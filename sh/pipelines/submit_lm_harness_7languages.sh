#!/bin/bash

# lm-evaluation-harness k-shot evaluation of the 7-language LoRA continued-pretraining runs
# (sh/pipelines/submit_lora_cpt_7languages.sh), one 1-GPU job per run, then one CPU job comparing them.
#
#   prefetch  (login node) every benchmark dataset + the base model into the shared HF cache; jobs run offline
#   eval      sh/eval/eval_lm_harness.sh per method: base = the untrained MODEL from the Hub, others = <run>/final.
#             A run still training (job g7-<method> queued or running) is evaluated after it succeeds; a run
#             with neither final/ nor a training job is skipped.
#             -> checkpoints/lora_cpt/<model>/<run>/lm_eval_<K>shot/{summary.md,summary.csv,results.json,...}
#   compare   after every eval job: macro-average per benchmark and per language, difference from random
#             -> checkpoints/lora_cpt/<model>/lm_eval_<K>shot_comparison.md
#
# Run from the project root, on the login node:  bash sh/pipelines/submit_lm_harness_7languages.sh
# Some methods only: METHODS="base random avg5" bash ...   0-shot: NUM_FEWSHOT=0 bash ...
# Additional benchmarks (SIB-200, SEA-NLI, FLORES+): BENCHMARKS=extra bash ...  -> lm_eval_extra_<K>shot/
#   (all: BENCHMARKS=all -> lm_eval_all_<K>shot/); per-benchmark k: SHOTS="sea_nli_normal=5 flores_plus=1"
# Other partition: PARTITION=cscc-gpu-p bash ...

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

METHODS="${METHODS:-base random edu cult avg5 rr5}"
BUDGET="${BUDGET:-50M}"
SUFFIX=_7languages
MODEL="${MODEL:-google/gemma-3-1b-pt}"
SEED="${SEED:-42}"                           # training seed (run folder name)
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BENCHMARKS="${BENCHMARKS:-core}"            # core | extra | all (see eval_lm_harness.py)
SHOTS="${SHOTS:-}"
TIME="${TIME:-12:00:00}"
PARTITION="${PARTITION:-long}"
declare -A PARTITION_QOS=([long]=gpu-12 [cscc-gpu-p]=cscc-gpu-qos)
QOS="${QOS:-${PARTITION_QOS[$PARTITION]}}"
EXCLUDE="${EXCLUDE:-gpu-05,gpu-12,gpu-15,gpu-51,gpu-59}"   # nodes whose GPUs failed earlier jobs
declare -A SHORT=([base]=base [random]=rand [edu]=edu [cult]=cult [avg4]=avg4 [avg5]=avg5 [rr5]=rr5)
ROOT="checkpoints/lora_cpt/${MODEL##*/}"
# core keeps the original folder name; other suites get their own folder
EVAL_NAME="lm_eval_$([ "$BENCHMARKS" = core ] || echo "${BENCHMARKS// /-}_")${NUM_FEWSHOT}shot"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate sea-rater
source .venv/lm-eval/bin/activate
set -a; source .env; set +a   # HF_TOKEN
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # shared with the compute nodes (login /tmp is node-local)

echo "=== prefetch: benchmark datasets + $MODEL ==="
python -m src.train_gpt2_from_scratch.eval_lm_harness prefetch --benchmarks $BENCHMARKS 2>&1 | grep -E "^cached|Error|error" || true
python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1]); print('cached', sys.argv[1])" "$MODEL"

EVAL_JOBS=""
OUT_DIRS=""
for M in $METHODS; do
  if [ "$M" = base ]; then RUN_DIR="$ROOT/base"; TARGET="$MODEL"; else RUN_DIR="$ROOT/${M}_${BUDGET}${SUFFIX}_ep1_seed${SEED}"; TARGET="$RUN_DIR/final"; fi
  DEP=""
  if [ "$M" != base ] && [ ! -f "$TARGET/config.json" ]; then
    TRAIN_JOB=$(squeue -h -u "$USER" -n "g7-${SHORT[$M]:-$M}" -o %i | head -1)
    if [ -z "$TRAIN_JOB" ]; then echo "eval   : $M skipped (no $TARGET and no training job g7-${SHORT[$M]:-$M})"; continue; fi
    DEP="--dependency=afterok:$TRAIN_JOB --kill-on-invalid-dep=yes"
  fi
  JOB=$(MODEL=$TARGET OUTPUT_DIR="$RUN_DIR/$EVAL_NAME" NUM_FEWSHOT=$NUM_FEWSHOT BENCHMARKS="$BENCHMARKS" SHOTS="$SHOTS" \
        sbatch --parsable $DEP --partition="$PARTITION" --qos="$QOS" --time="$TIME" --exclude="$EXCLUDE" \
        --job-name="h7-${SHORT[$M]:-$M}" sh/eval/eval_lm_harness.sh) || {
    # e.g. the QOS submit limit (long/gpu-12: 8 queued+running jobs per user): keep going so the others and the
    # comparison are still submitted; resubmit this one later, e.g. METHODS=$M PARTITION=cscc-gpu-p bash ...
    echo "eval   : $M NOT submitted (sbatch refused it, see the error above)" >&2; continue; }
  echo "eval   : $M job $JOB${DEP:+ (after training job ${DEP#--dependency=afterok:})} -> $RUN_DIR/$EVAL_NAME"
  EVAL_JOBS="$EVAL_JOBS:$JOB"
  OUT_DIRS="$OUT_DIRS $RUN_DIR/$EVAL_NAME"
done

if [ -n "$EVAL_JOBS" ]; then
  BASELINE="$ROOT/random_${BUDGET}${SUFFIX}_ep1_seed${SEED}/$EVAL_NAME"
  case " $OUT_DIRS " in *" $BASELINE "*) ;; *) BASELINE="" ;; esac
  JOB=$(sbatch --parsable --dependency=afterany${EVAL_JOBS} --partition=cscc-cpu-p --qos=cscc-cpu-qos --time=0:20:00 \
        --mem=4G --ntasks=1 --job-name=h7-compare --output=job-%j.out --error=job-%j.err --wrap \
        "source /apps/local/anaconda3/etc/profile.d/conda.sh && conda activate sea-rater && \
         RUNS=\"\"; for d in$OUT_DIRS; do [ -f \$d/summary.csv ] && RUNS=\"\$RUNS \$d\"; done; \
         python -m src.train_gpt2_from_scratch.eval_lm_harness compare --runs \$RUNS ${BASELINE:+--baseline $BASELINE} \
           --output $ROOT/${EVAL_NAME}_comparison.md")
  echo "compare: job $JOB (after all eval jobs) -> $ROOT/${EVAL_NAME}_comparison.md"
fi
echo "Check progress: squeue -u $USER"
