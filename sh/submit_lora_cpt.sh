#!/bin/bash

# Submit Gemma 3 270M + LoRA continued pretraining on the 20M selections, the untrained-model reference, and all
# evaluations, chained with SLURM dependencies (evaluations start after every training job succeeds).
#
#   train   base (no training) + random_20M, edu_20M, avg4_20M, avg5_20M   sh/lora_cpt.sh
#   eval    test split loss/PPL/bits per byte                               sh/eval_test_set.sh
#           human-annotated docs, high quality = avg5 top 50% / cleanliness 5   sh/eval_annotated_weighted.sh (x2)
#           Belebele/XCOPA accuracy + Wikipedia/SIB-200 loss/PPL/bits per byte   sh/eval_hub_model.sh
#             (with the GPT-2 random_20M run as an extra reference: compare it by bits per byte and accuracy only)
# Outputs: checkpoints/lora_cpt/gemma-3-270m/<run>/ and checkpoints/lora_cpt/gemma-3-270m/*comparison.md,
#          data/rater_dataset/annotated_eval_lora_gemma{,_cleanliness}/
# Loss/PPL compare only Gemma runs with each other (same tokenizer); "random_20M" vs edu/avg4/avg5 is the
# selection effect, "base" vs random_20M the effect of continued pretraining itself.
#
# Run from the project root with bash: bash sh/submit_lora_cpt.sh
# Only some stages: STAGES="eval" bash sh/submit_lora_cpt.sh   (after the training jobs finished)

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGES="${STAGES:-train eval}"
METHODS="${METHODS:-base random_20M edu_20M avg4_20M avg5_20M}"
ROOT="checkpoints/lora_cpt/gemma-3-270m"
has() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }
declare -A SHORT=([base]=base [random_20M]=rand [edu_20M]=edu [avg4_20M]=avg4 [avg5_20M]=avg5)

TRAIN_JOBS=""
if has train; then
  for M in $METHODS; do
    JOB=$(METHOD=$M sbatch --parsable --job-name="lg-${SHORT[$M]:-$M}" sh/lora_cpt.sh)
    echo "train  $M: job $JOB -> $ROOT/$([ "$M" = base ] && echo base || echo "${M}_ep1_seed42")"
    TRAIN_JOBS="${TRAIN_JOBS:+$TRAIN_JOBS:}$JOB"
  done
fi

if has eval; then
  RUNS="$ROOT/base"
  for M in random_20M edu_20M avg4_20M avg5_20M; do RUNS="$RUNS $ROOT/${M}_ep1_seed42"; done
  FINALS=""; for R in $RUNS; do FINALS="$FINALS $R/final"; done
  D=$(dep "$TRAIN_JOBS")
  JOB=$(RUN_DIRS="$RUNS" COMPARISON_FILE="$ROOT/test_comparison.md" \
        sbatch --parsable $D --job-name=lg-test sh/eval_test_set.sh)
  echo "eval   test split: job $JOB"
  JOB=$(RUN_ROOT="$ROOT" SCORE_METHODS="random edu avg4 avg5" BASELINE="$ROOT/base" \
        OUTPUT_DIR=data/rater_dataset/annotated_eval_lora_gemma sbatch --parsable $D --job-name=lg-hq sh/eval_annotated_weighted.sh)
  echo "eval   annotated (avg5 top 50%): job $JOB"
  JOB=$(RUN_ROOT="$ROOT" SCORE_METHODS="random edu avg4 avg5" BASELINE="$ROOT/base" HQ_BY=cleanliness HQ_MIN_SCORE=5 \
        OUTPUT_DIR=data/rater_dataset/annotated_eval_lora_gemma_cleanliness \
        sbatch --parsable $D --job-name=lg-clean sh/eval_annotated_weighted.sh)
  echo "eval   annotated (cleanliness == 5): job $JOB"
  JOB=$(MODELS="${FINALS# }" OUT_ROOT="$ROOT" sbatch --parsable $D --job-name=lg-down sh/eval_hub_model.sh)
  echo "eval   Belebele/XCOPA + Wikipedia/SIB-200: job $JOB"
fi
echo "Check progress: squeue -u $USER"
