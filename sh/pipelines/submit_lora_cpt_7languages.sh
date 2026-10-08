#!/bin/bash

# Gemma 3 1B (pt) + LoRA continued pretraining on the 7-language selections, TARGET_TOKENS (50M) Gemma tokens per
# language each (the selection counts with the trained model's own tokenizer), one 1-GPU job per selection. Validation/test loss come from the training run itself (split of
# data/pilot_corpus_7languages); no benchmark evaluation here.
#
#   prefetch  (login node) Gemma (model + tokenizer) into the Hugging Face cache; the jobs then run with
#             HF_HUB_OFFLINE=1, so compute nodes need no network
#   select    1 CPU job, sh/rater/select_llm.sh, skipped when every selection to train already exists
#             -> data/pilot_selected/{random,edu,cult,avg4,avg5,rr5}_50M_7languages/ (avg4 only for analysis)
#   train     one job per method, after select   sh/train/lora_cpt.sh
#             -> checkpoints/lora_cpt/gemma-3-1b-pt/<method>_50M_7languages_ep1_seed42/
#
# Run from the project root, on the login node:  bash sh/pipelines/submit_lora_cpt_7languages.sh
# Some methods only: METHODS="random avg5" bash ...   (add "base" for the untrained-model validation/test reference)
# Other seed / time limit: SEED=43 TIME=36:00:00 bash ...
# Selection still running (job 1234): WAIT_JOB=1234 bash ...   (no second select job; training starts after it succeeds)
# Other partition: PARTITION=cscc-gpu-p METHODS="avg5 rr5" bash ...   (cscc-gpu-qos: 2 running jobs, 48h; long/gpu-12: 8, 72h)
# Several runs per job, one per GPU (sh/train/lora_cpt_multi.sh): GPUS_PER_JOB=2 PARTITION=cscc-gpu-p \
#   METHODS="avg5 rr5 random edu" bash ...   -> 2 jobs x 2 GPUs, i.e. 4 runs within cscc-gpu-qos's 2-job limit

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

METHODS="${METHODS:-random edu cult avg5 rr5}"
TARGET_TOKENS="${TARGET_TOKENS:-50000000}"
BUDGET="${BUDGET:-50M}"
SUFFIX=_7languages
POOL_DIR=data/pilot_corpus_7languages
SCORES_ROOT=data/pilot_scores_7languages
MODEL="${MODEL:-google/gemma-3-1b-pt}"   # same tokenizer as gemma-3-270m, so its selections stay valid
SEED="${SEED:-42}"
# Training hyperparameters, sized for 1 A100 40GB: 32 x 2048 = 65,536 tokens per update. Gemma's 262k-vocab logits
# dominate memory (~5 GB per 2048-token sequence), so 2 sequences per forward/backward and 2 per eval batch; turn on
# GRADIENT_CHECKPOINTING=1 (or MICRO_BATCH_SIZE=1) only on CUDA out of memory.
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-1e-4}"                             # cosine, 3% warmup, grad clip 1.0 (continue_pretrain_lora.py defaults)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
TIME="${TIME:-20:00:00}"                     # measured ~12.3h per 350M-token run (8.3k tokens/s on an A100 40GB); no mid-run checkpoints, so keep headroom
EXCLUDE="${EXCLUDE:-gpu-05,gpu-12,gpu-15,gpu-51,gpu-59}"   # nodes whose GPUs failed earlier jobs (gpu-05: no GPU reachable; gpu-51: GPUs 2-3 missing)
PARTITION="${PARTITION:-long}"
declare -A PARTITION_QOS=([long]=gpu-12 [cscc-gpu-p]=cscc-gpu-qos)
QOS="${QOS:-${PARTITION_QOS[$PARTITION]}}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"           # >1: group METHODS into jobs of this many runs, one GPU each
declare -A SHORT=([base]=base [random]=rand [edu]=edu [cult]=cult [avg4]=avg4 [avg5]=avg5 [rr5]=rr5)
selection() { echo "data/pilot_selected/${1}_${BUDGET}${SUFFIX}"; }

echo "=== prefetch: $MODEL (also counts the selection budget) ==="
source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate sea-rater
set -a; source .env; set +a   # HF_TOKEN (Gemma is gated)
# Shared cache: the login shell's XDG_CACHE_HOME (/tmp/$USER/cache) is local to the login node, so compute nodes
# would not find the prefetched model there; sbatch passes HF_HOME on to the jobs.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
snapshot_download(sys.argv[1])
AutoTokenizer.from_pretrained(sys.argv[1])
print("cached")
EOF
export HF_HUB_OFFLINE=1   # sbatch passes the environment on to the jobs

SELECT_JOB="${WAIT_JOB:-}"             # a select job already running: train after it instead of submitting another
MISSING=""
for M in $METHODS; do
  [ "$M" = base ] || [ -f "$(selection "$M")/documents.csv" ] || MISSING="$MISSING $M"
done
if [ -n "$WAIT_JOB" ]; then
  echo "select : waiting for running job $WAIT_JOB"
elif [ -n "$MISSING" ]; then
  SELECT_JOB=$(SCORES_ROOT=$SCORES_ROOT POOL_DIR=$POOL_DIR N_LANGUAGES=7 OUT_PREFIX="" OUT_SUFFIX=$SUFFIX \
               SELECT_RANDOM=1 COMPARE_RATERS=0 ROUND_ROBIN=1 TARGET_TOKENS=$TARGET_TOKENS BUDGET=$BUDGET TOKENIZER=$MODEL \
               TOP_METHODS="edu:educational_value_mean cult:cultural_nuances_mean avg4:avg4_mean avg5:avg5_mean" \
               sbatch --parsable --time=4:00:00 --job-name=select-7l sh/rater/select_llm.sh)
  echo "select : job $SELECT_JOB (missing:$MISSING) -> data/pilot_selected/*_${BUDGET}${SUFFIX}"
else
  echo "select : every selection exists, skipped"
fi
DEP=${SELECT_JOB:+--dependency=afterok:$SELECT_JOB --kill-on-invalid-dep=yes}

run_name() { if [ "$1" = base ]; then echo base; else echo "${1}_${BUDGET}${SUFFIX}"; fi; }
out_dir() { echo "checkpoints/lora_cpt/${MODEL##*/}/$([ "$1" = base ] && echo base || echo "$(run_name "$1")_ep1_seed${SEED}")"; }
TRAIN_ENV=(POOL_DIR=$POOL_DIR MODEL=$MODEL SEED=$SEED SEQ_LEN=$SEQ_LEN LR=$LR WEIGHT_DECAY=$WEIGHT_DECAY
           MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING)

if [ "$GPUS_PER_JOB" -gt 1 ]; then
  read -ra ALL <<< "$METHODS"
  for ((i = 0; i < ${#ALL[@]}; i += GPUS_PER_JOB)); do
    GROUP=("${ALL[@]:i:GPUS_PER_JOB}")
    RUNS=""; NAMES=""
    for M in "${GROUP[@]}"; do RUNS="$RUNS $(run_name "$M")"; NAMES="$NAMES${NAMES:+-}${SHORT[$M]:-$M}"; done
    N=${#GROUP[@]}
    JOB=$(env "${TRAIN_ENV[@]}" METHODS="${RUNS# }" \
          sbatch --parsable $DEP --partition="$PARTITION" --qos="$QOS" --time="$TIME" --gres=gpu:$N \
          --cpus-per-task=$((4 * N)) --mem=$((64 * N))G --exclude="$EXCLUDE" --job-name="g7-$NAMES" \
          sh/train/lora_cpt_multi.sh)
    for M in "${GROUP[@]}"; do echo "train  : $(run_name "$M") job $JOB (log job-$JOB.$(run_name "$M").out) -> $(out_dir "$M")"; done
  done
else
  for M in $METHODS; do
    # base only evaluates validation/test: small, short job, so it fits gaps the training runs cannot
    if [ "$M" = base ]; then RES=(--time=3:00:00 --mem=24G); else RES=(--time="$TIME" --mem=64G); fi
    JOB=$(env "${TRAIN_ENV[@]}" METHOD="$(run_name "$M")" \
          sbatch --parsable $DEP --partition="$PARTITION" --qos="$QOS" "${RES[@]}" --exclude="$EXCLUDE" --job-name="g7-${SHORT[$M]:-$M}" \
          sh/train/lora_cpt.sh)
    echo "train  : $(run_name "$M") job $JOB -> $(out_dir "$M")"
  done
fi
echo "Check progress: squeue -u $USER"
