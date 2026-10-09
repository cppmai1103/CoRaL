#!/bin/bash

# OLMo 1B + LoRA continued pretraining on the 7-language pool, TARGET_TOKENS (50M) OLMo tokens per language, two runs
# on one GPU each:
#   random   the random selection, ordinary loss (the baseline)
#   wavg5    the SAME documents in the same order, quality-weighted loss (docs/05_loss.md): each document's token loss
#            is weighted by 0.5 + avg5/5 (human-rater 5-dimension average), normalised to a mean weight of 1 per language
#
#   prefetch  (login node) OLMo (model + tokenizer) into the shared Hugging Face cache; the jobs run with HF_HUB_OFFLINE=1
#   select    1 CPU job, skipped when the selection exists: random documents of the train split, counted with OLMo's own
#             tokenizer (the Gemma-counted random_50M_7languages holds ~1.2-3.9x more OLMo tokens per language)
#             -> data/pilot_selected/random_50M_7languages_olmo/
#   train     one 1-GPU job per method, after select   sh/train/lora_cpt.sh
#             -> checkpoints/lora_cpt/OLMo-1B-hf/{random,wavg5}_50M_7languages_ep1_seed42/
#
# Same hyperparameters as the Gemma 3 1B runs (sh/pipelines/submit_lora_cpt_7languages.sh): LoRA r=16/alpha=32, lr 1e-4,
# weight decay 0.01, 32 x 2048 tokens per update. Measured on an A100 40GB: ~13.8k tokens/s, 29.4 GiB peak, so ~7h per run.
#
# Run from the project root, on the login node:  bash sh/pipelines/submit_olmo_7languages.sh
# One method only: METHODS="wavg5" bash ...      Other seed / time limit: SEED=43 TIME=20:00:00 bash ...
# Other partition: PARTITION=cscc-gpu-p bash ...
# Evaluate afterwards with the 7-language evaluation launchers (same run folder names, under OLMo-1B-hf/):
#   MODEL=allenai/OLMo-1B-hf METHODS="base random wavg5" bash sh/pipelines/submit_lm_harness_7languages.sh
#   MODEL=allenai/OLMo-1B-hf METHODS="random base wavg5" bash sh/pipelines/submit_ppl_7languages.sh

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

METHODS="${METHODS:-random wavg5}"
MODEL="${MODEL:-allenai/OLMo-1B-hf}"
TARGET_TOKENS="${TARGET_TOKENS:-50000000}"
BUDGET="${BUDGET:-50M}"
SUFFIX=_7languages
POOL_DIR=data/pilot_corpus_7languages
SCORES_ROOT=data/pilot_scores_7languages
SELECTION="data/pilot_selected/random_${BUDGET}${SUFFIX}_olmo"   # counted with OLMo's tokenizer
SEED="${SEED:-42}"
SEQ_LEN="${SEQ_LEN:-2048}"                   # OLMo 1B's context length
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"    # 50k-vocab logits: 4 x 2048 fits in 29.4 GiB
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
TIME="${TIME:-14:00:00}"                     # ~7h measured; no mid-run checkpoints, so keep headroom
EXCLUDE="${EXCLUDE:-gpu-05,gpu-12,gpu-15,gpu-51,gpu-59}"   # nodes whose GPUs failed earlier jobs
PARTITION="${PARTITION:-long}"
declare -A PARTITION_QOS=([long]=gpu-12 [cscc-gpu-p]=cscc-gpu-qos)
QOS="${QOS:-${PARTITION_QOS[$PARTITION]}}"
# g7-<short>: the job names submit_lm_harness_7languages.sh waits for when a run is still training
declare -A SHORT=([random]=rand [wavg5]=wavg5)

echo "=== prefetch: $MODEL ==="
source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate sea-rater
set -a; source .env; set +a   # HF_TOKEN (lora_cpt.sh reads it too)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # shared with the compute nodes (login /tmp is node-local)
python - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
snapshot_download(sys.argv[1])
AutoTokenizer.from_pretrained(sys.argv[1])
print("cached")
EOF
export HF_HUB_OFFLINE=1   # sbatch passes the environment on to the jobs

SELECT_JOB=""
if [ -f "$SELECTION/documents.csv" ]; then
  echo "select : $SELECTION exists, skipped"
else
  SELECT_JOB=$(sbatch --parsable --partition=cscc-cpu-p --qos=cscc-cpu-qos --time=2:00:00 --cpus-per-task=8 --mem=32G \
               --job-name=select-olmo --output=job-%j.out --error=job-%j.err --wrap="
    set -eo pipefail
    source /apps/local/anaconda3/etc/profile.d/conda.sh && conda activate sea-rater
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    python -m src.train_gpt2_from_scratch.prepare_data --method random --pool-dir $POOL_DIR --output-dir $SELECTION \
      --tokenizer $MODEL --select-by tokens --target-tokens $TARGET_TOKENS")
  echo "select : job $SELECT_JOB -> $SELECTION"
fi
DEP=${SELECT_JOB:+--dependency=afterok:$SELECT_JOB --kill-on-invalid-dep=yes}

for M in $METHODS; do
  case "$M" in
    random) WEIGHTS="" ;;
    wavg5)  WEIGHTS="$SCORES_ROOT/avg5_mean" ;;
    *) echo "Unknown method $M (random, wavg5)" >&2; exit 1 ;;
  esac
  RUN="${M}_${BUDGET}${SUFFIX}"
  JOB=$(env POOL_DIR=$POOL_DIR MODEL=$MODEL SEED=$SEED SEQ_LEN=$SEQ_LEN LR=$LR WEIGHT_DECAY=$WEIGHT_DECAY \
        MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE METHOD=$RUN \
        TRAIN_DATA="$SELECTION/documents.csv" LOSS_WEIGHT_SCORES="$WEIGHTS" \
        sbatch --parsable $DEP --partition="$PARTITION" --qos="$QOS" --time="$TIME" --mem=64G --gres=gpu:1 \
        --exclude="$EXCLUDE" --job-name="g7-${SHORT[$M]}" sh/train/lora_cpt.sh)
  echo "train  : $RUN job $JOB -> checkpoints/lora_cpt/${MODEL##*/}/${RUN}_ep1_seed${SEED}"
done
echo "Check progress: squeue -u $USER"
