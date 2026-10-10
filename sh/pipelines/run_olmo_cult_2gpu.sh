#!/bin/bash

# Local (no SLURM): OLMo 1B + LoRA continued pretraining on the 50M-OLMo-token-per-language selection ranked by the
# cultural-nuances rater (data/pilot_scores_7languages/cultural_nuances_mean), ONE run split over two GPUs with torchrun
# (same batches and updates as on one GPU, ~2x faster; see continue_pretrain_lora.py). Same hyperparameters and output
# layout as sh/pipelines/run_olmo_7languages_local.sh (random, wavg5, edu, avg5, rr5).
#
#   cult   top documents by cultural nuance, data/pilot_selected/cult_50M_7languages_olmo, ordinary loss
#   -> checkpoints/lora_cpt/OLMo-1B-hf/cult_50M_7languages_ep1_seed42/  (log: <run folder>.log)
# The selection is made first if missing (~25 min, CPU). Checkpoints every 250 steps; rerunning the same command after
# a crash or kill resumes from the last one.
#
# Run from anywhere:  bash sh/pipelines/run_olmo_cult_2gpu.sh
# Other GPUs:  GPUS="2,3" bash ...      Follow:  tail -f checkpoints/lora_cpt/OLMo-1B-hf/cult_50M_7languages_ep1_seed42.log

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPUS="${GPUS:-0,1}"
PYTHON="${PYTHON:-.venv/sea-rater/bin/python}"
MODEL="${MODEL:-allenai/OLMo-1B-hf}"
TARGET_TOKENS="${TARGET_TOKENS:-50000000}"
BUDGET="${BUDGET:-50M}"
SEED="${SEED:-42}"
POOL_DIR=data/pilot_corpus_7languages
SELECTION="data/pilot_selected/cult_${BUDGET}_7languages_olmo"
RUN="cult_${BUDGET}_7languages"
ROOT="checkpoints/lora_cpt/${MODEL##*/}"
LOG="$ROOT/${RUN}_ep1_seed${SEED}.log"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

if [ ! -f "$SELECTION/documents.csv" ]; then
  echo "=== select: top cultural nuance, $TARGET_TOKENS OLMo tokens per language -> $SELECTION (~25 min) ==="
  "$PYTHON" -m src.train_gpt2_from_scratch.prepare_data --method top-score --pool-dir "$POOL_DIR" \
    --scores-dir data/pilot_scores_7languages/cultural_nuances_mean --output-dir "$SELECTION" \
    --tokenizer "$MODEL" --select-by tokens --target-tokens "$TARGET_TOKENS"
fi

mkdir -p "$ROOT"
CUDA_VISIBLE_DEVICES=$GPUS nohup "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  -m src.train_gpt2_from_scratch.continue_pretrain_lora \
  --model "$MODEL" --pool-dir "$POOL_DIR" --seed "$SEED" --method "$RUN" --train-data "$SELECTION/documents.csv" \
  --seq-len "${SEQ_LEN:-2048}" --lr "${LR:-1e-4}" --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-4}" --eval-batch-size "${EVAL_BATCH_SIZE:-4}" --no-progress --device cuda \
  > "$LOG" 2>&1 &
echo "train  : $RUN on GPUs $GPUS ($NPROC processes), pid $! -> $ROOT/${RUN}_ep1_seed${SEED}/  (log $LOG)"
echo "Continues after you log out. Follow with: tail -f $LOG"
