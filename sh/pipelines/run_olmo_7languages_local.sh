#!/bin/bash

# Local (no SLURM) version of sh/pipelines/submit_olmo_7languages.sh: OLMo 1B + LoRA continued pretraining on 50M OLMo
# tokens per language, random and wavg5 run in parallel in the background, one GPU each. Same data, hyperparameters
# and output folders as the SLURM launcher; uses the project's .venv/sea-rater instead of the cluster conda env.
#
#   random   data/pilot_selected/random_50M_7languages_olmo, ordinary loss
#   wavg5    the same documents in the same order, loss weighted by 0.5 + avg5/5 (normalised per language)
#   edu      the top documents by the educational-value rater, data/pilot_selected/edu_50M_7languages_olmo, ordinary loss
#   -> checkpoints/lora_cpt/OLMo-1B-hf/<method>_50M_7languages_ep1_seed42/  (log: <run folder>.log)
# Checkpoints every 250 steps (~20 min); rerunning the same command after a crash or kill resumes each run from its
# last checkpoint (continue_pretrain_lora.py --save-every / --no-resume).
#
# Run from anywhere:  bash sh/pipelines/run_olmo_7languages_local.sh
# Other GPUs / one method:  GPUS="2 3" bash ...   METHODS="edu" GPUS="1" bash ...
# Follow:  tail -f checkpoints/lora_cpt/OLMo-1B-hf/*_ep1_seed42.log      Stop:  kill <pid printed below>

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

METHODS="${METHODS:-random wavg5}"
GPUS="${GPUS:-0 1}"                          # one per method, in order
PYTHON="${PYTHON:-.venv/sea-rater/bin/python}"
MODEL="${MODEL:-allenai/OLMo-1B-hf}"
TARGET_TOKENS="${TARGET_TOKENS:-50000000}"
BUDGET="${BUDGET:-50M}"
SUFFIX=_7languages
POOL_DIR=data/pilot_corpus_7languages
SCORES_ROOT=data/pilot_scores_7languages
SELECTION="data/pilot_selected/random_${BUDGET}${SUFFIX}_olmo"           # random and wavg5
EDU_SELECTION="data/pilot_selected/edu_${BUDGET}${SUFFIX}_olmo"         # edu
SEED="${SEED:-42}"
TRAIN_ARGS=(--model "$MODEL" --pool-dir "$POOL_DIR" --seed "$SEED"
            --seq-len "${SEQ_LEN:-2048}" --lr "${LR:-1e-4}" --weight-decay "${WEIGHT_DECAY:-0.01}"
            --micro-batch-size "${MICRO_BATCH_SIZE:-4}" --eval-batch-size "${EVAL_BATCH_SIZE:-4}" --no-progress --device cuda)
ROOT="checkpoints/lora_cpt/${MODEL##*/}"

read -ra METHOD_LIST <<< "$METHODS"
read -ra GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt "${#METHOD_LIST[@]}" ]; then
  echo "ERROR: ${#METHOD_LIST[@]} methods but only ${#GPU_LIST[@]} GPUs in GPUS=\"$GPUS\"" >&2; exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

if [[ " $METHODS " == *" edu "* ]] && [ ! -f "$EDU_SELECTION/documents.csv" ]; then
  echo "=== select: top educational value, $TARGET_TOKENS OLMo tokens per language -> $EDU_SELECTION (~25 min) ==="
  "$PYTHON" -m src.train_gpt2_from_scratch.prepare_data --method top-score --pool-dir "$POOL_DIR" \
    --scores-dir "$SCORES_ROOT/educational_value_mean" --output-dir "$EDU_SELECTION" \
    --tokenizer "$MODEL" --select-by tokens --target-tokens "$TARGET_TOKENS"
fi
if [[ " $METHODS " == *" random "* || " $METHODS " == *" wavg5 "* ]] && [ ! -f "$SELECTION/documents.csv" ]; then
  echo "=== select: random $TARGET_TOKENS OLMo tokens per language -> $SELECTION (~25 min) ==="
  "$PYTHON" -m src.train_gpt2_from_scratch.prepare_data --method random --pool-dir "$POOL_DIR" --output-dir "$SELECTION" \
    --tokenizer "$MODEL" --select-by tokens --target-tokens "$TARGET_TOKENS"
fi

mkdir -p "$ROOT"
for i in "${!METHOD_LIST[@]}"; do
  M="${METHOD_LIST[$i]}"; GPU="${GPU_LIST[$i]}"
  case "$M" in
    random) EXTRA=(--train-data "$SELECTION/documents.csv") ;;
    wavg5)  EXTRA=(--train-data "$SELECTION/documents.csv" --loss-weight-scores "$SCORES_ROOT/avg5_mean") ;;
    edu)    EXTRA=(--train-data "$EDU_SELECTION/documents.csv") ;;
    *) echo "Unknown method $M (random, wavg5, edu)" >&2; exit 1 ;;
  esac
  RUN="${M}_${BUDGET}${SUFFIX}"
  LOG="$ROOT/${RUN}_ep1_seed${SEED}.log"
  CUDA_VISIBLE_DEVICES=$GPU nohup "$PYTHON" -m src.train_gpt2_from_scratch.continue_pretrain_lora \
    "${TRAIN_ARGS[@]}" --method "$RUN" "${EXTRA[@]}" > "$LOG" 2>&1 &
  echo "train  : $RUN on GPU $GPU, pid $! -> $ROOT/${RUN}_ep1_seed${SEED}/  (log $LOG)"
done
echo "Both runs continue after you log out. Follow with: tail -f $ROOT/*_ep1_seed${SEED}.log"
