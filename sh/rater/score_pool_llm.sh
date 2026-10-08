#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=2:00:00
#SBATCH --job-name=score-llm
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# Score the GPT-2 candidate pool (data/pilot_corpus train split, 48.5K documents x 6 languages) with ONE
# LLM-label rater (sh/rater/finetune_llm.sh), exactly like the human-label scoring in sh/train/train_gpt2.sh MODE=score
# but reading checkpoints/rater_llm/ and writing data/pilot_scores_llm/ (the human-rater scores in
# data/pilot_scores/ are untouched). Already-scored languages are skipped on a rerun (score_pool.py).
#
# Run: DIMENSION=educational_value sbatch sh/rater/score_pool_llm.sh
# Any other rater/pool, e.g. the 7-language human rater on the 7-language corpus:
#   DIMENSION=reasoning RATER_ROOT=checkpoints/rater_7languages/finetuned SCORES_ROOT=data/pilot_scores_7languages \
#     POOL_DIR=data/pilot_corpus_7languages sbatch sh/rater/score_pool_llm.sh
# Faster on N GPUs (languages split across them, one process each): sbatch --gres=gpu:2 sh/rater/score_pool_llm.sh

DIMENSION="${DIMENSION:-educational_value}"
POOLING="${POOLING:-mean}"
RATER_ROOT="${RATER_ROOT:-checkpoints/rater_llm/finetuned}"
SCORES_ROOT="${SCORES_ROOT:-data/pilot_scores_llm}"
POOL_DIR="${POOL_DIR:-data/pilot_corpus}"
ALL_DOCUMENTS="${ALL_DOCUMENTS:-0}"   # 1: score every document of POOL_DIR (no split manifest yet)
BATCH_SIZE="${BATCH_SIZE:-32}"        # documents per forward pass; ~7GB of an A100-40GB at 32
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | DIMENSION=$DIMENSION | rater $RATER_ROOT/$POOLING/$DIMENSION -> $SCORES_ROOT/${DIMENSION}_$POOLING"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)

OUT="$SCORES_ROOT/${DIMENSION}_$POOLING"
ARGS=(--rater-dir "$RATER_ROOT/$POOLING/$DIMENSION" --pool-dir "$POOL_DIR" --output-dir "$OUT" --batch-size "$BATCH_SIZE" --device cuda)
[ "$ALL_DOCUMENTS" = 1 ] && ARGS+=(--no-split-manifest)
read -ra GPUS <<< "$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' ' ')"

if [ "${#GPUS[@]}" -le 1 ]; then
  python -m src.train_gpt2_from_scratch.score_pool "${ARGS[@]}"
else
  # More than one GPU (sbatch --gres=gpu:N): one score_pool.py process per GPU, each on its own languages.
  # Languages go largest pool file first to the least-loaded GPU; already-scored ones count as no work.
  mapfile -t SHARDS < <(python - "$POOL_DIR" "$OUT" "${#GPUS[@]}" <<'EOF'
import sys
from pathlib import Path
pool, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
files = [p for p in pool.glob("*.csv") if p.stem != "split_manifest"]
work = {p.stem: 0 if (out / p.name).is_file() else p.stat().st_size for p in files}
shards = [[0, []] for _ in range(n)]
for lang in sorted(work, key=work.get, reverse=True):
    shard = min(shards, key=lambda s: s[0])
    shard[0] += work[lang]
    shard[1].append(lang)
for _, langs in shards:
    print(" ".join(langs))
EOF
  )
  PIDS=()
  for i in "${!GPUS[@]}"; do
    [ -n "${SHARDS[$i]}" ] || continue
    LOG="job-${SLURM_JOB_ID}.gpu$i.out"
    echo "  GPU ${GPUS[$i]}: ${SHARDS[$i]} -> log $LOG"
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m src.train_gpt2_from_scratch.score_pool "${ARGS[@]}" \
      --languages ${SHARDS[$i]} --no-plot > "$LOG" 2>&1 &
    PIDS+=($!)
  done
  FAILED=0
  for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
  [ "$FAILED" = 0 ] || { echo "ERROR: a scoring process failed, see job-${SLURM_JOB_ID}.gpu*.out" >&2; exit 1; }
  python -m src.train_gpt2_from_scratch.score_pool "${ARGS[@]}" --plot-only   # distribution plot over all languages
fi

echo "Done: $OUT/"
