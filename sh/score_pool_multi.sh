#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=10:00:00
#SBATCH --job-name=s7-multi
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:3
#SBATCH --qos=normal

set -eo pipefail

# Score the pool on several dimensions at once inside ONE job: one score_pool.py process per GPU, one dimension
# each (the QOS allows 2 running jobs but up to 4 GPUs per job). Same rater/pool/output as sh/score_pool_llm.sh;
# languages already scored are skipped, so a cancelled single-dimension job resumes here. Request as many GPUs
# as dimensions (at most 4):
#   DIMENSIONS="professionalism cleanliness cultural_nuances" sbatch --gres=gpu:3 sh/score_pool_multi.sh
# Logs: job-<id>.out (this script) and job-<id>.<dimension>.out (one per process).

DIMENSIONS="${DIMENSIONS:-professionalism cleanliness cultural_nuances}"
POOLING="${POOLING:-mean}"
RATER_ROOT="${RATER_ROOT:-checkpoints/rater_7languages/finetuned}"
SCORES_ROOT="${SCORES_ROOT:-data/pilot_scores_7languages}"
POOL_DIR="${POOL_DIR:-data/pilot_corpus_7languages}"
ALL_DOCUMENTS="${ALL_DOCUMENTS:-1}"   # 1: every document of POOL_DIR (no split manifest needed)

cd "$SLURM_SUBMIT_DIR"
read -ra DIMS <<< "$DIMENSIONS"
read -ra GPUS <<< "$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' ' ')"
echo "Working directory: $(pwd) | ${#DIMS[@]} dimensions on GPUs ${GPUS[*]}: ${DIMS[*]}"
if [ "${#DIMS[@]}" -gt "${#GPUS[@]}" ]; then
  echo "ERROR: ${#DIMS[@]} dimensions but ${#GPUS[@]} GPU(s); submit with --gres=gpu:${#DIMS[@]}" >&2; exit 1
fi

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate sea-rater
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
nvidia-smi

PIDS=()
for i in "${!DIMS[@]}"; do
  DIM=${DIMS[$i]}
  LOG="job-${SLURM_JOB_ID}.${DIM}.out"
  echo "  $DIM -> GPU ${GPUS[$i]}, log $LOG"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m src.train_gpt2_from_scratch.score_pool \
    --rater-dir "$RATER_ROOT/$POOLING/$DIM" \
    --pool-dir "$POOL_DIR" \
    $([ "$ALL_DOCUMENTS" = 1 ] && echo --no-split-manifest) \
    --output-dir "$SCORES_ROOT/${DIM}_$POOLING" \
    --device cuda > "$LOG" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  ${DIMS[$i]}: done"; else echo "  ${DIMS[$i]}: FAILED (see job-${SLURM_JOB_ID}.${DIMS[$i]}.out)" >&2; FAILED=1; fi
done
[ "$FAILED" = 0 ] && echo "Done: $SCORES_ROOT/{$(echo "$DIMENSIONS" | tr ' ' ',')}_$POOLING/" || exit 1
