#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=2:00:00
#SBATCH --job-name=score-llm
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Score the GPT-2 candidate pool (data/pilot_corpus train split, 48.5K documents x 6 languages) with ONE
# LLM-label rater (sh/finetune_llm.sh), exactly like the human-label scoring in sh/train_gpt2.sh MODE=score
# but reading checkpoints/rater_llm/ and writing data/pilot_scores_llm/ (the human-rater scores in
# data/pilot_scores/ are untouched). Already-scored languages are skipped on a rerun (score_pool.py).
#
# Run: DIMENSION=educational_value sbatch sh/score_pool_llm.sh

DIMENSION="${DIMENSION:-educational_value}"
POOLING="${POOLING:-mean}"
RATER_ROOT="${RATER_ROOT:-checkpoints/rater_llm/finetuned}"
SCORES_ROOT="${SCORES_ROOT:-data/pilot_scores_llm}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | DIMENSION=$DIMENSION | rater $RATER_ROOT/$POOLING/$DIMENSION -> $SCORES_ROOT/${DIMENSION}_$POOLING"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi

python -m src.train_gpt2_from_scratch.score_pool \
  --rater-dir "$RATER_ROOT/$POOLING/$DIMENSION" \
  --output-dir "$SCORES_ROOT/${DIMENSION}_$POOLING" \
  --device cuda

echo "Done: $SCORES_ROOT/${DIMENSION}_$POOLING/"
