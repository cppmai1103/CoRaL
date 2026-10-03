#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-2080ti
#SBATCH --time=0:30:00
#SBATCH --job-name=score-test
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# One-off: score the held-out TEST split (not the training candidate pool) with all 5 dimension
# raters, then combine into avg4/avg5, so the importance-score distribution of the test set can be
# compared against the training pool / selected sets. Writes to data/pilot_scores/<dim>_mean_test and
# data/pilot_scores/avg{4,5}_mean_test, kept separate from the candidate-pool scores in data/pilot_scores/<dim>_mean.
#
# Run: sbatch sh/score_test_set.sh

RATER_ROOT="checkpoints/rater/finetuned/mean"
DIMENSIONS="educational_value reasoning professionalism cleanliness cultural_nuances"
PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

for DIM in $DIMENSIONS; do
  echo "=== score test split: $DIM ==="
  python -m src.train_gpt2_from_scratch.score_pool \
    --rater-dir "$RATER_ROOT/$DIM" \
    --output-dir "data/pilot_scores/${DIM}_mean_test" \
    --splits test \
    --batch-size 8 \
    --device cuda
done

SCORE_DIRS=""
for DIM in $DIMENSIONS; do
  SCORE_DIRS="$SCORE_DIRS data/pilot_scores/${DIM}_mean_test"
done

echo "=== combine: avg4 (excludes cultural_nuances) ==="
python -m src.train_gpt2_from_scratch.combine_scores \
  --scores-dirs data/pilot_scores/educational_value_mean_test data/pilot_scores/reasoning_mean_test \
                data/pilot_scores/professionalism_mean_test data/pilot_scores/cleanliness_mean_test \
  --output-dir data/pilot_scores/avg4_mean_test

echo "=== combine: avg5 (all 5 dimensions) ==="
python -m src.train_gpt2_from_scratch.combine_scores \
  --scores-dirs $SCORE_DIRS \
  --output-dir data/pilot_scores/avg5_mean_test

echo "Done: data/pilot_scores/{educational_value,avg4,avg5}_mean_test"
