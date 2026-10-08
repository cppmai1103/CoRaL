#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=0:30:00
#SBATCH --job-name=eval-on-annotated
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# One-off: evaluate the trained GPT-2 pilot checkpoints against the human-annotated SEA-Rater
# reference dataset (data/rater_dataset/document_table.csv, ~5,698 unique documents / 6 languages --
# disjoint from and unseen by the GPT-2 pilot corpus). No rater model involved here; see
# score_annotated_dataset.py's docstring for the steps. Below: --skip-full-eval runs ONLY the
# high-quality-subset pass (top 50% per language by avg5), and --runs restricts it to the 5
# checkpoints still missing from gpt2_results_highquality.json (edu_10M/20M, avg4_5M/10M/20M --
# random_5M/10M/20M, edu_5M and avg5_5M/10M/20M are already done). Drop --runs for every method,
# drop --skip-full-eval too to also redo the full-dataset pass. Already-completed runs are skipped
# automatically even without --runs; add --force to redo them anyway.
#
# Run: sbatch sh/eval/score_annotated_dataset.sh

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.score_annotated_dataset --device cuda --skip-full-eval \
  --runs edu_10M_ep1_seed42 edu_20M_ep1_seed42 \
        avg4_5M_ep1_seed42 avg4_10M_ep1_seed42 avg4_20M_ep1_seed42

echo "Done: data/rater_dataset/annotated_eval/{importance_score_distribution.png, importance_scores.csv, gpt2_results_highquality.json, gpt2_summary_highquality.md}"
