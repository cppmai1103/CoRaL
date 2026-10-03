#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=2:00:00
#SBATCH --job-name=eval-wiki-sib200
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Wikipedia/SIB-200 loss+perplexity evaluation (docs/04_wikipedia_sib200_evaluation.md) of the
# 20M-token-budget GPT-2 pilot checkpoints: random, edu, avg4, avg5. Wikipedia: 1,000 eligible,
# training-deduplicated articles/language (all 6 languages). SIB-200: all 1,004 sentences/language
# pooled across train+dev+test, used as a text corpus (not for topic classification). The first
# --run-dir (random) is treated as the baseline for loss_diff_vs_random.
#
# First run downloads ~1.85GB of Wikipedia parquet (cached under ~/.cache/huggingface for the
# remaining 3 checkpoints in this same job).
#
# Run: sbatch sh/eval_wiki_sib200.sh

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

pip install pyarrow pandas >/dev/null  # Wikipedia and pilot_corpus overlap check both read parquet/csv via pandas

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.eval_wiki_sib200 \
  --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42 \
            checkpoints/gpt2_top_doc/edu_20M_ep1_seed42 \
            checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42 \
            checkpoints/gpt2_top_doc/avg5_20M_ep1_seed42 \
  --comparison-file checkpoints/gpt2_top_doc/wiki_sib200_comparison.md \
  --device cuda

echo "Done: checkpoints/gpt2_top_doc/{random,edu,avg4,avg5}_20M_ep1_seed42/wiki_sib200_eval/, wiki_sib200_comparison.md"
