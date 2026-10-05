#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=1:00:00
#SBATCH --job-name=eval-downstream
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Zero-shot downstream evaluation (docs/03_downstream_evaluation.md) of the 20M-token-budget GPT-2
# pilot checkpoints: random, edu, avg4, avg5. Belebele (reading comprehension, all 6 languages) and
# XCOPA (cause/effect, only indo/thai/vie have this dataset) only -- SIB-200 is commented out below
# (avg4/avg5 were consistently below random there, the one non-chance-level result across all three
# benchmarks so far; re-enable it by uncommenting --benchmarks if you want it back).
# 500 random examples per language, seed 42, full-candidate-text scoring. XCOPA's test split is
# exactly 500 examples/language, so this uses its entire test set, not a 500-of-many sample.
#
# Run: sbatch sh/eval_downstream.sh

PYTHON_VERSION=3.11
# Other runs (first = baseline): RUN_DIRS="..." COMPARISON_FILE=... sbatch sh/eval_downstream.sh
RUN_DIRS="${RUN_DIRS:-checkpoints/gpt2_top_doc/random_20M_ep1_seed42 checkpoints/gpt2_top_doc/edu_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg5_20M_ep1_seed42}"
COMPARISON_FILE="${COMPARISON_FILE:-checkpoints/gpt2_top_doc/downstream_comparison.md}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

pip install pyarrow pandas >/dev/null  # XCOPA ships as parquet

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.eval_downstream \
  --run-dir $RUN_DIRS \
  --benchmarks belebele xcopa \
  --comparison-file "$COMPARISON_FILE" \
  --num-samples 500 \
  --device cuda
  # --benchmarks belebele xcopa sib200 \  # swap in this line (and comment out the one above) for SIB-200 too

echo "Done: checkpoints/gpt2_top_doc/{random,edu,avg4,avg5}_20M_ep1_seed42/downstream_eval/, downstream_comparison.md"
