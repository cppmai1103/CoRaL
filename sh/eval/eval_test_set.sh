#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH --job-name=eval-test-set
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# Loss, perplexity and bits/byte of the weighted-loss GPT-2 runs on the pilot corpus held-out TEST
# split (1,000 documents/language, never trained on), next to the unweighted baseline trained on the
# same random_20M documents. The first run in RUN_DIRS is the baseline for the Δ columns.
# Per run: <run>/test_eval/{results.json,summary.md}; all runs: $COMPARISON_FILE.
# Loss/perplexity are per token (compare same-tokenizer runs only); bits/byte compares across tokenizers.
# See src/train_gpt2_from_scratch/eval_test_set.py. A few minutes per model on the A40.
#
# Run (SeaLLM runs):  sbatch sh/eval/eval_test_set.sh
# Custom tokenizer runs (after TOKENIZER=checkpoints/tokenizers/sea_bpe_16k sbatch sh/train/train_gpt2_weighted.sh):
#   RUN_ROOT=checkpoints/gpt2_weighted_loss_sea_bpe_16k BASELINE=checkpoints/gpt2_top_doc_sea_bpe_16k/random_20M_ep1_seed42 \
#     sbatch sh/eval/eval_test_set.sh
# Any set of runs, e.g. both tokenizers in one table (compare them by bits/byte):
#   RUN_DIRS="checkpoints/gpt2_top_doc/random_20M_ep1_seed42 checkpoints/gpt2_top_doc_sea_bpe_16k/random_20M_ep1_seed42" \
#     COMPARISON_FILE=checkpoints/tokenizer_test_comparison.md sbatch sh/eval/eval_test_set.sh

RUN_ROOT="${RUN_ROOT:-checkpoints/gpt2_weighted_loss}"
BASELINE="${BASELINE:-checkpoints/gpt2_top_doc/random_20M_ep1_seed42}"
SCORE_METHODS="${SCORE_METHODS:-edu avg4 avg5}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
if [ -z "$RUN_DIRS" ]; then
  RUN_DIRS="$BASELINE"
  for SCORE_METHOD in $SCORE_METHODS; do
    RUN_DIRS="$RUN_DIRS $RUN_ROOT/${SCORE_METHOD}_20M_ep${EPOCHS}_seed${SEED}"
  done
fi
COMPARISON_FILE="${COMPARISON_FILE:-$RUN_ROOT/test_comparison.md}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | RUN_DIRS=$RUN_DIRS | COMPARISON_FILE=$COMPARISON_FILE"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.eval_test_set \
  --run-dir $RUN_DIRS \
  --split test \
  --comparison-file "$COMPARISON_FILE" \
  --device cuda

echo "Done: <run>/test_eval/ for each run, $COMPARISON_FILE"
