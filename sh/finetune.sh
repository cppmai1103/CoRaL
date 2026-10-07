#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=3:00:00          # test mode; for MODE=full override: sbatch --time=12:00:00 ...
#SBATCH --job-name=finetune
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# MODE=test (default): one dimension, 2 epochs, both poolings. Checks that
#   fine-tuning fits in GPU memory at 4096 tokens, how long an epoch takes, and
#   that validation error improves. Weights are not saved.
# MODE=full: all 5 dimensions, both poolings, up to 10 epochs each.
#   sbatch --time=12:00:00 --export=ALL,MODE=full sh/finetune.sh
MODE="${MODE:-test}"
TEST_DIMENSION="${TEST_DIMENSION:-reasoning}"
# DATASET_DIR: output of src.prepare_dataset; OUTPUT_ROOT: checkpoints go to $OUTPUT_ROOT/<pooling>.
#   sbatch --time=12:00:00 --export=ALL,DATASET_DIR=data/rater_dataset_7languages,OUTPUT_ROOT=checkpoints/rater_7languages/finetuned sh/finetune.sh
DATASET_DIR="${DATASET_DIR:-data/rater_dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/rater/finetuned}"

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"   # own environment: other projects share "huhu" and changed its transformers version

# $SLURM_SUBMIT_DIR is the real submission dir; $0 points to SLURM's spool copy.
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | MODE=$MODE | DATASET_DIR=$DATASET_DIR | OUTPUT_ROOT=$OUTPUT_ROOT"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  echo "Env '${ENVIRONMENT_NAME}' not found, creating it with python=${PYTHON_VERSION}"
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
else
  echo "Env '${ENVIRONMENT_NAME}' already exists, skipping creation"
fi
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1      # use env torch, not ~/.local's (which is known broken here)
export PYTHONUNBUFFERED=1      # live progress in job-*.out

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.14.1" huggingface_hub matplotlib

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# mmBERT-base is downloaded from the Hugging Face Hub on first use (public model,
# no token needed). Lower --micro-batch-size or add --gradient-checkpointing if
# a run hits CUDA out-of-memory; the effective batch stays 16.

# if [ "$MODE" = "test" ]; then
#   for POOLING in cls mean; do
#     echo "=== TEST: $TEST_DIMENSION, pooling=$POOLING ==="
#     python -m src.train_rater.finetune \
#       --dimension "$TEST_DIMENSION" \
#       --pooling "$POOLING" \
#       --max-epochs 2 \
#       --patience 2 \
#       --no-save-model \
#       --output-dir "checkpoints/rater/finetuned/$POOLING" \
#       --device cuda
#   done
# elif [ "$MODE" = "full" ]; then
#   for POOLING in cls mean; do
#     echo "=== FULL: all dimensions, pooling=$POOLING ==="
#     python -m src.train_rater.finetune \
#       --dimension all \
#       --pooling "$POOLING" \
#       --output-dir "checkpoints/rater/finetuned/$POOLING" \
#       --device cuda
#   done
# else
#   echo "Unknown MODE=$MODE (use test or full)"; exit 1
# fi

for POOLING in mean; do
  for DIMENSION in all; do
    echo "Fine-tuning $DIMENSION, pooling=$POOLING"
    python -m src.train_rater.finetune \
      --dimension "$DIMENSION" \
      --pooling "$POOLING" \
      --document-table "$DATASET_DIR/document_table.csv" \
      --split-manifest-dir "$DATASET_DIR" \
      --output-dir "$OUTPUT_ROOT/$POOLING" \
      --device cuda
  done
done
