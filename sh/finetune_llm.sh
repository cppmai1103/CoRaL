#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=1:30:00
#SBATCH --job-name=finetune-llm
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Fine-tune one mmBERT-base rater (mean pooling) on LLM-score labels instead of human labels:
# data/rater_dataset_llm/ (src/prepare_llm_rater_dataset.py) -- the SAME human-annotated documents and
# per-dimension splits as data/rater_dataset/, with train/validation relabelled by the LLM and test keeping
# the human labels, so the only difference from checkpoints/rater/finetuned/mean is the training label.
# Same code and hyperparameters as sh/finetune.sh (10 epochs, patience 3; ~35 min per dimension).
# Output: checkpoints/rater_llm/finetuned/mean/<dimension>/ (model/, predictions, evaluation_report.md).
#
# Run one dimension per job (all five: bash sh/submit_llm_pipeline.sh):
#   DIMENSION=educational_value sbatch sh/finetune_llm.sh

DIMENSION="${DIMENSION:-educational_value}"
POOLING="${POOLING:-mean}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"   # lower to 2 (or 1) on CUDA out-of-memory; the effective batch stays 16
EXTRA_ARGS=()
[ -n "$GRADIENT_CHECKPOINTING" ] && EXTRA_ARGS+=(--gradient-checkpointing)   # GRADIENT_CHECKPOINTING=1: less memory, slower
DATA_DIR="${DATA_DIR:-data/rater_dataset_llm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/rater_llm/finetuned}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | DIMENSION=$DIMENSION | POOLING=$POOLING | MAX_EPOCHS=$MAX_EPOCHS | DATA_DIR=$DATA_DIR"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_rater.finetune \
  --dimension "$DIMENSION" \
  --pooling "$POOLING" \
  --document-table "$DATA_DIR/document_table.csv" \
  --split-manifest-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_ROOT/$POOLING" \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  "${EXTRA_ARGS[@]}" \
  --device cuda

echo "Done: $OUTPUT_ROOT/$POOLING/$DIMENSION/"
