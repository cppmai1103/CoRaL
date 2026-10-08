#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long # Partition is a queue for jobs
#SBATCH --time=4:00:00         # Time limit for the job
#SBATCH --job-name=mlp4096
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1               # Number of nodes you want to run your process on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --qos=gpu-12

set -eo pipefail

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"   # own environment: other projects share "huhu" and changed its transformers version

# Pin CWD to the directory `sbatch` was run from. Don't use $0 here -- on this
# cluster SLURM stages the submitted script into a per-job spool dir
# (/var/spool/slurm/d/job<ID>/) and runs it from there, so $0 resolves to that
# spool copy, not to CoRAL/. $SLURM_SUBMIT_DIR is set by SLURM itself
# to the real submission directory and isn't affected by the staging.
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"


source /apps/local/anaconda3/etc/profile.d/conda.sh


if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  echo "Env '${ENVIRONMENT_NAME}' not found, creating it with python=${PYTHON_VERSION}"
  conda create -n "${ENVIRONMENT_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "Env '${ENVIRONMENT_NAME}' already exists, skipping creation"
fi
conda activate "${ENVIRONMENT_NAME}"

export PYTHONNOUSERSITE=1      # use env torch, not ~/.local's (which is known broken here)
export PYTHONUNBUFFERED=1      # live progress in job-*.out

python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install "transformers==5.14.1" huggingface_hub matplotlib

# The public mmBERT checkpoint is downloaded on first use. Authentication is
# optional; if needed, export HF_TOKEN in the submitting shell.

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'This job requires a CUDA GPU'"

# Compare frozen-encoder MLPs at the fine-tuning run's 4096-token context length.
# Reuse the existing document table and per-dimension splits. The embedding
# builder processes all chunks for longer documents; the current corpus fits
# within 4096 tokens per document. Existing 2048-token runs remain separate.
RATER_MAX_CHUNK_TOKENS=4096
RATER_ENCODER="jhu-clsp/mmBERT-base"
RATER_DOCUMENT_TABLE="data/rater_dataset/document_table.csv"

for RATER_INPUT in "$RATER_DOCUMENT_TABLE" \
  data/rater_dataset/split_manifest_educational_value.csv \
  data/rater_dataset/split_manifest_reasoning.csv \
  data/rater_dataset/split_manifest_professionalism.csv \
  data/rater_dataset/split_manifest_cleanliness.csv \
  data/rater_dataset/split_manifest_cultural_nuances.csv; do
  if [ ! -f "$RATER_INPUT" ]; then
    echo "Missing required input: $RATER_INPUT" >&2
    exit 1
  fi
done

for RATER_POOLING in cls mean; do
  RATER_EMBEDDING_DIR="data/rater_dataset/embeddings_${RATER_POOLING}_${RATER_MAX_CHUNK_TOKENS}"
  RATER_OUTPUT_DIR="checkpoints/rater/frozen/mlp_${RATER_POOLING}_${RATER_MAX_CHUNK_TOKENS}"
  echo "Building $RATER_POOLING embeddings at $RATER_MAX_CHUNK_TOKENS tokens: $RATER_EMBEDDING_DIR"

  python -m src.train_rater.build_embeddings \
    --document-table "$RATER_DOCUMENT_TABLE" \
    --encoder "$RATER_ENCODER" \
    --pooling "$RATER_POOLING" \
    --max-chunk-tokens "$RATER_MAX_CHUNK_TOKENS" \
    --dtype float32 \
    --output-dir "$RATER_EMBEDDING_DIR" \
    --device cuda

  # Keep the original MLP recipe fixed for both poolings (training seed 42 is
  # fixed in src.train_rater.train). Only the encoder context length changes
  # relative to the corresponding 2048-token frozen-encoder baseline.
  echo "Training all five $RATER_POOLING MLP heads: $RATER_OUTPUT_DIR"
  python -m src.train_rater.train \
    --dimension all \
    --split-manifest-dir data/rater_dataset \
    --embeddings "$RATER_EMBEDDING_DIR/embeddings.pt" \
    --output-dir "$RATER_OUTPUT_DIR" \
    --hidden-size 128 \
    --dropout 0.1 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --batch-size 64 \
    --max-epochs 30 \
    --patience 3 \
    --min-delta 0.001 \
    --grad-clip-norm 1.0 \
    --device cuda
done
