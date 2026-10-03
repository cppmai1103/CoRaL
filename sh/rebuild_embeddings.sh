#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=2:00:00
#SBATCH --job-name=rebuild-embeddings
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Regenerates the 4 frozen mmBERT embedding caches under data/rater_dataset/ that were
# accidentally deleted: embeddings (mean@2048, the default), embeddings_cls (cls@2048),
# embeddings_mean_4096 (mean@4096), embeddings_cls_4096 (cls@4096). Deterministic given the
# same encoder/document_table.csv, so this reconstructs them exactly as they were built
# originally -- nothing else needs to change, checkpoints/rater/frozen/ already has its
# trained MLP heads and doesn't depend on this cache existing.
#
# Run: sbatch sh/rebuild_embeddings.sh

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo "=== mean @ 2048 (embeddings/) ==="
python -m src.train_rater.build_embeddings --pooling mean --max-chunk-tokens 2048 --device cuda

echo "=== cls @ 2048 (embeddings_cls/) ==="
python -m src.train_rater.build_embeddings --pooling cls --max-chunk-tokens 2048 --device cuda

echo "=== mean @ 4096 (embeddings_mean_4096/) ==="
python -m src.train_rater.build_embeddings --pooling mean --max-chunk-tokens 4096 --device cuda

echo "=== cls @ 4096 (embeddings_cls_4096/) ==="
python -m src.train_rater.build_embeddings --pooling cls --max-chunk-tokens 4096 --device cuda

echo "Done: data/rater_dataset/{embeddings,embeddings_cls,embeddings_mean_4096,embeddings_cls_4096}"
