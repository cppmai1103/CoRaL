#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=2:00:00
#SBATCH --job-name=sea-tokenizer
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32GB
#SBATCH --qos=cscc-cpu-qos

set -eo pipefail

# Train a small byte-level BPE tokenizer for the 6 SEA pilot languages on the TRAIN split of
# data/pilot_corpus only (validation/test never seen), balanced by characters per language, then
# compare it to the SeaLLM v2 tokenizer (48,384) on the validation split -> <output-dir>/report.md.
# CPU only; a few minutes. See src/train_gpt2_from_scratch/train_tokenizer.py.
#
# Run: sbatch sh/data/train_tokenizer.sh                    (16K -> checkpoints/tokenizers/sea_bpe_16k)
#      VOCAB_SIZE=32000 sbatch sh/data/train_tokenizer.sh   (32K -> checkpoints/tokenizers/sea_bpe_32k)
# Then train GPT-2 with it:
#      TOKENIZER=checkpoints/tokenizers/sea_bpe_16k sbatch sh/train/train_gpt2_weighted.sh

VOCAB_SIZE="${VOCAB_SIZE:-16000}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | VOCAB_SIZE=$VOCAB_SIZE"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS="${SLURM_NTASKS_PER_NODE:-8}"  # threads used by the Rust BPE trainer
[ -f .env ] && source .env  # HF_TOKEN, only to download the SeaLLM tokenizer for the comparison

python -m src.train_gpt2_from_scratch.train_tokenizer --vocab-size "$VOCAB_SIZE"
