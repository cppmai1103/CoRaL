#!/bin/bash

# CPU-only counterpart of script.sh -- for running the same active pipeline steps
# directly on a node with no GPU (e.g. a login/head node), no SLURM submission
# involved (run with `bash script_cpu.sh`, not `sbatch`). torch auto-detects
# cuda.is_available()==False and every phase2 script (evaluate.py, train.py) already
# falls back to CPU on its own -- this file just skips the SBATCH directives and
# nvidia-smi/GPU checks that don't apply here.
#
# Mirrors whichever steps are currently active (uncommented) in script.sh -- keep the
# two in sync. build_candidate_windows.py/tokenize_windows.py/train.py are commented
# out here too since checkpoints/hipe2020_fr/gliner/phase2/mbert_mlp.pt and
# data/hipe2020_fr/gliner/data_phase2/phase2_candidate_windows.jsonl already exist.

set -e

# sbatch runs a spooled copy of this file, so use the submit dir (project root) there; with plain
# `bash sh/data/script_cpu.sh` fall back to this file's parent dir. Either way cwd = project root.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}"
echo "Working directory: $(pwd)"

ENVIRONMENT_NAME="huhu"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# HF_TOKEN lives in the gitignored project-root .env (chmod 600), never in this script.
source .env

# python upload_old_to_hf.py cppmai/neurons-location-ex1 

# python 01_prepare_data.py

# python -m src.download_from_huggingface

python -m src.upload_to_huggingface