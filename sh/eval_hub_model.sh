#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=3:00:00
#SBATCH --job-name=ev-hub
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Evaluate a pretrained Hugging Face model (default google/gemma-3-270m) with the same two suites as the GPT-2
# pilot runs, next to reference runs (RUN_DIRS, default the random_20M baseline):
#   1. zero-shot accuracy: Belebele (6 languages) + XCOPA (indo/thai/vie), 500 examples/language
#   2. Wikipedia (1,000 articles/language) + SIB-200 (1,004 sentences/language): loss, PPL and bits per byte
# Loss/PPL are per token and differ by tokenizer (Gemma 262K vs SeaLLM 48K): compare the models by
# BITS PER BYTE and by accuracy. The pretrained model's context starts with its BOS token (<bos> for Gemma).
# Gemma is gated: HF_TOKEN comes from the gitignored project-root .env.
# Outputs: checkpoints/hub_models/<model name>/{downstream_eval,wiki_sib200_eval}/ and
#          checkpoints/hub_models/{downstream_comparison,wiki_sib200_comparison}.md
#
# Run: sbatch sh/eval_hub_model.sh
#      MODELS="google/gemma-3-270m google/gemma-3-1b-pt" RUN_DIRS="checkpoints/gpt2_top_doc/random_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42" sbatch sh/eval_hub_model.sh

MODELS="${MODELS:-google/gemma-3-270m}"
RUN_DIRS="${RUN_DIRS:-checkpoints/gpt2_top_doc/random_20M_ep1_seed42}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
OUT_ROOT="${OUT_ROOT:-checkpoints/hub_models}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | MODELS=$MODELS | RUN_DIRS=$RUN_DIRS"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
set -a; source .env; set +a   # HF_TOKEN for gated models

pip install pyarrow pandas >/dev/null  # XCOPA/Wikipedia parquet

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo "=== zero-shot accuracy (Belebele/XCOPA) ==="
python -m src.train_gpt2_from_scratch.eval_downstream \
  --run-dir $RUN_DIRS \
  --model $MODELS \
  --benchmarks belebele xcopa \
  --num-samples "$NUM_SAMPLES" \
  --comparison-file "$OUT_ROOT/downstream_comparison.md" \
  --device cuda

echo "=== Wikipedia/SIB-200 loss, PPL, bits per byte ==="
python -m src.train_gpt2_from_scratch.eval_wiki_sib200 \
  --run-dir $RUN_DIRS \
  --model $MODELS \
  --comparison-file "$OUT_ROOT/wiki_sib200_comparison.md" \
  --device cuda

echo "Done: $OUT_ROOT/{downstream_comparison,wiki_sib200_comparison}.md"
