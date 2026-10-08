#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH --job-name=eval-downstream
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# Zero-shot downstream evaluation (docs/03_downstream_evaluation.md) of the 20M-token-budget GPT-2
# pilot checkpoints: random, edu, avg4, avg5. Belebele (reading comprehension, all 6 languages) and
# XCOPA (cause/effect, only indo/thai/vie have this dataset) only -- SIB-200 is commented out below
# (avg4/avg5 were consistently below random there, the one non-chance-level result across all three
# benchmarks so far; re-enable it by uncommenting --benchmarks if you want it back).
# The full official test split of each benchmark (Belebele 900, XCOPA 500 per language), full-candidate-text
# scoring.
#
# Run: sbatch sh/eval/eval_downstream.sh
# Pretrained/LoRA-merged models (scored with their own BOS token) go in MODELS; RUN_DIRS="" for models only:
#   RUN_DIRS="" MODELS="checkpoints/lora_cpt/gemma-3-270m/base/final ..." COMPARISON_FILE=... sbatch sh/eval/eval_downstream.sh
# Few-shot: NUM_SHOTS=5 (or 10) -> <run>/downstream_eval_<K>shot/; MAX_CONTEXT caps --model prompts (default 2048).

PYTHON_VERSION=3.11
# Other runs (first = baseline): RUN_DIRS="..." COMPARISON_FILE=... sbatch sh/eval/eval_downstream.sh
RUN_DIRS="${RUN_DIRS-checkpoints/gpt2_top_doc/random_20M_ep1_seed42 checkpoints/gpt2_top_doc/edu_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg5_20M_ep1_seed42}"
NUM_SHOTS="${NUM_SHOTS:-0}"
SHOT_TAG=$([ "$NUM_SHOTS" -gt 0 ] && echo "_${NUM_SHOTS}shot" || true)
COMPARISON_FILE="${COMPARISON_FILE:-checkpoints/gpt2_top_doc/downstream_comparison${SHOT_TAG}.md}"
MAX_CONTEXT="${MAX_CONTEXT:-}"
MODELS="${MODELS:-}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | NUM_SHOTS=$NUM_SHOTS"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

pip install pyarrow pandas >/dev/null  # XCOPA ships as parquet

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.eval_downstream \
  ${RUN_DIRS:+--run-dir $RUN_DIRS} \
  ${MODELS:+--model $MODELS} \
  ${MAX_CONTEXT:+--max-context $MAX_CONTEXT} \
  --num-shots "$NUM_SHOTS" \
  --benchmarks belebele xcopa \
  --comparison-file "$COMPARISON_FILE" \
  --device cuda
  # --benchmarks belebele xcopa sib200 \  # swap in this line (and comment out the one above) for SIB-200 too

echo "Done: <run>/downstream_eval/ for every run, $COMPARISON_FILE"
