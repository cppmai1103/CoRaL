#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --job-name=lm-harness
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# k-shot evaluation of one model with lm-evaluation-harness on the 7 project languages. Suites: core = Belebele,
# Global PIQA, Global-MMLU Full, INCLUDE (docs/03_downstream_evaluation.md); extra = SIB-200, SEA-NLI normal/hard,
# FLORES+ eng->lang (docs/03b_additional_benchmarks.md, custom tasks in src/train_gpt2_from_scratch/lm_eval_tasks). See
# src/train_gpt2_from_scratch/eval_lm_harness.py. Runs in .venv/lm-eval (lm-eval on top of sea-rater; create it once:
#   conda activate sea-rater && python -m venv --system-site-packages .venv/lm-eval && source .venv/lm-eval/bin/activate \
#     && pip install "lm-eval[hf]"   (with torch/transformers pinned to the sea-rater versions)).
# Datasets and model must already be in the shared HF cache (sh/pipelines/submit_lm_harness_7languages.sh prefetches
# them on the login node), because the job runs offline.
#
# Run: MODEL=checkpoints/lora_cpt/gemma-3-1b-pt/avg5_50M_7languages_ep1_seed42/final \
#        OUTPUT_DIR=checkpoints/lora_cpt/gemma-3-1b-pt/avg5_50M_7languages_ep1_seed42/lm_eval_5shot sbatch sh/eval/eval_lm_harness.sh
#      all runs of the 7-language LoRA pipeline: bash sh/pipelines/submit_lm_harness_7languages.sh
# 0-shot instead: NUM_FEWSHOT=0 (nothing held out). Quick check: LIMIT=10 (not a final result).
# Additional benchmarks: BENCHMARKS=extra (or all, or names, e.g. "sib200 flores_plus"); use another OUTPUT_DIR
# (e.g. .../lm_eval_extra_5shot) so the core results are not overwritten. SEA-NLI follows NUM_FEWSHOT
# (0-shot as docs/03b recommends: SHOTS="sea_nli_normal=0 sea_nli_hard=0").
# Per-benchmark k: SHOTS="sea_nli_normal=5 sea_nli_hard=5 flores_plus=1" (overrides NUM_FEWSHOT for those).
# Belebele, Global-MMLU and FLORES+ are scored on fixed subsets (500/language, 100/subject category, 3/topic; the same
# items for every model, see subsets.json); FULL=1 scores the complete test sets, FULL=global_mmlu only that one.
# Also runs without SLURM (bash sh/eval/eval_lm_harness.sh from the project root; GPU=1 picks the GPU), e.g. on a
# rented server with the .venv/lm-eval venv and no /apps conda.

MODEL="${MODEL:?set MODEL to a Hub ID or a merged model folder}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
HOLDOUT="${HOLDOUT:-5}"
SEED="${SEED:-1234}"
BENCHMARKS="${BENCHMARKS:-core}"     # core | extra | all | benchmark names (xcopa disabled for now, see TASKS)
SHOTS="${SHOTS:-}"                   # per-benchmark k, e.g. "sib200=0 flores_plus=1"
LANGUAGES="${LANGUAGES:-burmese fil indo khmer malay thai vie}"
LIMIT="${LIMIT:-}"
FULL="${FULL:-}"
BATCH_SIZE="${BATCH_SIZE:-}"         # fixed batch size; default auto (can pick too large and run out of memory)
ARGS=(--model "$MODEL" --output-dir "$OUTPUT_DIR" --num-fewshot "$NUM_FEWSHOT" --holdout "$HOLDOUT" --seed "$SEED"
      --benchmarks $BENCHMARKS --languages $LANGUAGES)
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$SHOTS" ] && ARGS+=(--shots $SHOTS)
[ -n "$BATCH_SIZE" ] && ARGS+=(--batch-size "$BATCH_SIZE")
if [ "$FULL" = 1 ]; then ARGS+=(--full-test-sets); elif [ -n "$FULL" ]; then ARGS+=(--full-test-sets $FULL); fi

cd "${SLURM_SUBMIT_DIR:-.}"
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"
echo "Working directory: $(pwd) | MODEL=$MODEL | ${NUM_FEWSHOT}-shot | OUTPUT_DIR=$OUTPUT_DIR"

if [ -f /apps/local/anaconda3/etc/profile.d/conda.sh ]; then
  source /apps/local/anaconda3/etc/profile.d/conda.sh
  conda activate sea-rater
fi
source .venv/lm-eval/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # less fragmentation
set -a; source .env; set +a   # HF_TOKEN (Gemma and some benchmark datasets are gated)
# Shared Hugging Face cache: the login shell's XDG_CACHE_HOME (/tmp/$USER/cache) is node-local.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
python -c "import torch, lm_eval; print('torch', torch.__version__, '| lm_eval', lm_eval.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.eval_lm_harness eval "${ARGS[@]}"

echo "Done: $OUTPUT_DIR/summary.md"
