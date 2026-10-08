#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=2:00:00
#SBATCH --job-name=eval-ppl
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# Wikipedia/SIB-200 loss+perplexity evaluation (docs/04_wikipedia_sib200_evaluation.md) of the
# 20M-token-budget GPT-2 pilot checkpoints: random, edu, avg4, avg5. Wikipedia: 1,000 eligible,
# training-deduplicated articles/language (all 6 languages). SIB-200: all 1,004 sentences/language
# pooled across train+dev+test, used as a text corpus (not for topic classification). The first
# --run-dir (random) is treated as the baseline for loss_diff_vs_random.
#
# First run downloads ~1.85GB of Wikipedia parquet (cached under ~/.cache/huggingface for the
# remaining 3 checkpoints in this same job).
#
# Run: sbatch sh/eval/eval_wiki_sib200.sh

PYTHON_VERSION=3.11
# Other runs (first = baseline): RUN_DIRS="..." COMPARISON_FILE=... sbatch sh/eval/eval_wiki_sib200.sh
# Pretrained / merged LoRA models instead (first = baseline), e.g. the 7-language LoRA runs, as
# sh/pipelines/submit_ppl_7languages.sh does: RUN_DIRS="" MODELS="<run>/final google/gemma-3-1b-pt ..." \
#   DATASETS="wikipedia annotated" PILOT_CORPUS_DIR=data/pilot_corpus_7languages COMPARISON_FILE=... sbatch ...
# The Wikipedia sample must already be saved (data/eval_sets/, --prepare-only on the login node) when OFFLINE=1.
RUN_DIRS="${RUN_DIRS-checkpoints/gpt2_top_doc/random_20M_ep1_seed42 checkpoints/gpt2_top_doc/edu_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg5_20M_ep1_seed42}"
MODELS="${MODELS:-}"
DATASETS="${DATASETS:-wikipedia sib200}"
LANGUAGES="${LANGUAGES:-}"
OUT_NAME="${OUT_NAME:-}"              # per-model output folder name (default wiki_sib200_eval)
HQ_BY="${HQ_BY:-}"                    # human score(s) defining the high-quality annotated documents, e.g. "avg5 cleanliness"
PILOT_CORPUS_DIR="${PILOT_CORPUS_DIR:-data/pilot_corpus}"
COMPARISON_FILE="${COMPARISON_FILE:-checkpoints/gpt2_top_doc/wiki_sib200_comparison.md}"
OFFLINE="${OFFLINE:-}"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | RUN_DIRS=$RUN_DIRS | MODELS=$MODELS | DATASETS=$DATASETS"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
[ -f .env ] && { set -a; source .env; set +a; }   # HF_TOKEN (Gemma is gated)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[ -n "$OFFLINE" ] && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

ARGS=(--datasets $DATASETS --pilot-corpus-dir "$PILOT_CORPUS_DIR" --comparison-file "$COMPARISON_FILE" --device cuda)
[ -n "$RUN_DIRS" ] && ARGS+=(--run-dir $RUN_DIRS)
[ -n "$MODELS" ] && ARGS+=(--model $MODELS)
[ -n "$LANGUAGES" ] && ARGS+=(--languages $LANGUAGES)
[ -n "$HQ_BY" ] && ARGS+=(--hq-by $HQ_BY)
[ -n "$OUT_NAME" ] && ARGS+=(--out-name "$OUT_NAME")
python -m src.train_gpt2_from_scratch.eval_wiki_sib200 "${ARGS[@]}"

echo "Done: $COMPARISON_FILE"
