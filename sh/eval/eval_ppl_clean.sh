#!/bin/bash

# Loss / perplexity / bits per byte on clean, human-translated text, one GPU, no SLURM needed
# (src/train_gpt2_from_scratch/eval_wiki_sib200.py --datasets ...):
#   bible  the whole Bible, one open translation per language, one document per chapter (no Khmer; Malay: NT only)
#   ntrex  NTREX-128 news, all 123 articles per language
#   alt    ALT news, 1,000 articles per language (the same articles in every language)
#   flores_wikibooks / flores_wikivoyage   FLORES+ dev+devtest Wikibooks / Wikivoyage passages, one document per source
#          page (~179 / ~176 per language; gated dataset: HF_TOKEN in .env; SIB-200 and Belebele reuse FLORES sentences)
# The data is downloaded once to data/eval_sets/{bible,ntrex,alt,flores_plus}/ (here, before scoring), so scoring runs offline.
#
# The first model is the baseline of the comparison table (loss difference); bits per byte compares any models.
#   -> <run>/$OUT_NAME/ per model (Hub models: checkpoints/hub_models/<name>/$OUT_NAME/)
#   -> $COMPARISON_FILE
#
# Run from the project root (background, survives logout):
#   nohup bash sh/eval/eval_ppl_clean.sh > checkpoints/lora_cpt/OLMo-1B-hf/ppl_clean.log 2>&1 &
# Other models / GPU / datasets:
#   MODELS="<run>/final allenai/OLMo-1B-hf" GPU=1 DATASETS="ntrex alt" COMPARISON_FILE=... bash sh/eval/eval_ppl_clean.sh

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

R=checkpoints/lora_cpt/OLMo-1B-hf
MODELS="${MODELS:-$R/random_50M_7languages_ep1_seed42/final allenai/OLMo-1B-hf $R/wavg5_50M_7languages_ep1_seed42/final}"
DATASETS="${DATASETS:-bible ntrex alt flores_wikibooks flores_wikivoyage}"
ALT_ARTICLES="${ALT_ARTICLES:-1000}"
GPU="${GPU:-0}"
OUT_NAME="${OUT_NAME:-ppl_clean}"
COMPARISON_FILE="${COMPARISON_FILE:-$R/ppl_clean_comparison.md}"
PYTHON="${PYTHON:-.venv/sea-rater/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python   # cluster: the activated sea-rater conda env

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
[ -f .env ] && { set -a; source .env; set +a; }   # HF_TOKEN (gated models)
ARGS=(--datasets $DATASETS --alt-articles "$ALT_ARTICLES" --pilot-corpus-dir data/pilot_corpus_7languages)

echo "=== prepare: download $DATASETS (once) ==="
"$PYTHON" -m src.train_gpt2_from_scratch.eval_wiki_sib200 --prepare-only "${ARGS[@]}" 2>&1 | grep -vE "wikipedia/"

echo "=== score $(date '+%F %T') on GPU $GPU:$MODELS ==="
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON" -m src.train_gpt2_from_scratch.eval_wiki_sib200 "${ARGS[@]}" --model $MODELS \
  --out-name "$OUT_NAME" --comparison-file "$COMPARISON_FILE" --no-progress --device cuda
echo "Done $(date '+%F %T'): $COMPARISON_FILE"
