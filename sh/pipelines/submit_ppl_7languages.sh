#!/bin/bash

# Perplexity (loss, PPL, bits per byte) of the 7-language LoRA continued-pretraining runs and the untrained base model on
#   wikipedia     1,000 eligible Wikipedia articles per language (20231101 dump), none in data/pilot_corpus_7languages
#   annotated_hq  (default) the high-quality human-annotated SEA-Rater documents (data/rater_dataset_7languages, none in
#                 the training pool): top 50% per language by human avg5, and top 50% by human cleanliness (HQ_BY)
#   annotated     the same, plus all ~970 annotated documents per language
# See src/train_gpt2_from_scratch/eval_wiki_sib200.py.
#
#   prepare  (login node) sample the Wikipedia articles once into data/eval_sets/ (downloads ~2 GB of Wikipedia
#            parquet the first time) and cache the base model; the GPU job then runs offline
#   eval     ONE GPU job scoring every model in turn (first = baseline for the loss differences)
#            -> checkpoints/lora_cpt/<model>/<run>/$OUT_NAME/ (base: checkpoints/hub_models/<model>/$OUT_NAME/)
#            -> $COMPARISON_FILE (default checkpoints/lora_cpt/<model>/ppl_comparison.md)
#
# Run from the project root, on the login node:  bash sh/pipelines/submit_ppl_7languages.sh
# Default: only the high-quality annotated documents. Some methods only: METHODS="random avg5" bash ...
# More sets: DATASETS="wikipedia annotated sib200" TIME=6:00:00 bash ...
# Wikipedia apart from the annotated run (own folders, own comparison):
#   DATASETS=wikipedia OUT_NAME=ppl_wikipedia COMPARISON_FILE=checkpoints/lora_cpt/gemma-3-1b-pt/ppl_wikipedia_comparison.md bash ...
# Other partition: PARTITION=cscc-gpu-p bash ...

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

METHODS="${METHODS:-random base edu cult avg5 rr5}"   # first = baseline
BUDGET="${BUDGET:-50M}"
SUFFIX=_7languages
MODEL="${MODEL:-google/gemma-3-1b-pt}"
SEED="${SEED:-42}"                                     # training seed (run folder name)
DATASETS="${DATASETS:-annotated_hq}"   # also: wikipedia, annotated (all + high quality), sib200
HQ_BY="${HQ_BY:-avg5 cleanliness}"     # one high-quality set per human score (top 50% per language each)
PILOT_CORPUS_DIR=data/pilot_corpus_7languages          # Wikipedia articles found in the training pool are excluded
TIME="${TIME:-3:00:00}"
PARTITION="${PARTITION:-long}"
declare -A PARTITION_QOS=([long]=gpu-12 [cscc-gpu-p]=cscc-gpu-qos)
QOS="${QOS:-${PARTITION_QOS[$PARTITION]}}"
EXCLUDE="${EXCLUDE:-gpu-05,gpu-12,gpu-15,gpu-51,gpu-59}"   # nodes whose GPUs failed earlier jobs
ROOT="checkpoints/lora_cpt/${MODEL##*/}"
OUT_NAME="${OUT_NAME:-wiki_sib200_eval}"               # per-model output folder
COMPARISON_FILE="${COMPARISON_FILE:-$ROOT/ppl_comparison.md}"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate sea-rater
set -a; source .env; set +a   # HF_TOKEN
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # shared with the compute nodes (login /tmp is node-local)

MODELS=""
for M in $METHODS; do
  if [ "$M" = base ]; then MODELS="$MODELS $MODEL"; continue; fi
  FINAL="$ROOT/${M}_${BUDGET}${SUFFIX}_ep1_seed${SEED}/final"
  if [ -f "$FINAL/config.json" ]; then MODELS="$MODELS $FINAL"; else echo "eval   : $M skipped (no $FINAL)"; fi
done

echo "=== prepare: Wikipedia sample + $MODEL ==="
if [[ " $DATASETS " == *" wikipedia "* ]]; then
  python -m src.train_gpt2_from_scratch.eval_wiki_sib200 --prepare-only --pilot-corpus-dir "$PILOT_CORPUS_DIR"
fi
if [[ " $DATASETS " == *" sib200 "* ]]; then
  python -c "from src.train_gpt2_from_scratch.eval_wiki_sib200 import load_sib200_text as f, SIB200_LANGUAGES as L; [f(l) for l in L]; print('cached sib200')"
fi
python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1]); print('cached', sys.argv[1])" "$MODEL"

JOB=$(RUN_DIRS="" MODELS="${MODELS# }" DATASETS="$DATASETS" HQ_BY="$HQ_BY" OUT_NAME="$OUT_NAME" PILOT_CORPUS_DIR="$PILOT_CORPUS_DIR" \
      COMPARISON_FILE="$COMPARISON_FILE" OFFLINE=1 \
      sbatch --parsable --partition="$PARTITION" --qos="$QOS" --time="$TIME" --exclude="$EXCLUDE" \
      --job-name=ppl-7l sh/eval/eval_wiki_sib200.sh)
echo "eval   : job $JOB ->$MODELS"
echo "compare: $COMPARISON_FILE (written by the same job when it ends)"
echo "Check progress: squeue -u $USER; tail -f job-$JOB.out"
