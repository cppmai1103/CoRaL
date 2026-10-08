#!/bin/bash

# Submit the 20M top-select GPT-2 runs with the custom 16K SEA tokenizer, plus their evaluation:
#   1. train random_20M (baseline), edu_20M, avg4_20M, avg5_20M      -> one sh/train/train_gpt2.sh job each
#      (same documents as the SeaLLM runs: data/pilot_selected/<method>/documents.csv)
#   2. MODE=init: untrained-model (chance-level) reference           -> one sh/train/train_gpt2.sh job
#   3. test-split loss/perplexity/bits-per-byte of the 4 runs        -> sh/eval/eval_test_set.sh, starts only
#      after all 4 training jobs succeed (SLURM afterok dependency)
# Outputs: checkpoints/gpt2_top_doc_sea_bpe_16k/{random,edu,avg4,avg5}_20M_ep1_seed42/, init_seed42/,
#          test_comparison.md
#
# This only submits jobs; run it with bash (not sbatch), from the project root:
#   bash sh/pipelines/submit_top_select_16k.sh
# Other tokenizer / methods: TOKENIZER=... METHODS="random_20M avg4_20M" bash sh/pipelines/submit_top_select_16k.sh

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TOKENIZER="${TOKENIZER:-checkpoints/tokenizers/sea_bpe_16k}"
METHODS="${METHODS:-random_20M edu_20M avg4_20M avg5_20M}"
SEED="${SEED:-42}"
TAG="$(basename "$TOKENIZER" | tr '[:upper:]' '[:lower:]')"
RUN_ROOT="checkpoints/gpt2_top_doc_${TAG}"

if [ ! -f "$TOKENIZER/tokenizer.json" ]; then
  echo "ERROR: no tokenizer at $TOKENIZER -- train it first: sbatch sh/data/train_tokenizer.sh" >&2
  exit 1
fi
for M in $METHODS; do
  if [ ! -f "data/pilot_selected/$M/documents.csv" ]; then
    echo "ERROR: data/pilot_selected/$M/documents.csv is missing" >&2
    exit 1
  fi
done

TRAIN_JOBS=""
RUN_DIRS=""
for M in $METHODS; do
  JOB=$(METHOD=$M TOKENIZER=$TOKENIZER SEED=$SEED sbatch --parsable --job-name="gpt2-${M}-${TAG}" sh/train/train_gpt2.sh)
  echo "train $M: job $JOB -> $RUN_ROOT/${M}_ep1_seed${SEED}"
  TRAIN_JOBS="${TRAIN_JOBS:+$TRAIN_JOBS:}$JOB"
  RUN_DIRS="$RUN_DIRS $RUN_ROOT/${M}_ep1_seed${SEED}"
done

JOB=$(MODE=init TOKENIZER=$TOKENIZER SEED=$SEED sbatch --parsable --job-name="gpt2-init-${TAG}" sh/train/train_gpt2.sh)
echo "init reference: job $JOB -> $RUN_ROOT/init_seed${SEED}"

JOB=$(RUN_DIRS="${RUN_DIRS# }" COMPARISON_FILE="$RUN_ROOT/test_comparison.md" \
      sbatch --parsable --dependency=afterok:$TRAIN_JOBS --job-name="eval-test-${TAG}" sh/eval/eval_test_set.sh)
echo "test-split evaluation: job $JOB (after $TRAIN_JOBS) -> $RUN_ROOT/test_comparison.md"
echo "Check progress: squeue -u $USER"
