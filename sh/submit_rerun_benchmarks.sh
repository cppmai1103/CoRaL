#!/bin/bash

# Re-score every previously evaluated model on the FULL official test split of Belebele (900/language,
# previously a 500 sample) and XCOPA (500/language, unchanged), so all runs are compared on the same questions.
# Two jobs, run in parallel:
#
#   gpt2    GPT-2 pilot runs: random, edu, avg4, avg5 (human rater), llm_edu/avg4/avg5 (LLM rater),
#           weighted-loss edu/avg4/avg5      -> checkpoints/gpt2_top_doc/downstream_comparison_all.md
#           (names that collide, e.g. the two edu_20M runs, are prefixed with their folder)
#   gemma   Gemma 3 270M: untrained base + LoRA runs random/edu/avg4/avg5
#                                            -> checkpoints/lora_cpt/gemma-3-270m/downstream_comparison.md
#
# Each run's <run>/downstream_eval/ is overwritten with the full-split results. Wikipedia/SIB-200 loss is not
# rerun: it never sampled the benchmark test sets.
#
# Few-shot runs (one job per K and group) write <run>/downstream_eval_<K>shot/ and downstream_comparison_<K>shot.md;
# shots are balanced over the answer options and never taken from the scored test items (see eval_downstream.py).
# Gemma runs with an 8192-token window so 10 Belebele passages fit; GPT-2 has 1024 tokens, so it keeps fewer
# Belebele shots (the number used is in each summary.md).
#
# Run from the project root: bash sh/submit_rerun_benchmarks.sh                 (zero-shot)
# Only one group:            SETS="gemma" bash sh/submit_rerun_benchmarks.sh
# 5- and 10-shot:            SHOTS="5 10" SETS="gemma" bash sh/submit_rerun_benchmarks.sh

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SETS="${SETS:-gpt2 gemma}"
SHOTS="${SHOTS:-0}"
tag() { [ "$1" -gt 0 ] && echo "_${1}shot" || true; }
has() { case " $SETS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if has gpt2; then
  TOP=checkpoints/gpt2_top_doc
  WEIGHTED=checkpoints/gpt2_weighted_loss
  RUNS="$TOP/random_20M_ep1_seed42"
  for M in edu avg4 avg5 llm_edu llm_avg4 llm_avg5; do RUNS="$RUNS $TOP/${M}_20M_ep1_seed42"; done
  for M in edu avg4 avg5; do RUNS="$RUNS $WEIGHTED/${M}_20M_ep1_seed42"; done
  for K in $SHOTS; do
    JOB=$(RUN_DIRS="$RUNS" NUM_SHOTS=$K COMPARISON_FILE="$TOP/downstream_comparison_all$(tag $K).md" \
          sbatch --parsable --time=4:00:00 --mem=16G --job-name=bench-gpt2-${K}shot sh/eval_downstream.sh)
    echo "gpt2   10 runs, ${K}-shot: job $JOB -> $TOP/downstream_comparison_all$(tag $K).md"
  done
fi

if has gemma; then
  ROOT=checkpoints/lora_cpt/gemma-3-270m
  MODELS="$ROOT/base/final"
  for M in random edu avg4 avg5; do MODELS="$MODELS $ROOT/${M}_20M_ep1_seed42/final"; done
  for K in $SHOTS; do
    JOB=$(RUN_DIRS="" MODELS="$MODELS" NUM_SHOTS=$K MAX_CONTEXT=8192 \
          COMPARISON_FILE="$ROOT/downstream_comparison$(tag $K).md" \
          sbatch --parsable --time=4:00:00 --mem=16G --job-name=bench-gemma-${K}shot sh/eval_downstream.sh)
    echo "gemma  5 models, ${K}-shot: job $JOB -> $ROOT/downstream_comparison$(tag $K).md"
  done
fi
echo "Check progress: squeue -u $USER"
