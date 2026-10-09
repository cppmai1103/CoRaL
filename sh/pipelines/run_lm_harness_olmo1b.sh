#!/bin/bash
# 5-shot accuracy benchmarks (core + extra: Belebele, Global PIQA, Global-MMLU, INCLUDE, SIB-200, SEA-NLI) for the
# OLMo-1B runs: Global-MMLU on the complete test set, Belebele on its fixed 500/language subset. Each run waits for its
# training to write final/, then: random on GPU 0 followed by the base model, wavg5 on GPU 1. Then the comparison.
# Run from the project root inside tmux:  bash sh/pipelines/run_lm_harness_olmo1b.sh
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

R=checkpoints/lora_cpt/OLMo-1B-hf
NAME=lm_eval_all_fullmmlu_5shot

evaluate() {  # GPU, model (Hub ID or <run>/final), run folder
  local gpu=$1 model=$2 run=$3
  mkdir -p "$run"
  if [[ "$model" == checkpoints/* ]]; then
    until [ -f "$model/config.json" ]; do echo "$(date +%T) waiting for $model"; sleep 300; done
  fi
  echo "$(date +%T) eval $model on GPU $gpu -> $run/$NAME"
  GPU=$gpu MODEL=$model OUTPUT_DIR=$run/$NAME BENCHMARKS=all FULL=global_mmlu \
    bash sh/eval/eval_lm_harness.sh > "$run/$NAME.log" 2>&1 \
    && echo "$(date +%T) done $run/$NAME" || echo "$(date +%T) FAILED $model, see $run/$NAME.log"
}

(evaluate 0 $R/random_50M_7languages_ep1_seed42/final $R/random_50M_7languages_ep1_seed42
 evaluate 0 allenai/OLMo-1B-hf $R/base) &
evaluate 1 $R/wavg5_50M_7languages_ep1_seed42/final $R/wavg5_50M_7languages_ep1_seed42 &
wait

RUNS=""
for m in base random_50M_7languages_ep1_seed42 wavg5_50M_7languages_ep1_seed42; do
  [ -f "$R/$m/$NAME/summary.csv" ] && RUNS="$RUNS $R/$m/$NAME"
done
.venv/lm-eval/bin/python -m src.train_gpt2_from_scratch.eval_lm_harness compare --runs $RUNS \
  --baseline $R/random_50M_7languages_ep1_seed42/$NAME --output $R/${NAME}_comparison.md
echo "Comparison: $R/${NAME}_comparison.md"
