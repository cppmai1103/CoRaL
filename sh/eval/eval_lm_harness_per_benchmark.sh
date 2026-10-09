#!/bin/bash

# sh/eval/eval_lm_harness.sh one benchmark at a time: each benchmark is its own process (score -> save its results ->
# exit, which frees all GPU and CPU memory) before the next starts, so a crash loses only the benchmark it happened in
# and a rerun skips the benchmarks already done. Every benchmark is scored exactly as in one combined run (holdout and
# subsets are seeded per task).
#
#   -> $OUTPUT_DIR/<benchmark>/{summary.md,summary.csv,results.json,...}   as each benchmark finishes
#   -> $OUTPUT_DIR/{summary.md,summary.csv}                                 all benchmarks merged, at the end
# The merged folder works with `eval_lm_harness compare` like a single run.
#
# Run from the project root (no SLURM), e.g.:
#   MODEL=checkpoints/lora_cpt/OLMo-1B-hf/random_50M_7languages_ep1_seed42/final \
#   OUTPUT_DIR=checkpoints/lora_cpt/OLMo-1B-hf/random_50M_7languages_ep1_seed42/lm_eval_all_fullmmlu_5shot \
#   GPU=0 FULL=global_mmlu nohup bash sh/eval/eval_lm_harness_per_benchmark.sh > <log> 2>&1 &
# Every other setting (NUM_FEWSHOT, SHOTS, FULL, BATCH_SIZE, LIMIT, ...) is passed on to eval_lm_harness.sh.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:?set MODEL to a Hub ID or a merged model folder}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
# smallest first, so results arrive early; full Global-MMLU (~14k questions x 4 languages) last
BENCHMARK_LIST="${BENCHMARK_LIST:-belebele global_piqa include sib200 sea_nli_normal sea_nli_hard global_mmlu}"
export BATCH_SIZE="${BATCH_SIZE:-16}"   # auto picked 51 for OLMo 1B and ran out of memory on an A100 40GB
export MODEL GPU NUM_FEWSHOT HOLDOUT SEED SHOTS LANGUAGES LIMIT FULL

FAILED=""
for B in $BENCHMARK_LIST; do
  OUT="$OUTPUT_DIR/$B"
  if [ -f "$OUT/summary.csv" ]; then echo "=== $B: done already ($OUT), skipped"; continue; fi
  echo "=== $B: start $(date '+%F %T') -> $OUT"
  # its own process: everything it held on the GPU and in RAM is released when it exits
  if BENCHMARKS="$B" OUTPUT_DIR="$OUT" bash sh/eval/eval_lm_harness.sh; then
    echo "=== $B: done $(date '+%F %T')"
  else
    echo "=== $B: FAILED $(date '+%F %T') -- continuing with the next benchmark; rerun this script to retry it" >&2
    FAILED="$FAILED $B"
  fi
done

echo "=== merge -> $OUTPUT_DIR/summary.md"
source .venv/lm-eval/bin/activate
python - "$OUTPUT_DIR" "$MODEL" $BENCHMARK_LIST <<'EOF'
import csv, sys
from pathlib import Path
from src.train_gpt2_from_scratch.eval_lm_harness import write_summary

out, model, benches = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
rows, done = [], []
for b in benches:
    path = out / b / "summary.csv"
    if not path.is_file():
        continue
    done.append(b)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["score"] = float(r["score"]) if r["score"] not in ("", "None") else None
            r["stderr"] = float(r["stderr"]) if r["stderr"] not in ("", "None") else None
            r["n"], r["shots"] = int(r["n"]), int(r["shots"])
            rows.append(r)
if rows:
    missing = [b for b in benches if b not in done]
    write_summary(out, rows, f"# lm-evaluation-harness results: {model} (one run per benchmark: {', '.join(done)})"
                  + (f"\n\nNOT DONE: {', '.join(missing)}" if missing else ""))
EOF
[ -z "$FAILED" ] || { echo "Failed:$FAILED (rerun to retry; finished benchmarks are skipped)" >&2; exit 1; }
echo "Done: $OUTPUT_DIR/summary.md"
