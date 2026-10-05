#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --job-name=select-llm
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32GB
#SBATCH --qos=normal

set -eo pipefail

# After the 5 LLM-label raters are trained and have scored the pool (sh/finetune_llm.sh, sh/score_pool_llm.sh):
#   1. compare LLM-label vs human-label raters on the human test split -> checkpoints/rater_llm/rater_comparison.md
#   2. average the dimensions: avg4 (no cultural_nuances) and avg5 -> data/pilot_scores_llm/{avg4,avg5}_mean
#   3. select the top TARGET_TOKENS SeaLLM tokens per language for edu/avg4/avg5, the same way as the
#      human-rater 20M selections -> data/pilot_selected/llm_{edu,avg4,avg5}_20M
# CPU only (SeaLLM tokenizer for token counting). Then train each with sh/train_gpt2.sh METHOD=llm_<m>_20M.
#
# Run: sbatch sh/select_llm.sh

SCORES_ROOT="${SCORES_ROOT:-data/pilot_scores_llm}"
TARGET_TOKENS="${TARGET_TOKENS:-20000000}"
BUDGET="${BUDGET:-20M}"
GROUP_SIZE="${GROUP_SIZE:-5}"
DIMS4="educational_value reasoning professionalism cleanliness"
DIMS5="$DIMS4 cultural_nuances"
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | SCORES_ROOT=$SCORES_ROOT | TARGET_TOKENS=$TARGET_TOKENS"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# combine_scores.py silently keeps only languages present in every folder, so check all 6 are scored first.
for D in $DIMS5; do
  N=0
  for F in "$SCORES_ROOT/${D}_mean"/*.csv; do
    [ -e "$F" ] || continue
    case "$(basename "$F")" in thresholds.csv|dimension_correlations.csv) ;; *) N=$((N + 1)) ;; esac
  done
  if [ "$N" -lt 6 ]; then
    echo "ERROR: $SCORES_ROOT/${D}_mean has scores for $N of 6 languages -- rerun: DIMENSION=$D sbatch sh/score_pool_llm.sh" >&2
    exit 1
  fi
done

echo "=== 1. LLM-label vs human-label raters on the human test split ==="
python -m src.train_rater.compare_raters \
  --raters human=checkpoints/rater/finetuned/mean llm=checkpoints/rater_llm/finetuned/mean \
  --output checkpoints/rater_llm/rater_comparison.md

echo "=== 2. combine dimensions ==="
DIRS4=""; for D in $DIMS4; do DIRS4="$DIRS4 $SCORES_ROOT/${D}_mean"; done
DIRS5=""; for D in $DIMS5; do DIRS5="$DIRS5 $SCORES_ROOT/${D}_mean"; done
python -m src.train_gpt2_from_scratch.combine_scores --scores-dirs $DIRS4 --output-dir "$SCORES_ROOT/avg4_mean"
python -m src.train_gpt2_from_scratch.combine_scores --scores-dirs $DIRS5 --output-dir "$SCORES_ROOT/avg5_mean"

echo "=== 3. select the top $TARGET_TOKENS tokens per language ==="
for PAIR in edu:educational_value_mean avg4:avg4_mean avg5:avg5_mean; do
  M="${PAIR%%:*}"; SRC="${PAIR#*:}"
  python -m src.train_gpt2_from_scratch.prepare_data \
    --method top-score \
    --scores-dir "$SCORES_ROOT/$SRC" \
    --output-dir "data/pilot_selected/llm_${M}_${BUDGET}" \
    --select-by tokens --target-tokens "$TARGET_TOKENS" --group-size "$GROUP_SIZE" \
    --compare-with "data/pilot_selected/random_${BUDGET}/summary.json"
done

echo "Done: checkpoints/rater_llm/rater_comparison.md, $SCORES_ROOT/{avg4,avg5}_mean, data/pilot_selected/llm_{edu,avg4,avg5}_${BUDGET}"
