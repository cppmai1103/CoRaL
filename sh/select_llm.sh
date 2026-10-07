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
# Other scores/pool (e.g. the 7-language human rater; also re-selects random_<BUDGET> on that pool, since the
# methods must all come from the same pool, and skips the LLM-vs-human comparison):
#   SCORES_ROOT=data/pilot_scores_7languages POOL_DIR=data/pilot_corpus_7languages N_LANGUAGES=7 \
#     OUT_PREFIX="" OUT_SUFFIX=_7languages SELECT_RANDOM=1 COMPARE_RATERS=0 sbatch sh/select_llm.sh
#   -> data/pilot_selected/{random,edu,avg4,avg5}_20M_7languages

SCORES_ROOT="${SCORES_ROOT:-data/pilot_scores_llm}"
TARGET_TOKENS="${TARGET_TOKENS:-20000000}"
BUDGET="${BUDGET:-20M}"
GROUP_SIZE="${GROUP_SIZE:-5}"
POOL_DIR="${POOL_DIR:-data/pilot_corpus}"
N_LANGUAGES="${N_LANGUAGES:-6}"         # every dimension must be scored for this many languages
OUT_PREFIX="${OUT_PREFIX-llm_}"         # output: data/pilot_selected/<OUT_PREFIX><method>_<BUDGET><OUT_SUFFIX>
OUT_SUFFIX="${OUT_SUFFIX:-}"
SELECT_RANDOM="${SELECT_RANDOM:-0}"     # 1: also select random_<BUDGET><OUT_SUFFIX> from POOL_DIR (seed 42)
COMPARE_RATERS="${COMPARE_RATERS:-1}"   # 1: LLM-label vs human-label rater comparison (step 1)
RANDOM_DIR="data/pilot_selected/random_${BUDGET}${OUT_SUFFIX}"
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

# combine_scores.py silently keeps only languages present in every folder, so check all are scored first.
for D in $DIMS5; do
  N=0
  for F in "$SCORES_ROOT/${D}_mean"/*.csv; do
    [ -e "$F" ] || continue
    case "$(basename "$F")" in thresholds.csv|dimension_correlations.csv) ;; *) N=$((N + 1)) ;; esac
  done
  if [ "$N" -lt "$N_LANGUAGES" ]; then
    echo "ERROR: $SCORES_ROOT/${D}_mean has scores for $N of $N_LANGUAGES languages -- rerun: DIMENSION=$D sbatch sh/score_pool_llm.sh" >&2
    exit 1
  fi
done

if [ "$COMPARE_RATERS" = 1 ]; then
  echo "=== 1. LLM-label vs human-label raters on the human test split ==="
  python -m src.train_rater.compare_raters \
    --raters human=checkpoints/rater/finetuned/mean llm=checkpoints/rater_llm/finetuned/mean \
    --output checkpoints/rater_llm/rater_comparison.md
fi

echo "=== 2. combine dimensions ==="
DIRS4=""; for D in $DIMS4; do DIRS4="$DIRS4 $SCORES_ROOT/${D}_mean"; done
DIRS5=""; for D in $DIMS5; do DIRS5="$DIRS5 $SCORES_ROOT/${D}_mean"; done
python -m src.train_gpt2_from_scratch.combine_scores --scores-dirs $DIRS4 --output-dir "$SCORES_ROOT/avg4_mean"
python -m src.train_gpt2_from_scratch.combine_scores --scores-dirs $DIRS5 --output-dir "$SCORES_ROOT/avg5_mean"

if [ "$SELECT_RANDOM" = 1 ]; then
  echo "=== 3a. random baseline, $TARGET_TOKENS tokens per language -> $RANDOM_DIR ==="
  python -m src.train_gpt2_from_scratch.prepare_data \
    --method random \
    --pool-dir "$POOL_DIR" \
    --output-dir "$RANDOM_DIR" \
    --select-by tokens --target-tokens "$TARGET_TOKENS" --group-size "$GROUP_SIZE"
fi
echo "=== 3. select the top $TARGET_TOKENS tokens per language ==="
for PAIR in edu:educational_value_mean avg4:avg4_mean avg5:avg5_mean; do
  M="${PAIR%%:*}"; SRC="${PAIR#*:}"
  python -m src.train_gpt2_from_scratch.prepare_data \
    --method top-score \
    --pool-dir "$POOL_DIR" \
    --scores-dir "$SCORES_ROOT/$SRC" \
    --output-dir "data/pilot_selected/${OUT_PREFIX}${M}_${BUDGET}${OUT_SUFFIX}" \
    --select-by tokens --target-tokens "$TARGET_TOKENS" --group-size "$GROUP_SIZE" \
    --compare-with "$RANDOM_DIR/summary.json"
done

echo "Done: $SCORES_ROOT/{avg4,avg5}_mean, data/pilot_selected/${OUT_PREFIX}{edu,avg4,avg5}_${BUDGET}${OUT_SUFFIX}"
