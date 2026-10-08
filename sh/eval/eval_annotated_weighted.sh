#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH --job-name=eval-annot-weighted
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# Loss/perplexity of the quality-weighted-loss GPT-2 runs on the HUMAN-annotated SEA-Rater documents
# (data/rater_dataset/document_table.csv, ~5.7K documents / 6 languages, excluded from the pilot corpus
# so unseen by every GPT-2 run), next to the unweighted baseline trained on the same random_20M
# documents. Two passes (src/train_gpt2_from_scratch/score_annotated_dataset.py):
#   all annotated documents                 -> gpt2_summary.md / gpt2_results.json
#   per language, top HQ_FRACTION by the    -> gpt2_summary_highquality.md / gpt2_results_highquality.json
#   human avg5 score (high quality)
# If loss weighting helps, the weighted runs should close (or reverse) their gap to the baseline on the
# high-quality pass compared with the all-documents pass and the random test split.
# The first run (the baseline) is the reference for the Δ loss columns. Already-evaluated runs are
# skipped on a rerun (add FORCE=1 to redo them).
#
# Run: sbatch sh/eval/eval_annotated_weighted.sh
# Other high-quality definition (HQ_BY = edu/avg4/avg5 or one human dimension; HQ_MIN_SCORE = keep every
# document scoring at least this instead of the top HQ_FRACTION), plus the 20M top-select runs
# (checkpoints/gpt2_top_doc/{edu,avg4,avg5}_20M_ep1_seed42) for comparison -- e.g. cleanliness == 5:
#   HQ_BY=cleanliness HQ_MIN_SCORE=5 INCLUDE_TOP_SELECT=1 OUTPUT_DIR=data/rater_dataset/annotated_eval_cleanliness \
#     sbatch sh/eval/eval_annotated_weighted.sh
# Custom-tokenizer runs:
#   RUN_ROOT=checkpoints/gpt2_weighted_loss_sea_bpe_16k BASELINE=checkpoints/gpt2_top_doc_sea_bpe_16k/random_20M_ep1_seed42 \
#     OUTPUT_DIR=data/rater_dataset/annotated_eval_weighted_sea_bpe_16k sbatch sh/eval/eval_annotated_weighted.sh

RUN_ROOT="${RUN_ROOT:-checkpoints/gpt2_weighted_loss}"
BASELINE="${BASELINE:-checkpoints/gpt2_top_doc/random_20M_ep1_seed42}"
SCORE_METHODS="${SCORE_METHODS:-edu avg4 avg5}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
HQ_FRACTION="${HQ_FRACTION:-0.5}"
HQ_BY="${HQ_BY:-avg5}"
HQ_MIN_SCORE="${HQ_MIN_SCORE:-}"
INCLUDE_TOP_SELECT="${INCLUDE_TOP_SELECT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rater_dataset/annotated_eval_weighted}"
RUNS="$BASELINE"
for SCORE_METHOD in $SCORE_METHODS; do
  RUNS="$RUNS $RUN_ROOT/${SCORE_METHOD}_20M_ep${EPOCHS}_seed${SEED}"
done
if [ -n "$INCLUDE_TOP_SELECT" ]; then
  for SCORE_METHOD in $SCORE_METHODS; do
    RUNS="$RUNS checkpoints/gpt2_top_doc/${SCORE_METHOD}_20M_ep1_seed42"
  done
fi
HQ_ARGS=(--high-quality-by "$HQ_BY" --high-quality-fraction "$HQ_FRACTION")
[ -n "$HQ_MIN_SCORE" ] && HQ_ARGS+=(--high-quality-min-score "$HQ_MIN_SCORE")
FORCE_ARGS=()
[ -n "$FORCE" ] && FORCE_ARGS=(--force)
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | RUNS=$RUNS | HQ_BY=$HQ_BY HQ_MIN_SCORE=${HQ_MIN_SCORE:-none} HQ_FRACTION=$HQ_FRACTION | OUTPUT_DIR=$OUTPUT_DIR"

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.score_annotated_dataset \
  --runs $RUNS \
  --output-dir "$OUTPUT_DIR" \
  "${HQ_ARGS[@]}" \
  "${FORCE_ARGS[@]}" \
  --device cuda

echo "Done: $OUTPUT_DIR/{gpt2_summary.md, gpt2_summary_highquality.md, importance_score_distribution.png}"
