#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=6:00:00
#SBATCH --job-name=gpt2-weighted
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Quality-weighted training loss (docs/05_loss.md): instead of selecting which documents to train
# on, every document's rater importance score becomes a per-token loss weight. All three runs
# below train on the SAME documents -- data/pilot_selected/random_20M/documents.csv, the same 20M
# manifest used by the unweighted checkpoints/gpt2_top_doc/random_20M_ep1_seed42 baseline -- and
# from the same shared initial weights (checkpoints/gpt2_top_doc/init). Only the loss weighting
# differs between SCORE_METHODS:
#   edu    data/pilot_scores/educational_value_mean  (single dimension)
#   avg4   data/pilot_scores/avg4_mean                (4-dimension average)
#   avg5   data/pilot_scores/avg5_mean                (5-dimension average)
#
# Compare each result's validation/test (ordinary unweighted loss) against random_20M_ep1_seed42
# to see whether loss weighting beats the identical-data unweighted baseline.
#
# After all SCORE_METHODS are trained, this also runs the same two evaluation suites already used
# for checkpoints/gpt2_top_doc on these new checkpoints plus the random_20M_ep1_seed42 baseline:
#   docs/03_downstream_evaluation.md  zero-shot Belebele/XCOPA, 500 samples/language
#   docs/04_wikipedia_sib200_evaluation.md  Wikipedia/SIB-200 loss and perplexity
#
# Run: sbatch sh/train_gpt2_weighted.sh
# Override which methods/epochs run, e.g.: SCORE_METHODS="avg5" EPOCHS=4 sbatch sh/train_gpt2_weighted.sh

SCORE_METHODS="${SCORE_METHODS:-edu avg4 avg5}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | SCORE_METHODS=$SCORE_METHODS | EPOCHS=$EPOCHS | SEED=$SEED"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

for SCORE_METHOD in $SCORE_METHODS; do
  echo "=== sanity check: $SCORE_METHOD ==="
  python -m src.train_gpt2_from_scratch.train_gpt2_weighted \
    --score-method "$SCORE_METHOD" \
    --sanity-check \
    --seed "$SEED" \
    --device cuda

  echo "=== train GPT-2 with $SCORE_METHOD-weighted loss on random_20M: $EPOCHS epoch(s), seed $SEED ==="
  python -m src.train_gpt2_from_scratch.train_gpt2_weighted \
    --score-method "$SCORE_METHOD" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --device cuda
done

# Baseline first (trained on the same documents, unweighted), then every weighted run just trained.
RUN_DIRS="checkpoints/gpt2_top_doc/random_20M_ep1_seed42"
for SCORE_METHOD in $SCORE_METHODS; do
  RUN_DIRS="$RUN_DIRS checkpoints/gpt2_weighted_loss/${SCORE_METHOD}_20M_ep${EPOCHS}_seed${SEED}"
done

pip install pyarrow pandas >/dev/null  # XCOPA/Wikipedia parquet, pilot_corpus overlap check

echo "=== downstream zero-shot evaluation (Belebele/XCOPA) ==="
python -m src.train_gpt2_from_scratch.eval_downstream \
  --run-dir $RUN_DIRS \
  --benchmarks belebele xcopa \
  --comparison-file checkpoints/gpt2_weighted_loss/downstream_comparison.md \
  --num-samples 500 \
  --device cuda

echo "=== Wikipedia/SIB-200 loss+perplexity evaluation ==="
python -m src.train_gpt2_from_scratch.eval_wiki_sib200 \
  --run-dir $RUN_DIRS \
  --comparison-file checkpoints/gpt2_weighted_loss/wiki_sib200_comparison.md \
  --device cuda

echo "Done: checkpoints/gpt2_weighted_loss/{${SCORE_METHODS// /,}}_20M_ep${EPOCHS}_seed${SEED}/, " \
     "downstream_comparison.md, wiki_sib200_comparison.md"
