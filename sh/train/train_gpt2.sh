#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=long
#SBATCH --time=1:00:00          # generous default for MODE=score-all (~3-4h scoring); override for quicker modes,
#SBATCH --job-name=gpt2-pilot   # e.g. --time=1:00:00 for MODE=train, or pass --time=... on the sbatch command line
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-12

set -eo pipefail

# HF_TOKEN lives in the gitignored project-root .env (chmod 600), never in this script.
# sbatch runs a spooled copy of this file, so locate .env via the submit directory (project root).
source "${SLURM_SUBMIT_DIR:-.}/.env"

# One entry point for the whole GPT-2 pilot: score the candidate pool with one or more raters,
# select the top documents, and train, all in the same job. MODE picks which of those stages run:
#
#   MODE=train      (default) Just train on an already-prepared data/pilot_selected/$METHOD/documents.csv.
#                   This is the Random baseline path.        e.g. METHOD=random EPOCHS_LIST="1 4" sbatch sh/train/train_gpt2.sh
#
#   MODE=score      Score the pool with ONE rater (RATER_DIR), keep the top NUM_DOCS, then train.
#                   e.g. RATER_DIR=checkpoints/rater/finetuned/mean/educational_value SCORES_NAME=educational_value_mean \
#                        METHOD=edu_top10k sbatch --export=ALL,MODE=score sh/train/train_gpt2.sh
#
#   MODE=score-all  Score the pool on SCORE_DIMENSIONS (dimensions not yet scored; waits up to WAIT_MINUTES
#                   for any COMBINE_DIMENSIONS scored by another job, e.g. Educational Value from MODE=score),
#                   average all COMBINE_DIMENSIONS into one importance score, keep the top NUM_DOCS, then train.
#                   e.g. sbatch --export=ALL,MODE=score-all sh/train/train_gpt2.sh
#
#   MODE=combine    Like score-all but scores nothing itself: DIMENSIONS must already be fully scored
#                   (fails fast if not). Use this to try a different subset/combination of already-scored
#                   dimensions, e.g. the four below WITHOUT Cultural Nuances (the default for this mode):
#                   e.g. sbatch --export=ALL,MODE=combine sh/train/train_gpt2.sh
#                        DIMENSIONS="educational_value reasoning professionalism cleanliness cultural_nuances" \
#                          COMBINED_NAME=avg5_mean METHOD=avg5_top10k sbatch --export=ALL,MODE=combine sh/train/train_gpt2.sh
#
#   MODE=select     No scoring/combining: (re)select from an already-scored/combined source (or, for
#                   SELECTION_METHOD=random, no rater at all) with --num-docs OR a token budget, then
#                   train. This is how to sweep token budgets (5M/10M/25M/...) per language across
#                   methods whose scores already exist, without re-scoring or re-combining anything:
#                     SELECT_BY=tokens TARGET_TOKENS=5000000 METHOD=random_5M \
#                       sbatch --export=ALL,MODE=select sh/train/train_gpt2.sh
#                     SELECT_BY=tokens TARGET_TOKENS=5000000 SELECTION_METHOD=top-score \
#                       SCORES_DIR=data/pilot_scores/avg5_mean METHOD=avg5_5M \
#                       sbatch --export=ALL,MODE=select sh/train/train_gpt2.sh
#                   (SCORES_DIR is any score_pool.py/combine_scores.py output, e.g.
#                   data/pilot_scores/educational_value_mean, data/pilot_scores/avg4_mean, data/pilot_scores/avg5_mean.)
#
#   MODE=init       No scoring, no selection, no training: evaluate the untrained, freshly initialized
#                   (shared) GPT-2 weights on validation/test. This is the chance-level reference every
#                   trained run's summary.md should be compared against -- perplexity should land close to
#                   the vocabulary size (~48,384). Only depends on (architecture, SEED), not on METHOD;
#                   TRAIN_DATA is read only to discover the 6 languages, never encoded or trained on.
#                   e.g. sbatch --export=ALL,MODE=init sh/train/train_gpt2.sh
#
# All modes except init: TRAIN=0 stops after selection, without training (e.g. to inspect
# data/pilot_selected/$METHOD first).
# score/score-all/combine/select all select by SELECT_BY: docs (--num-docs documents per language,
# the default) or tokens (--target-tokens SeaLLM tokens per language, grown --group-size documents
# at a time -- see prepare_data.py --select-by tokens).
# Run every method with the same SELECT_BY/NUM_DOCS-or-TARGET_TOKENS, EPOCHS_LIST and SEED so the
# runs are comparable.
# Other tokenizer (MODE=train and MODE=init): set TOKENIZER to a Hub ID or a local tokenizer folder, e.g. the
# custom SEA BPE from sh/data/train_tokenizer.sh. Runs then go to checkpoints/gpt2_top_doc_<TOKENIZER_TAG>/
# (default tag: the tokenizer folder name) instead of checkpoints/gpt2_top_doc/, from shared initial weights
# for that vocab size. Training documents are unchanged: data/pilot_selected/$METHOD was sized with SeaLLM
# token counts, so the same documents give a different number of tokens under another tokenizer.
#   for M in random_20M edu_20M avg4_20M avg5_20M; do
#     METHOD=$M TOKENIZER=checkpoints/tokenizers/sea_bpe_16k sbatch sh/train/train_gpt2.sh; done
#   MODE=init TOKENIZER=checkpoints/tokenizers/sea_bpe_16k sbatch sh/train/train_gpt2.sh   # chance-level reference
# If a training step hits CUDA out-of-memory, lower --micro-batch-size in the train_gpt2 call below
# (8 by default; the effective batch stays 32 sequences).
MODE="${MODE:-train}"
TRAIN="${TRAIN:-1}"
EPOCHS_LIST="${EPOCHS_LIST:-1}"      # use the same epoch count as the runs you compare against
SEED="${SEED:-42}"
SELECT_BY="${SELECT_BY:-docs}"       # docs | tokens (score/score-all/combine/select only)
NUM_DOCS="${NUM_DOCS:-10000}"        # select-by=docs only
TARGET_TOKENS="${TARGET_TOKENS:-}"   # select-by=tokens only: SeaLLM tokens (with EOS) per language
GROUP_SIZE="${GROUP_SIZE:-5}"        # select-by=tokens only: grow the selection this many documents at a time

case "$MODE" in
  train)     METHOD="${METHOD:-random}" ;;
  score)     METHOD="${METHOD:-edu_top10k}" ;;
  score-all) METHOD="${METHOD:-avg5_top10k}" ;;
  combine)   METHOD="${METHOD:-avg4_top10k}" ;;
  select)    METHOD="${METHOD:-random_selected}" ;;
  init)      METHOD="${METHOD:-random}" ;;   # only used to pick a default TRAIN_DATA; irrelevant to the result
  *) echo "Unknown MODE=$MODE (use train, score, score-all, combine, select, or init)" >&2; exit 1 ;;
esac
TRAIN_DATA="${TRAIN_DATA:-data/pilot_selected/${METHOD}/documents.csv}"

TOKENIZER="${TOKENIZER:-}"
if [ -n "$TOKENIZER" ]; then
  TOKENIZER_TAG="${TOKENIZER_TAG:-$(basename "$TOKENIZER" | tr '[:upper:]' '[:lower:]')}"
  RUN_ROOT="checkpoints/gpt2_top_doc_${TOKENIZER_TAG}"
  TOKENIZER_ARGS=(--tokenizer "$TOKENIZER")
else
  RUN_ROOT="checkpoints/gpt2_top_doc"
  TOKENIZER_ARGS=()
fi

if [ "$SELECT_BY" = "tokens" ] && [ -z "$TARGET_TOKENS" ]; then
  echo "ERROR: TARGET_TOKENS is required when SELECT_BY=tokens (e.g. TARGET_TOKENS=5000000)" >&2
  exit 1
fi

# MODE=score only: which rater scores the pool.
RATER_DIR="${RATER_DIR:-checkpoints/rater/finetuned/mean/educational_value}"   # mean pooling: test MAE 0.409 (cls: 0.495)
SCORES_NAME="${SCORES_NAME:-educational_value_mean}"

# MODE=select only: random needs no rater; top-score reads an already-scored/combined folder.
SELECTION_METHOD="${SELECTION_METHOD:-random}"   # random | top-score
SCORES_DIR="${SCORES_DIR:-data/pilot_scores/educational_value_mean}"

# MODE=score-all only: dimensions to score here vs. to combine (the ones not scored here, e.g. Educational
# Value, must already be complete -- scored separately with MODE=score -- or this waits for them).
SCORE_DIMENSIONS="${SCORE_DIMENSIONS:-reasoning professionalism cleanliness cultural_nuances}"
COMBINE_DIMENSIONS_ALL="${COMBINE_DIMENSIONS:-educational_value reasoning professionalism cleanliness cultural_nuances}"
RATER_ROOT="${RATER_ROOT:-checkpoints/rater/finetuned/mean}"    # mean pooling: the better rater for every dimension
COMBINED_NAME_ALL="${COMBINED_NAME:-avg5_mean}"
WAIT_MINUTES="${WAIT_MINUTES:-180}"   # how long score-all waits for dimensions scored by another job

# MODE=combine only: dimensions to average, must already be fully scored (default: 4, WITHOUT Cultural Nuances).
DIMENSIONS="${DIMENSIONS:-educational_value reasoning professionalism cleanliness}"
COMBINED_NAME_SUBSET="${COMBINED_NAME:-avg4_mean}"

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea-rater"   # own environment: other projects share "huhu" and changed its transformers version

# $SLURM_SUBMIT_DIR is the real submission dir; $0 points to SLURM's spool copy.
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | MODE=$MODE | METHOD=$METHOD | SELECT_BY=$SELECT_BY" \
     "${TARGET_TOKENS:+TARGET_TOKENS=$TARGET_TOKENS }NUM_DOCS=$NUM_DOCS | EPOCHS_LIST=$EPOCHS_LIST | SEED=$SEED" \
     "| TOKENIZER=${TOKENIZER:-default (SeaLLM)} | RUN_ROOT=$RUN_ROOT"

# Number of usable score files (language CSVs; thresholds.csv / dimension_correlations.csv are not
# languages) in a folder. Pure-bash glob walk, not ls|grep|wc: with `set -o pipefail`, an empty/no-match
# glob makes an intermediate grep exit 1 (0 lines selected) even though wc -l would still print the
# right "0", and that stray nonzero exit status kills the whole script under `set -e` before any
# message prints.
count_score_files() {
  local dir="$1" count=0 f
  for f in "$dir"/*.csv; do
    [ -e "$f" ] || continue   # unmatched glob (dir missing or empty) stays literal here; skip it
    case "$(basename "$f")" in
      thresholds.csv|dimension_correlations.csv) ;;
      *) count=$((count + 1)) ;;
    esac
  done
  echo "$count"
}
scored_languages() { count_score_files "data/pilot_scores/${1}_mean"; }

# MODE=combine fails fast, before any environment setup, if a dimension has not been fully scored.
if [ "$MODE" = "combine" ]; then
  SCORE_DIRS=""
  for DIM in $DIMENSIONS; do
    COUNT=$(scored_languages "$DIM")
    if [ "$COUNT" -lt 6 ]; then
      echo "ERROR: data/pilot_scores/${DIM}_mean has scores for $COUNT of 6 languages. Score $DIM first (MODE=score-all or MODE=score)." >&2
      exit 1
    fi
    SCORE_DIRS="$SCORE_DIRS data/pilot_scores/${DIM}_mean"
  done
fi

# MODE=select (SELECTION_METHOD=top-score only) fails fast if SCORES_DIR isn't fully scored.
if [ "$MODE" = "select" ] && [ "$SELECTION_METHOD" != "random" ]; then
  COUNT=$(count_score_files "$SCORES_DIR")
  if [ "$COUNT" -lt 6 ]; then
    echo "ERROR: SCORES_DIR=$SCORES_DIR has scores for $COUNT of 6 languages. Point it at a folder written by " \
         "score_pool.py or combine_scores.py (e.g. data/pilot_scores/avg5_mean), fully scored for all 6 languages." >&2
    exit 1
  fi
fi

# select-by=tokens: shared argument list for prepare_data.py (docs count vs. token budget).
selection_args() {
  if [ "$SELECT_BY" = "tokens" ]; then
    echo "--select-by tokens --target-tokens $TARGET_TOKENS --group-size $GROUP_SIZE"
  else
    echo "--num-docs $NUM_DOCS"
  fi
}

source /apps/local/anaconda3/etc/profile.d/conda.sh

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  echo "Env '${ENVIRONMENT_NAME}' not found, creating it with python=${PYTHON_VERSION}"
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
else
  echo "Env '${ENVIRONMENT_NAME}' already exists, skipping creation"
fi
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1      # use env torch, not ~/.local's (which is known broken here)
export PYTHONUNBUFFERED=1      # live progress in job-*.out

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.14.1" huggingface_hub tqdm matplotlib

nvidia-smi || true   # only informational; exits non-zero on some nodes (e.g. corrupted infoROM)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# Data preparation (already done; uncomment to rebuild the corpus itself, independent of MODE):
# python -m src.train_gpt2_from_scratch.extract_corpus --target-per-language 50000
# python -m src.train_gpt2_from_scratch.split_corpus

# --- MODE=score: score the pool with one rater, then select -------------------------------------
if [ "$MODE" = "score" ]; then
  echo "=== 1. score the candidate pool: $SCORES_NAME ==="
  python -m src.train_gpt2_from_scratch.score_pool \
    --rater-dir "$RATER_DIR" \
    --output-dir "data/pilot_scores/$SCORES_NAME" \
    --top-k "$NUM_DOCS" \
    --device cuda

  echo "=== 2. keep the top $NUM_DOCS documents per language ==="
  python -m src.train_gpt2_from_scratch.prepare_data \
    --method top-score \
    --scores-dir "data/pilot_scores/$SCORES_NAME" \
    --output-dir "data/pilot_selected/$METHOD" \
    --num-docs "$NUM_DOCS"
fi

# --- MODE=score-all: score the remaining dimensions, wait for the rest, then combine + select ---
if [ "$MODE" = "score-all" ]; then
  for DIM in $SCORE_DIMENSIONS; do
    echo "=== 1. score the candidate pool: $DIM ==="
    python -m src.train_gpt2_from_scratch.score_pool \
      --rater-dir "$RATER_ROOT/$DIM" \
      --output-dir "data/pilot_scores/${DIM}_mean" \
      --top-k "$NUM_DOCS" \
      --device cuda
  done

  SCORE_DIRS=""
  for DIM in $COMBINE_DIMENSIONS_ALL; do
    SCORE_DIRS="$SCORE_DIRS data/pilot_scores/${DIM}_mean"
  done

  # Combining needs every dimension complete. Wait for the ones scored by another job (e.g. Educational Value).
  for DIM in $COMBINE_DIMENSIONS_ALL; do
    case " $SCORE_DIMENSIONS " in *" $DIM "*) continue ;; esac
    WAITED=0
    while [ "$(scored_languages "$DIM")" -lt 6 ]; do
      if [ "$WAITED" -ge "$WAIT_MINUTES" ]; then
        echo "ERROR: data/pilot_scores/${DIM}_mean has $(scored_languages "$DIM") of 6 languages after waiting ${WAIT_MINUTES} min." >&2
        echo "Finish scoring $DIM (MODE=score), then re-run this script: the scored dimensions are skipped." >&2
        exit 1
      fi
      echo "Waiting for $DIM scores ($(scored_languages "$DIM") of 6 languages) ... ${WAITED} min"
      sleep 300
      WAITED=$((WAITED + 5))
    done
    [ "$WAITED" -gt 0 ] && sleep 60   # let the other job finish writing its last file
  done

  echo "=== 2. average the dimensions into one importance score: $COMBINE_DIMENSIONS_ALL ==="
  python -m src.train_gpt2_from_scratch.combine_scores \
    --scores-dirs $SCORE_DIRS \
    --output-dir "data/pilot_scores/$COMBINED_NAME_ALL" \
    --top-k "$NUM_DOCS"

  echo "=== 3. keep the top $NUM_DOCS documents per language ==="
  python -m src.train_gpt2_from_scratch.prepare_data \
    --method top-score \
    --scores-dir "data/pilot_scores/$COMBINED_NAME_ALL" \
    --output-dir "data/pilot_selected/$METHOD" \
    --num-docs "$NUM_DOCS"
fi

# --- MODE=combine: average already-scored dimensions (no scoring here), then select -------------
if [ "$MODE" = "combine" ]; then
  echo "=== 1. average the dimensions: $DIMENSIONS ==="
  python -m src.train_gpt2_from_scratch.combine_scores \
    --scores-dirs $SCORE_DIRS \
    --output-dir "data/pilot_scores/$COMBINED_NAME_SUBSET" \
    --top-k "$NUM_DOCS"

  echo "=== 2. keep the top $NUM_DOCS documents per language ==="
  python -m src.train_gpt2_from_scratch.prepare_data \
    --method top-score \
    --scores-dir "data/pilot_scores/$COMBINED_NAME_SUBSET" \
    --output-dir "data/pilot_selected/$METHOD" \
    --num-docs "$NUM_DOCS"
fi

# --- MODE=select: (re)select from an already-scored/combined source (or random), then select ---
if [ "$MODE" = "select" ]; then
  if [ "$SELECTION_METHOD" = "random" ]; then
    echo "=== select ($SELECT_BY): random -> $METHOD ==="
    python -m src.train_gpt2_from_scratch.prepare_data \
      --method random \
      --output-dir "data/pilot_selected/$METHOD" \
      --seed "$SEED" \
      $(selection_args)
  else
    echo "=== select ($SELECT_BY): top-score from $SCORES_DIR -> $METHOD ==="
    python -m src.train_gpt2_from_scratch.prepare_data \
      --method top-score \
      --scores-dir "$SCORES_DIR" \
      --output-dir "data/pilot_selected/$METHOD" \
      $(selection_args)
  fi
fi

# --- MODE=init: evaluate the untrained initial weights, no selection or training --------------
# Needs data/pilot_corpus/split_manifest.csv and data/pilot_corpus/<language>.csv for the held-out validation/test
# texts, plus $TRAIN_DATA (only its language column is read). Writes to $RUN_ROOT/init_seed$SEED/.
if [ "$MODE" = "init" ]; then
  echo "=== evaluate the untrained initial weights, seed $SEED ==="
  python -m src.train_gpt2_from_scratch.train_gpt2 \
    --eval-init-only \
    --train-data "$TRAIN_DATA" \
    --output-dir "$RUN_ROOT/init_seed$SEED" \
    "${TOKENIZER_ARGS[@]}" \
    --seed "$SEED" \
    --device cuda
fi

# --- Train (train/score/score-all/combine/select; MODE=train needs data/pilot_selected/$METHOD/documents.csv ----
# to already exist). Needs data/pilot_corpus/split_manifest.csv and data/pilot_corpus/<language>.csv for the
# held-out validation/test texts.
if [ "$MODE" != "init" ]; then
  if [ "$TRAIN" = "1" ]; then
    for EPOCHS in $EPOCHS_LIST; do
      echo "=== train GPT-2 on $METHOD: $EPOCHS epoch(s), seed $SEED ==="
      python -m src.train_gpt2_from_scratch.train_gpt2 \
        --method "$METHOD" \
        --train-data "$TRAIN_DATA" \
        --output-dir "$RUN_ROOT/${METHOD}_ep${EPOCHS}_seed${SEED}" \
        "${TOKENIZER_ARGS[@]}" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --evals-per-epoch 2 \
        --device cuda
    done
  else
    echo "TRAIN=0: stopping after selection (data/pilot_selected/$METHOD)."
  fi
fi

# SELECT_BY=tokens TARGET_TOKENS=5000000 METHOD=random_5M \
#   sbatch --export=ALL,MODE=select sh/train/train_gpt2.sh

# SELECT_BY=tokens TARGET_TOKENS=5000000 SELECTION_METHOD=top-score \
#   SCORES_DIR=data/pilot_scores/educational_value_mean METHOD=edu_5M \
#   sbatch --export=ALL,MODE=select sh/train/train_gpt2.sh

# SELECT_BY=tokens TARGET_TOKENS=5000000 SELECTION_METHOD=top-score \
#   SCORES_DIR=data/pilot_scores/avg4_mean METHOD=avg4_5M \
#   sbatch --export=ALL,MODE=select sh/train/train_gpt2.sh

# SELECT_BY=tokens TARGET_TOKENS=5000000 SELECTION_METHOD=top-score \
#   SCORES_DIR=data/pilot_scores/avg5_mean METHOD=avg5_5M \
#   sbatch --export=ALL,MODE=select sh/train/train_gpt2.sh
