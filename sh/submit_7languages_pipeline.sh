#!/bin/bash

# Data for the 7-language experiments (6 pilot languages + Burmese), scored by the 7-language human rater.
# Chained SLURM jobs (afterok); nothing under data/pilot_corpus, data/pilot_scores or the 6-language
# selections is touched.
#
#   extract 7 jobs  sh/prepare_corpus_7languages.sh STEP=extract: a random sample of TARGET (300K) FineWeb2
#                   documents per language, one CPU job each   -> data/pilot_corpus_7languages/<lang>.csv
#   split   1 job   STEP=split, after extract: fresh 500 validation / 1000 test split of all 7 languages
#                                                    -> data/pilot_corpus_7languages/split_manifest.csv
#   score   5 jobs  sh/score_pool_llm.sh with checkpoints/rater_7languages/finetuned/mean/<dim>: EVERY extracted
#                   document (no split needed), ~5.5h each for 7 x 300K (after extract, and after RATER_JOB if the
#                   raters are still training)
#                                                    -> data/pilot_scores_7languages/<dim>_mean/
#   select  1 job   sh/select_llm.sh, after score + split: avg4/avg5 + random/edu/avg4/avg5/rr5 selections of
#                   TARGET_TOKENS (50M) per language from the train split (rr5: per batch the next 5 top documents
#                   of each of the 5 dimensions, duplicates dropped)
#                                                    -> data/pilot_selected/{random,edu,avg4,avg5,rr5}_<BUDGET>_7languages/
#
# Run from the project root:  RATER_JOB=22011 bash sh/submit_7languages_pipeline.sh
# Extraction only:            STAGES="extract" bash sh/submit_7languages_pipeline.sh
# The rest later:             STAGES="split score select" bash sh/submit_7languages_pipeline.sh

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGES="${STAGES:-extract split score select}"
RATER_JOB="${RATER_JOB:-}"     # job id still training checkpoints/rater_7languages (scoring waits for it)
POOL_DIR=data/pilot_corpus_7languages
RATER_ROOT=checkpoints/rater_7languages/finetuned
SCORES_ROOT=data/pilot_scores_7languages
DIMENSIONS="educational_value reasoning professionalism cleanliness cultural_nuances"
declare -A SHORT=([educational_value]=edu [reasoning]=reas [professionalism]=prof [cleanliness]=clean [cultural_nuances]=cult)
has() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dep() { local d; d=$(echo "$*" | tr ' ' '\n' | grep -v '^$' | paste -sd:); [ -n "$d" ] && echo "--dependency=afterok:$d" || true; }

TARGET="${TARGET:-300000}"             # documents extracted per language
TARGET_TOKENS="${TARGET_TOKENS:-50000000}"   # selection budget: SeaLLM tokens per language
BUDGET="${BUDGET:-50M}"                     # name tag of the selections
EXTRACT_JOBS=""
if has extract; then
  for L in fil indo khmer malay thai vie burmese; do
    JOB=$(STEP=extract LANGUAGE=$L TARGET=$TARGET sbatch --parsable --job-name="ex-$L" sh/prepare_corpus_7languages.sh)
    echo "extract: $L job $JOB"
    EXTRACT_JOBS="$EXTRACT_JOBS $JOB"
  done
fi

SPLIT_JOB=""
if has split; then
  SPLIT_JOB=$(STEP=split TARGET=$TARGET sbatch --parsable $(dep $EXTRACT_JOBS) --time=2:00:00 \
              --job-name=split-7l sh/prepare_corpus_7languages.sh)
  echo "split  : job $SPLIT_JOB -> $POOL_DIR/split_manifest.csv"
fi

SCORE_JOBS=""
if has score; then
  for DIM in $DIMENSIONS; do
    JOB=$(DIMENSION=$DIM RATER_ROOT=$RATER_ROOT SCORES_ROOT=$SCORES_ROOT POOL_DIR=$POOL_DIR ALL_DOCUMENTS=1 \
          sbatch --parsable $(dep $EXTRACT_JOBS $RATER_JOB) --time=10:00:00 --mem=32G \
          --job-name="s7-${SHORT[$DIM]}" sh/score_pool_llm.sh)
    echo "score  : $DIM job $JOB"
    SCORE_JOBS="$SCORE_JOBS $JOB"
  done
fi

if has select; then
  JOB=$(SCORES_ROOT=$SCORES_ROOT POOL_DIR=$POOL_DIR N_LANGUAGES=7 OUT_PREFIX="" OUT_SUFFIX=_7languages \
        SELECT_RANDOM=1 COMPARE_RATERS=0 ROUND_ROBIN=1 TARGET_TOKENS=$TARGET_TOKENS BUDGET=$BUDGET \
        sbatch --parsable $(dep $SCORE_JOBS $SPLIT_JOB) --time=4:00:00 --job-name=select-7l sh/select_llm.sh)
  echo "select : job $JOB -> data/pilot_selected/{random,edu,avg4,avg5,rr5}_${BUDGET}_7languages"
fi
echo "Check progress: squeue -u $USER"
