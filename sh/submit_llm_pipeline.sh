#!/bin/bash

# Submit the whole LLM-label pipeline as chained SLURM jobs (each step starts only after the jobs it needs
# succeed -- afterok). Prerequisite, run once on the login node (CPU, seconds):
#   python -m src.extract_llm_scores           # LLM scores of the human-annotated docs -> data/rater_dataset/llm_scores_annotated.{json,md}
#   python -m src.prepare_llm_rater_dataset    # human-annotated docs + human splits, LLM labels -> data/rater_dataset_llm/
#
# Stages (STAGES, default all, in this order):
#   rater   5 jobs  sh/finetune_llm.sh, one per dimension          -> checkpoints/rater_llm/finetuned/mean/<dim>/
#   score   5 jobs  sh/score_pool_llm.sh, each after its rater      -> data/pilot_scores_llm/<dim>_mean/
#   select  1 job   sh/select_llm.sh: rater comparison vs human raters, avg4/avg5, top-20M selection
#                                                                   -> data/pilot_selected/llm_{edu,avg4,avg5}_20M/
#   gpt2    3 jobs  sh/train_gpt2.sh METHOD=llm_<m>_20M             -> checkpoints/gpt2_top_doc/llm_<m>_20M_ep1_seed42/
#   eval    5 jobs  on random_20M + LLM-rater runs + human-rater runs (first = baseline):
#                   test split, annotated high-quality (avg5 top 50%, cleanliness == 5), Belebele/XCOPA, Wikipedia/SIB-200
# A later stage only waits for earlier stages submitted in the same call; to resume, e.g. after the raters
# are done: STAGES="score select gpt2 eval" bash sh/submit_llm_pipeline.sh
#
# This only submits jobs; run it with bash from the project root:
#   bash sh/submit_llm_pipeline.sh

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGES="${STAGES:-rater score select gpt2 eval}"
DIMENSIONS="educational_value reasoning professionalism cleanliness cultural_nuances"
METHODS="llm_edu llm_avg4 llm_avg5"
# Short job names (<= 8 characters, so the default squeue column shows them in full)
declare -A SHORT=([educational_value]=edu [reasoning]=reas [professionalism]=prof [cleanliness]=clean [cultural_nuances]=cult)
BASELINE="checkpoints/gpt2_top_doc/random_20M_ep1_seed42"
has() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

if has rater && [ ! -f data/rater_dataset_llm/split_manifest_educational_value.csv ]; then
  echo "ERROR: data/rater_dataset_llm/ is missing -- run: python -m src.prepare_llm_rater_dataset" >&2
  exit 1
fi

declare -A RATER_JOB
SCORE_JOBS=""
for DIM in $DIMENSIONS; do
  if has rater; then
    RATER_JOB[$DIM]=$(DIMENSION=$DIM sbatch --parsable --job-name="ft-${SHORT[$DIM]}" sh/finetune_llm.sh)
    echo "rater   $DIM: job ${RATER_JOB[$DIM]}"
  fi
  if has score; then
    JOB=$(DIMENSION=$DIM sbatch --parsable $(dep "${RATER_JOB[$DIM]}") --job-name="sc-${SHORT[$DIM]}" sh/score_pool_llm.sh)
    echo "score   $DIM: job $JOB${RATER_JOB[$DIM]:+ (after ${RATER_JOB[$DIM]})}"
    SCORE_JOBS="${SCORE_JOBS:+$SCORE_JOBS:}$JOB"
  fi
done

SELECT_JOB=""
if has select; then
  SELECT_JOB=$(sbatch --parsable $(dep "$SCORE_JOBS") --job-name=select sh/select_llm.sh)
  echo "select     : job $SELECT_JOB${SCORE_JOBS:+ (after $SCORE_JOBS)}"
fi

GPT2_JOBS=""
if has gpt2; then
  for M in $METHODS; do
    JOB=$(METHOD=${M}_20M sbatch --parsable $(dep "$SELECT_JOB") --job-name="gpt-${M#llm_}" sh/train_gpt2.sh)
    echo "gpt2    ${M}_20M: job $JOB${SELECT_JOB:+ (after $SELECT_JOB)}"
    GPT2_JOBS="${GPT2_JOBS:+$GPT2_JOBS:}$JOB"
  done
fi

if has eval; then
  RUN_DIRS="$BASELINE"
  for M in $METHODS edu avg4 avg5; do RUN_DIRS="$RUN_DIRS checkpoints/gpt2_top_doc/${M}_20M_ep1_seed42"; done
  D=$(dep "$GPT2_JOBS")
  JOB=$(RUN_DIRS="$RUN_DIRS" COMPARISON_FILE=checkpoints/gpt2_top_doc/llm_test_comparison.md \
        sbatch --parsable $D --job-name=ev-test sh/eval_test_set.sh)
  echo "eval    test split: job $JOB"
  JOB=$(RUN_ROOT=checkpoints/gpt2_top_doc SCORE_METHODS="$METHODS edu avg4 avg5" BASELINE="$BASELINE" \
        OUTPUT_DIR=data/rater_dataset/annotated_eval_llm sbatch --parsable $D --job-name=ev-hq sh/eval_annotated_weighted.sh)
  echo "eval    annotated (avg5 top 50%): job $JOB"
  JOB=$(RUN_ROOT=checkpoints/gpt2_top_doc SCORE_METHODS="$METHODS edu avg4 avg5" BASELINE="$BASELINE" \
        HQ_BY=cleanliness HQ_MIN_SCORE=5 OUTPUT_DIR=data/rater_dataset/annotated_eval_cleanliness_llm \
        sbatch --parsable $D --job-name=ev-clean sh/eval_annotated_weighted.sh)
  echo "eval    annotated (cleanliness == 5): job $JOB"
  JOB=$(RUN_DIRS="$RUN_DIRS" COMPARISON_FILE=checkpoints/gpt2_top_doc/llm_downstream_comparison.md \
        sbatch --parsable $D --job-name=ev-down sh/eval_downstream.sh)
  echo "eval    Belebele/XCOPA: job $JOB"
  JOB=$(RUN_DIRS="$RUN_DIRS" COMPARISON_FILE=checkpoints/gpt2_top_doc/llm_wiki_sib200_comparison.md \
        sbatch --parsable $D --job-name=ev-wiki sh/eval_wiki_sib200.sh)
  echo "eval    Wikipedia/SIB-200: job $JOB"
fi
echo "Check progress: squeue -u $USER"
