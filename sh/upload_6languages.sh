#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --job-name=upload-6l
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --qos=normal

set -eo pipefail

# Upload the 6-language work (data6languages/, checkpoints6languages/) to the private Hub repos
# cppmai/sea-rater-data and cppmai/sea-rater-models, in the same layout as the earlier uploads (repo root).
# src/upload_to_huggingface.py first copies everything into a staging folder under .scratch/hf_upload
# (~35 GB), then uploads; the staging folder is deleted when the upload succeeds.
#
# Run: sbatch sh/upload_6languages.sh            (ONLY=dataset or ONLY=model for one repo; DRY_RUN=1 to only package)

ONLY="${ONLY:-both}"
DRY_RUN="${DRY_RUN:-0}"
D=data6languages
C=checkpoints6languages
STAGING=.scratch/hf_upload/6languages_$SLURM_JOB_ID

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | ONLY=$ONLY DRY_RUN=$DRY_RUN | staging $STAGING"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate sea-rater
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
set -a; source .env; set +a   # HF_TOKEN

ARGS=(
  --dataset-repo cppmai/sea-rater-data --model-repo cppmai/sea-rater-models --only "$ONLY"
  --staging-dir "$STAGING"
  --dataset-dir data/annotation_batches --prepared-dir $D/rater_dataset
  --pilot-corpus-dir $D/pilot_corpus --pilot-scores-dir $D/pilot_scores --pilot-selected-dir $D/pilot_selected
  --llm-score-dir data/llm_score --rater-dataset-llm-dir $D/rater_dataset_llm --pilot-scores-llm-dir $D/pilot_scores_llm
  --checkpoint-dir $C/rater/frozen --finetuned-dir $C/rater/finetuned --rater-llm-dir $C/rater_llm
  --gpt2-dir $C/gpt2_top_doc --weighted-dir $C/gpt2_weighted_loss --gpt2-16k-dir $C/gpt2_top_doc_sea_bpe_16k
  --tokenizers-dir $C/tokenizers --lora-cpt-dir $C/lora_cpt --hub-models-dir $C/hub_models
)
[ "$DRY_RUN" = 1 ] && ARGS+=(--dry-run)

python -m src.upload_to_huggingface "${ARGS[@]}"

if [ "$DRY_RUN" = 1 ]; then
  echo "Dry run: packages kept in $STAGING for review (delete with: rm -rf $STAGING)"
else
  rm -rf "$STAGING"
  echo "Done: uploaded; removed $STAGING"
fi
