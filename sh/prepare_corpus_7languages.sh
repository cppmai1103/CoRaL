#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --job-name=corpus-7l
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --qos=normal

set -eo pipefail

# Build the 7-language pilot corpus (6 pilot languages + Burmese, TARGET = 300K documents each) in its own folder;
# data/pilot_corpus/ (the 6-language experiments) is never written.
#
#   STEP=extract LANGUAGE=<lang>   a seeded RANDOM sample of TARGET documents from the whole FineWeb2 config (random
#                                  row groups, random rows; same filters as data/pilot_corpus; the annotated
#                                  documents are excluded by id and text) -> $DST/<lang>.csv
#   STEP=split                     after all languages: full extraction report, then a fresh split of all 7
#                                  languages (500 validation / 1000 test per language, by website, seed 42)
#                                  -> $DST/split_manifest.csv, split_report.md
# A random sample does not contain data/pilot_corpus's validation/test documents, so those are not reused.
#
# Run all of it with: bash sh/submit_7languages_pipeline.sh (one extract job per language, then split)

STEP="${STEP:?set STEP=extract, truncate or split}"
LANGUAGE="${LANGUAGE:-}"
TARGET="${TARGET:-300000}"
SRC=data/pilot_corpus
DST="${DST:-data/pilot_corpus_7languages}"
LANGUAGES="fil indo khmer malay thai vie burmese"
ENVIRONMENT_NAME="sea_rater_env"   # has datatrove (FineWeb2 reader + banned-words list)

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | STEP=$STEP ${LANGUAGE:+LANGUAGE=$LANGUAGE }TARGET=$TARGET -> $DST"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p "$DST"
# A symlink here would make the extraction overwrite the linked file (e.g. data/pilot_corpus/<lang>.csv).
for F in "$DST"/*; do
  if [ -L "$F" ]; then echo "ERROR: $F is a symlink; remove it so nothing outside $DST is written" >&2; exit 1; fi
done

if [ "$STEP" = "extract" ]; then
  [ -n "$LANGUAGE" ] || { echo "ERROR: STEP=extract needs LANGUAGE" >&2; exit 1; }
  # Per-language report name, so parallel jobs do not overwrite each other's extraction_report.md.
  python -m src.train_gpt2_from_scratch.extract_corpus \
    --languages "$LANGUAGE" --target-per-language "$TARGET" --output-dir "$DST" --sampling random --seed 42
  mv "$DST/extraction_report.md" "$DST/extraction_report_$LANGUAGE.md"
  echo "Done: $DST/$LANGUAGE.csv"

elif [ "$STEP" = "truncate" ]; then
  # Cut languages extracted with a larger target to their first TARGET documents (random order, so still random).
  for L in ${LANGUAGE:-$LANGUAGES}; do
    python -m src.train_gpt2_from_scratch.extract_corpus --truncate --languages "$L" \
      --target-per-language "$TARGET" --output-dir "$DST"
  done

elif [ "$STEP" = "split" ]; then
  echo "=== extraction report for all languages (every language already extracted, so nothing is streamed) ==="
  for L in $LANGUAGES; do
    python - "$DST/$L.stats.json" "$TARGET" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
assert s["kept"] == s["target"] == int(sys.argv[2]), f"{sys.argv[1]}: kept {s['kept']} of target {s['target']}"
EOF
  done
  python -m src.train_gpt2_from_scratch.extract_corpus \
    --languages $LANGUAGES --target-per-language "$TARGET" --output-dir "$DST" --sampling random --seed 42
  rm -f "$DST"/extraction_report_*.md

  echo "=== split all 7 languages: 500 validation / 1000 test per language, by website ==="
  python -m src.train_gpt2_from_scratch.split_corpus --corpus-dir "$DST"
  echo "Done: $DST/{split_manifest.csv,split_report.md,extraction_report.md}"
else
  echo "Unknown STEP=$STEP (extract, truncate or split)" >&2; exit 1
fi
