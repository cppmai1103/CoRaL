#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=cscc-gpu-p
#SBATCH --time=48:00:00
#SBATCH --job-name=lora-multi
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --qos=cscc-gpu-qos

set -eo pipefail

# Several LoRA continued-pretraining runs inside ONE job: one sh/train/lora_cpt.sh process per GPU, one method each
# (cscc-gpu-qos allows 2 running jobs but up to 4 GPUs per user, so 2 jobs x 2 GPUs = 4 runs at once). Every other
# setting (MODEL, POOL_DIR, SEQ_LEN, LR, ...) is passed through the environment exactly as for lora_cpt.sh. Request
# one GPU and ~64 GB per method:
#   METHODS="avg5_50M_7languages rr5_50M_7languages" POOL_DIR=data/pilot_corpus_7languages ... \
#     sbatch --gres=gpu:2 --mem=128G sh/train/lora_cpt_multi.sh
# Usually submitted by sh/pipelines/submit_lora_cpt_7languages.sh with GPUS_PER_JOB=2.
# Logs: job-<id>.out (this script) and job-<id>.<method>.out (one per run).

METHODS="${METHODS:?set METHODS to the runs to train, one per GPU}"

cd "$SLURM_SUBMIT_DIR"
read -ra RUNS <<< "$METHODS"
read -ra GPUS <<< "$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' ' ')"
echo "Working directory: $(pwd) | ${#RUNS[@]} runs on GPUs ${GPUS[*]}: ${RUNS[*]}"
if [ "${#RUNS[@]}" -gt "${#GPUS[@]}" ]; then
  echo "ERROR: ${#RUNS[@]} runs but ${#GPUS[@]} GPU(s); submit with --gres=gpu:${#RUNS[@]}" >&2; exit 1
fi

PIDS=()
for i in "${!RUNS[@]}"; do
  M=${RUNS[$i]}
  LOG="job-${SLURM_JOB_ID}.${M}.out"
  echo "  $M -> GPU ${GPUS[$i]}, log $LOG"
  METHOD=$M CUDA_VISIBLE_DEVICES=${GPUS[$i]} bash sh/train/lora_cpt.sh > "$LOG" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  ${RUNS[$i]}: done"; else echo "  ${RUNS[$i]}: FAILED (see job-${SLURM_JOB_ID}.${RUNS[$i]}.out)" >&2; FAILED=1; fi
done
[ "$FAILED" = 0 ] && echo "Done: ${RUNS[*]}" || exit 1
