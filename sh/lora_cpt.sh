#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=4:00:00
#SBATCH --job-name=lora-cpt
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

set -eo pipefail

# Continued pretraining of google/gemma-3-270m with LoRA on one 20M selection (same documents as the GPT-2 runs),
# or, with METHOD=base, evaluation of the untrained pretrained model (the reference for every LoRA run).
# See src/train_gpt2_from_scratch/continue_pretrain_lora.py. Output: checkpoints/lora_cpt/gemma-3-270m/<run>/
# (results.json/summary.md with validation/test loss, adapter/, final/ = merged model + tokenizer).
# Gemma is gated: HF_TOKEN comes from the gitignored project-root .env.
#
# Run: METHOD=avg4_20M sbatch sh/lora_cpt.sh        (random_20M, edu_20M, avg4_20M, avg5_20M)
#      METHOD=base sbatch sh/lora_cpt.sh            (no training)
#      all of them + evaluations: bash sh/submit_lora_cpt.sh
# Other base model / LoRA size: MODEL=... LORA_R=32 LORA_ALPHA=64 METHOD=... sbatch sh/lora_cpt.sh
# CUDA out of memory: MICRO_BATCH_SIZE=2 (same 32 sequences per step) or GRADIENT_CHECKPOINTING=1.
# A smaller budget from a larger nested selection (its first part, as prepare_data.py would have selected it), trained
# as its own run (shuffled, cosine): METHOD=avg5_50M_7languages BUDGET=10M POOL_DIR=data/pilot_corpus_7languages \
#   sbatch sh/lora_cpt.sh   -> checkpoints/lora_cpt/<model>/avg5_50M_7languages_budget10M_ep1_seed42/

METHOD="${METHOD:-random_20M}"
MODEL="${MODEL:-google/gemma-3-270m}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-2e-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
BUDGET="${BUDGET:-}"                   # e.g. 10M (SeaLLM tokens per language); default: the whole selection
TRAIN_DATA="${TRAIN_DATA:-}"           # default data/pilot_selected/$METHOD/documents.csv
POOL_DIR="${POOL_DIR:-}"               # default data/pilot_corpus (validation/test documents)
ARGS=(--model "$MODEL" --epochs "$EPOCHS" --seed "$SEED" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lr "$LR"
      --micro-batch-size "$MICRO_BATCH_SIZE")
[ -n "$GRADIENT_CHECKPOINTING" ] && ARGS+=(--gradient-checkpointing)
[ -n "$BUDGET" ] && ARGS+=(--budget "$BUDGET")
[ -n "$TRAIN_DATA" ] && ARGS+=(--train-data "$TRAIN_DATA")
[ -n "$POOL_DIR" ] && ARGS+=(--pool-dir "$POOL_DIR")
if [ "$METHOD" = "base" ]; then ARGS+=(--eval-base-only); else ARGS+=(--method "$METHOD"); fi
ENVIRONMENT_NAME="sea-rater"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd) | METHOD=$METHOD${BUDGET:+ | BUDGET=$BUDGET} | MODEL=$MODEL | LoRA r=$LORA_R alpha=$LORA_ALPHA | LR=$LR | EPOCHS=$EPOCHS | SEED=$SEED"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
set -a; source .env; set +a   # HF_TOKEN (gated model)

pip install peft >/dev/null

nvidia-smi
python -c "import torch, peft; print('torch', torch.__version__, '| peft', peft.__version__, '| cuda available:', torch.cuda.is_available())"

python -m src.train_gpt2_from_scratch.continue_pretrain_lora "${ARGS[@]}" --device cuda

echo "Done: METHOD=$METHOD"
