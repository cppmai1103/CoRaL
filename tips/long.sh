#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=train-%j.out
#SBATCH --error=job-%j.err
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

hostname
nvidia-smi

echo "$CUDA_VISIBLE_DEVICES"

