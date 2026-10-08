#!/bin/bash

#SBATCH --job-name=gpu-test
#SBATCH --output=output-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=00:03:00

source /apps/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mai.chau/.conda/envs/test_env

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# hostname
nvidia-smi
python test.py