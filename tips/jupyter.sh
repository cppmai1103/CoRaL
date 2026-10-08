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

# source /apps/local/anaconda3/etc/profile.d/conda.sh
# conda activate /home/mai.chau/.conda/envs/test_env

ENV_NAME="test_env"
PORT=$(shuf -i 20000-65000 -n 1)

# NOTE: please update the path to conda_init.sh below
srun --export=ALL bash -lc "
  . /apps/local/anaconda2023/conda_init.sh # Adjust path as needed, script that initializes conda
  conda activate ${ENV_NAME} || { echo 'Env ${ENV_NAME} not found'; exit 1; }
  python -m pip install -q --upgrade jupyterlab ipykernel
  python -m ipykernel install --user --name ${ENV_NAME} --display-name 'Python (${ENV_NAME})' >/dev/null 2>&1 || true
  jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT}
"