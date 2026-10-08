#!/bin/bash

#SBATCH --job-name=cpu-test
#SBATCH --output=cpu-output-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --time=12:00:00

hostname
python test.py