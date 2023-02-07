#!/bin/bash

#SBATCH --job-name=SIA
#SBATCH --partition=a3000
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16384
#SBATCH --cpus-per-task=16
#SBATCH --output=./log_slurm/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate py38

COMMAND="python main.py"
srun $COMMAND