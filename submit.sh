#!/bin/bash

#SBATCH --job-name=SIA
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16384
#SBATCH --cpus-per-task=8
#SBATCH --output=./log_slurm/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate py38

COMMAND="python main.py \
--mode=train \
--dataset=amazon_beauty \
--num_epochs=50 \
--lr=1e-3 \
--latent_dim=64 \
--item_dim_hidden=64 \
--item_num_heads=1 \
--attn_dim_head=64 \
--attn_num_heads=1 \
--log_dir=dim64_lr1e-3
"
srun $COMMAND