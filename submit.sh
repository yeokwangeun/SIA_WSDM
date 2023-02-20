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
--num_epochs=5 \
--lr=1e-2 \
--latent_dim=16 \
--item_dim_hidden=16 \
--item_num_heads=1 \
--attn_dim_head=16 \
--attn_num_heads=1 \
"
if [ ! -z "$1" ]; then
  COMMAND="$1"
fi
srun $COMMAND