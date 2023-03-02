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
--early_stop=50 \
--lr=5e-4 \
--latent_dim=128 \
--feature_dim=128 \
--attn_dim_head=128 \
--attn_num_heads=2 \
--attn_depth=1 \
--attn_self_per_cross=2 \
--attn_dropout=0.5 \
--attn_ff_dropout=0.5 \
--weight_decay=1 \
--log_dir=dim128_lr5e-4
"
if [ ! -z "$1" ]; then
  COMMAND="$1"
fi
srun $COMMAND
