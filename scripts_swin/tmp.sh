#!/bin/bash
# Activate virtual environment
source ../llm-baselines/optim_venv/bin/activate

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=2
cd /data/users/shkodnik1917/llm-baselines-warmup

python src/main.py \
  --model swin \
  --dataset imagenet \
  --swin_model_name swin_base_patch4_window7_224 \
  --batch_size 128 \
  --lr 0.05 \
  --momentum 0.95 \
  --iterations 100000 \
  --eval_interval 500 \
  --eval_batches 20 \
  --log_interval 10 \
  --opt sgd \
  --weight_decay 0.05 \
  --warmup_steps 20000 \
  --scheduler cos \
  --device cuda:0 \
  --num_workers 4 \
  --wandb