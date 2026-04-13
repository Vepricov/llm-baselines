#!/bin/bash
# Activate virtual environment
source ../llm-baselines/optim_venv/bin/activate

export CUDA_VISIBLE_DEVICES=7

for ws in 10000
do
  python src/main.py \
    --model swin \
    --dataset imagenet \
    --swin_model_name swin_base_patch4_window7_224 \
    --optimizer d-muon \
    --lr 1e-3 \
    --batch_size 128 \
    --iterations 50000 \
    --weight_decay 0.1 \
    --warmup_steps $ws \
    --scheduler cos \
    --device cuda:0 \
    --num_workers 4 \
    --eval_interval 500 \
    --eval_batches 20 \
    --log_interval 10 \
    --wandb
done