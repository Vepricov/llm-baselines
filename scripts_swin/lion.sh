#!/bin/bash
# Activate virtual environment
source ../llm-baselines/optim_venv/bin/activate

export CUDA_VISIBLE_DEVICES=4

# for ws in 1000 10000 30000 50000
for ws in 60000 70000 80000
do
  python src/main.py \
    --model swin \
    --dataset imagenet \
    --swin_model_name swin_base_patch4_window7_224 \
    --optimizer lion-ada \
    --lr 1e-3 \
    --batch_size 128 \
    --iterations 100000 \
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