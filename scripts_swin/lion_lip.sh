#!/bin/bash
# Activate virtual environment
source ../llm-baselines/optim_venv/bin/activate

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=2

for fstar in 0.1 0.5 0
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
    --scheduler cos \
    --lipschitz_sigma_F 0.001 \
    --lipschitz_rho 2.0 \
    --use_lip_warmup \
    --lipschitz_mode linear_and_cos \
    --lipschitz_loss_star $fstar \
    --device cuda:0 \
    --num_workers 4 \
    --eval_interval 500 \
    --eval_batches 20 \
    --log_interval 10 \
    --wandb
done