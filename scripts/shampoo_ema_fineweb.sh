#!/bin/bash

# Vanilla Shampoo baseline: same machinery as shampoo_dykaf but with the standard
# full-matrix g g^T / g^T g EMA for L/R (precond_mode="ema"). Direct comparison
# point for shampoo_dykaf (which uses the fast DyKAF proj_split for L/R).
# 124M llama, batch_size 32, sequence_length 512 -- matches the time_abl setup.

export CUDA_VISIBLE_DEVICES=1
BASE_PORT=29712
for iterations in 64000
do
    python \
    ./src/main.py \
        --run_prefix time_abl \
        --model llama \
        --dataset fineweb \
        --optimizer shampoo_ema \
        --dykaf_init eps \
        --shampoo_beta 0.999 \
        --precondition_frequency 10 \
        --max_precond_dim 10000 \
        --lr 5e-4 \
        --iterations $iterations \
        --n_embd 768 \
        --n_head 12 \
        --n_layer 12 \
        --batch_size 32 \
        --sequence_length 512 \
        --acc_steps 1 \
        --warmup_steps 3000 \
        --grad_clip 0.5 \
        --seed 0 \
        --weight_decay 1e-1 \
        --scheduler cos \
        --beta1 0.9 --beta2 0.999 \
        --dropout 0.0 \
        --eval_interval 115 --latest_ckpt_interval 1000 \
        --log_interval 1 \
        --do_not_auto_resume \
        --wandb_project MIKOLA_DROP_SOAP \
        --wandb
done
