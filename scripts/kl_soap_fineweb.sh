#!/bin/bash

# torchrun --nproc_per_node=2 --master_port=$((BASE_PORT++)) \
# --distributed_backend nccl \
export CUDA_VISIBLE_DEVICES=4
for iterations in 128000 256000 512000
do
    python \
    ./src/main.py \
        --model llama \
        --dataset fineweb \
        --optimizer kl_soap \
        --precondition_frequency 10 \
        --lr 5e-1 \
        --iterations $iterations \
        --n_embd 768 \
        --n_head 12 \
        --n_layer 12 \
        --batch_size 32 \
        --sequence_length 512 \
        --acc_steps 1 \
        --warmup_steps 3000 \
        --grad_clip 0.5 \
        --seed 1 \
        --weight_decay 1e-4 \
        --scheduler cos \
        --beta1 0.95 --beta2 0.999 \
        --dropout 0.0 \
        --eval_interval 115 --latest_ckpt_interval 1000 \
        --log_interval 1 \
        --do_not_auto_resume \
        --wandb_project dykaf_fineweb \
        --wandb # \ --do_not_auto_resume \
done
