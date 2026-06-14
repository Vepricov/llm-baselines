#!/bin/bash

export CUDA_VISIBLE_DEVICES=5,7
BASE_PORT=29522
for iterations in 8000 16000 32000 48000 64000
do
    sleep 5
    torchrun --nproc_per_node=2 --master_port=$((BASE_PORT++)) \
    ./src/main.py \
        --run_prefix time_abl \
        --distributed_backend nccl \
        --model llama \
        --dataset fineweb \
        --optimizer muon \
        --lr 5e-4 \
        --iterations $iterations \
        --n_embd 768 \
        --n_head 12 \
        --n_layer 12 \
        --batch_size 256 \
        --sequence_length 512 \
        --acc_steps 1 \
        --warmup_steps 3000 \
        --grad_clip 0.5 \
        --seed 0 \
        --weight_decay 0.1 \
        --scheduler cos \
        --beta1 0.8 --beta2 0.999 \
        --dropout 0.0 \
        --eval_interval 115 --latest_ckpt_interval 1000 \
        --log_interval 1 \
        --do_not_auto_resume \
        --wandb # \ --do_not_auto_resume \
done