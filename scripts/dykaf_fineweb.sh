#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
# torchrun --nproc_per_node=2 --master_port=$((BASE_PORT++)) \
#         --distributed_backend nccl \
# for iterations in 8000 16000 32000 48000 64000 128000
BASE_PORT=29501
for iterations in 64000
do
    python \
    ./src/main.py \
        --run_prefix time_abl \
        --model llama \
        --dataset fineweb \
        --optimizer dykaf \
        --dykaf_init kron \
        --precondition_frequency 10 \
        --lr 1e-3 \
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
        --wandb # \ --do_not_auto_resume \
done
