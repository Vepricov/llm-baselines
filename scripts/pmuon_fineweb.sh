#!/bin/bash

# PMuon (Preconditioned Muon) baseline, ported from the modded-nanogpt record
# (gamma=0.3, cov_beta=0.95, lr=0.035, nesterov). 124M llama, fineweb.
# Baseline for the planned pmuon_dykaf (proj_split factors) comparison.

export CUDA_VISIBLE_DEVICES=5
for iterations in 64000
do
    python \
    ./src/main.py \
        --run_prefix time_abl \
        --model llama \
        --dataset fineweb \
        --optimizer pmuon \
        --muon_lr_factor 0.035 \
        --pmuon_gamma 0.3 \
        --pmuon_cov_beta 0.95 \
        --nesterov True \
        --muon_ns_steps 5 \
        --lr 3e-4 \
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
