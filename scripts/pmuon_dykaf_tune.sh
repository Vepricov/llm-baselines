#!/bin/bash

# Parametrized pmuon_dykaf tuning launcher (bs32, vs SOAP ref ~3.45).
# One knob per run; env-var controlled so adding a variant is one line.
#   GPU TAG ITERS GAMMA COVBETA PFREQ WARMUP WD LR ADAMW
# Baseline: GAMMA=0.3 COVBETA=0.95 PFREQ=10 WARMUP=1000 WD=0.1 LR=0.035 ADAMW=5e-4

GPU=${GPU:-2}
TAG=${TAG:-pmt}
ITERS=${ITERS:-16000}
GAMMA=${GAMMA:-0.3}
COVBETA=${COVBETA:-0.95}
PFREQ=${PFREQ:-10}
WARMUP=${WARMUP:-1000}
WD=${WD:-0.1}
LR=${LR:-0.035}
ADAMW=${ADAMW:-5e-4}

export CUDA_VISIBLE_DEVICES=$GPU
python \
./src/main.py \
    --run_prefix "$TAG" \
    --model llama \
    --dataset fineweb \
    --optimizer pmuon_dykaf \
    --muon_lr_factor "$LR" \
    --pmuon_gamma "$GAMMA" \
    --pmuon_cov_beta "$COVBETA" \
    --precondition_frequency "$PFREQ" \
    --nesterov True \
    --muon_ns_steps 5 \
    --lr "$ADAMW" \
    --iterations "$ITERS" \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 512 \
    --acc_steps 1 \
    --warmup_steps "$WARMUP" \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay "$WD" \
    --scheduler cos \
    --beta1 0.9 --beta2 0.999 \
    --dropout 0.0 \
    --eval_interval 200 --latest_ckpt_interval 100000 \
    --log_interval 50 \
    --do_not_auto_resume \
    --wandb_project MIKOLA_DROP_SOAP \
    --wandb
