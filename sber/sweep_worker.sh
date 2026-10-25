#!/bin/bash
# sber/sweep_worker.sh <line index> - one sweep line on one GPU.
# Called by slurm_startup.sh with CUDA_VISIBLE_DEVICES set; needs TASK_DIR and DATA_DIR.
# A line whose run dir has DONE is skipped, so resubmitting a sweep only redoes the rest,
# and an interrupted run continues from exps/<name>/ckpts/latest (auto resume of main.py).
set -uo pipefail

IDX="${1:?line index}"
PY="${PY:-python}"
cd "${TASK_DIR:?}"
SWEEP="$TASK_DIR/sweep.txt"

LINE="$(grep -v -e '^#' -e '^@' -e '^[[:space:]]*$' "$SWEEP" | sed -n "$((IDX + 1))p")"
[ -n "$LINE" ] || exit 0
COMMON="$(grep -m1 '^@common' "$SWEEP" | sed 's/^@common//')"
OBJECTIVE="$(grep -m1 '^@objective' "$SWEEP" | sed 's/^@objective *//')"
NAME="$(echo "$LINE" | cut -d'|' -f1 | xargs)"
FLAGS="$(echo "$LINE" | cut -d'|' -f2)"
POST="$(echo "$LINE" | cut -s -d'|' -f3 | xargs)"

RUN="$TASK_DIR/runs/$NAME"
mkdir -p "$RUN"
if [ -f "$RUN/DONE" ]; then
    echo "$NAME: already done, skipped"
    exit 0
fi
rm -f "$RUN/STATUS"

SMOKE_FLAGS=""
NSEQ="${MEASURE_N_SEQS:-32}"
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAGS="--iterations 40 --eval_interval 20 --eval_batches 2 --log_interval 10 --warmup_steps 5 --latest_ckpt_interval 20"
    [ "$POST" = "measure" ] && SMOKE_FLAGS="$SMOKE_FLAGS --permanent_ckpt_interval 20"
    NSEQ=2
fi
# EXTRA_ARGS is for local rehearsal only (tiny model on cpu)
ARGS="$COMMON $FLAGS $SMOKE_FLAGS ${EXTRA_ARGS:-}"
PLACE="--datasets_dir $DATA_DIR --device ${DEVICE:-cuda:0}"

echo "$(date '+%d-%m-%Y %H:%M:%S') host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} args=$ARGS" >> "$RUN/attempts.txt"
echo "$NAME: training"
# shellcheck disable=SC2086
"$PY" src/main.py $ARGS $PLACE \
    --experiment_name "$NAME" --results_base_folder "$TASK_DIR/exps" \
    --lab_run_work token-grad-averaging --lab_run_objective "$OBJECTIVE" \
    >> "$RUN/train.log" 2>&1
RC=$?

if [ "$RC" -eq 0 ] && [ "$POST" = "measure" ]; then
    echo "$NAME: measuring token gradients"
    for CKPT in "$TASK_DIR/exps/$NAME/ckpts/"[0-9]*; do
        [ -d "$CKPT" ] || continue
        OUT="$RUN/tgc_$(printf '%06d' "$(basename "$CKPT")").pt"
        [ -f "$OUT" ] && continue
        # shellcheck disable=SC2086
        "$PY" src/token_grad_cov.py $ARGS $PLACE --ckpt_path "$CKPT" \
            --out_path "$OUT" --n_seqs "$NSEQ" >> "$RUN/measure.log" 2>&1 || { RC=$?; break; }
    done
    if [ "$RC" -eq 0 ]; then
        PLOT=""
        "$PY" -c "import matplotlib" 2>/dev/null && PLOT="--plot $RUN/tgc_report.png"
        # shellcheck disable=SC2086
        "$PY" src/analysis/token_grad_cov_report.py "$RUN"/tgc_*.pt \
            --out_json "$RUN/tgc_report.json" --save_weights "$RUN/gls_weights.npy" $PLOT \
            >> "$RUN/measure.log" 2>&1 || RC=$?
    fi
fi

echo "$RC" > "$RUN/STATUS"
[ "$RC" -eq 0 ] && touch "$RUN/DONE"
echo "$NAME: exit $RC"
exit 0
