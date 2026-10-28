#!/bin/bash
# gigalearn/line.sh - one sweep line, called by gigalearn worker.sh in code/ with
# NAME ARGS POST SMOKE OBJECTIVE RUN_DIR EXPS_DIR PY DATA_DIR in the environment.
# Trains with src/main.py (auto resume from exps/<name>/ckpts/latest), then for
# post "measure" measures token gradients on every permanent checkpoint.
set -uo pipefail

PLACE="--datasets_dir $DATA_DIR --device ${DEVICE:-cuda:0}"
EXTRA=""
NSEQ="${MEASURE_N_SEQS:-32}"
if [ "${SMOKE:-0}" = "1" ] && [ "$POST" = "measure" ]; then
    EXTRA="--permanent_ckpt_interval 20"
    NSEQ=2
fi

# HGX preempts and requeues our jobs (PreemptMode=REQUEUE): FineWeb lines checkpoint every
# 200 steps (~1 min) so a requeued attempt loses little; later flags win over the sweep's value
TRAIN_EXTRA=""
case "$ARGS" in
    *"--dataset fineweb"*) [ "${SMOKE:-0}" = "1" ] || TRAIN_EXTRA="--latest_ckpt_interval 200" ;;
esac

# shellcheck disable=SC2086
"$PY" src/main.py $ARGS $EXTRA $TRAIN_EXTRA $PLACE \
    --experiment_name "$NAME" --results_base_folder "$EXPS_DIR" \
    --lab_run_work token-grad-averaging --lab_run_objective "$OBJECTIVE" \
    >> "$RUN_DIR/train.log" 2>&1 || exit $?

[ "$POST" = "measure" ] || exit 0
for CKPT in "$EXPS_DIR/$NAME/ckpts/"[0-9]*; do
    [ -d "$CKPT" ] || continue
    OUT="$RUN_DIR/tgc_$(printf '%06d' "$(basename "$CKPT")").pt"
    [ -f "$OUT" ] && continue
    # shellcheck disable=SC2086
    "$PY" src/token_grad_cov.py $ARGS $EXTRA $PLACE --ckpt_path "$CKPT" \
        --out_path "$OUT" --n_seqs "$NSEQ" >> "$RUN_DIR/measure.log" 2>&1 || exit $?
done
PLOT=""
"$PY" -c "import matplotlib" 2>/dev/null && PLOT="--plot $RUN_DIR/tgc_report.png"
# shellcheck disable=SC2086
"$PY" src/analysis/token_grad_cov_report.py "$RUN_DIR"/tgc_*.pt \
    --out_json "$RUN_DIR/tgc_report.json" --save_weights "$RUN_DIR/gls_weights.npy" $PLOT \
    >> "$RUN_DIR/measure.log" 2>&1
