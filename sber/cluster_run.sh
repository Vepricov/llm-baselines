#!/bin/bash
# sber/cluster_run.sh - запускается НА КЛАСТЕРЕ (его вызывает run.sh).
# Собирает отдельную папку прогона, подставляет пути, ставит задачу, ждёт, собирает результаты.
set -euo pipefail

EXP="${1:?experiment}"
STAMP="${2:?stamp}"

ROOT="$HOME/gigalearn"
SRC="$ROOT/repo/sber"
TASK="$ROOT/runs/${EXP}_${STAMP}"
BASE="$ROOT/base"

mkdir -p "$TASK/logs" "$TASK/results"

[ -f "$SRC/experiments/$EXP/train_slurm.sh" ] || {
    echo "error: нет эксперимента '$EXP' (ожидался $SRC/experiments/$EXP)"; exit 2; }

cp "$SRC/experiments/$EXP/"*.sh "$TASK/"
cp "$SRC/experiments/$EXP/README.md" "$TASK/" 2>/dev/null || true
cp "$SRC/plot_metrics.py" "$SRC/run_bundle.sh" "$TASK/" 2>/dev/null || true

# Постоянные тяжёлые вещи лежат в base, в прогон не копируются.
for d in pylibs tiktoken_cache data; do
    [ -e "$BASE/$d" ] && ln -sfn "$BASE/$d" "$TASK/$d"
done

IMAGE="${GIGALEARN_IMAGE:-}"
if [ -z "$IMAGE" ] && [ -f "$HOME/gigalearn/config.env" ]; then
    IMAGE="$(grep -m1 '^GIGALEARN_IMAGE=' "$HOME/gigalearn/config.env" | cut -d= -f2- || true)"
fi
[ -n "$IMAGE" ] || { echo "error: не задан GIGALEARN_IMAGE"; exit 2; }

sed -i "s|@LOG_DIR@|$TASK/logs|g; s|@IMAGE@|$IMAGE|g" "$TASK/train_slurm.sh" "$TASK/slurm_startup.sh"

if [ ! -f "$BASE/data/fineweb-100BT/train.bin" ]; then
    echo "WARNING: нет датасета $BASE/data/fineweb-100BT/train.bin"
fi

echo "== задача: $EXP ($STAMP)"
cd "$TASK"
bash run_bundle.sh --watch

# Результаты в отдельную папку прогона (ничего не перезатирается: у каждого прогона свой STAMP).
sacct -j "$(basename "$(ls -t report_*.txt | head -1)" .txt | tr -dc '0-9')" \
      --format=JobID,State,ExitCode,Elapsed 2>/dev/null | head -6 > "$TASK/results/status.txt" || true
cp "$TASK"/report_*.txt "$TASK/results/" 2>/dev/null || true
for f in metrics.csv training_curve.svg training_curve.png; do
    [ -f "$TASK/$f" ] && cp "$TASK/$f" "$TASK/results/"
done

echo "== результаты: $TASK/results"
ls -la "$TASK/results"
