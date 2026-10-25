#!/bin/bash
# sber/cluster_run.sh - запускается НА КЛАСТЕРЕ, его вызывает sber/run.sh.
# Собирает отдельную папку прогона, пишет README прогона, ставит задачу, ждёт, собирает результаты.
set -euo pipefail

EXP="${1:?experiment}"
RUN_ID="${2:?run id}"

ROOT="$HOME/gigalearn"
PROJECT="${PROJECT:-${BRANCH:-sber}}"
SRC="$ROOT/$PROJECT/repo/sber"
TASK="$ROOT/$PROJECT/runs/${EXP}_${RUN_ID}"
BASE="$ROOT/base"

# experiments with sweep_<dataset>.txt are Slurm arrays, see sber/cluster_sweep.sh
if ls "$SRC/experiments/$EXP"/sweep_*.txt >/dev/null 2>&1; then
    exec bash "$SRC/cluster_sweep.sh" "$EXP" "$RUN_ID"
fi

mkdir -p "$TASK/logs" "$TASK/results"

[ -f "$SRC/experiments/$EXP/train_slurm.sh" ] || {
    echo "error: нет эксперимента '$EXP' (ожидался $SRC/experiments/$EXP)"; exit 2; }

cp "$SRC/experiments/$EXP/"*.sh "$TASK/"
cp "$SRC/experiments/$EXP/README.md" "$TASK/experiment.md" 2>/dev/null || true
cp "$SRC/experiments/$EXP/idea.md" "$TASK/idea.md" 2>/dev/null || true
cp "$SRC/plot_metrics.py" "$SRC/run_bundle.sh" "$TASK/" 2>/dev/null || true
# slurm_startup.sh runs src/main.py relative to the run dir, so the code goes there too
cp -r "$ROOT/$PROJECT/repo/src" "$TASK/"

# Постоянные тяжёлые вещи живут в base и в прогон не копируются.
for d in pylibs tiktoken_cache data; do
    [ -e "$BASE/$d" ] && ln -sfn "$BASE/$d" "$TASK/$d"
done

# Значения по умолчанию совпадают с slurm_startup.sh, чтобы README прогона был правдой.
export DATASET="${DATASET:-fineweb}"
export OPTIMIZER="${OPTIMIZER:-adamw}"
export LR="${LR:-3e-3}"
export ITERATIONS="${ITERATIONS:-20000}"
export WARMUP="${WARMUP:-2000}"
export BATCH="${BATCH:-256}"
export SEQ="${SEQ:-512}"
export N_EMBD="${N_EMBD:-768}"
export N_HEAD="${N_HEAD:-12}"
export N_LAYER="${N_LAYER:-12}"

IMAGE="${GIGALEARN_IMAGE:-}"
[ -n "$IMAGE" ] || { echo "error: не задан GIGALEARN_IMAGE"; exit 2; }

sed -i "s|@LOG_DIR@|$TASK/logs|g; s|@IMAGE@|$IMAGE|g" "$TASK/train_slurm.sh" "$TASK/slurm_startup.sh"

IDEA="$(cat "$TASK/idea.md" 2>/dev/null || echo '(идея не описана: добавь experiments/'"$EXP"'/idea.md)')"

cat > "$TASK/README.md" <<EOF
# Прогон ${EXP} — ${RUN_ID}

## Идея
${IDEA}

## Параметры
- dataset: ${DATASET}
- optimizer: ${OPTIMIZER}
- lr: ${LR}, warmup: ${WARMUP}
- iterations: ${ITERATIONS}
- batch: ${BATCH} x seq ${SEQ}, acc_steps 1
- model: llama ${N_EMBD}/${N_HEAD}/${N_LAYER}
- image: ${IMAGE}

## Воспроизводимость
- ветка кода: ${BRANCH:-sber}
- sha256 архива кода: ${CODE_SHA:-неизвестно}
- запуск: $(date '+%d-%m-%Y %H:%M:%S')
- узел: (заполнится после старта)

## Где что
- логи: logs/
- отчёт: results/report_*.txt
- метрики: results/metrics.csv
- график: results/training_curve.svg (и .png если есть matplotlib)
- чекпоинты: exps/

## Результат
(заполнить после прогона: финальный val_loss, наблюдения, вывод)
EOF

if [ ! -f "$BASE/data/fineweb-100BT/train.bin" ] && [ "$DATASET" = "fineweb" ]; then
    echo "WARNING: нет датасета $BASE/data/fineweb-100BT/train.bin — задача, скорее всего, упадёт"
fi

echo "== задача: $EXP, прогон $RUN_ID"
cd "$TASK"
bash run_bundle.sh --watch

REPORT="$(ls -t report_*.txt 2>/dev/null | head -1 || true)"
JOB="$(grep -m1 '^job id:' "$REPORT" 2>/dev/null | tr -dc '0-9' || true)"
NODE="$(sacct -j "$JOB" --format=NodeList -n 2>/dev/null | head -1 | tr -d ' ' || true)"

{
    echo "job: $JOB"
    echo "run: $RUN_ID"
    echo "date: $(date '+%d-%m-%Y %H:%M:%S')"
    echo "node: $NODE"
    sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed 2>/dev/null | head -6
} > "$TASK/results/status.txt"

if [ -n "$NODE" ]; then
    sed -i "s|^- узел: .*|- узел: $NODE|" "$TASK/README.md"
fi

cp "$TASK/README.md" "$TASK/idea.md" "$TASK/results/" 2>/dev/null || true
cp "$TASK"/report_*.txt "$TASK/results/" 2>/dev/null || true
for f in metrics.csv training_curve.svg training_curve.png; do
    [ -f "$TASK/$f" ] && cp "$TASK/$f" "$TASK/results/"
done

echo "== результаты: $TASK/results"
ls -la "$TASK/results"
