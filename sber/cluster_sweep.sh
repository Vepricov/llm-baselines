#!/bin/bash
# sber/cluster_sweep.sh - runs ON THE CLUSTER, cluster_run.sh hands over experiments
# that have sweep_<dataset>.txt. Snapshots code and sweep into runs/<exp>_<run id>/,
# submits one Slurm array (one task = one node = 8 lines) and returns at once.
# The same run id again = resume: finished lines are skipped, the rest continue.
set -euo pipefail

EXP="${1:?experiment}"
RUN_ID="${2:?run id}"

ROOT="$HOME/gigalearn"
PROJECT="${PROJECT:-${BRANCH:-sber}}"
REPO="$ROOT/$PROJECT/repo"
EXP_SRC="$REPO/sber/experiments/$EXP"
TASK="$ROOT/$PROJECT/runs/${EXP}_${RUN_ID}"
BASE="$ROOT/base"
JOB_NAME="${EXP}_${RUN_ID}"
# GPUs per array task: 8 = whole nodes, 1 = one job per line (gets through a busy queue
# faster). A short honest TIME_LIMIT lets Slurm backfill the jobs into gaps.
GPUS="${GPUS_PER_NODE:-8}"
PARTITION="${PARTITION:-HGX}"

if [ -d "$TASK" ]; then
    echo "== продолжение свипа $JOB_NAME"
    if squeue -h -n "$JOB_NAME" -t PENDING,RUNNING 2>/dev/null | grep -q .; then
        echo "error: задачи свипа ещё в очереди (squeue -n $JOB_NAME), сначала дождись или scancel"
        exit 2
    fi
    # shellcheck disable=SC1091
    . "$TASK/sweep.env"
    # the code is refreshed, so a fix pushed after a failure reaches the remaining lines;
    # sweep.txt and the finished runs stay as they were
    rm -rf "$TASK/src" "$TASK/sber"
    cp -r "$REPO/src" "$REPO/sber" "$TASK/"
else
    export DATASET="${DATASET:-shakespeare-char}"
    export SMOKE="${SMOKE:-0}"
    SWEEP_SRC="$EXP_SRC/sweep_${DATASET}.txt"
    [ -f "$SWEEP_SRC" ] || { echo "error: нет $SWEEP_SRC (свип под датасет $DATASET не описан)"; exit 2; }
    mkdir -p "$TASK/logs" "$TASK/results" "$TASK/runs"
    # code snapshot: later pushes to the branch never change a running sweep
    cp -r "$REPO/src" "$REPO/sber" "$TASK/"
    cp "$SWEEP_SRC" "$TASK/sweep.txt"
    cp "$EXP_SRC/idea.md" "$TASK/idea.md" 2>/dev/null || true
    for d in pylibs tiktoken_cache data; do
        [ -e "$BASE/$d" ] && ln -sfn "$BASE/$d" "$TASK/$d"
    done
    printf 'export DATASET=%q\nexport SMOKE=%q\n' "$DATASET" "$SMOKE" > "$TASK/sweep.env"
fi

case "$DATASET" in
    shakespeare-char) NEED="$TASK/sber/data/shakespeare/raw.txt" ;;
    fineweb) NEED="$BASE/data/fineweb-100BT/train.bin" ;;
    finewebedu) NEED="$BASE/data/fineweb-edu-100BT/train.bin" ;;
    *) NEED="" ;;
esac
if [ -n "$NEED" ] && [ ! -f "$NEED" ]; then
    echo "error: нет данных $NEED, свип не ставлю"
    exit 2
fi

if [ -z "${TIME_LIMIT:-}" ]; then
    if [ "$SMOKE" = "1" ]; then TIME_LIMIT="00:30:00"; else TIME_LIMIT="24:00:00"; fi
fi
IMAGE="${GIGALEARN_IMAGE:-$(cat "$TASK/image.txt" 2>/dev/null || true)}"
[ -n "$IMAGE" ] || { echo "error: не задан GIGALEARN_IMAGE"; exit 2; }
echo "$IMAGE" > "$TASK/image.txt"
# train_slurm.sh is rendered anew on every submission, so a resume may change these
sed "s|@LOG_DIR@|$TASK/logs|g; s|@IMAGE@|$IMAGE|g; s|@GPUS@|$GPUS|g; s|@PARTITION@|$PARTITION|g; s|@TIME_LIMIT@|$TIME_LIMIT|g" \
    "$TASK/sber/experiments/$EXP/train_slurm.sh" > "$TASK/train_slurm.sh"
cp "$TASK/sber/experiments/$EXP/slurm_startup.sh" "$TASK/"

N="$( (grep -v -e '^#' -e '^@' -e '^[[:space:]]*$' "$TASK/sweep.txt" || true) | wc -l | tr -d ' ')"
DONE="$( (ls "$TASK"/runs/*/DONE 2>/dev/null || true) | wc -l | tr -d ' ')"
TASKS=$(( (N + GPUS - 1) / GPUS ))
MAX_NODES="${MAX_NODES:-$TASKS}"
[ "$MAX_NODES" -gt 0 ] || MAX_NODES=1

cat > "$TASK/README.md" <<README
# Свип ${EXP} — ${RUN_ID}

## Идея
$(cat "$TASK/idea.md" 2>/dev/null || echo '(нет idea.md)')

## Параметры
- dataset: ${DATASET}, smoke: ${SMOKE}
- строк: ${N}, задач массива: ${TASKS} по ${GPUS} GPU (${PARTITION}, лимит ${TIME_LIMIT}), одновременно не больше ${MAX_NODES}
- ветка кода: ${BRANCH:-?}, sha256 архива: ${CODE_SHA:-?} (все попытки: code_history.txt)
- образ: $(grep -m1 'container-image' "$TASK/train_slurm.sh" | sed 's/.*="//; s/".*//')
- каждая строка свипа: sweep.txt, своя папка runs/<имя>/, чекпоинты exps/<имя>/

## Как продолжить
Тот же запуск с RESUME_RUN=${RUN_ID}: готовые строки пропускаются, прерванные
продолжаются с последнего чекпоинта.
README

echo "$(date '+%d-%m-%Y %H:%M:%S') branch=${BRANCH:-?} sha256=${CODE_SHA:-?}" >> "$TASK/code_history.txt"

cd "$TASK"
OUT="$(sbatch --job-name="$JOB_NAME" --array="0-$((TASKS - 1))%${MAX_NODES}" train_slurm.sh)"
echo "$OUT"
{
    echo "sweep: $JOB_NAME"
    echo "submitted: $(date '+%d-%m-%Y %H:%M:%S')"
    echo "lines: $N (already done: $DONE), array tasks: $TASKS, max nodes: $MAX_NODES"
    echo "dataset: $DATASET, smoke: $SMOKE, partition: $PARTITION, gpus per task: $GPUS, time: $TIME_LIMIT"
    echo "$OUT"
} >> "$TASK/results/submitted.txt"
cp "$TASK/README.md" "$TASK/results/"
echo "== свип поставлен: $N строк, $TASKS задач по $GPUS GPU на $PARTITION, лимит $TIME_LIMIT, уже готово $DONE"
