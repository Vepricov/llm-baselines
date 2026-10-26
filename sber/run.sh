#!/bin/bash
# sber/run.sh - одна команда на рабочей машине:
#   код с GitHub -> кластер -> запуск -> результаты обратно в отдельную папку.
#
# Использование:
#   bash run.sh <experiment> [host]
#   DATASET=shakespeare-char ITERATIONS=200 bash run.sh mlmc_pretrain
#
# Конфиг (один раз): $HOME/gigalearn/config.env
#   GIGALEARN_HOST=user@cluster-host
#   GIGALEARN_IMAGE=/path/to/image.sqsh
#
# Раскладка и на рабочей машине, и на кластере: gigalearn/<проект>/runs/<эксперимент>_<дата-время>/,
# проект по умолчанию = ветка (PROJECT=...). Общие тяжёлые вещи кластера: gigalearn/base/.
#
# Переменные: PROJECT, GIGALEARN_REPO, GIGALEARN_BRANCH, GIGALEARN_HOME,
#             DATASET, OPTIMIZER, LR, ITERATIONS, WARMUP, BATCH, SEQ, N_EMBD, N_HEAD, N_LAYER
set -euo pipefail

EXP="${1:?usage: run.sh <experiment> [host]}"
[ -f "$HOME/gigalearn/config.env" ] && . "$HOME/gigalearn/config.env"
HOST="${2:-${GIGALEARN_HOST:-}}"
[ -n "$HOST" ] || { echo "error: задай GIGALEARN_HOST или положи его в \$HOME/gigalearn/config.env"; exit 2; }

REPO="${GIGALEARN_REPO:-Vepricov/llm-baselines}"
BRANCH="${GIGALEARN_BRANCH:-tga}"
BASE="${GIGALEARN_HOME:-$HOME/gigalearn}"
PROJECT="${PROJECT:-$BRANCH}"
RUN_ID="$(date +%d-%m-%Y_%H-%M-%S)"
RUN_DIR="$BASE/$PROJECT/runs/${EXP}_${RUN_ID}"
mkdir -p "$RUN_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 1/3 код: $REPO@$BRANCH"
curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP/src.tar.gz"
CODE_SHA="$(shasum -a 256 "$TMP/src.tar.gz" | cut -c1-12)"
echo "   $(du -h "$TMP/src.tar.gz" | cut -f1), sha256: $CODE_SHA"

# Переменные запуска едут на кластер явно: ssh сам их не пробрасывает.
REMOTE_ENV="DATASET='${DATASET:-}' OPTIMIZER='${OPTIMIZER:-}' LR='${LR:-}' \
ITERATIONS='${ITERATIONS:-}' WARMUP='${WARMUP:-}' BATCH='${BATCH:-}' SEQ='${SEQ:-}' \
N_EMBD='${N_EMBD:-}' N_HEAD='${N_HEAD:-}' N_LAYER='${N_LAYER:-}' \
SMOKE='${SMOKE:-}' MAX_NODES='${MAX_NODES:-}' GPUS_PER_NODE='${GPUS_PER_NODE:-}' \
PARTITION='${PARTITION:-}' TIME_LIMIT='${TIME_LIMIT:-}' \
GIGALEARN_IMAGE='${GIGALEARN_IMAGE:-}' CODE_SHA='$CODE_SHA' BRANCH='$BRANCH' PROJECT='$PROJECT'"

REMOTE_HOME="$(ssh -n "$HOST" 'printf %s "$HOME"')"
echo "== 2/3 кластер: $HOST ($REMOTE_HOME/gigalearn/$PROJECT)"
ssh "$HOST" "set -e; R=\$HOME/gigalearn/$PROJECT/repo; mkdir -p \$R; tar xzf - -C \$R --strip-components=1; env $REMOTE_ENV bash \$R/sber/cluster_run.sh '$EXP' '$RUN_ID'" < "$TMP/src.tar.gz"

echo "== 3/3 результаты"
if scp -q -r "$HOST:$REMOTE_HOME/gigalearn/$PROJECT/runs/${EXP}_${RUN_ID}/results/." "$RUN_DIR/" 2>/dev/null; then
    echo "готово: $RUN_DIR"
    ls -la "$RUN_DIR"
else
    echo "результатов нет; лог на кластере: $REMOTE_HOME/gigalearn/$PROJECT/runs/${EXP}_${RUN_ID}/"
fi
