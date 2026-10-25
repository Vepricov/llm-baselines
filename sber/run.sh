#!/bin/bash
# sber/run.sh - одна команда на рабочей машине: код с GitHub -> кластер -> запуск -> результаты обратно.
#
# Использование:
#   GIGALEARN_HOST=user@cluster-host bash run.sh <experiment>
#   GIGALEARN_HOST=user@cluster-host bash run.sh <experiment> <host>
#
# Конфиг (необязательно): $HOME/gigalearn/config.env с переменными
#   GIGALEARN_HOST=user@cluster-host
#   GIGALEARN_IMAGE=/path/to/image.sqsh
#
# Переменные:
#   GIGALEARN_REPO   (по умолчанию Vepricov/llm-baselines)
#   GIGALEARN_BRANCH (по умолчанию sber)
#   GIGALEARN_HOME   (по умолчанию $HOME/gigalearn)
set -euo pipefail

EXP="${1:?usage: run.sh <experiment> [host]}"
[ -f "$HOME/gigalearn/config.env" ] && . "$HOME/gigalearn/config.env"
HOST="${2:-${GIGALEARN_HOST:-}}"
[ -n "$HOST" ] || { echo "error: задай GIGALEARN_HOST (user@host) или положи его в \$HOME/gigalearn/config.env"; exit 2; }

REPO="${GIGALEARN_REPO:-Vepricov/llm-baselines}"
BRANCH="${GIGALEARN_BRANCH:-sber}"
BASE="${GIGALEARN_HOME:-$HOME/gigalearn}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$BASE/runs/${EXP}_${STAMP}"
mkdir -p "$RUN_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 1/3 код: $REPO@$BRANCH"
curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP/src.tar.gz"
echo "   $(du -h "$TMP/src.tar.gz" | cut -f1)"

echo "== 2/3 кластер: $HOST"
ssh "$HOST" "set -e; mkdir -p ~/gigalearn/repo; tar xzf - -C ~/gigalearn/repo --strip-components=1; GIGALEARN_IMAGE='${GIGALEARN_IMAGE:-}' bash ~/gigalearn/repo/sber/cluster_run.sh '$EXP' '$STAMP'" < "$TMP/src.tar.gz"

echo "== 3/3 результаты"
if scp -q -r "$HOST:~/gigalearn/runs/${EXP}_${STAMP}/results" "$RUN_DIR/" 2>/dev/null; then
    echo "готово: $RUN_DIR"
    ls -la "$RUN_DIR/results" 2>/dev/null | head -12
else
    echo "результатов нет, смотри лог на кластере: ~/gigalearn/runs/${EXP}_${STAMP}/"
fi
