#!/bin/bash
# sber/giga.sh - short commands for sweeps on GigaLearn, for any project.
#
# Install once (then open a new terminal):
#   curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/giga.sh -o $HOME/gigalearn/giga && bash $HOME/gigalearn/giga setup
#
#   giga smoke [project]    new short test sweep, every line 40 steps
#   giga start [project]    new real sweep
#   giga status [project]   summary of the last sweep; send summary.tar.gz from the printed path
#   giga resume [project]   cancel what of the last sweep is still queued and submit the rest again
#   giga cancel [project]   cancel every queued or running sweep job of the project
#   giga queue              my jobs and the partitions
#   giga update             fetch the newest giga
#
# A project is a branch of the code repository with sber/experiments/<project>_sweep/.
# Without a project the last used one is taken (first time: tga).
# Settings go in front: PARTITION=H200 giga resume, GPUS_PER_NODE=8 giga start.
# Folders: $HOME/gigalearn/<project>/runs/<sweep>/ here and on the cluster.
set -euo pipefail

BASE="$HOME/gigalearn"
[ -f "$BASE/config.env" ] && . "$BASE/config.env"
HOST="${GIGALEARN_HOST:?задай GIGALEARN_HOST в $BASE/config.env}"
REPO="${GIGALEARN_REPO:-Vepricov/llm-baselines}"
CMD="${1:-}"

PROJECT="${2:-$(cat "$BASE/last_project" 2>/dev/null || echo tga)}"
BRANCH="${GIGALEARN_BRANCH:-$PROJECT}"
EXP="${EXP:-${PROJECT}_sweep}"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH/sber"
STATE="$BASE/$PROJECT/last_sweep"
export PROJECT GIGALEARN_BRANCH="$BRANCH"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
mkdir -p "$BASE/$PROJECT"
echo "$PROJECT" > "$BASE/last_project"

fetch() {
    # the query string defeats the few-minute cache of raw.githubusercontent.com
    curl -fsSL "$RAW/$1?$(date +%s)"
}

last_id() {
    if [ -f "$STATE" ]; then
        cat "$STATE"
    else
        ssh -n "$HOST" "cd \$HOME/gigalearn/$PROJECT/runs 2>/dev/null && ls -td ${EXP}_*/ 2>/dev/null | head -1" \
            | sed "s|/$||; s|^${EXP}_||"
    fi
}

submit() {
    # $1 = run id: a new one starts a sweep, an existing one continues it
    echo "$1" > "$STATE"
    fetch run.sh | RESUME_RUN="$1" bash -s -- "$EXP"
}

case "$CMD" in
    smoke)  SMOKE=1 submit "$(date +%d-%m-%Y_%H-%M-%S)" ;;
    start)  SMOKE=0 submit "$(date +%d-%m-%Y_%H-%M-%S)" ;;
    status) fetch collect.sh | bash -s -- "${EXP}_$(last_id)" ;;
    resume)
        ID="$(last_id)"
        [ -n "$ID" ] || { echo "error: у проекта $PROJECT ещё нет свипов, сначала giga smoke"; exit 2; }
        ssh -n "$HOST" "scancel -n ${EXP}_$ID"
        sleep 5
        submit "$ID"
        ;;
    cancel)
        ssh -n "$HOST" "ids=\$(squeue -u \$USER -h -o '%A %j' | awk '\$2 ~ /^${EXP}_/ {print \$1}' | sort -u)
            [ -n \"\$ids\" ] && scancel \$ids && echo \"снято: \$(echo \$ids)\" || echo 'снимать нечего'"
        ;;
    queue)  ssh -n "$HOST" 'squeue -u $USER; echo; sinfo -s' ;;
    update)
        fetch giga.sh > "$BASE/giga.new" && mv "$BASE/giga.new" "$BASE/giga" && chmod +x "$BASE/giga"
        echo "обновлено: $BASE/giga"
        ;;
    setup)
        chmod +x "$BASE/giga"
        for rc in "$HOME/.bash_profile" "$HOME/.zshrc"; do
            grep -q 'alias giga=' "$rc" 2>/dev/null || echo "alias giga=\"$BASE/giga\"" >> "$rc"
        done
        echo "готово: открой новый терминал и пиши giga status"
        ;;
    *)  sed -n '2,19p' "$BASE/giga" 2>/dev/null || sed -n '2,19p' "$0"; exit 2 ;;
esac
