#!/bin/bash
# sber/giga.sh - short commands for sweeps on GigaLearn, for any project.
#
# Install once (then open a new terminal):
#   curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/giga.sh -o $HOME/gigalearn/giga && bash $HOME/gigalearn/giga setup
#
# The project is the folder you stand in: cd $HOME/gigalearn/<project>, then
#   giga smoke      new short test sweep, every line 40 steps
#   giga start      new real sweep
#   giga status     summary of the last sweep; send summary.tar.gz from the printed path
#   giga resume     cancel what of the last sweep is still queued and submit the rest again
#   giga cancel     cancel every queued or running sweep job of the project
#   giga queue      my jobs and the partitions (any folder)
#   giga new NAME   make the folder of a new project (any folder)
#   giga update     fetch the newest giga (any folder)
#
# A project is a branch of the code repository with sber/experiments/<project>_sweep/.
# Settings go in front: PARTITION=H200 giga resume, GPUS_PER_NODE=8 giga start.
# Folders: $HOME/gigalearn/<project>/runs/<sweep>/ here and on the cluster.
set -euo pipefail

BASE="$HOME/gigalearn"
[ -f "$BASE/config.env" ] && . "$BASE/config.env"
HOST="${GIGALEARN_HOST:?задай GIGALEARN_HOST в $BASE/config.env}"
REPO="${GIGALEARN_REPO:-Vepricov/llm-baselines}"
CMD="${1:-}"

# the project is the first folder under $BASE on the way to the current folder
case "$PWD/" in
    "$BASE"/*/*) PROJECT="${PROJECT:-$(echo "${PWD#"$BASE"/}" | cut -d/ -f1)}" ;;
    *) PROJECT="${PROJECT:-}" ;;
esac
case "$PROJECT" in runs|base) PROJECT="" ;; esac
if [ -z "$PROJECT" ]; then
    case "$CMD" in
        smoke|start|status|resume|cancel)
            echo "error: зайди в папку проекта, например: cd $BASE/tga"
            echo "проекты: $(cd "$BASE" && ls -d */ 2>/dev/null | tr -d / | grep -v -x -e runs -e base | tr '\n' ' ')"
            exit 2
            ;;
    esac
    PROJECT=none
fi
BRANCH="${GIGALEARN_BRANCH:-$PROJECT}"
EXP="${EXP:-${PROJECT}_sweep}"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH/sber"
STATE="$BASE/$PROJECT/last_sweep"
export PROJECT GIGALEARN_BRANCH="$BRANCH"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

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
    new)
        NAME="${2:?giga new <имя проекта>}"
        mkdir -p "$BASE/$NAME"
        echo "папка проекта: $BASE/$NAME"
        echo "дальше: cd $BASE/$NAME и giga smoke (нужна ветка $NAME с sber/experiments/${NAME}_sweep)"
        ;;
    update)
        fetch giga.sh > "$BASE/giga.new" && mv "$BASE/giga.new" "$BASE/giga" && chmod +x "$BASE/giga"
        echo "обновлено: $BASE/giga"
        ;;
    setup)
        chmod +x "$BASE/giga"
        for rc in "$HOME/.bash_profile" "$HOME/.zshrc"; do
            grep -q 'alias giga=' "$rc" 2>/dev/null || echo "alias giga=\"$BASE/giga\"" >> "$rc"
        done
        mkdir -p "$BASE/tga"
        echo "готово: открой новый терминал, затем"
        echo "  cd $BASE/tga"
        echo "  giga status"
        ;;
    *)  echo "проект здесь: $PROJECT"; sed -n '6,19p' "$BASE/giga" 2>/dev/null || sed -n '6,19p' "$0"; exit 2 ;;
esac
