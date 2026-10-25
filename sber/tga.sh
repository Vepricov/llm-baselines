#!/bin/bash
# sber/tga.sh - short commands for the token-grad-averaging sweep on the work machine.
#
# Install once:
#   curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/tga.sh -o ~/tga && chmod +x ~/tga
#
#   ~/tga smoke     new short test sweep (every line 40 steps)
#   ~/tga start     new real sweep
#   ~/tga status    summary of the last sweep (send me summary.md / the tar.gz)
#   ~/tga resume    cancel what of the last sweep is still queued and submit the rest again
#   ~/tga cancel    cancel the last sweep
#   ~/tga queue     my jobs and the partitions
#
# Settings go in front: PARTITION=H200 ~/tga resume, GPUS_PER_NODE=8 ~/tga start.
# Every call fetches the current scripts from the branch, so fixes arrive by themselves.
set -euo pipefail

EXP=tga_sweep
RAW="https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber"
[ -f "$HOME/gigalearn/config.env" ] && . "$HOME/gigalearn/config.env"
HOST="${GIGALEARN_HOST:?задай GIGALEARN_HOST в \$HOME/gigalearn/config.env}"
STATE="$HOME/gigalearn/last_sweep"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

fetch() {
    # the query string defeats the few-minute cache of raw.githubusercontent.com
    curl -fsSL "$RAW/$1?$(date +%s)"
}

last_id() {
    if [ -f "$STATE" ]; then
        cat "$STATE"
    else
        ssh -n "$HOST" "cd \$HOME/gigalearn/runs && for d in \$(ls -td ${EXP}_*/ 2>/dev/null); do echo \${d%/}; break; done" \
            | sed "s/^${EXP}_//"
    fi
}

submit() {
    # $1 = run id: a new one starts a sweep, an existing one continues it
    echo "$1" > "$STATE"
    fetch run.sh | RESUME_RUN="$1" bash -s -- "$EXP"
}

cancel() {
    ssh -n "$HOST" "scancel -n ${EXP}_$1 && echo 'снято: ${EXP}_$1'"
}

case "${1:-}" in
    smoke)  SMOKE=1 submit "$(date +%d-%m-%Y_%H-%M-%S)" ;;
    start)  SMOKE=0 submit "$(date +%d-%m-%Y_%H-%M-%S)" ;;
    status) fetch collect.sh | bash -s -- "${EXP}_$(last_id)" ;;
    resume)
        ID="$(last_id)"
        cancel "$ID"
        sleep 5
        submit "$ID"
        ;;
    cancel) cancel "$(last_id)" ;;
    queue)  ssh -n "$HOST" 'squeue -u $USER; echo; sinfo -s' ;;
    *)      sed -n '2,15p' "$0"; exit 2 ;;
esac
