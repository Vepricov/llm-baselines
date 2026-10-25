#!/bin/bash
# sber/collect.sh - on the work machine: build the sweep summary on the cluster and copy it back.
# Usually called by giga (giga status). By hand:
#   curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/<branch>/sber/collect.sh \
#       | PROJECT=<project> bash -s -- [<sweep name or its beginning>]
#
# Without a name takes the project's most recent sweep. Prints summary.md and leaves
#   <gigalearn>/<project>/runs/<sweep>/summary/          the summary
#   <gigalearn>/<project>/runs/<sweep>/summary.tar.gz    the same packed, to send on
set -euo pipefail

[ -f "$HOME/gigalearn/config.env" ] && . "$HOME/gigalearn/config.env"
HOST="${GIGALEARN_HOST:?задай GIGALEARN_HOST в $HOME/gigalearn/config.env}"
BASE="${GIGALEARN_HOME:-$HOME/gigalearn}"
PROJECT="${PROJECT:?задай PROJECT}"
NAME="${1:-}"
RUNS="gigalearn/$PROJECT/runs"   # relative to the home dir on the cluster

REMOTE_HOME="$(ssh -n "$HOST" 'printf %s "$HOME"')"
# no name: newest sweep; a cut or shortened name is completed by prefix if unambiguous
NAME="$(ssh -n "$HOST" "cd \$HOME/$RUNS 2>/dev/null || exit 0
    if [ -z '$NAME' ]; then
        for d in \$(ls -td */ 2>/dev/null); do [ -f \"\$d/sweep.txt\" ] && { echo \"\${d%/}\"; break; }; done
    else
        m=\$(ls -d '$NAME'* 2>/dev/null)
        [ \$(echo \"\$m\" | grep -c .) -eq 1 ] && echo \"\${m%/}\" || echo '$NAME'
    fi")"
[ -n "$NAME" ] || { echo "error: в $REMOTE_HOME/$RUNS нет ни одного свипа"; exit 2; }
echo "== свип: $NAME"

ssh -n "$HOST" "cd \$HOME/$RUNS/$NAME && python3 sber/sweep_summary.py . > /dev/null && \
    { squeue -h -n '$NAME' -o '%i %T %M %R' > summary/queue.txt 2>&1 || true; }"

LOCAL="$BASE/$PROJECT/runs/$NAME"
mkdir -p "$LOCAL"
rm -rf "$LOCAL/summary"
scp -q -r "$HOST:$REMOTE_HOME/$RUNS/$NAME/summary" "$LOCAL/"
tar czf "$LOCAL/summary.tar.gz" -C "$LOCAL" summary

cat "$LOCAL/summary/summary.md"
echo
echo "== в очереди:"
cat "$LOCAL/summary/queue.txt"
echo
echo "== отправить: $LOCAL/summary.tar.gz"
