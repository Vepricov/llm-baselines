#!/bin/bash
# sber/collect.sh - on the work machine: build the sweep summary on the cluster and copy it back.
#
#   curl -sL https://raw.githubusercontent.com/Vepricov/llm-baselines/tga/sber/collect.sh | bash
#   curl -sL .../collect.sh | bash -s -- tga_sweep_25-10-2026_21-40-00
#
# Without an argument takes the most recent sweep. Prints summary.md and leaves
# $HOME/gigalearn/runs/<sweep>/summary/ plus <sweep>_summary.tar.gz to send on.
set -euo pipefail

[ -f "$HOME/gigalearn/config.env" ] && . "$HOME/gigalearn/config.env"
HOST="${GIGALEARN_HOST:?задай GIGALEARN_HOST в \$HOME/gigalearn/config.env}"
BASE="${GIGALEARN_HOME:-$HOME/gigalearn}"
NAME="${1:-}"

REMOTE_HOME="$(ssh -n "$HOST" 'printf %s "$HOME"')"
if [ -z "$NAME" ]; then
    NAME="$(ssh -n "$HOST" 'cd $HOME/gigalearn/runs && for d in $(ls -td */); do [ -f "$d/sweep.txt" ] && { echo "${d%/}"; break; }; done')"
    [ -n "$NAME" ] || { echo "error: на кластере нет ни одного свипа"; exit 2; }
fi
# a cut or shortened name is completed by prefix, if that is unambiguous
NAME="$(ssh -n "$HOST" "cd \$HOME/gigalearn/runs && m=\$(ls -d '$NAME'* 2>/dev/null); \
    [ \$(echo \"\$m\" | grep -c .) -eq 1 ] && echo \"\$m\" || echo '$NAME'")"
NAME="${NAME%/}"
echo "== свип: $NAME"

ssh -n "$HOST" "cd \$HOME/gigalearn/runs/$NAME && python3 sber/sweep_summary.py . > /dev/null && \
    { squeue -h -n '$NAME' -o '%i %T %M %R' > summary/queue.txt 2>&1 || true; }"

mkdir -p "$BASE/runs/$NAME"
rm -rf "$BASE/runs/$NAME/summary"
scp -q -r "$HOST:$REMOTE_HOME/gigalearn/runs/$NAME/summary" "$BASE/runs/$NAME/"
tar czf "$BASE/runs/${NAME}_summary.tar.gz" -C "$BASE/runs/$NAME" summary

cat "$BASE/runs/$NAME/summary/summary.md"
echo
echo "== в очереди:"
cat "$BASE/runs/$NAME/summary/queue.txt"
echo
echo "== отправить: $BASE/runs/${NAME}_summary.tar.gz"
