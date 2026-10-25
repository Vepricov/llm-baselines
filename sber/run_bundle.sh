#!/bin/bash
# run_bundle.sh - unpack a task bundle on GigaLearn and submit it with sbatch.
#
# Run it ON THE CLUSTER (front node), inside the unpacked bundle dir or with
# a tarball path:
#     bash run_bundle.sh                 # uses cwd (unpacked bundle)
#     bash run_bundle.sh bundle.tar.gz   # unpacks, then submits
#
# Options:
#     --watch        tail the log until the job ends, then write report_<jobid>.txt
#     --no-watch     submit and print the job id only (default)
#     --launcher X   sbatch script name (default: train_slurm.sh)
#
# The #SBATCH --output/--error directories are created before sbatch,
# because Slurm requires the log dir to exist when the job starts.

set -euo pipefail

# Чистим прошлые артефакты, чтобы не притащить устаревшие metrics/plot.
rm -f "$HOME/report_latest.txt" "$HOME/metrics_latest.csv" "$HOME/training_curve_latest.png" "$HOME/training_curve_latest.svg"


WATCH=0
LAUNCHER="train_slurm.sh"
TARBALL=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --watch)   WATCH=1; shift ;;
        --no-watch) WATCH=0; shift ;;
        --launcher) LAUNCHER="$2"; shift 2 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) TARBALL="$1"; shift ;;
    esac
done

if [ -n "$TARBALL" ]; then
    tar xzf "$TARBALL"
    DIR="$(tar tzf "$TARBALL" 2>/dev/null | head -1 | cut -d/ -f1)"
    cd "$DIR"
fi

[ -f "$LAUNCHER" ] || { echo "error: $LAUNCHER not found in $(pwd)" >&2; exit 1; }

# Create the log directories from #SBATCH --output/--error before submitting.
for key in output error; do
    path="$(grep -m1 "^#SBATCH --$key=" "$LAUNCHER" | cut -d= -f2-)"
    if [ -n "$path" ]; then
        dir="$(dirname "$path")"
        if [ "$dir" != "." ]; then
            mkdir -p "$dir"
            echo "log dir ready: $dir"
        fi
    fi
done

chmod +x ./*.sh 2>/dev/null || true

echo "submitting: $LAUNCHER from $(pwd)"
OUT="$(sbatch "$LAUNCHER")"
echo "$OUT"
JOBID="$(echo "$OUT" | awk '{print $NF}')"

if [ "$WATCH" -eq 0 ]; then
    echo "job id: $JOBID"
    exit 0
fi

LOG="$(grep -m1 "^#SBATCH --output=" "$LAUNCHER" | cut -d= -f2- | sed "s/%j/$JOBID/g")"
echo "watching job $JOBID, log: $LOG"

while squeue -j "$JOBID" -h 2>/dev/null | grep -q .; do
    sleep 15
done

echo "job $JOBID finished"
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    tail -n 120 "$LOG"
else
    echo "(log file not found at $LOG)"
fi

REPORT="report_${JOBID}.txt"
{
    echo "job id: $JOBID"
    echo "finished: $(date '+%F %T')"
    echo ""
    echo "===== sacct ====="
    sacct -j "$JOBID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList 2>&1 | head -20
    echo ""
    echo "===== log tail ====="
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        tail -n 150 "$LOG"
    else
        echo "(log not found)"
    fi
    echo ""
    echo "===== err tail ====="
    ERR="$(grep -m1 "^#SBATCH --error=" "$LAUNCHER" | cut -d= -f2- | sed "s/%j/$JOBID/g")"
    if [ -n "$ERR" ] && [ -f "$ERR" ]; then
        tail -n 60 "$ERR"
    else
        echo "(no stderr file)"
    fi
} > "$REPORT"

echo "report written: $REPORT"

# Страховка: если график в контейнере не получился, строим его здесь же,
# на фронт-ноде (python3 есть всегда, matplotlib не нужен - пишем SVG).
if [ -n "$LOG" ] && [ -f "$LOG" ] && [ -f plot_metrics.py ]; then
    python3 plot_metrics.py "$LOG" . && echo "metrics+plot built on front node"
fi

# Кладём в домашку под фиксированными именами, чтобы runner.sh просто их
# забирал (без glob'ов и ошибок "not a directory").
cp "$REPORT" "$HOME/report_latest.txt"
echo "report: $HOME/report_latest.txt"

if [ -f metrics.csv ]; then
    cp metrics.csv "$HOME/metrics_latest.csv"
    echo "metrics: $HOME/metrics_latest.csv"
fi
if [ -f training_curve.png ]; then
    cp training_curve.png "$HOME/training_curve_latest.png"
    echo "plot: $HOME/training_curve_latest.png"
fi
if [ -f training_curve.svg ]; then
    cp training_curve.svg "$HOME/training_curve_latest.svg"
    echo "plot: $HOME/training_curve_latest.svg"
fi

echo "take ONLY these files back through the approved channel."
