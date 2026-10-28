# gigalearn/prepare.sh - sourced by gigalearn node.sh once per array task, in code/.
# shakespeare is tokenized once per sweep in a private temp dir and moved into place with
# one rename, then never rewritten: a requeued (preempted) task must not touch files that
# another attempt of a line is still reading.
export TIKTOKEN_CACHE_DIR="$GIGA_BASE/tiktoken_cache"
export LAB_RUN_OFFLINE=1
if grep -q -- '--dataset shakespeare-char' "$TASK_DIR/sweep.txt"; then
    export DATA_DIR="$TASK_DIR/data/shared"
    if [ ! -f "$DATA_DIR/shakespeare/test.npy" ]; then
        TMP_DATA="$TASK_DIR/data/tmp.${SLURM_JOB_ID:-0}.${SLURM_ARRAY_TASK_ID:-0}.$$"
        mkdir -p "$TMP_DATA/shakespeare"
        cp gigalearn/data/shakespeare/raw.txt "$TMP_DATA/shakespeare/raw.txt"
        "$PY" -c "import sys; sys.path.insert(0, 'src'); from data.shakespeare import get_shakespeare_data; get_shakespeare_data('$TMP_DATA')"
        # os.rename refuses an existing non-empty dir: the first finished task wins
        "$PY" -c "import os, sys; os.rename(sys.argv[1], sys.argv[2])" "$TMP_DATA" "$DATA_DIR" 2>/dev/null \
            || rm -rf "$TMP_DATA"
    fi
else
    export DATA_DIR="$GIGA_BASE/data"
fi
