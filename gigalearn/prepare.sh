# gigalearn/prepare.sh - sourced by gigalearn node.sh once per array task, in code/.
# shakespeare is tokenized per array task, so parallel tasks never write the same file.
export TIKTOKEN_CACHE_DIR="$GIGA_BASE/tiktoken_cache"
export LAB_RUN_OFFLINE=1
if grep -q -- '--dataset shakespeare-char' "$TASK_DIR/sweep.txt"; then
    export DATA_DIR="$TASK_DIR/data_task${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$DATA_DIR/shakespeare"
    cp gigalearn/data/shakespeare/raw.txt "$DATA_DIR/shakespeare/raw.txt"
    "$PY" -c "import sys; sys.path.insert(0, 'src'); from data.shakespeare import get_shakespeare_data; get_shakespeare_data('$DATA_DIR')"
else
    export DATA_DIR="$GIGA_BASE/data"
fi
