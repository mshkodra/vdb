#!/usr/bin/env bash
# Resets vdb_server's persisted state: stops the server if it's running on
# `port`, then deletes `data_dir` — the WAL (<data_dir>/wal/) and snapshot
# (<data_dir>/snapshot) DurableVDB keeps there. The next
# `vdb_server <data_dir> <port>` start recovers nothing and comes back empty,
# since data_dir's contents are the *only* thing that determines what state
# comes back on restart (see DurableVDB's own recover_()).
#
# Usage: server/reset.sh [-f|--force] <data_dir> [port]
#   -f, --force   skip the "type yes to confirm" prompt (for repeated use while
#                 iterating on a corpus, not the default — this is a permanent,
#                 irreversible delete)
set -euo pipefail

FORCE=0
while [[ "${1:-}" == -* ]]; do
    case "$1" in
        -f|--force) FORCE=1 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ $# -lt 1 ]]; then
    echo "usage: $0 [-f|--force] <data_dir> [port]" >&2
    exit 1
fi

DATA_DIR="$1"
PORT="${2:-50051}"

# Stop the running server on this port, if any — an empty match here isn't an
# error, the server might already be stopped.
PID="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$PID" ]]; then
    echo "stopping vdb_server (pid $PID) on port $PORT..."
    kill "$PID"
    # Wait for the port to actually free up rather than racing the delete below
    # against a process that's still mid-shutdown.
    for _ in $(seq 1 20); do
        lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 0.2
    done
else
    echo "no server currently listening on port $PORT (nothing to stop)"
fi

if [[ ! -e "$DATA_DIR" ]]; then
    echo "$DATA_DIR does not exist — nothing to reset"
    exit 0
fi

if [[ "$FORCE" -ne 1 ]]; then
    echo "about to permanently delete $DATA_DIR (WAL + snapshot — every row inserted so far)"
    read -r -p "type 'yes' to confirm: " CONFIRM
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "aborted, nothing deleted"
        exit 1
    fi
fi

rm -rf "$DATA_DIR"
echo "reset complete. restart with:"
echo "  ./build/vdb_server $DATA_DIR $PORT"
