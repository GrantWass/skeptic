#!/usr/bin/env bash
# collect_loop.sh — run collect_prices.py overnight with auto-restart on crash.
#
# Usage:
#   bash scripts/collect_loop.sh
#   bash scripts/collect_loop.sh --interval 1 --assets BTC ETH SOL
#
# Logs to: data/collect_loop.log
# Stop cleanly with Ctrl+C (or kill the PID shown at startup).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$ROOT_DIR/data/collect_loop.log"
RESTART_DELAY=5
MAX_RESTARTS=100   # safety cap — stops after 100 crashes (implies a hard bug, not transient)

mkdir -p "$ROOT_DIR/data"

# Reset log file on each run
> "$LOG_FILE"

# Trap Ctrl+C and SIGTERM so the loop exits cleanly
_stop=0
trap '_stop=1' INT TERM

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "=== collect_loop starting (PID $$) ==="
log "Log file: $LOG_FILE"
log "Args passed to collect_prices.py: $*"

# Activate virtual environment
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE"
    log "Activated venv: $VENV_ACTIVATE"
else
    log "WARNING: venv not found at $VENV_ACTIVATE — using system Python"
fi

restarts=0

while [[ $_stop -eq 0 && $restarts -lt $MAX_RESTARTS ]]; do
    log "Starting collect_prices.py (attempt $((restarts + 1)))"

    # caffeinate -s: prevent macOS sleep while this runs
    # tee -a: echo stdout/stderr to terminal AND append to log file
    if caffeinate -s python "$SCRIPT_DIR/collect_prices.py" "$@" 2>&1 | tee -a "$LOG_FILE"; then
        # Clean exit (Ctrl+C inside the script exits 0 via KeyboardInterrupt handler)
        log "collect_prices.py exited cleanly."
        break
    else
        exit_code=$?
        restarts=$((restarts + 1))
        if [[ $_stop -eq 1 ]]; then
            log "Shutdown requested — not restarting."
            break
        fi
        log "collect_prices.py crashed (exit $exit_code). Restart $restarts/$MAX_RESTARTS in ${RESTART_DELAY}s..."
        sleep "$RESTART_DELAY"
    fi
done

if [[ $restarts -ge $MAX_RESTARTS ]]; then
    log "ERROR: hit max restarts ($MAX_RESTARTS). Giving up — check $LOG_FILE for details."
    exit 1
fi

log "=== collect_loop stopped ==="
