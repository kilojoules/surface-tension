#!/bin/bash
# HTTP-probing watchdog for vLLM inference boxes.
# Different from scripts/watchdog.sh (which uses SSH ps-grep) — vLLM-as-Docker-entrypoint
# boxes don't reliably give SSH proc visibility into the container, but the HTTP endpoint
# is the real liveness signal anyway.
#
# Destroys the instance when ANY of the following is true:
#   - 5 consecutive /v1/models probes fail (vLLM crashed or host died)
#   - elapsed time exceeds MAX_MIN (default 90)
#   - sentinel file exists (caller-driven success signal)
#
# Reads instance info from $LOCAL/vast_current.env:
#   <INSTANCE_ID> <SSH_HOST> <SSH_PORT>
# Reads HTTP endpoint from $LOCAL/vast_vllm.env:
#   ENDPOINT_URL=http://<host>:<ext_port>/v1
#   API_KEY=<api-key>
#
# Usage: bash scripts/vllm_watchdog.sh [sentinel_path]
#   default sentinel: $LOCAL/vast_logs/<INST>/.eval_done

set -u
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_current.env"
ENDPOINT_FILE="$LOCAL/vast_vllm.env"

[ ! -f "$INSTANCE_FILE" ] && echo "missing $INSTANCE_FILE" && exit 1
[ ! -f "$ENDPOINT_FILE" ] && echo "missing $ENDPOINT_FILE" && exit 1
read INST HOST PORT < "$INSTANCE_FILE"
# shellcheck disable=SC1090
source "$ENDPOINT_FILE"

SENTINEL="${1:-$LOCAL/vast_logs/$INST/.eval_done}"
INTERVAL=${VLLM_WATCHDOG_INTERVAL:-60}
MAX_MIN=${VLLM_WATCHDOG_MAX_MIN:-90}
MAX_FAILS=${VLLM_WATCHDOG_MAX_FAILS:-5}
# vLLM cold start (image pull + 4B/31B model download + warmup) can take 10-20 min.
# Don't count probe failures against MAX_FAILS until we've seen at least one 200,
# but DO enforce STARTUP_GRACE_MIN as a hard cap on "never came up at all".
STARTUP_GRACE_MIN=${VLLM_STARTUP_GRACE_MIN:-25}

LOG="$LOCAL/vast_logs/vllm_watchdog_${INST}.log"
mkdir -p "$(dirname "$LOG")" "$(dirname "$SENTINEL")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" ; }
probe() {
    curl -sf -m 10 "${ENDPOINT_URL}/models" \
        -H "Authorization: Bearer ${API_KEY}" >/dev/null 2>&1
}

destroy_and_exit() {
    local reason="$1"
    local rc="${2:-0}"
    log "=== DESTROY: $reason ==="
    echo "n" | vastai destroy instance "$INST" 2>&1 | grep -v "Update" | tee -a "$LOG"
    log "=== DONE ==="
    exit "$rc"
}

fails=0
ever_responded=0
start=$(date +%s)

log "vllm_watchdog start: instance=$INST  endpoint=$ENDPOINT_URL"
log "  interval=${INTERVAL}s max=${MAX_MIN}m max_fails=$MAX_FAILS startup_grace=${STARTUP_GRACE_MIN}m"
log "  sentinel: $SENTINEL"

while true; do
    now=$(date +%s)
    elapsed_min=$(( (now - start) / 60 ))

    # 1) hard cap
    if [ "$elapsed_min" -ge "$MAX_MIN" ]; then
        destroy_and_exit "max runtime ${MAX_MIN}m" 1
    fi

    # 2) caller-driven sentinel (success signal from sweep.py / launch script)
    if [ -f "$SENTINEL" ]; then
        destroy_and_exit "sentinel $SENTINEL" 0
    fi

    # 3) HTTP liveness
    if probe; then
        if [ "$ever_responded" = "0" ]; then
            log "  [up] vLLM came up after ${elapsed_min}m (entering normal-watch mode)"
            ever_responded=1
        else
            log "  [ok] vLLM responsive (${elapsed_min}m elapsed)"
        fi
        fails=0
    else
        # During startup grace, probe failures are normal (image pull, model download, warmup).
        # Only enforce MAX_FAILS once vLLM has responded at least once.
        if [ "$ever_responded" = "0" ]; then
            log "  [warmup] probe still failing (${elapsed_min}m / ${STARTUP_GRACE_MIN}m grace)"
            if [ "$elapsed_min" -ge "$STARTUP_GRACE_MIN" ]; then
                destroy_and_exit "vLLM never came up within ${STARTUP_GRACE_MIN}m" 1
            fi
        else
            fails=$((fails + 1))
            log "  [fail] probe $fails/$MAX_FAILS (${elapsed_min}m elapsed)"
            if [ "$fails" -ge "$MAX_FAILS" ]; then
                destroy_and_exit "vLLM unresponsive ($fails consecutive probe failures)" 1
            fi
        fi
    fi

    sleep "$INTERVAL"
done
