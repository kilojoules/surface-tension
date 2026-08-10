#!/bin/bash
# Watchdog — adapted from turnstile/scripts/watchdog.sh.
# Polls instance, syncs results, auto-destroys on completion or stall.
#
# Reads instance info from vast_current.env (written by launch_dpo.sh):
#   <INSTANCE_ID> <SSH_HOST> <SSH_PORT>
#
# Tunables (env):
#   WATCHDOG_INTERVAL  default 180  (poll every 3 min)
#   WATCHDOG_GRACE_MIN default 10   (idle this long after process exit -> destroy)
#   WATCHDOG_STALL_MIN default 30   (log frozen this long with proc still alive -> destroy)
#   WATCHDOG_MAX_HOURS default 12   (hard cap)

set -u

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="${WATCHDOG_INSTANCE_FILE:-$LOCAL/vast_current.env}"

[ ! -f "$INSTANCE_FILE" ] && echo "missing $INSTANCE_FILE" && exit 1
read INST HOST PORT < "$INSTANCE_FILE"

INTERVAL=${WATCHDOG_INTERVAL:-180}
GRACE_MIN=${WATCHDOG_GRACE_MIN:-10}
STALL_MIN=${WATCHDOG_STALL_MIN:-30}
MAX_HOURS=${WATCHDOG_MAX_HOURS:-12}
MAX_SSH_FAILS=${WATCHDOG_MAX_SSH_FAILS:-20}

LOG="$LOCAL/vast_logs/watchdog_${INST}.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" ; }
do_ssh() { ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "root@$HOST" "$@" 2>/dev/null ; }

sync_all() {
    local out="$LOCAL/vast_logs/${INST}"
    mkdir -p "$out"
    # Build excludes — adapter dirs are pushed to Hub, no need locally; user can override via WATCHDOG_RSYNC_EXCLUDES
    local excludes="--exclude='*.pyc' --exclude='__pycache__' --exclude='outputs/*/checkpoint-*' --exclude='outputs/*/final_adapter' --exclude='outputs/*/best_val_adapter'"
    [ -n "${WATCHDOG_RSYNC_EXCLUDES:-}" ] && excludes="$excludes ${WATCHDOG_RSYNC_EXCLUDES}"
    eval rsync -az -e \"ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30\" \
        \"root@$HOST:/workspace/\" \"$out/\" $excludes 2>/dev/null \
        || log "  [sync] failed"
}

destroy_and_exit() {
    local reason="$1"
    log "=== DESTROY: $reason ==="
    sync_all
    sync_all
    case "${PROVIDER:-vast}" in
        runpod)
            python3 "$LOCAL/scripts/runpod_kill.py" "$INST" 2>&1 | tee -a "$LOG"
            ;;
        *)
            echo "n" | vastai destroy instance "$INST" 2>&1 | grep -v Update | tee -a "$LOG"
            ;;
    esac
    log "=== DONE ==="
    exit "${2:-0}"
}

ssh_fails=0
idle_since=""
last_log_mtime=""
log_stall_since=""
start=$(date +%s)

log "watchdog start: instance=$INST host=$HOST:$PORT"
log "  interval=${INTERVAL}s grace=${GRACE_MIN}m stall=${STALL_MIN}m max=${MAX_HOURS}h"

while true; do
    now=$(date +%s)
    elapsed_h=$(( (now - start) / 3600 ))
    [ "$elapsed_h" -ge "$MAX_HOURS" ] && destroy_and_exit "max runtime ${MAX_HOURS}h"

    if ! do_ssh "echo ok" >/dev/null; then
        ssh_fails=$((ssh_fails + 1))
        log "  [ssh] fail $ssh_fails/$MAX_SSH_FAILS"
        if [ "$ssh_fails" -ge "$MAX_SSH_FAILS" ]; then
            log "  [ssh] dead — final sync attempt"
            sync_all || true
            destroy_and_exit "ssh unreachable" 1
        fi
        sleep "$INTERVAL"; continue
    fi
    ssh_fails=0

    # Done sentinel
    if do_ssh "[ -f /workspace/all_done ]"; then
        destroy_and_exit "all_done sentinel"
    fi

    # Process alive?
    procs=$(do_ssh "ps aux | grep -E 'quadrant_pod_runner|dpo_train|sft_train|kto_train|grpo_train|build_sft|build_expanded|build_rationale|build_dpo_pairs|build_dpo_dataset|build_kto_dataset|build_onpolicy_dataset|sweep_local|aggregate.py|push_adapter|recheck_eval|recheck_threemetric|propensity_scorer|propensity_generate|propensity_interrupt' | grep -v grep | wc -l" | tr -d ' ')
    procs=${procs:-0}

    # Log activity
    info=$(do_ssh "ls -t /workspace/*.log 2>/dev/null | head -1 | xargs -r stat -c '%Y'")
    cur_mtime=$(echo "$info" | awk '{print $1}')
    if [ -n "$cur_mtime" ] && [ "$cur_mtime" = "$last_log_mtime" ]; then
        [ -z "$log_stall_since" ] && log_stall_since=$now
        stall_min=$(( (now - log_stall_since) / 60 ))
        if [ "$stall_min" -ge "$STALL_MIN" ] && [ "$procs" -gt 0 ]; then
            log "  [stall] log frozen ${stall_min}m with $procs procs — killing"
            destroy_and_exit "log stall ${stall_min}m"
        fi
    else
        log_stall_since=""
        last_log_mtime=$cur_mtime
    fi

    if [ "$procs" = "0" ]; then
        if [ -z "$idle_since" ]; then
            idle_since=$now
            log "  [idle] no procs — grace ${GRACE_MIN}m starts"
        fi
        idle_min=$(( (now - idle_since) / 60 ))
        log "  [idle] ${idle_min}m (grace=${GRACE_MIN}m)"
        if [ "$idle_min" -ge "$GRACE_MIN" ]; then
            destroy_and_exit "process exited, idle ${idle_min}m"
        fi
    else
        idle_since=""
        log "  [ok] $procs procs running, ${elapsed_h}h elapsed"
    fi

    sync_all
    sleep "$INTERVAL"
done
