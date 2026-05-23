#!/bin/bash
# Babysitter for the -final clean-17 eval. Catches the CUDA-crash failure mode that wasted
# the previous run: once CUDA dies, every gen errors instantly with "illegal memory access"
# and the pipeline "completes" the eval in minutes with 135/136 gen_errors. The main
# watchdog can't catch this (it sees all_done get touched). We detect by pulling the CSV
# periodically and checking the gen_error rate.
set -u
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
[ ! -s "$LOCAL/vast_eval_clean17_final.env" ] && { echo "no env file"; exit 1; }
read INST HOST PORT < "$LOCAL/vast_eval_clean17_final.env"
LOG="$LOCAL/vast_logs/babysit_${INST}.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

POLL=${POLL:-180}
HARD_CAP_HOURS=${HARD_CAP_HOURS:-4}
ERR_RATE_THRESHOLD=${ERR_RATE_THRESHOLD:-0.5}
MIN_ROWS_BEFORE_KILL=${MIN_ROWS_BEFORE_KILL:-10}
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"

log "babysit_clean17_final start: instance=$INST host=$HOST:$PORT poll=${POLL}s"
log "  intervention rules:"
log "    - kill if >${ERR_RATE_THRESHOLD} gen_error rate after ${MIN_ROWS_BEFORE_KILL}+ rows (CUDA-crash detection)"
log "    - hard cap: kill if still running after ${HARD_CAP_HOURS}h"

start=$(date +%s)
ssh_fails=0
while true; do
    now=$(date +%s)
    elapsed_h=$(( (now - start) / 3600 ))

    # Hard time cap
    if [ "$elapsed_h" -ge "$HARD_CAP_HOURS" ]; then
        log "ABORT: hard cap reached (${HARD_CAP_HOURS}h elapsed)"
        echo "n" | vastai destroy instance "$INST" 2>&1 | tee -a "$LOG"
        break
    fi

    # Quick SSH liveness check
    if ! $SSH "root@$HOST" "echo ok" >/dev/null 2>&1; then
        ssh_fails=$((ssh_fails + 1))
        log "  SSH fail $ssh_fails (waiting...)"
        if [ "$ssh_fails" -ge 8 ]; then
            log "ABORT: SSH dead 8x — instance probably already gone or wedged; forcing destroy"
            echo "n" | vastai destroy instance "$INST" 2>&1 | tee -a "$LOG"
            break
        fi
        sleep 60; continue
    fi
    ssh_fails=0

    # Did the pipeline finish on its own?
    if $SSH "root@$HOST" "[ -f /workspace/all_done ]" 2>/dev/null; then
        log "all_done sentinel present — pipeline finished naturally, watchdog will destroy"
        break
    fi

    # Pull the CSV row counts + gen_error rate
    STATS=$($SSH "root@$HOST" '
      CSV=/workspace/st/results/raw/eval_clean17_final.csv
      if [ -f "$CSV" ]; then
        TOTAL=$(($(wc -l < "$CSV") - 1))
        ERRS=$(awk -F, "NR>1 && \$9 != \"\" {n++} END{print n+0}" "$CSV")
        CUDA_ERRS=$(grep -c "CUDA error" "$CSV" 2>/dev/null || echo 0)
        echo "$TOTAL $ERRS $CUDA_ERRS"
      else
        echo "0 0 0"
      fi
    ' 2>/dev/null)
    TOTAL=$(echo "$STATS" | awk "{print \$1}")
    ERRS=$(echo "$STATS" | awk "{print \$2}")
    CUDA_ERRS=$(echo "$STATS" | awk "{print \$3}")
    log "  rows=${TOTAL:-0}  gen_errors=${ERRS:-0}  cuda_errors=${CUDA_ERRS:-0}  elapsed=${elapsed_h}h"

    # CUDA-crash detection
    if [ "${TOTAL:-0}" -ge "$MIN_ROWS_BEFORE_KILL" ]; then
        # bash float comparison via python
        IS_BAD=$(python3 -c "print(1 if $ERRS / $TOTAL > $ERR_RATE_THRESHOLD else 0)")
        if [ "$IS_BAD" = "1" ]; then
            log "ABORT: gen_error rate $ERRS/$TOTAL > $ERR_RATE_THRESHOLD (CUDA crash or similar fatal failure)"
            log "  cuda_error count: $CUDA_ERRS — if >0, GPU is dead"
            echo "n" | vastai destroy instance "$INST" 2>&1 | tee -a "$LOG"
            break
        fi
    fi

    sleep "$POLL"
done
log "babysit_clean17_final exit"
