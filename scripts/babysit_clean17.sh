#!/bin/bash
# Babysitter for the clean-17 eval. Waits for done_clean17_bestval (= eval #1 finished),
# then kills the pipeline (skip eval #2 -final, which we don't need given the deflationary
# clean-set signal), runs recheck_eval.py on the bestval CSV, touches /workspace/all_done →
# main watchdog destroys the pod with a final sync. Robust against SSH choking.
set -u
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
[ ! -f "$LOCAL/vast_eval_clean17.env" ] && { echo "no env file"; exit 1; }
read INST HOST PORT < "$LOCAL/vast_eval_clean17.env"
LOG="$LOCAL/vast_logs/babysit_${INST}.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

POLL=${POLL:-90}
SSH_FAIL_LIMIT=${SSH_FAIL_LIMIT:-8}
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20"

log "babysit_clean17 start: instance=$INST host=$HOST:$PORT (poll=${POLL}s)"
log "  trigger: /workspace/done_clean17_bestval appears (eval #1 finished)"
log "  action:  pkill pipeline → recheck → touch all_done → main watchdog destroys"

ssh_fails=0
while true; do
    # Cheap path: did the watchdog already sync the sentinel locally?
    if [ -f "$LOCAL/vast_logs/$INST/done_clean17_bestval" ]; then
        log "done_clean17_bestval seen in synced dir — confirming via SSH..."
    fi

    # Authoritative check on the pod
    if $SSH "root@$HOST" "[ -f /workspace/done_clean17_bestval ]" 2>/dev/null; then
        log "CONFIRMED on pod. Killing pipeline + recheck + touching all_done..."
        $SSH "root@$HOST" '
            pkill -9 -f "sweep_local.py" 2>/dev/null || true
            pkill -9 -f "bash -uo pipefail" 2>/dev/null || true
            sleep 3
            cd /workspace/st/src
            echo "[babysit-remote] running recheck_eval on eval_clean17_bestval.csv..."
            python -u recheck_eval.py ../results/raw/eval_clean17_bestval.csv 2>&1 | tee /workspace/recheck_summary.txt || true
            cp ../results/raw/recheck_summary.json /workspace/ 2>/dev/null || true
            touch /workspace/all_done
            echo "[babysit-remote] all_done touched at $(date)"
        ' 2>&1 | tee -a "$LOG"
        log "done; main watchdog will sync + destroy within ~2 min"
        break
    fi

    # SSH probe failed — also check if the pod might just be gone
    if ! $SSH "root@$HOST" "echo ok" >/dev/null 2>&1; then
        ssh_fails=$((ssh_fails + 1))
        log "  SSH fail $ssh_fails/$SSH_FAIL_LIMIT"
        if [ $ssh_fails -ge $SSH_FAIL_LIMIT ]; then
            log "  SSH dead. Pod may already be destroyed or wedged. Forcing destroy via vastai..."
            echo "n" | vastai destroy instance "$INST" 2>&1 | tee -a "$LOG"
            log "  exiting"
            break
        fi
    else
        ssh_fails=0
    fi

    sleep "$POLL"
done
log "babysit_clean17 exit"
