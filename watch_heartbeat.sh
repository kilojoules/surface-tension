#!/bin/bash
# Watches the streamed heartbeat file for staleness. Emits one line per check
# (so Monitor can pick it up as events). Beeps the terminal if heartbeat is
# stale — meaning the box has gone unresponsive even if SSH still answers.
#
# Run alongside stream_log.sh. The heartbeat itself is written by the orchestrator
# on the box (touched every 60s).
#
# Usage: watch_heartbeat.sh <ssh_host> <ssh_port>
#   STALE_THRESHOLD_S — seconds without heartbeat update before alerting (default 300)

set -uo pipefail

SSH_HOST="${1:?Usage: $0 <ssh_host> <ssh_port>}"
SSH_PORT="${2:?Usage: $0 <ssh_host> <ssh_port>}"
STALE=${STALE_THRESHOLD_S:-300}

HB_FILE="./vast_logs/${SSH_HOST//./_}_${SSH_PORT}/heartbeat.txt"

while true; do
  ts=$(date -u +%H:%M:%S)
  if [ ! -f "$HB_FILE" ]; then
    echo "[$ts] WAITING — no heartbeat file yet at $HB_FILE"
  else
    # Heartbeat file is "epoch_ms\tphase" — we just check the mtime locally
    # since stream_log.sh rsyncs it every 30s.
    age=$(( $(date +%s) - $(stat -f %m "$HB_FILE" 2>/dev/null || stat -c %Y "$HB_FILE") ))
    last=$(tail -1 "$HB_FILE" 2>/dev/null)
    if [ "$age" -gt "$STALE" ]; then
      echo "[$ts] STALE — heartbeat ${age}s old (>$STALE), last: $last"
      printf '\a'  # terminal bell
    else
      echo "[$ts] OK — heartbeat ${age}s old, last: $last"
    fi
  fi
  sleep 60
done
