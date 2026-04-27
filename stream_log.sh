#!/bin/bash
# Streams logs from a vast.ai instance to local disk every 30s.
# Run this in a separate terminal alongside an active experiment so logs
# survive a host crash.
#
# Usage: stream_log.sh <ssh_host> <ssh_port>
# e.g.   stream_log.sh ssh5.vast.ai 16464
#
# Output: ./vast_logs/<host>_<port>/{run,vllm,vllm_dpo,dpo_train,heartbeat}.log
#
# Stop with Ctrl-C — files persist.

set -uo pipefail

SSH_HOST="${1:?Usage: $0 <ssh_host> <ssh_port>}"
SSH_PORT="${2:?Usage: $0 <ssh_host> <ssh_port>}"

OUT_DIR="./vast_logs/${SSH_HOST//./_}_${SSH_PORT}"
mkdir -p "$OUT_DIR"
echo "[stream_log] streaming to $OUT_DIR/ (Ctrl-C to stop)"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o ServerAliveInterval=10"

while true; do
  ts=$(date -u +%H:%M:%S)
  for f in run.log vllm.log vllm_dpo.log dpo_train.log heartbeat.txt; do
    # rsync each file independently so a missing file doesn't break the others
    rsync -azq $SSH_OPTS -e "ssh -p $SSH_PORT" \
      "root@$SSH_HOST:/workspace/$f" "$OUT_DIR/$f" 2>/dev/null || true
  done
  # Print a compact heartbeat to the terminal so the user can see we're alive
  if [ -f "$OUT_DIR/heartbeat.txt" ]; then
    hb=$(cat "$OUT_DIR/heartbeat.txt" 2>/dev/null | tail -1)
    echo "[$ts] hb=$hb run.log=$(wc -l < $OUT_DIR/run.log 2>/dev/null || echo 0)L"
  else
    echo "[$ts] (no heartbeat yet) run.log=$(wc -l < $OUT_DIR/run.log 2>/dev/null || echo 0)L"
  fi
  sleep 30
done
