#!/bin/bash
# Launch the JBB-prefill baseline check ($2 step-0 of
# prereg/jbb_prefill_2026-09-03.md). Ships only harmless inputs (public JBB
# behaviors + a benign neutral prefix). ~82 short generations on the 8B
# victim: model load + gen ~15-20 min ≈ $0.40-0.60; MAX_HOURS=2 caps at $3.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
INSTANCE_FILE="$LOCAL/vast_jbb_baseline.env"

# ---- gate 1: prereg externally timestamped -------------------------------
PREREG="$LOCAL/prereg/jbb_prefill_2026-09-03.md"
if grep -q '`____________`' "$PREREG"; then
    echo "REFUSING TO LAUNCH: $PREREG still has the external-timestamp"
    echo "placeholder. Anchor it (public gist) and record the link first."
    exit 1
fi

# ---- gate 2: baseline items present and harmless --------------------------
ITEMS="$LOCAL/data/jbb_prefill/baseline_items_highharm.json"
[ -f "$ITEMS" ] || { echo "FATAL: no baseline_items.json — run scripts/harvest_jbb_prefixes.py"; exit 1; }
python3 - "$ITEMS" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
bad = [g for g, m in d["behaviors"].items() if set(m) - {"category", "neutral_prefix"}]
assert not bad, f"baseline items carry unexpected (possibly harmful) fields: {bad[:3]}"
print(f"  baseline items OK: {d['n_behaviors']} behaviors, benign prefixes only")
PY

echo "=== launch_jbb_baseline gpu=$GPU cloud=$CLOUD ==="
python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-jbb-baseline" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 120 --wait-min 22 || exit 1
read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps (pinned to the 2026-08-11 propensity freeze)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     apt-get update -qq; apt-get install -y -qq rsync; which rsync
     pip install -q 'torch==2.5.1'
     pip install -q 'transformers==5.14.1' 'accelerate==1.14.0' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3'
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,accelerate; print(\"deps ok\", torch.__version__, transformers.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

echo "uploading src + runner + harmless baseline items..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data/jbb_prefill,results/raw}"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/run_jbb_baseline.sh" "root@$HOST:/workspace/st/scripts/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$ITEMS" "root@$HOST:/workspace/st/data/jbb_prefill/"

# ---- gate 3: verified upload, and re-assert no harmful fields on the pod ---
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     n_src=\$(ls /workspace/st/src/*.py | wc -l)
     [ \"\$n_src\" -ge 15 ] || { echo 'FATAL: src upload incomplete'; exit 1; }
     python3 -c \"
import json
d=json.load(open('/workspace/st/data/jbb_prefill/baseline_items_highharm.json'))
bad=[g for g,m in d['behaviors'].items() if set(m)-{'category','neutral_prefix'}]
assert not bad, 'harmful fields on pod'
print('  pod items OK:', d['n_behaviors'])\""

echo "launching baseline runner..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_jbb_baseline.sh && \
     nohup env HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash scripts/run_jbb_baseline.sh > /workspace/pipeline_jbb.log 2>&1 &
     echo \"runner launched, pid=\$!\"" </dev/null

cat <<EOF

=== LAUNCHED: JBB-prefill baseline check ===
  Pod: $INST   (ssh -p $PORT root@$HOST)
  ~82 short generations, ~15-20 min, ~\$0.40-0.60.
  Reads: results/raw/jbb_baseline_summary.json for the PASS/FAIL verdict.
Watchdog:
  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 \\
  WATCHDOG_GRACE_MIN=10 WATCHDOG_STALL_MIN=30 WATCHDOG_MAX_HOURS=2 \\
  WATCHDOG_RSYNC_EXCLUDES="--exclude=*.jsonl.gz" \\
  WATCHDOG_LOG=/tmp/watchdog_jbb_baseline.log bash $LOCAL/scripts/watchdog.sh
EOF
