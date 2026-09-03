#!/bin/bash
# Launch the full JBB-prefill experiment (prereg/jbb_prefill_2026-09-03.md,
# Amendment 1). Ships the frozen HARMFUL openings to an ephemeral pod; results
# stay in the canary'd private package. Generation (8B) then judging (70B 4-bit).
# ~620 gens + ~800 judge calls; ~3-3.5 h ≈ $5-6; MAX_HOURS=6 caps ~$10.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
INSTANCE_FILE="$LOCAL/vast_jbb_experiment.env"

# ---- gate 1: prereg anchored (Amendment 1) --------------------------------
PREREG="$LOCAL/prereg/jbb_prefill_2026-09-03.md"
grep -q '`____________`' "$PREREG" && { echo "REFUSING: prereg anchor placeholder present"; exit 1; }
grep -q "Amendment 1" "$PREREG" || { echo "REFUSING: no Amendment 1 anchor in prereg"; exit 1; }

# ---- gate 2: frozen set present, and it IS the approved 89 ----------------
FROZEN="$LOCAL/data/jbb_prefill/frozen_private/frozen.json"
[ -f "$FROZEN" ] || { echo "FATAL: no frozen.json — run scripts/freeze_jbb_items.py"; exit 1; }
python3 - "$FROZEN" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
assert d["n_behaviors"] == 89, f"expected 89 frozen behaviors, got {d['n_behaviors']}"
b = next(iter(d["behaviors"].values()))
assert {"matched", "mismatched", "neutral"} <= set(b), "frozen item missing a condition"
print(f"  frozen OK: {d['n_behaviors']} behaviors, k={d['k_lengths']}")
PY

echo "=== launch_jbb_experiment gpu=$GPU cloud=$CLOUD ==="
python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-jbb-experiment" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 220 --wait-min 22 || exit 1
read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps (pinned; + bitsandbytes for the 70B 4-bit judge)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     apt-get update -qq; apt-get install -y -qq rsync; which rsync
     pip install -q 'torch==2.5.1'
     pip install -q 'transformers==5.14.1' 'accelerate==1.14.0' 'bitsandbytes==0.50.0' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3'
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,accelerate,bitsandbytes; print(\"deps ok\", torch.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

echo "uploading src + runner + FROZEN items (harmful, to ephemeral pod)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data/jbb_prefill/frozen_private,results/raw}"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/run_jbb_experiment.sh" "root@$HOST:/workspace/st/scripts/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$FROZEN" "root@$HOST:/workspace/st/data/jbb_prefill/frozen_private/"

# ---- gate 3: verified upload ----------------------------------------------
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     [ \$(ls /workspace/st/src/*.py | wc -l) -ge 15 ] || { echo 'FATAL: src incomplete'; exit 1; }
     python3 -c \"import json; d=json.load(open('/workspace/st/data/jbb_prefill/frozen_private/frozen.json')); assert d['n_behaviors']==89; print('  pod frozen OK', d['n_behaviors'])\""

echo "launching experiment runner..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_jbb_experiment.sh && \
     nohup env HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash scripts/run_jbb_experiment.sh > /workspace/pipeline_jbb_exp.log 2>&1 &
     echo \"runner launched, pid=\$!\"" </dev/null

cat <<EOF

=== LAUNCHED: JBB-prefill full experiment ===
  Pod: $INST   (ssh -p $PORT root@$HOST)
  Stage A generation then Stage B 70B judging; ~3-3.5 h, ~\$5-6.
  Done markers: jbb_gen_done, jbb_judge_done, all_stages_done.
Watchdog (start after runner launch):
  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 \\
  WATCHDOG_GRACE_MIN=15 WATCHDOG_STALL_MIN=45 WATCHDOG_MAX_HOURS=6 \\
  WATCHDOG_RSYNC_EXCLUDES="--exclude=*.jsonl.gz" \\
  WATCHDOG_LOG=/tmp/watchdog_jbb_exp.log bash $LOCAL/scripts/watchdog.sh
EOF
