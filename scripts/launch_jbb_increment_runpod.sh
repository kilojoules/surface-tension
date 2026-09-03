#!/bin/bash
# Launch the N=10,15 increment. Seeds the prior N=5,20 gens+judged (from
# SEED_DIR), uploads the 4-length frozen set, and runs gen->judge so only the
# new cells are produced. ~534 new gens + ~714 judge calls; ~2 h ≈ $3-4;
# MAX_HOURS=4. SEED_DIR must hold the COMPLETE N=5,20 run.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
INSTANCE_FILE="$LOCAL/vast_jbb_increment.env"
SEED_DIR="${SEED_DIR:?set SEED_DIR to the complete N=5,20 results/raw dir}"

# ---- gates ----------------------------------------------------------------
PREREG="$LOCAL/prereg/jbb_prefill_2026-09-03.md"
grep -q "Qi et al. 2023 GPT-4-Judge" "$PREREG" || { echo "REFUSING: prereg not on Qi rubric"; exit 1; }
FROZEN="$LOCAL/data/jbb_prefill/frozen_private/frozen.json"
python3 - "$FROZEN" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
assert d["k_lengths"] == [5, 10, 15, 20], f"frozen not 4-length: {d['k_lengths']}"
assert d["n_behaviors"] == 89
print(f"  frozen OK: 4 lengths, {d['n_behaviors']} behaviors")
PY
for f in jbb_full.jsonl jbb_full_judged.jsonl jbb_openings.json; do
    [ -f "$SEED_DIR/$f" ] || { echo "FATAL: seed missing $SEED_DIR/$f"; exit 1; }
done
sg=$(wc -l < "$SEED_DIR/jbb_full.jsonl"); sj=$(wc -l < "$SEED_DIR/jbb_full_judged.jsonl")
echo "  seed: $sg gens, $sj judged (expect ~623 each)"
[ "$sg" -ge 600 ] && [ "$sj" -ge 590 ] || { echo "FATAL: seed incomplete"; exit 1; }

echo "=== launch_jbb_increment gpu=$GPU cloud=$CLOUD ==="
python3 "$LOCAL/scripts/runpod_launch.py" --gpu "$GPU" --cloud "$CLOUD" --name "st-jbb-increment" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 220 --wait-min 22 || exit 1
read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break; sleep 10
done

echo "installing deps (hardened)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     apt-get update -qq; apt-get install -y -qq rsync; which rsync
     pip install -q --retries 8 --timeout 120 'torch==2.5.1'
     pip install -q --retries 8 --timeout 120 'transformers==5.14.1' 'accelerate==1.14.0' 'bitsandbytes==0.50.0' 'huggingface_hub==1.27.0' 'numpy==1.26.3'
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,accelerate,bitsandbytes; print(\"deps ok\", torch.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

echo "uploading src + runner + frozen(4-length) + SEED (N=5,20) data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/src /workspace/st/scripts /workspace/st/data/jbb_prefill/frozen_private /workspace/st/results/raw"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/scripts/run_jbb_increment.sh" "root@$HOST:/workspace/st/scripts/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$FROZEN" "root@$HOST:/workspace/st/data/jbb_prefill/frozen_private/"
for f in jbb_full.jsonl jbb_full_judged.jsonl jbb_openings.json; do
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$SEED_DIR/$f" "root@$HOST:/workspace/st/results/raw/"
done

# ---- verify upload + seed on pod ------------------------------------------
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     [ \$(ls /workspace/st/src/*.py | wc -l) -ge 15 ] || { echo 'FATAL: src incomplete'; exit 1; }
     python3 -c \"import json; d=json.load(open('/workspace/st/data/jbb_prefill/frozen_private/frozen.json')); assert d['k_lengths']==[5,10,15,20]; print('  pod frozen 4-length OK')\"
     echo \"  pod seed gens \$(wc -l < /workspace/st/results/raw/jbb_full.jsonl), judged \$(wc -l < /workspace/st/results/raw/jbb_full_judged.jsonl)\""

echo "launching increment runner..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_jbb_increment.sh && \
     nohup env HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash scripts/run_jbb_increment.sh > /workspace/pipeline_jbb_inc.log 2>&1 &
     echo \"runner launched, pid=\$!\"" </dev/null

cat <<EOF

=== LAUNCHED: JBB N=10,15 increment ===
  Pod: $INST   (ssh -p $PORT root@$HOST)
  gen(534) -> judge(~714), seeded past N=5,20; ~2 h, ~\$3-4.
  Watchdog:
  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 \\
  WATCHDOG_GRACE_MIN=15 WATCHDOG_STALL_MIN=45 WATCHDOG_MAX_HOURS=4 \\
  bash $LOCAL/scripts/watchdog.sh
EOF
