#!/bin/bash
# Launch ONE pod that extends ONE step-0 prefill cell to n=8/problem (/136).
# Prereg: prereg/step0_substitution_power_2026-09-02.md (gist-anchored
# bf0660fd… on 2026-09-02, before any new generation).
#
# ARM=rsft|vanilla   (call twice, one pod per arm — independent failure
# domains, same total bill; parallelism cuts wall clock, not cost).
#
# Cost expectation, from measured throughput (51 gens in 3h37m on this exact
# recipe = 4.25 min/gen): 85 new gens ≈ 6.0 h + ~0.4 h setup ≈ $9.6/pod at
# A100-SXM4-80GB SECURE $1.49/h. Two pods ≈ $19 expected. WATCHDOG_MAX_HOURS
# =13 (2.03× best case) → $39 hard ceiling for the pair.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
ARM="${ARM:?set ARM=rsft|vanilla}"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"

case "$ARM" in
  rsft)    SEED_DIR="$LOCAL/data/evidence/step0_v1/st-step0_yjljmvip0iz88b"
           TAG=eval_step0_prefill ;;
  vanilla) SEED_DIR="$LOCAL/data/evidence/step0_v1/st-step0b_tdag86rvn5jowy"
           TAG=eval_step0b_sft_prefill ;;
  *) echo "FATAL: unknown ARM=$ARM"; exit 1 ;;
esac
INSTANCE_FILE="$LOCAL/vast_step0pow_$ARM.env"

# ---- gate 1: prereg externally timestamped before data generation ---------
PREREG="$LOCAL/prereg/step0_substitution_power_2026-09-02.md"
if grep -q '`____________`' "$PREREG"; then
    echo "REFUSING TO LAUNCH: $PREREG still has the external-timestamp"
    echo "placeholder. Anchor it (public gist) and record the link first."
    exit 1
fi

# ---- gate 2: seed state must exist and be the expected size ---------------
[ -f "$SEED_DIR/$TAG.csv" ] || { echo "FATAL: no seed CSV at $SEED_DIR/$TAG.csv"; exit 1; }
rows=$(( $(wc -l < "$SEED_DIR/$TAG.csv") - 1 ))
srcs=$(ls "$SEED_DIR/sources_$TAG" | wc -l | tr -d ' ')
echo "seed: $rows CSV rows, $srcs source files (expect 51 / 102)"
[ "$rows" -eq 51 ] || { echo "FATAL: expected 51 seed rows, got $rows"; exit 1; }

echo "=== launch_step0_power arm=$ARM gpu=$GPU cloud=$CLOUD ==="
python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-step0pow-$ARM" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 150 --wait-min 22 || exit 1
read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

# ---- deps: pinned to the 2026-08-11 propensity pod's verified freeze ------
# (unsilenced + verified: a silent `apt-get install rsync` failure on
# 2026-08-13 made uploads bounce and crashed the runner instantly.)
echo "installing deps (pinned)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     apt-get update -qq
     apt-get install -y -qq rsync
     which rsync
     pip install -q 'torch==2.5.1'
     pip install -q 'transformers==5.14.1' 'peft==0.20.0' 'bitsandbytes==0.50.0' \
                    'accelerate==1.14.0' 'datasets==3.6.0' 'pandas==3.0.5' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3'
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,peft,bitsandbytes,accelerate,datasets,pandas; \
       print(\"deps ok\", torch.__version__, transformers.__version__, peft.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

echo "uploading src + runner + deck + seed state..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data,results/raw}"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/run_step0_power.sh" "root@$HOST:/workspace/st/scripts/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" "root@$HOST:/workspace/st/data/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$SEED_DIR/$TAG.csv" "root@$HOST:/workspace/st/results/raw/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$SEED_DIR/sources_$TAG/" "root@$HOST:/workspace/st/results/raw/sources_$TAG/"

# ---- gate 3: verified file counts before the runner starts ----------------
# (the 2026-08-13 loss was a runner launched onto an empty /workspace.)
echo "verifying upload..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     n_src=\$(ls /workspace/st/src/*.py | wc -l)
     n_seed=\$(ls /workspace/st/results/raw/sources_$TAG | wc -l)
     n_rows=\$(wc -l < /workspace/st/results/raw/$TAG.csv)
     echo \"  src=\$n_src seed_sources=\$n_seed csv_lines=\$n_rows\"
     [ \"\$n_src\" -ge 15 ] || { echo 'FATAL: src upload incomplete'; exit 1; }
     [ \"\$n_seed\" -eq 102 ] || { echo 'FATAL: seed sources incomplete'; exit 1; }
     [ \"\$n_rows\" -eq 52 ] || { echo 'FATAL: seed csv wrong size'; exit 1; }
     [ -f /workspace/st/data/problems_lcb_clean17.jsonl ] || { echo 'FATAL: no deck'; exit 1; }"

echo "launching runner (arm=$ARM)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_step0_power.sh && \
     nohup env ARM=$ARM HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash scripts/run_step0_power.sh > /workspace/pipeline_$ARM.log 2>&1 &
     echo \"runner launched, pid=\$!\"" </dev/null

cat <<EOF

=== LAUNCHED: step-0 power run, arm=$ARM ===
  Pod: $INST   (ssh -p $PORT root@$HOST)
  Expect 85 new generations, ~6.0 h, ~\$9.6.
Watchdog (start this now):
  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 \\
  WATCHDOG_GRACE_MIN=15 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=13 \\
  WATCHDOG_RSYNC_EXCLUDES="--exclude=problems_lcb_*.jsonl" \\
  WATCHDOG_LOG=/tmp/watchdog_step0pow_$ARM.log bash $LOCAL/scripts/watchdog.sh
EOF
