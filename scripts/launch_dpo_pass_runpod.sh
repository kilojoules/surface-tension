#!/bin/bash
# Launch the DPO pass-round (pair policy pass-v2) on RunPod.
# Mirrors launch_dpo_round_runpod.sh (the r1/r2 launcher) — same GPU/image/
# deps — but drives scripts/run_dpo_pass_round.sh on the pod instead of an
# inline pipeline, and refuses to launch until the prereg is externally
# timestamped.
#
# Cost/time expectation (from the r1 round's measured 19 h on this exact
# config, minus the val eval which auto-skips): ~14–17 h expected;
# A100-SXM4-80GB SECURE ≈ $1.49/h → ~$21–25 expected, $60 worst case at the
# 40 h watchdog cap. COMMUNITY is ~20–40% cheaper, less reliable.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_dpo_pass.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
START_ADAPTER="${START_ADAPTER:-kilojoules/surface-tension-dpo-r1-r32-final}"
HUB_REPO="${HUB_REPO:-kilojoules/surface-tension-dpo-pass-r32}"

# ---- gate 1: prereg must be externally timestamped before data generation --
PREREG="$LOCAL/prereg/dpo_pass_round_2026-07-23.md"
if grep -q '`____________`' "$PREREG"; then
    echo "REFUSING TO LAUNCH: $PREREG still has the external-timestamp"
    echo "placeholder. Register it (OSF/Zenodo, or a signed pushed tag) and"
    echo "record the link in the prereg first — repo policy (prereg/README.md)."
    exit 1
fi

# ---- gate 2: balance precheck (manual disaster #3 from 2026-05-28) ---------
python3 - <<'PY' || true
import runpod, os
runpod.api_key = open(os.path.expanduser("~/.run.pod")).read().strip()
try:
    u = runpod.get_user()
    bal = u.get("clientBalance") if isinstance(u, dict) else None
    print(f"RunPod balance: {bal}")
    if isinstance(bal, (int, float)) and bal < 40:
        print("WARNING: balance below the $40 worst-case cap — top up first.")
except Exception as e:
    print(f"(balance check unavailable via SDK: {e} — CHECK THE CONSOLE MANUALLY)")
PY
read -p "Balance covers the $60 worst case? [y/N] " ok
[ "$ok" = "y" ] || exit 1

echo "=== launch_dpo_pass (gpu=$GPU cloud=$CLOUD; init=$START_ADAPTER hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-dpo-pass \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 150 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate 'datasets<4.0' pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x; deps install failed\")' && python3 -c 'import transformers, peft, bitsandbytes, datasets'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + scripts + problem data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data,results/raw,outputs} /workspace/st/prereg" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/make_dpo_pass_pool.py" \
    "$LOCAL/scripts/run_dpo_pass_round.sh" \
    "root@$HOST:/workspace/st/scripts/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PREREG" "root@$HOST:/workspace/st/prereg/" 2>/dev/null

echo "launching pass-round pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_dpo_pass_round.sh && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       START_ADAPTER='$START_ADAPTER' HUB_REPO='$HUB_REPO' \
       bash scripts/run_dpo_pass_round.sh > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (DPO pass-round) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "  Tail: ssh -p $PORT root@$HOST 'tail -f /workspace/pipeline.log'"
echo ""
echo "Now start the watchdog (40 h cap = 2x the 19 h best case; excludes are"
echo "INPUT-side only — every output must sync):"
echo "  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE \\"
echo "  WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 \\"
echo "  WATCHDOG_MAX_HOURS=40 \\"
echo "  WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl --exclude=problems_dpo_pass45.jsonl\" \\"
echo "  WATCHDOG_LOG=/tmp/watchdog_dpo_pass.log bash $LOCAL/scripts/watchdog.sh"
