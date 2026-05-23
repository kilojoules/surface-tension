#!/bin/bash
# Zero-compute pre-check (well, ~$2 of compute): can the BASE Gemma 4 31B-it model,
# with constraint hint + rationale disclosure in the prompt at INFERENCE time,
# produce loop-free passing solutions for the 5 problems that rationale-SFT
# trained-model fails on at bare-prompt time?
# This disambiguates whether the always-violate tail is a hard ceiling.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_alwaysviolate_check.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"

echo "=== launch_alwaysviolate_check (5 problems × n=4, base model + hint + disclosure) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-alwaysviolate-check \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 130 --wait-min 22 || exit 1

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
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch, transformers; print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + 5-problem file..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_alwaysviolate5.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== always-violate pre-check (n=4, base + hint + disclosure) ==========\"
         python -u build_rationale_dataset.py \
           --problems ../data/problems_lcb_alwaysviolate5.jsonl \
           --out /workspace/alwaysviolate_acceptances.jsonl \
           --n-samples 4 --max-new-tokens 2048 --temperature 0.7 \
           2>&1 | tee /workspace/alwaysviolate_check.log
         echo \"========== ACCEPTANCES (compliant+passing) ==========\"
         wc -l /workspace/alwaysviolate_acceptances.jsonl || echo \"file missing\"
         echo \"========== per-problem summary ==========\"
         python3 -c \"
import json, collections
counts = collections.Counter()
try:
    for line in open(\\\"/workspace/alwaysviolate_acceptances.jsonl\\\"):
        counts[json.loads(line)[\\\"problem_id\\\"]] += 1
except FileNotFoundError:
    pass
for pid in [\\\"lcb/abc356_c\\\",\\\"lcb/abc374_d\\\",\\\"lcb/abc376_b\\\",\\\"lcb/arc183_a\\\",\\\"lcb/arc189_a\\\"]:
    print(f\\\"  {pid}: {counts.get(pid, 0)}/4 compliant+passing\\\")
\"
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=20 WATCHDOG_STALL_MIN=60 WATCHDOG_MAX_HOURS=3 bash $LOCAL/scripts/watchdog.sh"
