#!/bin/bash
# DPO pair-yield probe — round-2 viability check.
# Sample the current adapter (default: DPO-r1) on the 45-problem LCB pool, count pairs.
# No DPO training, no eval — just stage 1 of the loop.
#
# Decision rule:
#   >=80 pairs  → round 2 worth running on this pool
#   40-79 pairs → marginal; consider a wider pool
#   <40 pairs   → pool exhausted; switch pools BEFORE next round
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_dpo_probe.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
ADAPTER="${ADAPTER:-kilojoules/surface-tension-dpo-r1-r32-final}"
TAG="${TAG:-dpo_r1_probe}"

echo "=== launch_dpo_yieldprobe (gpu=$GPU cloud=$CLOUD; sample $ADAPTER) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-dpo-probe \
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
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x\")'
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problem pool..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_dpopool.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching pair-sampling probe..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== probe: sample $ADAPTER on 45-problem pool ==========\"
         python -u build_dpo_pairs.py \
           --problems ../data/problems_lcb_dpopool.jsonl \
           --out /workspace/dpo_pairs_$TAG.jsonl \
           --adapter $ADAPTER \
           --n-samples 8 --max-new-tokens 2048 --temperature 0.9 --max-pairs 6 \
           2>&1 | tee /workspace/dpo_pairs_$TAG.log
         echo \"========== final pair count ==========\"
         wc -l /workspace/dpo_pairs_$TAG.jsonl
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"probe launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (DPO probe: $ADAPTER) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=20 WATCHDOG_STALL_MIN=60 WATCHDOG_MAX_HOURS=5 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_dpopool.jsonl\" WATCHDOG_LOG=/tmp/watchdog_dpo_probe.log bash $LOCAL/scripts/watchdog.sh"
