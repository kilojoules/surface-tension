#!/bin/bash
# Bare-prompt eval-only sweep against ANY problems.jsonl using a Hub-hosted adapter.
# No training; just rent → load base+adapter → sweep_local --constraints (skip with-hint)
# → aggregate → all_done → watchdog destroys.
#
# Usage:
#   bash scripts/launch_eval_only.sh <problems_jsonl> <hub_adapter> <eval_tag> [gpu_filter]
# e.g.:
#   bash scripts/launch_eval_only.sh data/problems_mbpp30.jsonl kilojoules/surface-tension-sft v8_xb_mbpp A100_SXM4

set -e
PROBLEMS="${1:?need problems jsonl path (relative to repo root)}"
HUB_ADAPTER="${2:?need hub adapter repo id}"
TAG="${3:?need eval tag}"
GPU_FILTER="${4:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
INSTANCE_FILE="$LOCAL/vast_current.env"

case "$GPU_FILTER" in
    RTX_4090)   DPH_MAX=0.50 ;;
    A100_SXM4)  DPH_MAX=1.20 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    H100_NVL)   DPH_MAX=2.50 ;;
    *)          DPH_MAX=2.50 ;;
esac
RELIABILITY="${RELIABILITY:-0.99}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
N_SAMPLES="${N_SAMPLES:-3}"

if [ "$HUB_ADAPTER" = "none" ]; then
    ADAPTER_FLAG=""
else
    ADAPTER_FLAG="--adapter $HUB_ADAPTER"
fi

echo "=== launch_eval_only ==="
echo "  problems: $PROBLEMS"
echo "  adapter: $HUB_ADAPTER"
echo "  tag: $TAG"
echo "  gpu: $GPU_FILTER"

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=80 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-eval-${TAG}-noautokill" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && { echo "rent failed"; exit 1; }
echo "instance: $INST"

for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    sleep 15
done
[ "$STATUS" != "running" ] && { echo "FAIL"; echo "n" | vastai destroy instance "$INST"; exit 1; }

SSH_INFO=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST"

for i in $(seq 1 20); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "pip install -q transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading code + problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --include='*.txt' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/$PROBLEMS" "root@$HOST:/workspace/st/$PROBLEMS" 2>/dev/null

echo "launching bare-prompt-only eval..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup bash -euo pipefail -c '
       echo \"=== bare-prompt eval (no with-hint) ===\"
       python -u sweep_local.py \
         --problems ../$PROBLEMS \
         --csv ../results/raw/${TAG}_raw.csv \
         --source-dir ../results/raw/sources_$TAG \
         --base-model google/gemma-4-31B-it \
         $ADAPTER_FLAG \
         --n-samples $N_SAMPLES \
         --max-new-tokens $MAX_NEW_TOKENS \
         --constraints 2>&1 | tee /workspace/${TAG}.log

       python -u aggregate.py \
         --csv ../results/raw/${TAG}_raw.csv \
         --summary-csv ../results/${TAG}_summary.csv \
         --summary-md ../results/${TAG}_summary.md \
         --sources-dir ../results/raw/sources_$TAG 2>&1 | tee /workspace/${TAG}_agg.log

       touch /workspace/all_done
     ' > /workspace/pipeline.log 2>&1 &
     echo pid=\$!"

echo ""
echo "=== LAUNCHED ==="
echo "  next: WATCHDOG_MAX_HOURS=4 bash scripts/watchdog.sh"
