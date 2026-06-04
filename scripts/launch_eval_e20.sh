#!/bin/bash
# Complete the v9 SFT eval (the previous run was cut off at 60/90 by the 3h cap).
# Drops --constraints so it only generates the bare-prompt samples we actually need
# (halves the work). Adapter is already on Hub: kilojoules/surface-tension-sft-phaseA-lr1e4_e20.

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"
N_SAMPLES="${N_SAMPLES:-3}"
PROBLEMS_FILE="${PROBLEMS_FILE:-problems_lcb.jsonl}"
CSV_TAG="${CSV_TAG:-sft_v9}"
LIMIT="${LIMIT:-30}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-120}"
INSTANCE_FILE="$LOCAL/vast_eval_e20.env"
RELIABILITY="${RELIABILITY:-0.99}"
DPH_MAX=2.00

echo "=== launch_eval_v9 (LCB-30 bare prompt, n=$N_SAMPLES, no --constraints) ==="

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
    --label "st-eval-e20-noautokill" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && { echo "rent failed: $RESULT"; exit 1; }
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

echo "uploading..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/$PROBLEMS_FILE" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching eval..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         echo \"=== eval v9 on LCB-30 bare prompt, n=$N_SAMPLES ===\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/$PROBLEMS_FILE \
           --csv ../results/raw/eval_${CSV_TAG}_lcb30.csv \
           --source-dir ../results/raw/sources_eval_${CSV_TAG}_lcb30 \
           --base-model google/gemma-4-31B-it \
           --adapter kilojoules/surface-tension-sft-phaseA-lr1e4_e20 \
           --n-samples $N_SAMPLES --max-new-tokens 1024 --limit-problems $LIMIT \
           2>&1 | tee /workspace/eval_v9.log
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"eval launched, pid=\$!\""

echo ""
echo "=== LAUNCHED (v9 eval, complete) ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 WATCHDOG_GRACE_MIN=10 WATCHDOG_MAX_HOURS=8 bash scripts/watchdog.sh"
