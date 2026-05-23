#!/bin/bash
# SFT data-scaling ablation: train at N values in {N_VALUES}, single seed,
# fixed hyperparameters. Each (N) gets its own adapter saved + pushed to hub.
# All runs share one rented box (model reloads between cells; ~15min × N
# overhead is cheaper than per-cell renting).
#
# Usage:
#   N_VALUES="50 100 200 300 500" bash scripts/launch_ablation_sft.sh
#
# Defaults:
#   N_VALUES        "50 100 200 300 500"
#   HUB_PREFIX      kilojoules/st-sft-n
#   SFT_TRAIN_FILE  ../data/sft_full.jsonl  (use sft_full from datagen merge)
#   SFT_LR          1e-5
#   SFT_EPOCHS      3

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"
N_VALUES="${N_VALUES:-50 100 200 300 500}"
HUB_PREFIX="${HUB_PREFIX:-kilojoules/st-sft-n}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-../data/sft_full.jsonl}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SEED="${SEED:-0}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-80}"
INSTANCE_FILE="$LOCAL/vast_current.env"

case "$GPU_FILTER" in
    RTX_4090)   DPH_MAX=0.50 ;;
    A100_SXM4)  DPH_MAX=1.20 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    *)          DPH_MAX=2.50 ;;
esac
RELIABILITY="${RELIABILITY:-0.99}"

echo "=== launch_ablation_sft N_VALUES=[$N_VALUES] ==="

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
    --label "st-sft-ablation-noautokill" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && { echo "rent failed: $RESULT"; exit 1; }

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

echo "uploading code + data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/" "root@$HOST:/workspace/st/data/" --exclude='*.csv' 2>/dev/null

echo "launching SFT ablation pipeline (N values: $N_VALUES)..."
# Build the loop body as a string of python invocations
LOOP_BODY=""
for N in $N_VALUES; do
    LOOP_BODY+="        echo \"=== SFT N=$N seed=$SEED ===\"
        SFT_TRAIN='$SFT_TRAIN_FILE' \
        SFT_OUTPUT='../outputs/ablation/sft_n${N}_s${SEED}' \
        SFT_LIMIT_N='$N' \
        SFT_LR='$SFT_LR' SFT_EPOCHS='$SFT_EPOCHS' \
        EVAL_EVERY='${EVAL_EVERY:-0}' \
        SEED='$SEED' \
        HUB_REPO='${HUB_PREFIX}${N}-s${SEED}' \
        python -u sft_train.py 2>&1 | tee /workspace/sft_n${N}_s${SEED}.log
"
done

ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
$LOOP_BODY
        touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "  Pipeline: ssh -p $PORT root@$HOST 'tail -f /workspace/pipeline.log'"
echo ""
echo "Next: WATCHDOG_MAX_HOURS=10 bash scripts/watchdog.sh   # in another terminal"
