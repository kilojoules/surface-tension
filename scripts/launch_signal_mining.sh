#!/bin/bash
# Mine "signal-rich" problems for GRPO training: problems where base-model bare-prompt
# compliance is in the discriminating zone [0.3, 0.7]. Avoids the saturation problem
# we hit with MBPP-30 (baseline 0.69 → groups all-1).
#
# Pipeline (all on remote):
#   Stage A: bare-prompt sweep with no adapter on problems_expanded.jsonl, N=4 samples
#   Stage B: filter problems by per-problem compliance ∈ [0.3, 0.7]
#   Stage C: write data/problems_signal_rich.jsonl
#
# Usage:
#   bash scripts/launch_signal_mining.sh

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"
LIMIT_PROBLEMS="${LIMIT_PROBLEMS:-200}"  # subset of expanded set to keep it fast
N_SAMPLES="${N_SAMPLES:-4}"
COMPL_LOW="${COMPL_LOW:-0.3}"
COMPL_HIGH="${COMPL_HIGH:-0.7}"
ADAPTER="${ADAPTER:-}"               # e.g. kilojoules/surface-tension-sft for v8-active mining
TAG="${TAG:-signal_mine}"            # output tag; e.g. signal_mine_v8 for v8 run

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-120}"
INSTANCE_FILE="$LOCAL/vast_current.env"

case "$GPU_FILTER" in
    RTX_4090)   DPH_MAX=0.50 ;;
    A100_SXM4)  DPH_MAX=1.20 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    *)          DPH_MAX=2.50 ;;
esac
RELIABILITY="${RELIABILITY:-0.99}"

echo "=== launch_signal_mining (limit=$LIMIT_PROBLEMS, n=$N_SAMPLES, comp ∈ [$COMPL_LOW, $COMPL_HIGH]) ==="

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
    --label "st-signal-mine-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
    "$LOCAL/data/problems_expanded.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching mining pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"=== Stage A: bare-prompt sweep on $LIMIT_PROBLEMS expanded problems (adapter=${ADAPTER:-none}) ===\"
         python -u sweep_local.py \
           --problems ../data/problems_expanded.jsonl \
           --csv ../results/raw/${TAG}_raw.csv \
           --source-dir ../results/raw/sources_${TAG} \
           --base-model google/gemma-4-31B-it \
           ${ADAPTER:+--adapter $ADAPTER} \
           --n-samples $N_SAMPLES \
           --max-new-tokens 1024 \
           --limit-problems $LIMIT_PROBLEMS \
           --constraints 2>&1 | tee /workspace/${TAG}_sweep.log

         echo \"=== Stage A complete; filter step is run locally after sync ===\"
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_GRACE_MIN=60 WATCHDOG_MAX_HOURS=4 bash scripts/watchdog.sh"
