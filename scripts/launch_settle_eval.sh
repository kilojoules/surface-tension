#!/bin/bash
# Instance 1: settle the flip-rate / regression claims.
# Stage A: n=20 base + n=20 v8-SFT bare-prompt eval on LCB-30 (all 30 problems).
# Stage B: forward-pass loss for v8 SFT on the 146 training pairs.
#
# Outputs: vast_logs/$INST/st/results/raw/{eval_settle_*,per_pair_loss_v8.csv}

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-120}"
INSTANCE_FILE="$LOCAL/vast_settle_eval.env"
RELIABILITY="${RELIABILITY:-0.99}"

case "$GPU_FILTER" in
    A100_SXM4)  DPH_MAX=1.50 ;;
    H100_SXM)   DPH_MAX=4.50 ;;
    *)          DPH_MAX=1.50 ;;
esac
GPU_RAM_MIN="${GPU_RAM_MIN:-70}"   # 70GB ⇒ A100 80GB / H100 80GB

echo "=== launch_settle_eval (gpu=$GPU_FILTER ram>=${GPU_RAM_MIN}GB) ==="

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER gpu_ram>$GPU_RAM_MIN num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=80 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-settle-eval-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
    "$LOCAL/data/problems_lcb.jsonl" "$LOCAL/data/sft_all.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"=== Stage A1: base bare-prompt n=20 on LCB-30 ===\"
         python -u sweep_local.py \
           --problems ../data/problems_lcb.jsonl \
           --csv ../results/raw/eval_settle_base_lcb30.csv \
           --source-dir ../results/raw/sources_eval_settle_base_lcb30 \
           --base-model google/gemma-4-31B-it \
           --n-samples 20 \
           --max-new-tokens 1024 \
           --limit-problems 30 \
           --constraints 2>&1 | tee /workspace/settle_base.log

         touch /workspace/stage_a1_done

         echo \"=== Stage A2: v8 SFT bare-prompt n=20 on LCB-30 ===\"
         python -u sweep_local.py \
           --problems ../data/problems_lcb.jsonl \
           --csv ../results/raw/eval_settle_sft_lcb30.csv \
           --source-dir ../results/raw/sources_eval_settle_sft_lcb30 \
           --base-model google/gemma-4-31B-it \
           --adapter kilojoules/surface-tension-sft \
           --n-samples 20 \
           --max-new-tokens 1024 \
           --limit-problems 30 \
           --constraints 2>&1 | tee /workspace/settle_sft.log

         touch /workspace/stage_a2_done

         echo \"=== Stage B: per-pair forward-pass loss for v8 SFT ===\"
         ADAPTER=kilojoules/surface-tension-sft \
         PAIRS=../data/sft_all.jsonl \
         OUT=../results/raw/per_pair_loss_v8.csv \
           python -u per_pair_loss.py 2>&1 | tee /workspace/per_pair_loss.log

         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 WATCHDOG_GRACE_MIN=10 WATCHDOG_MAX_HOURS=2 bash scripts/watchdog.sh"
