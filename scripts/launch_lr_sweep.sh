#!/bin/bash
# LR × epoch sweep on bf16 base + LoRA rank=32 (matches v8 SFT capacity).
# Tests whether training compute / LR was a confound for the rank result.
#
# Vary: SFT_LR ∈ {1e-4, 1e-5, 1e-6} — order-of-magnitude bracket
# Hold: rank=32, alpha=16, epochs=20, all other hyperparameters
# bf16 base avoids the bnb 4-bit rank-shape CUDA bug.
#
# Outputs (per LR, with $TAG = lr2e5 etc.):
#   Hub: kilojoules/surface-tension-sft-bf16-$TAG
#   eval CSV: ../results/raw/eval_bf16_$TAG_lcb30.csv

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"
GPU_RAM_MIN="${GPU_RAM_MIN:-70}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-200}"
INSTANCE_FILE="$LOCAL/vast_lr_sweep.env"
RELIABILITY="${RELIABILITY:-0.99}"

case "$GPU_FILTER" in
    A100_SXM4)  DPH_MAX=1.50 ;;
    H100_SXM)   DPH_MAX=4.50 ;;
    *)          DPH_MAX=1.50 ;;
esac

echo "=== launch_lr_sweep (LRs={1e-4,1e-5,1e-6}, epochs=20, bf16, rank=32) ==="

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER gpu_ram>$GPU_RAM_MIN num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=160 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-lr-sweep-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
    "pip install -q transformers peft accelerate datasets pandas 2>&1 | tail -1
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
       bash -uo pipefail -c '
         for ENTRY in 1e-4:lr1e4 1e-5:lr1e5 1e-6:lr1e6; do
           LR=\${ENTRY%%:*}
           TAG=\${ENTRY##*:}
           REPO=kilojoules/surface-tension-sft-bf16-\$TAG
           echo \"=== training lr=\$LR (tag=\$TAG, rank=32, 20 epochs, bf16) ===\"
           if SFT_TRAIN=../data/sft_all.jsonl \
              SFT_EVAL=../data/sft_all.jsonl \
              SFT_OUTPUT=../outputs/sft_bf16_\$TAG \
              LORA_RANK=32 LORA_ALPHA=16 \
              SFT_LR=\$LR SFT_EPOCHS=20 \
              EVAL_EVERY=0 EVAL_N=0 \
              QUANT_BIT=0 \
                python -u sft_train.py 2>&1 | tee /workspace/sft_\$TAG.log; then
             touch /workspace/stage_\${TAG}_trained
             echo \"--- pushing \$TAG to Hub ---\"
             timeout 900 python -u push_adapter.py ../outputs/sft_bf16_\$TAG/final_adapter \$REPO 2>&1 | tee /workspace/push_\$TAG.log || \
               echo \"WARN: push \$TAG failed\"
             touch /workspace/stage_\${TAG}_pushed
             echo \"--- eval \$TAG on LCB-30 ---\"
             QUANT_BIT=0 python -u sweep_local.py \
               --problems ../data/problems_lcb.jsonl \
               --csv ../results/raw/eval_bf16_\${TAG}_lcb30.csv \
               --source-dir ../results/raw/sources_eval_bf16_\${TAG}_lcb30 \
               --base-model google/gemma-4-31B-it \
               --adapter \$REPO \
               --n-samples 3 \
               --max-new-tokens 1024 \
               --limit-problems 30 \
               --constraints 2>&1 | tee /workspace/eval_\$TAG.log || \
               echo \"WARN: eval \$TAG failed\"
             touch /workspace/stage_\${TAG}_evaled
           else
             echo \"WARN: training \$TAG failed\"
             touch /workspace/stage_\${TAG}_failed
           fi
         done
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED (LR sweep) ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 WATCHDOG_GRACE_MIN=15 WATCHDOG_MAX_HOURS=10 bash scripts/watchdog.sh"
