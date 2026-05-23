#!/bin/bash
# v9 SFT — the FIRST SFT run that actually trains the model. Same config as v8
# (4-bit, rank=32, alpha=16, lr=1e-5, 3 epochs, sft_all.jsonl 146 pairs) but with
# STRIP_WRAPPERS=1 so LoRA gradients actually flow. v8 was a no-op (zero gradient).
#
# Verifies: grad_norm > 0 (sft_train.py hard-fails otherwise), loss trend, adapter
# B matrices nonzero. Pushes to Hub as kilojoules/surface-tension-sft-v9.
# Then evals on LCB-30 bare prompt, n=3 — the headline number.

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-150}"
INSTANCE_FILE="$LOCAL/vast_sft_v9.env"
RELIABILITY="${RELIABILITY:-0.99}"
DPH_MAX=1.20

echo "=== launch_sft_v9 (4-bit, rank=32, lr=1e-5, 3 epochs, STRIP_WRAPPERS=1) ==="

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=100 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-sft-v9-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
    "$LOCAL/data/sft_all.jsonl" "$LOCAL/data/problems_lcb.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         echo \"=== Stage A: train v9 SFT (4-bit, rank=32, lr=1e-5, 3 epochs, STRIP_WRAPPERS=1) ===\"
         SFT_TRAIN=../data/sft_all.jsonl SFT_EVAL=../data/sft_all.jsonl \
         SFT_OUTPUT=../outputs/sft_v9 \
         LORA_RANK=32 LORA_ALPHA=16 SFT_LR=1e-5 SFT_EPOCHS=3 \
         QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 \
         EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=5 \
           python -u sft_train.py 2>&1 | tee /workspace/sft_v9.log
         touch /workspace/stage_a_done

         echo \"=== Stage B: push v9 adapter to Hub ===\"
         timeout 900 python -u push_adapter.py ../outputs/sft_v9/final_adapter kilojoules/surface-tension-sft-v9 2>&1 | tee /workspace/push_v9.log || echo \"WARN: push timed out\"
         touch /workspace/stage_b_done

         echo \"=== Stage C: eval v9 on LCB-30 bare prompt, n=3 ===\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb.jsonl \
           --csv ../results/raw/eval_sft_v9_lcb30.csv \
           --source-dir ../results/raw/sources_eval_sft_v9_lcb30 \
           --base-model google/gemma-4-31B-it \
           --adapter kilojoules/surface-tension-sft-v9 \
           --n-samples 3 --max-new-tokens 1024 --limit-problems 30 \
           --constraints 2>&1 | tee /workspace/eval_v9.log || echo \"WARN: eval failed\"
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED (v9 SFT — the real one) ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 WATCHDOG_GRACE_MIN=10 WATCHDOG_MAX_HOURS=3 bash scripts/watchdog.sh"
