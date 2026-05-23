#!/bin/bash
# Training-fix diagnostic v2. Tests whether stripping the Gemma4ClippableLinear
# wrappers before attaching LoRA restores gradient flow.
#
#   1. q4_strip_nogc      QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=0  lr=1e-5  — candidate fix
#   2. q4_strip_gc        QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1  lr=1e-5  — fix + non-reentrant GC
#   3. q4_nostrip_nogc    QUANT_BIT=4 STRIP_WRAPPERS=0 USE_GC=0  lr=1e-5  — control (wrapper vs GC)
#   4. q4_strip_gc_lr1e3  QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1  lr=1e-3  — sanity: high LR must move loss
#
# Reads: grad-check line (n params with nonzero grad), per-2-step loss + grad_norm.
# Win condition: strip runs show grad_norm > 0 and loss that visibly moves; nostrip run
# reproduces grad_norm = 0. (Note: sft_train.py now HARD-FAILS on zero grad, so a broken
# run will RuntimeError after step 0 rather than running silently.)

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-120}"
INSTANCE_FILE="$LOCAL/vast_train_diag.env"
RELIABILITY="${RELIABILITY:-0.99}"
DPH_MAX=1.20

echo "=== launch_train_diagnostic v2 (wrapper-strip test) ==="

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
    --label "st-train-diag-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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

echo "launching diagnostic..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         run() { TAG=\$1; shift
           echo \"========== DIAG \$TAG ==========\"
           env \"\$@\" \
             SFT_TRAIN=../data/sft_all.jsonl SFT_EVAL=../data/sft_all.jsonl \
             SFT_OUTPUT=../outputs/diag_\$TAG \
             LORA_RANK=32 LORA_ALPHA=16 SFT_EPOCHS=2 \
             EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=2 \
             python -u sft_train.py 2>&1 | tee /workspace/diag_\$TAG.log || echo \"WARN: \$TAG ended non-zero\"
           touch /workspace/diag_\${TAG}_done
         }
         run q4_strip_nogc      QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=0 SFT_LR=1e-5
         run q4_strip_gc        QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 SFT_LR=1e-5
         run q4_nostrip_nogc    QUANT_BIT=4 STRIP_WRAPPERS=0 USE_GC=0 SFT_LR=1e-5
         run q4_strip_gc_lr1e3  QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 SFT_LR=1e-3
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"diagnostic launched, pid=\$!\""

echo ""
echo "=== LAUNCHED (train diagnostic v2) ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=60 WATCHDOG_GRACE_MIN=8 WATCHDOG_MAX_HOURS=2 bash scripts/watchdog.sh"
