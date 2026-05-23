#!/bin/bash
# Phase A of the SFT hyperparameter sweep (idea-critic design): train-only, no LCB eval.
# Four configs run sequentially on one instance, 8-epoch ones first:
#   lr1e4_e8   SFT_LR=1e-4 SFT_EPOCHS=8
#   lr3e4_e8   SFT_LR=3e-4 SFT_EPOCHS=8
#   lr1e4_e20  SFT_LR=1e-4 SFT_EPOCHS=20
#   lr3e4_e20  SFT_LR=3e-4 SFT_EPOCHS=20
# All: rank=32, alpha=64 (bump LoRA gain vs v8's 0.5x), dropout=0, linear-decay-to-10%
# schedule (not cosine-to-zero), STRIP_WRAPPERS=1 (the gradient-flow fix), 4-bit + GC.
# Each run: fine-grained loss logging (LOG_EVERY=2), end-of-run fit probe (teacher-forced
# NLL on train+eval, adapter on vs off), push adapter to Hub. NO LCB-30 eval here — that's
# Phase B, on whichever config fits best.
#
# Outputs: vast_logs/$INST/sft_phaseA_<tag>.log ; Hub: kilojoules/surface-tension-sft-phaseA-<tag>

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-150}"
INSTANCE_FILE="$LOCAL/vast_sft_phaseA.env"
RELIABILITY="${RELIABILITY:-0.99}"
DPH_MAX=1.20

echo "=== launch_sft_phaseA (4 configs, train-only, linear schedule, alpha=64) ==="

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
    --label "st-sft-phaseA-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
    "$LOCAL/data/sft_all.jsonl" "$LOCAL/data/sft_eval.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching Phase A pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         run() {  # \$1=tag  \$2=lr  \$3=epochs
           TAG=\$1; LR=\$2; EP=\$3
           REPO=kilojoules/surface-tension-sft-phaseA-\$TAG
           echo \"========== PHASE-A \$TAG  (lr=\$LR epochs=\$EP) ==========\"
           if SFT_TRAIN=../data/sft_all.jsonl SFT_EVAL=../data/sft_eval.jsonl \
              SFT_OUTPUT=../outputs/phaseA_\$TAG \
              LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
              SFT_LR=\$LR SFT_EPOCHS=\$EP LR_SCHEDULE=linear \
              QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 \
              EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=2 \
                python -u sft_train.py 2>&1 | tee /workspace/sft_phaseA_\$TAG.log; then
             touch /workspace/phaseA_\${TAG}_trained
             timeout 900 python -u push_adapter.py ../outputs/phaseA_\$TAG/final_adapter \$REPO 2>&1 | tee /workspace/push_\$TAG.log || echo \"WARN: push \$TAG failed\"
             touch /workspace/phaseA_\${TAG}_pushed
           else
             echo \"WARN: \$TAG training failed/aborted\"
             touch /workspace/phaseA_\${TAG}_failed
           fi
         }
         run lr1e4_e8   1e-4 8
         run lr3e4_e8   3e-4 8
         run lr1e4_e20  1e-4 20
         run lr3e4_e20  3e-4 20
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"Phase A launched, pid=\$!\""

echo ""
echo "=== LAUNCHED (SFT Phase A: 4 train-only configs) ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=12 WATCHDOG_MAX_HOURS=12 bash scripts/watchdog.sh"
