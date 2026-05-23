#!/bin/bash
# LoRA rank sweep at ranks {128, 256, 512} on the 146-pair sft_all dataset.
# Replaces the rank-sweep portion of launch_capacity_onpolicy.sh with two fixes:
#   1. EVAL_EVERY=0 disables in-training mini-eval abort
#   2. Hub push happens as a separate `timeout 900` Python step (not via
#      sft_train.py's model.push_to_hub which hung last time).
#
# rank=32 is v8 (already on Hub); rank=64 is already evaluated; this run
# fills in the rest. Skip on-policy for this run; rank-capacity question first.
#
# Outputs: vast_logs/$INST/st/results/raw/eval_r{128,256,512}_lcb30.csv
# Hub: kilojoules/surface-tension-sft-r{128,256,512}

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-180}"
INSTANCE_FILE="$LOCAL/vast_rank_sweep.env"
RELIABILITY="${RELIABILITY:-0.99}"

case "$GPU_FILTER" in
    A100_SXM4)  DPH_MAX=1.20 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    *)          DPH_MAX=1.50 ;;
esac

echo "=== launch_rank_sweep ==="

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=120 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-rank-sweep-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
       bash -uo pipefail -c '
         for RANK in 128 256 512; do
           ALPHA=\$((RANK / 2))
           REPO=kilojoules/surface-tension-sft-r\$RANK
           echo \"=== training rank=\$RANK alpha=\$ALPHA ===\"
           if SFT_TRAIN=../data/sft_all.jsonl \
              SFT_EVAL=../data/sft_all.jsonl \
              SFT_OUTPUT=../outputs/sft_r\$RANK \
              LORA_RANK=\$RANK \
              LORA_ALPHA=\$ALPHA \
              SFT_LR=1e-5 SFT_EPOCHS=3 \
              EVAL_EVERY=0 EVAL_N=0 \
                python -u sft_train.py 2>&1 | tee /workspace/sft_r\$RANK.log; then
             touch /workspace/stage_r\${RANK}_trained

             echo \"--- pushing rank=\$RANK to Hub (timeout 900s) ---\"
             timeout 900 python -u push_adapter.py ../outputs/sft_r\$RANK/final_adapter \$REPO 2>&1 | tee /workspace/push_r\$RANK.log || \
               echo \"WARN: push rank=\$RANK timed out or failed; eval will skip\"
             touch /workspace/stage_r\${RANK}_pushed

             echo \"--- eval rank=\$RANK on LCB-30 ---\"
             python -u sweep_local.py \
               --problems ../data/problems_lcb.jsonl \
               --csv ../results/raw/eval_r\${RANK}_lcb30.csv \
               --source-dir ../results/raw/sources_eval_r\${RANK}_lcb30 \
               --base-model google/gemma-4-31B-it \
               --adapter \$REPO \
               --n-samples 3 \
               --max-new-tokens 1024 \
               --limit-problems 30 \
               --constraints 2>&1 | tee /workspace/eval_r\$RANK.log || \
               echo \"WARN: eval rank=\$RANK failed\"
             touch /workspace/stage_r\${RANK}_evaled
           else
             echo \"WARN: rank=\$RANK training failed\"
             touch /workspace/stage_r\${RANK}_failed
           fi
         done
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 WATCHDOG_GRACE_MIN=15 WATCHDOG_MAX_HOURS=4 bash scripts/watchdog.sh"
