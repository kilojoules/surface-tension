#!/bin/bash
# GRPO data-scaling ablation. Two arms:
#   ARM=rl_only   : start from base, no SFT init, ADAPTER_INIT=none
#   ARM=sft_rl    : start from corresponding SFT-N adapter
#
# For sft_rl we pull the SFT adapter from HF Hub at HUB_PREFIX_SFT${N}-s${SEED}.
#
# Usage:
#   ARM=rl_only N_VALUES="50 100 200 300 500" bash scripts/launch_ablation_grpo.sh
#   ARM=sft_rl  N_VALUES="50 100 200 300 500" bash scripts/launch_ablation_grpo.sh

set -e
ARM="${ARM:?need ARM=rl_only or ARM=sft_rl}"
case "$ARM" in
    rl_only|sft_rl) ;;
    *) echo "ARM must be rl_only or sft_rl"; exit 1 ;;
esac

GPU_FILTER="${GPU_FILTER:-A100_SXM4}"
N_VALUES="${N_VALUES:-50 100 200 300 500}"
HUB_PREFIX_GRPO="${HUB_PREFIX_GRPO:-kilojoules/st-${ARM}-n}"
HUB_PREFIX_SFT="${HUB_PREFIX_SFT:-kilojoules/st-sft-n}"
GRPO_TRAIN_PROBLEMS="${GRPO_TRAIN_PROBLEMS:-../data/problems_expanded.jsonl}"
GRPO_EVAL_PROBLEMS="${GRPO_EVAL_PROBLEMS:-../data/problems_grpo_eval.jsonl}"
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_STEPS="${GRPO_STEPS:-30}"
G="${G:-8}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"
KL_BETA="${KL_BETA:-0.02}"
TEMPERATURE="${TEMPERATURE:-0.9}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-10}"
ABORT_PASS_FLOOR="${ABORT_PASS_FLOOR:-0.20}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
INSTANCE_FILE="$LOCAL/vast_current.env"

case "$GPU_FILTER" in
    RTX_4090)   DPH_MAX=0.50 ;;
    A100_SXM4)  DPH_MAX=1.20 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    *)          DPH_MAX=2.50 ;;
esac
RELIABILITY="${RELIABILITY:-0.99}"

echo "=== launch_ablation_grpo ARM=$ARM N_VALUES=[$N_VALUES] ==="

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
    --label "st-${ARM}-ablation-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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

LOOP_BODY=""
for N in $N_VALUES; do
    if [ "$ARM" = "rl_only" ]; then
        ADAPTER_ARG="ADAPTER_INIT='none'"
    else
        ADAPTER_ARG="ADAPTER_INIT='${HUB_PREFIX_SFT}${N}-s${SEED}'"
    fi
    LOOP_BODY+="        echo \"=== ${ARM} N=$N seed=$SEED ===\"
        $ADAPTER_ARG \
        GRPO_TRAIN_PROBLEMS='$GRPO_TRAIN_PROBLEMS' \
        GRPO_EVAL_PROBLEMS='$GRPO_EVAL_PROBLEMS' \
        GRPO_OUTPUT='../outputs/ablation/${ARM}_n${N}_s${SEED}' \
        GRPO_LIMIT_N='$N' \
        GRPO_LR='$GRPO_LR' GRPO_STEPS='$GRPO_STEPS' \
        G='$G' PROMPTS_PER_STEP='$PROMPTS_PER_STEP' \
        TEMPERATURE='$TEMPERATURE' KL_BETA='$KL_BETA' \
        EVAL_EVERY='$EVAL_EVERY' ABORT_PASS_FLOOR='$ABORT_PASS_FLOOR' \
        SEED='$SEED' \
        HUB_REPO='${HUB_PREFIX_GRPO}${N}-s${SEED}' \
        python -u grpo_train.py 2>&1 | tee /workspace/${ARM}_n${N}_s${SEED}.log || echo \"  (cell ${ARM}_n${N} failed but continuing)\"
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
echo "  Instance: $INST  ARM=$ARM"
echo "  SSH: ssh -p $PORT root@$HOST"
echo ""
echo "Next: WATCHDOG_MAX_HOURS=10 bash scripts/watchdog.sh"
