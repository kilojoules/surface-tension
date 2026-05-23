#!/bin/bash
# GRPO continuation from v8 SFT adapter. Train on MBPP-30, eval cross-benchmark on
# LCB-30 + MBPP-30. Stacked-adapter trick anchors KL to v8.
#
# Usage:
#   bash scripts/launch_grpo.sh [adapter_init] [hub_repo] [gpu_filter]
#
# Smoke-test override (small/fast):
#   GRPO_STEPS=5 G=2 PROMPTS_PER_STEP=2 RUN_FULL_EVAL=0 bash scripts/launch_grpo.sh ...
#
# Full run (default):
#   bash scripts/launch_grpo.sh

set -e
ADAPTER_INIT="${1:-kilojoules/surface-tension-sft}"
HUB_REPO="${2:-kilojoules/surface-tension-grpo}"
GPU_FILTER="${3:-A100_SXM4}"

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
GRPO_STEPS="${GRPO_STEPS:-50}"
G="${G:-8}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"
GRPO_LR="${GRPO_LR:-2e-6}"
KL_BETA="${KL_BETA:-0.02}"
KL_REFERENCE="${KL_REFERENCE:-v8}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_N="${EVAL_N:-8}"
ABORT_VAR_FLOOR="${ABORT_VAR_FLOOR:-0.0}"
ABORT_VAR_CHECK_STEP="${ABORT_VAR_CHECK_STEP:-20}"
N_SAMPLES_EVAL="${N_SAMPLES_EVAL:-3}"
LCB_LIMIT="${LCB_LIMIT:-30}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"

echo "=== launch_grpo: init=$ADAPTER_INIT → $HUB_REPO ==="
echo "  steps=$GRPO_STEPS G=$G prompts/step=$PROMPTS_PER_STEP lr=$GRPO_LR"
echo "  kl_beta=$KL_BETA kl_reference=$KL_REFERENCE temperature=$TEMPERATURE"

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=80 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1
echo "offer: $OFFER_ID"

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-grpo-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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

echo "uploading code + GRPO data + eval problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --include='*.txt' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_grpo_train.jsonl" \
    "$LOCAL/data/problems_grpo_eval.jsonl" \
    "$LOCAL/data/problems_lcb.jsonl" \
    "$LOCAL/data/problems_mbpp30.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching GRPO + cross-benchmark eval pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       ADAPTER_INIT='$ADAPTER_INIT' \
       HUB_REPO='$HUB_REPO' \
       GRPO_TRAIN_PROBLEMS='../data/problems_grpo_train.jsonl' \
       GRPO_EVAL_PROBLEMS='../data/problems_grpo_eval.jsonl' \
       GRPO_LR='$GRPO_LR' GRPO_STEPS='$GRPO_STEPS' \
       G='$G' PROMPTS_PER_STEP='$PROMPTS_PER_STEP' \
       TEMPERATURE='$TEMPERATURE' KL_BETA='$KL_BETA' KL_REFERENCE='$KL_REFERENCE' \
       MAX_LENGTH='$MAX_LENGTH' MAX_NEW_TOKENS='$MAX_NEW_TOKENS' \
       EVAL_EVERY='$EVAL_EVERY' EVAL_N='$EVAL_N' \
       ABORT_VAR_FLOOR='$ABORT_VAR_FLOOR' ABORT_VAR_CHECK_STEP='$ABORT_VAR_CHECK_STEP' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"=== Stage A: GRPO training (continue from $ADAPTER_INIT) ===\"
         python -u grpo_train.py 2>&1 | tee /workspace/grpo_train.log

         if [ \"$RUN_FULL_EVAL\" = \"1\" ]; then
           echo \"=== Stage B1: GRPO adapter, BARE prompt, LCB (limit=$LCB_LIMIT) ===\"
           python -u sweep_local.py \
             --problems ../data/problems_lcb.jsonl \
             --csv ../results/raw/grpo_eval_lcb_bare.csv \
             --source-dir ../results/raw/sources_grpo_eval_lcb_bare \
             --base-model \"\$BASE_MODEL\" \
             --adapter ../outputs/grpo_run1/final_adapter \
             --n-samples $N_SAMPLES_EVAL \
             --max-new-tokens \"\$MAX_NEW_TOKENS\" \
             --limit-problems $LCB_LIMIT \
             --constraints 2>&1 | tee /workspace/grpo_eval_lcb.log

           echo \"=== Stage B2: GRPO adapter, BARE prompt, MBPP-30 ===\"
           python -u sweep_local.py \
             --problems ../data/problems_mbpp30.jsonl \
             --csv ../results/raw/grpo_eval_mbpp_bare.csv \
             --source-dir ../results/raw/sources_grpo_eval_mbpp_bare \
             --base-model \"\$BASE_MODEL\" \
             --adapter ../outputs/grpo_run1/final_adapter \
             --n-samples $N_SAMPLES_EVAL \
             --max-new-tokens \"\$MAX_NEW_TOKENS\" \
             --constraints 2>&1 | tee /workspace/grpo_eval_mbpp.log
         fi

         touch /workspace/all_done
       ' \
       > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "  Pipeline: ssh -p $PORT root@$HOST 'tail -f /workspace/pipeline.log'"
