#!/bin/bash
# Cross-benchmark eval for the 15 ablation adapters. Loops over a list of
# HF Hub repos, runs sweep_local.py with each as adapter on LCB-30 + MBPP-30
# bare-prompt eval. CSVs land in results/raw/eval_batch_<repo_tag>_*.csv.
#
# Usage:
#   ADAPTER_REPOS="repo1 repo2 ..." bash scripts/launch_eval_batch.sh
#
# Or the typical 15-adapter ablation:
#   PREFIXES="st-sft-n st-rl_only-n st-sft_rl-n" \
#   N_VALUES="50 100 200 300 500" \
#   SEED=0 \
#   bash scripts/launch_eval_batch.sh

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

# Build adapter list
if [ -n "${ADAPTER_REPOS:-}" ]; then
    REPOS="$ADAPTER_REPOS"
else
    PREFIXES="${PREFIXES:-st-sft-n st-rl_only-n st-sft_rl-n}"
    N_VALUES="${N_VALUES:-50 100 200 300 500}"
    HUB_USER="${HUB_USER:-kilojoules}"
    SEED="${SEED:-0}"
    REPOS=""
    for p in $PREFIXES; do
        for n in $N_VALUES; do
            REPOS+=" ${HUB_USER}/${p}${n}-s${SEED}"
        done
    done
fi
N_SAMPLES="${N_SAMPLES:-2}"
LCB_LIMIT="${LCB_LIMIT:-30}"
RUN_LCB="${RUN_LCB:-1}"
RUN_MBPP="${RUN_MBPP:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

echo "=== launch_eval_batch ==="
echo "$REPOS" | tr ' ' '\n' | grep -v '^$' | nl

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
    --label "st-eval-batch-noautokill" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && { echo "rent failed"; exit 1; }

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

ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "pip install -q transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb.jsonl" \
    "$LOCAL/data/problems_mbpp30.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

LOOP_BODY=""
for repo in $REPOS; do
    tag=$(echo "$repo" | tr '/' '_' | tr -c '[:alnum:]_' '_')
    if [ "$RUN_LCB" = "1" ]; then
        LOOP_BODY+="        echo \"=== eval $repo on LCB ===\"
        python -u sweep_local.py \
          --problems ../data/problems_lcb.jsonl \
          --csv ../results/raw/eval_${tag}_lcb.csv \
          --source-dir ../results/raw/sources_eval_${tag}_lcb \
          --base-model google/gemma-4-31B-it \
          --adapter '$repo' \
          --n-samples $N_SAMPLES \
          --max-new-tokens $MAX_NEW_TOKENS \
          --limit-problems $LCB_LIMIT \
          --constraints 2>&1 | tee /workspace/eval_${tag}_lcb.log || echo \"  (cell failed but continuing)\"
"
    fi
    if [ "$RUN_MBPP" = "1" ]; then
        LOOP_BODY+="        echo \"=== eval $repo on MBPP-30 ===\"
        python -u sweep_local.py \
          --problems ../data/problems_mbpp30.jsonl \
          --csv ../results/raw/eval_${tag}_mbpp.csv \
          --source-dir ../results/raw/sources_eval_${tag}_mbpp \
          --base-model google/gemma-4-31B-it \
          --adapter '$repo' \
          --n-samples $N_SAMPLES \
          --max-new-tokens $MAX_NEW_TOKENS \
          --constraints 2>&1 | tee /workspace/eval_${tag}_mbpp.log || echo \"  (cell failed but continuing)\"
"
    fi
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
echo "Next: WATCHDOG_MAX_HOURS=14 bash scripts/watchdog.sh"
