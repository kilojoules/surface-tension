#!/bin/bash
# Launch a DPO+eval run on a single 4090 vast.ai instance, modeled on turnstile's
# launch_focused_dpo.sh + run_seed.sh. The "stack of fragile abstractions" lessons:
#   - rsync code from local (NOT git clone on box; avoids HF dataset loader breakage)
#   - install minimal pip deps unpinned (no vllm, no trl)
#   - 4090 not H100 (cheaper, more available)
#   - --direct ssh
#   - watchdog runs locally with auto-destroy on completion
#
# Usage: bash scripts/launch_dpo.sh [base_model] [hub_repo]
#   defaults: google/gemma-4-31B-it   kilojoules/surface-tension-dpo

set -e
BASE_MODEL="${1:-google/gemma-4-31B-it}"
HUB_REPO="${2:-kilojoules/surface-tension-dpo}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
INSTANCE_FILE="$LOCAL/vast_current.env"

echo "=== launch_dpo: $BASE_MODEL → $HUB_REPO ==="

# 1. Find a cheap reliable 4090
OFFER_ID=$(echo "n" | vastai search offers \
    'gpu_name=RTX_4090 num_gpus=1 dph<0.50 inet_down>500 disk_space>=80 reliability>0.998' \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1
echo "offer: $OFFER_ID"

# 2. Rent
RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-dpo" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "instance: $INST"

# 3. Wait for actual_status=running
for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    echo "  status: $STATUS ($i/40)"
    sleep 15
done
if [ "$STATUS" != "running" ]; then
    echo "FAIL — destroying $INST"
    echo "n" | vastai destroy instance "$INST" 2>&1 | grep -v Update
    exit 1
fi

# 4. SSH info → instance file
SSH_INFO=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST"

# 5. Wait for SSH
for i in $(seq 1 20); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

# 6. Install minimal deps. Note: NO trl, NO vllm.
echo "installing deps..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "pip install -q transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true"

# 7. HF auth (token from local file)
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

# 8. rsync code + DPO pairs + LCB problems
echo "uploading code, pairs, problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --include='*.txt' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/dpo_pairs_train.jsonl" \
    "$LOCAL/data/dpo_pairs_eval.jsonl" \
    "$LOCAL/data/dpo_pairs_all.jsonl" \
    "$LOCAL/data/problems_lcb.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

# 9. Launch the DPO+eval pipeline. nohup so we can disconnect.
echo "launching pipeline on box..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='$BASE_MODEL' HUB_REPO='$HUB_REPO' \
       bash -c 'python -u dpo_train.py 2>&1 | tee /workspace/dpo_train.log
                python -u sweep_local.py \
                  --problems ../data/problems_lcb.jsonl \
                  --csv ../results/raw/pilot_v6_dpo_raw.csv \
                  --source-dir ../results/raw/sources_v6_dpo \
                  --base-model \"\$BASE_MODEL\" \
                  --adapter ../outputs/dpo_run1/final_adapter \
                  --n-samples 3 \
                  --constraints no_loops_no_recursion 2>&1 | tee /workspace/sweep_local.log
                python -u aggregate.py \
                  --csv ../results/raw/pilot_v6_dpo_raw.csv \
                  --summary-csv ../results/pilot_v6_dpo_summary.csv \
                  --summary-md ../results/pilot_v6_dpo_summary.md \
                  --sources-dir ../results/raw/sources_v6_dpo 2>&1 | tee /workspace/aggregate.log
                touch /workspace/all_done' \
       > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "instance file: $INSTANCE_FILE"
echo "next: bash scripts/watchdog.sh   # in this terminal — destroys instance on completion"
echo "tail: ssh -p $PORT root@$HOST 'tail -f /workspace/dpo_train.log'"
