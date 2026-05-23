#!/bin/bash
# Mine more SFT pairs by running constrained-prompt sweep on the expanded
# problem set (HumanEval + MBPP-sanitized full + LCB). Pipeline:
#
#   Stage A: build_expanded_problems.py  → data/problems_expanded.jsonl
#   Stage B: sweep_local.py --skip-bare --constraints no_loops_no_recursion
#            → results/raw/datagen_expanded_raw.csv + sources_*
#   Stage C: build_sft_dataset.py → data/expanded/sft_{train,eval,all}.jsonl
#   Stage D: optionally merge with existing 146 to write data/sft_full.jsonl
#
# Usage:
#   bash scripts/launch_datagen.sh [gpu_filter]

set -e
GPU_FILTER="${1:-A100_SXM4}"

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
N_SAMPLES="${N_SAMPLES:-2}"            # constrained samples per problem
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LIMIT_PROBLEMS="${LIMIT_PROBLEMS:-0}"  # 0 = no limit; >0 for smoke tests

echo "=== launch_datagen on $GPU_FILTER (n_samples=$N_SAMPLES) ==="

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
    --label "st-datagen-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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

echo "uploading code + held-out set..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --include='*.txt' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
# Upload held-out file so build_expanded_problems.py can exclude it.
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_mbpp30.jsonl" \
    "$LOCAL/data/problems_lcb.jsonl" \
    "$LOCAL/data/sft_all.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching datagen pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       MAX_NEW_TOKENS='$MAX_NEW_TOKENS' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"=== Stage A: build expanded problems ===\"
         python -u build_expanded_problems.py 2>&1 | tee /workspace/datagen_build.log

         echo \"=== Stage B: constrained sweep on expanded set ===\"
         python -u sweep_local.py \
           --problems ../data/problems_expanded.jsonl \
           --csv ../results/raw/datagen_expanded_raw.csv \
           --source-dir ../results/raw/sources_datagen_expanded \
           --base-model google/gemma-4-31B-it \
           --n-samples $N_SAMPLES \
           --max-new-tokens \"\$MAX_NEW_TOKENS\" \
           --limit-problems $LIMIT_PROBLEMS \
           --constraints no_loops_no_recursion \
           --skip-bare 2>&1 | tee /workspace/datagen_sweep.log

         echo \"=== Stage C: build SFT pool from new sweep ===\"
         python -u build_sft_dataset.py \
           --csv ../results/raw/datagen_expanded_raw.csv \
           --sources-dir ../results/raw/sources_datagen_expanded \
           --problems ../data/problems_expanded.jsonl \
           --out-dir ../data/expanded \
           --holdout-frac 0.0 2>&1 | tee /workspace/datagen_buildsft.log

         echo \"=== Stage D: merge with existing 146 ===\"
         python -u -c \"
import json
existing = [json.loads(l) for l in open(\\\"../data/sft_all.jsonl\\\")]
new = [json.loads(l) for l in open(\\\"../data/expanded/sft_all.jsonl\\\")]
seen_codes = {(e[\\\"problem_id\\\"], e[\\\"completion\\\"]) for e in existing}
merged = list(existing)
added = 0
for n in new:
    key = (n[\\\"problem_id\\\"], n[\\\"completion\\\"])
    if key in seen_codes: continue
    seen_codes.add(key); merged.append(n); added += 1
with open(\\\"../data/sft_full.jsonl\\\", \\\"w\\\") as f:
    for m in merged:
        f.write(json.dumps(m) + \\\"\\\n\\\")
print(f\\\"merged: existing={len(existing)} + new={len(new)} (added={added}) = {len(merged)}\\\")
\" 2>&1 | tee /workspace/datagen_merge.log

         touch /workspace/all_done
       ' \
       > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "  Pipeline: ssh -p $PORT root@$HOST 'tail -f /workspace/pipeline.log'"
echo ""
echo "Next: WATCHDOG_MAX_HOURS=10 bash scripts/watchdog.sh   # in another terminal"
