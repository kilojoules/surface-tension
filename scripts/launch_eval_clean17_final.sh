#!/bin/bash
# Clean-17 eval for rankcurve-r32-FINAL (the over-fit, epoch-20 checkpoint) — the experiment
# that answers "did we deploy the right checkpoint?" Same setup as the -bestval run for direct
# comparability (max_new_tokens=1024, n=8, T=0.7, bare prompt). 136 generations, ~$3-4 on a
# reliability-filtered vast A100. Pipeline auto-runs recheck → all_done → watchdog destroys.
set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"
RELIABILITY="${RELIABILITY:-0.99}"
DPH_MAX="${DPH_MAX:-1.30}"
N_SAMPLES="${N_SAMPLES:-8}"
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-130}"
INSTANCE_FILE="$LOCAL/vast_eval_clean17_final.env"
ADAPTER="${ADAPTER:-kilojoules/surface-tension-sft-rankcurve-r32-final}"
TAG="${TAG:-clean17_final}"

echo "=== launch_eval_clean17_final (vast --direct, reliability>$RELIABILITY; $ADAPTER on clean-17, n=$N_SAMPLES) ==="

if [ -n "${OFFER_ID:-}" ]; then
    echo "offer: $OFFER_ID (explicit override)"
else
    OFFER_ID=$(echo "n" | vastai search offers \
        "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=110 reliability>$RELIABILITY" \
        --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin); print(d[0]['id'] if d else '')
")
    [ -z "$OFFER_ID" ] && echo "no offers" && exit 1
    echo "offer: $OFFER_ID"
fi

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-eval-clean17-final-noautokill" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && { echo "rent failed: $RESULT"; exit 1; }
echo "instance: $INST"

for i in $(seq 1 50); do
    STATUS=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    sleep 15
done
[ "$STATUS" != "running" ] && { echo "FAIL"; echo "n" | vastai destroy instance "$INST"; exit 1; }

SSH_INFO=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo "$SSH_INFO" | cut -d' ' -f1); PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (instance $INST)"

for i in $(seq 1 25); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: SSH"; echo "n" | vastai destroy instance "$INST"; exit 1; }

echo "installing deps + GPU sanity check..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "pip install -q transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,peft,bitsandbytes as b; print(\"torch\",torch.__version__,\"| cuda\",torch.cuda.is_available(),\"| transformers\",transformers.__version__,\"peft\",peft.__version__,\"bnb\",b.__version__)'
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); print(f\"GPU matmul-50x: {time.time()-t:.2f}s (expect <2s on a healthy A100)\")'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + clean-17 problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching eval pipeline ($TAG only)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         echo \"========== eval $TAG (adapter=$ADAPTER) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_$TAG.csv --source-dir ../results/raw/sources_eval_$TAG \
           --base-model google/gemma-4-31B-it --adapter $ADAPTER \
           --n-samples $N_SAMPLES --max-new-tokens 1024 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_$TAG.log
         cp ../results/raw/eval_$TAG.csv /workspace/ 2>/dev/null || true
         echo \"========== re-checking compliance from sources ==========\"
         python -u recheck_eval.py ../results/raw/eval_$TAG.csv 2>&1 | tee /workspace/recheck_summary.txt
         cp ../results/raw/recheck_summary.json /workspace/ 2>/dev/null || true
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"eval launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on vast.ai ($TAG, $N_SAMPLES samples × 17 problems = 136 gens) ==="
echo "  Instance: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=vast WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=10 WATCHDOG_MAX_HOURS=6 bash $LOCAL/scripts/watchdog.sh"
