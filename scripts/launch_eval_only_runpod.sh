#!/bin/bash
# Eval-only: load a Hub adapter and run val (12) + clean (17) sweeps.
# Used when an adapter is already pushed but eval needs to run on a different pod
# (e.g. previous pod was slow or died after push).
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_eval_only.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
ADAPTER="${ADAPTER:?ADAPTER required (Hub path, e.g. kilojoules/...-final)}"
TAG="${TAG:-evalonly}"

echo "=== launch_eval_only (gpu=$GPU cloud=$CLOUD; adapter=$ADAPTER tag=$TAG) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-eval-$TAG" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 130 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__)'
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + eval problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

# Pod-speed probe: time 50 matmuls. If >5s, kill and retry on another pod.
echo "GPU speed probe..."
SPEED=$(ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); print(round(time.time()-t,2))'" 2>&1 | tail -1)
echo "  GPU matmul-50x: ${SPEED}s (expect <2s healthy A100)"
if [ "$(echo "$SPEED > 5" | bc 2>/dev/null)" = "1" ]; then
    echo "FATAL: GPU too slow (${SPEED}s) — abort + destroy"
    python3 "$LOCAL/scripts/runpod_kill.py" "$INST"
    exit 2
fi

echo "launching eval pipeline (val + clean)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== eval val (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_${TAG}_val.csv --source-dir ../results/raw/sources_eval_${TAG}_val \
           --base-model google/gemma-4-31B-it --adapter $ADAPTER \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_${TAG}_val.log
         python -u recheck_eval.py ../results/raw/eval_${TAG}_val.csv 2>&1 | tee /workspace/recheck_${TAG}_val.txt

         echo \"========== eval clean (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_${TAG}_clean.csv --source-dir ../results/raw/sources_eval_${TAG}_clean \
           --base-model google/gemma-4-31B-it --adapter $ADAPTER \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_${TAG}_clean.log
         python -u recheck_eval.py ../results/raw/eval_${TAG}_clean.csv 2>&1 | tee /workspace/recheck_${TAG}_clean.txt

         cp ../results/raw/eval_${TAG}_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (eval-only $TAG: $ADAPTER) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=8 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl\" WATCHDOG_LOG=/tmp/watchdog_eval_${TAG}.log bash $LOCAL/scripts/watchdog.sh"
