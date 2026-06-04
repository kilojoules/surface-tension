#!/bin/bash
# Resume the BASE-MODEL (no adapter) eval on clean-17 after the prior pod was
# killed mid-eval (watchdog max_hours hit). Uploads the partial CSV + sources;
# sweep_local picks up from the next missing key. H100 for reliable throughput.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_eval_base_resume.env"
GPU="${GPU:-NVIDIA H100 80GB HBM3}"
CLOUD="${CLOUD:-SECURE}"
PARTIAL_CSV="$LOCAL/vast_logs/7fjlynzsv7yiec/st/results/raw/eval_base_clean17.csv"
PARTIAL_SOURCES="$LOCAL/vast_logs/7fjlynzsv7yiec/st/results/raw/sources_eval_base_clean17"
TAG="base_clean17"

[ -f "$PARTIAL_CSV" ] || { echo "FATAL: partial CSV not found at $PARTIAL_CSV"; exit 1; }
[ -d "$PARTIAL_SOURCES" ] || { echo "FATAL: partial sources dir not found at $PARTIAL_SOURCES"; exit 1; }
N_ALREADY=$(($(wc -l < "$PARTIAL_CSV") - 1))
echo "=== launch_eval_base_resume (gpu=$GPU cloud=$CLOUD; $N_ALREADY rows already done; tag=$TAG) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-eval-base-resume \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 130 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: pod up but SSH not responding"; python3 "$LOCAL/scripts/runpod_kill.py" "$INST"; exit 1; }

echo "installing deps + GPU sanity check..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__)'
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); s=time.time()-t; print(f\"GPU matmul-50x: {s:.2f}s\"); assert s < 2.5, f\"FATAL slow GPU silicon (matmul-50x={s:.2f}s > 2.5s threshold) — kill + relaunch\"'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problems + partial CSV + partial sources..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "mkdir -p /workspace/st/{src,data,results/raw/sources_eval_$TAG}"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" "root@$HOST:/workspace/st/data/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PARTIAL_CSV" "root@$HOST:/workspace/st/results/raw/eval_$TAG.csv"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PARTIAL_SOURCES/" "root@$HOST:/workspace/st/results/raw/sources_eval_$TAG/"

echo "launching eval pipeline (resume from $N_ALREADY rows, no adapter)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         echo \"========== eval $TAG resume (base, no adapter) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_$TAG.csv --source-dir ../results/raw/sources_eval_$TAG \
           --base-model google/gemma-4-31B-it \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
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
echo "=== LAUNCHED on RunPod (base resume: $N_ALREADY rows uploaded, will finish remaining) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=10 WATCHDOG_STALL_MIN=90 WATCHDOG_MAX_HOURS=4 bash $LOCAL/scripts/watchdog.sh"
