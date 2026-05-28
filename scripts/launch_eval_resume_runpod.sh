#!/bin/bash
# Resume an interrupted sweep_local eval by uploading the partial CSV + sources to
# a fresh pod and letting sweep_local's built-in resume support pick up where the
# prior pod died. Useful when an external pod-kill cost us mid-eval progress.
#
# Required env:
#   ADAPTER          Hub path of the adapter to eval (e.g. kilojoules/...-final)
#   PARTIAL_CSV      local path to the partial CSV synced from the dead pod
#   PARTIAL_SOURCES  local path to the partial source-dir synced from the dead pod
#   PROBLEMS         local data/problems_*.jsonl matching the original eval (val or clean)
#   TAG              tag for new csv/source-dir on the pod (e.g. dpo_stripped_clean)
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_eval_resume.env"
GPU="${GPU:-NVIDIA H100 80GB HBM3}"
CLOUD="${CLOUD:-SECURE}"
ADAPTER="${ADAPTER:?ADAPTER required}"
PARTIAL_CSV="${PARTIAL_CSV:?PARTIAL_CSV required}"
PARTIAL_SOURCES="${PARTIAL_SOURCES:?PARTIAL_SOURCES required (local dir)}"
PROBLEMS="${PROBLEMS:?PROBLEMS required (local jsonl path)}"
TAG="${TAG:?TAG required}"

[ -f "$PARTIAL_CSV" ] || { echo "FATAL: partial CSV not found at $PARTIAL_CSV"; exit 1; }
[ -d "$PARTIAL_SOURCES" ] || { echo "FATAL: partial sources dir not found at $PARTIAL_SOURCES"; exit 1; }
PROBLEMS_BASENAME=$(basename "$PROBLEMS")
N_ALREADY=$(($(wc -l < "$PARTIAL_CSV") - 1))
echo "=== launch_eval_resume (gpu=$GPU cloud=$CLOUD; $N_ALREADY rows already done; adapter=$ADAPTER tag=$TAG) ==="

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

echo "uploading src + problems + partial CSV + partial sources..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PROBLEMS" "root@$HOST:/workspace/st/data/$PROBLEMS_BASENAME" 2>/dev/null
# Upload partial CSV + sources to the *exact paths* sweep_local will pick up
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PARTIAL_CSV" "root@$HOST:/workspace/st/results/raw/eval_${TAG}.csv" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PARTIAL_SOURCES/" "root@$HOST:/workspace/st/results/raw/sources_eval_${TAG}/" 2>/dev/null

echo "verify partial data on pod..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "echo CSV rows: \$(($(wc -l < /workspace/st/results/raw/eval_${TAG}.csv) - 1))
     echo source files: \$(ls /workspace/st/results/raw/sources_eval_${TAG}/ | wc -l)"

# Pod-speed probe — kill fast if matmul is slow
echo "GPU speed probe..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); print(f\"matmul-50x: {time.time()-t:.2f}s\")'"

echo "launching resume eval ($N_ALREADY rows already in CSV, will skip and continue)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== resume eval ($TAG): $N_ALREADY rows already present ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/$PROBLEMS_BASENAME \
           --csv ../results/raw/eval_${TAG}.csv --source-dir ../results/raw/sources_eval_${TAG} \
           --base-model google/gemma-4-31B-it --adapter $ADAPTER \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_${TAG}.log
         python -u recheck_eval.py ../results/raw/eval_${TAG}.csv 2>&1 | tee /workspace/recheck_${TAG}.txt
         cp ../results/raw/eval_${TAG}.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED resume eval $TAG (adapter=$ADAPTER, already done=$N_ALREADY) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=150 WATCHDOG_MAX_HOURS=12 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=$PROBLEMS_BASENAME\" WATCHDOG_LOG=/tmp/watchdog_eval_resume.log bash $LOCAL/scripts/watchdog.sh"
