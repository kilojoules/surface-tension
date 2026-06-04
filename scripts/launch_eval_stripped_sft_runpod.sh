#!/bin/bash
# Eval SFT-stripped (rationale-stripped) adapter on val (12) + clean (17) held-outs.
# Bare-prompt only (--constraints with no values), n=8, max_new=3072.
# Fills the gap so we can do the rationale-SFT vs rationale-stripped comparison
# rigorously (rechecked from saved sources) for the rationale_summary figure.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_eval_stripped_sft.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
ADAPTER="${ADAPTER:-kilojoules/surface-tension-sft-rationale-stripped-r32-final}"

echo "=== launch_eval_stripped_sft (gpu=$GPU cloud=$CLOUD; $ADAPTER on val+clean) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-eval-stripped-sft \
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

echo "uploading src + problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/"

echo "launching eval pipeline (val + clean)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         echo \"========== eval stripped val (12 problems) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_stripped_sft_val.csv --source-dir ../results/raw/sources_eval_stripped_sft_val \
           --base-model google/gemma-4-31B-it --adapter $ADAPTER \
           --n-samples $N_SAMPLES --max-new-tokens $MAX_NEW_TOKENS --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_stripped_sft_val.log
         python -u recheck_eval.py ../results/raw/eval_stripped_sft_val.csv 2>&1 | tee /workspace/recheck_stripped_val.txt
         cp ../results/raw/eval_stripped_sft_val.csv /workspace/ 2>/dev/null || true

         echo \"========== eval stripped clean (17 problems) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_stripped_sft_clean.csv --source-dir ../results/raw/sources_eval_stripped_sft_clean \
           --base-model google/gemma-4-31B-it --adapter $ADAPTER \
           --n-samples $N_SAMPLES --max-new-tokens $MAX_NEW_TOKENS --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_stripped_sft_clean.log
         python -u recheck_eval.py ../results/raw/eval_stripped_sft_clean.csv 2>&1 | tee /workspace/recheck_stripped_clean.txt
         cp ../results/raw/eval_stripped_sft_clean.csv /workspace/ 2>/dev/null || true

         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"eval launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (stripped-SFT eval val+clean) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=10 WATCHDOG_STALL_MIN=90 WATCHDOG_MAX_HOURS=8 bash $LOCAL/scripts/watchdog.sh"
