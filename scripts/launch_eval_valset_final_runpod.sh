#!/bin/bash
# -final on the SFT val set (12 problems used for early-stopping but not training).
# Pairs with the existing -bestval val-set number (34/93 = 0.37) to fill in the val cell
# of the 2x3 bestval/final x train/val/clean grid. ~96 gens, expected <1.5h on Secure A100.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_eval_valset_final.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
N_SAMPLES="${N_SAMPLES:-8}"
ADAPTER="${ADAPTER:-kilojoules/surface-tension-sft-rankcurve-r32-final}"
TAG="${TAG:-valset_final}"
PROBLEMS_FILE="${PROBLEMS_FILE:-problems_lcb_sfteval.jsonl}"

echo "=== launch_eval_valset_final_runpod (gpu=$GPU cloud=$CLOUD; $ADAPTER on $PROBLEMS_FILE, n=$N_SAMPLES) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-eval-valset-final \
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
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x; deps install failed\")' && python3 -c 'import transformers, peft, bitsandbytes' && python3 -c 'import torch,transformers,peft,bitsandbytes as b; print(\"torch\",torch.__version__,\"| cuda\",torch.cuda.is_available(),\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); print(f\"GPU matmul-50x: {time.time()-t:.2f}s (expect <2s healthy A100)\")'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + val-set problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/$PROBLEMS_FILE" "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching eval pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         echo \"========== eval $TAG (adapter=$ADAPTER, problems=$PROBLEMS_FILE) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/$PROBLEMS_FILE \
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
echo "=== LAUNCHED on RunPod ($TAG: $ADAPTER on $PROBLEMS_FILE, n=$N_SAMPLES) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=15 WATCHDOG_MAX_HOURS=3 bash $LOCAL/scripts/watchdog.sh"
