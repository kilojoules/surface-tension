#!/bin/bash
# RunPod eval: bare-prompt compliance of the rank-curve val-min checkpoint vs the over-fit
# final, split by in-training (28 problems, n=2) vs held-out (12 problems, n=8). Re-derives
# compliance from sources at the end (the CSV `compliant` column is unreliable).
# Adapters already on the Hub. Watchdog auto-destroys the pod on completion.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_eval_bestval.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-COMMUNITY}"
N_HELDOUT="${N_HELDOUT:-8}"
N_INTRAIN="${N_INTRAIN:-2}"

echo "=== launch_eval_bestval_runpod (gpu=$GPU cloud=$CLOUD; held-out n=$N_HELDOUT, in-train n=$N_INTRAIN) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-eval-bestval \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 120 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: pod up but SSH not responding"; python3 "$LOCAL/scripts/runpod_kill.py" "$INST"; exit 1; }

echo "installing deps (rsync + torch 2.5.1 + transformers/peft/bnb latest)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,peft,bitsandbytes as b; print(\"torch\",torch.__version__,\"| cuda_avail\",torch.cuda.is_available(),\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problem splits..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_sfttrain.jsonl" "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching eval pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         eval_one() {  # \$1=adapter_repo  \$2=tag  \$3=problems_file  \$4=n_samples
           echo \"========== eval \$2 (adapter=\$1, problems=\$3, n=\$4) ==========\"
           LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
             --problems ../data/\$3 \
             --csv ../results/raw/eval_\$2.csv \
             --source-dir ../results/raw/sources_eval_\$2 \
             --base-model google/gemma-4-31B-it --adapter \$1 \
             --n-samples \$4 --max-new-tokens 1024 --temperature 0.7 --constraints \
             2>&1 | tee /workspace/eval_\$2.log
           cp ../results/raw/eval_\$2.csv /workspace/ 2>/dev/null || true
           touch /workspace/done_\$2
         }
         eval_one kilojoules/surface-tension-sft-rankcurve-r32-bestval r32bestval_heldout problems_lcb_sfteval.jsonl  $N_HELDOUT
         eval_one kilojoules/surface-tension-sft-rankcurve-r32-bestval r32bestval_intrain problems_lcb_sfttrain.jsonl $N_INTRAIN
         eval_one kilojoules/surface-tension-sft-rankcurve-r32-final   r32final_heldout   problems_lcb_sfteval.jsonl  $N_HELDOUT
         eval_one kilojoules/surface-tension-sft-rankcurve-r32-final   r32final_intrain   problems_lcb_sfttrain.jsonl $N_INTRAIN
         echo \"========== re-checking compliance from sources ==========\"
         python -u recheck_eval.py ../results/raw/eval_r32bestval_heldout.csv ../results/raw/eval_r32bestval_intrain.csv ../results/raw/eval_r32final_heldout.csv ../results/raw/eval_r32final_intrain.csv 2>&1 | tee /workspace/recheck_summary.txt
         cp ../results/raw/recheck_summary.json /workspace/ 2>/dev/null || true
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"eval launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (bestval/final compliance eval) ==="
echo "  Pod: $INST"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=10 WATCHDOG_MAX_HOURS=10 bash $LOCAL/scripts/watchdog.sh"
