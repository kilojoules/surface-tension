#!/bin/bash
# Resume-style DPO: skip sampling (pairs file is uploaded from local), go straight
# to dpo_train + push + eval. Used when a prior pod died mid-pipeline and we want
# to reuse the pairs already sampled.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_dpo_resume.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
ADAPTER_INIT="${ADAPTER_INIT:?ADAPTER_INIT required (e.g. kilojoules/surface-tension-sft-rationale-stripped-r32-final)}"
HUB_REPO="${HUB_REPO:?HUB_REPO required (e.g. kilojoules/surface-tension-dpo-from-stripped-r32)}"
PAIRS_LOCAL="${PAIRS_LOCAL:?PAIRS_LOCAL required (e.g. vast_logs/.../data/dpo_pairs_rstripped.jsonl)}"
TAG="${TAG:-resume}"

[ -f "$PAIRS_LOCAL" ] || { echo "FATAL: pairs file not found at $PAIRS_LOCAL"; exit 1; }
N_PAIRS=$(wc -l < "$PAIRS_LOCAL")
echo "=== launch_dpo_from_pairs ($N_PAIRS pairs from $PAIRS_LOCAL; init=$ADAPTER_INIT hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-dpo-$TAG" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 150 --wait-min 22 || exit 1

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
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x\")'
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + pairs + eval problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PAIRS_LOCAL" "root@$HOST:/workspace/st/data/dpo_pairs_${TAG}.jsonl" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching DPO ($TAG) — skipping sampling, going straight to train+push+eval..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== STAGE 1: DPO train ($N_PAIRS pairs, reference = $ADAPTER_INIT) ==========\"
         wc -l ../data/dpo_pairs_${TAG}.jsonl
         ADAPTER_INIT=$ADAPTER_INIT \
         DPO_TRAIN=../data/dpo_pairs_${TAG}.jsonl \
         DPO_OUTPUT=../outputs/dpo_${TAG} \
         DPO_BETA=0.1 DPO_LR=5e-6 DPO_EPOCHS=3 \
         MAX_LENGTH=2048 \
           python -u dpo_train.py 2>&1 | tee /workspace/dpo_train_${TAG}.log
         [ -d ../outputs/dpo_${TAG}/final_adapter ] || { echo \"FATAL: no final_adapter\"; exit 1; }

         echo \"========== push adapter to Hub ($HUB_REPO-final) ==========\"
         timeout 900 python -u push_adapter.py ../outputs/dpo_${TAG}/final_adapter ${HUB_REPO}-final 2>&1 | tee /workspace/push_dpo_${TAG}.log || echo \"WARN: push failed\"

         echo \"========== STAGE 2a: eval val (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_dpo_${TAG}_val.csv --source-dir ../results/raw/sources_eval_dpo_${TAG}_val \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_dpo_${TAG}_val.log
         python -u recheck_eval.py ../results/raw/eval_dpo_${TAG}_val.csv 2>&1 | tee /workspace/recheck_${TAG}_val.txt

         echo \"========== STAGE 2b: eval clean (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_dpo_${TAG}_clean.csv --source-dir ../results/raw/sources_eval_dpo_${TAG}_clean \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_dpo_${TAG}_clean.log
         python -u recheck_eval.py ../results/raw/eval_dpo_${TAG}_clean.csv 2>&1 | tee /workspace/recheck_${TAG}_clean.txt

         cp ../results/raw/eval_dpo_${TAG}_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (DPO-from-pairs $TAG: $N_PAIRS pairs) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=14 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl --exclude=dpo_pairs_${TAG}.jsonl\" WATCHDOG_LOG=/tmp/watchdog_dpo_${TAG}.log bash $LOCAL/scripts/watchdog.sh"
