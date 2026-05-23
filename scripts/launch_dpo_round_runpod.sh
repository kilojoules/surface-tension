#!/bin/bash
# One round of the iterated DPO alignment loop.
# Pipeline:
#   1. build_dpo_pairs: sample the current policy (ADAPTER_INIT) bare-prompt on 28 train
#      problems, label with AST checker + tests, form (compliant ≻ violating) pairs.
#   2. dpo_train: DPO from ADAPTER_INIT, reference anchored to ADAPTER_INIT (B1++).
#   3. Push round adapter to Hub.
#   4. Eval val (12) + clean (17) at max_new=3072.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_dpo_round.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
ROUND="${ROUND:-1}"
ADAPTER_INIT="${ADAPTER_INIT:-kilojoules/surface-tension-sft-b1plus-r32-final}"
HUB_REPO="${HUB_REPO:-kilojoules/surface-tension-dpo-r${ROUND}-r32}"

echo "=== launch_dpo_round (round=$ROUND gpu=$GPU cloud=$CLOUD; init=$ADAPTER_INIT hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-dpo-r${ROUND} \
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
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x; deps install failed\")' && python3 -c 'import transformers, peft, bitsandbytes' && python3 -c 'import torch,transformers,peft,bitsandbytes as b; print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problem data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_dpopool.jsonl" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching DPO round-$ROUND pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== STAGE 1: build DPO pairs (sample $ADAPTER_INIT on fresh 45-problem pool) ==========\"
         python -u build_dpo_pairs.py \
           --problems ../data/problems_lcb_dpopool.jsonl \
           --out ../data/dpo_pairs_r${ROUND}.jsonl \
           --adapter $ADAPTER_INIT \
           --n-samples 8 --max-new-tokens 2048 --temperature 0.9 --max-pairs 6 \
           2>&1 | tee /workspace/dpo_pairs_r${ROUND}.log
         [ -s ../data/dpo_pairs_r${ROUND}.jsonl ] || { echo \"FATAL: empty pairs file\"; exit 1; }
         cp ../data/dpo_pairs_r${ROUND}.jsonl /workspace/
         wc -l ../data/dpo_pairs_r${ROUND}.jsonl

         echo \"========== STAGE 2: DPO train (reference = $ADAPTER_INIT) ==========\"
         ADAPTER_INIT=$ADAPTER_INIT \
         DPO_TRAIN=../data/dpo_pairs_r${ROUND}.jsonl \
         DPO_OUTPUT=../outputs/dpo_r${ROUND} \
         DPO_BETA=0.1 DPO_LR=5e-6 DPO_EPOCHS=3 \
         MAX_LENGTH=2048 \
           python -u dpo_train.py 2>&1 | tee /workspace/dpo_train_r${ROUND}.log
         [ -d ../outputs/dpo_r${ROUND}/final_adapter ] || { echo \"FATAL: no final_adapter\"; exit 1; }

         echo \"========== push round-$ROUND adapter to Hub ==========\"
         timeout 900 python -u push_adapter.py ../outputs/dpo_r${ROUND}/final_adapter ${HUB_REPO}-final 2>&1 | tee /workspace/push_dpo_r${ROUND}.log || echo \"WARN: push failed\"

         echo \"========== STAGE 3a: eval val (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_dpo_r${ROUND}_val.csv --source-dir ../results/raw/sources_eval_dpo_r${ROUND}_val \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_dpo_r${ROUND}_val.log
         python -u recheck_eval.py ../results/raw/eval_dpo_r${ROUND}_val.csv 2>&1 | tee /workspace/recheck_dpo_r${ROUND}_val.txt

         echo \"========== STAGE 3b: eval clean (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_dpo_r${ROUND}_clean.csv --source-dir ../results/raw/sources_eval_dpo_r${ROUND}_clean \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_dpo_r${ROUND}_clean.log
         python -u recheck_eval.py ../results/raw/eval_dpo_r${ROUND}_clean.csv 2>&1 | tee /workspace/recheck_dpo_r${ROUND}_clean.txt

         cp ../results/raw/eval_dpo_r${ROUND}_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (DPO round $ROUND) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=16 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl\" WATCHDOG_LOG=/tmp/watchdog_dpo_r${ROUND}.log bash $LOCAL/scripts/watchdog.sh"
