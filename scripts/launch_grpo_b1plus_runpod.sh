#!/bin/bash
# GRPO from B1++ checkpoint with multi-tier reward (penalize cheating).
# Pipeline:
#   1. GRPO train: 100 steps, G=8 rollouts/prompt, 4 prompts/step, KL anchor to B1++,
#      REWARD_MODE=multitier (+2 compliant∧pass, +1 compliant-substantive, -1 cheat, 0 else)
#   2. Push -grpo adapter to Hub
#   3. Eval val (12) + clean (17) at max_new_tokens=3072
# Watchdog cap 12h (max from this provider).
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_grpo_b1plus.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
HUB_REPO="${HUB_REPO:-kilojoules/surface-tension-grpo-b1plus-r32}"
ADAPTER_INIT="${ADAPTER_INIT:-kilojoules/surface-tension-sft-b1plus-r32-final}"

echo "=== launch_grpo_b1plus (gpu=$GPU cloud=$CLOUD; KL-anchor=$ADAPTER_INIT, hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-grpo-b1plus \
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
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__, \"| peft\", peft.__version__, \"| bnb\", b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problem data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_sfttrain.jsonl" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching GRPO pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== STAGE 1: GRPO train (multi-tier reward, KL-anchored to B1++) ==========\"
         ADAPTER_INIT=$ADAPTER_INIT \
         GRPO_TRAIN_PROBLEMS=../data/problems_lcb_sfttrain.jsonl \
         GRPO_EVAL_PROBLEMS=../data/problems_lcb_sfteval.jsonl \
         GRPO_LR=2e-6 GRPO_STEPS=100 G=8 PROMPTS_PER_STEP=4 \
         TEMPERATURE=0.9 KL_BETA=0.02 \
         GRPO_MODE=single \
         REWARD_MODE=multitier \
         ABORT_VAR_FLOOR=0.05 ABORT_PASS_FLOOR=0.20 \
         ABORT_ZERO_ACTIVE_FRAC=0.7 ABORT_ZERO_ACTIVE_WINDOW=12 \
         QUANT_BIT=4 STRIP_WRAPPERS=1 \
         GRPO_OUTPUT=../outputs/grpo_b1plus \
         LOG_EVERY=2 \
           python -u grpo_train.py 2>&1 | tee /workspace/grpo_b1plus.log
         [ -d ../outputs/grpo_b1plus/final_adapter ] || { echo \"FATAL: no final_adapter\"; exit 1; }

         echo \"========== push -grpo adapter to Hub ==========\"
         timeout 900 python -u push_adapter.py ../outputs/grpo_b1plus/final_adapter ${HUB_REPO}-final 2>&1 | tee /workspace/push_grpo.log || echo \"WARN: push failed\"

         echo \"========== STAGE 3a: eval on val set (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_grpo_val.csv --source-dir ../results/raw/sources_eval_grpo_val \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_grpo_val.log
         python -u recheck_eval.py ../results/raw/eval_grpo_val.csv 2>&1 | tee /workspace/recheck_grpo_val.txt

         echo \"========== STAGE 3b: eval on clean set (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_grpo_clean.csv --source-dir ../results/raw/sources_eval_grpo_clean \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_grpo_clean.log
         python -u recheck_eval.py ../results/raw/eval_grpo_clean.csv 2>&1 | tee /workspace/recheck_grpo_clean.txt

         cp ../results/raw/eval_grpo_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (GRPO from B1++) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=12 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl\" WATCHDOG_LOG=/tmp/watchdog_grpo_b1plus.log bash $LOCAL/scripts/watchdog.sh"
