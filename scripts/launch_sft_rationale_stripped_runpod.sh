#!/bin/bash
# Matched-control SFT: same 66 rationale-SFT demos, with prose rationale stripped
# from the targets (just the ```python code block kept). Trains SFT identically to
# rationale-SFT proper (r=32, lr=1e-4, 20 epochs, linear decay). The only variable
# that differs is the presence/absence of rationale prose in the targets.
#
# Used to ablate: in the SFT->DPO chain, how much of the DPO ceiling on clean
# came from the rationale prose specifically, vs the rest of the pipeline?
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_sft_stripped.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
HUB_REPO="${HUB_REPO:-kilojoules/surface-tension-sft-rationale-stripped-r32}"

echo "=== launch_sft_rationale_stripped (gpu=$GPU cloud=$CLOUD; r=32, 20 epochs, hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-sft-stripped \
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
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x\")'
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + stripped dataset + eval problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/sft_rationale_stripped_train.jsonl" \
    "$LOCAL/data/sft_rationale_stripped_eval.jsonl" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching SFT-stripped pipeline (train → push → eval val+clean)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== STAGE 1: SFT train (r=32, 20 epochs) on rationale-stripped 66-demo set ==========\"
         SFT_TRAIN=../data/sft_rationale_stripped_train.jsonl SFT_EVAL=../data/sft_rationale_stripped_eval.jsonl \
         SFT_OUTPUT=../outputs/rationale_stripped_r32 \
         LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
         SFT_LR=1e-4 SFT_EPOCHS=20 LR_SCHEDULE=linear \
         QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 \
         MAX_LENGTH=2048 MAX_PROMPT_LENGTH=768 \
         EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=5 VAL_EVERY=24 VAL_N=48 \
           python -u sft_train.py 2>&1 | tee /workspace/sft_stripped.log
         [ -d ../outputs/rationale_stripped_r32/final_adapter ] || { echo \"FATAL: no final_adapter\"; exit 1; }

         echo \"========== push -final adapter to Hub ==========\"
         timeout 900 python -u push_adapter.py ../outputs/rationale_stripped_r32/final_adapter ${HUB_REPO}-final 2>&1 | tee /workspace/push_final.log || echo \"WARN: push final failed\"
         if [ -d ../outputs/rationale_stripped_r32/best_val_adapter ]; then
           timeout 900 python -u push_adapter.py ../outputs/rationale_stripped_r32/best_val_adapter ${HUB_REPO}-bestval 2>&1 | tee /workspace/push_bestval.log || echo \"WARN: push bestval failed\"
         fi

         echo \"========== STAGE 2a: eval val (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_stripped_val.csv --source-dir ../results/raw/sources_eval_stripped_val \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_stripped_val.log
         python -u recheck_eval.py ../results/raw/eval_stripped_val.csv 2>&1 | tee /workspace/recheck_val.txt

         echo \"========== STAGE 2b: eval clean (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_stripped_clean.csv --source-dir ../results/raw/sources_eval_stripped_clean \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_stripped_clean.log
         python -u recheck_eval.py ../results/raw/eval_stripped_clean.csv 2>&1 | tee /workspace/recheck_clean.txt

         cp ../results/raw/eval_stripped_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (SFT rationale-stripped) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=14 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl --exclude=sft_rationale_stripped_*.jsonl\" WATCHDOG_LOG=/tmp/watchdog_sft_stripped.log bash $LOCAL/scripts/watchdog.sh"
