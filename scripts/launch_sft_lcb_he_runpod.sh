#!/bin/bash
# Combined rationale-SFT: LCB-28 (B1++ data, already on disk) + 100 HumanEval
# problems generated fresh on the pod, same recipe (n=8 max_new=4096 substantive
# filter). Trains SFT on the combined set, evaluates on the original LCB val (12)
# and clean (17) held-outs. The key question: does rationale-SFT trained across
# two different coding-problem distributions improve generalization on LCB clean?
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_sft_combined.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
HUB_REPO="${HUB_REPO:-kilojoules/surface-tension-sft-lcb28-he100-r32}"

echo "=== launch_sft_lcb_he (gpu=$GPU cloud=$CLOUD; hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-sft-lcb-he \
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
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__)'
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| peft\",peft.__version__,\"| bnb\",b.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problems + existing B1++ data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_humaneval100.jsonl" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null
# upload existing B1++ training + eval data so we can concat with new HE data
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/vast_logs/rdqb499a37k1tv/sft_b1plus_train.jsonl" \
    "$LOCAL/vast_logs/rdqb499a37k1tv/sft_b1plus_eval.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching combined pipeline (HE rationale data gen → combine → SFT → push → eval val+clean)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== STAGE 1: HumanEval rationale data gen (100 problems, n=8, substantive filter) ==========\"
         python -u build_rationale_dataset.py \
           --problems ../data/problems_humaneval100.jsonl \
           --out ../data/sft_he_rationale.jsonl \
           --n-samples 8 --max-new-tokens 4096 --temperature 0.7 --filter substantive \
           2>&1 | tee /workspace/datagen_he.log
         [ -s ../data/sft_he_rationale.jsonl ] || { echo \"FATAL: empty HE rationale file\"; exit 1; }
         cp ../data/sft_he_rationale.jsonl /workspace/

         echo \"========== STAGE 2: combine LCB-28 (B1++ existing) + HE-100 (new) ==========\"
         cat ../data/sft_b1plus_train.jsonl ../data/sft_he_rationale.jsonl > ../data/sft_combined_train.jsonl
         # SFT eval split stays as LCB-only (B1++ eval data) — used only for training-time NLL monitoring
         cp ../data/sft_b1plus_eval.jsonl ../data/sft_combined_eval.jsonl
         echo \"  train demos: \$(wc -l < ../data/sft_combined_train.jsonl)\"
         echo \"  eval demos:  \$(wc -l < ../data/sft_combined_eval.jsonl)\"

         echo \"========== STAGE 3: SFT train (r=32, 20 epochs) on combined data ==========\"
         SFT_TRAIN=../data/sft_combined_train.jsonl SFT_EVAL=../data/sft_combined_eval.jsonl \
         SFT_OUTPUT=../outputs/lcb_he_r32 \
         LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
         SFT_LR=1e-4 SFT_EPOCHS=20 LR_SCHEDULE=linear \
         QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 \
         MAX_LENGTH=2048 MAX_PROMPT_LENGTH=768 \
         EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=5 VAL_EVERY=24 VAL_N=48 \
           python -u sft_train.py 2>&1 | tee /workspace/sft_lcb_he.log
         [ -d ../outputs/lcb_he_r32/final_adapter ] || { echo \"FATAL: no final_adapter\"; exit 1; }

         echo \"========== push -final adapter to Hub ==========\"
         timeout 900 python -u push_adapter.py ../outputs/lcb_he_r32/final_adapter ${HUB_REPO}-final 2>&1 | tee /workspace/push_final.log || echo \"WARN: push final failed\"
         if [ -d ../outputs/lcb_he_r32/best_val_adapter ]; then
           timeout 900 python -u push_adapter.py ../outputs/lcb_he_r32/best_val_adapter ${HUB_REPO}-bestval 2>&1 | tee /workspace/push_bestval.log || echo \"WARN: push bestval failed\"
         fi

         echo \"========== STAGE 4a: eval LCB val (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_lcb_he_val.csv --source-dir ../results/raw/sources_eval_lcb_he_val \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_lcb_he_val.log
         python -u recheck_eval.py ../results/raw/eval_lcb_he_val.csv 2>&1 | tee /workspace/recheck_val.txt

         echo \"========== STAGE 4b: eval LCB clean (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_lcb_he_clean.csv --source-dir ../results/raw/sources_eval_lcb_he_clean \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_lcb_he_clean.log
         python -u recheck_eval.py ../results/raw/eval_lcb_he_clean.csv 2>&1 | tee /workspace/recheck_clean.txt

         cp ../results/raw/eval_lcb_he_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (combined LCB-28 + HE-100 rationale-SFT) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=150 WATCHDOG_MAX_HOURS=20 WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_*.jsonl --exclude=sft_b1plus_*.jsonl --exclude=sft_he_rationale.jsonl --exclude=sft_combined_*.jsonl\" WATCHDOG_LOG=/tmp/watchdog_sft_lcb_he.log bash $LOCAL/scripts/watchdog.sh"
