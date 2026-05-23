#!/bin/bash
# Rationale-augmented SFT pipeline on RunPod. Three stages on one pod:
#   1. Data gen: base Gemma 4 31B-it + hint + "discuss why" disclosure on the 28
#      sft_train problems (n=4 samples each), filter to passing+compliant,
#      save (bare_prompt, full_rationale+code_response) pairs.
#      Same for the 12 sft_eval problems (used for training-time val NLL).
#   2. SFT train: LoRA r=32, alpha=64, lr=1e-4, linear-decay, 20 epochs.
#      Same recipe as rankcurve-r32; just the data is rationale-augmented.
#   3. Eval: sweep_local on val (12) + clean (17) at max_new_tokens=3072
#      (bumped because rationale prefix eats tokens), then recheck_eval.
# Watchdog auto-destroys when all_done is touched.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_rationale_sft.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
HUB_REPO="${HUB_REPO:-kilojoules/surface-tension-sft-rationale-r32}"

echo "=== launch_rationale_sft_pipeline_runpod (gpu=$GPU cloud=$CLOUD; r=32, 20 epochs, hub=$HUB_REPO) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-rationale-sft \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 150 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: pod up but SSH not responding"; python3 "$LOCAL/scripts/runpod_kill.py" "$INST"; exit 1; }

echo "installing deps (latest, matches the recent eval recipe — needed for gemma4 model_type)..."
# transformers==4.46.3 (rank-curve pin) does NOT have gemma4 support; the recent eval
# runs proved that the latest transformers + torch 2.5.1 + bnb 0.49 stack works for
# both inference AND training on Gemma 4 31B-it.
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch, transformers, peft, bitsandbytes as b; print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__, \"| peft\", peft.__version__, \"| bnb\", b.__version__)'
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); print(f\"GPU matmul-50x: {time.time()-t:.2f}s (expect <2s healthy A100)\")'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problem data..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
# Problem-set files for data gen + eval
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_sfttrain.jsonl" \
    "$LOCAL/data/problems_lcb_sfteval.jsonl" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching pipeline (gen → train → eval) with set -e..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"========== STAGE 1: rationale data gen ==========\"
         python -u build_rationale_dataset.py \
           --problems ../data/problems_lcb_sfttrain.jsonl \
           --out ../data/sft_rationale_train.jsonl \
           --n-samples 4 --max-new-tokens 2048 --temperature 0.7 \
           2>&1 | tee /workspace/datagen_train.log
         [ -s ../data/sft_rationale_train.jsonl ] || { echo \"FATAL: empty sft_rationale_train.jsonl\"; exit 1; }
         python -u build_rationale_dataset.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --out ../data/sft_rationale_eval.jsonl \
           --n-samples 4 --max-new-tokens 2048 --temperature 0.7 \
           2>&1 | tee /workspace/datagen_eval.log
         [ -s ../data/sft_rationale_eval.jsonl ] || { echo \"FATAL: empty sft_rationale_eval.jsonl\"; exit 1; }
         echo \"========== STAGE 1 done. train.jsonl + eval.jsonl sizes: ==========\"
         wc -l ../data/sft_rationale_train.jsonl ../data/sft_rationale_eval.jsonl
         cp ../data/sft_rationale_train.jsonl ../data/sft_rationale_eval.jsonl /workspace/

         echo \"========== STAGE 2: SFT train (r=32, 20 epochs) ==========\"
         SFT_TRAIN=../data/sft_rationale_train.jsonl SFT_EVAL=../data/sft_rationale_eval.jsonl \
         SFT_OUTPUT=../outputs/rationale_r32 \
         LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
         SFT_LR=1e-4 SFT_EPOCHS=20 LR_SCHEDULE=linear \
         QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 \
         MAX_LENGTH=2048 MAX_PROMPT_LENGTH=768 \
         EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=5 VAL_EVERY=24 VAL_N=48 \
           python -u sft_train.py 2>&1 | tee /workspace/sft_rationale.log
         [ -d ../outputs/rationale_r32/final_adapter ] || { echo \"FATAL: no final_adapter dir after training\"; exit 1; }

         echo \"========== push -final adapter to Hub ==========\"
         timeout 900 python -u push_adapter.py ../outputs/rationale_r32/final_adapter ${HUB_REPO}-final 2>&1 | tee /workspace/push_final.log || echo \"WARN: push final failed (continuing with eval, will use local adapter)\"
         if [ -d ../outputs/rationale_r32/best_val_adapter ]; then
           timeout 900 python -u push_adapter.py ../outputs/rationale_r32/best_val_adapter ${HUB_REPO}-bestval 2>&1 | tee /workspace/push_bestval.log || echo \"WARN: push bestval failed\"
         fi

         echo \"========== STAGE 3a: eval on val set (12 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_sfteval.jsonl \
           --csv ../results/raw/eval_rationale_val.csv --source-dir ../results/raw/sources_eval_rationale_val \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_rationale_val.log
         python -u recheck_eval.py ../results/raw/eval_rationale_val.csv 2>&1 | tee /workspace/recheck_val.txt

         echo \"========== STAGE 3b: eval on clean set (17 problems, max_new=3072) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_rationale_clean.csv --source-dir ../results/raw/sources_eval_rationale_clean \
           --base-model google/gemma-4-31B-it --adapter ${HUB_REPO}-final \
           --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_rationale_clean.log
         python -u recheck_eval.py ../results/raw/eval_rationale_clean.csv 2>&1 | tee /workspace/recheck_clean.txt

         cp ../results/raw/eval_rationale_*.csv /workspace/
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (rationale-SFT pipeline) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=25 WATCHDOG_MAX_HOURS=10 WATCHDOG_LOG=/tmp/watchdog_rationale.log bash $LOCAL/scripts/watchdog.sh"
