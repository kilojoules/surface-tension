#!/bin/bash
# RunPod version of launch_rank_curve.sh. Spins up a single A100 SXM 80GB pod (RunPod
# Community Cloud, ~$1.39/hr), runs 3 ranks sequentially (r=8/32/128, lr=1e-4, linear,
# 20 epochs) with periodic train/val NLL logging + best-val-checkpoint snapshotting
# (the post-fix sft_train.py), pushes both <repo>-final and <repo>-bestval per rank,
# touches /workspace/all_done at the end. Watchdog terminates the pod on completion.

set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_rank_curve.env"   # reused name; format is identical "<id> <ip> <port>"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-COMMUNITY}"

echo "=== launch_rank_curve_runpod (gpu=$GPU cloud=$CLOUD, 3 ranks, train/val + snapshots) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-rank-curve \
    --env-file "$INSTANCE_FILE" --disk 150 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

# RunPod pods sometimes take an extra 30-60s for sshd to fully start after IP is assigned.
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: pod up but SSH not responding"; python3 "$LOCAL/scripts/runpod_kill.py" "$INST"; exit 1; }

echo "installing deps (rsync + version-pinned python libs for torch 2.4)..."
# torch 2.4 + latest bitsandbytes breaks (torch.library infer_schema error on the
# int8-matmul custom op). bnb 0.44.1 uses the classic CUDA-extension path → works on
# torch 2.4; transformers 4.46.x accepts bnb >= 0.43 for 4-bit, so this combo is consistent.
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q 'transformers==4.46.3' 'peft==0.13.2' 'bitsandbytes==0.44.1' 'accelerate>=0.34,<1.1' datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch, transformers, peft, bitsandbytes; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, \"peft\", peft.__version__, \"bnb\", bitsandbytes.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/sft_train.jsonl" "$LOCAL/data/sft_eval.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching rank-curve pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -uo pipefail -c '
         run() {  # \$1=rank
           R=\$1; A=\$((R * 2))
           REPO=kilojoules/surface-tension-sft-rankcurve-r\$R
           echo \"========== RANK-CURVE r=\$R alpha=\$A ==========\"
           if SFT_TRAIN=../data/sft_train.jsonl SFT_EVAL=../data/sft_eval.jsonl \
              SFT_OUTPUT=../outputs/rankcurve_r\$R \
              LORA_RANK=\$R LORA_ALPHA=\$A LORA_DROPOUT=0.0 \
              SFT_LR=1e-4 SFT_EPOCHS=20 LR_SCHEDULE=linear \
              QUANT_BIT=4 STRIP_WRAPPERS=1 USE_GC=1 \
              EVAL_EVERY=0 EVAL_N=0 LOG_EVERY=5 VAL_EVERY=24 VAL_N=48 \
                python -u sft_train.py 2>&1 | tee /workspace/sft_rankcurve_r\$R.log; then
             touch /workspace/rankcurve_r\${R}_trained
             cp ../outputs/rankcurve_r\$R/val_curve.jsonl /workspace/val_curve_r\$R.jsonl 2>/dev/null || true
             cp ../outputs/rankcurve_r\$R/best_val.json /workspace/best_val_r\$R.json 2>/dev/null || true
             timeout 900 python -u push_adapter.py ../outputs/rankcurve_r\$R/final_adapter \${REPO}-final 2>&1 | tee /workspace/push_r\${R}_final.log || echo \"WARN: push final r\$R failed\"
             if [ -d ../outputs/rankcurve_r\$R/best_val_adapter ]; then
               timeout 900 python -u push_adapter.py ../outputs/rankcurve_r\$R/best_val_adapter \${REPO}-bestval 2>&1 | tee /workspace/push_r\${R}_bestval.log || echo \"WARN: push bestval r\$R failed\"
             fi
             touch /workspace/rankcurve_r\${R}_pushed
           else
             echo \"WARN: r\$R training failed/aborted\"
             touch /workspace/rankcurve_r\${R}_failed
           fi
         }
         run 8
         run 32
         run 128
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"rank-curve launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (rank curve: r=8/32/128) ==="
echo "  Pod: $INST"
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=12 WATCHDOG_MAX_HOURS=12 bash $LOCAL/scripts/watchdog.sh"
