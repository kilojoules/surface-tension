#!/bin/bash
# Rank sweep with train/val loss curves. 3 configs run sequentially, train-only (no LCB eval):
#   rank=8   (alpha=16)
#   rank=32  (alpha=64)   <- the v9/PhaseA reference rank
#   rank=128 (alpha=256)
# All: lr=1e-4, linear-decay-to-10%, 20 epochs, dropout=0, STRIP_WRAPPERS=1, 4-bit + GC.
# Trains on sft_train.jsonl (91 pairs / 28 problems); validates (teacher-forced NLL, cheap,
# no generation) on sft_eval.jsonl (55 pairs / 12 PROBLEM-LEVEL-HELD-OUT problems) every
# ~2 epochs. Logs train_nll + val_nll vs step → val_curve.jsonl. End-of-run fit probe too.
# Push each adapter to Hub. No LCB-30 behavioral eval here.
#
# Outputs: vast_logs/$INST/sft_rankcurve_r<R>.log + .../outputs/rankcurve_r<R>/val_curve.jsonl
# Hub: kilojoules/surface-tension-sft-rankcurve-r<R>

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-150}"
INSTANCE_FILE="$LOCAL/vast_rank_curve.env"
RELIABILITY="${RELIABILITY:-0.99}"
DPH_MAX="${DPH_MAX:-2.00}"

echo "=== launch_rank_curve (rank 8/32/128, lr=1e-4, 20ep, train/val curves) ==="

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=100 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-rank-curve-noautokill" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && { echo "rent failed: $RESULT"; exit 1; }
echo "instance: $INST"

for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    sleep 15
done
[ "$STATUS" != "running" ] && { echo "FAIL"; echo "n" | vastai destroy instance "$INST"; exit 1; }

SSH_INFO=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST"

for i in $(seq 1 20); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "pip install -q transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true"

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
             # copy the val curve + best-val metadata out where the watchdog will sync it
             cp ../outputs/rankcurve_r\$R/val_curve.jsonl /workspace/val_curve_r\$R.jsonl 2>/dev/null || true
             cp ../outputs/rankcurve_r\$R/best_val.json /workspace/best_val_r\$R.json 2>/dev/null || true
             # push BOTH checkpoints: the final (most-trained, possibly overfit) and the
             # val-min (early-stop point, plausibly more general). Eval them separately.
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
echo "=== LAUNCHED (rank curve: r=8/32/128, train/val) ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=12 WATCHDOG_MAX_HOURS=12 bash /Users/julianquick/portfolio_copy/surface_tension/scripts/watchdog.sh"
