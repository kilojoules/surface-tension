#!/bin/bash
# Recovery pod for the dead r8 rank-sweep pod (2zuutp8uolh0iz, died ~16:43
# UTC 2026-08-08, ~10h into the stripped_b149_r8 eval). Everything else from
# that pod is safe: rationale_b149_r8 fully evaled + synced, the stripped
# adapter pushed to the Hub, 50/136 stripped eval rows synced locally.
#
# This pod ONLY finishes the stripped_b149_r8 eval: pulls the adapter from
# the Hub, seeds the partial CSV + sources so sweep_local resumes at ~86
# pending, then runs run_sft_scaling.sh with ARMS=stripped_b149_r8 and
# pre-planted train/push markers. Best case ~8 h -> MAX_HOURS=20.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
DEAD=2zuutp8uolh0iz
INSTANCE_FILE="$LOCAL/vast_scaling_r8rec.env"
MIRROR="$LOCAL/vast_logs/$DEAD/st/results/raw"
[ -f "$MIRROR/eval_scaling_stripped_b149_r8.csv" ] || { echo "FATAL: no partial CSV in mirror"; exit 1; }

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "NVIDIA A100-SXM4-80GB" --cloud SECURE --name st-scal-r8rec \
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
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -2
     pip install -q --upgrade transformers peft bitsandbytes accelerate 'datasets<4.0' pandas 2>&1 | tail -2
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\")' && python3 -c 'import transformers, peft, bitsandbytes, datasets'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + scripts + data + partial eval state..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data/sft_scaling,results/raw,outputs}" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/run_sft_scaling.sh" "root@$HOST:/workspace/st/scripts/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/sft_scaling/" "root@$HOST:/workspace/st/data/sft_scaling/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" "root@$HOST:/workspace/st/data/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$MIRROR/eval_scaling_stripped_b149_r8.csv" \
    "root@$HOST:/workspace/st/results/raw/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$MIRROR/sources_eval_scaling_stripped_b149_r8/" \
    "root@$HOST:/workspace/st/results/raw/sources_eval_scaling_stripped_b149_r8/" 2>/dev/null

echo "seeding adapter from Hub + markers, launching resume pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_sft_scaling.sh && \
     python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('kilojoules/surface-tension-scaling-stripped-b149-r8-final',
                  local_dir='/workspace/st/outputs/scaling_stripped_b149_r8/final_adapter')
print('adapter seeded')\" && \
     touch /workspace/scal_stripped_b149_r8_train_done /workspace/scal_stripped_b149_r8_push_done && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       ARMS='stripped_b149_r8' HUB_PREFIX='kilojoules/surface-tension-scaling' \
       bash scripts/run_sft_scaling.sh > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\"" </dev/null

echo ""
echo "=== RECOVERY LAUNCHED (stripped_b149_r8 eval resume) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "Watchdog:"
echo "  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 \\"
echo "  WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=20 \\"
echo "  WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl --exclude=sft_scaling/\" \\"
echo "  WATCHDOG_LOG=/tmp/watchdog_scaling_r8rec.log bash $LOCAL/scripts/watchdog.sh"
