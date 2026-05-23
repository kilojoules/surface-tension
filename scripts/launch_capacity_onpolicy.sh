#!/bin/bash
# Instance 2: capacity (LoRA rank sweep) + on-policy SFT comparison.
#
# Stage C: SFT at LORA_RANK ∈ {64, 128, 256} on the same 146 pairs as v8.
#          Each pushed to Hub as kilojoules/surface-tension-sft-r{N}.
# Stage D: Sample base bare-prompt n=10 on 50 LCB problems, filter to
#          compliant+passing, train SFT at matched N, push as
#          kilojoules/surface-tension-sft-onpolicy.
# Stage E: Eval each new adapter on LCB-30 bare prompt at n=3.
#
# Outputs: vast_logs/$INST/st/results/raw/{eval_*,sft_onpolicy.jsonl}
# Hub: kilojoules/surface-tension-sft-{r64,r128,r256,onpolicy}

set -e
GPU_FILTER="${GPU_FILTER:-A100_SXM4}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK="${DISK:-180}"
INSTANCE_FILE="$LOCAL/vast_capacity_onpolicy.env"
RELIABILITY="${RELIABILITY:-0.99}"

case "$GPU_FILTER" in
    A100_SXM4)  DPH_MAX=1.20 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    *)          DPH_MAX=1.50 ;;
esac

echo "=== launch_capacity_onpolicy ==="

OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=120 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers" && exit 1

RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-cap-onpolicy-noautokill" --raw 2>&1 | grep -v "Update\|selected")
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
    "$LOCAL/data/problems_lcb.jsonl" "$LOCAL/data/sft_all.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

echo "launching pipeline..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash -euo pipefail -c '
         echo \"=== Stage C: LoRA rank sweep ===\"
         for RANK in 64 128 256; do
           ALPHA=\$((RANK / 2))
           echo \"--- training rank=\$RANK alpha=\$ALPHA ---\"
           if SFT_TRAIN=../data/sft_all.jsonl \
              SFT_EVAL=../data/sft_all.jsonl \
              SFT_OUTPUT=../outputs/sft_r\$RANK \
              LORA_RANK=\$RANK \
              LORA_ALPHA=\$ALPHA \
              HUB_REPO=kilojoules/surface-tension-sft-r\$RANK \
              SFT_LR=1e-5 SFT_EPOCHS=3 \
              EVAL_EVERY=0 \
                python -u sft_train.py 2>&1 | tee /workspace/sft_r\$RANK.log; then
             touch /workspace/stage_c_r\${RANK}_trained

             echo \"--- eval rank=\$RANK on LCB-30 ---\"
             python -u sweep_local.py \
               --problems ../data/problems_lcb.jsonl \
               --csv ../results/raw/eval_r\${RANK}_lcb30.csv \
               --source-dir ../results/raw/sources_eval_r\${RANK}_lcb30 \
               --base-model google/gemma-4-31B-it \
               --adapter kilojoules/surface-tension-sft-r\$RANK \
               --n-samples 3 \
               --max-new-tokens 1024 \
               --limit-problems 30 \
               --constraints 2>&1 | tee /workspace/eval_r\$RANK.log || \
               echo \"WARN: eval rank=\$RANK failed but training succeeded\"
             touch /workspace/stage_c_r\${RANK}_evaled
           else
             echo \"WARN: rank=\$RANK training failed; skipping its eval\"
             touch /workspace/stage_c_r\${RANK}_failed
           fi
         done
         touch /workspace/stage_c_done

         echo \"=== Stage D1: mine on-policy positives ===\"
         # Sample base bare-prompt n=10 on first 50 LCB problems, filter compliant+passing.
         python -u sweep_local.py \
           --problems ../data/problems_lcb.jsonl \
           --csv ../results/raw/onpolicy_mine_lcb50.csv \
           --source-dir ../results/raw/sources_onpolicy_mine_lcb50 \
           --base-model google/gemma-4-31B-it \
           --n-samples 10 \
           --max-new-tokens 1024 \
           --limit-problems 50 \
           --constraints 2>&1 | tee /workspace/onpolicy_mine.log

         python -u build_onpolicy_dataset.py \
           --csv ../results/raw/onpolicy_mine_lcb50.csv \
           --sources ../results/raw/sources_onpolicy_mine_lcb50 \
           --problems ../data/problems_lcb.jsonl \
           --limit-problems 50 \
           --out ../data/sft_onpolicy.jsonl 2>&1 | tee /workspace/onpolicy_filter.log
         touch /workspace/stage_d1_done

         echo \"=== Stage D2: train on-policy SFT ===\"
         if SFT_TRAIN=../data/sft_onpolicy.jsonl \
            SFT_EVAL=../data/sft_onpolicy.jsonl \
            SFT_OUTPUT=../outputs/sft_onpolicy \
            LORA_RANK=32 LORA_ALPHA=16 \
            HUB_REPO=kilojoules/surface-tension-sft-onpolicy \
            SFT_LR=1e-5 SFT_EPOCHS=3 \
            EVAL_EVERY=0 \
              python -u sft_train.py 2>&1 | tee /workspace/sft_onpolicy.log; then
           touch /workspace/stage_d2_done

           echo \"=== Stage D3: eval on-policy SFT on LCB-30 ===\"
           python -u sweep_local.py \
             --problems ../data/problems_lcb.jsonl \
             --csv ../results/raw/eval_onpolicy_lcb30.csv \
             --source-dir ../results/raw/sources_eval_onpolicy_lcb30 \
             --base-model google/gemma-4-31B-it \
             --adapter kilojoules/surface-tension-sft-onpolicy \
             --n-samples 3 \
             --max-new-tokens 1024 \
             --limit-problems 30 \
             --constraints 2>&1 | tee /workspace/eval_onpolicy.log || \
             echo \"WARN: on-policy eval failed but training succeeded\"
         else
           echo \"WARN: on-policy training failed\"
           touch /workspace/stage_d2_failed
         fi
         touch /workspace/all_done
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "Next: WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=90 WATCHDOG_GRACE_MIN=15 WATCHDOG_MAX_HOURS=6 bash scripts/watchdog.sh"
