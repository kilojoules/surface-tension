#!/bin/bash
# v8: SFT distillation pipeline. Runs end-to-end on a single vast.ai box:
#
#   Stage A: data gen — sweep_local on bare-prompt + constrained, n_samples=4
#   Stage B: build SFT dataset (filter compliant+passing, dedupe)
#   Stage C: SFT train (with mid-eval + abort rules)
#   Stage D: final eval — 3 controls (trained-bare, trained-constrained, base-bare)
#                         on held-out problems, n_samples=3
#
# Why this beats v6/v7 DPO: pure positive token-CE signal, no preference collapse,
# bounded loss. See README "Next steps" for the design rationale.
#
# Usage:
#   bash scripts/launch_sft.sh [base_model] [hub_repo] [gpu_filter]
#     defaults: google/gemma-3-4b-it   kilojoules/surface-tension-sft   RTX_4090
#
# For the full 31B run:
#   bash scripts/launch_sft.sh google/gemma-4-31B-it kilojoules/surface-tension-sft A100_SXM4

set -e
BASE_MODEL="${1:-google/gemma-3-4b-it}"
HUB_REPO="${2:-kilojoules/surface-tension-sft}"
GPU_FILTER="${3:-RTX_4090}"

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
INSTANCE_FILE="$LOCAL/vast_current.env"

case "$GPU_FILTER" in
    RTX_4090)   DPH_MAX=0.50 ;;
    H100_SXM)   DPH_MAX=2.50 ;;
    H100_NVL)   DPH_MAX=2.50 ;;
    H100_PCIE)  DPH_MAX=2.50 ;;
    H200)       DPH_MAX=3.50 ;;
    A100_SXM4)  DPH_MAX=1.20 ;;
    *)          DPH_MAX=2.50 ;;
esac
RELIABILITY="${RELIABILITY:-0.995}"
N_SAMPLES_DATAGEN="${N_SAMPLES_DATAGEN:-4}"
N_SAMPLES_EVAL="${N_SAMPLES_EVAL:-3}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK / 2))}"
SFT_LR="${SFT_LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

echo "=== launch_sft: $BASE_MODEL → $HUB_REPO on $GPU_FILTER ==="

# 1. find offer
OFFER_ID=$(echo "n" | vastai search offers \
    "gpu_name=$GPU_FILTER num_gpus=1 dph<$DPH_MAX inet_down>200 disk_space>=80 reliability>$RELIABILITY" \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "no offers matching filter" && exit 1
echo "offer: $OFFER_ID"

# 2. rent
RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "st-sft" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "
import re, sys
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")
[ -z "$INST" ] && echo "rent failed: $RESULT" && exit 1
echo "instance: $INST"

# 3. wait for running
for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    echo "  status: $STATUS ($i/40)"
    sleep 15
done
[ "$STATUS" != "running" ] && { echo "FAIL"; echo "n" | vastai destroy instance "$INST" 2>&1 | grep -v Update; exit 1; }

# 4. ssh info
SSH_INFO=$(echo "n" | vastai show instance "$INST" --raw 2>&1 | grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST"

# 5. wait for ssh
for i in $(seq 1 20); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

# 6. install minimal deps (no trl, no vllm)
echo "installing deps..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "pip install -q transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -1
     pip uninstall hf-xet -y 2>&1 | tail -1 || true"

# 7. HF auth
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

# 8. rsync code + LCB problems + (existing) DPO sources for re-using v4 chosen completions
echo "uploading code + problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}" 2>/dev/null
rsync -az --include='*.py' --include='*.txt' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb.jsonl" \
    "root@$HOST:/workspace/st/data/" 2>/dev/null

# 9. launch the 4-stage pipeline
echo "launching SFT pipeline on box..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env \
       BASE_MODEL='$BASE_MODEL' HUB_REPO='$HUB_REPO' \
       SFT_LR='$SFT_LR' SFT_EPOCHS='$SFT_EPOCHS' \
       LORA_RANK='$LORA_RANK' LORA_ALPHA='$LORA_ALPHA' \
       MAX_LENGTH='$MAX_LENGTH' MAX_NEW_TOKENS='$MAX_NEW_TOKENS' \
       N_SAMPLES_DATAGEN='$N_SAMPLES_DATAGEN' N_SAMPLES_EVAL='$N_SAMPLES_EVAL' \
       bash -euo pipefail -c '
         echo \"=== Stage A: data-gen sweep ===\"
         python -u sweep_local.py \
           --problems ../data/problems_lcb.jsonl \
           --csv ../results/raw/sft_datagen_raw.csv \
           --source-dir ../results/raw/sources_sft_datagen \
           --base-model \"\$BASE_MODEL\" \
           --n-samples \"\$N_SAMPLES_DATAGEN\" \
           --max-new-tokens \"\$MAX_NEW_TOKENS\" \
           --temperature 0.8 \
           --constraints no_loops_no_recursion 2>&1 | tee /workspace/datagen.log

         echo \"=== Stage B: build SFT dataset ===\"
         python -u build_sft_dataset.py \
           --csv ../results/raw/sft_datagen_raw.csv \
           --sources-dir ../results/raw/sources_sft_datagen \
           --constraint no_loops_no_recursion \
           --holdout-frac 0.3 2>&1 | tee /workspace/build_sft.log
         N=\$(wc -l < ../data/sft_train.jsonl)
         echo \"  sft_train.jsonl has \$N examples\"
         if [ \"\$N\" -lt 50 ]; then
           echo \"  FATAL: too few SFT examples; aborting before training\"
           exit 1
         fi

         echo \"=== Stage C: SFT training ===\"
         HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
           python -u sft_train.py 2>&1 | tee /workspace/sft_train.log

         echo \"=== Stage D: final eval (3 controls) ===\"
         # D1: trained model, bare prompt (the headline)
         python -u sweep_local.py \
           --problems ../data/problems_lcb.jsonl \
           --csv ../results/raw/sft_eval_trained_bare.csv \
           --source-dir ../results/raw/sources_sft_eval_trained_bare \
           --base-model \"\$BASE_MODEL\" \
           --adapter ../outputs/sft_run1/final_adapter \
           --n-samples \"\$N_SAMPLES_EVAL\" \
           --max-new-tokens \"\$MAX_NEW_TOKENS\" \
           --constraints no_loops_no_recursion 2>&1 | tee /workspace/eval_trained_bare.log
         python -u aggregate.py \
           --csv ../results/raw/sft_eval_trained_bare.csv \
           --summary-csv ../results/sft_eval_trained_bare_summary.csv \
           --summary-md ../results/sft_eval_trained_bare_summary.md \
           --sources-dir ../results/raw/sources_sft_eval_trained_bare 2>&1 | tee /workspace/aggregate.log

         touch /workspace/all_done
       ' \
       > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "  Pipeline: ssh -p $PORT root@$HOST 'tail -f /workspace/pipeline.log'"
echo ""
echo "Next: bash scripts/watchdog.sh   # in another terminal — destroys on completion"
