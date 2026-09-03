#!/bin/bash
# Judge-only re-run: score the EXISTING 623 generations with the 70B (bnb 4-bit)
# under the Qi 1-5 rubric. No generation. Ships the generations (harmful) to an
# ephemeral pod; judged results stay in the canary'd local package.
# ~800 judge calls; ~1-1.5 h ≈ $2-3; MAX_HOURS=4.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
INSTANCE_FILE="$LOCAL/vast_jbb_judge.env"
GENS="${GENS:-$LOCAL/vast_logs/in2i48j52f7iub/st/results/raw/jbb_full.jsonl}"

# ---- gate 1: prereg anchored (Qi correction) ------------------------------
PREREG="$LOCAL/prereg/jbb_prefill_2026-09-03.md"
grep -q "Qi et al. 2023 GPT-4-Judge" "$PREREG" || { echo "REFUSING: prereg not on the Qi rubric"; exit 1; }

# ---- gate 2: existing generations present ---------------------------------
[ -f "$GENS" ] || { echo "FATAL: no generations at $GENS"; exit 1; }
n=$(wc -l < "$GENS"); echo "  generations to judge: $n rows"
[ "$n" -ge 600 ] || { echo "FATAL: expected ~623 generations, got $n"; exit 1; }

echo "=== launch_jbb_judge_only gpu=$GPU cloud=$CLOUD ==="
python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-jbb-judge" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 220 --wait-min 22 || exit 1
read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done

echo "installing deps (pinned; + bitsandbytes for the 70B 4-bit judge)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     apt-get update -qq; apt-get install -y -qq rsync; which rsync
     pip install -q 'torch==2.5.1'
     pip install -q 'transformers==5.14.1' 'accelerate==1.14.0' 'bitsandbytes==0.50.0' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3'
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,accelerate,bitsandbytes; print(\"deps ok\", torch.__version__)'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

echo "uploading src + runner + existing generations..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/src /workspace/st/scripts /workspace/st/results/raw"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/run_jbb_judge_only.sh" "root@$HOST:/workspace/st/scripts/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$GENS" "root@$HOST:/workspace/st/results/raw/jbb_full.jsonl"

# ---- gate 3: verified upload ----------------------------------------------
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     [ \$(ls /workspace/st/src/*.py | wc -l) -ge 15 ] || { echo 'FATAL: src incomplete'; exit 1; }
     n=\$(wc -l < /workspace/st/results/raw/jbb_full.jsonl)
     [ \"\$n\" -ge 600 ] || { echo \"FATAL: gens upload incomplete (\$n)\"; exit 1; }
     echo \"  pod gens OK \$n\"
     grep -q 'Qi' /workspace/st/src/jbb_prefill_judge.py && echo '  judge on Qi rubric OK'"

echo "launching judge-only runner..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_jbb_judge_only.sh && \
     nohup env HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash scripts/run_jbb_judge_only.sh > /workspace/pipeline_jbb_judge.log 2>&1 &
     echo \"runner launched, pid=\$!\"" </dev/null

cat <<EOF

=== LAUNCHED: JBB judge-only re-run (Qi 1-5 rubric) ===
  Pod: $INST   (ssh -p $PORT root@$HOST)
  ~800 judge calls on the 623 existing generations; ~1-1.5 h, ~\$2-3.
  Done marker: all_stages_done.
Watchdog (start after runner launch):
  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 \\
  WATCHDOG_GRACE_MIN=15 WATCHDOG_STALL_MIN=45 WATCHDOG_MAX_HOURS=4 \\
  WATCHDOG_LOG=/tmp/watchdog_jbb_judge.log bash $LOCAL/scripts/watchdog.sh
EOF
