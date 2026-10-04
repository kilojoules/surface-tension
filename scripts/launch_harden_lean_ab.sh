#!/usr/bin/env bash
# Lean hardening run — Stage A+B on RunPod (split + P0 gate + harvest candidates)
# per docs/harden_lean_plan_2026-10-04.md. Launches ONE pod, runs everything,
# pulls results back, and DESTROYS THE POD in every exit path (trap).
# No active GPUs are left behind. Total wall ~35-50 min; ~$1-2.
#
# Usage: bash scripts/launch_harden_lean_ab.sh
# Env:   GPU (default "NVIDIA A40"), CLOUD (default COMMUNITY)
#
# Prereqs on this Mac (checked before any spend):
#   ~/.run.pod   — RunPod API key
#   ~/.hf_token  — HuggingFace token (gated meta-llama models)

set -euo pipefail

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_FILE="$LOCAL/harden_lean_ab.instance"
POD_ID=""
GPU="${GPU:-NVIDIA A40}"
CLOUD="${CLOUD:-COMMUNITY}"

say() { echo -e "\n=== $* ==="; }

kill_pod() {
    if [ -n "$POD_ID" ]; then
        say "DESTROYING POD $POD_ID (no active GPUs left behind)"
        python3 "$LOCAL/scripts/runpod_kill.py" "$POD_ID" || true
    fi
}
trap kill_pod EXIT

# ---- pre-flight ----
say "checking local credentials"
# Credential files live at nonstandard names on this Mac; symlink to the paths
# the shared tooling (runpod_launch.py / runpod_kill.py / prior launch scripts)
# expects. Idempotent.
[ -e "$HOME/.run.pod" ] || ln -s "$HOME/.super_lab_run.pod" "$HOME/.run.pod"
[ -e "$HOME/.hf_token" ] || ln -s "$HOME/.hf_key" "$HOME/.hf_token"
[ -f "$HOME/.run.pod" ] || { echo "FATAL: no RunPod API key (~/.super_lab_run.pod)"; exit 1; }
[ -f "$HOME/.hf_token" ] || { echo "FATAL: no HuggingFace token (~/.hf_key)"; exit 1; }

# ---- launch (try GPU candidates in price order until one is available) ----
say "launching pod (cloud=$CLOUD, candidates: ${GPU_CANDIDATES:-default list})"
IFS='|' read -ra CANDS <<< "${GPU_CANDIDATES:-NVIDIA A40|NVIDIA A5000|NVIDIA RTX 4090|NVIDIA A6000|NVIDIA L40S|NVIDIA A100-SXM4-80GB}"
launched=0
for g in "${CANDS[@]}"; do
    say "trying gpu=$g"
    if python3 "$LOCAL/scripts/runpod_launch.py" \
        --gpu "$g" --cloud "$CLOUD" --name "st-harden-lean-ab" \
        --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
        --env-file "$INSTANCE_FILE" --disk 60 --wait-min 15; then
        launched=1; GPU="$g"; break
    fi
    say "gpu=$g unavailable (or never SSH-ready) — trying next candidate"
done
[ "$launched" -eq 1 ] || { echo "FATAL: no GPU available from candidates"; exit 1; }
read -r INST HOST PORT < "$INSTANCE_FILE"
POD_ID="$INST"
say "pod ready on $GPU: $INST  (ssh -p $PORT root@$HOST)"

# ---- wait for ssh ----
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" >/dev/null 2>&1 && break
    sleep 10
done

# ---- deps + HF login ----
say "installing deps"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     pip install -q 'torch==2.5.1' 'transformers==5.14.1' 'accelerate==1.14.0' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3' \
                    'peft==0.17.0' 'bitsandbytes==0.48.0' 'scipy' 2>&1 | tail -2
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,accelerate; print(\"deps ok\", torch.__version__, transformers.__version__)'"
say "huggingface login (gated Llama)"
scp -q -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

# ---- upload (code, then the four public datasets) ----
say "uploading code + data"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/src /workspace/st/results/raw /workspace/st/data/harden/private"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
for d in harmbench gsm8k xstest alpaca; do
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "$LOCAL/data/$d/" "root@$HOST:/workspace/st/data/$d/"
done

say "verifying upload"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     n_src=\$(ls /workspace/st/src/*.py | wc -l)
     [ \"\$n_src\" -ge 40 ] || { echo \"FATAL: src upload incomplete (\$n_src files)\"; exit 1; }
     [ -s /workspace/st/data/harmbench/harmbench_behaviors_text_all.csv ] || { echo 'FATAL: harmbench csv missing'; exit 1; }
     [ -s /workspace/st/data/gsm8k/train.jsonl ] || { echo 'FATAL: gsm8k missing'; exit 1; }
     [ -s /workspace/st/data/xstest/xstest.jsonl ] || { echo 'FATAL: xstest missing'; exit 1; }
     [ -s /workspace/st/data/alpaca/alpaca.jsonl ] || { echo 'FATAL: alpaca missing'; exit 1; }
     echo 'pod upload OK'"

# ---- Stage A: split (pure) ----
say "STAGE A: behavior split"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && PYTHONPATH=src python src/harden_behavior_split.py \
        --behaviors data/harmbench/harmbench_behaviors_text_all.csv \
        --out-dir data/harden/split --seed 0"

# ---- Stage A: P0 baseline gate (GPU) ----
say "STAGE A: P0 baseline gate (base refusal >= 0.90, prefill plumbing)"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && PYTHONPATH=src python src/harden_p0_baseline.py \
        --behaviors data/harmbench/harmbench_behaviors_text_all.csv \
        --out results/raw/harden_p0_baseline.jsonl 2>&1 | tail -40"

# ---- Stage B: harvest candidates (GPU; approval stays with the author) ----
say "STAGE B: harvesting harmful-opening candidates (train + eval splits)"
# QUANT_BIT=0: bf16 7B fits an A40-48GB without bitsandbytes quantization.
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && QUANT_BIT=0 PYTHONPATH=src python src/harden_harvest_openings.py harvest \
        --behaviors data/harden/split/victim_train.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct --N 128 \
        --out-dir data/harden/private/openings_train 2>&1 | tail -8"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && QUANT_BIT=0 PYTHONPATH=src python src/harden_harvest_openings.py harvest \
        --behaviors data/harden/split/eval.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct --N 128 \
        --out-dir data/harden/private/openings_eval 2>&1 | tail -8"

# ---- pull results back (scores + split + private candidates for YOUR review) ----
say "pulling results back"
mkdir -p "$LOCAL/results/raw" "$LOCAL/data/harden/split" "$LOCAL/data/harden/private"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/st/results/raw/" "$LOCAL/results/raw/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/st/data/harden/split/" "$LOCAL/data/harden/split/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/st/data/harden/private/" "$LOCAL/data/harden/private/"

say "P0 summary"
cat "$LOCAL/results/raw/harden_p0_baseline_summary.json" 2>/dev/null || true

say "DONE — pod will now be destroyed by the exit trap; review candidates:"
echo "  data/harden/private/openings_train/REVIEW.html"
echo "  data/harden/private/openings_eval/REVIEW.html"
