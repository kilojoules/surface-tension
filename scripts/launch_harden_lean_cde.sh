#!/usr/bin/env bash
# Lean hardening run — Stage C+D+E on RunPod (pairs -> DPO train -> defensive eval)
# per docs/harden_lean_plan_2026-10-04.md. Runs AFTER the author has approved
# openings (launch_harden_lean_ab.sh produced candidates; the author ran
# harden_harvest_openings.py freeze -> approved_train.json / approved_eval.json).
#
# Same guarantees as the A+B script: ONE pod, results always pulled back before
# exit, pod DESTROYED in every exit path (trap). No active GPUs left behind.
# Wall ~2.5-4 h on an L40S/A100; ~$3-6.
#
# Usage:   bash scripts/launch_harden_lean_cde.sh
# Env:     GPU_CANDIDATES (default "NVIDIA A40|NVIDIA RTX A6000|NVIDIA GeForce RTX 4090|NVIDIA RTX 5000 Ada Generation|NVIDIA A100-SXM4-80GB"),
#          CLOUD (default COMMUNITY)
#
# Local prereqs (checked before any spend):
#   data/harden/split/{victim_train,eval}.jsonl     (from the A+B run)
#   data/harden/private/approved_train.json         (your freeze)
#   data/harden/private/approved_eval.json          (your freeze)
#   data/gsm8k, data/xstest, data/alpaca           (staged on this Mac)
#   ~/.super_lab_run.pod, ~/.hf_key                 (symlinked by the script)

set -euo pipefail

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_FILE="$LOCAL/harden_lean_cde.instance"
POD_ID=""
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
say "checking local prerequisites"
[ -e "$HOME/.run.pod" ] || ln -s "$HOME/.super_lab_run.pod" "$HOME/.run.pod"
[ -e "$HOME/.hf_token" ] || ln -s "$HOME/.hf_key" "$HOME/.hf_token"
[ -f "$HOME/.run.pod" ] || { echo "FATAL: no RunPod API key"; exit 1; }
[ -f "$HOME/.hf_token" ] || { echo "FATAL: no HF token"; exit 1; }
[ -s "$LOCAL/data/harden/private/approved_train.json" ] || { echo "FATAL: data/harden/private/approved_train.json missing — run the approval freeze (Gate B) first"; exit 1; }
[ -s "$LOCAL/data/harden/private/approved_eval.json" ] || { echo "FATAL: data/harden/private/approved_eval.json missing — run the approval freeze (Gate B) first"; exit 1; }
[ -s "$LOCAL/data/harden/split/victim_train.jsonl" ] || { echo "FATAL: data/harden/split/victim_train.jsonl missing — run launch_harden_lean_ab.sh first"; exit 1; }
[ -s "$LOCAL/data/gsm8k/train.jsonl" ] || { echo "FATAL: gsm8k not staged"; exit 1; }
[ -s "$LOCAL/data/xstest/xstest.jsonl" ] || { echo "FATAL: xstest not staged"; exit 1; }
[ -s "$LOCAL/data/alpaca/alpaca.jsonl" ] || { echo "FATAL: alpaca not staged"; exit 1; }
python3 -c "import json; a=json.load(open('$LOCAL/data/harden/private/approved_train.json')); print(f'approved_train: {len(a)} openings')"

# ---- launch (fallback list, price order) ----
say "launching pod (cloud=$CLOUD)"
IFS='|' read -ra CANDS <<< "${GPU_CANDIDATES:-NVIDIA A40|NVIDIA RTX A6000|NVIDIA GeForce RTX 4090|NVIDIA RTX 5000 Ada Generation|NVIDIA A100-SXM4-80GB}"
launched=0
for g in "${CANDS[@]}"; do
    say "trying gpu=$g"
    if python3 "$LOCAL/scripts/runpod_launch.py" \
        --gpu "$g" --cloud "$CLOUD" --name "st-harden-lean-cde" \
        --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
        --env-file "$INSTANCE_FILE" --disk 80 --wait-min 15; then
        launched=1; GPU="$g"; break
    fi
    say "gpu=$g unavailable — trying next"
done
[ "$launched" -eq 1 ] || { echo "FATAL: no GPU available"; exit 1; }
read -r INST HOST PORT < "$INSTANCE_FILE"
POD_ID="$INST"
say "pod ready on $GPU: $INST  (ssh -p $PORT root@$HOST)"

# ---- ssh + deps + cuda check ----
for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=10 "root@$HOST" "echo ok" >/dev/null 2>&1 && break
    sleep 10
done

say "installing deps"
# peft is needed for dpo_train (LoRA) — pin a build compatible with transformers 5.14.1.
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" \
    "set -e
     apt-get update -qq && apt-get install -y -qq rsync
     pip install -q 'torch==2.5.1' 'transformers==5.14.1' 'accelerate==1.14.0' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3' 'scipy' 2>&1 | tail -2
     pip install -q 'peft==0.16.0' 'bitsandbytes==0.48.0' 2>&1 | tail -2
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch,transformers,accelerate,peft; print(\"deps ok\", torch.__version__, transformers.__version__, peft.__version__)'
     python3 -c 'import torch; torch.cuda.init(); assert torch.cuda.is_available(), \"no CUDA\"; print(\"cuda ok:\", torch.cuda.get_device_name(0))'"

say "huggingface login (gated Llama)"
scp -q -P "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

# ---- upload ----
say "uploading code + data (public sets, split, private approved)"
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" \
    "mkdir -p /workspace/st/src /workspace/st/results/raw /workspace/st/outputs \
               /workspace/st/data/harden/private /workspace/st/data/harden/pairs"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
for d in gsm8k xstest alpaca; do
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
        "$LOCAL/data/$d/" "root@$HOST:/workspace/st/data/$d/"
done
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "$LOCAL/data/harden/split/" "root@$HOST:/workspace/st/data/harden/split/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "$LOCAL/data/harden/private/approved_train.json" \
    "$LOCAL/data/harden/private/approved_eval.json" \
    "root@$HOST:/workspace/st/data/harden/private/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "$LOCAL/scripts/stage_e_pod.sh" "root@$HOST:/workspace/st/stage_e.sh"

say "verifying upload"
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" \
    "set -e
     [ \"\$(ls /workspace/st/src/*.py | wc -l)\" -ge 40 ] || { echo 'FATAL: src incomplete'; exit 1; }
     [ -s /workspace/st/data/harden/private/approved_train.json ] || { echo 'FATAL: approved_train missing'; exit 1; }
     [ -s /workspace/st/data/harden/private/approved_eval.json ] || { echo 'FATAL: approved_eval missing'; exit 1; }
     echo 'pod upload OK'"

# ---- Stage C: build DPO pairs ----
FAILED_STAGE=""
say "STAGE C: building DPO pairs (refusal | math | benign x2 | merge)"
# Single heredoc'd remote script: correct exit propagation for every sub-step,
# no shell-precedence pitfalls from chaining ssh "A && B; rc=$?" fragments.
if ! ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" 'bash -s' <<'REMOTE'
    set -e
    cd /workspace/st
    PYTHONPATH=src python src/harden_build_dpo_pairs.py refusal \
        --behaviors data/harden/split/victim_train.jsonl \
        --completions data/harden/private/approved_train.json \
        --n-per-behavior 2 --out data/harden/pairs/refusal.jsonl
    run_logged() { local name="$1"; shift; "$@" > "/workspace/st/$name.log" 2>&1 \
        || { local rc=$?; tail -5 "/workspace/st/$name.log"; exit $rc; }; \
        tail -3 "/workspace/st/$name.log"; }
    run_logged math env PYTHONPATH=src QUANT_BIT=4 python3 src/harden_build_dpo_pairs.py math \
        --gsm8k data/gsm8k/train.jsonl --n-problems 300 --n-samples 4 \
        --victim meta-llama/Llama-3.1-8B-Instruct \
        --out data/harden/pairs/math.jsonl
    run_logged benign_x env PYTHONPATH=src QUANT_BIT=4 python3 src/harden_build_dpo_pairs.py benign \
        --format xstest --benign data/xstest/xstest.jsonl --n-prompts 100 --n-samples 2 \
        --victim meta-llama/Llama-3.1-8B-Instruct \
        --out data/harden/pairs/benign_xstest.jsonl
    run_logged benign_a env PYTHONPATH=src QUANT_BIT=4 python3 src/harden_build_dpo_pairs.py benign \
        --format alpaca --benign data/alpaca/alpaca.jsonl --n-prompts 100 --n-samples 2 \
        --victim meta-llama/Llama-3.1-8B-Instruct \
        --out data/harden/pairs/benign_alpaca.jsonl
    PYTHONPATH=src python src/harden_build_dpo_pairs.py merge \
        --inputs data/harden/pairs/refusal.jsonl data/harden/pairs/math.jsonl \
                 data/harden/pairs/benign_xstest.jsonl data/harden/pairs/benign_alpaca.jsonl \
        --out data/harden/pairs/victim_dpo_v1.jsonl --seed 0
REMOTE
then
    FAILED_STAGE="pairs"; fi

# ---- Stage D: DPO train ----
if [ -z "$FAILED_STAGE" ]; then
    say "STAGE D: DPO-harden the victim (~1h on L40S)"
    if ! ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" \
        "cd /workspace/st && BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
            DPO_TRAIN=data/harden/pairs/victim_dpo_v1.jsonl \
            DPO_OUTPUT=outputs/harden_dpo_v1 DPO_BETA=0.1 DPO_LR=5e-6 DPO_EPOCHS=3 \
            LORA_RANK=32 MAX_LENGTH=2048 QUANT_BIT=4 \
            PYTHONPATH=src python src/dpo_train.py > /workspace/st/dpo.log 2>&1; \
            rc=\$?; tail -25 /workspace/st/dpo.log; exit \$rc"; then
        FAILED_STAGE="dpo_train"; fi
fi

# ---- Stage E: defensive eval on base + hardened ----
if [ -z "$FAILED_STAGE" ]; then
    say "STAGE E: defensive eval (refusal / overrefusal / gsm8k / battery / prefill) x (base, dpo)"
    if ! ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" \
        "cd /workspace/st && bash stage_e.sh > /workspace/st/stage_e.log 2>&1; \
            rc=\$?; tail -30 /workspace/st/stage_e.log; exit \$rc"; then
        FAILED_STAGE="stage_e"; fi
fi

# ---- pull everything back BEFORE deciding exit ----
say "pulling results back"
mkdir -p "$LOCAL/results/raw" "$LOCAL/data/harden/pairs" "$LOCAL/outputs"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "root@$HOST:/workspace/st/results/raw/" "$LOCAL/results/raw/" || true
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "root@$HOST:/workspace/st/data/harden/pairs/" "$LOCAL/data/harden/pairs/" || true
# adapter ~100MB (LoRA r=32); pull for later use without retraining
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "root@$HOST:/workspace/st/outputs/" "$LOCAL/outputs/" || true

if [ -n "$FAILED_STAGE" ]; then
    say "FAILED at stage: $FAILED_STAGE — pod destroyed by exit trap"
    exit 1
fi
say "DONE (C+D+E) — pod destroyed by exit trap. Summaries:"
for f in "$LOCAL"/results/raw/harden_*_summary.json; do
    echo "--- $f"; cat "$f" 2>/dev/null || echo "(missing)"
done
