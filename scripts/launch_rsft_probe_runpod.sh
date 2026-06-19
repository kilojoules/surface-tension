#!/bin/bash
# R-SFT-only probe regenerate. Single arm. Captures residual activations
# at the prose-end token. Used for the lean version of Task 5: prose-promises
# vs r0-violates probe.
#
# Two-phase like the 3-arm launcher but with only one adapter:
#   Phase 1: ONE-sample verification with strict preamble gate (R-SFT must
#            emit real prose, not bare code) AND that *__resid.pt lands.
#   Phase 2: full 17 x N_SAMPLES generation with raw + activations saved.
#
# Per-arm checkpoint files (arm_rsft_done) sync home via the watchdog.
# Incomplete-arm rule: if Phase 2 dies mid-run, relaunch the missing
# problems, never analyze a partial corpus.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/runpod_rsft_probe.env"
GPU="${GPU:-NVIDIA H100 80GB HBM3}"
CLOUD="${CLOUD:-SECURE}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"

R_SFT="kilojoules/surface-tension-sft-b1plus-r32-final"

echo "=== launch_rsft_probe (gpu=$GPU cloud=$CLOUD; R-SFT only, activations on) ==="
echo "  adapter:              $R_SFT"
echo "  samples per problem:  $N_SAMPLES"
echo "  max_new_tokens:       $MAX_NEW_TOKENS"
echo "  CAPTURE_ACTIVATIONS:  1"

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-rsft-probe \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 130 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: pod up but SSH not responding"; python3 "$LOCAL/scripts/runpod_kill.py" "$INST"; exit 1; }

echo "installing deps + GPU sanity check..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__)'
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); s=time.time()-t; print(f\"GPU matmul-50x: {s:.2f}s\"); assert s < 2.5, \"FATAL slow GPU\"'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + problems..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" "mkdir -p /workspace/st/{src,data,results/raw}"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/problems_lcb_clean17.jsonl" \
    "root@$HOST:/workspace/st/data/"
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "head -1 /workspace/st/data/problems_lcb_clean17.jsonl > /workspace/st/data/verify_one_problem.jsonl"

echo "launching pipeline: verify R-SFT (strict gate + activation file lands) → full eval..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       R_SFT='$R_SFT' \
       N_SAMPLES=$N_SAMPLES MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
       bash -uo pipefail -c '
         set -e
         echo \"========== Phase 1: verify (1 sample, strict gate + *__resid.pt check) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 CAPTURE_ACTIVATIONS=1 python -u sweep_local.py \
           --problems ../data/verify_one_problem.jsonl \
           --csv ../results/raw/verify_rsft.csv \
           --source-dir ../results/raw/sources_verify_rsft \
           --base-model google/gemma-4-31B-it --adapter \$R_SFT \
           --n-samples 1 --max-new-tokens \$MAX_NEW_TOKENS --temperature 0.7 --constraints \
           2>&1 | tee /workspace/verify_rsft.log
         python -c \"
import glob
src = \\\"../results/raw/sources_verify_rsft\\\"
raw_files  = sorted(glob.glob(src + \\\"/*__raw.txt\\\"))
py_files   = sorted(glob.glob(src + \\\"/*.py\\\"))
act_files  = sorted(glob.glob(src + \\\"/*__resid.pt\\\"))
assert raw_files, f\\\"FAIL: no *__raw.txt in {src}\\\"
assert py_files,  f\\\"FAIL: no *.py in {src}\\\"
assert act_files, f\\\"FAIL: no *__resid.pt in {src} — activation capture broken\\\"
raw = open(raw_files[0]).read()
fence_at = raw.find(\\\"\`\`\`\\\")
preamble = raw[:fence_at] if fence_at >= 0 else raw
preamble_alpha = sum(1 for c in preamble if c.isalpha())
print(f\\\"VERIFY raw_chars={len(raw)} fence_at={fence_at} preamble_alpha={preamble_alpha}\\\")
print(f\\\"VERIFY *__resid.pt exists ({act_files[0]})\\\")
assert fence_at >= 200, f\\\"FAIL: code fence at char {fence_at} — no real preamble surface\\\"
assert preamble_alpha >= 100, f\\\"FAIL: preamble has only {preamble_alpha} alpha chars\\\"
import torch
t = torch.load(act_files[0])
print(f\\\"VERIFY activation tensor shape={tuple(t.shape)} dtype={t.dtype}\\\")
print(\\\"VERIFY PASS\\\")
\"
         echo \"=== Phase 1 PASSED ===\"
         touch /workspace/verifications_done

         echo \"========== Phase 2: full eval (17 x \$N_SAMPLES) ==========\"
         LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 CAPTURE_ACTIVATIONS=1 python -u sweep_local.py \
           --problems ../data/problems_lcb_clean17.jsonl \
           --csv ../results/raw/eval_rsft_probe_clean17.csv \
           --source-dir ../results/raw/sources_eval_rsft_probe_clean17 \
           --base-model google/gemma-4-31B-it --adapter \$R_SFT \
           --n-samples \$N_SAMPLES --max-new-tokens \$MAX_NEW_TOKENS --temperature 0.7 --constraints \
           2>&1 | tee /workspace/eval_rsft_probe.log
         python -u recheck_eval.py ../results/raw/eval_rsft_probe_clean17.csv 2>&1 | tee /workspace/recheck_rsft_probe.txt
         cp ../results/raw/eval_rsft_probe_clean17.csv /workspace/ 2>/dev/null || true
         touch /workspace/arm_rsft_done
         echo \"=== checkpoint: arm_rsft_done ===\"

         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (R-SFT-only probe regenerate; raw + activations) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo ""
echo "Phase 1 (verify, ~10 min): one sample + strict preamble gate + *__resid.pt check."
echo "Phase 2 (full eval, ~5 hr on H100): 17 problems × $N_SAMPLES samples × R-SFT adapter."
echo ""
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE \\"
echo "      WATCHDOG_INTERVAL=300 WATCHDOG_GRACE_MIN=10 WATCHDOG_STALL_MIN=120 \\"
echo "      WATCHDOG_MAX_HOURS=10 bash $LOCAL/scripts/watchdog.sh"
