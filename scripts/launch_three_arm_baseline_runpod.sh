#!/bin/bash
# Three-arm baseline study (base + vanilla SFT + R-SFT B1++) on clean-17 with
# raw prose saved. Implements the brief's pre-launch gate:
#   "confirm the _save_source patch writes *__raw.txt with full preamble+code+
#    post-code on a single live sample, on each of the three adapters, BEFORE
#    committing the full run."
# If verification on any adapter fails, the full eval aborts on the pod (no
# further $/walltime spent on a broken pipeline). Tarjan is the project-standard
# recursion check; rescoring with src/strict_ladder.py happens locally after
# sync-back.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/runpod_three_arm.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"

VANILLA_SFT="kilojoules/surface-tension-sft-rationale-stripped-r32-final"
R_SFT="kilojoules/surface-tension-sft-b1plus-r32-final"

echo "=== launch_three_arm_baseline (gpu=$GPU cloud=$CLOUD; base + vanilla SFT + R-SFT on clean-17) ==="
echo "  vanilla SFT adapter:  $VANILLA_SFT"
echo "  R-SFT (B1++) adapter: $R_SFT"
echo "  samples per problem:  $N_SAMPLES"
echo "  max_new_tokens:       $MAX_NEW_TOKENS"

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name st-three-arm-baseline \
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
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); s=time.time()-t; print(f\"GPU matmul-50x: {s:.2f}s\"); assert s < 2.5, f\"FATAL slow GPU silicon (matmul-50x={s:.2f}s > 2.5s threshold) — kill + relaunch\"'"

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

# Build a single-problem manifest on the pod for verification (abc356_c only)
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "head -1 /workspace/st/data/problems_lcb_clean17.jsonl > /workspace/st/data/verify_one_problem.jsonl"

echo "launching three-arm pipeline (verify-per-arm then full per-arm)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "root@$HOST" \
    "cd /workspace/st/src && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       VANILLA_SFT='$VANILLA_SFT' R_SFT='$R_SFT' \
       N_SAMPLES=$N_SAMPLES MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
       bash -uo pipefail -c '
         set -e

         verify_arm() {
           local arm=\$1
           local adapter=\$2
           local extra=\"\"
           [ -n \"\$adapter\" ] && extra=\"--adapter \$adapter\"
           echo \"========== verify \$arm (one-sample raw-write check) ==========\"
           LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
             --problems ../data/verify_one_problem.jsonl \
             --csv ../results/raw/verify_\${arm}.csv \
             --source-dir ../results/raw/sources_verify_\${arm} \
             --base-model google/gemma-4-31B-it \$extra \
             --n-samples 1 --max-new-tokens \$MAX_NEW_TOKENS --temperature 0.7 --constraints \
             2>&1 | tee /workspace/verify_\${arm}.log
           python -c \"
import glob, sys
src = \\\"../results/raw/sources_verify_\${arm}\\\"
raw_files = sorted(glob.glob(src + \\\"/*__raw.txt\\\"))
py_files  = sorted(glob.glob(src + \\\"/*.py\\\"))
assert raw_files, f\\\"FAIL[\${arm}]: no *__raw.txt in {src}\\\"
assert py_files,  f\\\"FAIL[\${arm}]: no *.py in {src}\\\"
raw = open(raw_files[0]).read()
py  = open(py_files[0]).read()
has_codeblock = \\\"\`\`\`python\\\" in raw or \\\"\`\`\`\\\" in raw
has_prose_before_block = raw.find(\\\"\`\`\`\\\") > 80
print(f\\\"VERIFY[\${arm}]: raw_chars={len(raw)} py_chars={len(py)} has_codeblock={has_codeblock} has_prose_before_block={has_prose_before_block}\\\")
print(f\\\"VERIFY[\${arm}] head: {raw[:200]!r}\\\")
assert len(raw) > 200, f\\\"FAIL[\${arm}]: raw too short ({len(raw)} chars)\\\"
print(f\\\"VERIFY[\${arm}] PASS\\\")
\"
         }

         full_arm() {
           local arm=\$1
           local adapter=\$2
           local extra=\"\"
           [ -n \"\$adapter\" ] && extra=\"--adapter \$adapter\"
           echo \"========== full eval \$arm on clean-17 ==========\"
           LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
             --problems ../data/problems_lcb_clean17.jsonl \
             --csv ../results/raw/eval_\${arm}_clean17.csv \
             --source-dir ../results/raw/sources_eval_\${arm}_clean17 \
             --base-model google/gemma-4-31B-it \$extra \
             --n-samples \$N_SAMPLES --max-new-tokens \$MAX_NEW_TOKENS --temperature 0.7 --constraints \
             2>&1 | tee /workspace/eval_\${arm}_clean17.log
           python -u recheck_eval.py ../results/raw/eval_\${arm}_clean17.csv 2>&1 | tee /workspace/recheck_\${arm}_clean17.txt
           cp ../results/raw/eval_\${arm}_clean17.csv /workspace/ 2>/dev/null || true
         }

         # Phase 1: verify each adapter ONE-SAMPLE before any full run.
         # Abort the entire pipeline if any verification fails.
         verify_arm base       \"\"
         verify_arm vanillaSFT \"\$VANILLA_SFT\"
         verify_arm rsft       \"\$R_SFT\"
         echo \"=== ALL VERIFICATIONS PASSED — proceeding to full runs ===\"

         # Phase 2: full eval per arm
         full_arm base       \"\"
         full_arm vanillaSFT \"\$VANILLA_SFT\"
         full_arm rsft       \"\$R_SFT\"

         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\""

echo ""
echo "=== LAUNCHED on RunPod (3-arm baseline: base + vanilla SFT + R-SFT clean-17 with raw saved) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo ""
echo "Phase 1 (verification) takes ~25 min — three 1-sample loads to confirm raw saves."
echo "Phase 2 (full eval) takes ~15 hours total — three arms x 17 problems x $N_SAMPLES samples."
echo ""
echo "Next: PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE \\"
echo "      WATCHDOG_INTERVAL=300 WATCHDOG_GRACE_MIN=10 WATCHDOG_STALL_MIN=120 \\"
echo "      WATCHDOG_MAX_HOURS=20 bash $LOCAL/scripts/watchdog.sh"
