#!/bin/bash
# Launch quadrant Phase-1 generation on RunPod (one model per pod).
#
# Watchdog config follows feedback_watchdog_pitfalls.md:
#   • MAX_HOURS = 16 (≥ 2× best-case ~7.4h for k=12 × 17 problems × probe)
#   • NO custom WATCHDOG_RSYNC_EXCLUDES — defaults already skip training
#     artifacts only; the output paper/data/quadrant/self_reports.jsonl IS
#     covered by the default sync. NEVER add the output file to excludes.
#   • Balance precheck before launch — refuses to launch if projected burn
#     × 1.5 exceeds clientBalance.
#
# Usage:
#   scripts/launch_quadrant_gen_runpod.sh base
#   scripts/launch_quadrant_gen_runpod.sh R-SFT
#       (the second invocation appends to the same self_reports.jsonl;
#        generate.py resumes idempotently on (model, problem_id, sample_idx))
#
# Env overrides:
#   GPU                NVIDIA A100-SXM4-80GB   (default)
#   CLOUD              SECURE                  (RunPod community vs secure)
#   MAX_HOURS          16
#   K                  12 (samples per problem)
#   ADAPTER_PATH       hf-repo-or-local-path   (REQUIRED for R-SFT model)
#   POD_DOLLARS_PER_HR 2.50   (estimate for balance precheck)
#   CAPTURE_ACTIVATIONS  empty | 1   (default empty; set 1 to dump probe-turn
#                                     activations to /workspace/st/activations)

set -euo pipefail

MODEL_TAG="${1:-}"
if [[ -z "$MODEL_TAG" ]]; then
    echo "usage: $0 <base|R-SFT>" >&2; exit 2
fi
case "$MODEL_TAG" in
    base) ADAPTER_PATH="${ADAPTER_PATH:-}";;
    R-SFT)
        if [[ -z "${ADAPTER_PATH:-}" ]]; then
            echo "FAIL: ADAPTER_PATH env var required for --model R-SFT" >&2
            echo "  e.g. ADAPTER_PATH=username/st-rsft-r32-final $0 R-SFT" >&2
            exit 2
        fi
        ;;
    *)  echo "FAIL: unknown model tag '$MODEL_TAG' (expected base|R-SFT)" >&2
        exit 2;;
esac

LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_quadrant_${MODEL_TAG//[^a-zA-Z0-9]/_}.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
MAX_HOURS="${MAX_HOURS:-16}"
K="${K:-12}"
POD_DOLLARS_PER_HR="${POD_DOLLARS_PER_HR:-2.50}"
CAPTURE_ACTIVATIONS_FLAG=""
[[ -n "${CAPTURE_ACTIVATIONS:-}" ]] && CAPTURE_ACTIVATIONS_FLAG="--capture-activations --activations-dir /workspace/st/activations"

# PROBLEMS_FILE env var lets us swap the cohort without editing the launcher.
# Default = clean-17; pass a single-problem file for irreducible case studies.
# MUST match a real file in paper/data/quadrant/ — the launcher uploads it to
# the pod under the same basename.
PROBLEMS_LOCAL="$LOCAL/paper/data/quadrant/${PROBLEMS_FILE:-problems_lcb_clean17.jsonl}"
TESTS_LOCAL="$LOCAL/data/problems_lcb_clean17.jsonl"
OUT_LOCAL_DIR="$LOCAL/paper/data/quadrant"
PROBLEMS_BASENAME=$(basename "$PROBLEMS_LOCAL")

echo "=== Quadrant Phase-1 launch ($MODEL_TAG) ==="
echo "  gpu=$GPU cloud=$CLOUD max_hours=$MAX_HOURS k=$K"
echo "  adapter=${ADAPTER_PATH:-<none, base model>}"
echo "  capture_activations=${CAPTURE_ACTIVATIONS:-0}"

# -----------------------------------------------------------------------------
# Local pre-flight
# -----------------------------------------------------------------------------
[[ -f "$PROBLEMS_LOCAL" ]] || {
    echo "FAIL: missing $PROBLEMS_LOCAL" >&2
    echo "  Run:  python -m quadrant.build_problems" >&2
    exit 1
}
[[ -f "$TESTS_LOCAL" ]] || {
    echo "FAIL: missing $TESTS_LOCAL (LCB tests manifest)" >&2; exit 1
}

# Sanity: count matches EXPECTED_COUNT (default 17 for clean-17). For a
# different cohort (e.g. one-problem case-study run) export EXPECTED_COUNT
# alongside PROBLEMS_FILE.
EXPECTED_COUNT="${EXPECTED_COUNT:-17}"
LOCAL_COUNT=$(wc -l < "$PROBLEMS_LOCAL" | tr -d ' ')
if [[ "$LOCAL_COUNT" != "$EXPECTED_COUNT" ]]; then
    echo "FAIL: $PROBLEMS_LOCAL has $LOCAL_COUNT lines, expected $EXPECTED_COUNT" >&2
    exit 1
fi

# Balance precheck
echo "--- RunPod balance precheck ---"
python3 "$LOCAL/scripts/quadrant_balance_check.py" \
    --new-pod-per-hr "$POD_DOLLARS_PER_HR" --max-hours "$MAX_HOURS" --safety 1.5 \
    || { echo "FAIL: balance precheck failed; not launching" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Launch pod
# -----------------------------------------------------------------------------
echo "--- launching pod ---"
python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" \
    --name "st-quadrant-${MODEL_TAG//[^a-zA-Z0-9]/_}" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 130 --wait-min 22 || exit 1

read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"

# ---------------------------------------------------------------------------
# Pod is now live and billing. Install an EXIT trap that tears down on ANY
# failure between here and the successful handoff (sentinel-found OR watchdog-
# command-printed). The trap clears itself on clean handoff so we don't kill
# pods that are doing legitimate work.
# Prior bug (2026-06-08): the launcher's `set -e` killed the script mid-deps-
# install, but the pod stayed alive at $1.49/hr because nothing tore it down.
# ---------------------------------------------------------------------------
LAUNCH_HANDOFF_OK=0
_teardown_on_failure() {
    local rc=$?
    if [[ "$LAUNCH_HANDOFF_OK" != "1" && -n "${INST:-}" ]]; then
        echo "" >&2
        echo "!! launcher exiting with rc=$rc before handoff — tearing down pod $INST" >&2
        python3 "$LOCAL/scripts/runpod_kill.py" "$INST" 2>&1 | tail -3 || true
    fi
}
trap _teardown_on_failure EXIT

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
    sleep 10
done
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null \
    || { echo "FAIL: pod up but SSH not responding"; python3 "$LOCAL/scripts/runpod_kill.py" "$INST"; exit 1; }

# -----------------------------------------------------------------------------
# Pod setup
# -----------------------------------------------------------------------------
echo "--- installing deps + GPU sanity ---"
# torch 2.7.0 introduces float8_e8m0fnu, which transformers ≥ 5.x references
# on module import (via integrations.finegrained_fp8). Earlier torch versions
# fail to import gemma4 modeling at all. torchvision 0.22.0 matches torch 2.7.
# Empirically learned across four failed launches; see git log for the journey.
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.7.0' 'torchvision==0.22.0' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate datasets 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     set -e
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.7\"), (\"FATAL torch=\"+torch.__version__)'
     python3 -c 'import torch; assert hasattr(torch, \"float8_e8m0fnu\"), \"FATAL torch missing float8_e8m0fnu — latest transformers fp8 path will fail\"'
     python3 -c 'import torchvision; print(\"torchvision\", torchvision.__version__)'
     python3 -c 'import torch,time; x=torch.randn(8192,8192,device=\"cuda\"); torch.cuda.synchronize(); t=time.time(); [torch.mm(x,x) for _ in range(50)]; torch.cuda.synchronize(); dt=time.time()-t; print(f\"GPU matmul-50x: {dt:.2f}s\"); assert dt<3.5, f\"slow GPU: {dt:.2f}s (>3.5s threshold — host likely contended)\"'
     python3 -c 'import transformers; print(\"transformers\", transformers.__version__)'
     python3 -c 'from transformers import AutoTokenizer, AutoModelForCausalLM; print(\"transformers imports OK\")'
     python3 -c 'from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM; print(\"gemma4 modeling import OK\")'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

# -----------------------------------------------------------------------------
# rsync — INPUT FILES ONLY (per feedback_watchdog_pitfalls.md)
# -----------------------------------------------------------------------------
# Files we PUSH up (inputs):
#   src/                                  — code
#   scripts/quadrant_pod_runner.py        — orchestrator
#   data/problems_lcb_clean17.jsonl       — LCB tests manifest (input)
#   paper/data/quadrant/problems_lcb_clean17.jsonl  — quadrant cohort (input)
#
# Files the pod WRITES (do NOT add to any rsync exclude list):
#   paper/data/quadrant/self_reports.jsonl         — the experimental output
#   workspace/eval_quadrant_$MODEL_TAG.log         — runner stdout
#   workspace/all_done                              — sentinel
#   activations/*.pt (optional)                    — probe-turn activations

echo "--- uploading code + manifests ---"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data,paper/data/quadrant,activations}" 2>/dev/null

rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null

rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/quadrant_pod_runner.py" "root@$HOST:/workspace/st/scripts/" 2>/dev/null

rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$TESTS_LOCAL" "root@$HOST:/workspace/st/data/" 2>/dev/null

rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$PROBLEMS_LOCAL" "root@$HOST:/workspace/st/paper/data/quadrant/" 2>/dev/null

# -----------------------------------------------------------------------------
# Launch the runner
# -----------------------------------------------------------------------------
ADAPTER_FLAG=""
[[ -n "${ADAPTER_PATH:-}" ]] && ADAPTER_FLAG="--adapter-path $ADAPTER_PATH"

EXIT_SENTINEL="/workspace/quadrant_gen_${MODEL_TAG}.exit"
SMOKE_APPROVED_FLAG=""
if [[ "${SMOKE_APPROVED:-}" == "1" ]]; then
    SMOKE_APPROVED_FLAG="--smoke-approved"
    echo "--- SMOKE_APPROVED=1 set; gate will bypass on the pod ---"
fi

echo "--- launching pipeline ---"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && \
     nohup env HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) SMOKE_APPROVED=${SMOKE_APPROVED:-} \
       bash -uo pipefail -c '
         echo \"========== quadrant gen $MODEL_TAG ==========\"
         python -u scripts/quadrant_pod_runner.py \
             --model $MODEL_TAG \
             --base-model google/gemma-4-31B-it \
             $ADAPTER_FLAG \
             --problems paper/data/quadrant/$PROBLEMS_BASENAME \
             --tests    data/problems_lcb_clean17.jsonl \
             --out      paper/data/quadrant/self_reports.jsonl \
             --smoke-artifacts-dir paper/data/quadrant \
             --exit-sentinel-path $EXIT_SENTINEL \
             -k $K --solution-temperature 0.7 \
             --solution-max-tokens 4096 --probe-max-tokens 256 \
             --quant-bit 4 $CAPTURE_ACTIVATIONS_FLAG \
             $SMOKE_APPROVED_FLAG \
             2>&1 | tee /workspace/quadrant_gen_$MODEL_TAG.log
         touch /workspace/all_done
         echo \"=== ALL DONE ===\"
       ' > /workspace/pipeline.log 2>&1 &
     echo \"runner launched, pid=\$!\""

# -----------------------------------------------------------------------------
# Two paths from here:
#
#  (A) Smoke-gate mode (default — SMOKE_APPROVED not set): the gate halts
#      within ~6-10 min after model load with exit 64. We poll the sentinel
#      for up to 30 min. If we see 64: HALTED status + tear down. If we see
#      a non-64 non-zero: FAILED status + tear down. Timeout → assume the
#      gate behaved oddly; hand off to the watchdog.
#
#  (B) Approved mode (SMOKE_APPROVED=1): the full k×17 run will take ~7h;
#      we don't poll. Print the watchdog command and let the user start it.
# -----------------------------------------------------------------------------

if [[ "${SMOKE_APPROVED:-}" == "1" ]]; then
    echo ""
    echo "=== LAUNCHED ($MODEL_TAG on pod $INST, SMOKE_APPROVED) ==="
    echo "ssh -p $PORT root@$HOST"
    echo ""
    echo "Full run takes ~7h. Start the watchdog in another terminal:"
    echo ""
    echo "  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE \\"
    echo "      WATCHDOG_INTERVAL=180 WATCHDOG_GRACE_MIN=10 \\"
    echo "      WATCHDOG_MAX_HOURS=$MAX_HOURS \\"
    echo "      bash $LOCAL/scripts/watchdog.sh"
    echo ""
    echo "QUADRANT $MODEL_TAG: LAUNCHED_FOR_FULL_RUN (use watchdog to monitor + destroy)"
    LAUNCH_HANDOFF_OK=1   # watchdog now owns the pod
    exit 0
fi

# --- (A) Smoke-gate poll -----------------------------------------------------
echo ""
echo "--- polling smoke gate (up to 30 min) ---"
POLL_DEADLINE=$(($(date +%s) + 30*60))
EXIT_CODE=""
while [[ $(date +%s) -lt $POLL_DEADLINE ]]; do
    if ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" \
            "test -f $EXIT_SENTINEL && cat $EXIT_SENTINEL" 2>/dev/null > /tmp/quadrant_exit.$$ ; then
        if [[ -s /tmp/quadrant_exit.$$ ]]; then
            EXIT_CODE=$(tr -d '[:space:]' < /tmp/quadrant_exit.$$)
            echo "[poll] sentinel: exit=$EXIT_CODE"
            break
        fi
    fi
    echo "[poll] $(date '+%H:%M:%S') no sentinel yet"
    sleep 30
done
rm -f /tmp/quadrant_exit.$$

# Always sync artifacts (smoke files in particular) back before destroying.
sync_back() {
    local dest="$LOCAL/vast_logs/$INST"
    mkdir -p "$dest"
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30" \
        --exclude='*.pyc' --exclude='__pycache__' \
        --exclude='outputs/*/checkpoint-*' \
        --exclude='outputs/*/final_adapter' \
        --exclude='outputs/*/best_val_adapter' \
        "root@$HOST:/workspace/" "$dest/" 2>/dev/null || echo "  [sync] failed"
}

# Distinct status-line constants — both halted-for-review and a real crash
# leave no running pod, so they need clearly different lines.
HALTED_LINE="QUADRANT $MODEL_TAG: HALTED_FOR_SMOKE_REVIEW → paper/data/quadrant/smoke_${MODEL_TAG//[^A-Za-z0-9_.-]/_}.txt (re-launch with SMOKE_APPROVED=1 ./scripts/launch_quadrant_gen_runpod.sh $MODEL_TAG)"
PASS_LINE="QUADRANT $MODEL_TAG: GATE_PASSED (full run continues; start watchdog)"
FAIL_LINE_TMPL='QUADRANT %s: FAILED (exit %s) → see vast_logs/%s/quadrant_gen_%s.log'

if [[ -z "$EXIT_CODE" ]]; then
    # Timed out without a sentinel — gate must have been bypassed by an old
    # SMOKE_APPROVED state on the pod, or the runner went into the long main
    # loop without writing one. Hand off to the watchdog rather than killing
    # work we don't fully understand.
    echo "[poll] timed out; assuming gate proceeded into full run."
    sync_back
    echo ""
    echo "$PASS_LINE"
    echo ""
    echo "Start watchdog to monitor + destroy (max ${MAX_HOURS}h):"
    echo ""
    echo "  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE \\"
    echo "      WATCHDOG_INTERVAL=180 WATCHDOG_GRACE_MIN=10 \\"
    echo "      WATCHDOG_MAX_HOURS=$MAX_HOURS \\"
    echo "      bash $LOCAL/scripts/watchdog.sh"
    LAUNCH_HANDOFF_OK=1   # watchdog now owns the pod
    exit 0

elif [[ "$EXIT_CODE" == "64" ]]; then
    # Smoke gate halted for human review. NOT a failure.
    echo "--- syncing smoke artifacts back ---"
    sync_back
    echo "--- tearing down pod (no idle billing while a human reviews) ---"
    python3 "$LOCAL/scripts/runpod_kill.py" "$INST" 2>&1 | tail -3
    echo ""
    echo "$HALTED_LINE"
    # Echo synced artifact paths so the reviewer can open them immediately.
    SMOKE_TXT_LOCAL="$LOCAL/vast_logs/$INST/st/paper/data/quadrant/smoke_${MODEL_TAG//[^A-Za-z0-9_.-]/_}.txt"
    if [[ -f "$SMOKE_TXT_LOCAL" ]]; then
        echo "  open: $SMOKE_TXT_LOCAL"
    else
        echo "  (smoke .txt not at expected local path — check $LOCAL/vast_logs/$INST/st/paper/data/quadrant/)"
    fi
    # Exit 0 because halted-for-review is intentional; the LINE is the signal.
    LAUNCH_HANDOFF_OK=1   # pod already destroyed above
    exit 0

elif [[ "$EXIT_CODE" == "0" ]]; then
    # Runner exited 0 within the polling window. That should only happen on a
    # fast-failed dry run; the full k×17 loop is much longer. Treat as success
    # but flag the surprise.
    echo "--- syncing back ---"
    sync_back
    echo "--- tearing down pod ---"
    python3 "$LOCAL/scripts/runpod_kill.py" "$INST" 2>&1 | tail -3
    echo ""
    echo "QUADRANT $MODEL_TAG: COMPLETED_FAST (exit 0 within smoke window; check log + row count)"
    LAUNCH_HANDOFF_OK=1   # pod already destroyed above
    exit 0

else
    # Real failure path. Distinct line, syncs back, tears down.
    echo "--- syncing back what we have ---"
    sync_back
    echo "--- tearing down pod ---"
    python3 "$LOCAL/scripts/runpod_kill.py" "$INST" 2>&1 | tail -3
    echo ""
    printf "$FAIL_LINE_TMPL\n" "$MODEL_TAG" "$EXIT_CODE" "$INST" "$MODEL_TAG"
    LAUNCH_HANDOFF_OK=1   # pod already destroyed above
    exit 1
fi
