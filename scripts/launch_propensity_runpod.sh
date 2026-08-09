#!/bin/bash
# Launch the propensity panel on RunPod. Hard gates per ADDENDUM F6:
#   (a) the item-freeze addendum must be anchored (no placeholder);
#   (b) a writable off-pod sync target verified by write-read-delete BEFORE
#       the pipeline starts, and the watchdog (which does the per-batch
#       pull-sync) must be running before the main stages begin.
# Watchdog MAX_HOURS is sized from the on-pod pilot (2x extrapolated total),
# not guessed: this launcher blocks until pilot_timing.json syncs back.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
INSTANCE_FILE="$LOCAL/vast_propensity.env"

# ---- gate a: addendum anchored ---------------------------------------------
ADD="$LOCAL/data/propensity/ADDENDUM.md"
if grep -q '`____________`' "$ADD"; then
  echo "REFUSING TO LAUNCH: $ADD anchor placeholder still present."; exit 1
fi
# freeze integrity: every frozen file must match the manifest
python3 - <<'PY' || exit 1
import hashlib, json
m = json.load(open("/Users/julianquick/portfolio_copy/surface_tension/data/propensity/frozen/MANIFEST.json"))
for f, h in m["outputs"].items():
    p = f"/Users/julianquick/portfolio_copy/surface_tension/data/propensity/frozen/{f}"
    assert hashlib.sha256(open(p,'rb').read()).hexdigest() == h, f"sha mismatch: {f}"
print(f"freeze integrity ok ({len(m['outputs'])} files)")
PY

# ---- gate b: writable off-pod sync target -----------------------------------
SYNC="$LOCAL/vast_logs/propensity_pending"   # watchdog renames to pod id later
mkdir -p "$SYNC"
probe="$SYNC/.write_probe_$$"
echo probe > "$probe" && [ "$(cat "$probe")" = "probe" ] && rm "$probe" \
  || { echo "REFUSING TO LAUNCH: off-pod sync target not writable"; exit 1; }
echo "sync target verified: $SYNC"

# ---- balance ----------------------------------------------------------------
read -p "RunPod balance covers ~\$60 worst case? [y/N] " ok; [ "$ok" = "y" ] || exit 1

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "NVIDIA A100-SXM4-80GB" --cloud SECURE --name st-propensity \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --env-file "$INSTANCE_FILE" --disk 150 --wait-min 22 || exit 1
read INST HOST PORT < "$INSTANCE_FILE"
echo "ssh: ssh -p $PORT root@$HOST  (pod $INST)"
for i in $(seq 1 30); do
  ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" 2>/dev/null && break
  sleep 10
done

echo "installing deps (pinned)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "apt-get update -qq >/dev/null 2>&1 || true; apt-get install -y -qq rsync >/dev/null 2>&1 || true
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -2
     pip install -q --upgrade transformers peft bitsandbytes accelerate 'datasets<4.0' pandas 2>&1 | tail -2
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     pip freeze > /workspace/pip_freeze.txt
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\")'"
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + scripts + frozen items..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,results/raw} /workspace/st/data/propensity" 2>/dev/null
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/scripts/run_propensity_panel.sh" "root@$HOST:/workspace/st/scripts/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/propensity/" "root@$HOST:/workspace/st/data/propensity/" 2>/dev/null

echo "launching pipeline (pilot first)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_propensity_panel.sh && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       bash scripts/run_propensity_panel.sh > /workspace/pipeline.log 2>&1 &
     echo \"pipeline pid=\$!\"" </dev/null

echo "waiting for pilot timing (blocks; ~30-50 min incl. model download)..."
for i in $(seq 1 90); do
  T=$(ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" \
      "cat /workspace/st/results/raw/pilot_timing.json 2>/dev/null" 2>/dev/null)
  [ -n "$T" ] && break
  sleep 60
done
[ -n "${T:-}" ] || { echo "FATAL: pilot never reported; investigate pod"; exit 1; }
echo "pilot: $T"
python3 - "$T" <<'PY'
import json, sys
t = json.loads(sys.argv[1])
spr = t["secs_per_record"]
# full panel records: ~2 orders x ~7600 AB/pair items + 600 persona x3 + BBQ 600x3 etc ~= 19k rec/arm
per_arm_h = 19000 * spr / 3600
total_h = per_arm_h * 8 + 2 * per_arm_h * 0.35 + 3 * 0.6 * per_arm_h + 8 * 1.5  # panel + rescore/steer + bf16 + battery
print(f"secs/record {spr:.2f} -> per-arm ~{per_arm_h:.1f}h, extrapolated total ~{total_h:.1f}h")
print(f"RECOMMENDED WATCHDOG_MAX_HOURS={int(total_h*2)+1}")
PY
echo ""
echo "Start the watchdog NOW with the recommended cap:"
echo "  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE WATCHDOG_INTERVAL=120 \\"
echo "  WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 WATCHDOG_MAX_HOURS=<from above> \\"
echo "  WATCHDOG_RSYNC_EXCLUDES=\"--exclude=data/propensity/\" \\"
echo "  WATCHDOG_LOG=/tmp/watchdog_propensity.log bash $LOCAL/scripts/watchdog.sh"
