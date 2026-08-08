#!/bin/bash
# Launch one pod of the SFT token-scaling grid on RunPod. One pod per budget
# level (b37 / b75 / b149), each running BOTH arms at that level
# (rationale_<lvl> + stripped_<lvl>) — so every pod yields a complete matched
# rationale-vs-stripped contrast even if the other pods die.
#
#   LEVEL=b37  bash scripts/launch_sft_scaling_runpod.sh
#   LEVEL=b75  bash scripts/launch_sft_scaling_runpod.sh
#   LEVEL=b149 bash scripts/launch_sft_scaling_runpod.sh
#
# Cost/time per pod: 2 trains ~1h each + 2 clean-17 evals ~10h each
# (measured on the DPO rounds, same eval config) ≈ 22 h best case;
# A100-SXM4-80GB SECURE ≈ $1.49/h → ~$33/pod, ~$100 all three.
# MAX_HOURS=44 (2x best case) → $66/pod worst case.
set -e
LOCAL=/Users/julianquick/portfolio_copy/surface_tension
LEVEL="${LEVEL:?set LEVEL=b37|b75|b149|r8|r128}"
case "$LEVEL" in
  b37|b75|b149) ARMS="rationale_${LEVEL} stripped_${LEVEL}"; DSET_LEVEL=$LEVEL;;
  # rank-sweep pods: both arms at the top budget, non-default LoRA rank
  # (prereg/rank_sweep_2026-08-08.md; run_sft_scaling.sh parses the _rK suffix)
  r8|r128)      ARMS="rationale_b149_${LEVEL} stripped_b149_${LEVEL}"; DSET_LEVEL=b149;;
  *) echo "bad LEVEL=$LEVEL"; exit 1;;
esac
INSTANCE_FILE="$LOCAL/vast_scaling_${LEVEL}.env"
GPU="${GPU:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
HUB_PREFIX="${HUB_PREFIX:-kilojoules/surface-tension-scaling}"

# ---- gate 1: prereg must be externally timestamped before data generation --
PREREG="$LOCAL/prereg/sft_scaling_2026-08-06.md"
case "$LEVEL" in r8|r128) PREREG="$LOCAL/prereg/rank_sweep_2026-08-08.md";; esac
if grep -q '`____________`' "$PREREG"; then
    echo "REFUSING TO LAUNCH: $PREREG still has the external-timestamp"
    echo "placeholder. Register it (public gist / OSF / signed pushed tag) and"
    echo "record the link in the prereg first — repo policy (prereg/README.md)."
    exit 1
fi

# ---- gate 2: training sets must exist and match the committed manifest -----
python3 - "$DSET_LEVEL" <<'PY' || exit 1
import hashlib, json, sys
lvl = sys.argv[1]
root = "/Users/julianquick/portfolio_copy/surface_tension/data/sft_scaling"
m = json.load(open(f"{root}/manifest.json"))
for arm in ("rationale", "stripped"):
    name = f"{arm}_{lvl}"
    h = hashlib.sha256(open(f"{root}/{name}.jsonl", "rb").read()).hexdigest()
    assert h == m["sets"][name]["sha256"], f"{name}: sha mismatch vs manifest — regenerate"
    print(f"ok {name}: {m['sets'][name]['n_demos']} demos, "
          f"{m['sets'][name]['completion_chars']:,} chars")
PY

# ---- gate 3: balance precheck (manual disaster #3 from 2026-05-28) ---------
python3 - <<'PY' || true
import runpod, os
runpod.api_key = open(os.path.expanduser("~/.run.pod")).read().strip()
try:
    u = runpod.get_user()
    bal = u.get("clientBalance") if isinstance(u, dict) else None
    print(f"RunPod balance: {bal}")
except Exception as e:
    print(f"(balance check unavailable via SDK: {e} — CHECK THE CONSOLE MANUALLY)")
PY
read -p "Balance covers this pod's \$66 worst case (\$200 if launching all three)? [y/N] " ok
[ "$ok" = "y" ] || exit 1

echo "=== launch_sft_scaling LEVEL=$LEVEL (arms: $ARMS; gpu=$GPU cloud=$CLOUD) ==="

python3 "$LOCAL/scripts/runpod_launch.py" \
    --gpu "$GPU" --cloud "$CLOUD" --name "st-scal-$LEVEL" \
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
     pip install -q --upgrade 'torch==2.5.1' 2>&1 | tail -3
     pip install -q --upgrade transformers peft bitsandbytes accelerate 'datasets<4.0' pandas 2>&1 | tail -3
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; assert torch.__version__.startswith(\"2.5\"), (\"FATAL torch=\"+torch.__version__+\" expected 2.5.x; deps install failed\")' && python3 -c 'import transformers, peft, bitsandbytes, datasets'"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token" 2>/dev/null
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "uploading src + scripts + scaling sets + eval deck..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/{src,scripts,data/sft_scaling,results/raw,outputs,prereg}" 2>/dev/null
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
    "$PREREG" "root@$HOST:/workspace/st/prereg/" 2>/dev/null

echo "launching scaling pipeline (ARMS=$ARMS)..."
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && chmod +x scripts/run_sft_scaling.sh && \
     nohup env BASE_MODEL='google/gemma-4-31B-it' \
       HUGGING_FACE_HUB_TOKEN=\$(cat /root/.hf_token) \
       ARMS='$ARMS' HUB_PREFIX='$HUB_PREFIX' \
       bash scripts/run_sft_scaling.sh > /workspace/pipeline.log 2>&1 &
     echo \"pipeline launched, pid=\$!\"" </dev/null

echo ""
echo "=== LAUNCHED on RunPod (SFT scaling, $LEVEL) ==="
echo "  Pod: $INST   (ssh -p $PORT root@$HOST)"
echo "  Tail: ssh -p $PORT root@$HOST 'tail -f /workspace/pipeline.log'"
echo ""
echo "Now start the watchdog (44 h cap = 2x the 22 h best case; excludes are"
echo "INPUT-side only — every output must sync):"
echo "  PROVIDER=runpod WATCHDOG_INSTANCE_FILE=$INSTANCE_FILE \\"
echo "  WATCHDOG_INTERVAL=120 WATCHDOG_GRACE_MIN=30 WATCHDOG_STALL_MIN=120 \\"
echo "  WATCHDOG_MAX_HOURS=44 \\"
echo "  WATCHDOG_RSYNC_EXCLUDES=\"--exclude=problems_lcb_*.jsonl --exclude=sft_scaling/\" \\"
echo "  WATCHDOG_LOG=/tmp/watchdog_scaling_$LEVEL.log bash $LOCAL/scripts/watchdog.sh"
