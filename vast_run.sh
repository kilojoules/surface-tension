#!/bin/bash
# Resilient orchestrator that runs ON the vast box.
# Writes heartbeats. Pushes adapter to HF Hub during training. Idempotent phase markers.
# NO self-destruct (we want logs to survive failures so we can debug).
#
# Required env (passed in by the launching SSH command):
#   HF_TOKEN     — for HF Hub auth (model download + adapter push)
#   HUB_REPO     — destination for adapter pushes, e.g. kilojoules/gemma-4-31b-st-dpo
#   RESUME       — "1" to pull adapter from HUB_REPO before training (after a crash)
#
# After completion: leaves /workspace/all_done. Instance kept alive — destroy manually.

set -uo pipefail
exec > /workspace/run.log 2>&1

LOG()  { echo "[$(date -u +%H:%M:%S)] $*"; }
PHASE(){ LOG ""; LOG "================ phase $1: $2 ================"; }
DONE() { touch "/workspace/phase_$1_done"; LOG ">>> phase $1 complete"; }
FAIL() { touch "/workspace/phase_$1_failed"; LOG "!!! phase $1 FAILED — instance kept alive for debug"; exit 1; }
HB()   { date -u +"%s\t%Y-%m-%dT%H:%M:%SZ\t$1" > /workspace/heartbeat.txt; }

# Heartbeat writer — runs forever in background, updates every 60s.
# Polled remotely via stream_log.sh + watch_heartbeat.sh.
HB "starting"
( while true; do HB "${CURRENT_PHASE:-running}"; sleep 60; done ) &
HB_PID=$!

cd /
[ -d /workspace ] || mkdir -p /workspace
cd /workspace

# ---- repo ----
export CURRENT_PHASE="0a-clone"
PHASE "0a" "clone"
[ -d st ] && (cd st && git pull) || git clone https://github.com/kilojoules/surface-tension.git st || FAIL "0a"
cd st
LOG "repo at $(git rev-parse HEAD)"
DONE "0a"

# ---- deps (vllm pinned to 0.19.1; no flash-attn; sdpa fallback in dpo_train.py) ----
export CURRENT_PHASE="0b-deps"
PHASE "0b" "deps"
if [ ! -f /workspace/phase_0b_done ]; then
  pip install --no-cache-dir -q "vllm==0.19.1" 2>&1 | tail -3 || FAIL "0b"
  pip install --no-cache-dir -q -r src/requirements_dpo.txt 2>&1 | tail -3 || FAIL "0b"
  pip show vllm trl peft 2>/dev/null | grep -E "^(Name|Version):" | paste - -
fi
DONE "0b"

# ---- problems data: skip the broken HF loader; expect data/problems_lcb.jsonl pre-uploaded ----
export CURRENT_PHASE="1b-problems"
PHASE "1b" "verify-problems"
if [ ! -s data/problems_lcb.jsonl ]; then
  LOG "data/problems_lcb.jsonl missing or empty — must be SCP'd up before launch"
  FAIL "1b"
fi
LOG "problems: $(wc -l < data/problems_lcb.jsonl) entries"
DONE "1b"

# ---- launch vLLM ----
export CURRENT_PHASE="1a-vllm-up"
PHASE "1a" "vllm-up"
setsid nohup vllm serve google/gemma-4-31B-it \
  --port 8000 --host 0.0.0.0 \
  --dtype bfloat16 --max-model-len 8192 \
  --api-key sk-vast-local --gpu-memory-utilization 0.92 \
  > /workspace/vllm.log 2>&1 < /dev/null &
VLLM_PID=$!
echo "$VLLM_PID" > /workspace/vllm.pid
LOG "vllm pid $VLLM_PID; waiting for /v1/models (up to 10 min)..."
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/v1/models -H "Authorization: Bearer sk-vast-local" >/dev/null 2>&1; then
    LOG "vllm up after ${i}x10s"; break
  fi
  if [ "$i" = "60" ]; then tail -40 /workspace/vllm.log; FAIL "1a"; fi
  sleep 10
done
DONE "1a"

# ---- phase 1c: sweep on all problems ----
export CURRENT_PHASE="1c-sweep"
PHASE "1c" "sweep n=6"
cd src
RUNNER=http MODEL_ENDPOINT=http://localhost:8000/v1 MODEL_API_KEY=sk-vast-local \
MODEL_NAME=google/gemma-4-31B-it MODEL_MAX_TOKENS=2048 \
python sweep.py \
  --problems ../data/problems_lcb.jsonl \
  --csv ../results/raw/pilot_v5_raw.csv \
  --source-dir ../results/raw/sources_v5 \
  --workers 8 --n-samples 6 \
  --constraints no_loops_no_recursion 2>&1 | tail -30 || FAIL "1c"
DONE "1c"

# ---- phase 2: build DPO triples ----
export CURRENT_PHASE="2-build-dpo"
PHASE "2" "build-dpo-dataset"
python build_dpo_dataset.py \
  --csv ../results/raw/pilot_v5_raw.csv \
  --sources-dir ../results/raw/sources_v5 \
  --constraint no_loops_no_recursion \
  --holdout-frac 0.3 2>&1 | tail -8 || FAIL "2"
DONE "2"

# ---- phase 3a: kill vLLM (full process group) ----
export CURRENT_PHASE="3a-kill-vllm"
PHASE "3a" "kill-vllm"
for pgid in $(ps -eo pgid,cmd | grep -E "vllm serve" | grep -v grep | awk '{print $1}' | sort -u); do
  LOG "SIGKILL pgid $pgid"; kill -KILL "-$pgid" 2>/dev/null || true
done
pkill -KILL -f "vllm serve" 2>/dev/null || true
sleep 30
nvidia-smi --query-gpu=memory.used --format=csv,noheader
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$mem" -gt 10000 ] && { LOG "GPU still holding ${mem} MiB; waiting more"; sleep 60; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
DONE "3a"

# ---- phase 3b: DPO training. Pushes adapter to HF Hub every save_steps. ----
export CURRENT_PHASE="3b-dpo-train"
PHASE "3b" "dpo-train"
HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
HUB_REPO="${HUB_REPO:-}" \
RESUME_FROM_HUB="${RESUME:-}" \
python dpo_train.py 2>&1 | tee /workspace/dpo_train.log || FAIL "3b"
DONE "3b"

# ---- phase 4: eval the adapter ----
export CURRENT_PHASE="4a-vllm-lora"
PHASE "4a" "vllm-up-with-adapter"
ADAPTER_DIR=/workspace/st/outputs/dpo_run1/final_adapter
[ -d "$ADAPTER_DIR" ] || { ls /workspace/st/outputs/dpo_run1/ || true; FAIL "4a"; }
setsid nohup vllm serve google/gemma-4-31B-it \
  --port 8000 --host 0.0.0.0 \
  --dtype bfloat16 --max-model-len 8192 \
  --api-key sk-vast-local --gpu-memory-utilization 0.92 \
  --enable-lora --lora-modules dpo=$ADAPTER_DIR \
  > /workspace/vllm_dpo.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/v1/models -H "Authorization: Bearer sk-vast-local" >/dev/null 2>&1; then
    LOG "vllm-lora up after ${i}x10s"; break
  fi
  [ "$i" = "60" ] && { tail -40 /workspace/vllm_dpo.log; FAIL "4a"; }
  sleep 10
done
DONE "4a"

export CURRENT_PHASE="4b-eval"
PHASE "4b" "eval-sweep on adapter"
RUNNER=http MODEL_ENDPOINT=http://localhost:8000/v1 MODEL_API_KEY=sk-vast-local \
MODEL_NAME=dpo MODEL_MAX_TOKENS=2048 \
python sweep.py \
  --problems ../data/problems_lcb.jsonl \
  --csv ../results/raw/pilot_v5_dpo_raw.csv \
  --source-dir ../results/raw/sources_v5_dpo \
  --workers 8 --n-samples 3 \
  --constraints no_loops_no_recursion 2>&1 | tail -20 || FAIL "4b"
DONE "4b"

export CURRENT_PHASE="4c-aggregate"
PHASE "4c" "aggregate"
python aggregate.py \
  --csv ../results/raw/pilot_v5_dpo_raw.csv \
  --summary-csv ../results/pilot_v5_dpo_summary.csv \
  --summary-md ../results/pilot_v5_dpo_summary.md \
  --sources-dir ../results/raw/sources_v5_dpo 2>&1 | tail -3 || FAIL "4c"
DONE "4c"

LOG ""
LOG "================ ALL PHASES COMPLETE ================"
HB "done"
touch /workspace/all_done

# Stop the heartbeat writer
kill "$HB_PID" 2>/dev/null || true
