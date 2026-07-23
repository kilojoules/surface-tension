#!/bin/bash
# DPO pass-round (pair policy pass-v2): put gradient on PASSING within
# compliance, starting from DPO-r1 — the direct A/B against DPO-r2, which
# started from the same checkpoint on the same 45-problem pool but with the
# compliance-v1 policy and regressed (cheating -6pts bought with -8pts pass,
# -13pts compliance; Pareto-dominated).
#
# Pre-registered predictions & decision rules: prereg/dpo_pass_round_*.md —
# EXTERNALLY TIMESTAMP IT (OSF/Zenodo or signed pushed tag) BEFORE launching;
# that is repo policy (prereg/README.md).
#
# Runs ON the pod (bash scripts/run_dpo_pass_round.sh from the repo root).
# Prereqs: repo at /workspace/st, deps from src/requirements_dpo.txt,
# HF_TOKEN exported. Phases are idempotent via /workspace/passround_*_done
# markers, so a crashed pod can resume by re-running the script.
#
# ---- LAUNCH CHECKLIST (all three items have burned money before) ----------
#  1. Watchdog MAX_HOURS >= 40  (best case ~19 h: pairs ~4.5h + train ~1h +
#     val eval ~3h + clean eval ~10h; >=2x best case per checklist).
#  2. rsync excludes are INPUT-side only. These OUTPUTS must sync
#     continuously (the judgments.jsonl loss):
#       /workspace/st/data/problems_dpo_pass45.jsonl
#       /workspace/st/data/dpo_pairs_pass.jsonl
#       /workspace/st/outputs/dpo_pass/
#       /workspace/st/results/raw/eval_dpo_pass_*  (csv + sources_*/ dirs)
#       /workspace/*.log /workspace/st/*.log
#  3. Provider balance precheck; monitor ~every 5 min during model load and
#     the first optimizer steps (assert grad/loss movement — the zero-gradient
#     LoRA no-op happened before).
# ---------------------------------------------------------------------------

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
LOG() { echo "[$(date -u +%H:%M:%S)] $*"; }
MARK=/workspace; [ -d /workspace ] || MARK=$ROOT   # marker dir (pod vs local dry-run)

START_ADAPTER=${START_ADAPTER:-kilojoules/surface-tension-dpo-r1-r32-final}
HUB_REPO=${HUB_REPO:-kilojoules/surface-tension-dpo-pass-r32}
N_SAMPLES=${N_SAMPLES:-8}
MAX_NEW=${MAX_NEW:-2048}
TEMP=${TEMP:-0.9}

cd src

# ---- STAGE 0: materialize the r2-matched 45-problem pool -------------------
if [ ! -f "$MARK/passround_0_done" ]; then
  LOG "========== STAGE 0: materialize pool (45 ids from the r2 round) =========="
  python ../scripts/make_dpo_pass_pool.py --out ../data/problems_dpo_pass45.jsonl || exit 1
  touch "$MARK/passround_0_done"
fi

# ---- STAGE 1: build pairs under pass-v2 (sample DPO-r1, T=0.9, n=8) --------
if [ ! -f "$MARK/passround_1_done" ]; then
  LOG "========== STAGE 1: build pairs (policy pass-v2, sample $START_ADAPTER) =========="
  python build_dpo_pairs.py \
      --problems ../data/problems_dpo_pass45.jsonl \
      --out ../data/dpo_pairs_pass.jsonl \
      --adapter "$START_ADAPTER" \
      --pair-policy pass-v2 \
      --n-samples "$N_SAMPLES" --max-new-tokens "$MAX_NEW" --temperature "$TEMP" \
      2>&1 | tee ../dpo_pairs_pass.log || exit 1
  # sanity gates: enough pairs, and the new pair type actually present
  python - <<'PY' || exit 1
import json, sys
pairs = [json.loads(l) for l in open("../data/dpo_pairs_pass.jsonl")]
probs = {p["problem_id"] for p in pairs}
print(f"[gate] {len(pairs)} pairs from {len(probs)} problems")
if len(pairs) < 60 or len(probs) < 12:
    sys.exit("FATAL: pair supply below floor (60 pairs / 12 problems) — "
             "do not train on this; investigate sampling first.")
PY
  touch "$MARK/passround_1_done"
fi

# ---- STAGE 2: DPO from DPO-r1, reference anchored to DPO-r1 ----------------
# Hyperparameters mirror the r1/r2 rounds exactly (launch_dpo_round_runpod.sh):
# beta=0.1, lr=5e-6, 3 epochs, max_length 2048 — only the pair policy differs.
if [ ! -f "$MARK/passround_2_done" ]; then
  LOG "========== STAGE 2: dpo_train (ADAPTER_INIT=$START_ADAPTER, beta=0.1, 3 epochs) =========="
  ADAPTER_INIT="$START_ADAPTER" \
  DPO_TRAIN=../data/dpo_pairs_pass.jsonl \
  DPO_OUTPUT=../outputs/dpo_pass \
  DPO_BETA=${DPO_BETA:-0.1} DPO_LR=${DPO_LR:-5e-6} \
  DPO_EPOCHS=${DPO_EPOCHS:-3} MAX_LENGTH=${MAX_LENGTH:-2048} \
  python dpo_train.py 2>&1 | tee ../dpo_train_pass.log || exit 1
  [ -d ../outputs/dpo_pass/final_adapter ] || { LOG "FATAL: no final_adapter"; exit 1; }
  timeout 900 python push_adapter.py ../outputs/dpo_pass/final_adapter "${HUB_REPO}-final" \
      2>&1 | tee ../push_dpo_pass.log || LOG "WARN: Hub push failed (adapter still on disk)"
  touch "$MARK/passround_2_done"
fi

# ---- STAGE 3: bare-prompt evals (same protocol as r1/r2 rounds) ------------
# 3a val-12 (early signal, ~3 h) then 3b clean-17 (the reported set, ~10 h).
ADAPTER_DIR=../outputs/dpo_pass/final_adapter
[ -d "$ADAPTER_DIR" ] || { LOG "FATAL: $ADAPTER_DIR missing"; exit 1; }

if [ ! -f "$MARK/passround_3a_done" ] && [ ! -f ../data/problems_lcb_val12.jsonl ]; then
  LOG "STAGE 3a SKIPPED: ../data/problems_lcb_val12.jsonl not present (the val-12"
  LOG "file was never re-synced; clean-17 below is the reported set). To include"
  LOG "it, materialize the val set first."
  touch "$MARK/passround_3a_done"
fi
# Eval config mirrors the r1/r2 rounds: BARE prompt only (`--constraints` with
# no args clears the constrained condition — compliance here means the trained
# DEFAULT), 4-bit with wrapper stripping (the zero-gradient/no-op trap).
if [ ! -f "$MARK/passround_3a_done" ]; then
  LOG "========== STAGE 3a: eval val (12 problems, n=8, max_new=3072) =========="
  LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python sweep_local.py \
      --problems ../data/problems_lcb_val12.jsonl \
      --csv ../results/raw/eval_dpo_pass_val.csv \
      --source-dir ../results/raw/sources_eval_dpo_pass_val \
      --base-model "${BASE_MODEL:-google/gemma-4-31B-it}" --adapter "$ADAPTER_DIR" \
      --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
      2>&1 | tee ../eval_dpo_pass_val.log || exit 1
  python recheck_eval.py ../results/raw/eval_dpo_pass_val.csv | tee -a ../eval_dpo_pass_val.log
  touch "$MARK/passround_3a_done"
fi

if [ ! -f "$MARK/passround_3b_done" ]; then
  LOG "========== STAGE 3b: eval clean (17 problems, n=8, max_new=3072) =========="
  LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python sweep_local.py \
      --problems ../data/problems_lcb_clean17.jsonl \
      --csv ../results/raw/eval_dpo_pass_clean.csv \
      --source-dir ../results/raw/sources_eval_dpo_pass_clean \
      --base-model "${BASE_MODEL:-google/gemma-4-31B-it}" --adapter "$ADAPTER_DIR" \
      --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
      2>&1 | tee ../eval_dpo_pass_clean.log || exit 1
  python recheck_eval.py ../results/raw/eval_dpo_pass_clean.csv | tee -a ../eval_dpo_pass_clean.log
  touch "$MARK/passround_3b_done"
fi

LOG "ALL DONE. Compare against DPO-r1 (0.65 / 0.32 / 0.07) and DPO-r2"
LOG "(0.52 / 0.30 / 0.01) on compliance / cmp-and-pass / cheating, clean-17."
LOG "Keep the pod until every output above has synced off it."
touch "$MARK/passround_all_done"
