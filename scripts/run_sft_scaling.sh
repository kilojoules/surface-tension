#!/bin/bash
# SFT token-scaling grid, one pod's share: train + eval each arm named in $ARMS
# (space-separated names matching data/sft_scaling/<name>.jsonl, e.g.
# "rationale_b149 stripped_b149"). The launcher runs one pod per budget level
# so every pod yields a complete matched rationale-vs-stripped contrast even
# if the other pods die.
#
# Recipe is the ORIGINAL R-SFT recipe, byte-for-byte
# (launch_rationale_sft_pipeline_runpod.sh): r=32 alpha=64 dropout=0 lr=1e-4
# epochs=20 linear-decay max_length=2048. Fixed across all budgets — tokens
# SEEN scales with unique tokens; that is the quantity under study (prereg).
# Deviation from the original run: VAL_EVERY=0 (no best-val tracking; -final
# is the deployed convention and val tracking does not alter training).
#
# Pre-registered: prereg/sft_scaling_2026-08-06.md — externally timestamped
# BEFORE launch (launcher gates on it).
#
# LAUNCH CHECKLIST (each item has burned money before):
#   1. Watchdog MAX_HOURS >= 2x best case. Best case/pod: 2 trains ~2h +
#      2 evals ~20h = ~22h  ->  MAX_HOURS=44.
#   2. rsync excludes INPUT-side only; outputs that must sync continuously:
#        /workspace/st/outputs/scaling_*/  /workspace/st/results/raw/eval_scaling_*
#        /workspace/*.log /workspace/st/*.log
#   3. Balance precheck; watch the first optimizer steps for loss movement
#      (the zero-gradient LoRA no-op happened before; STRIP_WRAPPERS=1 is the
#      fix and is sft_train.py's default).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
LOG() { echo "[$(date -u +%H:%M:%S)] $*"; }
MARK=/workspace; [ -d /workspace ] || MARK=$ROOT

: "${ARMS:?set ARMS, e.g. ARMS='rationale_b149 stripped_b149'}"
BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
HUB_PREFIX=${HUB_PREFIX:-kilojoules/surface-tension-scaling}

cd src
for NAME in $ARMS; do
  TRAIN_SET=../data/sft_scaling/${NAME}.jsonl
  [ -f "$TRAIN_SET" ] || { LOG "FATAL: $TRAIN_SET missing"; exit 1; }
  OUT=../outputs/scaling_${NAME}

  if [ ! -f "$MARK/scal_${NAME}_train_done" ]; then
    LOG "========== TRAIN $NAME (original R-SFT recipe, 20 epochs) =========="
    BASE_MODEL="$BASE_MODEL" \
    SFT_TRAIN="$TRAIN_SET" SFT_OUTPUT="$OUT" \
    LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.0 \
    SFT_LR=1e-4 SFT_EPOCHS=20 LR_SCHEDULE=linear \
    MAX_LENGTH=2048 MAX_PROMPT_LENGTH=768 \
    EVAL_EVERY=0 EVAL_N=0 VAL_EVERY=0 VAL_N=0 LOG_EVERY=5 \
    python -u sft_train.py 2>&1 | tee ../sft_scaling_${NAME}.log || exit 1
    [ -d "$OUT/final_adapter" ] || { LOG "FATAL: no final_adapter for $NAME"; exit 1; }
    # the zero-gradient trap: a no-op run leaves the loss flat — refuse to continue
    python - "$NAME" <<'PY' || exit 1
import re, sys
name = sys.argv[1]
txt = open(f"../sft_scaling_{name}.log").read()
losses = [float(m) for m in re.findall(r"loss=([0-9]+\.[0-9]+)", txt)]
grads = [float(m) for m in re.findall(r"grad_norm=([0-9.]+e[+-][0-9]+)", txt)]
assert len(losses) >= 2, "no loss lines parsed from the train log"
first, last = losses[0], losses[-1]
print(f"[gate] {name}: loss {first:.4f} -> {last:.4f}; max grad_norm {max(grads):.3e}")
assert max(grads) > 0, "FATAL: grad_norm never above 0 — the zero-gradient no-op"
assert last < first * 0.9, (
    f"FATAL: loss barely moved ({first:.4f} -> {last:.4f}) — "
    "suspect the zero-gradient no-op; do not eval this adapter.")
PY
    touch "$MARK/scal_${NAME}_train_done"
  fi

  if [ ! -f "$MARK/scal_${NAME}_push_done" ]; then
    LOG "========== PUSH $NAME -> ${HUB_PREFIX}-${NAME//_/-}-final =========="
    timeout 900 python push_adapter.py "$OUT/final_adapter" "${HUB_PREFIX}-${NAME//_/-}-final" \
        2>&1 | tee ../push_scaling_${NAME}.log \
      || LOG "WARN: Hub push failed for $NAME (adapter still on disk)"
    touch "$MARK/scal_${NAME}_push_done"
  fi

  # Eval mirrors every DPO/SFT round: clean-17, BARE prompt (`--constraints`
  # with no args), n=8, max_new 3072, T=0.7, 4-bit with wrapper stripping.
  if [ ! -f "$MARK/scal_${NAME}_eval_done" ]; then
    LOG "========== EVAL $NAME on clean-17 (n=8, max_new=3072) =========="
    LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
        --problems ../data/problems_lcb_clean17.jsonl \
        --csv ../results/raw/eval_scaling_${NAME}.csv \
        --source-dir ../results/raw/sources_eval_scaling_${NAME} \
        --base-model "$BASE_MODEL" --adapter "$OUT/final_adapter" \
        --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
        2>&1 | tee ../eval_scaling_${NAME}.log || exit 1
    python recheck_eval.py ../results/raw/eval_scaling_${NAME}.csv \
        | tee -a ../eval_scaling_${NAME}.log
    touch "$MARK/scal_${NAME}_eval_done"
  fi
done

LOG "ALL ARMS DONE on this pod: $ARMS"
LOG "Keep the pod until every output above has synced off it."
touch "$MARK/scal_all_done"
# watchdog.sh auto-destroys (after a final sync) on this sentinel:
sleep 300 && touch "$MARK/all_done"
