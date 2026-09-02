#!/bin/bash
# Pod-side runner for prereg/step0_substitution_power_2026-09-02.md.
#
# Extends ONE step-0 prefill cell from n=3 to n=8 samples/problem (the /136
# convention). The pod is seeded with the committed 51-row CSV + its sources,
# so sweep_local's resume key (problem_id, constraint, condition, sample_idx)
# leaves exactly 85 new generations pending. Recipe is byte-for-byte the
# 2026-08-13 step-0 runners; only --n-samples changes.
#
# ARM=rsft|vanilla selects the cell.
set -uo pipefail
cd /workspace/st/src

case "${ARM:?set ARM=rsft|vanilla}" in
  rsft)    AD=kilojoules/surface-tension-sft-rationale-r32-final
           TAG=eval_step0_prefill ;;
  vanilla) AD=kilojoules/surface-tension-sft-rankcurve-r32-final
           TAG=eval_step0b_sft_prefill ;;
  *)       echo "FATAL: unknown ARM=$ARM"; exit 1 ;;
esac

echo "=== arm=$ARM adapter=$AD tag=$TAG ==="
echo "seeded rows: $(wc -l < ../results/raw/$TAG.csv) (expect 52 = header + 51)"
echo "seeded sources: $(ls ../results/raw/sources_$TAG | wc -l) (expect 102)"

LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4 python -u sweep_local.py \
  --problems ../data/problems_lcb_clean17.jsonl \
  --csv ../results/raw/$TAG.csv \
  --source-dir ../results/raw/sources_$TAG \
  --base-model google/gemma-4-31B-it --adapter $AD \
  --n-samples 8 --max-new-tokens 3072 --temperature 0.7 --constraints \
  --prefill "$(printf '\x60\x60\x60python\n')" \
  > /workspace/${ARM}_power.log 2>&1

python recheck_eval.py ../results/raw/$TAG.csv >> /workspace/${ARM}_power.log 2>&1
touch /workspace/${ARM}_done

# Deliberately NO `sleep N && touch /workspace/all_done`: on 2026-08-13 a
# crashed runner's orphaned sleep-subshell re-created that sentinel after
# manual cleanup and the watchdog destroyed a freshly relaunched pod. The
# watchdog's own post-process grace timer handles teardown instead.
