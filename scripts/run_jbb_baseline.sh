#!/bin/bash
# Pod-side runner for the JBB-prefill baseline check (the $2 step-0 of
# prereg/jbb_prefill_2026-09-03.md). Ships NO harmful content: it prompts with
# the plain (public JailbreakBench) behaviors and prefills only the benign
# "Sure, I can help with that." neutral prefix. Refusal-substring only; the
# 70B judge is not needed to answer the gate question.
set -uo pipefail
cd /workspace/st

echo "=== jbb baseline: $(python3 -c 'import json;print(json.load(open("data/jbb_prefill/baseline_items_highharm.json"))["n_behaviors"])') behaviors ==="
PYTHONPATH=src python3 -u src/jbb_prefill_gen.py \
  --mode baseline \
  --frozen data/jbb_prefill/baseline_items_highharm.json \
  --out results/raw/jbb_baseline.jsonl \
  --victim meta-llama/Llama-3.1-8B-Instruct \
  --max-new-tokens 512 \
  > /workspace/jbb_baseline.log 2>&1

touch /workspace/jbb_baseline_done
# No `sleep N && touch all_done` sentinel (the 2026-08-13 orphaned-sentinel
# race). The watchdog's post-process grace timer handles teardown.
