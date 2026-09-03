#!/bin/bash
# Pod-side JUDGE-ONLY runner: scores pre-existing generations (uploaded from a
# prior run) with the 70B (bnb 4-bit) under the Qi 1-5 rubric. No generation.
set -uo pipefail
cd /workspace/st
echo "=== JUDGE-ONLY: $(wc -l < results/raw/jbb_full.jsonl) generations ==="
PYTHONPATH=src python3 -u src/jbb_prefill_judge.py \
  --gens results/raw/jbb_full.jsonl \
  --out results/raw/jbb_full_judged.jsonl \
  --openings-out results/raw/jbb_openings.json \
  > /workspace/jbb_judge.log 2>&1
echo "  judged rows: $(wc -l < results/raw/jbb_full_judged.jsonl)"
touch /workspace/jbb_judge_done /workspace/all_stages_done
