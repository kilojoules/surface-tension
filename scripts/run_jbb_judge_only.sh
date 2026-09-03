#!/bin/bash
# Pod-side JUDGE-ONLY runner: scores pre-existing generations with the 70B
# (bnb 4-bit) under the Qi 1-5 rubric. No generation.
set -uo pipefail
cd /workspace/st
NGEN=$(wc -l < results/raw/jbb_full.jsonl)
echo "=== JUDGE-ONLY: $NGEN generations ==="
PYTHONPATH=src python3 -u src/jbb_prefill_judge.py \
  --gens results/raw/jbb_full.jsonl \
  --out results/raw/jbb_full_judged.jsonl \
  --openings-out results/raw/jbb_openings.json \
  > /workspace/jbb_judge.log 2>&1
NJUD=$(wc -l < results/raw/jbb_full_judged.jsonl 2>/dev/null || echo 0)
echo "  judged rows: $NJUD / $NGEN"
# Only declare done if judging actually completed (>= 95% of gens judged). A
# crash (network/DNS/OOM) leaves NO done-marker, so the watchdog/monitor treat
# it as incomplete instead of a false completion.
if [ "$NJUD" -ge $(( NGEN * 95 / 100 )) ]; then
    touch /workspace/jbb_judge_done /workspace/all_stages_done
    echo "  DONE ($NJUD rows)"
else
    echo "  INCOMPLETE ($NJUD/$NGEN) — no done-marker written; judge crashed?"
    tail -5 /workspace/jbb_judge.log | grep -vE 'Loading weights|Fetching' || true
    exit 1
fi
