#!/bin/bash
# Pod-side INCREMENT runner: adds N=10,15 to an existing N=5,20 dataset.
# Seeded with the prior jbb_full.jsonl / jbb_full_judged.jsonl / jbb_openings.json,
# the resumable gen + judge runners produce ONLY the new N=10,15 cells and append
# them, yielding a unified 4-length dataset. Frozen items carry all 4 lengths;
# the 5/20 prefixes are byte-identical to the prior freeze (verified), so the
# seed rows stay valid.
set -uo pipefail
cd /workspace/st
echo "=== INCREMENT: seed rows gen=$(wc -l < results/raw/jbb_full.jsonl 2>/dev/null || echo 0) judged=$(wc -l < results/raw/jbb_full_judged.jsonl 2>/dev/null || echo 0) ==="

echo "--- STAGE A: generate N=10,15 (8B victim; resumes past N=5,20) ---"
PYTHONPATH=src python3 -u src/jbb_prefill_gen.py \
  --mode full --frozen data/jbb_prefill/frozen_private/frozen.json \
  --out results/raw/jbb_full.jsonl --victim meta-llama/Llama-3.1-8B-Instruct \
  --max-new-tokens 512 > /workspace/jbb_gen.log 2>&1
NGEN=$(wc -l < results/raw/jbb_full.jsonl)
echo "  total generations now: $NGEN (expect 1157 = 623 + 534)"
touch /workspace/jbb_gen_done

echo "--- STAGE B: judge (70B; resumes past already-judged) ---"
PYTHONPATH=src python3 -u src/jbb_prefill_judge.py \
  --gens results/raw/jbb_full.jsonl \
  --out results/raw/jbb_full_judged.jsonl \
  --openings-out results/raw/jbb_openings.json > /workspace/jbb_judge.log 2>&1
NJUD=$(wc -l < results/raw/jbb_full_judged.jsonl 2>/dev/null || echo 0)
echo "  total judged now: $NJUD / $NGEN"
if [ "$NJUD" -ge $(( NGEN * 95 / 100 )) ]; then
    touch /workspace/jbb_judge_done /workspace/all_stages_done
    echo "  DONE ($NJUD rows)"
else
    echo "  INCOMPLETE ($NJUD/$NGEN) — no done-marker; judge crashed?"
    grep -vE 'Loading weights|Fetching' /workspace/jbb_judge.log | tail -5 || true
    exit 1
fi
