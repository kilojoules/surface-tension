#!/bin/bash
# Pilot v2: LiveCodeBench medium (post-cutoff) × 4 stronger constraints.
# Run after Gemini quota resets (~14:18 local on 2026-04-26).
#
# Design choices:
# - Benchmark: LCB stdin-mode medium, contest_date >= 2024-06-01 (post-Flash-cutoff).
#   ~57 problems. Each has 8 tests (public + first private up to 8).
# - Constraints: no_recursion, no_loops, no_helpers, stdlib_whitelist.
#   The 3 weak constraints from v1 (no_classes/no_nested_functions/no_mutation) are dropped —
#   v1 showed they were trivially compliant at 1.0 with no binding signal.
# - Workers: 4 (vs 16 in v1) to stay under per-minute throttles. v1 burned 56% of attempts to 429s.
# - Calls: 57 × (3 unconstrained + 4 constraints × 3 samples) = 57 × 15 = 855. Fits in free-tier daily.

set -e
cd "$(dirname "$0")/src"

# Use flash-lite (2.5-flash quota was exhausted on initial v1 run).
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"

[ -f ../data/problems_lcb.jsonl ] || python loaders_lcb.py

python sweep.py \
  --problems ../data/problems_lcb.jsonl \
  --csv ../results/raw/pilot_v2_raw.csv \
  --source-dir ../results/raw/sources_v2 \
  --workers 4 \
  --constraints no_recursion no_loops no_helpers stdlib_whitelist

python aggregate.py \
  --csv ../results/raw/pilot_v2_raw.csv \
  --summary-csv ../results/pilot_v2_summary.csv \
  --summary-md ../results/pilot_v2_summary.md \
  --sources-dir ../results/raw/sources_v2

echo "Done. See results/pilot_v2_summary.md"
