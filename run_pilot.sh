#!/bin/bash
# Reproduce the pilot end-to-end. Resumable.
set -e
cd "$(dirname "$0")/src"
[ -f ../data/problems.jsonl ] || python loaders.py
python sweep.py --workers 16
python aggregate.py
echo "Outputs: results/pilot_summary.csv  results/pilot_summary.md"
