#!/bin/bash
# Retry the missing 1220 generations after Gemini quota resets.
# - Strips 429-failed rows from the CSV (already done once on the initial run).
# - Re-runs sweep with workers=4 to stay well under per-minute throttles.
# - Re-aggregates.
#
# Quota reset estimate: 23h17m from initial exhaustion at ~14:32 local on 2026-04-25,
# i.e. roughly 2026-04-26 13:50 local. Run after that.

set -e
cd "$(dirname "$0")/src"

CSV=../results/raw/pilot_raw.csv
BACKUP=../results/raw/pilot_raw.before_retry.csv

cp "$CSV" "$BACKUP"

# Strip any 429 / cli_exit / cli_timeout rows that may have accumulated, so resume picks them up.
python3 - <<EOF
import csv
src = "$BACKUP"
dst = "$CSV"
with open(src, newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)
def bad(r):
    e = r.get('gen_error') or ''
    return 'QUOTA' in e or 'cli_exit' in e or 'cli_timeout' in e
kept = [r for r in rows if not bad(r)]
print(f'kept {len(kept)} of {len(rows)}; will retry {len(rows) - len(kept)} (plus any never-attempted)')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in kept: w.writerow(r)
EOF

# Lower worker count to avoid retripping per-minute throttle
python sweep.py --workers 4

python aggregate.py
echo "Done. See results/pilot_summary.md"
