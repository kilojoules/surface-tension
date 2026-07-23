"""Materialize the DPO pass-round problem pool: the EXACT 45-problem pool the
r2 round sampled (so DPO-pass vs DPO-r2 isolates the pair policy — same
starting checkpoint, same problems, different objective).

The r2 pool file itself was never synced off the pod (input-side rsync
excludes); the 45 problem ids below are reconstructed from the r2 round's
build log (vast_logs/va1c66tygkxlcc/pipeline.log, lines "[i/45] lcb/...").
This script re-materializes the pool from LiveCodeBench via the repo's own
loader and refuses to write anything partial.

Run from src/ (or with src/ on PYTHONPATH), typically on the pod:
    python ../scripts/make_dpo_pass_pool.py --out ../data/problems_dpo_pass45.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from loaders_lcb import load_lcb_problems  # noqa: E402

# From the r2 round log, in log order.
POOL_IDS = [
    "lcb/1883_B", "lcb/1883_C", "lcb/abc301_c", "lcb/abc301_d", "lcb/abc302_b",
    "lcb/abc302_c", "lcb/abc302_d", "lcb/abc303_c", "lcb/abc303_d", "lcb/abc304_c",
    "lcb/abc304_d", "lcb/abc305_c", "lcb/abc306_c", "lcb/abc306_d", "lcb/abc307_c",
    "lcb/abc307_d", "lcb/abc308_c", "lcb/abc308_d", "lcb/abc309_c", "lcb/abc309_d",
    "lcb/abc310_c", "lcb/abc310_d", "lcb/abc311_c", "lcb/abc311_d", "lcb/abc312_c",
    "lcb/abc312_d", "lcb/abc313_b", "lcb/abc313_c", "lcb/abc314_c", "lcb/abc314_d",
    "lcb/abc315_c", "lcb/abc315_d", "lcb/abc318_c", "lcb/abc318_d", "lcb/abc319_c",
    "lcb/abc319_d", "lcb/abc320_c", "lcb/abc320_d", "lcb/abc321_c", "lcb/abc321_d",
    "lcb/abc322_c", "lcb/abc322_d", "lcb/abc323_c", "lcb/abc324_c", "lcb/abc325_b",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/problems_dpo_pass45.jsonl")
    args = ap.parse_args()

    assert len(POOL_IDS) == 45 and len(set(POOL_IDS)) == 45
    by_id = {p["id"]: p for p in load_lcb_problems()}
    missing = [i for i in POOL_IDS if i not in by_id]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} pool ids not found in the LCB "
                         f"loader output (first: {missing[:3]}). Refusing to "
                         "write a partial pool — the r2 comparison needs all 45.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for pid in POOL_IDS:              # keep the r2 log order
            f.write(json.dumps(by_id[pid]) + "\n")
    print(f"wrote {len(POOL_IDS)} problems to {args.out}")


if __name__ == "__main__":
    main()
