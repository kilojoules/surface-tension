"""Build a unified eval manifest from the merged cohort.

Reads `paper/data/cohort/labels_merged.jsonl` to get the cohort id list, then
loads the corresponding problem records from whichever source they came from
(LCB-medium, LCB-hard, or trees), and writes a single
`paper/data/cohort/problems_eval.jsonl` for the eval harness.

This is the **one code path for medium, hard, and trees** the brief requires.
The output preserves `runtime_mode` so the harness can branch on
"stdin" (LCB) vs "stdin-with-prelude" (trees). No per-slice hard-coding —
which problems are in the cohort comes from the labels file alone, so adding
or dropping any problem only requires editing `labels_*.jsonl`.

Run from repo root:
    python src/build_eval_manifest.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "paper" / "data" / "cohort"

# Source -> (jsonl_path, default_runtime_mode).
SOURCES = {
    "lcb-medium":     (ROOT / "data" / "problems_lcb.jsonl",                  "stdin"),
    "lcb-hard":       (ROOT / "data" / "problems_lcb_hard.jsonl",             "stdin"),
    "tree-classic":   (COHORT / "problems_trees.jsonl",                       None),
    "tree-perturbed": (COHORT / "problems_trees.jsonl",                       None),
    "tree-synthetic": (COHORT / "problems_trees.jsonl",                       None),
}


def _index_problems(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def main() -> None:
    labels = []
    with (COHORT / "labels_merged.jsonl").open() as f:
        for line in f:
            labels.append(json.loads(line))

    # Lazy-index each source on first reference.
    src_index: dict[str, dict[str, dict]] = {}

    eval_records = []
    missing = []
    for lab in labels:
        src = lab["source"]
        if src not in SOURCES:
            raise ValueError(f"unknown source {src!r} on {lab['id']}")
        if src not in src_index:
            src_index[src] = _index_problems(SOURCES[src][0])
        problems = src_index[src]
        pid = lab["id"]
        if pid not in problems:
            missing.append((pid, src, str(SOURCES[src][0].relative_to(ROOT))))
            continue
        rec = dict(problems[pid])
        # Ensure runtime_mode is set; tree records already carry it.
        rec.setdefault("runtime_mode", SOURCES[src][1] or "stdin")
        # Attach the cohort label (so the eval harness knows the cell, ceiling,
        # quadrant-eligibility, etc., without a separate join later).
        rec["cohort_label"] = lab
        eval_records.append(rec)

    out_path = COHORT / "problems_eval.jsonl"
    with out_path.open("w") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote: {out_path.relative_to(ROOT)}  (n={len(eval_records)})")

    # Sanity report
    by_mode = defaultdict(int)
    by_source = defaultdict(int)
    for r in eval_records:
        by_mode[r["runtime_mode"]] += 1
        by_source[r["cohort_label"]["source"]] += 1
    print("  by runtime_mode:", dict(by_mode))
    print("  by source:      ", dict(by_source))
    if missing:
        print()
        print(f"MISSING ({len(missing)}):")
        for pid, src, where in missing:
            print(f"  {pid}  ({src})  not in {where}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
