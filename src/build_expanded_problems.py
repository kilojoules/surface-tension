"""Build the expanded problem set for the data-scaling ablation.

Loads HumanEval (164 problems, full) and MBPP-sanitized (full = train+validation+
test+prompt = 427 problems) using the existing loaders. Filters out any problem
already used in problems_mbpp30.jsonl (the held-out cross-benchmark eval set)
to keep eval clean. Writes data/problems_expanded.jsonl.

The output schema matches loaders._humaneval_problem and loaders._mbpp_problem.
Used as the input to the data-mining sweep that expands the SFT pool.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset

from loaders import _humaneval_problem, _mbpp_problem


def main():
    out_dir = os.environ.get("OUT_DIR", "../data")
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(here, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load HumanEval (164 problems)
    he_rows = list(load_dataset("openai_humaneval", split="test"))
    he_problems = [_humaneval_problem(r) for r in he_rows]
    print(f"HumanEval: {len(he_problems)} problems")

    # Load MBPP-sanitized across all splits
    mbpp_problems: List[Dict] = []
    for split in ["test", "train", "validation", "prompt"]:
        try:
            rows = list(load_dataset("mbpp", "sanitized", split=split))
            split_problems = [p for p in (_mbpp_problem(r) for r in rows) if p is not None]
            mbpp_problems.extend(split_problems)
            print(f"MBPP-sanitized {split}: {len(rows)} rows -> {len(split_problems)} usable")
        except Exception as e:
            print(f"MBPP {split} failed: {type(e).__name__}: {e}")

    # Dedup MBPP by id
    seen = set()
    mbpp_unique = []
    for p in mbpp_problems:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        mbpp_unique.append(p)
    print(f"MBPP unique: {len(mbpp_unique)} problems")

    # Exclude held-out evaluation set (problems_mbpp30.jsonl)
    held_out_ids = set()
    held_out_path = os.path.join(out_dir, "problems_mbpp30.jsonl")
    if os.path.exists(held_out_path):
        with open(held_out_path) as f:
            for line in f:
                held_out_ids.add(json.loads(line)["id"])
        print(f"Held-out IDs (excluded): {len(held_out_ids)}")

    all_problems = he_problems + mbpp_unique
    expanded = [p for p in all_problems if p["id"] not in held_out_ids]
    print(f"Expanded problem set (excl held-out): {len(expanded)}")

    out = os.path.join(out_dir, "problems_expanded.jsonl")
    with open(out, "w") as f:
        for p in expanded:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
