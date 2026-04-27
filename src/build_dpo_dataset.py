"""Build (prompt, chosen, rejected) DPO triples from a sweep CSV + source dump.

Cross-pairs all (passing unconstrained) × (passing AND compliant constrained) within each
problem. The prompt is the bare-problem prompt — no mention of the constraint anywhere.
That is the whole experiment: the model has to learn the preference latently, not from
instruction-following.

Outputs problem-level train/eval split (held-out problems → eval set, never seen during DPO).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ast_checks import CHECKS
from loaders import load_problems_jsonl
from sweep import build_prompt


def _src_path(sources_dir: str, row: Dict[str, str]) -> str:
    safe = row["problem_id"].replace("/", "_")
    return os.path.join(
        sources_dir,
        f"{safe}__{row['constraint']}__{row['condition']}__s{row['sample_idx']}.py",
    )


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            txt = f.read()
        return txt if txt.strip() else None
    except FileNotFoundError:
        return None


def _format_completion(code: str) -> str:
    """Wrap bare code in the same fenced format the model normally outputs."""
    return f"```python\n{code.rstrip()}\n```"


def _index(rows: List[Dict[str, str]], constraint: str) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    """Per-problem indexes of (passing unconstrained) and (passing AND compliant constrained-with-target-constraint)."""
    unc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    con: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        if int(r.get("test_passed") or 0) != 1:
            continue
        if r["condition"] == "unconstrained":
            unc[r["problem_id"]].append(r)
        elif r["constraint"] == constraint and r["condition"] == "constrained":
            if int(r.get("compliant") or 0) == 1:
                con[r["problem_id"]].append(r)
    return unc, con


def build_pairs(
    problems_path: str,
    csv_path: str,
    sources_dir: str,
    constraint: str,
) -> List[Dict[str, str]]:
    problems = {p["id"]: p for p in load_problems_jsonl(problems_path)}
    rows = list(csv.DictReader(open(csv_path)))
    unc, con = _index(rows, constraint)

    constraint_check = CHECKS[constraint]
    pairs: List[Dict[str, str]] = []
    n_skipped_compliant_rejected = 0
    for pid in con:
        if pid not in unc or pid not in problems:
            continue
        prompt = build_prompt(problems[pid], constraint=None)  # bare prompt — no constraint hint
        for chosen_row in con[pid]:
            chosen_code = _read(_src_path(sources_dir, chosen_row))
            if not chosen_code:
                continue
            for rejected_row in unc[pid]:
                rejected_code = _read(_src_path(sources_dir, rejected_row))
                if not rejected_code:
                    continue
                if chosen_code == rejected_code:
                    continue
                # Skip pairs where the unconstrained "rejected" happens to satisfy the constraint —
                # those teach DPO contradictory preferences (compliant > compliant).
                if constraint_check(rejected_code):
                    n_skipped_compliant_rejected += 1
                    continue
                pairs.append({
                    "problem_id": pid,
                    "prompt": prompt,
                    "chosen": _format_completion(chosen_code),
                    "rejected": _format_completion(rejected_code),
                })
    if n_skipped_compliant_rejected:
        print(f"Skipped {n_skipped_compliant_rejected} pairs where the unconstrained sample was accidentally compliant.")
    return pairs


def split_by_problem(pairs: List[Dict[str, str]], holdout_frac: float = 0.3, seed: int = 0) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split so no problem appears in both train and eval — measures generalization, not memorization."""
    pids = sorted({p["problem_id"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_eval = max(1, int(len(pids) * holdout_frac))
    eval_pids = set(pids[:n_eval])
    train = [p for p in pairs if p["problem_id"] not in eval_pids]
    eval_ = [p for p in pairs if p["problem_id"] in eval_pids]
    return train, eval_


def write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            # TRL DPOTrainer only needs prompt/chosen/rejected; problem_id retained for traceability
            f.write(json.dumps(it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../data/problems_lcb.jsonl")
    ap.add_argument("--csv", default="../results/raw/pilot_v4_raw.csv")
    ap.add_argument("--sources-dir", default="../results/raw/sources_v4")
    ap.add_argument("--constraint", default="no_loops_no_recursion")
    ap.add_argument("--out-dir", default="../data")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    def absify(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    pairs = build_pairs(absify(args.problems), absify(args.csv), absify(args.sources_dir), args.constraint)
    print(f"Built {len(pairs)} pairs from {len({p['problem_id'] for p in pairs})} problems.")

    train, eval_ = split_by_problem(pairs, args.holdout_frac, args.seed)
    print(f"Train: {len(train)} pairs / {len({p['problem_id'] for p in train})} problems")
    print(f"Eval:  {len(eval_)} pairs / {len({p['problem_id'] for p in eval_})} problems")

    out_dir = absify(args.out_dir)
    write_jsonl(train, os.path.join(out_dir, "dpo_pairs_train.jsonl"))
    write_jsonl(eval_, os.path.join(out_dir, "dpo_pairs_eval.jsonl"))
    write_jsonl(pairs, os.path.join(out_dir, "dpo_pairs_all.jsonl"))
    print(f"Wrote {out_dir}/dpo_pairs_{{train,eval,all}}.jsonl")


if __name__ == "__main__":
    main()
