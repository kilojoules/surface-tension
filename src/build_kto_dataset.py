"""Build KTO dataset by reusing existing SFT positives + mining bare-prompt
non-compliant-but-passing rows from prior raw CSVs as negatives.

Positives = current data/sft_train.jsonl + data/sft_eval.jsonl (label=desirable).
Negatives = bare-prompt rows across all known raw CSVs where:
  - constraint == "none"  (the bare-prompt sweep, what we actually want to push away from)
  - test_passed == 1      (matched output distribution: same generation params, passing code)
  - constraint check FAILS on the saved source (i.e., contains loops/recursion)
each labeled undesirable.

Problem-level holdout is preserved: we keep the same set of held-out problem_ids
that already exist in data/sft_eval.jsonl. Negatives whose problem_id is in the
holdout set go to data/kto_eval.jsonl; the rest go to data/kto_train.jsonl.

Output schema (jsonl):
  {"problem_id": str, "prompt": str, "completion": str, "label": "desirable"|"undesirable"}
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

from ast_checks import CHECKS, CONSTRAINT_INSTRUCTIONS
from loaders import load_problems_jsonl


def build_prompt(problem: dict, constraint: str | None) -> str:
    """Bare-prompt builder. Inlined to avoid importing sweep_local (which pulls in peft)."""
    base = problem["prompt"]
    if constraint is None:
        instruction = ("Provide a complete, working solution. Return only Python source "
                       "inside a single ```python code block.")
    else:
        instruction = (f"Constraint: {CONSTRAINT_INSTRUCTIONS[constraint]}\n\n"
                       "Provide a complete, working solution that satisfies this constraint. "
                       "Return only Python source inside a single ```python code block.")
    return f"{base}\n\n{instruction}"


def _format_completion(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def _src_path_for_row(parent_dir: str, tag: str, row: Dict[str, str]) -> str | None:
    safe = row["problem_id"].replace("/", "_")
    fname = f"{safe}__{row['constraint']}__{row['condition']}__s{row['sample_idx']}.py"
    for sdir in (
        os.path.join(parent_dir, f"sources_{tag}"),
        os.path.join(parent_dir, "sources_local"),
    ):
        p = os.path.join(sdir, fname)
        if os.path.exists(p):
            return p
    return None


def mine_negatives(csv_paths: List[str], problems_by_id: Dict[str, dict],
                   constraint: str) -> List[Dict[str, str]]:
    check = CHECKS[constraint]
    seen = set()  # (problem_id, code) — dedup
    out: List[Dict[str, str]] = []

    for csv_path in sorted(set(csv_paths)):
        parent = os.path.dirname(csv_path)
        tag = os.path.basename(csv_path).replace("_raw.csv", "").replace(".csv", "")
        try:
            rows = list(csv.DictReader(open(csv_path)))
        except Exception:
            continue
        for r in rows:
            if r.get("constraint") != "none":
                continue
            if int(r.get("test_passed") or 0) != 1:
                continue
            pid = r["problem_id"]
            if pid not in problems_by_id:
                continue
            src = _src_path_for_row(parent, tag, r)
            if not src:
                continue
            code = open(src).read().strip()
            if not code:
                continue
            try:
                if check(code):
                    continue  # already compliant — not a negative
            except Exception:
                continue  # AST failure: skip
            key = (pid, code)
            if key in seen:
                continue
            seen.add(key)
            prompt = build_prompt(problems_by_id[pid], constraint=None)
            out.append({
                "problem_id": pid,
                "prompt": prompt,
                "completion": _format_completion(code),
                "label": "undesirable",
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../data/problems_lcb.jsonl")
    ap.add_argument("--sft-train", default="../data/sft_train.jsonl")
    ap.add_argument("--sft-eval", default="../data/sft_eval.jsonl")
    ap.add_argument("--csv-globs", nargs="+", default=[
        "../results/raw/*.csv",
        "../vast_logs/*/st/results/raw/*.csv",
    ])
    ap.add_argument("--constraint", default="no_loops_no_recursion")
    ap.add_argument("--out-dir", default="../data")
    ap.add_argument("--max-neg-ratio", type=float, default=3.0,
                    help="cap negatives at this multiple of positives (KTO needs balance)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    def absify(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(here, p)

    problems = load_problems_jsonl(absify(args.problems))
    problems_by_id = {p["id"]: p for p in problems}

    pos_train = [json.loads(l) for l in open(absify(args.sft_train))]
    pos_eval = [json.loads(l) for l in open(absify(args.sft_eval))]
    eval_pids = {e["problem_id"] for e in pos_eval}
    print(f"positives: {len(pos_train)} train + {len(pos_eval)} eval ({len(eval_pids)} held-out problems)")

    csv_paths: List[str] = []
    for g in args.csv_globs:
        csv_paths.extend(glob.glob(absify(g)))
    print(f"scanning {len(set(csv_paths))} raw CSVs...")

    negs = mine_negatives(csv_paths, problems_by_id, args.constraint)
    print(f"mined {len(negs)} unique non-compliant passing negatives "
          f"({len({n['problem_id'] for n in negs})} unique problems)")

    # Balance: cap negatives at max_neg_ratio × positives
    n_pos = len(pos_train) + len(pos_eval)
    cap = int(n_pos * args.max_neg_ratio)
    if len(negs) > cap:
        # Prefer broad problem coverage: round-robin over problems
        from collections import defaultdict
        by_pid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for n in negs:
            by_pid[n["problem_id"]].append(n)
        picked = []
        while len(picked) < cap and any(by_pid.values()):
            for pid in list(by_pid.keys()):
                if not by_pid[pid]:
                    continue
                picked.append(by_pid[pid].pop(0))
                if len(picked) >= cap:
                    break
        negs = picked
        print(f"capped to {len(negs)} negatives (max_neg_ratio={args.max_neg_ratio})")

    # Split negatives by the existing problem-level holdout
    neg_eval = [n for n in negs if n["problem_id"] in eval_pids]
    neg_train = [n for n in negs if n["problem_id"] not in eval_pids]

    # Tag positives with label
    pos_train_l = [{**e, "label": "desirable"} for e in pos_train]
    pos_eval_l = [{**e, "label": "desirable"} for e in pos_eval]

    train = pos_train_l + neg_train
    eval_ = pos_eval_l + neg_eval

    print(f"KTO train: {len(pos_train_l)} pos + {len(neg_train)} neg = {len(train)}")
    print(f"KTO eval:  {len(pos_eval_l)} pos + {len(neg_eval)} neg = {len(eval_)}")

    out_dir = absify(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kto_train.jsonl"), "w") as f:
        for x in train:
            f.write(json.dumps(x) + "\n")
    with open(os.path.join(out_dir, "kto_eval.jsonl"), "w") as f:
        for x in eval_:
            f.write(json.dumps(x) + "\n")
    print(f"wrote {out_dir}/kto_train.jsonl, kto_eval.jsonl")


if __name__ == "__main__":
    main()
