"""Build on-policy SFT dataset from a base bare-prompt sweep CSV.

Filters to (constraint='none', condition='unconstrained') samples that are
both compliant under no_loops_no_recursion AND test-passing, and writes
{problem_id, prompt, completion} pairs.

Usage:
  python3 build_onpolicy_dataset.py \
    --csv ../results/raw/onpolicy_mine_lcb50.csv \
    --sources ../results/raw/sources_onpolicy_mine_lcb50 \
    --problems ../data/problems_lcb.jsonl \
    --limit-problems 50 \
    --out ../data/sft_onpolicy.jsonl
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--limit-problems", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(args.problems) if l.strip()]
    if args.limit_problems:
        problems = problems[: args.limit_problems]
    pdict = {p["id"]: p for p in problems}

    rows = list(csv.DictReader(open(args.csv)))
    keep = []
    for r in rows:
        if r["constraint"] != "none":
            continue
        if not int(r.get("compliant", 0)) or not int(r.get("test_passed", 0)):
            continue
        safe = r["problem_id"].replace("/", "_")
        fname = os.path.join(args.sources, f'{safe}__none__unconstrained__s{r["sample_idx"]}.py')
        if not os.path.exists(fname):
            continue
        if r["problem_id"] not in pdict:
            continue
        code = open(fname).read().strip()
        if not code:
            continue
        keep.append({
            "problem_id": r["problem_id"],
            "prompt": pdict[r["problem_id"]]["prompt"],
            "completion": code,
        })

    pids = {k["problem_id"] for k in keep}
    print(f"on-policy compliant+passing: {len(keep)} examples from {len(pids)} problems")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for k in keep:
            f.write(json.dumps(k) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
