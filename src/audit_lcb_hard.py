"""LCB-hard audit helper.

Extracts (id, contest_date, prompt_first_lines) from data/problems_lcb_hard.jsonl
so the auditor (a human or an LLM) can read problem statements without loading
the 670 MB file each time.

Outputs:
    paper/data/cohort/lcb_hard_prompts.md  — pretty markdown for review
    paper/data/cohort/lcb_hard_index.jsonl — (id, contest_date) tuples, light file

Then a separate hand-authored audit fills in labels_lcb_hard.jsonl.

Run from repo root:
    python src/audit_lcb_hard.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "problems_lcb_hard.jsonl"
OUT_MD = ROOT / "paper" / "data" / "cohort" / "lcb_hard_prompts.md"
OUT_IDX = ROOT / "paper" / "data" / "cohort" / "lcb_hard_index.jsonl"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    problems = []
    with SRC.open() as f:
        for line in f:
            d = json.loads(line)
            problems.append({
                "id": d["id"],
                "contest_date": d.get("contest_date", ""),
                "prompt": d.get("prompt", ""),
            })
    # group by contest week (truncate to YYYY-MM-DD)
    by_week = defaultdict(list)
    for p in problems:
        day = (p["contest_date"] or "")[:10]
        by_week[day].append(p)

    with OUT_IDX.open("w") as f:
        for p in problems:
            f.write(json.dumps({"id": p["id"], "contest_date": p["contest_date"]}) + "\n")

    lines = []
    lines.append(f"# LCB-hard audit pool — {len(problems)} problems, "
                 f"{len(by_week)} contest weeks")
    lines.append("")
    lines.append("Grouped by contest day. The audit picks at most one irreducible "
                 "per contest week (cap rule from the brief — LCB-hard clusters "
                 "graph/DP by contest, so don't let the off-diagonal cell become "
                 "ten variants of the same algorithm).")
    lines.append("")
    for day in sorted(by_week.keys()):
        lines.append(f"## {day}  —  {len(by_week[day])} problem(s)")
        lines.append("")
        for p in by_week[day]:
            # truncate prompt to ~30 lines for readability
            prompt_head = "\n".join(p["prompt"].split("\n")[:35])
            lines.append(f"### `{p['id']}`")
            lines.append("")
            lines.append("```")
            lines.append(prompt_head)
            lines.append("```")
            lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote: {OUT_MD.relative_to(ROOT)}  ({len(problems)} problems, {len(by_week)} weeks)")
    print(f"wrote: {OUT_IDX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
