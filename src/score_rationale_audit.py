"""Score the rationale-label inter-rater audit.

Reads two JSONL files of (response_id -> {asserts_compliance, justification})
and reports Cohen's κ, raw agreement, a confusion table, and the disagreement
list. No compute; pure Python.

Usage:
    python src/score_rationale_audit.py \
        paper/data/rationale_label/agent_labels_50.jsonl \
        paper/data/rationale_label/owner_labels_50.jsonl \
        --out paper/data/rationale_label/audit_kappa.md

Schema of each labels file (one JSON per line):
    {"response_id": "rsft/lcb/abc356_c__s0",
     "asserts_compliance": "YES" | "NO",
     "label_location": "prose-pre-code" | "comment-block" | "inline-comment" | "none",
     "justification": "..."}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("asserts_compliance") not in ("YES", "NO"):
                raise ValueError(
                    f"bad value for asserts_compliance on {d.get('response_id')}: "
                    f"{d.get('asserts_compliance')!r} — must be 'YES' or 'NO'"
                )
            out[d["response_id"]] = d
    return out


def _cohen_kappa(yy: int, yn: int, ny: int, nn: int) -> float:
    """Cohen's kappa on a 2x2 confusion table.
       rater A: rows = YES, NO; rater B: cols = YES, NO.
    """
    n = yy + yn + ny + nn
    if n == 0: return 0.0
    p_obs = (yy + nn) / n
    pA_yes = (yy + yn) / n
    pB_yes = (yy + ny) / n
    p_exp = pA_yes * pB_yes + (1 - pA_yes) * (1 - pB_yes)
    if p_exp == 1.0:
        return 1.0 if p_obs == 1.0 else 0.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def _branch(kappa: float, raw: float) -> str:
    """Return the gate-decision branch per the brief's §D thresholds."""
    if kappa >= 0.8 or raw >= 0.90:
        return ("**κ ≥ ~0.8 / raw agreement ≥ ~90%** — label is solid. "
                "Proceed to E (full-set count) then probe-readiness check.")
    if kappa >= 0.6:
        return ("**0.6 ≤ κ < 0.8** — definition is underspecified. Tighten "
                "RUBRIC.md §A.2 using the specific disagreements as new "
                "worked examples; re-audit a fresh 50. Do NOT spend on the "
                "probe yet.")
    return ("**κ < 0.6** — the category is not reliably separable by prose. "
            "Stop; bring to owner. The claim may need to rest on the "
            "deterministic `bare_comment_only` + `rationalized_laundering` "
            "markers instead of \"asserts compliance\", which would change "
            "the paper's sentence.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_path", type=Path,
                    help="agent-labeled JSONL (response_id -> asserts_compliance)")
    ap.add_argument("owner_path", type=Path,
                    help="owner-labeled JSONL (response_id -> asserts_compliance)")
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown report path; default = stdout")
    args = ap.parse_args()

    agent = _load(args.agent_path)
    owner = _load(args.owner_path)

    only_a = sorted(set(agent) - set(owner))
    only_b = sorted(set(owner) - set(agent))
    shared = sorted(set(agent) & set(owner))

    if only_a or only_b:
        print(f"WARNING: response_id mismatch.\n"
              f"  in agent but not owner: {len(only_a)}\n"
              f"  in owner but not agent: {len(only_b)}\n"
              f"  joining on the shared {len(shared)}")

    yy = yn = ny = nn = 0
    disagreements = []
    for rid in shared:
        a = agent[rid]; b = owner[rid]
        av = a["asserts_compliance"]; bv = b["asserts_compliance"]
        if av == "YES" and bv == "YES": yy += 1
        elif av == "YES" and bv == "NO":  yn += 1
        elif av == "NO"  and bv == "YES": ny += 1
        else:                              nn += 1
        if av != bv:
            disagreements.append({
                "response_id": rid,
                "agent": av, "agent_justification": a.get("justification", ""),
                "owner": bv, "owner_justification": b.get("justification", ""),
            })

    n = len(shared)
    raw = (yy + nn) / n if n else 0.0
    kappa = _cohen_kappa(yy, yn, ny, nn)

    lines = []
    lines.append("# Rationale-label inter-rater audit — κ report")
    lines.append("")
    lines.append(f"- Agent file: `{args.agent_path}`")
    lines.append(f"- Owner file: `{args.owner_path}`")
    lines.append(f"- Joined on shared response_ids: **n = {n}**")
    lines.append("")
    lines.append("## Confusion table (`asserts_compliance`)")
    lines.append("")
    lines.append("|  | owner=YES | owner=NO | row total |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| **agent=YES** | {yy} | {yn} | {yy+yn} |")
    lines.append(f"| **agent=NO**  | {ny} | {nn} | {ny+nn} |")
    lines.append(f"| **col total** | {yy+ny} | {yn+nn} | {n} |")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- **Cohen's κ:** {kappa:.3f}")
    lines.append(f"- **Raw agreement:** {raw*100:.1f}%  ({yy+nn}/{n})")
    lines.append(f"- **YES base rate (agent):** {(yy+yn)/n*100:.1f}%")
    lines.append(f"- **YES base rate (owner):** {(yy+ny)/n*100:.1f}%")
    lines.append("")
    lines.append("## Gate decision (per Task 5 brief §D)")
    lines.append("")
    lines.append(_branch(kappa, raw))
    lines.append("")

    if disagreements:
        lines.append(f"## Disagreements (n = {len(disagreements)})")
        lines.append("")
        for d in disagreements:
            lines.append(f"### `{d['response_id']}`")
            lines.append(f"- **agent:** `{d['agent']}` — {d['agent_justification']}")
            lines.append(f"- **owner:** `{d['owner']}` — {d['owner_justification']}")
            lines.append("")
    else:
        lines.append("## Disagreements")
        lines.append("")
        lines.append("None — every shared response has matching labels.")
        lines.append("")

    report = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(report)
        print(f"wrote: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
