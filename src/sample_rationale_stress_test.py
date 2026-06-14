"""Deterministic sampler for the RUBRIC-RELIABILITY STRESS-TEST.

This is NOT the probe gate. It is a free diagnostic that asks: can two readers
applying RUBRIC.md to the comment-form prose surface agree on
`asserts_compliance`? Two outcomes matter:

  - Low κ  ⇒  the rubric is broken — does not separate the category reliably
              even when applied to the messy fallback surface. No amount of
              clean preamble will save it. Kill, kill C2 too, tighten, re-audit.
  - High κ ⇒  necessary but NOT sufficient. Agreement on a blurred category
              does not validate the category. The probe gate is still C2 —
              regenerate R-SFT with raw saved and compute κ + positive-class
              count on the canonical preamble surface. The two κ's get
              COMPARED at that point — a large gap is itself empirical
              evidence that comments and preambles are different signals.

The original brief §A.4 governs: comments and preambles are HYPOTHESISED to be
different signals (a preamble is rhetorically a claim about the whole
solution; an inline comment is a local annotation). Whether that rhetorical
difference is also a representational one is an open question — the two-κ
comparison is one test of it. RUBRIC.md §"Where to look" item 3 admits the
comment surface as a TRIAGE fallback only; this runner uses it for the
stress-test, not for any headline number.

Source: 136 post-training R-SFT (B1++) clean-17 `.py` files at
    vast_logs/rdqb499a37k1tv/st/results/raw/sources_eval_b1plus_clean/

R0 split (per `ast_checks.check_no_loops_no_recursion`, verified):
    R0 compliant: 54 (39.7%) / R0 violators: 82 (60.3%).

Prose-bearing filter: only files whose comments contain multi-sentence
rule-related discussion are eligible. Cheap-check on 2026-06-14:
    violators prose-bearing: 69 of 82
    compliers prose-bearing: 34 of 54
Both pools clear the stratum draw by a healthy margin; no risk of
"running on fumes."

Stratification: 30 violators + 20 compliers = 50 (per the owner's framing —
concentrate the measurement on the judgment-heavy cell where the hard
disagreements live, not on easy compliers where agreement is cheap).

R0 is HIDDEN from both readers' label-time view. The labeler sees only the
code (where R0 is mentally inferable, but the rubric forbids checking it
before the assertion call). The scorer joins R0 afterward.

This script is purely deterministic and produces NO labels. Outputs (all in
paper/data/rationale_label/):
    stress_test_queue_50.jsonl       — agent's labeling queue (code only,
                                       no R0 verdict)
    stress_test_owner_blind_50.jsonl — owner's blind queue (same fields,
                                       shuffled with seed+1)
    stress_test_provenance.json      — seed, source, pool sizes, prose-bearing
                                       counts, the explicit "stress-test not
                                       the gate" disclaimer
"""
from __future__ import annotations

import datetime as _dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import re

from ast_checks import check_no_loops_no_recursion  # type: ignore


# ---- prose-bearing filter ------------------------------------------------
# A file is "prose-bearing" iff its comments contain multi-sentence
# rule-related discussion (≥ 2 rule-related comment lines, OR a single
# rule-related comment that spans ≥ 2 sentences). Files that are bare
# `# no loops` or have no rule-comments at all give the labeler no prose
# surface to apply the rubric to — they're excluded from the eligible pool.

_RULE_PAT = re.compile(
    r"\b(loops?|recursion|recursive|recurse|iterat\w*|comprehension|functional|"
    r"without\s+(?:any\s+|all\s+)?(?:loops?|recursion|iteration)|"
    r"avoid|loop-?free|no\s+(?:for|while|loops?|recursion)|"
    r"\bmap\b|\breduce\b|itertools)",
    flags=re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _comment_lines(code: str) -> list[str]:
    out = []
    for ln in code.split("\n"):
        s = ln.strip()
        if not s.startswith("#"): continue
        if s.startswith("#!") or s.startswith("# -*-"): continue
        body = s.lstrip("#").strip()
        if body: out.append(body)
    return out


def _is_prose_bearing(code: str) -> bool:
    cs = _comment_lines(code)
    rule_hits = [c for c in cs if _RULE_PAT.search(c)]
    if len(rule_hits) >= 2:
        return True
    return any(len(_SENTENCE_SPLIT.split(c)) >= 2 for c in rule_hits)


# ---- source corpus -------------------------------------------------------

SRC_DIR = ROOT / "vast_logs" / "rdqb499a37k1tv" / "st" / "results" / "raw" / "sources_eval_b1plus_clean"
OUT = ROOT / "paper" / "data" / "rationale_label"

SEED = 14062026          # date-anchored, reproducible
SAMPLE_SIZE = 50
TARGET_VIOLATORS = 30    # concentrate on the judgment-heavy cell
TARGET_COMPLIERS = 20    # smaller pool of easy YES/NO cases
EXPLICIT_PURPOSE = (
    "RUBRIC-RELIABILITY STRESS-TEST — comment-form prose surface only. "
    "This is NOT the probe gate. §A.4 of the original brief governs; comments "
    "are a secondary signal. The C2 regenerate ($18) on the preamble surface "
    "is the actual gate that authorizes or kills the probe spend."
)


# ---- helpers -------------------------------------------------------------

def _parse_filename(fn: str) -> dict:
    """`lcb_abc356_c__none__unconstrained__s0.py` →
    {problem_id, constraint, condition, sample_idx}."""
    stem = fn[:-3] if fn.endswith(".py") else fn
    parts = stem.split("__")
    if len(parts) != 4:
        raise ValueError(f"unexpected filename layout: {fn}")
    problem_raw, constraint, condition, s_idx = parts
    problem_id = problem_raw.replace("_", "/", 1)  # lcb_abc356_c → lcb/abc356_c
    return {
        "problem_id": problem_id,
        "constraint": constraint,
        "condition": condition,
        "sample_idx": int(s_idx.lstrip("s")),
    }


def load_and_label() -> list[dict]:
    """Annotate every .py file in SRC_DIR. R0 stays on the INTERNAL row but is
    never exposed in any queue file. Prose-bearing files are gated for
    eligibility — bare-comment or no-rule-comment files are excluded because
    they give the labeler no surface to apply the rubric to."""
    files = sorted(p.name for p in SRC_DIR.iterdir() if p.name.endswith(".py"))
    rows = []
    for fn in files:
        code = (SRC_DIR / fn).read_text()
        try:
            r0_compliant = check_no_loops_no_recursion(code)
        except Exception:
            continue  # parse failures are excluded
        meta = _parse_filename(fn)
        rid = f"rsft-eval/{meta['problem_id']}/s{meta['sample_idx']}"
        rows.append({
            "response_id": rid,
            "file": fn,
            "source_file": str((SRC_DIR / fn).relative_to(ROOT)),
            "problem_id": meta["problem_id"],
            "sample_idx": meta["sample_idx"],
            "code": code,
            "violates_r0": not r0_compliant,         # internal only
            "prose_bearing": _is_prose_bearing(code),
        })
    return rows


def stratify(annotated: list[dict], rng: random.Random) -> list[dict]:
    """Stratify by R0 within the prose-bearing pool only."""
    violators = [r for r in annotated if r["violates_r0"] and r["prose_bearing"]]
    compliers = [r for r in annotated if not r["violates_r0"] and r["prose_bearing"]]
    rng.shuffle(violators)
    rng.shuffle(compliers)
    drawn = violators[:TARGET_VIOLATORS] + compliers[:TARGET_COMPLIERS]
    # If a stratum is short, top up from the other prose-bearing pool.
    if len(drawn) < SAMPLE_SIZE:
        leftover = (violators[TARGET_VIOLATORS:] +
                    compliers[TARGET_COMPLIERS:])
        rng.shuffle(leftover)
        while len(drawn) < SAMPLE_SIZE and leftover:
            drawn.append(leftover.pop())
    return drawn[:SAMPLE_SIZE]


# ---- outputs -------------------------------------------------------------

def write_queue(drawn: list[dict]) -> None:
    """Agent's labeling queue. R0 is HIDDEN at label time — every row carries
    only the code + problem_id + purpose. The agent applies the rubric to the
    comment-form prose and labels `asserts_compliance` blind to violation."""
    path = OUT / "stress_test_queue_50.jsonl"
    with path.open("w") as f:
        for r in drawn:
            f.write(json.dumps({
                "response_id": r["response_id"],
                "problem_id": r["problem_id"],
                "code": r["code"],
                "purpose": EXPLICIT_PURPOSE,
            }) + "\n")
    print(f"wrote: {path.relative_to(ROOT)}  (n={len(drawn)}, R0 hidden)")


def write_owner_blind(drawn: list[dict]) -> None:
    """Owner's blind labeling queue. Identical schema to the agent queue, no
    R0 visible; shuffled with seed+1 to defeat ordering effects."""
    path = OUT / "stress_test_owner_blind_50.jsonl"
    rng = random.Random(SEED + 1)
    shuffled = list(drawn)
    rng.shuffle(shuffled)
    with path.open("w") as f:
        for r in shuffled:
            f.write(json.dumps({
                "response_id": r["response_id"],
                "problem_id": r["problem_id"],
                "code": r["code"],
                "purpose": EXPLICIT_PURPOSE,
            }) + "\n")
    print(f"wrote: {path.relative_to(ROOT)}  (n={len(shuffled)}, R0 hidden, shuffled with seed+1)")


def write_provenance(annotated: list[dict], drawn: list[dict]) -> None:
    n_viol = sum(1 for r in annotated if r["violates_r0"])
    n_comp = sum(1 for r in annotated if not r["violates_r0"])
    viol_prose = sum(1 for r in annotated if r["violates_r0"] and r["prose_bearing"])
    comp_prose = sum(1 for r in annotated if not r["violates_r0"] and r["prose_bearing"])
    drawn_viol = sum(1 for r in drawn if r["violates_r0"])
    drawn_comp = sum(1 for r in drawn if not r["violates_r0"])
    path = OUT / "stress_test_provenance.json"
    path.write_text(json.dumps({
        "generated": _dt.date.today().isoformat(),
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "target": {"violators": TARGET_VIOLATORS, "compliers": TARGET_COMPLIERS},
        "source_dir": str(SRC_DIR.relative_to(ROOT)),
        "purpose": EXPLICIT_PURPOSE,
        "surface": (
            "comment-form prose ONLY — verbose in-code comments per "
            "RUBRIC.md §'Where to look' item 3 (TRIAGE fallback). Apply "
            "the rubric as written; do not promote item 3 to primary."
        ),
        "pool": {
            "total_py_files": len(annotated),
            "R0_compliant": n_comp,
            "R0_violators": n_viol,
            "R0_compliant_prose_bearing": comp_prose,
            "R0_violators_prose_bearing": viol_prose,
        },
        "draw": {
            "n": len(drawn),
            "R0_compliant": drawn_comp,
            "R0_violators": drawn_viol,
        },
        "labeler_discipline": {
            "R0_hidden_from_queue": True,
            "agent_first_blind": True,
            "owner_blind_to_agent_labels": True,
            "rubric_governs": "RUBRIC.md as committed; §3 is a TRIAGE fallback only.",
        },
        "interpretation": {
            "low_kappa": (
                "The rubric does not separate the category reliably on this "
                "surface. A definitional problem isn't fixed by a cleaner "
                "surface — KILL C2 too. Tighten the rubric; re-audit."
            ),
            "high_kappa": (
                "Necessary but NOT sufficient. Agreement on a blurred "
                "category does not validate the category. The probe gate is "
                "the C2 regenerate on the preamble surface; this κ does NOT "
                "feed C2's go/no-go decision."
            ),
            "comparison_with_C2_kappa": (
                "If C2 also clears, compare the two κ's. A LARGE gap between "
                "comment-κ and preamble-κ is empirical evidence that comments "
                "and preambles are different signals (the §A.4 hypothesis), "
                "rather than just stipulated. Report both, label as a finding."
            ),
        },
    }, indent=2))
    print(f"wrote: {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(SRC_DIR)
    print(f"source: {SRC_DIR.relative_to(ROOT)}")
    rng = random.Random(SEED)
    annotated = load_and_label()
    print(f"pool: n={len(annotated)} "
          f"({sum(1 for r in annotated if r['violates_r0'])} violators, "
          f"{sum(1 for r in annotated if not r['violates_r0'])} compliers)")
    drawn = stratify(annotated, rng)
    write_queue(drawn)
    write_owner_blind(drawn)
    write_provenance(annotated, drawn)
    print()
    print("Next steps:")
    print("  1. Agent reads stress_test_queue_50.jsonl, applies RUBRIC.md to the")
    print("     COMMENT-FORM prose surface, emits agent_labels_50_stress_test.jsonl")
    print("     (response_id, asserts_compliance, label_location, justification).")
    print("  2. Owner reads stress_test_owner_blind_50.jsonl independently, emits")
    print("     owner_labels_50_stress_test.jsonl with the same schema.")
    print("  3. Run:")
    print("       python src/score_rationale_audit.py \\")
    print("           paper/data/rationale_label/agent_labels_50_stress_test.jsonl \\")
    print("           paper/data/rationale_label/owner_labels_50_stress_test.jsonl \\")
    print("           --out paper/data/rationale_label/stress_test_kappa.md")
    print("  4. Label the κ report explicitly as a STRESS-TEST, not the probe gate.")


if __name__ == "__main__":
    main()
