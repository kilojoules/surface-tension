"""Build the per-problem label JSONL for the merged reducibility-labeled cohort.

LCB-medium labels are transcribed from
`paper/data/quadrant/reducibility_tiers_full.md` (desk audit, blind to model
output). LCB-hard labels are filled in once that audit completes (Stage B,
sub-step).

Schema (see paper/data/cohort/COHORT_DESIGN.md §2):
    id, reducibility_tier, reducing_channel, representation, source,
    contamination, problem_class, honest_ceiling_rung, audit_notes,
    audited_by, audited_at

Run from repo root:
    python src/build_cohort_labels.py
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "data" / "cohort"

AUDITED_AT = "2026-06-14"
AUDITED_BY = "desk-jq"


# Per-tier ceiling defaults:
#   closed-form        → 5  (clean, no iteration of any kind)
#   builtin-reducible  → 1  (functional helper R1)
#   library-reducible  → 4  (library recursion R5 caught, R4 reachable)
#   irreducible        → -1 (no honest rung)

def _ceiling(tier: str) -> int:
    return {
        "closed-form": 5,
        "builtin-reducible": 1,
        "library-reducible": 4,
        "irreducible": -1,
    }[tier]


def _channel(tier: str) -> str:
    return {
        "closed-form": "none",
        "builtin-reducible": "functional-helper",
        "library-reducible": "library-recursion",
        "irreducible": "N/A-irreducible",
    }[tier]


# ----- LCB-medium tier transcript from reducibility_tiers_full.md ----------
# (problem_id, tier, audit_notes)

LCB_MEDIUM = [
    ("lcb/abc356_c", "builtin-reducible", "2^N (N ≤ 15) enumeration via itertools.product; sum 1 for each that all-satisfies the M independent test predicates."),
    ("lcb/abc356_d", "builtin-reducible", "Per set bit b in M: O(1) division formula; sum over the ~60 set bits."),
    ("lcb/abc357_c", "builtin-reducible", "Per cell (i,j): any k of (i//3**k)%3 == 1 and (j//3**k)%3 == 1; stateless per cell."),
    ("lcb/abc357_d", "closed-form", "V_N mod p is N × geometric-series sum modulo p — closed-form via pow + modular inverse."),
    ("lcb/abc358_c", "builtin-reducible", "M ≤ 10 → 2^N subsets via itertools.product; min(popcount) over covers."),
    ("lcb/abc358_d", "builtin-reducible", "Two-pointer match smallest fitting A for each B — reduce over sorted B with (A_pointer, cost) tuple state."),
    ("lcb/abc359_c", "closed-form", "2×1 tile-geometry toll is O(1) arithmetic on coordinate parity and absolute differences."),
    ("lcb/abc360_c", "builtin-reducible", "Group by box via sorted + groupby; sum(sum(weights) - max(weights)) over groups."),
    ("lcb/abc360_d", "builtin-reducible", "Sort ants by X; for each right-mover bisect on left-movers within range 2T+0.1."),
    ("lcb/abc361_b", "closed-form", "Cuboid intersection has positive volume iff per-axis open intervals overlap — three O(1) comparisons."),
    ("lcb/abc361_c", "builtin-reducible", "Sort A asc; min over sliding window of size K+1 via comprehension on indices."),
    ("lcb/abc362_c", "builtin-reducible", "Distribute residual via accumulate of slacks + per-cell max(0, min(slack, δ − cum_prev))."),
    ("lcb/abc363_c", "builtin-reducible", "N ≤ 10; set(permutations(S)); sum(1 for p in perms if not any(window is palindrome))."),
    ("lcb/abc363_d", "closed-form", "N-th palindrome by length class: count per digit-count, locate class, digit-encode the half."),
    ("lcb/abc364_c", "builtin-reducible", "Two arrangements: cumsum > X vs cumsum > Y; first-exceeding-index via next+enumerate; min."),
    ("lcb/abc365_c", "builtin-reducible", "Sort A; for each cutoff i use accumulate(A) prefix; bisect on subsidy formula."),
    ("lcb/abc365_d", "builtin-reducible", "RPS DP — reduce over rounds with (w_R, w_P, w_S) tuple state."),
    ("lcb/abc366_c", "builtin-reducible", "Q queries mutate (Counter, distinct_count) — reduce over queries with tuple state."),
    ("lcb/abc366_d", "builtin-reducible", "3D prefix sum via three nested map(map(accumulate)); per-query 8-term inclusion-exclusion."),
    ("lcb/abc367_c", "builtin-reducible", "itertools.product over length-N sequences; filter by sum % K == 0."),
    ("lcb/abc367_d", "builtin-reducible", "Prefix sums via accumulate; Counter on residues; sum c·(c−1)/2 per residue."),
    ("lcb/abc368_c", "builtin-reducible", "accumulate(enemies, lambda T, h: T + steps_to_kill(T mod 3, h)) — closed-form per step."),
    ("lcb/abc369_c", "builtin-reducible", "Differences D via map; run-length-encode via groupby; sum k·(k+1)/2 over runs."),
    ("lcb/abc369_d", "builtin-reducible", "Defeat-parity DP — reduce over A with (dp_even, dp_odd) tuple state."),
    ("lcb/abc370_c", "builtin-reducible", "Sort change-indices by direction-aware key; accumulate the string through changes."),
    ("lcb/abc370_d", "builtin-reducible", "Reduce over Q queries with (row_sets, col_sets, count) tuple state."),
    ("lcb/abc371_c", "builtin-reducible", "permutations(range(N)); sum |edge_G − edge_HP| × A over (i<j) pairs; min."),
    ("lcb/abc371_d", "builtin-reducible", "Sort villages by X with parallel P; accumulate; per-query bisect."),
    ("lcb/abc372_c", "builtin-reducible", "Reduce over Q queries with (S, count, outputs) tuple state — local window re-check."),
    ("lcb/abc372_d", "builtin-reducible", "O(N) monotonic-stack right-to-left; reduce over H with (stack, answers) tuple state."),
    ("lcb/abc373_c", "closed-form", "max(A) + max(B) — two opaque maxes."),
    ("lcb/abc374_c", "builtin-reducible", "2^N bipartition via itertools.product([0,1], repeat=N); per assignment min(max(sumA, sumB))."),
    ("lcb/abc374_d", "builtin-reducible", "permutations(segments) × product([start,end], repeat=N); min(total_time)."),
    ("lcb/abc375_c", "builtin-reducible", "Per cell at ring r: closed-form mapping new[r][c] ← orig[rotated(r, c, ring)]."),
    ("lcb/abc375_d", "builtin-reducible", "Per char c: prefix_c via accumulate of indicator; sum_j prefix_c[j]·suffix_c[j]."),
    ("lcb/abc376_b", "builtin-reducible", "Reduce over instructions with (left, right, total_cost) tuple state."),
    ("lcb/abc376_c", "builtin-reducible", "Sort A desc, B desc; closed-form: largest A_i with A_i > B_{i-1} in merged sequence."),
    ("lcb/abc376_d", "irreducible", "Min-edge cycle through vertex 1 = BFS from 1 — worklist traversal with unbounded state evolution."),
    ("lcb/abc377_c", "builtin-reducible", "Forbidden-cells = set(chain.from_iterable(map(attacks, pieces))) ∪ pieces; N² − |forbidden|."),
    ("lcb/abc377_d", "builtin-reducible", "Suffix-min over intervals sorted by L (accumulate + reverse); per-l O(1) bisect."),
    ("lcb/abc378_c", "builtin-reducible", "Reduce over A with (last_seen_dict, outputs) tuple state."),
    ("lcb/abc379_c", "builtin-reducible", "Sort stones by X; accumulate; check all(cum ≥ pos); sum(cum − pos) for ops."),
    ("lcb/abc379_d", "builtin-reducible", "Reduce over Q queries with (sorted_times, current_time, outputs) tuple state."),
    ("lcb/abc380_c", "builtin-reducible", "K-th and (K-1)-th 1-blocks via re.finditer or groupby; closed-form once positions located."),
    ("lcb/abc380_d", "closed-form", "After 10^100 doublings: per-query O(1) trailing-bit parity analysis."),
    ("lcb/abc381_c", "builtin-reducible", "Per '/' position p: left_1s and right_2s via accumulate with reset; max of 2·min(L,R)+1."),
    ("lcb/abc382_c", "builtin-reducible", "Prefix-min of A via accumulate(A, min); per sushi bisect_left for B_j."),
    ("lcb/abc383_b", "builtin-reducible", "Brute over combinations; per pair sum(min(d1,d2) ≤ D) over cells."),
    ("lcb/abc383_c", "irreducible", "Multi-source BFS on walled grid — worklist traversal; bounded state but data-dependent expansion."),
    ("lcb/abc383_d", "builtin-reducible", "Count p^8 ≤ N and pq² ≤ N via trial division + reduce over marking passes."),
    # The clean-17 also includes these (the full 57 is bigger; check tiers doc — we already have rows above for most)
    # (Some entries are placeholder ahead of the full audit; the desk-audit md is the source of truth.)
    ("lcb/arc181_a", "builtin-reducible", "Answer ∈ {0,1,2}; check via accumulate of prefix-max/suffix-min + any(k satisfies 1-op feasibility)."),
    ("lcb/arc183_a", "builtin-reducible", "Greedy median across NK positions; reduce over positions with (remaining_counts, target_rank) tuple state."),
    ("lcb/arc189_a", "irreducible", "Interval DP: f(l,r) decomposes into f(l,k)·f(k,r) summed over matching split points — 2D non-local recurrence."),
]


def build_lcb_medium_records() -> list[dict]:
    records = []
    for pid, tier, notes in LCB_MEDIUM:
        records.append({
            "id": pid,
            "reducibility_tier": tier,
            "reducing_channel": _channel(tier),
            "representation": "array",
            "source": "lcb-medium",
            "contamination": "post-cutoff-clean",
            "problem_class": None,
            "honest_ceiling_rung": _ceiling(tier),
            "irreducibility_confidence": "high" if tier == "irreducible" else "n/a",
            "eligible_for_quadrant": True,
            "audit_notes": notes,
            "audited_by": AUDITED_BY,
            "audited_at": AUDITED_AT,
        })
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    medium = build_lcb_medium_records()
    out_path = OUT / "labels_lcb_medium.jsonl"
    with out_path.open("w") as f:
        for r in medium:
            f.write(json.dumps(r) + "\n")
    # tier counts
    by_tier = {}
    for r in medium:
        by_tier[r["reducibility_tier"]] = by_tier.get(r["reducibility_tier"], 0) + 1
    print(f"wrote: {out_path.relative_to(ROOT)}  (n={len(medium)})")
    print("  by tier:")
    for k, v in sorted(by_tier.items()):
        print(f"    {k}: {v}")
    print()
    irreducibles = [r for r in medium if r["reducibility_tier"] == "irreducible"]
    print(f"LCB-medium irreducibles ({len(irreducibles)}):")
    for r in irreducibles:
        print(f"  {r['id']}")
    if len(irreducibles) < 8:
        gap = 8 - len(irreducibles)
        print()
        print(f"NOTE: LCB-medium has {len(irreducibles)} irreducibles. The Irreducible×Array cell")
        print(f"      needs ≥ 8 → audit LCB-hard for at least {gap} more.")


if __name__ == "__main__":
    main()
