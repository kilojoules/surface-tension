"""Deterministic 3-way behavior split for the hardening-depth experiment
(prereg/hardening_depth_2026-10-03.md).

Partitions a behavior pool into three DISJOINT, category-stratified, seeded sets
and writes a hashed manifest (the repo's freeze/hashing norm):
  - victim_train   : behaviors the hardening arms learn to refuse
  - attacker_train : behaviors the DPO'd attacker + frozen prompt pool are built on
  - eval           : held out from both — where the robustness-coherence frontier
                     is measured (the confirmatory anchors use this set)

Disjointness is the contamination guard: nothing the arms or the attacker train
on may appear in eval. Behaviors are keyed by a stable hash of the behavior text,
so the split is reproducible and overlap is detectable.

NOTE ON SIZE: HarmBench's ~200 standard text behaviors are not enough for a
~300-behavior eval AND disjoint train sets. In practice the victim-train refusal
corpus usually comes from a SEPARATE dataset (leaving all of HarmBench to split
between attacker_train and eval), or the pool is augmented. Tune --fracs / the
input pool accordingly; eval n drives the power the prereg discusses.

Pure (no GPU, no model). The manifest stores hashes + IDs + counts (no behavior
text), so it is safe to commit; the per-split JSONL carries the public HarmBench
text and is written under data/harden/ (gitignored) by default.

Usage:
  PYTHONPATH=src python src/harden_behavior_split.py \
      --behaviors data/harmbench/harmbench_behaviors_text_all.csv \
      --out-dir data/harden/split --seed 0 --fracs 0.34 0.33 0.33
"""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict

from harden_p0_baseline import load_harmbench_behaviors

SPLITS = ("victim_train", "attacker_train", "eval")


def bhash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stratified_split(behaviors, fracs, seed):
    """Category-stratified, seeded, disjoint split. Deterministic in (seed, pool)."""
    assert abs(sum(fracs) - 1.0) < 1e-6, "fracs must sum to 1"
    by_cat = defaultdict(list)
    for b in behaviors:
        by_cat[b.get("category") or "uncat"].append(b)
    out = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda b: b["hash"])  # stable pre-shuffle order
        rng.shuffle(items)
        n = len(items)
        n_v = int(round(fracs[0] * n))
        n_a = int(round(fracs[1] * n))
        out["victim_train"] += items[:n_v]
        out["attacker_train"] += items[n_v:n_v + n_a]
        out["eval"] += items[n_v + n_a:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.34, 0.33, 0.33],
                    metavar=("VICTIM", "ATTACKER", "EVAL"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    behaviors = load_harmbench_behaviors(args.behaviors, args.limit)
    if not behaviors:
        raise SystemExit(f"no behaviors loaded from {args.behaviors}")

    # Dedupe by behavior text (a repeat must not straddle two splits).
    seen, uniq = set(), []
    for b in behaviors:
        h = bhash(b["goal"])
        if h not in seen:
            seen.add(h)
            b["hash"] = h
            uniq.append(b)
    split = stratified_split(uniq, args.fracs, args.seed)

    # Disjointness assertion (by hash) — the contamination guard.
    hs = {s: {b["hash"] for b in split[s]} for s in SPLITS}
    for a in SPLITS:
        for c in SPLITS:
            if a < c:
                assert not (hs[a] & hs[c]), f"overlap between {a} and {c}"

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = dict(source=os.path.basename(args.behaviors), seed=args.seed,
                    fracs=args.fracs, n_total=len(uniq), splits={})
    for s in SPLITS:
        items = split[s]
        with open(os.path.join(args.out_dir, f"{s}.jsonl"), "w") as f:
            for b in items:
                f.write(json.dumps(dict(id=b.get("id", ""), goal=b["goal"],
                                        category=b.get("category", ""),
                                        hash=b["hash"])) + "\n")
        ids_hash = bhash("".join(sorted(b["hash"] for b in items)))
        manifest["splits"][s] = dict(n=len(items), ids_sha256=ids_hash,
                                     categories=sorted({b.get("category", "") for b in items}))
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)

    print("=== behavior split ===")
    for s in SPLITS:
        m = manifest["splits"][s]
        print(f"  {s:14s} n={m['n']:4d}  ids_sha256={m['ids_sha256'][:12]}…")
    print(f"  total unique: {len(uniq)}  seed: {args.seed}  disjoint: OK")
    print(f"  manifest (safe to commit): {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
