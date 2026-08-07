"""Materialize the SFT token-scaling grid: 2 arms x 3 completion-char budgets.

Motivation (prereg/sft_scaling_2026-08-06.md): the "rationale prose is
load-bearing" claim (README) rests on rationale-vs-stripped comparisons that
are confounded by token count — the rationale targets carry ~25% more
loss-bearing chars (186,429 vs 148,996 over the same 66 demos). This grid
de-confounds it: both arms are subsampled from the SAME 66 demos / 22
problems in the SAME order, to MATCHED completion-char budgets, so at each
budget the arms differ only in whether the prose is present.

Design choices, fixed here and mirrored in the prereg:
  - Budgets: 37,250 / 74,500 / 148,996 chars (quarter / half / full of the
    stripped corpus; the top budget is the largest where both arms can match).
  - Selection order: problems shuffled once (seed 0), demos shuffled within
    problem, then round-robin across problems — a prefix of this order is
    automatically problem-stratified. The SAME index order serves both arms;
    each arm takes the prefix that first reaches the budget (so the rationale
    arm uses fewer demos for the same tokens — that is the point).
  - Loss-bearing tokens = completion chars only (sft_train.py masks the
    prompt), so budgets are stated in completion chars.

Writes data/sft_scaling/{rationale,stripped}_{b37,b75,b149}.jsonl plus
manifest.json (row counts, char totals, demo indices, sha256 of inputs and
outputs). Deterministic: re-running must reproduce byte-identical sets.

Run from the repo root:  python scripts/make_sft_scaling_sets.py
"""
import hashlib
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAT_IN = os.path.join(ROOT, "data", "sft_rationale_train.jsonl")
STRIP_IN = os.path.join(ROOT, "data", "sft_rationale_stripped_train.jsonl")
OUT_DIR = os.path.join(ROOT, "data", "sft_scaling")
SEED = 0
BUDGETS = {"b37": 37_250, "b75": 74_500, "b149": 148_996}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rat = [json.loads(l) for l in open(RAT_IN)]
    strip = [json.loads(l) for l in open(STRIP_IN)]
    assert len(rat) == len(strip) == 66, (len(rat), len(strip))
    for a, b in zip(rat, strip):
        assert a["problem_id"] == b["problem_id"] and a["prompt"] == b["prompt"]
        assert b["completion"].strip() in a["completion"], a["problem_id"]

    # Round-robin over shuffled problems -> problem-stratified prefixes.
    by_prob = {}
    for i, r in enumerate(rat):
        by_prob.setdefault(r["problem_id"], []).append(i)
    rng = random.Random(SEED)
    probs = sorted(by_prob)
    rng.shuffle(probs)
    for p in probs:
        rng.shuffle(by_prob[p])
    order = []
    while any(by_prob[p] for p in probs):
        for p in probs:
            if by_prob[p]:
                order.append(by_prob[p].pop(0))
    assert sorted(order) == list(range(66))

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {
        "seed": SEED,
        "budgets_completion_chars": BUDGETS,
        "selection_order": order,
        "inputs": {os.path.basename(p): sha256(p) for p in (RAT_IN, STRIP_IN)},
        "sets": {},
    }
    for arm, rows in (("rationale", rat), ("stripped", strip)):
        for tag, budget in BUDGETS.items():
            chosen, total = [], 0
            for i in order:
                if total >= budget:
                    break
                chosen.append(i)
                total += len(rows[i]["completion"])
            name = f"{arm}_{tag}"
            out = os.path.join(OUT_DIR, f"{name}.jsonl")
            with open(out, "w") as f:
                for i in chosen:
                    f.write(json.dumps(rows[i]) + "\n")
            probs_cov = {rows[i]["problem_id"] for i in chosen}
            manifest["sets"][name] = {
                "n_demos": len(chosen),
                "n_problems": len(probs_cov),
                "completion_chars": total,
                "demo_indices": chosen,
                "sha256": sha256(out),
            }
            print(f"{name:16s} {len(chosen):3d} demos  {len(probs_cov):2d} problems  "
                  f"{total:7,d} chars")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {os.path.join(OUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
