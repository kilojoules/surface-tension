"""Build DPO preference pairs for the lean victim-hardening run
(docs/harden_lean_plan_2026-10-04.md, Stage C).

Three pair families + merge:
  refusal : chosen = a varied benign refusal template, rejected = an APPROVED
            harmful completion (from the human-gated freeze, Gate B). This
            script AUTHORS NO HARMFUL CONTENT — the rejected side is loaded
            from the private approved file only, and outputs containing it
            stay under data/harden/ (gitignored, canary'd).
  math    : chosen/rejected = correct/incorrect GSM8K samples from the victim
            (exact-match judge: last number in the generation vs the `#### N`
            gold). Coherence pressure — keeps math ability while refusing.
  benign  : anti-over-refusal: chosen = a non-refusal sample (Alpaca reference
            preferred when present), rejected = a refusal the model actually
            produced on a benign prompt (XSTest or Alpaca).
  merge   : deterministic shuffle (+ optional --dup applied to ALL inputs of
            that call) into the final training file; writes a commit-safe
            manifest (counts + hashes only, no text).

Output line format (dpo_train.py): {"prompt", "chosen", "rejected"} plus
family/behavior_hash bookkeeping. GPU subcommands (math, benign) import torch
lazily; refusal/merge run anywhere.

Usage: see docs/harden_lean_plan_2026-10-04.md Stage C.
"""
import argparse
import hashlib
import json
import os
import random
import re

from harden_build_corpus import RATIONALE, TERSE


def bhash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _refusal_variants():
    """Mixed terse/reflective chosen sides (benign text only)."""
    return TERSE + [f"{r} {t}" for r in RATIONALE for t in TERSE[:2]]


# ---------------------------------------------------------------- refusal ---
def build_refusal(args):
    behaviors = [json.loads(ln) for ln in open(args.behaviors) if ln.strip()]
    approved = json.load(open(args.completions))  # {behavior_hash: completion}
    variants = _refusal_variants()
    rng = random.Random(args.seed)
    pairs, missing = [], 0
    for b in behaviors:
        h = b.get("hash") or bhash(b["goal"])
        rej = approved.get(h) or approved.get(b.get("id", ""))
        if not rej:
            missing += 1
            continue
        for j in range(args.n_per_behavior):
            pairs.append(dict(family="refusal", behavior_hash=h, prompt=b["goal"],
                              chosen=variants[(len(pairs) + j) % len(variants)],
                              rejected=rej))
    _write(args.out, pairs)
    print(f"refusal pairs: {len(pairs)} ({missing}/{len(behaviors)} behaviors had "
          f"no approved completion — skipped)")


# ------------------------------------------------------------------- math ---
def parse_gold(answer: str) -> str:
    """GSM8K gold from the `#### N` tail (commas stripped)."""
    tail = answer.split("####")[-1]
    return tail.strip().replace(",", "").replace("$", "").rstrip(".")


def last_number(text: str):
    """Last numeric token in text (commas allowed), or None."""
    nums = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return None
    return nums[-1].replace(",", "").replace("$", "").rstrip(".")


def build_math(args):
    problems = [json.loads(ln) for ln in open(args.gsm8k) if ln.strip()][: args.n_problems]
    rng = random.Random(args.seed)
    from harden_eval import load_victim_adapter  # lazy torch chain
    model, tok = load_victim_adapter(args.victim, args.adapter)
    from jbb_prefill_gen import generate_with_prefill

    pairs, skipped = [], 0
    for i, p in enumerate(problems):
        gold = parse_gold(p["answer"])
        gens = [generate_with_prefill(model, tok, p["question"], prefill="",
                                      max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature)
                for _ in range(args.n_samples)]
        correct = [g[1] for g in gens if last_number(g[1]) == gold]
        wrong = [g[1] for g in gens if last_number(g[1]) != gold]
        if not correct or not wrong:
            skipped += 1
            continue
        # clearest contrast: shortest correct vs longest incorrect
        pairs.append(dict(family="math", behavior_hash=bhash(p["question"]),
                          prompt=p["question"], chosen=min(correct, key=len),
                          rejected=max(wrong, key=len)))
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(problems)}] pairs so far: {len(pairs)}", flush=True)
    _write(args.out, pairs)
    print(f"math pairs: {len(pairs)} ({skipped} problems lacked a correct or "
          f"an incorrect sample)")


# ----------------------------------------------------------------- benign ---
def load_benign_prompts(path, fmt):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if fmt == "xstest":
        return [(r.get("hash") or bhash(r["prompt"]), r["prompt"], None) for r in rows]
    return [(r.get("hash") or bhash(r["instruction"]), r["instruction"], r.get("output")) for r in rows]


def build_benign(args):
    prompts = load_benign_prompts(args.benign, args.format)[: args.n_prompts]
    from harden_eval import load_victim_adapter  # lazy torch chain
    from jbb_prefill_gen import generate_with_prefill, is_refusal
    model, tok = load_victim_adapter(args.victim, args.adapter)

    pairs, skipped = [], 0
    for i, (h, prompt, reference) in enumerate(prompts):
        gens = [generate_with_prefill(model, tok, prompt, prefill="",
                                      max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature)[1]
                for _ in range(args.n_samples)]
        helpful = [g for g in gens if not is_refusal(g)]
        refusals = [g for g in gens if is_refusal(g)]
        chosen = reference if (reference and not is_refusal(reference)) else (
            max(helpful, key=len) if helpful else None)
        if not chosen or not refusals:
            skipped += 1
            continue
        pairs.append(dict(family="benign", behavior_hash=h, prompt=prompt,
                          chosen=chosen, rejected=min(refusals, key=len)))
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(prompts)}] pairs so far: {len(pairs)}", flush=True)
    _write(args.out, pairs)
    print(f"benign ({args.format}) pairs: {len(pairs)} ({skipped} prompts lacked "
          f"a helpful or a refusing sample — no over-refusal pressure needed there)")


# ------------------------------------------------------------------ merge ---
def merge(args):
    rng = random.Random(args.seed)
    merged, per_input = [], {}
    for path in args.inputs:
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        rows = rows * args.dup
        per_input[os.path.basename(path)] = len(rows)
        merged += rows
    rng.shuffle(merged)
    _write(args.out, merged)
    fam = {}
    for r in merged:
        fam[r.get("family", "?")] = fam.get(r.get("family", "?"), 0) + 1
    manifest = dict(n_total=len(merged), families=fam,
                    per_input=per_input, dup=args.dup, seed=args.seed,
                    sha256=bhash("".join(r["chosen"] + r["rejected"] for r in merged)))
    mpath = args.out.replace(".jsonl", "_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"merged {len(merged)} pairs -> {args.out}  (families: {fam})")
    print(f"commit-safe manifest -> {mpath}")


def _write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refusal")
    r.add_argument("--behaviors", required=True, help="victim_train.jsonl (with hashes)")
    r.add_argument("--completions", required=True, help="approved_train.json {hash: text}")
    r.add_argument("--n-per-behavior", type=int, default=2)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)

    m = sub.add_parser("math")
    m.add_argument("--gsm8k", required=True, help="gsm8k train.jsonl")
    m.add_argument("--n-problems", type=int, default=300)
    m.add_argument("--n-samples", type=int, default=4)
    m.add_argument("--max-new-tokens", type=int, default=512)
    m.add_argument("--temperature", type=float, default=0.9)
    m.add_argument("--victim", default="meta-llama/Llama-3.1-8B-Instruct")
    m.add_argument("--adapter", default="", help="optional policy adapter to sample from")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)

    b = sub.add_parser("benign")
    b.add_argument("--format", choices=["xstest", "alpaca"], required=True)
    b.add_argument("--benign", required=True)
    b.add_argument("--n-prompts", type=int, default=100)
    b.add_argument("--n-samples", type=int, default=2)
    b.add_argument("--max-new-tokens", type=int, default=256)
    b.add_argument("--temperature", type=float, default=0.9)
    b.add_argument("--victim", default="meta-llama/Llama-3.1-8B-Instruct")
    b.add_argument("--adapter", default="")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)

    g = sub.add_parser("merge")
    g.add_argument("--inputs", nargs="+", required=True)
    g.add_argument("--dup", type=int, default=1,
                   help="replicate ALL inputs of this call (Gate E remedy: "
                        "merge the benign inputs with --dup 2 first, then merge all)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)

    args = ap.parse_args()
    dict(refusal=build_refusal, math=build_math, benign=build_benign,
         merge=merge)[args.cmd](args)


if __name__ == "__main__":
    main()
