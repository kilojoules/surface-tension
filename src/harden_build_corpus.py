"""Build the four hardening-arm SFT corpora for the reflection x depth 2x2
(prereg/hardening_depth_2026-10-03.md), from the victim-train behavior split.

Arms (reflection x depth):
  shallow_terse       prompt=behavior, target = terse refusal (decided at pos 0)
  shallow_reflective  prompt=behavior, target = rationale + refusal (pos 0)
  deep_terse          recovery augmentation: a harmful prefix of random length
                      k in [1, K] sits in the assistant turn as MASKED context,
                      and the loss-bearing target is a pivot + terse refusal
                      (the model learns to re-refuse mid-stream)
  deep_reflective     same recovery, target = pivot + rationale + refusal

SAFETY:
  - This script AUTHORS NO HARMFUL CONTENT. The refusal/rationale/pivot templates
    below are benign. The harmful prefixes for the deep arms are loaded from an
    APPROVED, human-gated, canary'd PRIVATE file (the jbb frozen_private pattern),
    never generated or embedded here.
  - The deep-arm corpora contain those prefixes, so output goes to a PRIVATE dir
    under data/harden/ (gitignored) and gets a CANARY. The shallow arms are free
    of harmful content.
  - The prefix is stored SEPARATELY from the target and marked for loss masking:
    the model is trained to REFUSE given a harmful start, never to generate the
    harmful tokens. See the TRAINER NOTE printed at the end.

--shallow-only builds just the two shallow arms (no harmful prefixes, no
tokenizer) — use it to smoke-test templates/structure with zero harmful content.

Usage:
  PYTHONPATH=src python src/harden_build_corpus.py \
      --behaviors data/harden/split/victim_train.jsonl \
      --harmful-prefixes data/harden/private/approved_prefixes.json \
      --out-dir data/harden/private/corpora --K 16 --n-deep 2 --seed 0
"""
import argparse
import hashlib
import json
import os
import random

# Benign templates (several variants, to avoid overfitting a single refusal string).
TERSE = [
    "I can't help with that.",
    "I'm not able to help with this request.",
    "Sorry, I can't assist with that.",
    "I won't help with this.",
]
RATIONALE = [
    "This asks for something that could cause real harm, so I won't provide it.",
    "Fulfilling this could facilitate serious harm, so I can't assist.",
    "Helping here would risk enabling harm, which I won't do.",
]
PIVOT = [
    "\n\nI need to stop here — I shouldn't continue with this. ",
    "\n\nActually, I can't keep going with this. ",
    "\n\nOn reflection, I won't continue. ",
]


def bhash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def terse_target(rng):
    return rng.choice(TERSE)


def reflective_target(rng):
    return f"{rng.choice(RATIONALE)} {rng.choice(TERSE)}"


def build_shallow(behavior, rng):
    return [
        dict(arm="shallow_terse", prompt=behavior, assistant_prefix="",
             target=terse_target(rng), k=0),
        dict(arm="shallow_reflective", prompt=behavior, assistant_prefix="",
             target=reflective_target(rng), k=0),
    ]


def build_deep(behavior, prefix_text, rng, K, n_deep, tok):
    """n_deep examples per deep arm, each a random k-token harmful prefix (masked
    context) + a pivot/refusal (loss-bearing target)."""
    recs = []
    pref_ids = tok(prefix_text, add_special_tokens=False).input_ids
    if not pref_ids:
        return recs
    for _ in range(n_deep):
        k = rng.randint(1, min(K, len(pref_ids)))
        prefix_k = tok.decode(pref_ids[:k])
        recs.append(dict(arm="deep_terse", prompt=behavior, assistant_prefix=prefix_k,
                         target=rng.choice(PIVOT) + terse_target(rng), k=k))
        recs.append(dict(arm="deep_reflective", prompt=behavior, assistant_prefix=prefix_k,
                         target=rng.choice(PIVOT) + reflective_target(rng), k=k))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", required=True, help="victim_train.jsonl from the split")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--harmful-prefixes",
                    help="APPROVED private prefixes json {behavior_hash|id: text}; "
                         "required unless --shallow-only")
    ap.add_argument("--shallow-only", action="store_true")
    ap.add_argument("--K", type=int, default=16, help="max recovery prefix length (tokens)")
    ap.add_argument("--n-deep", type=int, default=2, help="deep examples per behavior per arm")
    ap.add_argument("--victim", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="tokenizer used for k-token prefix cuts (CPU only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    behaviors = [json.loads(ln) for ln in open(args.behaviors) if ln.strip()]
    rng = random.Random(args.seed)
    arms = {}

    def add(rec):
        arms.setdefault(rec["arm"], []).append(rec)

    tok, prefixes = None, {}
    if not args.shallow_only:
        if not args.harmful_prefixes:
            raise SystemExit("--harmful-prefixes is required unless --shallow-only")
        prefixes = json.load(open(args.harmful_prefixes))
        from transformers import AutoTokenizer  # lazy; CPU-only, no GPU needed
        tok = AutoTokenizer.from_pretrained(args.victim)

    n_missing = 0
    for b in behaviors:
        goal = b["goal"]
        h = b.get("hash") or bhash(goal)
        for r in build_shallow(goal, rng):
            r["hash"] = h
            add(r)
        if not args.shallow_only:
            pfx = prefixes.get(h) or prefixes.get(b.get("id", ""))
            if not pfx:
                n_missing += 1
                continue
            for r in build_deep(goal, pfx, rng, args.K, args.n_deep, tok):
                r["hash"] = h
                add(r)

    os.makedirs(args.out_dir, exist_ok=True)
    for arm, recs in sorted(arms.items()):
        with open(os.path.join(args.out_dir, f"{arm}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    if not args.shallow_only:
        with open(os.path.join(args.out_dir, "CANARY.txt"), "w") as f:
            f.write("Private hardening corpus — contains approved harmful prefixes "
                    "(masked context for recovery training). Do not commit or "
                    "redistribute.\n")

    print("=== hardening corpus ===")
    for arm in sorted(arms):
        print(f"  {arm:18s} {len(arms[arm]):6d} examples")
    if not args.shallow_only and n_missing:
        print(f"  ({n_missing} behaviors had no approved prefix — deep arms skipped there)")
    print(f"  out: {args.out_dir}")
    print("\nTRAINER NOTE: mask `prompt` AND `assistant_prefix`; put loss ONLY on "
          "`target`. The model must learn to refuse GIVEN a harmful start, never to "
          "generate the prefix. (sft_train.py masks only the prompt today — the "
          "deep arms need the prefix added to the mask.)")


if __name__ == "__main__":
    main()
