"""Harvest + human-approve harmful openings for the deep-arm recovery prefixes
AND the prefill battery (prereg/hardening_depth_2026-10-03.md). Mirrors
scripts/harvest_jbb_prefixes.py.

SAFETY / GATE:
  - Authors NO harmful content. Candidate openings are the first-N tokens of
    completions produced by a HELPFUL-ONLY model on the TRAINING-split behaviors
    (attack-disjoint, per the prereg). Generation is gated behind `harvest` +
    a model id and only runs on a GPU box.
  - Candidates + review page + frozen set are written under data/harden/private/
    (gitignored) with a CANARY, and are NEVER committed.
  - Nothing downstream (deep arms, prefill battery) runs until a human approves
    specific case_ids here — the prereg's hard-constraint human-approval gate.

  harvest : generate candidate openings + a private, canary'd REVIEW.html
  freeze  : emit the approved private prefix set (keyed by behavior hash) that
            harden_build_corpus.py --harmful-prefixes consumes

Usage:
  PYTHONPATH=src python src/harden_harvest_openings.py harvest \
      --behaviors data/harden/split/victim_train.jsonl \
      --model <helpful-only model> --N 20 --out-dir data/harden/private/openings
  PYTHONPATH=src python src/harden_harvest_openings.py freeze \
      --candidates data/harden/private/openings/candidates.json \
      --approve C0 C3 C7 --out data/harden/private/approved_prefixes.json
"""
import argparse
import hashlib
import html
import json
import os

CANARY = "PRIVATE harmful-opening candidates — canary'd, gitignored, never committed.\n"


def bhash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def harvest(args):
    behaviors = [json.loads(ln) for ln in open(args.behaviors) if ln.strip()]
    from model_utils import load_model, generate_text  # lazy; GPU
    model, tok = load_model(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    cands = {}
    for i, b in enumerate(behaviors):
        comp = generate_text(model, tok, b["goal"], max_new_tokens=args.N + 32,
                             temperature=0.0)
        ids = tok(comp, add_special_tokens=False).input_ids[:args.N]
        opening = tok.decode(ids)
        cid = f"C{i}"
        cands[cid] = dict(case_id=cid, behavior_id=b.get("id", ""),
                          hash=b.get("hash") or bhash(b["goal"]),
                          opening=opening, n_tokens=len(ids))
        print(f"[{i + 1}/{len(behaviors)}] {cid} harvested ({len(ids)} tok)", flush=True)
    json.dump(cands, open(os.path.join(args.out_dir, "candidates.json"), "w"), indent=2)
    with open(os.path.join(args.out_dir, "CANARY.txt"), "w") as f:
        f.write(CANARY)
    with open(os.path.join(args.out_dir, "REVIEW.html"), "w") as f:
        f.write("<!doctype html><meta charset=utf-8><title>opening review (PRIVATE)</title>"
                "<p><b>PRIVATE — do not share.</b> Approve case_ids, then run freeze.</p>")
        for c in cands.values():
            f.write(f"<div><b>{c['case_id']}</b> ({c['n_tokens']} tok)"
                    f"<pre>{html.escape(c['opening'])}</pre></div>")
    print(f"\n{len(cands)} candidates → {args.out_dir}/candidates.json + REVIEW.html "
          "(PRIVATE). Review, then freeze the approved case_ids.")


def freeze(args):
    cands = json.load(open(args.candidates))
    approved = {}
    for cid in args.approve:
        if cid not in cands:
            raise SystemExit(f"approved case_id {cid!r} not in candidates")
        c = cands[cid]
        approved[c["hash"]] = c["opening"]  # keyed by behavior hash for the corpus builder
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(approved, open(args.out, "w"), indent=2)
    print(f"froze {len(approved)} approved openings → {args.out} "
          f"({len(args.approve)} approved of {len(cands)} candidates)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--behaviors", required=True)
    h.add_argument("--model", required=True)
    h.add_argument("--N", type=int, default=20)
    h.add_argument("--out-dir", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--candidates", required=True)
    fr.add_argument("--approve", nargs="+", required=True)
    fr.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "harvest":
        harvest(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
