# JBB prefill item set (candidates → review → approved freeze)

Supports `prereg/jbb_prefill_2026-09-03.md`. Tests whether refusal is a
token-borne gate by continuing the assistant turn from the opening of a
response that already succeeded on the same JailbreakBench behavior.

## Flow

1. `scripts/harvest_jbb_prefixes.py` reads Turnstile's existing single-turn
   steering sweep and generates **candidate** prefixes: 41 JBB-100 behaviors ×
   3 structural start points (S0 opening / S1 post-preamble / S2 first step) ×
   k ∈ {8,16,32,64} victim tokens. No new harmful text is generated.
2. Candidates render to `frozen_private/REVIEW.html` — a **local, private,
   gitignored** page. The reviewer approves specific `case_id`s for content
   appropriateness and a suitable opening.
3. Only approved cases become the frozen item set (prereg Amendment 1,
   externally re-anchored before any run). Its public manifest (sha256 hashes,
   token lengths — no harmful plaintext) is committed at that point.

## What is committed here (public)

- `baseline_items.json` — the public JailbreakBench behaviors + a benign
  "Sure, I can help…" neutral prefix. Ships to the $2 baseline plumbing pod;
  contains **no** harmful completions.
- This README.

## What is NOT committed (private, gitignored, canary'd)

- `frozen_private/candidates.json`, `frozen_private/REVIEW.html` — the harvested
  jailbreak openings themselves. Never committed, mirrored, quoted, or logged.

## Reproducing the openings (not uploaded — pinned + reconstructible)

The prefill openings are pinned by `frozen_manifest.json` (sha256 of every
opening). Two ways to obtain them without re-publishing harmful text:

1. **Rebuild from the run's own seed (drift-proof):**
   `scripts/build_jbb_frozen_from_seed.py <seed jbb_full.jsonl>` — the exact
   openings used are in the seed's `prefill` field; short lengths are prefixes
   of the seed's N=20 (verified to reproduce N=5/10/15 byte-for-byte). This is
   how the N=1,2,3 cells were derived.
2. **Harvest from public Turnstile at the pinned commit `fc0a7f4`.** ⚠ The
   Turnstile source grows over time — a fresh harvest against a *later*
   Turnstile drifts (a 2026-09-09 addition changed one behavior and added a
   90th), so pin the commit. `frozen_manifest.json` lets you verify either way.

Only the victim's harmful *continuations* stay private, and those are
regenerated, not needed as input.

## Attribution

JailbreakBench behaviors are the public JailbreakBench/JBB-Behaviors benchmark
(committed with attribution, as `data/DECKS.md` does for LiveCodeBench). The
harmful **completions** are model outputs and stay private.
