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

## Reproducing the openings (they are not uploaded — they are reconstructible)

The prefill openings are deterministic first-N-token cuts of harm-judged
responses in the **public** Turnstile repo (`github.com/kilojoules/turnstile`).
`frozen_manifest.json` (committed) carries the sha256 of every opening, so a
fresh `harvest_jbb_prefixes.py` → `freeze_jbb_items.py` against a Turnstile
checkout rebuilds the exact set and you can check the hashes match (verified
byte-for-byte, including the mismatched derangement). This is why the openings
need not be re-published: every experimental input is public or
reconstructible-from-public; only the victim's harmful *continuations* stay
private, and those are regenerated, not needed as input.

## Attribution

JailbreakBench behaviors are the public JailbreakBench/JBB-Behaviors benchmark
(committed with attribution, as `data/DECKS.md` does for LiveCodeBench). The
harmful **completions** are model outputs and stay private.
