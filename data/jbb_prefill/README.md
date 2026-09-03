# JBB prefill frozen item set

Frozen for `prereg/jbb_prefill_2026-09-03.md`. 41 JBB-100
behaviors, each with the opening tokens of a real successful single-turn
jailbreak response harvested from Turnstile's `single_prompt_steering_v2`
sweep (alpha=0, coherent, 70B-judged unsafe).

- `manifest_public.json` (committed) — sha256 hashes, token lengths, source
  record ids, the mismatch derangement, and the benign neutral-control
  template. No harmful plaintext.
- `frozen_private/prefixes.json` (**gitignored**, canary'd) — the harmful
  matched/neutral prefixes themselves. Never committed or mirrored publicly.

Prefix lengths (victim tokens): [8, 16, 32, 64]. Mismatched control = a seeded
fixed-point-free derangement of the behaviors (each behavior gets another
behavior's matched prefix). Neutral control = the benign template cut to k.
