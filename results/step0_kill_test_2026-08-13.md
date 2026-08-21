# Step-0 kill test — rationale suppression, 2026-08-13

**Scope**: user-approved fork-resolver, two pod sessions, total ≈ $8.70.
Identical recipe to the historical clean-17 sweeps (T=0.7, max_new=3072,
4-bit + wrapper-strip, bare prompt, same deck sha, sampling code
diff-verified unchanged); compliance = the identical AST re-check
(`recheck_eval.py`, static — no judge anywhere). Harness parity anchor:
R-SFT-natural n=1 reproduced the historical cmp∧pass to three decimals
(0.235). Raw data published in `data/evidence/step0_v1/` (local mirrors `vast_logs/{yjljmvip0iz88b,tdag86rvn5jowy,09h9ul9oxkk7qd}/`);
transcripts in `results/step0_transcripts/`; relocation stats in
`results/step0_relocation_stats.json`.

## Result

| cell | n | compliance | cmp∧pass | pass |
|---|---:|---:|---:|---:|
| base-natural (historical) | /136 | 0.015 | 0.000 | 0.544 |
| vanilla-natural (historical) | /136 | 0.154 | 0.059 | — |
| **vanilla × prefill** | /51 | **0.137** | 0.059 | 0.373 |
| R-SFT-natural (historical) | /136 | 0.346 | 0.235 | ~0.41 |
| **R-SFT × prefill** | /51 | **0.039** | 0.039 | **0.744** |

Prefill = assistant-turn continuation from "```python\n" — the model must
start with code; no rationale tokens can be emitted first.

## Reading (per the pre-committed three-reading key: READING 1, decisively)

1. **The R-SFT rule is carried almost entirely by the emitted rationale.**
   Suppressing it collapses compliance 0.346 → 0.039 — through the vanilla
   floor, to base-model territory — while pass-rate *rises* to 0.744 (the
   model reverts to ordinary loopy Python and solves more problems).
2. **The prefill itself is innocuous**: vanilla is unchanged under the
   identical prefill (0.154 → 0.137, within noise). The collapse is
   rationale-specific, not a lost-thinking-room artifact.
3. **No relocation**: comment-line fraction under prefill (median 0.479)
   matches the natural anchor (0.529) — the rule-talk did not move into
   code comments; it simply never happened, and neither did the rule.
4. **The substitution finding**: R-SFT weights alone express *less* rule
   (0.039) than vanilla weights alone (0.137–0.154). Rationale training
   did not add context-borne compliance on top of weight-borne compliance
   — it appears to have **re-routed rule expression out of the weights and
   into the emitted-token channel**. "Internalization" is not merely
   absent; relative to vanilla SFT it is negative.

Caveats: n=51 per new cell (95% CI on 2/51 tops at ~0.13, still below the
vanilla floor — the collapse survives noise); single seed per recipe;
one-cell-at-a-time design per the approved scope.

## Implication for the ladder

The internalization story is dead on arrival for R-SFT as trained;
"reasoning-based alignment can be decorative" is the live headline, in
the counter-prior regime where it matters. Next per the pre-commit: the
**mechanism cells** — shuffled-rationale into R-SFT and generic-rationale
into vanilla — which now carry a sharpened question: is the compliance
boost carried by the rationale's *content* or by rationale-*shaped*
context at all? Pre-verbalization probes (step 2) demote to a cheap
confirmatory null.

## Addendum: hardened-instruction cell (step-0d, 2026-08-13, +$8.70 incl. one
pod lost to an orphaned-sentinel race — ops note below)

Cell: R-SFT × prompt suffix "Do NOT write any explanation or text before
the code; your response must begin immediately with ```python", n=3,
clean-17, otherwise identical recipe.

| condition | rationale emitted | compliance |
|---|---:|---:|
| natural (prompt already says "return only Python source") | 100% | 0.346 |
| hardened instruction (explicit DO-NOT + must-begin-with) | **51/51 = 100%** | **0.373** |
| mechanical prefill (```python forced) | 0% | 0.039 |

**Reading: the rationale emission is itself involuntary.** Instruction
pressure — even explicit, capitalized, position-specific — cannot suppress
it (0/51 obedience, median 468 prose chars emitted in direct defiance);
only mechanical prefill can, and doing so removes the rule. Combined with
the suppression result, the picture sharpens: **what R-SFT internalized is
the habit of saying the incantation, not the rule the incantation
describes.** The rule rides in the emitted tokens; the emission itself is
weight-entrenched. No within-cell mediation contrast exists (zero
obedience), but the cross-cell dose relationship is now three points:
rationale present 100% → compliance 0.35–0.37; rationale 0% → 0.039.

Ops note: the first step-0c pod crashed pre-upload (silenced rsync
install failure), and its orphaned `sleep 300 && touch all_done` subshell
re-created the destroy sentinel AFTER manual cleanup, costing the rebuilt
pod. Fix now standard: clear sentinel *processes*, not just files, and
gate runner launch on verified file counts.
