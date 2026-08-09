# DPO pass-round (pair policy pass-v2) — results, 2026-07-24

**Pre-registered** before data generation: `prereg/dpo_pass_round_2026-07-23.md`,
externally anchored at
https://gist.github.com/kilojoules/f53653978b644d5dc02458a732f566d5.
**Verdict: the primary prediction FAILED.** Reported per the prereg's
decision rules — negative with the same prominence as a positive, committed
convention throughout (/136 total attempts, 17 clean problems × n=8;
gen-errors/truncations count as non-compliant; AST re-checked from saved
sources). Raw evidence synced in `vast_logs/bpj6wmav30iudo/` (eval CSV,
sources, pairs, full logs); adapter:
`kilojoules/surface-tension-dpo-pass-r32-final`. Pod cost ≈ $27.

## Run integrity

- Pool: exact r2 45-problem pool re-materialized 45/45 by id.
- Pairs (sampling DPO-r1, T=0.9, n=8): **141 pairs / 24 problems** (gate
  floor 60/12; r1 had 113/19, r2 126/21). Composition ≈ half the new
  `comp∧pass ≻ comp∧fail` type, ~20% `≻ cheat`, ~30% `≻ viol∧fail`.
  Sampling stats: DPO-r1 is 80% compliant on its own training pool
  (209 comp∧pass / 70 comp∧fail / 34 cheat / 37 viol∧fail of ~350).
- Training verified real (loss 0.68–2.05 → 0.000, pref-acc 50% → 100% by
  step ~370 of 423; β=0.1, lr=5e-6, 3 epochs, max_length 2048, reference
  anchored to DPO-r1 — identical to r1/r2 except the pair policy).

## Headline table (clean-17, /136, committed convention)

| metric | DPO-r1 | DPO-r2 | **DPO-pass** | prereg prediction | verdict |
|---|---:|---:|---:|---|---|
| compliance | 0.647 | 0.515 | **0.574** | within 5 pts of 0.647 | ✗ (−7.3) |
| cmp∧pass (primary) | 0.324 | 0.301 | **0.279** | > 0.301 | **✗ FAILED** |
| cheating | 0.074 | 0.015 | **0.022** (3/136, ≤0.056¹) | in [0.015, 0.074] | ✓ |
| total pass | 0.39 | 0.31 | **0.301** | recover toward 0.39 | ✗ |
| no-code output | 0.169 | 0.221 | **0.257** | (not predicted) | — |

¹ one-sided 95% Clopper–Pearson; never a bare near-zero.

Secondary (descriptive, code-emitting rows only, n=99): compliance 0.788,
cmp∧pass 0.384, conditional pass 0.414. The two denominators give opposite
signs on the primary — the /136 committed convention governs, as
pre-registered; both are reported so nobody has to trust the choice.

## Reading

**Putting gradient on passing did not recover passing.** Relative to DPO-r2
(same start checkpoint, same pool, same hyperparameters, only the pair
policy changed): compliance partially recovered (+6 pts), cheating
re-priced slightly upward as predicted (0.015 → 0.022), but total pass
stayed at r2's floor and the win quadrant *fell* (0.301 → 0.279).

Where the mass went: **no-code responses, 26%** — continuing a monotone
trend across DPO rounds (r1 17% → r2 22% → pass 26%; r1-vs-pass diff
≈ 1.8 SE, a trend not a certainty). Mechanism hypothesis, post-hoc and
labeled as such: in every round so far — v1 and v2 policies alike — **every
rejected example contains code**, so each DPO round pushes probability mass
away from code-emission per se; suppressing the code block is the cheapest
way to increase preference margin on 141 pairs (training reached
loss ≈ 0 / 100% accuracy — the pairs are fully separable, so whatever
feature separates them cheapest gets amplified). The logical extreme of
"don't emit failing code" is "don't emit code" — the same Goodhart the
pass-v2 policy was built to fix, one level up.

**Implication: the alignment tax at rounds ≥2 lives in the iterated-DPO
mechanics at this scale, not in the pair policy.** Pair-policy engineering
addressed the objective and the objective improved (cheating stayed low,
compliance partially recovered) — but generation itself degrades each
round. Candidate next steps, in rough order of cheapness:

1. **Early stopping / fewer steps** — loss hit ~0 by step ~370; 1 epoch may
   capture the signal before the degenerate margin-maximizer takes over.
2. **No-code outputs as rejected examples** — directly punish silence
   (`comp∧pass ≻ no-code`), making code-emission itself carry positive
   gradient for the first time.
3. **Verifiable-reward RL revisited**: this eval shows 3 problems in the
   [0.3, 0.7] discriminating zone and 10 strictly-mixed — the bimodality
   that zeroed GRPO's advantage in May has softened under the trained
   policies; the multitier reward (+2 comp∧pass / +1 comp∧fail / −1 cheat)
   may now have gradient to work with.

## Addendum 2026-08-09: "no-code" is truncation-by-deliberation, not code suppression

A transcript audit (all 35 no-code raw outputs re-read; verification script
result: 35/35 end inside an **unclosed** ```python fence, median 91% of
non-blank in-fence lines are `#`-comment deliberation, 0 refusals, 0
prose-only, 0 empty) shows the "no-code output" category is mechanically
**truncation at the token cap during in-code rule-deliberation** — long
`# Wait, the prompt forbids...` self-argument chains about a constraint the
bare prompt never stated — not suppression of code-emission. The same
mechanism accounts for the no-code rows in the SFT scaling arms (22/22 and
29/29 audited for rationale_b149/stripped_b149).

This revises the mechanism hypothesis above: iterated DPO's monotone
no-code trend (17% → 22% → 26%) is **escalating compliance-deliberation
verbosity** (raw mean 929 words vs 448 in the chosen training examples),
which overruns max_new_tokens — likelihood displacement showing up as
anxious rule-lawyering, not as declining code probability. The behavioral
tax is real either way (nothing extractable = fails /136), but the story
"every rejected example contains code → mass moves away from code-emission
per se" overstated the case; the displaced mass went into deliberation
*about the rule*, inside the code block. Full audit in the 2026-08-09
prose-theme study (prereg/propensity_panel_2026-08-09.md, motivation
section).

## Prereg accounting

Predictions 1 (primary), 2, and 4: **failed**. Prediction 3 (cheating
re-prices upward, within [0.015, 0.074]): held. Pre-committed language
applies: the cheating movement is "re-priced cheating under a pass
objective," not "deception" (no claims axis was measured). No gate
failures: pair supply 2.3× the floor; eval wrote all 136 rows (the 37
non-generations are model behavior — 35 no-code, 2 over-cap — not
harness errors; 0 harness exceptions in the eval log).
