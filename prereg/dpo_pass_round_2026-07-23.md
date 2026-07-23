# Pre-registration — DPO pass-round (pair policy pass-v2)

**Stamp:** 2026-07-23 (authored; no data generated)
**External timestamp:** ⚠ REQUIRED BEFORE LAUNCH per `prereg/README.md` policy —
register this file on OSF/Zenodo, or push a signed tag containing it, and
record the link/tag here: `____________`. Do not start Stage 1 of
`scripts/run_dpo_pass_round.sh` until this line is filled in.

## Motivation

Every objective that has actually trained in this project put gradient on
compliance and against cheating, never on passing (measured pair
compositions: 22% of DPO-r1 and 44% of DPO-r2 chosen examples fail their own
tests; the sole pass-rewarding recipe, GRPO, never took a step). Round 2
Goodharted exactly as that predicts: cheating −6 pts bought with −8 pts total
pass and −13 pts compliance; DPO-r2 is Pareto-dominated by DPO-r1. This round
re-runs "round 2" with the corrected objective (pass-v2: chosen must be
compliant∧passing; compliant∧failing joins the rejected pool) — same starting
checkpoint (DPO-r1), same 45-problem pool the r2 round sampled, same β/LR/
sampling — so the only changed variable is the pair policy.

## Design (frozen)

- Start/reference adapter: `kilojoules/surface-tension-dpo-r1-r32-final`.
- Pool: the r2 round's 45 LCB problems, re-materialized by id
  (`scripts/make_dpo_pass_pool.py`; ids reconstructed from the r2 build log).
- Pair policy `pass-v2` (`build_dpo_pairs.form_pairs`, tested against a
  replay of the r1 log for the v1 baseline): chosen ∈ compliant∧pass only;
  rejected ranked cheat → violating∧fail → compliant∧fail; row-major
  enumeration, max 6 pairs/problem; n=8, T=0.9, max_new=2048.
- Train: `dpo_train.py`, β=0.1, lr=5e-6, defaults otherwise (matches r1/r2).
- Eval: bare-prompt clean-17, n=8, T=0.7, max_new=3072, AST re-checked from
  saved sources (`recheck_eval.py`) — identical protocol and comparison
  numbers as the committed figures: DPO-r1 0.647/0.324/0.074 and DPO-r2
  0.515/0.301/0.015 on compliance / cmp∧pass / cheating (/136).
- Gate before training: ≥60 pairs from ≥12 problems, else abort and
  investigate sampling (no silent under-powered round).

## Pre-registered predictions

1. **Primary: cmp∧pass (win quadrant) beats DPO-r2's 0.301** — the corrected
   objective should not pay round 2's alignment tax. Directional bar:
   cmp∧pass(pass-round) > 0.301; success if it also ≥ DPO-r1's 0.324.
2. **Total pass rate recovers toward DPO-r1's 0.39** (r2 fell to 0.31).
3. **Cheating rises relative to DPO-r2's 0.015.** Rewarding passing while
   constraining style re-values the loop solution; we predict cheating lands
   between DPO-r2 (0.015) and DPO-r1 (0.074). If it exceeds DPO-r1's 0.074,
   that is the interesting outcome: a measured, trained-in comply-vs-perform
   tension — the phenomenon the pressure experiment is built to dose — and it
   must be reported as such, not as a failed run.
4. **Compliance stays within 5 pts of DPO-r1's 0.647** (the pass gradient
   should mostly convert compliant∧fail → compliant∧pass, not de-train
   compliance).

## Decision rules (pre-committed language)

- Every rate is reported /136 with the same conventions as the committed
  figures; zeros are reported with one-sided 95% Clopper-Pearson upper
  bounds, never as "0.000".
- If prediction 3's cheating increase occurs, the word for it is
  "re-priced cheating under a pass objective" — NOT "deception" (no claims
  axis is measured in this round; deception language requires the quadrant
  protocol).
- If the pair gate fails or ≥3 problems hit gen-errors, the round is
  reported as inconclusive-by-supply, not as evidence about the objective.
- Negative result (cmp∧pass ≤ 0.301) is reported with the same prominence
  as a positive one.

## Known limitations (pre-registered)

- One round, one seed, 17 eval problems: arm contrasts of a few points will
  have overlapping CIs; the primary comparison is directional against two
  fixed, already-published numbers.
- The pool ids were reconstructed from the r2 log; if LCB upstream has
  changed any of those problems' tests since May, `make_dpo_pass_pool.py`
  aborts rather than substituting problems.
- Base-model quantization and the Gemma wrapper-stripping caveat apply as
  in all prior rounds (assert nonzero grad movement early — the zero-gradient
  no-op has happened before).
