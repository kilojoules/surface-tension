# Prereg: powering the step-0 substitution contrast to the /136 convention

**Registered:** 2026-09-02, BEFORE any new generation.
**Parent result:** `results/step0_kill_test_2026-08-13.md` (the cells being
extended) and its Reading #4, the claim under test.
**External timestamp anchor:**
https://gist.github.com/kilojoules/bf0660fd4b2f42948e4517c97430acee
(public gist, created 2026-09-02 before launch; anchors the pre-URL version
of this file, sha256 `bc9af8d7…4742` — the gist carries the full anchored
text.)

## Why this run exists (full disclosure of what is known at registration)

A cluster-aware re-analysis of the *already published* step-0 cells was run
on 2026-09-02, before this prereg, and motivated it. Everything it found is
disclosed here; nothing below is conditioned on data that does not yet exist.

Re-derived from the committed evidence package with the same AST re-check
(`src/recheck_eval.py` over `data/evidence/{step0_v1,naturals_v1}/`), all
counts reproducing the published doc exactly:

| cell | count | rate |
|---|---|---|
| R-SFT natural | 47/136 | 0.346 |
| vanilla natural | 21/136 | 0.154 |
| base natural | 2/136 | 0.015 |
| R-SFT × prefill | 2/51 | 0.039 |
| vanilla × prefill | 7/51 | 0.137 |

Cluster bootstrap over the 17 problems (20k reps, resampling problems as the
unit; measured design effect 1.97× for the R-SFT prefill cell and 1.67× for
the vanilla prefill cell):

| contrast | Δ | 95% CI | p |
|---|---:|---|---:|
| collapse: R-SFT natural − prefill | +0.306 | [+0.164, +0.463] | <0.0001 |
| innocuous: vanilla natural − prefill | +0.017 | [−0.054, +0.083] | 0.65 |
| **substitution: vanilla prefill − R-SFT prefill** | **+0.098** | **[−0.039, +0.255]** | **0.25** |
| R-SFT prefill − base natural | +0.025 | [−0.044, +0.118] | 0.78 |

So: the collapse is robust; the innocuousness null is intact; **the
substitution ordering that Reading #4 asserts is not established**, and the
R-SFT prefill cell's entire compliant signal is **one problem of seventeen**
(`lcb/abc363_c`, 2 of 3 samples). Vanilla's is 4 of 17.

This run exists to replace a one-problem estimate with a properly sampled one.

## Question

At the study's committed sampling convention, does R-SFT's weight-borne
compliance (compliance under rationale suppression) sit **below** vanilla
SFT's — i.e. does the negative-internalization / substitution reading of
Reading #4 survive?

## Design (fixed before launch)

- **Two cells extended, nothing else touched**: R-SFT × prefill and
  vanilla × prefill, from n=3 to **n=8 samples per problem** — the /136
  convention every other cell in this study already uses (17 clean problems
  × 8). 85 new generations per arm, 170 total.
- **Pooling with the existing 51**: identical recipe in every respect
  (below), sampling is unseeded at T=0.7, so old and new draws are
  exchangeable. `sweep_local.py` resumes on the key
  `(problem_id, constraint, condition, sample_idx)`; each pod is seeded with
  the committed CSV + sources and generates only `sample_idx` 3–7.
  Pooling is **gated** on the homogeneity check in Prediction 4.
- **Recipe, byte-for-byte from `vast_logs/{yjljmvip0iz88b,tdag86rvn5jowy}/run_step0*.sh`**:
  `LOAD_STRIP_WRAPPERS=1 QUANT_BIT=4`, base `google/gemma-4-31B-it`,
  adapters `kilojoules/surface-tension-sft-rationale-r32-final` (R-SFT) and
  `kilojoules/surface-tension-sft-rankcurve-r32-final` (vanilla),
  `--max-new-tokens 3072 --temperature 0.7 --constraints` (no values → bare
  prompt only), prefill = the three-backtick `python` fence opener + newline,
  deck `data/problems_lcb_clean17.jsonl`. **Quantization stays 4-bit**: bf16
  would be ~3× cheaper but is a different measurement, and every historical
  number in this contrast is 4-bit.
- **Ops**: one pod per arm (independent failure domains, same total bill),
  A100-SXM4-80GB SECURE ≈ $1.49/h. Measured throughput 4.25 min/generation
  (51 gens in 3h37m) → 85 gens ≈ 6.0 h + ~0.4 h setup ≈ **$9.6/pod, ~$19
  expected**. `MAX_HOURS=13` per pod (2.03× best case) → **$39 hard ceiling**.
  No `sleep N && touch` sentinel anywhere (the 2026-08-13 orphaned-sentinel
  race destroyed a pod); watchdog process grep must match the runner before
  launch.

## Power, disclosed up front

At n=136/arm, the observed Δ=0.098 and design effect 1.8, SE(Δ) ≈ 0.045 →
**power ≈ 58% two-sided (≈70% one-sided)**. This is an **estimation run, not
a detect-or-bust run**. It halves the CI half-width (0.147 → ≈0.089). A null
will therefore be reported as a *bound*, never as evidence of no effect.
Reaching 80% power would need n≈14/problem ≈ $40, which was declined on cost.

## Predictions

1. **(Primary) Substitution:** the cluster-bootstrap 95% CI on
   (vanilla prefill − R-SFT prefill) at /136 **excludes 0 in the positive
   direction**. Failure = CI contains 0, or is negative.
2. **Collapse replicates:** R-SFT × prefill ≤ 0.10 at /136. (Point estimate
   0.039; this fails only if the published collapse was itself a small-n
   artifact.)
3. **Vanilla stable:** vanilla × prefill within [0.06, 0.25] at /136 —
   consistent with both its own 0.137 and its natural 0.154.
4. **(Gate) Homogeneity:** within each arm, the new 85 draws vs the existing
   51 differ at Fisher p ≥ 0.01. Failure → **do not pool**; report the cells
   separately and investigate environment drift before any reading.
5. **Spread diagnostic:** if R-SFT's true prefilled rate is ≈0.04, ≥3 of 17
   problems should show ≥1 compliant generation at n=8. If the signal stays
   concentrated on `abc363_c` alone, the cell is problem-specific and is
   reported that way regardless of the rate.

Tracked secondaries, no predictions: pass rate (published 0.744 for R-SFT
prefill — the "reverts to loopy code and solves more" claim), cmp∧pass,
no-code rate, per-problem rates for both arms.

## Decision rules (pre-committed language)

- **P1 holds** → the substitution reading stands. `results/step0_kill_test_2026-08-13.md`
  Reading #4 is restated with the /136 numbers and the CI, and
  `paper/figs/step0_substitution` keeps its claim but gains bootstrap CIs.
- **P1 fails with a tight interval** (half-width ≤ 0.09) → the
  negative-internalization / "re-routed the rule out of the weights"
  language is **retracted at equal prominence** in the results doc, the
  README, and the figure title/docstring. The replacement claim is the one
  the data support: *with the rationale suppressed, R-SFT retains no more
  rule than base, and cannot be distinguished from vanilla.* The collapse
  (P2) is unaffected and is stated on its own.
- **P1 fails with a wide interval** → reported as inconclusive plus the
  bound; no claim in either direction; the doc says so.
- **P2 fails** → a larger correction than P1: the headline collapse is
  weaker than published and gets equal-prominence treatment immediately.
- **P4 fails** → cells are not pooled; this is an integrity finding about the
  harness/environment, reported before any substantive reading.
- Regardless of outcome, the figure gains uncertainty intervals; shipping
  bar charts with no error bars is what let a one-problem estimate read as a
  result.
- All rates reported /136. Clopper–Pearson bounds for any near-zero cell;
  never a bare zero.
