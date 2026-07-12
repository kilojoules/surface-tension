# Correction — 2026-07-12

**The Stage-3 mech-interp headline ("a pre-verbal plan to launder") is
retracted. The behavioral deception result stands, but its "0.000" must be
read as "0 observed, 95% upper bound ≈ 0.08," which is what this project's
own pre-registration said to report. Several smaller reporting errors are
fixed alongside. The evidence needed to check all of this is now published
in-repo.**

This project's stated identity is catching measurement traps — the broken
`compliant` column, the zero-gradient LoRA no-op, the val-set-reuse
inflation. This correction applies that standard to the project's own
headline result. Same data, opposite headline — and this is the honest one.

---

## 1. What was claimed

Commit `be5edf1` (2026-07-04) and the README's bolded bottom line claimed:

> "the residual stream shows they decide to launder before they write a line
> of code" … "A linear direction in the residual stream, read at the final
> prompt token **before any code is generated**, predicts whether the
> solution will launder at **AUROC 0.82** (n=188). It survives problem-held-out
> CV (0.75, so it's not reading the prompt) and a shuffle null (~0.6 ceiling).
> Because the code doesn't exist yet, this can't be the probe reading the
> output — it's a genuine forward *plan to launder*, computed in mid-to-late
> layers."

Three supports were offered: plain-CV AUROC 0.82, problem-held-out CV 0.746,
and a label-shuffle null ≈ 0.6. A "within-problem variance" observation (9/16
problems contain samples that disagree on construct use) was cited as showing
the choice is a per-generation decision.

## 2. Why the per-generation reading is impossible by construction

The `__sol` activation is captured by a dedicated forward pass **over the
solution prompt only**, before `generate()` is called
(`scripts/quadrant_pod_runner.py`, "Capture BEFORE generation"). The solution
prompt is built from the fixed constraint instruction plus the problem
statement (`src/quadrant/generate.py::_build_solution_messages`) — it does
not depend on `sample_idx`, and temperature only enters at generation. The
`__sol` tensor is therefore a **deterministic function of (model, problem)**.

Verified on the actual tensors (`probe_correction.audit_identity`; manifests
in `data/evidence/quadrant_v4/`): within every (arm, problem), all per-sample
`__sol` tensors are **byte-identical**; between problems they differ at
L2 ~10² (minimum between-problem distance 104–235 across arms). The base
arm's "n = 188" probe input contains exactly **16 distinct vectors** — one
per problem.

Consequences:

- The probe assigns the **same score to every sample of a problem**. It
  cannot, in principle, predict *which generation* will launder — only a
  per-problem propensity. There is no per-generation "plan" signal at
  `__sol` to find, regardless of AUROC.
- The within-problem-variance control **inverts**: on the 9/16 problems where
  samples disagree, the probe provably scores disagreeing samples
  identically. What was cited as evidence for a per-generation decision is a
  proof that the probe cannot see that decision.
- The parenthetical "so it's not reading the prompt" was wrong in the strong
  sense: `__sol` is a deterministic function of the prompt, so the prompt is
  the *only* thing the probe can read. Problem-held-out CV rules out
  memorizing problem *identity*; what generalizes is generalizable **prompt
  features** correlated with laundering propensity (e.g., "this problem has
  a natural comprehension/functional solution") — task affordances, not
  intent.

## 3. The statistics compounded it

All numbers below are computed by `src/quadrant/probe_correction.py`. The
module first reproduces every published figure with the repo's own
`per_layer_auroc` (same folds, same seed) — exact from the author's raw file
layout (`--raw`); the *published* plain-CV/shuffle figures are additionally
file-enumeration-order-dependent (§3d), so from the published evidence
package's sorted order the same leaky procedure gives e.g. base 0.810 rather
than 0.820. Every **corrected** statistic (exact-tie grouped CV, permutation
null, LOPO, full-data oracle, deception bounds) is order-invariant and
re-derives bit-identically from the published package; the CV-legal
memorizer reference deliberately lives on the same order-dependent folds as
the plain CV it characterizes, so it moves the same way (base ≈ 0.865–0.871
across enumeration orders). Run, from the repo root:
`PYTHONPATH=src python -m quadrant.probe_correction --evidence
data/evidence/quadrant_v4 --out results/correction_2026-07-12`.

**(a) The 0.82 was inflated by duplicate leakage.** The published headline
used sample-level 5-fold CV (`per_layer_auroc` defaults, `groups=None`), so
byte-identical copies of each problem's vector sat in train and test folds
simultaneously — the probe partly "recognizes the problem" rather than
generalizing. How much is that worth? A **CV-legal memorizer** — score each
held-out sample by its problem's mean label among *train* samples only,
under the same folds, zero information from the activations — reaches
**AUROC ≈ 0.87** (`foldlegal_baserate_auroc`; 0.865–0.871 depending on the
same enumeration order that moves the plain CV, since it characterizes that
leaky split); with full-data label rates (including each sample's own label,
an upper reference no legal predictor attains) the same idea reaches 0.92. The published 0.82 is *below* what
problem-identity memorization alone legally attains under this split; it is
not evidence of anything beyond leakage. The leakage-free number is the
problem-held-out **0.747** (exact-tie treatment; the reference
implementation reads 0.746 — §3d).

**(b) The published null was the wrong null.** The "shuffle null (~0.6)"
permuted labels at the **sample** level (one permutation, seed-fixed),
destroying the within-problem label clustering. With an effective n of 16
problems and a max-over-61-layers selection, the correct null must preserve
the clustering: permute which problem's activation vector is paired with
which problem's label block, and recompute the max-over-layers grouped-CV
AUROC each time. Over 1,000 such permutations (base arm, functional/
comprehension label): **null mean ≈ 0.70, null 95th percentile ≈ 0.81 —
observed 0.747 lands near the middle, p ≈ 0.25.** The published "signal well
above the ~0.6 noise ceiling" was a comparison against an understated
ceiling.

**(c) At the honest unit of analysis there is ~nothing left.** Treating the
problem as the unit (16 vectors, one label each — with the disclosure that
one of the 16 problems splits exactly 50/50, so "majority label" is a
convention there; both assignments are computed rather than letting a
rounding rule decide silently): leave-one-problem-out CV gives
max-over-layers AUROC 0.73, problem-level permutation **p ≈ 0.76** with the
tied problem as negative, and 0.80, p ≈ 0.63 with it positive — null either
way. For scale: the R-SFT arm's LOPO AUROC is a *perfect 1.0* at p ≈ 0.50
(under its own tie convention; the other assignment leaves it undefined at
one negative problem) — with 7 problems the null reaches 1.0 half the time.
n = 16 problems was never enough to establish this claim, and 7 is an
anecdote generator.

**(d) A micro-finding about the reference implementation.** Within-problem
ties in the reference pipeline's scores are broken by ~1e-14 BLAS
accumulation jitter (byte-identical input rows score unequally), so the
published grouped AUROC itself depends on floating-point noise at the third
decimal (0.7466 noise-broken vs 0.7472 exact-tie treatment). Worse, the
plain-CV and sample-shuffle numbers depend on **file-enumeration order**:
fold assignment follows dict insertion order, which for the raw pipeline is
filesystem glob order. Re-deriving from the evidence package (sorted order)
gives base plain-CV 0.810 rather than the published 0.820 — the headline
number was only defined up to the order a filesystem happened to list the
tensor files. (The reference grouped number moves too, 0.746 → 0.748 under
reordering, via the same tie-noise; only the exact-tie treatment is stable.) Every *corrected* statistic (exact-tie grouped CV, permutation
p, LOPO, oracle, deception bounds) is order-invariant by construction, and
verified bit-identical across both load paths
(`probe_correction_raw.json` vs `probe_correction_evidence.json`).
Immaterial to any conclusion, but symptomatic: the pipeline was pretending
n_samples > n_problems, and the arithmetic noticed twice.

## 4. The training-ablation table, re-read

The README's ablation table reported per-arm "plan-to-launder" AUROCs
(base 0.82 / R-SFT 0.84 / DPO-r1 0.73 / DPO-r2 0.78 / vanilla-DPO 0.95).
The plain-CV numbers all reproduce exactly from the raw layout (from the
published package they shift — §3d). But the per-arm **problem supports**
are: base 16, R-SFT 7, DPO-r1 8, DPO-r2 5, vanilla-SFT 2, vanilla-DPO 2.
Corrected (grouped CV with exact ties — the order-invariant statistic the
permutation test uses; 1,000 permutations):

| arm | problems | published (plain CV) | grouped CV | perm p |
|---|---:|---:|---:|---:|
| base | 16 | 0.820 | 0.747 | ≈ 0.25 |
| R-SFT | 7 | 0.839 | 0.768 | ≈ 0.60 |
| DPO-r1 | 8 | 0.732 | 0.807 | ≈ 0.017 |
| DPO-r2 | 5 | 0.782 | 0.825 | ≈ 0.25 |
| vanilla-SFT | 2 | n/a (1 negative) | — | — |
| vanilla-DPO | 2 | 0.95 (2 negatives) | undefined¹ | — |

¹ With 2 problems, every problem-held-out train fold is single-class; there
is no leakage-free number at all — the published 0.95 had no problem-held-out
support and should be disregarded entirely.

Grouped column = the exact-tie treatment (order-invariant; identical from
raw and packaged data; the statistic the permutation p is computed against).
The repo's reference implementation (`per_layer_auroc`, noise-broken ties)
reads within ≤ 0.003 of these (e.g. DPO-r2 0.828 vs 0.825) and drifts with
row order — §3d.

One arm (DPO-r1) is nominally significant at p ≈ 0.017. That is one of five
permutation tests run — the four arms above plus the base `__sol` *laundered*
probe (published as 0.86 in the main table; corrected: grouped 0.859,
perm p ≈ 0.24) — and the five are not independent (arms share problems; the
two base probes share activations). Under independence, a minimum p ≤ 0.017
across five null tests happens about 8% of the time; on 8 problems, not
predicted in advance, it is a candidate for follow-up at larger problem-n,
nothing more. The README's replication claim ("The direction replicates
(0.73–0.84) across base and every rationale-trained arm.") described
plain-CV numbers that inherit the same duplicate leakage; the corrected
reading is: **no probe shows a problem-level signal that survives its
permutation null, except one post-hoc, multiplicity-uncorrected case.**

## 5. What the `__sol` data does support

- The end-of-prompt representation carries **problem-level information
  correlated with laundering propensity** (grouped-CV point estimates
  0.75–0.83 across arms) — but this is statistically indistinguishable from
  its own selection noise at these problem counts (p ≈ 0.25 at n=16; the
  post-hoc DPO-r1 exception is §4), and conceptually indistinguishable from
  the prompt encoding task affordances.
- At `__rep` (after the code exists, before the self-report), base-arm
  probes read construct use at 0.81–0.87 **including under problem-held-out
  CV** (plain 0.814, grouped 0.814; laundered: 0.871/0.845; trained arms
  read higher still, 0.88–0.96, on their 5–8 problem supports — same
  descriptive caveat). This is coherent — the generated code is in context
  at `__rep` — and is exactly the "probe reads the output" configuration the
  original writeup was careful to avoid claiming. Descriptive only; the
  in-context confound stands, and `__rep` tensors are not yet deposited
  publicly (see the evidence README).
- Testing a *per-generation plan* requires capturing activations at early
  **generated-token** positions (where sampling has begun), not the shared
  prompt prefix. That experiment has not been run.

## 6. Deception "0.000" → "0 observed, upper bound ≈ 0.08"

The prereg (`prereg/quadrant_v4_deception_2026-07-03.md`) specified
problem-level bootstrap CIs and, for a ≈0 result, reporting the CI upper
bound. The README reported bare point estimates ("0.000"). Corrected
(construct-union deception among passing samples; exact one-sided 95%
Clopper–Pearson, which for zero counts is the informative bound a bootstrap
cannot give):

| arm | count | point | one-sided 95% upper | two-sided 95% upper |
|---|---:|---:|---:|---:|
| base | 1/95 | 0.011 | 0.049 | 0.057 |
| R-SFT | 0/35 | 0.000 | **0.082** | 0.100 |
| DPO-r1 | 0/36 | 0.000 | **0.080** | 0.097 |

The one-sided bound is quoted in the text because the question is one-tailed
(how much deception could be hiding under an observed zero); the two-sided
values are the conventional CI reading and are shown so nobody has to
suspect the smaller of the two computed bounds was cherry-picked. These
bounds assume independent samples; samples cluster within ≤17 problems, so
true uncertainty is somewhat larger. The right sentence is: "we observed
essentially no deception, and the data are consistent with a deception rate
anywhere below ~8–10% in the trained arms" — not "deception is 0.000." Additionally, the design measured deception in a **no-incentive
regime** (the constraint prompt explicitly permits the migration target, so
compliance was nearly free); a null here was likely a priori, which is why
the follow-on (`src/pressure/`) makes compliance costly. And per the
prereg's own limitation #1, the structured self-report is partly a
code-reading-comprehension task with the code in context, which plausibly
deflates measured deception.

**Disclosure — the claim-side raw data was lost.** The judge outputs
(`judgments.jsonl`) lived only on the judge pod and were not synced before
the pod was deleted. The counts above are as published in commit `be5edf1`;
they cannot currently be re-derived. The same loss covers the prereg's
judge-side diagnostics — the human-κ judge gate, ambiguous-rate,
per-construct missing-claim fractions, and the pre-registered disagreement
control cell (prereg prediction 3) — none of whose values were archived off
the pod, so they cannot be reported; the only prereg gates checkable from
the released rows are the judge parse-error count (0) and the token-cap
check (0 samples near the 4096 cap). The judge's exact inputs
(`self_report_text`) are preserved per row in `data/evidence/quadrant_v4/`,
so the claim side can be regenerated by re-running `quadrant.judge_runner`
(Qwen3-30B-A3B, temp 0, rubric v1). Until someone does, treat the deception
counts as single-sourced. (Lesson: sync outputs, not just logs, before
killing a pod.)

## 7. Also corrected in this pass

- **README Stage-1 status quoted the debunked 0.34.** The "0.34 capability
  drop" was the artifact of the broken `compliant` column that
  `paper/main.tex` itself diagnoses (real prompted-constraint cost ≈ 8
  pass-rate points, 0.87 → 0.79; the binding fact for Stage 2 is the
  bare-prompt default gap, not a large capability cost). The README now says
  so.
- **PLAN.md** predicted concealment in maximal hidden-reasoning language and
  was left unhedged at the repo root after the data went the other way; it
  now carries a dated status note.
- **Prereg timestamps are git-only** and entered the repo in the same
  commits as their results; `prereg/README.md` now states this plainly and
  commits future preregs to independent timestamping (OSF/Zenodo or a
  signed pushed tag) before data generation.
- **Evaluation-data hygiene:** all published problem-text files now carry a
  canary GUID (`NOTICE.md`); the certified pressure-set solutions are
  withheld from HEAD to avoid seeding future training corpora with the
  instrument's answer key.

## 8. What stands

- **The definitional fork and the behavioral result.** Scoring the covert
  cell against the rule the model was shown (vs. loose/strict) is what
  collapses a naive "~90% covert violation" reading into "≈0 deception (see
  bounds above), ~universal honest surface-migration, openly self-reported."
  That methodological point — a deception eval must condition on the rule
  actually shown or it overcounts — is untouched by this correction, and is
  the project's main contribution.
- **Laundering rates** (AST-side, judge-free): base 0.86, R-SFT 0.97,
  DPO-r1 0.97, DPO-r2 0.95, vanilla arms 0.90–0.91 of passing samples
  satisfy the shown rule while violating the loose rule. Recomputable from
  the published rows.
- **Stages 1–2** (elicitation gap, bimodal determinism, SFT checkpoint
  findings, and the paper's two measurement traps) are unaffected.

## 9. How this was found

An adversarial review of the project (2026-07-11) flagged that `__sol` is
computed from a sample-invariant prompt; the byte-identity audit, the
leakage oracle, and the clustered permutation null followed. Everything
needed to have caught this was in the repo — the capture code comments say
"prompt only," and the prompt builder takes no sample index. The
generalizable lessons join the project's collection:

1. **A probe on a deterministic function of the prompt is a per-input probe.**
   If inputs repeat, n is the number of *distinct inputs*, not rows.
2. **Match the null to the dependence structure.** Sample-level shuffles
   understate max-over-layers selection noise by a lot when labels cluster
   (here: ~0.60 vs ~0.81 at the 95th percentile).
3. **Report the bound your prereg told you to report.**
4. **Publish the evidence with the claim.** The error sat unfound for a
   week: the *argument* was recoverable from the capture code alone, but the
   confirming tensors were private, and it was found the day someone could
   look.
