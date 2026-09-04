# Surface Tension

![A hot air balloon at altitude](docs/balloon.jpg)

A hot air balloon at a given altitude could be there because it's genuinely light, or because it's carrying sandbags it could drop. Same altitude, two states — from below you can't tell. When we fine-tune a model to follow a rule ("don't write loops," or "don't blackmail the user"), we want the light balloon: a real change in what the model *is*. Surface Tension is a controlled testbed for telling those regimes apart, using arbitrary coding rules (easy to define, AST-verifiable) as a stand-in for safety constraints.

Where the metaphor landed: we went looking for hidden sandbags and found a balloon that, when asked, reads out its ballast accurately — in a regime where carrying sandbags cost nothing. The honest-measurement machinery built along the way (and once turned on this project's own headline — see the correction) is the contribution.

> **⚠ Correction (2026-07-12).** The claim previously made here — that a
> linear probe on the pre-code residual stream revealed "a genuine forward
> plan to launder" (AUROC 0.82) — is **retracted**: the pre-code activation
> is a deterministic function of the problem prompt (byte-identical across
> all samples of a problem — verified), so the probe could only ever read
> per-problem propensity, and the result does not survive a correctly
> clustered permutation test (p ≈ 0.25). The behavioral deception result
> stands but is now reported with its pre-registered upper bound
> (0 observed → ≤ ~0.08) rather than as "0.000". Full analysis, mechanism,
> and re-derivable evidence: [`docs/correction_2026-07-12.md`](docs/correction_2026-07-12.md);
> data: [`data/evidence/quadrant_v4/`](data/evidence/quadrant_v4/README.md).
> The error was found by adversarial review one week after publication;
> everything needed to catch it was in the repo. Same data, opposite
> headline — and this is the honest one.

---

## Bottom line

**A naive deception metric reads this testbed as ~90% covert rule-violation. Conditioning on the rule the model was actually shown collapses that to ≈0 deception (bounded, in a no-incentive condition where compliance was nearly free) — what looked covert is near-universal, openly self-reported migration into explicitly permitted constructs. Any deception eval whose graded rule is stricter than the rule the model was shown will overcount whenever models migrate into the definitional gap — here the overcount was nearly total.**

1. **Deception ≈ 0, bounded.** Scored against the rule *as shown* (loops/recursion forbidden; comprehensions and `map`/`filter`/`reduce` explicitly allowed), the models essentially never deny a forbidden construct they used: base **1/95** passing samples (point 0.011, 95% upper bound 0.049), R-SFT **0/35** (≤ 0.082), DPO-r1 **0/36** (≤ 0.080). Two caveats before quoting this anywhere: the constraint left a **free legal escape** (the migration target was permitted), so this is the *no-incentive control condition*, not evidence that models under pressure don't deceive — the costly-compliance experiment is scoped in [`src/pressure/`](src/pressure/README.md); and the self-report probe shows the model its own code, so it partly measures reading comprehension (prereg limitation #1).
2. **Surface migration ("laundering") ≈ universal, and honest.** 86–97% of passing solutions satisfy the shown rule by moving iteration into permitted comprehension/functional style (AST-side, judge-free: base 0.86, R-SFT 0.97, DPO-r1 0.97), and the models *say so accurately* when asked. "Laundering" here is a term of art for surface-form migration — it is instructed, permitted, honestly-reported behavior, not misconduct; the point of the word is that a strict checker *mislabels* it as covert violation.
3. **The mech-interp result is a null (corrected).** The end-of-prompt (`__sol`) representation carries some problem-level correlation with migration propensity (problem-held-out AUROC 0.75 on 16 problems), but this does not beat a properly clustered permutation null (p ≈ 0.25), and by construction it cannot reflect a per-generation "plan" — the activation is identical for every sample of a problem. Post-code (`__rep`) probes read construct use at 0.81–0.87 even across held-out problems, which is coherent (the code is in context) and deliberately not claimed as intent. There is no deception direction to find because there was essentially no deception in this no-incentive setting; there is no established plan-to-launder direction either.

The methodological spine survives its own correction: **measure claims against the rule the model was shown; report bounds, not zeros; publish the evidence with the claim.**

---

## The research arc

Arbitrary coding rules stand in for safety constraints because they're cheap to impose, AST-verifiable, and benchmark-scorable.

1. **Establish binding pressure** — find a constraint the model doesn't satisfy by default but can satisfy when told. ✅ *Stage 1.*
2. **Bake the constraint in** — fine-tune so the model obeys with no prompt hint. ✅ *Stage 2* (rationale-SFT distillation → DPO).
3. **Detect the sandbag** — quadrant the trained model: complying, laundering, or covertly cheating — and probe what it represents. ✅ *Stage 3, with the 2026-07-12 correction.*
4. **Costly compliance** — re-run Stage 3 where the legal solution is expensive and an overseer rewards claimed compliance: a real comply-vs-cheat choice. *Apparatus built (52 certified problems), not yet run.*
5. **Aligner agent / adversarial trainer** — scoped, not built (see PLAN.md status note).

## Status at a glance

- **Stage 1 — constraint discovery: done.** `no_loops_no_recursion` on Gemma-4-31B, LCB-medium (post-cutoff): bare-prompt compliance is near zero, prompted compliance is 0.65, and the prompted-constraint cost is **≈ 8 pass-rate points** (0.87 → 0.79, n=54). An earlier "0.34 capability drop" figure was an artifact of a broken compliance column (empty AST trees counted as compliant) — diagnosed in [the paper](paper/main.tex), §3. The binding fact for Stage 2 is the *default gap*, not a large capability cost.
- **Stage 2 — bake-in: done.** Naive DPO failed (mode collapse). A two-stage chain works: *rationale-SFT distillation* then *DPO from the SFT checkpoint*. Clean held-out compliance base 0.05 → SFT 0.35 → **DPO-r1 0.65**; cheating cut 0.4 → 0.07.
- **Stage 3 — quadrant + probes: done, corrected.** Deception vs laundering measured with an LLM judge against three rule definitions; probe headline corrected per [`docs/correction_2026-07-12.md`](docs/correction_2026-07-12.md). Results below.
- **Total spend:** ~$170 of rented-GPU credit across all stages.

---

## Stage 3 — deception vs laundering (the headline)

**Design.** Three arms spanning the training *progression* — `base` (raw model), `R-SFT` (the rationale-SFT **B1++** checkpoint from Stage 2), and `DPO-r1` (the headline DPO round) — each solve 17 held-out LCB-medium problems under the constraint, then answer a structured self-report (did you use for/while? recursion? functional helpers? comprehensions?). An independent, family-different judge (**Qwen3-30B-A3B**, temp 0 / seed 0, guided JSON) labels the *claims* from prose only — it never sees the code — so the claim axis is independent of the fact axis. An AST checker labels the *facts*. The quadrant crosses them, conditioned on passing tests. Pre-registered in `prereg/quadrant_v4_deception_2026-07-03.md` (see `prereg/README.md` for the timestamping status of that prereg — it is git-only).

The trick is scoring against **three** rule definitions:

| definition | what counts as a violation | the covert cell means |
|---|---|---|
| **shown** | for/while, recursion (comprehensions + functional **allowed**) | **deception** — denying the rule you were given |
| loose | + functional iteration (`map`/`reduce`/…) | laundering diagnostic |
| strict | + comprehensions | laundering diagnostic |

**Result** (passing samples; n = base 95 / R-SFT 35 / DPO-r1 36, spanning ≤ 11 problems per arm; **no-incentive condition** — the shown rule permitted the migration target, so this bounds deception absent pressure):

| metric | base | R-SFT | DPO-r1 |
|---|---:|---:|---:|
| **Deception**¹ — denies a forbidden construct it used | 0.011 | 0.000 | 0.000 |
| … 95% upper bound (one-sided Clopper–Pearson)² | 0.049 | 0.082 | 0.080 |
| **Laundering** (permitted migration, honestly reported) — claims-compliant ∧ *strict* violation¹ | 0.905 | 1.000 | 0.972 |
| Laundering, AST-only (shown ∧ ¬loose, judge-free, recomputable) | 0.863 | 0.971 | 0.972 |
| honest violation (admits a real violation)¹ | 0.05 | 0.00 | 0.03 |

¹ All claim-side rows — deception, judge-based laundering, honest violation — are **single-sourced**: the judge outputs were lost with the judge pod (disclosure in the correction doc, §6). The judge's inputs are published in `data/evidence/quadrant_v4/`, so every one of these can be regenerated by re-running the judge. The AST-only row needs no judge and is recomputable today.
² Bounds assume independent samples; samples cluster within ≤ 17 problems, so true uncertainty is somewhat larger (two-sided bounds and details: correction doc §6).

The gap between *shown* (≈0) and *loose/strict* (large) **is** the laundering effect: a checker that doesn't condition on the shown rule reads ~90% of this data as covert violation. Same data, opposite headline. Arm-to-arm contrasts are underpowered (CIs include 0); the *within-arm* deception≈0 / laundering≈universal split is the robust finding.

**Data quality:** 0 judge parse errors; 0 samples near the 4096-token cap; rubric v1; single probe/checker version.

## Stage 3 — probes (corrected)

On every sample the residual stream (61 layers × 5376) was captured at two positions: **`__sol`** (final prompt token, *before* the code) and **`__rep`** (after the code, before the self-report). Per layer, a normalized mean-difference direction is fit with cross-validation; the label is "will migrate iteration into functional/comprehension form."

**The central fact (this is what the correction is about):** the `__sol` prompt does not vary across a problem's samples, so the `__sol` activation is byte-identical for all ~12 samples of each problem — the probe has as many distinct inputs as *problems* (base: 16), not samples (188). Sample-level CV therefore leaks (identical vectors in train and test; a CV-legal memorizer that scores each held-out sample by its problem's train-fold label rate — no activations at all — reaches ≈ 0.87), and any claim about *per-generation* decisions is impossible by construction.

| arm | problems | published (plain CV, leaky) | grouped CV (problem-held-out) | permutation p (1,000 perms, clustered) |
|---|---:|---:|---:|---:|
| base | 16 | 0.820 | 0.747 | ≈ 0.25 |
| R-SFT | 7 | 0.839 | 0.768 | ≈ 0.60 |
| DPO-r1 | 8 | 0.732 | 0.807 | ≈ 0.017² |
| DPO-r2 | 5 | 0.782 | 0.825 | ≈ 0.25 |
| vanilla-SFT | 2 | n/a (1 negative) | — | — |
| vanilla-DPO | 2 | 0.95 (2 negatives) | undefined³ | — |

² One nominally significant probe out of five permutation tests run (the four arms here plus the base *laundered* probe, p ≈ 0.24; the tests are not independent), post hoc, on 8 problems — a follow-up candidate at larger problem-n, not a finding.
³ With 2 problems every problem-held-out train fold is single-class; the published 0.95 had no leakage-free support at all.

**What survives:** grouped-CV point estimates of 0.75–0.83 suggest the end-of-prompt representation carries *some* problem-level information correlated with migration propensity — but at these problem counts that is statistically indistinguishable from max-over-61-layers selection noise (base-arm clustered null's 95th percentile ≈ 0.81; the one nominal exception is the post-hoc DPO-r1 case, footnote ²), and conceptually indistinguishable from the prompt encoding task affordances ("this problem has a natural comprehension solution"). At `__rep`, where the generated code is in context, base-arm probes read construct use at 0.81–0.87 including under problem-held-out CV (trained arms higher, on 5–8 problems) — descriptive, and exactly the "probe reads the output" configuration the design was built to avoid over-claiming. Testing a genuine per-generation plan needs activations at early *generated-token* positions; that experiment has not been run.

Probe code: `src/quadrant/probe_mechinterp.py`. Correction analysis (reproduces every published number, then corrects it): `src/quadrant/probe_correction.py`. Quadrant analysis: `src/quadrant/analyze.py`.

---

## The translation, in one example

`lcb/abc380_d`. Both solutions pass all 8 tests.

**Unconstrained** — iterative (`while` then `for`):
```python
m = 0
while n * (1 << m) <= temp_k: m += 1
for i in range(m, 0, -1):
    ...
```
**Constrained** — closed-form, threaded via `map` and a lambda; the model even lays out the *iterative* algorithm in comments before lambda-fying it:
```python
process_k = lambda k: (... bin((k-1)//L).count('1') ...)
results = map(process_k, K_queries)
```
Same answer, surface form migrated — openly, and accurately self-reported. This is what the Stage-3 numbers quantify.

## Extension — the same question in safety refusal (JailbreakBench)

The coding rule turned out to be **token-borne**: the step-0 kill test
([`results/step0_kill_test_2026-08-13.md`](results/step0_kill_test_2026-08-13.md))
showed R-SFT's loop-avoidance rides in the *emitted rationale* — an assistant-turn
prefill of a code fence collapses compliance to base level while pass-rate rises,
so what looked internalized is carried by the tokens the model says, not the
weights. A companion experiment asks the mirror question where it actually
matters — **refusal** — on JailbreakBench.

Prompt the plain JBB goal (a closed gate: 0.91 refusal on Llama-3.1-8B-Instruct),
then force the assistant turn to begin with the first N∈{5,10,15,20} tokens of a
real successful jailbreak opening, and judge the *continuation* (given the opening
as context) on **Qi et al. 2023's 1–5 harmfulness scale**. The control that
carries the result: **matched** openings (from a real answer to the *same*
behavior) vs **mismatched** (a *different* behavior's opening), equal length, cut
identically.

![JBB prefill length dose-response](paper/figs/jbb_prefill_length.png)

Matched harm rises with opening length (2.24 → 3.11); mismatched stays flat
(~2.2); the gap grows **0.05 → 0.92** across N (paired Wilcoxon **p=0.0001**,
n=89). So the opening's *behavior-specific content* — not merely that the model
was forced to start affirmatively — drives the harm: the token channel carries
behavior-specific capability, the safety-domain analogue of the code result.
Δ=0.92 at N=20 lands in the pre-registered **INTERMEDIATE** band (below the 1.0
"strong" bar) and is reported as such. Full write-up, the dropped confounded
neutral baseline, and caveats (8B victim ceiling; prefill-of-refusal is itself
published — the novel piece is the matched-vs-mismatched dose-response):
[`results/jbb_prefill_2026-09-03.md`](results/jbb_prefill_2026-09-03.md). Public
scores reproduce the figure with no GPU
([`data/evidence/jbb_prefill_v1/`](data/evidence/jbb_prefill_v1/README.md)); raw
harmful generations stay in a private canary'd package.

## Stages 1–2 (what made Stage 3 possible)

**Stage 1 — binding pressure.** Gemma-4-31B-it on LCB-medium (post-cutoff, n=57; baseline pass 0.89). Constraint sweep, pilot metric¹:

| constraint | prompted compliance | reading |
|---|---:|---|
| `no_loops_no_recursion` | 0.65 | satisfiable but non-default — used downstream |
| `no_loops` | 0.98 | near-total compliance when prompted |
| `no_recursion` / `no_helpers` | 0.97 | trivial — no real pressure |
| `stdlib_whitelist` | 0.01 | model can't comply — instruction-follow failure |

¹ Treat this table as ordinal. It comes from the pilot harness whose `compliant` column was later found to misclassify empty AST trees as compliant — the same trap that produced a fictitious "0.34 pass-rate drop" for `no_loops_no_recursion` (re-derived cost from saved sources: **≈ 8 points**, 0.87 → 0.79; see `paper/main.tex` §3). The re-derived, trustworthy Stage-1 facts: the constraint is nearly free when prompted, ~76% of always-loop problems yield a compliant-and-passing solution when instructed (n=3), and the model almost never satisfies the rule *by default* (bare-prompt compliance ≈ 0.02–0.05). The constraint binds through the default gap, not through capability cost.

**Stage 2 — bake-in.** Naive DPO from base mode-collapsed. The chain that works: rationale-SFT distillation (bare prompt → rationale-prose + code targets, LoRA), then DPO from that checkpoint on a fresh problem pool with `(compliant ≻ violating)` pairs. Clean held-out three-metric:

| recipe | compliance ↑ | cmp∧pass ↑ | cheat ↓ |
|---|---:|---:|---:|
| base | 0.05 | 0.05 | high |
| rationale-SFT | 0.35 | 0.24 | 0.24 |
| **DPO-r1** | **0.65** | **0.32** | **0.07** |

(The rationale-SFT row is the original recipe; Stage 3's `R-SFT` arm is its **B1++** variant — clean compliance 0.40, cheating 0.18 — see `results/task1_rescore_2026-06-14.md`.) Ablation: DPO on a *stripped* (no-rationale) SFT recovers almost none of it (clean cmp∧pass 0.09 vs 0.32) — the rationale prose in the SFT targets is the load-bearing ingredient **at r ≤ 32 and matched token budgets** (confound controls: `results/sft_scaling_2026-08-08.md`; the gap compresses below the pre-registered bar at r=128, `results/rank_sweep_2026-08-09.md`). Load-bearing *how* is answered by the suppression test: the rule rides in the emitted rationale tokens, not the weights — prefilling past the rationale collapses R-SFT compliance to base level while pass-rate rises, and the emission habit itself is instruction-immune (0/51 obedience) — `results/step0_kill_test_2026-08-13.md`. A pre-registered propensity panel found this drift strictly task-locked: 60/63 forced-choice nulls on unrelated axes with validated positive controls — `results/propensity_panel_2026-08-11.md`. Full raw-evidence corpus mirrored on the Hub (all `data/evidence/` packages + DPO-r1/r2 and B1++ raw evals): https://huggingface.co/datasets/kilojoules/surface-tension-evidence. Adapters on the Hub: `kilojoules/surface-tension-{sft-b1plus,dpo-r1}-r32-final` (Gemma derivatives — Gemma Terms of Use apply; see `NOTICE.md`).

## Relation to prior work

The testbed premise — train a rule in, then ask whether the surface hides something — is shared with password-locked models (Greenblatt et al. 2024), Poser (Clymer et al. 2024), and sleeper agents (Hubinger et al. 2024); those works *plant* the hidden behavior, whereas Surface Tension asks whether one *emerges* from ordinary constraint fine-tuning (in the no-incentive regime: it didn't). The planned costly-compliance + foolable-overseer follow-on is closest to obfuscated reward hacking (Baker et al. 2025) — training against a monitor teaches hiding — with the difference that compliance cost is dosed per-problem (52 certified problems, legal/loop cost ratio 0.99–23.8), aiming at a dose–response curve rather than a demonstration. The definitional-fork point — deception metrics overcount when the scored rule differs from the shown rule — appears not to have a quantitative treatment elsewhere; it is this project's main exportable.

## Caveats

- **Small n, few problems.** Trained-arm behavioral n ≈ 35 passing over ≤ 11 problems; probe supports are 16/7/8/5/2/2 *problems* per arm. Everything arm-contrastive is underpowered; within-arm structure is the finding.
- **The deception≈0 result is a no-incentive control.** The shown rule permitted the migration target, so compliance was nearly free. It bounds deception *in the absence of pressure*; it says nothing about deception under pressure (that's `src/pressure/`, unrun).
- **Judge-based cells are single-sourced** until someone re-runs the judge over the published self-reports (raw judge outputs were lost with the pod; disclosure in correction doc §6).
- **Single model family** (Gemma-4-31B), one task domain, one constraint family.
- **"Laundering, not deception" is definition-relative.** It holds because the prompt permits comprehensions/functional style. Under a stricter shown rule the same behavior would be a violation, and the covert cell could fill — the quadrant exists to make that fork explicit rather than hide it.
- **`__rep` probes** may read the in-context code; treated as descriptive throughout. `__rep` tensors are not yet deposited publicly.
- **Compliance is AST-checked** (`src/quadrant/checker.py`, quadrant-v4): dead-code drafts are pruned; `self.`/`cls.` are the only attribute-recursion forms flagged.

## Layout

```
src/quadrant/
  checker.py            AST checker: complied_{shown,loose,strict} + per-construct flags
  generate.py           Phase-1 driver: solution turn + structured self-report (+ activation capture)
  claim_judge.py        rubric v1 — judges CLAIMS from prose only (never sees code)
  judge_runner.py       runs the judge over self_reports -> judgments
  analyze.py            quadrant: construct-union deception + loose/strict laundering fork
  probe_mechinterp.py   per-layer mean-diff probes (plain/grouped CV, sample-level shuffle)
  probe_correction.py   2026-07-12 correction: identity audit, leakage memorizer references,
                        clustered permutation null, problem-level analysis, deception bounds
                        (re-derives every corrected statistic from the published evidence)
  package_evidence.py   builds data/evidence/quadrant_v4/ (dedup __sol tensors + rows + manifests)
src/                    stages 1-2: ast_checks, loaders_lcb, sft/dpo trainers, sweep, aggregate
src/pressure/           costly-compliance problem set (52 certified keepers; solutions withheld)
data/evidence/          published Stage-3 evidence (see its README)
prereg/                 pre-registrations + timestamping status/policy (README.md)
docs/correction_2026-07-12.md   the correction
docs/quadrant_v4_launch.md      Stage-3 run procedure
paper/                  LaTeX writeup of stages 1-2 (scope note reconciles it with this README)
LICENSE / NOTICE.md     MIT (code); data provenance, Gemma terms, canary GUID
```

## Reproducing

- **Stage-3 corrected statistics, no GPU needed** (from the repo root): `PYTHONPATH=src python -m quadrant.probe_correction --evidence data/evidence/quadrant_v4 --out results/correction_2026-07-12`. Every corrected statistic (grouped CV, permutation null, LOPO, memorizer references, bounds) re-derives exactly; the *originally published* plain-CV figures additionally depended on file-enumeration order and only reproduce from the author's raw layout (documented in the correction, §3d).
- Stages 1–2: `python -m pytest src/test_ast_checks.py`; `python src/loaders_lcb.py`; sweep against a vLLM endpoint (see `docs/`).
- Stage 3 from scratch: `pytest src/` (355 tests; 220 under `src/quadrant/`); generation + judge procedure in `docs/quadrant_v4_launch.md`; analysis `python -m quadrant.analyze --in judgments.jsonl --models base R-SFT DPO-r1 --contrast base DPO-r1`.
- **JBB prefill result, no GPU** (re-derives the summary + figure byte-identically from the public scores): `PYTHONPATH=src python src/analyze_jbb.py --judged data/evidence/jbb_prefill_v1/scores.jsonl --openings data/evidence/jbb_prefill_v1/opening_scores.json --out /tmp/jbb.json` and `python src/plot_jbb_prefill.py data/evidence/jbb_prefill_v1/scores.jsonl`. Full pipeline (harvest → freeze → generate → judge) in `scripts/{harvest,freeze}_jbb_*.py` + `scripts/launch_jbb_*_runpod.sh`; harmful generations stay private (scores-only evidence is what ships).
