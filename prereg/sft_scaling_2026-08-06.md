# Prereg: SFT token-scaling grid (rationale vs stripped at matched budgets)

**Registered:** 2026-08-06, BEFORE any training or evaluation of these arms.
**External timestamp anchor:**
https://gist.github.com/kilojoules/38fd6e9009940b77bb104de434682223
(public gist, created 2026-08-06 before launch; contains this document's
sha256 `6e1b1ea7…65b7` and the dataset manifest's sha256 `08d0d603…878c`
— note the gist anchors the pre-URL version of this file, so re-hashing the
committed file after this link was inserted will not reproduce the hash;
the gist itself carries the full anchored text.)
**Author:** Julian Quick (quectojoules@gmail.com)

## Question

The README claims the rationale prose in R-SFT targets is "the load-bearing
ingredient." That claim is confounded: rationale targets carry ~25% more
loss-bearing (completion) characters than the stripped targets built from the
same 66 demos (186,429 vs 148,996 chars), and the deployed B1++ variant
carries 2.5–3.5× the tokens of any comparison arm. Does the rationale
advantage survive when completion-token budgets are **matched**? And what is
the data-scaling behaviour of each target type? (This also replaces the
zero-gradient-era data-scaling sweep, which was invalid.)

## Design (fixed before launch)

- **Arms:** 2 target types × 3 completion-char budgets, subsampled from the
  SAME 66 demos / 22 problems in the SAME pre-committed order
  (`scripts/make_sft_scaling_sets.py`, seed 0, problem-stratified round-robin;
  sets + sha256 manifest committed in `data/sft_scaling/`):

  | set | demos | completion chars |
  |---|---:|---:|
  | rationale_b37 / stripped_b37 | 13 / 15 | 38,209 / 38,853 |
  | rationale_b75 / stripped_b75 | 23 / 30 | 75,017 / 77,664 |
  | rationale_b149 / stripped_b149 | 50 / 66 | 150,178 / 148,996 |

  Budgets match within 3% at every level; the arms differ at a matched
  budget only in whether the rationale prose is present (stripped completions
  are byte-subsets of their rationale counterparts, verified 66/66).
- **Recipe:** the original R-SFT recipe, identical for all six runs
  (`scripts/run_sft_scaling.sh`): Gemma-4-31B-it 4-bit, LoRA r=32 α=64
  dropout=0, lr=1e-4, 20 epochs, linear decay, max_length 2048. Fixed-epochs
  design: tokens *seen* scales with unique tokens — that is the quantity
  under study. Deviation from the original run: VAL_EVERY=0 (no best-val
  tracking; `-final` is the deployed convention and val snapshots do not
  alter training).
- **Eval:** clean-17 (OOD; zero problem overlap with the 22 training
  problems), bare prompt, n=8, max_new 3072, T=0.7 — the same protocol and
  **/136 committed convention** as every prior round (generation errors,
  truncations, and no-code outputs count as non-compliant; AST re-checked
  from saved sources via `recheck_eval.py`).
- **Ops:** one pod per budget level (both arms), so each pod yields a
  complete matched contrast; adapters pushed to
  `kilojoules/surface-tension-scaling-*-final` before eval; zero-gradient
  gate on every train log before its eval runs.

## Predictions

1. **(Primary) Matched-budget contrast at the top level:** rationale_b149
   clean-17 compliance exceeds stripped_b149 by **≥ 10 points** (/136).
2. **Scaling:** rationale compliance rises with budget — rationale_b149
   exceeds rationale_b37 by ≥ 8 points.
3. **Sign consistency:** rationale ≥ stripped on compliance at **all three**
   matched budgets.
4. (Secondary, descriptive) rationale_b149 lands near the original R-SFT
   result (compliance 0.43 at 186k chars): within ±10 points of 0.43.

Tracked secondaries, no predictions: cmp∧pass, cheating (reported with a
one-sided 95% Clopper–Pearson bound whenever near zero, never a bare zero),
pass rate, and **no-code rate** (given the iterated-DPO finding).

## Decision rules (pre-committed language)

- If Prediction 1 **fails** (gap < 10 points, or reversed): the README's
  "rationale prose is the load-bearing ingredient" claim is **corrected** to
  "not distinguishable from a token-budget effect at this scale," with the
  same prominence as the original claim. No re-runs, no denominator swaps.
- If Prediction 1 holds but Prediction 3 fails: report as "load-bearing only
  at scale" — the claim narrows, it does not generalize downward.
- If any train-log gate fails (zero-gradient / flat loss): that cell is
  **invalid, not negative** — fix and re-run before any cross-arm reading.
- Results are reported on /136 regardless of direction; the code-emitting-rows
  denominator may be shown alongside but never governs.
- The comparison arm for the original R-SFT number (0.43) is descriptive
  context only; the six grid cells are the confirmatory comparison set.
