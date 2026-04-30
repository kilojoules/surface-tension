# Pilot Study: Binding Coding Constraints

**Goal:** find (benchmark, constraint) pairs with high `binding_score = performance_drop × compliance_rate`.

Model: `gemini-2.5-flash` (spec called for 2.0-flash, but only 2.5-flash is available via the CLI as of generation date).
Raw data: `pilot_v6_dpo_raw.csv` (342 rows).

## Data sufficiency
**1 of 1 pairs are undersampled** (< 30 kept samples in either condition):
- `lcb_medium` × `no_loops_no_recursion`: n_unc=9, n_con=1 — likely from API quota / failures, results below are unreliable

Re-run after quota resets to get coverage on all benchmarks. See `retry_pilot.sh`.

## Success-criteria check
Pilot is informative if some pair has `drop_overall ≥ 0.15` (unbiased binding pressure).

**Cannot evaluate**: no pair has sufficient sample size. Re-run is required.

## Full ranking

Sorted by `drop_overall` (unbiased — counts non-compliant constrained samples as failures).
`drop_among_compliant` is the spec metric, kept for reference but biased upward when compliance < 1.

| benchmark | constraint | n_unc | n_con | n_compl | rej_unc | rej_con | pass_unc | **pass_overall** | **drop_overall** | compliance | pass_compl | drop_compl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lcb_medium | no_loops_no_recursion | 9 | 1 | 1 | 0.9474 | 0.9942 | 0.8889 | **0.0** | **0.8889** | 1.0 | 0.0 | 0.8889 |

## Top 3 examples

### 1. `lcb_medium` × `no_loops_no_recursion` (drop_overall=0.8889)

- pass_unconstrained: **0.8889**, pass_overall: **0.0**
- compliance_rate: **1.0**, drop_among_compliant (biased): 0.8889

_(no compliant constrained sample with matching unconstrained sample available)_