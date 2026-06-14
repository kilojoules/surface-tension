# Two free checks before any C2 / regenerate spend

**Stamp:** 2026-06-14 • **Scope:** Task 5 sequencing. Gates the multi-arm-baseline-vs-R-SFT-only spend decision.

## Check 1 — Does the laundering gap survive the Tarjan recursion fix?

Same 5 arms as Task 1/2, matched cohort n = 110. Two loose checks side-by-side: `ast_checks.check_no_loops_no_recursion` (the over-aggressive helper-chain-counts-as-recursion version used in Task 1 and to label `violates_r0` in the agent's 50-sample) vs Tarjan SCC cycle detection (used at R0 in `src/strict_ladder.py`).

| arm | n | ast_checks R0 (over-flag) | Tarjan R0 (cycles only) | R4 strict-max | Gap R0 → R4 |
|---|---:|---:|---:|---:|---:|
| base | 110 | 1.8% | 3.6% | 0.0% | **+3.6 pp** |
| R-SFT (B1++) | 110 | 46.4% | 69.1% | 0.0% | **+69.1 pp** |
| vanilla DPO | 110 | 25.5% | 52.7% | 0.0% | **+52.7 pp** |
| DPO-r1 | 110 | 71.8% | 75.5% | 0.9% | **+74.5 pp** |
| DPO-r2 | 110 | 60.9% | 64.5% | 0.0% | **+64.5 pp** |

The laundering gap **survives** under Tarjan and **widens** for R-SFT (+23 pp) and vanilla DPO (+27 pp). The widening comes from the helper-chain over-flag suppressing the loose number under `ast_checks`. The Task 2 finding ("DPO crushes literal loops but the violations migrate into functional helpers/comprehensions, leaving the strict-judge gap intact") is stronger under the corrected checker, not weaker.

DPO-r1 and DPO-r2 have small shifts (+3.7 / +3.6 pp) — they use fewer helper functions than R-SFT and vanilla DPO. Task 2's "DPO is functional-helper-heavy" was already visible; this just confirms it on the loose-judge side under Tarjan.

**Reading for the spend question:** the gap exists and is large under any sensible checker. The headline is intact. The question is no longer "does the phenomenon exist" — it's "does R-SFT do it more than vanilla SFT," which is the missing control.

## Check 2 — Did any arm preserve preamble-form prose?

Same 5 arms (each evaluated on clean-17 with `condition=unconstrained`):

| arm | n .py | n `*__raw.txt` | .py starts with | preamble retained? |
|---|---:|---:|---|:---:|
| base | 115 | 0 | `import sys` (code) | **No** |
| R-SFT (B1++) | 136 | 0 | `import sys` (code) | **No** |
| vanilla DPO | 136 | 0 | `import sys` (code) | **No** |
| DPO-r1 | 136 | 0 | `import sys` (code) | **No** |
| DPO-r2 | 136 | 0 | `import sys` (code) | **No** |

**0 / 5 arms** preserved the prose preamble. `_save_source` discarded raw on every success branch across every run; the bug is uniform. The `save_raw` patch is now in place (committed at `74a8d67`, dry-run verified), but it does not retroactively recover what was discarded.

## Check 2b — Where is vanilla SFT on clean-17?

Hunted across all `vast_logs` for `sources_eval_*stripped_sft*clean*` and `sources_eval_*sft*clean*`. **It does not exist.** Stripped-SFT was evaluated on the val set (`vast_logs/308960zaa2vqut/.../sources_eval_stripped_sft_val`, `vast_logs/jy8taoitdmkzhg/.../sources_eval_stripped_val`, etc.) but never on clean-17. The Task 1/2 cohort has no vanilla-SFT-only data point; the curve jumps base → vanilla-DPO directly, treating vanilla DPO as if it stood in for the SFT-only adapter.

This is the gap your message named: the causal claim "rationalization-augmented SFT (R-SFT) introduces the decoupling" needs vanilla SFT as the control, and the existing data has neither prose nor a clean-17 SFT-only data point to anchor it.

## What we can and can't conclude from existing data

| question | can answer from existing data? |
|---|---|
| Does R-SFT show loose-but-not-strict laundering? | **Yes** — Task 2, confirmed under Tarjan today (+69 pp gap). |
| What is R-SFT's `asserts_compliance` rate on the comment surface? | **Yes** — Task 5 turn-N: 47/50 = 94% (with the borderlines noted). |
| What is R-SFT's `rationale_violation` rate on the conservative R0? | **Yes** — Task 5 today: 4/50 = 8% genuine decoupling. |
| Is the laundering gap caused by R-SFT specifically, vs vanilla SFT? | **No** — no vanilla SFT eval on clean-17 exists. |
| Is `asserts_compliance` rate higher in R-SFT than vanilla SFT or base? | **No** — no prose for any other arm. Cross-arm assertion-rate comparison is structurally unavailable. |
| Does the comment-surface vs preamble-surface gap differ across arms? | **No** — no preamble surface anywhere. |

**The causal claim "rationalization manufactures the surface on which decoupling occurs" is unsupported from any data we currently have.** Not weakly unsupported — *structurally* unsupported, because the prose surface needed to measure assertion rate on non-R-SFT arms does not exist on disk.

## Implication for spend scope

Reframing the C2 question per your message:

**Previous framing:** "$18 to regenerate R-SFT with raw saved to clear the probe positive-class floor."
**Reframed:** "Multi-arm baseline study — base + vanilla-SFT + R-SFT (+ optionally DPO arms) on clean-17 with raw saved, Tarjan applied from the start. The vanilla-SFT and base arms are the causal-claim control; the R-SFT arm clears the probe-readiness floor as a side effect."

The vanilla-SFT-on-clean-17 data point doesn't exist *at all* — it's not a "regenerate to get raw" but "run for the first time, with raw saved." That's load-bearing for the causal claim.

Pre-registered prediction per your suggestion: **vanilla SFT will assert compliance at a much lower rate than R-SFT** (it wasn't trained to narrate). If that's true, the finding sharpens from "decoupling rate" (which would need many positives in vanilla SFT to compare) to "rationalization manufactures the assertion surface on which decoupling can occur" — cleaner and more interesting than a rate delta. Report both rates either way.

## Three open questions for the owner (no spend until decided)

1. **Scope of the multi-arm baseline.** Minimum sufficient is base + vanilla SFT + R-SFT on clean-17 with raw saved. Maximum is all 5 (current) arms + vanilla SFT = 6 arms.
2. **Cost.** Conservative estimate at clean-17 (17 problems × 8 samples = 136 generations per arm) × ~100–200 s/sample × ~$3/hr ≈ $5–15 per arm. 6 arms = $30–90. Roughly 2–4× the original $18-for-R-SFT-only budget, but unlocks the causal claim instead of just clearing the probe floor.
3. **Should the loose/strict numbers in Task 1/2 / the paper switch from `ast_checks` to Tarjan retroactively?** The widening means the headline gets stronger, not weaker — but it's a methodology change touching every published number. Surfacing as a separate decision.

## Sequencing per your plan

- [x] Tarjan re-run of Task 1/2 — **done, gap survives, widens**.
- [x] Per-arm prose retention table — **done, 0/5 retained; vanilla-SFT clean-17 missing**.
- [ ] Scope call: qualitative existence proof on the 4 R-SFT decoupling cases vs full base → SFT → R-SFT → DPO curve. **Awaiting your decision** with both check results in hand.

No regenerate launched. No probe. No labels flipped. Two near-misses (`abc362_c/s6`, `abc370_d/s2`) still flagged for your call.
