# LoRA rank sweep — results, 2026-08-09

**Pre-registered** before any rank≠32 training:
`prereg/rank_sweep_2026-08-08.md`, externally anchored at
https://gist.github.com/kilojoules/207763a19c1a6a460f7097790dcfeb21.
**Verdict: Prediction 1 (primary) FAILED at r=128; Predictions 2 and 3
held.** Reported per the pre-committed decision rules: the matched-token
"rationale prose is load-bearing" reading is hereby **qualified as
rank-dependent, with r=128 as the failing rank.** Committed convention
throughout: /136, AST re-checked from saved sources. Raw evidence in
`vast_logs/{2zuutp8uolh0iz,3ikana4iiah82i,ore52jv7w4b5iz}/`; adapters
`kilojoules/surface-tension-scaling-*-r{8,128}-final`. Cost ≈ $95 including
a dead-pod recovery (below).

## Full rank grid at the 149k budget (/136; r=32 = parent grid)

| rank | rationale cmp | stripped cmp | gap | rat cmp∧pass | strip cmp∧pass | rat / strip no-code |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.316 | 0.191 | **+12.5** | 0.199 | 0.096 | 0.221 / 0.272 |
| 32 | 0.360 | 0.176 | **+18.4** | 0.213 | 0.044 | 0.162 / 0.213 |
| 128 | 0.324 | 0.235 | **+8.9** | 0.147 | 0.110 | 0.169 / 0.221 |

Binomial SE at these rates ≈ ±3.3–3.7 pts per cell (n=136).

## Prereg accounting

1. **(Primary) Arm gap ≥ 10 pts at both new ranks: FAILED.** Held at r=8
   (+12.5) but not at r=128 (+8.9, 1.1 pts under the pre-committed bar).
   Per the decision rule, the load-bearing-prose claim is qualified as
   rank-dependent. Context the rule requires reporting alongside: the gap
   remains positive at every rank, and the r=128 shortfall is within one
   SE of the threshold — this is a compression, not a reversal.
2. **Rank invariance of the rationale arm: HELD.** 0.316 / 0.360 / 0.324
   across a 16× capacity range — both new ranks within ±4.4 pts of r=32
   (band was ±10). The May NLL-based "capacity is not the bottleneck"
   conclusion replicates on the behavioral endpoint for rationale targets.
3. **Stripped ≤ 0.26 at both ranks: HELD** (0.191, 0.235).

## Reading

**What moved is the stripped arm, not the rationale arm.** Rationale
compliance is flat in rank; stripped compliance rises from 0.176 (r=32) to
0.235 (r=128), a +5.9-pt move (~1.7 SE — suggestive, not decisive), and its
cmp∧pass rises 0.044 → 0.110. Under the capacity reading: prose supplies
something that low-capacity adapters can already encode, while code-only
targets can partially compensate for the missing prose **only when given
substantially more adapter capacity** — capacity partially substitutes for
prose, never fully (the gap stays positive everywhere), and prose never
needs the capacity (r=8 rationale ≈ r=128 rationale).

Honest alternatives the design cannot exclude: at fixed α=2r the effective
update geometry changes with rank, so "capacity" and "optimization scale"
are entangled (disclosed in the prereg); and with one seed per cell, the
stripped-r128 rise rests on a single training run.

**Net effect on the scaling-grid conclusion** (`results/
sft_scaling_2026-08-08.md`): unchanged at the recipe the project actually
uses (r=32, where the gap is +18.4), but the claim now carries the
qualifier "at r ≤ 32"; at r=128 the matched-token gap compresses below the
pre-registered materiality bar. Both documents cross-reference this.

## Ops note (dead pod, contained)

The r8 pod (2zuutp8uolh0iz) died provider-side ~19 h in, mid-way through
the stripped_b149_r8 eval. Loss was limited to ~86 in-flight generations
(~$12): the rationale_b149_r8 cell was fully synced, the stripped adapter
was already on the Hub, and 50/136 eval rows had synced locally. A recovery
pod (ore52jv7w4b5iz, `scripts/launch_r8_recovery_runpod.sh`) resumed the
eval from the seeded partial CSV ("already done 50; pending 86") and
completed it. The Hub-checkpoint + continuous-rsync policy exists for
exactly this failure and worked as designed.
