# SFT token-scaling grid — results, 2026-08-08

**Pre-registered** before any training: `prereg/sft_scaling_2026-08-06.md`,
externally anchored at
https://gist.github.com/kilojoules/38fd6e9009940b77bb104de434682223.
**Verdict: all four predictions held.** The "rationale prose is
load-bearing" claim **survives de-confounding from token count**.
Committed convention throughout: /136 (17 clean problems × n=8;
gen-errors, truncations, and no-code outputs count as non-compliant; AST
re-checked from saved sources via `recheck_eval.py`). Raw evidence synced
in `vast_logs/{bs5rgbob4bwiu1,zga2oxifpys5ra,94eoa6wag8bzjj}/`; adapters:
`kilojoules/surface-tension-scaling-{rationale,stripped}-{b37,b75,b149}-final`.
Cost: 3 pods ≈ $105 total (~23–25 h each).

## Design recap

2 target types × 3 matched completion-char budgets, subsampled from the
SAME 66 demos / 22 problems in the same pre-committed order
(`data/sft_scaling/manifest.json`; budgets match within 3% at every level;
stripped completions are byte-subsets of their rationale counterparts).
Original R-SFT recipe fixed across all cells (r=32, α=64, lr 1e-4,
20 epochs, linear decay). Loss is completion-only, so budgets are stated in
loss-bearing chars. Eval: clean-17 (zero problem overlap with training),
bare prompt, n=8, max_new 3072, T=0.7. All six trainings passed the
zero-gradient/flat-loss gate (loss 0.55–0.61 → 0.005–0.03).

## Headline grid (/136)

| budget (chars) | arm | demos | compliance | cmp∧pass | pass | no-code |
|---|---|---:|---:|---:|---:|---:|
| ~38k | rationale | 13 | **0.221** | 0.096 | 0.441 | 0.221 |
| ~38k | stripped | 15 | **0.066** | 0.007 | 0.441 | 0.228 |
| ~76k | rationale | 23 | **0.257** | 0.191 | 0.485 | 0.206 |
| ~76k | stripped | 30 | **0.132** | 0.051 | 0.493 | 0.191 |
| ~149k | rationale | 50 | **0.360** | 0.213 | 0.419 | 0.162 |
| ~149k | stripped | 66 | **0.176** | 0.044 | 0.412 | 0.213 |

Reference points, same convention: base model compliance 0.015
(2/136, pass 0.544); original R-SFT (66 demos, 186k chars) 0.346.

## Prereg accounting — 4/4 held

1. **(Primary) Matched-budget contrast at 149k: rationale − stripped =
   +18.4 pts** (0.360 vs 0.176), ≥ the pre-committed 10. **HELD.**
2. **Rationale scaling: +13.9 pts** b37→b149 (0.221 → 0.360), ≥ 8. **HELD.**
3. **Sign consistency: 3/3 budgets** — gaps +15.5 / +12.5 / +18.4, all in
   the same direction (and all ≥ 10, though only b149 was predicted at that
   size). **HELD.**
4. rationale_b149 (0.360) within ±10 pts of the original R-SFT — holds
   against the prereg's written reference (0.43, from the era's reporting)
   and even more tightly against the same-convention /136 recomputation of
   that run (0.346, gap +1.4 pts). **HELD.**

## Reading

**The rationale advantage is not a token-budget effect.** At every matched
loss-bearing-token budget, rationale targets beat stripped targets by
12–18 pts compliance while pass rates are statistically indistinguishable
(0.41–0.49 both arms, no consistent direction). The prior confound —
rationale completions carry ~25% more chars — cannot explain a gap that
persists when the budgets are equalized, is largest at the top budget, and
appears at the smallest budget (where 38k chars of code-only targets leave
compliance at 0.066, barely above the base model's 0.015).

**Both curves do scale** (rationale 0.221→0.360, stripped 0.066→0.176), so
tokens matter too — but the prose contributes a roughly constant multiple
(~2–3× the stripped compliance at every budget) rather than a head start
that more code-only data could close. Extrapolating the stripped curve's
~+5.5 pts per doubling, code-only targets would need several more doublings
of data (which does not exist in this corpus) to reach the rationale arm's
current level.

**cmp∧pass separates even harder**: the rationale arm converts compliance
into compliant-and-passing at 4–5× the stripped rate at every budget
(0.213 vs 0.044 at the top). Stripped models that comply rarely also pass —
consistent with the prose teaching *how to satisfy the constraint while
solving the problem*, not just raising a compliance flag.

**No-code rates are flat across arms and budgets** (0.16–0.23 everywhere)
— unlike the iterated-DPO rounds, SFT does not tax code-emission; the
no-code mass here looks like a property of the base+small-SFT regime, not
of the target type.

Ops note: stripped-arm evals ran ~40–70% slower per task (longer
generations toward the 3072 cap) — a behavioral difference in its own
right, disclosed here since it shows up as wall-clock, not in any metric.

## Follow-up: rank sweep (completed 2026-08-09)

`results/rank_sweep_2026-08-09.md`: the rationale arm is rank-invariant
(0.316/0.360/0.324 at r=8/32/128) and the gap holds at r=8 (+12.5), but the
matched-token gap **compresses to +8.9 at r=128** (stripped rises to
0.235), failing that prereg's ≥10-pt bar. The headline above therefore
carries the qualifier **"at r ≤ 32"** — the recipe this project actually
uses — and the load-bearing-prose reading is rank-dependent at high
adapter capacity.
