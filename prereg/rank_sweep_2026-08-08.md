# Prereg amendment: LoRA-rank sweep on the matched-token contrast

**Registered:** 2026-08-08, BEFORE any rank≠32 training or evaluation.
**Parent prereg:** `prereg/sft_scaling_2026-08-06.md` (gist-anchored 2026-08-06).
**External timestamp anchor:**
https://gist.github.com/kilojoules/207763a19c1a6a460f7097790dcfeb21
(public gist, created 2026-08-08 before launch; anchors the pre-URL version
of this file, sha256 `d68151c1…5954` — the gist carries the full anchored
text.)

## Disclosure of what is already known at registration time

The parent grid's rationale arm is fully evaluated (compliance /136:
b37 0.221, b75 0.257, b149 0.360 — parent predictions 2 and 4 hold). The
three stripped-arm evals are **still running**; no stripped number is known.
Nothing in this amendment is conditioned on them beyond reusing the parent's
pre-committed 0.26 threshold, which was fixed before any data.

## Question

Is the rationale-vs-stripped gap at matched token budgets an artifact of
adapter capacity at r=32? And does the May-2026 NLL-based finding ("more
rank → worse val-min; capacity is not the bottleneck") replicate on the
behavioral endpoint?

## Design (fixed before launch)

- **4 new cells**: {rationale_b149, stripped_b149} × LoRA rank {8, 128},
  giving a 3-point rank curve {8, 32, 128} per arm at the top budget (the
  r=32 points are the parent grid's b149 cells).
- **Identical to the parent in every other respect**: same committed
  datasets (sha-gated against `data/sft_scaling/manifest.json`), same recipe
  (lr 1e-4, 20 epochs, linear decay, max_length 2048, dropout 0), same
  clean-17 bare-prompt /136 eval, same zero-gradient gates.
- **α = 2·rank** (α=16 at r=8, α=256 at r=128), holding the parent's
  r=32/α=64 ratio, so the swept quantity is capacity at fixed α/r. Caveat
  stated up front: rank and effective step geometry are not perfectly
  separable in LoRA even at fixed α/r; this sweep reads as "capacity under
  the standard scaling convention."
- Ops: one pod per rank (both arms), MAX_HOURS=54 (2× the ~27 h best case
  now that stripped-arm evals are measured slower), adapters pushed to
  `kilojoules/surface-tension-scaling-*-r{8,128}-final` before eval.

## Predictions

1. **(Primary) Arm effect is capacity-robust:** rationale exceeds stripped
   on compliance (/136) by ≥ 10 points at **both** r=8 and r=128.
2. **Rank invariance of the rationale arm:** rationale_b149 compliance at
   r=8 and at r=128 each within ±10 points of the r=32 value (0.360) — the
   behavioral replication of "capacity is not the bottleneck."
3. Stripped_b149 compliance ≤ 0.26 (the parent primary's threshold) at both
   new ranks.

Tracked secondaries, no predictions: cmp∧pass, pass, cheating
(Clopper–Pearson-bounded near zero), no-code rate per cell.

## Decision rules (pre-committed language)

- If Prediction 1 fails at either rank, the matched-token "rationale prose
  is load-bearing" reading (whatever the parent primary concludes) must be
  qualified as **rank-dependent**, with the failing rank named.
- If Prediction 2 fails in the upward direction (r=128 ≥ 10 points above
  r=32), the May NLL-based rank conclusion does not transfer to the
  behavioral endpoint and gets corrected where cited.
- Zero-gradient/flat-loss gate failure → the cell is invalid, not negative;
  fix and re-run before any cross-cell reading.
- All cells reported /136 regardless of direction; code-emitting-rows
  denominators may be shown alongside but never govern.
