# Rubric-reliability stress-test — plan and framing

**Stamp:** 2026-06-14 • **Status:** DRAFT — awaits owner approval before the sampler runs.

This document fixes the framing error in the previous turn: rubric §3 ("verbose
in-code comments count as the prose surface") is a TRIAGE fallback, not a
license to promote comments to the headline measurement. The original brief
§A.4 governs — a `#`-comment is demoted to secondary because comments are a
**different cognitive signal** from a prose preamble, and the concealment-vs-
confabulation question lives precisely in that difference. This plan therefore
splits the work into two pieces that buy two different things, and only one of
them needs the $18.

## What buys what

| step | corpus / surface | what it measures | what it does NOT measure |
|---|---|---|---|
| **stress-test (free, now)** | 136 post-training R-SFT clean-17 `.py` files, comment-form prose surface | rubric *reliability* under the messy fallback surface — can two readers using `RUBRIC.md` agree on `asserts_compliance` when they have only comments to look at? | (1) whether the category is the right category — agreement on a blurred surface does not validate the category itself; (2) the positive-class count for the probe gate |
| **C2 gate ($18, conditional)** | regenerated R-SFT clean-17 with raw saved → preamble-form prose | the actual probe gate: positive-class count of `rationale_violation` on the canonical preamble surface; rubric reliability on that same surface | nothing — this is the headline measurement the brief is asking for |

The stress-test is **diagnostic only**:

- **Low κ kills the rubric.** If two readers can't agree on `asserts_compliance`
  even when looking at messy in-code prose, no amount of clean preamble will
  save the category — the rubric must be tightened before any spend.
- **High κ is necessary, NOT sufficient.** Agreement on a blurred category
  does not validate the category. **We hypothesise** that comments and
  preambles are different signals — rhetorically, a preamble is a claim
  about the whole solution while an inline comment is a local annotation
  — but the temporal framing earlier in this document overstated that.
  Generation is left-to-right, so a comment above a function is emitted
  *before* the function's tokens; "post-hoc" is a rhetorical claim, not a
  generation-order one. Whether the rhetorical difference is also a
  representational one is exactly the open question. The two-κ comparison
  at C2 is one empirical test of it (see below); the C2 spend is therefore
  still required, but its justification is "the hypothesis is unproven and
  the comparison tests it," not "comments are obviously a different
  cognitive event."

## Why we can't get the gate result for free

`build_rationale_dataset.py:127` rejects R0-non-compliant samples before
writing — so every prose-bearing corpus on disk (`sft_full.jsonl`, `sft_b1plus_*`,
`sft_rationale_*`) is 100% R0-compliant by construction. There is **no on-disk
corpus** that contains both preamble-form prose AND R0-violators, because the
pipeline was built specifically to exclude that combination from training
data. The only place violator-prose has ever existed is the raw post-training
R-SFT response that `_save_source` discarded.

The save_raw patch (`src/raw_response_save.py`, dry-run 7/7 green; spec in
`SAVE_RAW_PATCH.md`) closes that hole prospectively. The C2 regenerate spends
$18 to produce one cohort of post-training R-SFT outputs *with raw saved*,
which is the only artifact that contains the phenomenon in its native form.

## Stress-test source corpus

`vast_logs/rdqb499a37k1tv/st/results/raw/sources_eval_b1plus_clean/`:

- 136 `.py` files — the post-training R-SFT clean-17 generations
- R0 split verified: **54 compliers (39.7%) / 82 violators (60.3%)** —
  matches the Task 1 number exactly
- All files parse; zero parse-failures
- Surface: in-code comments only (no preamble — eval pipeline stripped it).
  Most contain multi-sentence rationale prose embedded as `# ...` blocks
  (item 3 of `RUBRIC.md §"Where to look"`).

## Stratification

Per the owner's framing — **concentrate the measurement on the
judgment-heavy cell**, not "pressure the harder side." Easy compliers
("implemented without loops" + no loop = trivial YES) inflate κ; the
hard, paper-relevant calls (is this prose a claim or strategy musing?)
cluster in the violator set, which is also where the eventual positive
class lives:

- 30 violators (drawn from the 69-file prose-bearing violator pool)
- 20 compliers (drawn from the 34-file prose-bearing complier pool)
- Total: 50

**Prose-bearing filter** (applied to both strata before drawing): a
file is eligible iff its `#`-comments contain ≥ 2 rule-related lines
OR a single rule-related comment of ≥ 2 sentences. Bare-`# no loops`
or no-rule-comment files give the labeler no surface to apply the
rubric to and are excluded.

Cheap check on 2026-06-14: 69 of 82 violators and 34 of 54 compliers
clear the filter. Draw is well under each pool, no risk of running on
fumes. If either count had been thin, that would itself have been
soft evidence that C2 is the only viable surface — but both clear.

## R0 hidden at label time

Both label files contain only `(response_id, problem_id, code,
purpose)`. The R0 verdict is **stripped from the queue entirely** so
neither reader sees it while judging `asserts_compliance`. The scorer
joins R0 after both label files are produced. The rubric instructs
labelers to decide `asserts_compliance` before checking R0; this
removes the temptation to peek.

## Labeling protocol

1. Sampler emits `stress_test_queue_50.jsonl` (with R0 visible to the agent)
   and `stress_test_owner_blind_50.jsonl` (with R0 hidden, shuffled with
   `seed+1`). Both files carry an `EXPLICIT_PURPOSE` string in every row that
   says "rubric-reliability stress-test, not the probe gate" so neither
   labeler accidentally writes a probe-readiness conclusion.
2. Agent reads `stress_test_queue_50.jsonl`, applies `RUBRIC.md` to the
   comment-form prose, emits `agent_labels_50_stress_test.jsonl` with
   `(response_id, asserts_compliance, label_location, justification)`.
3. Owner reads `stress_test_owner_blind_50.jsonl` independently, emits
   `owner_labels_50_stress_test.jsonl` with the same schema.
4. Existing `src/score_rationale_audit.py` computes κ. The output markdown
   `stress_test_kappa.md` carries a header reminding the reader that the
   result is diagnostic, not the gate.

## What the agent will NOT do during labeling

- Will not look at the R0 verdict before deciding `asserts_compliance` on a
  given row (per the rubric: label assertion first, deterministically, before
  joining with the AST verdict).
- Will not extend the rubric. Applies `RUBRIC.md` as written, including the
  rule that a bare `# no loops` comment counts as `bare_comment_only`
  (secondary) and not as `asserts_compliance`.
- Will not interpret comment-form prose more generously than preamble-form
  prose; if anything will lean toward NO when the surface is ambiguous,
  because the rubric's preference order is preamble > top-of-code comment
  block > inline comments.

## C2 two-κ comparison (when C2 runs)

Per the owner's extension to Q3: the stress-test κ does NOT feed C2's
go/no-go decision, but the pair of κ's IS interpretable together.
After C2 runs and produces a preamble-surface κ, compare:

- `kappa_comment` (this run) vs `kappa_preamble` (C2 run)
- If both high and close: rubric is reliable across surfaces.
- If both low: the rubric itself is broken (already caught by the
  stress-test in the kill-switch).
- **If they diverge — high preamble-κ, low comment-κ, or vice
  versa — that is itself empirical evidence that comments and
  preambles are different signals**, rather than a stipulated
  difference. Report as a finding, not as noise.

The comparison is reported in C2's `kappa.md`, not in this stress-test's
output (it doesn't exist yet at this point). This stress-test's deliverable
is purely the comment-surface κ + the kill-switch verdict.

## What stays unchanged from the previous turn

- `src/raw_response_save.py` — helper module, 7/7 dry-run tests green.
- `src/test_save_raw.py` — the dry-run tests.
- `paper/data/rationale_label/RUBRIC.md` — the rubric. (§3 not promoted;
  comments remain a triage fallback only.)
- `paper/data/rationale_label/SAVE_RAW_PATCH.md` — the patch spec for
  `sweep.py` / `sweep_local.py`. Independent of the κ stress-test; blocks
  the C2 regenerate.
- `src/score_rationale_audit.py` — the κ tool. Outputs a κ + 2×2 +
  disagreement list regardless of which corpus the labels came from.

## What gets discarded

- `src/sample_rationale_audit.py` (B1++ filtered-pool sampler) — deleted.
- `paper/data/rationale_label/KAPPA_AUDIT_PLAN.md` (B1++ plan) — deleted.
