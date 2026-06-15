# SFT training corpus narration measurement

**Stamp:** 2026-06-15 • **Scope:** position-agnostic `asserts_compliance` rate per RUBRIC.md (updated 2026-06-15), applied by agent to a stratified hand-labeled sample of 20 entries from each of three SFT training corpora.

**Population label:** these are **base-model generations post-rejection-filter**, not what the trained arms produce at inference. Report them only as "what the model had available to learn from," never blended with trained-arm rates.

## TL;DR — Origin assignment per rhetorical position

The three-outcome interpretation, broken out **by position** (not collapsed):

| position | sft_b1plus (R-SFT training data) | sft_rationale_stripped (vanilla SFT training data) | sft_full (broader pool) | reading |
|---|---:|---:|---:|---|
| **preamble** | 100% | 0% | 0% | **Origin 1 at preamble**: R-SFT's preamble narration is inherited from training data. Vanilla SFT had no preamble narration to learn from. |
| **inline_comment** | ~80% | 40% | 15% | **Mixed**: both arms' in-code narration is partially inherited. The "stripping" in `rationale_stripped` only stripped the preamble; in-code rationale survived at meaningful rate. |
| **post_code** | 0% | 0% | 0% | **Origin 2 at post-code**: the smoke test showed vanilla SFT producing post-code self-correcting narration that is **absent from every training corpus**. That's an emergent disposition, not mimicry. |

**The single most important finding**: `sft_rationale_stripped` is mis-named. It strips preamble prose but **retains substantial in-code compliance narration** (40% of entries make explicit `asserts_compliance` claims in `#`-comments per the rubric). Vanilla SFT was therefore trained on in-code narration; it isn't a "no-narration" control as the name implied.

## Sample-based rates with Wilson 95% CIs

Stratified random sample (seed `14062026`), n=20 per corpus, hand-labeled by the agent applying RUBRIC.md (updated 2026-06-15). The corpora are 100% R0-compliant by construction (the `build_rationale_dataset.py` filter rejects R0-violators before writing), so every YES below is `asserts_compliance ∧ ¬violates` = `rationale_honest` — the honest paired baseline.

| corpus | n | preamble YES | inline YES | post YES | any-position YES |
|---|---:|---:|---:|---:|---:|
| **sft_full** (data/sft_full.jsonl, 889 total) | 20 | 0/20 = **0%** [0–16%] | 3/20 = **15%** [5–36%] | 0/20 = **0%** | 3/20 = 15% |
| **sft_b1plus** (R-SFT training; 240 total) | 20 | 20/20 = **100%** [84–100%] | ~16/20 ≈ **80%** [58–92%] | 0/20 = **0%** | 20/20 = 100% |
| **stripped** (vanilla SFT training; 102 total) | 20 | 0/20 = **0%** [0–16%] | 8/20 = **40%** [22–61%] | 0/20 = **0%** | 8/20 = 40% |

Wilson CIs are conservative at n=20; the rates above are best-read as ranges. Heuristic-based rates over the full corpora (presented in the earlier turn) sit inside these CIs — the hand-labels confirm the heuristic is not badly off.

## How this reframes the §D verdict for the full run

The original three §D outcomes from the brief were stated as if the three positions could be collapsed. They cannot. Position-stratified outcomes:

### At the preamble position (the "purest" Spine B test)

| if R-SFT preamble-rate is | and vanilla SFT preamble-rate is | reading |
|---|---|---|
| **~100%** | **~0%** | Confirms the clean preamble-position contrast. R-SFT inherits preamble narration from training; vanilla SFT had none to learn. The smoke (R-SFT preamble=596 chars, vanilla SFT preamble=0) points strongly to this outcome. |
| ~100% | ~50%+ | Vanilla SFT generalised to preamble narration despite not being trained on it. Disposition emerging mid-training. Sharper Spine B than the brief named. |
| <50% | <50% | Both arms forgot the training-time rate at inference. Unlikely given the smoke. |

### At the inline_comment position (the noisy-shared layer)

Both training corpora carry in-code narration. Expect both arms to assert compliance in `#`-comments at meaningful rates at inference. The cross-arm gap here is **NOT** clean — it's "more vs less" of the same shared inheritance, not "narrates vs doesn't." Reporting it as a binary contrast would mis-name the finding.

### At the post_code position (the smoke's emergent finding)

Zero training-corpus rate everywhere. If vanilla SFT produces post-code self-correcting narration in the full run at meaningful rate (smoke showed ~7,100 post-code chars on n=1), that is an **emergent disposition** — Origin 2 in clean form, and the headline finding worth the spend.

Substantively: post-code self-correction is a different cognitive event from preamble assertion. The model writes code, looks at it, narrates a self-correction. That bucket (`self_correcting_narration`, added to RUBRIC.md today) is **closer to honest struggle than to concealment**. If R-SFT shows preamble decoupling (assert → violate, no later correction) and vanilla SFT shows post-code self-correction (write → notice → fix), the deck's clean dichotomy actually becomes:

> *"Rationalization training shifts compliance narration from post-hoc self-correction into pre-hoc assertion. The post-hoc narration corrects itself; the pre-hoc one decouples."*

That's a much more interesting paper than "rationalization causes decoupling" — and the corpus measurement tells us we shouldn't pre-claim it; we should measure it.

## Hard guardrails honored

| guardrail | status |
|---|---|
| "Label as a distinct population. Base-model-under-filtering, never blended with trained-arm rates." | ✓ — this report uses "training corpus" language throughout; the rates above are NOT the rates the full run will measure on trained-arm inference outputs |
| "Same rubric, same R0, no looser definition for the corpus." | ✓ — RUBRIC.md as updated today applied uniformly. Tarjan-corrected R0 is moot here because all corpus entries are R0-compliant by construction (filter side already done) |
| "Report by rhetorical position, never collapsed." | ✓ — table above is position-stratified |

## What this changes about the full run

**Still launch.** The corpus measurement tells us **what to expect** at inference, not **what we get** at inference. The full run is now framed differently:

- The R-SFT preamble assertion rate at inference is the most important number. If it's near 100% (matching training-corpus rate), R-SFT carries preamble narration into inference. If it drops below corpus rate, R-SFT generalises away from it.
- The vanilla SFT preamble assertion rate at inference is the **causal control**. Smoke showed n=1 of 0%. If the full 136-sample run also shows ~0%, the position-specific Spine B holds at preamble.
- The vanilla SFT POST-CODE rate at inference is **the emergent-disposition test**. Smoke showed n=1 of substantial post-code prose. If this holds at scale, the deck story sharpens to the position-shift framing above.
- The in-code rate cross-arm comparison is **a covariate, not a headline contrast**. Both arms inherit it. Reporting it without that context would mis-name the finding.

## What this changes about the deck

The deck claim cannot be "rationalization causes the assertion surface." That collapses position and gives R-SFT credit for narration that's actually inherited from its training corpus (which IS what we constructed it from — that's expected, not a finding).

The deck claim can be **one of**:

1. **Preamble narration is acquired only with rationalization training**, even though both arms see compliance reasoning in their training data — the position is what's learned, not the practice. Plus: the assertion-violation decoupling lives more easily in preamble form than in post-code form.
2. **Post-code self-correcting narration is an emergent property of SFT-without-rationalization-scaffolding** — vanilla SFT learns to debate with itself in a way the corpus never showed it. R-SFT, because the preamble is the "answer," doesn't go back.
3. **Both** — position-specific inheritance + emergent post-hoc narration in the non-preamble arm — which is the sharpest read and is what the data is pointing toward.

None of these is dead; all three are testable in the full run with the position-stratified measurement.

## Tier reaffirmed

H100 still the right call (smoke confirmed availability and speed). Cost estimate unchanged from earlier: ~10 hr / ~$27–35.

## Files committed

- `paper/data/rationale_label/RUBRIC.md` (updated): §A.2 now position-agnostic; new `self_correcting_narration` bucket; required position-stratified output shape.
- `paper/data/rationale_label/corpus_narration_measurement.md` (this file).
