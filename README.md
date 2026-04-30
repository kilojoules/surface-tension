# Surface Tension

**Detecting hidden reasoning in constraint-trained code models.**

Models trained under stylistic constraints (no recursion, no OOP, etc.) might satisfy those constraints at the surface while internally reasoning in the forbidden style and translating at output. If this happens, chain-of-thought monitoring breaks down: the stated reasoning doesn't match the actual computation.

This repo is **stage 1** of the research arc in [`PLAN.md`](PLAN.md): empirically establish whether *binding pressure* — a measurable performance gap when a constraint is imposed — exists at all, on the model we'd actually fine-tune. Without binding pressure, there's nothing to translate, and the rest of the program (mech-interp probes, an aligner agent, an adversarial trainer) has no signal to detect.

## Headline finding

On **Gemma 4 31B-it** evaluated against **LiveCodeBench medium** (post-training-cutoff problems only, n=57):

| constraint                  | drop_overall | compliance | reading                                          |
|-----------------------------|-------------:|-----------:|--------------------------------------------------|
| `no_loops_no_recursion`     |     **0.34** |       0.65 | **strong, satisfiable binding** — best for stage 1 |
| `no_loops`                  |     **0.14** |       0.98 | clean ~14pp drop, near-total compliance          |
| `no_helpers`                |         0.03 |       0.99 | trivial — no real pressure                       |
| `no_recursion`              |        −0.01 |       0.97 | trivial — model rarely recurses unconstrained anyway |
| `stdlib_whitelist`          |         0.89 |       0.01 | model can't comply at all (instruction-follow failure, not binding) |

`pass_unconstrained = 0.89`. The combined `no_loops_no_recursion` constraint clears the 0.15 spec threshold by 2× and is *the* artifact stage 2 would study: same model, same problem, two surface forms, both correct.

### The translation, in one example

`lcb/abc380_d` (a Thue-Morse-style query problem). The unconstrained Gemma generation traces the recurrence iteratively; the `no_loops_no_recursion` Gemma generation finds the closed-form `bin((k-1) // L).count('1')` and threads queries via `map(...)` and a lambda IIFE. Both pass all 8 tests.

[`results/examples/abc380_d_unconstrained.py`](results/examples/abc380_d_unconstrained.py) (uses a `while` to find the level, then a `for` to count flips):
```python
m = 0
while n * (1 << m) <= temp_k:
    m += 1
for i in range(m, 0, -1):
    half_len = n * (1 << (i - 1))
    if temp_k >= half_len:
        num_flips += 1
        temp_k -= half_len
```

[`results/examples/abc380_d_no_loops_no_recursion.py`](results/examples/abc380_d_no_loops_no_recursion.py) (closed-form, declarative iteration via `map`):
```python
process_k = lambda k: (
    lambda x, idx: (
        S_orig[idx] if bin(x).count('1') % 2 == 0 else
        (S_orig[idx].upper() if S_orig[idx].islower() else S_orig[idx].lower())
    )
)((k - 1) // L, (k - 1) % L)
results = map(process_k, K_queries)
```

The constrained version actually contains comments stepping through the iterative algorithm before lambda-fying — behavioral evidence that the model's working representation is iterative even when the output isn't. This is exactly the surface/internal mismatch the program hypothesizes.

40 of 57 LCB problems produced the "constrained-compliant AND passed" pattern, so there is plenty of paired data for downstream mech-interp work.

## What's a "binding score"?

Two metrics are reported. Only one is interpretable.

**`drop_overall`** (unbiased, headline): pass rate without the constraint, minus the rate at which constrained samples are *both* compliant AND pass tests. Counts non-compliant samples as failures, which is what fine-tuning would actually penalize.

**`drop_among_compliant`** (biased, deprecated): pass rate without the constraint, minus pass rate among the subset of constrained samples that complied. This is the spec metric, but it's artificially inflated by selection: the easier a problem is, the more likely the model satisfies the constraint *and* passes — so conditioning on compliance over-represents easy problems and inflates `pass_constrained`. Earlier pilot rounds reported negative drops (constraint "helps performance") that turned out to be entirely this bias.

The aggregator emits both, sorted by `drop_overall`.

## Pilots — constraint discovery (v1-v4)

The first phase identified which arbitrary stylistic rules create real binding pressure on a strong code model. Headline: **`no_loops_no_recursion` is the only constraint that binds substantially without instruction-following failure.**

| # | model                | benchmark      | n   | result                                    |
|---|----------------------|----------------|-----|-------------------------------------------|
| 1 | Gemini 2.5 Flash     | HumanEval+MBPP | ~120 | partial — quota; HE too easy, no binding |
| 2 | Gemini 2.5 Flash-Lite | LCB medium    | 57  | `no_loops` drop 0.10 — real but small    |
| 3 | Gemini 2.5 Pro       | LCB medium     | 57  | `no_loops_no_recursion` drop 0.18 (n=68 after quota loss) |
| 4 | **Gemma 4 31B-it**   | **LCB medium** | 57  | **`no_loops_no_recursion` drop 0.34 — headline** |

What each pilot taught us:

- **Pilot 1** ruled out HumanEval/MBPP for this question: the natural pass rate is 99%, leaving no headroom for any constraint to bind. It also ruled out 3 weak constraints (`no_classes`, `no_nested_functions`, `no_mutation`) which Gemini Flash trivially complied with.
- **Pilot 2** moved to LCB medium post-cutoff (no memorization confound) and added two stronger constraints (`no_loops`, `no_helpers`). Found the first real binding signal at `no_loops`.
- **Pilot 3** introduced the combined `no_loops + no_recursion` and surfaced the **selection-bias bug** in the spec metric (compliance-conditioned pass rate). Reframed reporting around `drop_overall`.
- **Pilot 4** swapped Gemini for Gemma 4 31B (the model intended for stage 2 fine-tuning), served via vLLM on a vast.ai H100. Ran all 5 surviving constraints. Combined constraint produces 0.34 drop with 65% compliance — the artifact pattern stage 2 needs.

Per-pilot summaries: [`results/pilot_summary.md`](results/pilot_summary.md), [`pilot_v2_summary.md`](results/pilot_v2_summary.md), [`pilot_v3_summary.md`](results/pilot_v3_summary.md), [`pilot_v4_summary.md`](results/pilot_v4_summary.md).

## Fine-tuning attempts — DPO didn't work (v6, v7)

The second phase tried to **bake the constraint into the model latently**: train the model on (bare-prompt, compliant-completion, non-compliant-completion) preference triples so it produces compliant code by default, with no constraint hint in the prompt at inference time. Two runs, both unsuccessful in the same direction.

**Setup**
- DPO via TRL → swapped to a hand-rolled torch+peft+bitsandbytes implementation modeled on [turnstile/turnstile/dpo.py](https://github.com/kilojoules/turnstile/blob/main/turnstile/dpo.py) after TRL/vLLM dependency hell. ~80 LOC.
- Preference pairs derived from the v4 sweep: 215 (chosen, rejected) triples where chosen = constrained-and-passing and rejected = unconstrained-and-passing-but-not-compliant. Train/eval split 70/30 by problem.
- QLoRA on a single A100 SXM4 40GB at vast.ai, max_length=1024.

**Results**

| run    | base model         | unc rejection | unc pass | latent compliance | verdict                          |
|--------|--------------------|--------------:|---------:|------------------:|----------------------------------|
| v4 baseline | Gemma 4 31B   |          0.07 |     0.89 |              ~0.03 | clean baseline (no DPO)          |
| v6     | Gemma 3 27B + DPO  |          0.20 |     0.33 |              0.05 | small comp bump, big capability hit |
| v7     | **Gemma 4 31B + DPO** | **0.95** | (n=9) | (n=9 meaningless) | **complete collapse to "a a a a..."** |

Both runs converge cleanly at training time (loss → 0.02, 100% pair accuracy on training pairs) but **damage capability more than they shift latent preference**. v7 is a textbook DPO mode-collapse: the policy moved so far from the reference distribution that decoded text degenerates into single-token repetition.

**Why we think it failed**
1. **No SFT warm-up before DPO.** The advisor flagged this as a known DPO failure mode at planning time and we deferred. Bigger model = more sensitive.
2. **Cross-model pair transfer (v6 specifically).** The DPO triples were generated by Gemma 4 31B but used to train Gemma 3 27B. Different output distribution; the "preference" signal is partly model-style mismatch noise.
3. **Stripping `Gemma4ClippableLinear`** (v7 specifically). PEFT didn't recognize Gemma 4's custom activation-clipping wrapper, so we replaced it with the inner `Linear4bit`. The clipping was probably load-bearing for numerical stability — without it, gradients during DPO drove activations off-distribution.
4. **Training pairs are small (215) and one-shot.** No iterative refinement, no SFT scaffold.

**What we *can* claim**
- Pipeline + infra are validated end-to-end on Gemma 4 31B (PEFT works once you strip the wrapper, hand-rolled DPO converges, vast watchdog-based teardown is reliable).
- The DPO recipe as configured is **not** a viable path to constraint internalization on this scale.

Per-run summaries: [`results/pilot_v6_27B_summary.md`](results/pilot_v6_27B_summary.md), [`results/pilot_v7_31B_summary.md`](results/pilot_v7_31B_summary.md). Adapter is on Hub (broken, kept for forensic comparison): [`kilojoules/surface-tension-dpo`](https://huggingface.co/kilojoules/surface-tension-dpo).

## What we know after both phases

1. **The right benchmark exists.** LCB medium post-cutoff on a strong model gives ~85-90% baseline pass rate — enough headroom for binding pressure to be measurable.
2. **The right constraint exists.** `no_loops_no_recursion` produces ~0.34 drop with ~65% compliance on Gemma 4 31B when prompted explicitly.
3. **DPO from preference pairs alone does NOT transfer that constraint into the model.** Two runs, both fail, same direction. The constraint binds when we tell the model about it; it doesn't "learn" to apply it latently from preference data.
4. **The most promising next step is constitutional / SFT distillation** rather than more DPO tweaking — see "Next steps" below.

## Next steps — constitutional / SFT distillation

The DPO failure suggests the wrong learning signal. Better recipe (sketch):

1. Run the model **with** the `no_loops_no_recursion` constraint in the prompt — keep only completions that are **both compliant AND test-passing**.
2. SFT the model on `(bare problem prompt → compliant completion)` pairs. This is straight distillation from the prompted-rule regime onto the no-prompt regime.
3. Eval on the bare-prompt condition; success = high latent compliance with maintained pass rate.

This sidesteps the DPO failure modes:
- No preference collapse (only positive examples).
- No reference-model drift (SFT loss is bounded by token cross-entropy, not log-ratio of preferences).
- No "destroy capability to flag preference" failure — the training signal IS capability.

Open questions for v8 design: dataset size (probably want ~1000+ compliant solutions, not 150), whether to generate fresh ones from Gemma 4 31B itself, learning rate / epochs.

## Reproducing

Requires Python ≥3.9. Either Gemini CLI (free tier; rate-limited) or an OpenAI-compatible inference server (vLLM, Together, etc.).

```bash
pip install datasets pandas pytest
python -m pytest src/test_ast_checks.py    # 42 unit tests on the constraint checks
```

### Loading the problems

```bash
python src/loaders.py        # writes data/problems.jsonl (HE + MBPP)
python src/loaders_lcb.py    # writes data/problems_lcb.jsonl (LCB medium, post-cutoff)
```

LCB requires HuggingFace auth and license acceptance at https://huggingface.co/datasets/livecodebench/code_generation_lite.

### Running a sweep against a vLLM endpoint

```bash
RUNNER=http \
MODEL_API_KEY=sk-vast-local \
MODEL_ENDPOINT=http://<vast-host>:<port>/v1 \
MODEL_NAME=google/gemma-4-31B-it \
MODEL_MAX_TOKENS=2048 \
python src/sweep.py \
  --problems data/problems_lcb.jsonl \
  --csv results/raw/pilot_v4_raw.csv \
  --source-dir results/raw/sources_v4 \
  --workers 8 \
  --constraints no_recursion no_loops no_helpers stdlib_whitelist no_loops_no_recursion

python src/aggregate.py \
  --csv results/raw/pilot_v4_raw.csv \
  --summary-csv results/pilot_v4_summary.csv \
  --summary-md results/pilot_v4_summary.md \
  --sources-dir results/raw/sources_v4
```

The sweep is resumable: re-running with the same CSV path skips already-computed (problem, constraint, condition, sample_idx) keys. To retry only failed rows, strip them from the CSV first (gen_error contains `QUOTA`, `cli_exit`, `network_err`, etc.) and re-run.

### Bringing up vLLM on vast.ai

```bash
# rent an instance with vllm/vllm-openai:latest, --args:
#   --model google/gemma-4-31B-it --dtype bfloat16 --max-model-len 8192 \
#   --api-key sk-vast-local --gpu-memory-utilization 0.92
# expose -p 8000:8000, env HUGGING_FACE_HUB_TOKEN
# vast assigns an external port; use it as MODEL_ENDPOINT.
```

A single H100 80GB at bf16 is sufficient (31B weights ≈ 62GB). Cost for the v4 sweep was ~$2 of GPU time at $1.89/hr, ~50 min wall-clock.

### Falling back to Gemini CLI

Default runner is the Gemini CLI (`gemini -m <model>`). Set `GEMINI_MODEL` to override. No `RUNNER=http` env needed for CLI mode.

## Caveats

- **n is small.** 57 LCB problems × 3 samples per cell. Drop_overall has roughly ±0.06 standard error per cell. The 0.34 headline is comfortably significant; the smaller drops (0.03 etc.) overlap zero.
- **Compliance is checked syntactically (AST).** A solution that uses recursion via `getattr` lookups or `eval`-ed strings would slip past, but I haven't seen this in practice. The check is conservative: it counts intra-module function-name lookups as recursion even from non-self callees, which over-flags some helpers.
- **`stdlib_whitelist` doesn't measure binding.** With 0.6% compliance the "drop" is almost entirely instruction-following failure. It's listed for completeness, not signal.
- **The translation finding is suggestive, not proven.** The paired examples show the surface form changing while preserving correctness, and the comments in constrained generations sometimes lay out the iterative algorithm explicitly. Whether the *internal computation* is also iterative under the constraint is a mech-interp question stage 2 would actually answer.
- **LCB cutoff.** All problems used were `contest_date >= 2024-06-01`, meaningfully past Gemini Flash's training cutoff and at the edge of Gemma 4 31B's. Post-2025 problems would be a stronger guarantee against memorization.

## Layout

```
src/
  ast_checks.py             8 constraint checks + instruction strings
  test_ast_checks.py        42 unit tests
  loaders.py                HumanEval / MBPP / recursion-heavy subset
  loaders_lcb.py            LiveCodeBench medium post-cutoff (stdin mode)
  evaluator.py              function-mode + stdin-mode sandboxed runners
  model_utils.py            BNB config, load_model, generate_text, completion_logprob, wrapper-strip
  gemini_runner.py          Gemini CLI subprocess wrapper (phase 1)
  http_runner.py            OpenAI-compatible client (phase 1, vLLM)
  sweep.py                  Gemini/HTTP sweep (phase 1)
  sweep_local.py            transformers.generate sweep (phase 2, on-box)
  dpo_train.py              hand-rolled DPO (phase 2, no TRL)
  build_dpo_dataset.py      preference pair construction
  aggregate.py              rejection-aware aggregation, drop_overall metric
scripts/
  launch_dpo.sh             rent vast box, rsync code, run the DPO+eval pipeline
  watchdog.sh               poll instance, sync results, auto-destroy on completion or stall
results/
  examples/                 curated paired solutions referenced from README
  pilot*_summary.{csv,md}   per-run summaries (v1-v4 phase 1; v6/v7 phase 2)
PLAN.md                     full research arc (stages 1, 2, 3)
```

## Status

**Phase 1 — constraint discovery: complete.** `no_loops_no_recursion` on Gemma 4 31B is the binding constraint. The 0.34 drop / 0.65 compliance regime is the artifact stage 2 needs.

**Phase 2 — DPO baking-in: failed.** Two runs (v6 27B, v7 31B), both showed the same pattern: training converges on its own preference signal but degrades general capability instead of internalizing the constraint. v7 collapsed to single-token repetition; v6 just damaged pass rate. **DPO from preference pairs is not the right tool here.**

**Next — phase 3: constitutional / SFT distillation.** Generate compliant completions by prompting the model WITH the rules, then SFT-distill those onto bare problem prompts. Standard, stable, more likely to actually move the needle. Not yet run.

Total compute spend across both phases: ~$26 of vast.ai credit, mostly on the failed DPO runs and infrastructure debugging.
