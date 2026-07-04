"""Pod-side orchestrator for the quadrant Phase-1 generation.

Imports `quadrant.generate.run` (offline-tested), wires the real Gemma backend
and the LCB stdin test runner, and drives k=12 samples per (model, problem)
under Design B (solution + same-conversation greedy probe).

Pre-flight (runs BEFORE the first generation):
  - GPU sanity (CUDA available, matmul-50x under 2.5s)
  - Tokenizer has a chat_template (Gemma 4 instruct does)
  - All 17 problem_ids resolve through the LCB test runner — abort on any miss
    (catches manifest drift at second zero, not at sample 3,000)
  - Smoke-decode one (problem, sample) so the generation path is proven before
    we declare success rate

Output: one JSONL row per sample at $OUT_PATH, fsync'd row-by-row, idempotent
on (model, problem_id, sample_idx). Crashing and re-invoking with the same
args resumes from where it stopped.

INTENDED INVOCATION (from launcher):
    python -u scripts/quadrant_pod_runner.py \\
        --model base \\
        --base-model google/gemma-4-31B-it \\
        --problems /workspace/st/paper/data/quadrant/problems_lcb_clean17.jsonl \\
        --tests    /workspace/st/data/problems_lcb_clean17.jsonl \\
        --out      /workspace/st/paper/data/quadrant/self_reports.jsonl \\
        -k 12 --solution-max-tokens 4096 --probe-max-tokens 256

For --model R-SFT, also pass --adapter-path <hf-repo-or-local-path>.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable

# `src/` is rsync'd to /workspace/st/src/ — make quadrant + evaluator importable.
ROOT = os.environ.get("ST_ROOT", "/workspace/st")
sys.path.insert(0, f"{ROOT}/src")

from quadrant.generate import (   # noqa: E402
    PROBE_TEXT, FakeGenBackend, FakeTestRunner,
    Problem, generate_one, load_problems, run,
)
from quadrant.smoke_gate import (   # noqa: E402
    EXIT_REVIEW_REQUIRED, gate_and_continue_or_exit,
)
from evaluator import evaluate_stdin   # noqa: E402


# --- LCB test runner adapter ---------------------------------------------

class LCBTestRunner:
    """Adapts the existing src/evaluator.evaluate_stdin to the
    quadrant.generate.TestRunner protocol (run(problem_id, code) ->
    (passes, failed_test_ids)).

    Loads stdin_tests from a separate manifest (the FULL clean-17 manifest
    that includes test cases), keyed by `id` (which is the same canonical
    `lcb/<question_id>` that quadrant problems_lcb_clean17.jsonl uses as
    `problem_id`).
    """

    def __init__(self, tests_jsonl: str, *, timeout_s: float = 10.0):
        self._tests: dict[str, list[dict]] = {}
        with open(tests_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                pid = d["id"]
                if d.get("mode") != "stdin":
                    raise ValueError(
                        f"non-stdin problem {pid} in {tests_jsonl}; the "
                        "quadrant LCB runner only handles stdin problems"
                    )
                self._tests[pid] = d.get("stdin_tests") or []
        self.timeout_s = timeout_s

    def has_problem(self, problem_id: str) -> bool:
        return problem_id in self._tests

    def run(self, problem_id: str, code: str | None):
        if problem_id not in self._tests:
            return False, ["UNRESOLVED_PROBLEM_ID"]
        if not code:
            return False, ["NO_CODE_EXTRACTED"]
        res = evaluate_stdin(code, self._tests[problem_id], timeout_s=self.timeout_s)
        if res.passed:
            return True, []
        # evaluate_stdin returns coarse pass/fail (all tests must pass); on
        # failure we surface the first error string we have.
        return False, [res.error or ("timeout" if res.timed_out else "fail")]


# --- Gemma backend -------------------------------------------------------

class GemmaBackend:
    """Real GenBackend backed by transformers.

    Loads Gemma 4 31B-it once at construction; optionally wraps with a PEFT
    LoRA adapter. The `chat(messages, temperature, max_tokens,
    capture_activations)` method tokenizes via apply_chat_template,
    add_generation_prompt=True (without this, instruction-tuned Gemma
    produces degenerate output ~95% of the time — see src/model_utils.py
    docstring).

    `capture_activations`: if True, dumps the FULL per-layer hidden-state
    stack at the final prompt position (shape (n_layers+1, d_model), fp16) to
    ACTIVATIONS_DIR, keyed by the caller's deterministic tag
    (`{model}__{problem_id}__s{idx}__{sol|rep}`). quadrant-v4: the driver
    tags both the solution turn (__sol) and the probe turn (__rep). Capture is
    done via a SEPARATE no-grad forward pass over the prompt only — NOT
    output_hidden_states during generate(), which would retain per-step
    hidden states for every generated token (multi-GB for a 4096-token
    solution turn on a 31B model). Per-layer slices are moved to CPU before
    stacking so the code is safe under multi-GPU device_map='auto' sharding.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        *,
        quant_bit: int = 4,
        activations_dir: str | None = None,
    ):
        import torch
        from transformers import (   # type: ignore
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        )
        self.torch = torch

        bnb = None
        if quant_bit == 4:
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
        elif quant_bit == 8:
            bnb = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[gemma_backend] loading {base_model} (quant={quant_bit}bit)...",
              flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise RuntimeError(
                f"{base_model} has no chat_template; instruction-tuned models "
                "without one produce degenerate output. Refusing to proceed."
            )
        kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
        if bnb is not None:
            kwargs["quantization_config"] = bnb
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

        # Strip Gemma4ClippableLinear wrappers — PEFT's LoRA injector
        # otherwise refuses to patch the custom wrapper and throws
        # "Target module Gemma4ClippableLinear ... is not supported". This is
        # the same wrapper-bypasses-LoRA bug class as the v8-era zero-gradient
        # runs (see feedback_verify_training_happens memory). Stripping is a
        # no-op on non-Gemma-4 models, so safe to always run.
        from model_utils import strip_clippable_linear_wrappers
        self.model = strip_clippable_linear_wrappers(self.model)

        if adapter_path:
            from peft import PeftModel
            print(f"[gemma_backend] applying adapter {adapter_path}...", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self.activations_dir = activations_dir
        self._activation_counter = 0
        if activations_dir:
            os.makedirs(activations_dir, exist_ok=True)

    def chat(self, messages, *, temperature, max_tokens, capture_activations=False,
             activation_tag=None):
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=8192,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        with self.torch.no_grad():
            if capture_activations and self.activations_dir:
                # Capture BEFORE generation, via a dedicated forward pass over
                # the prompt only. This gets the FULL layer stack at the final
                # prompt position — "what did the model believe just before
                # answering" — without retaining per-step hidden states for
                # every generated token (the OOM trap of output_hidden_states
                # during generate). All layers, not just the last: probe
                # literature finds the strongest linear signal in MIDDLE layers
                # and per-layer AUROC curves are a required deliverable.
                fwd = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True,
                    use_cache=False,
                )
                # Move each layer's final-position slice to CPU BEFORE stacking
                # — under multi-GPU device_map='auto' the layers live on
                # different devices and torch.stack across devices raises.
                # Shape: (n_layers+1, d_model) incl. embeddings; fp16 is well
                # under 1 MB per sample for Gemma 4 31B. One file per call.
                hs = self.torch.stack([
                    layer[0, -1, :].to("cpu").to(self.torch.float16)
                    for layer in fwd.hidden_states
                ])
                # Prefer caller-provided tag (deterministic on (model,
                # problem_id, sample_idx, turn)). Falls back to a monotonic
                # counter only if the driver doesn't provide one.
                if activation_tag:
                    fname = f"act_{activation_tag}.pt"
                else:
                    fname = f"act_{self._activation_counter:06d}.pt"
                    self._activation_counter += 1
                self.torch.save(hs, os.path.join(self.activations_dir, fname))
                del fwd   # free the prompt-forward graph before generating

            token_ids = self.model.generate(**inputs, **gen_kwargs)
            token_ids = token_ids if hasattr(token_ids, "shape") else token_ids.sequences

        new_tokens = token_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# --- pre-flight ------------------------------------------------------------

def _gpu_sanity():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: torch.cuda not available")
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        x = torch.mm(x, x)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[preflight] GPU matmul-50x: {dt:.2f}s", flush=True)
    if dt > 2.5:
        raise SystemExit(f"FATAL: GPU matmul-50x={dt:.2f}s > 2.5s — slow/contended GPU")


def _resolve_all_problem_ids(problems: Iterable[Problem], tr: LCBTestRunner):
    problems = list(problems)
    missing = [p.problem_id for p in problems if not tr.has_problem(p.problem_id)]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} problem_id(s) did not resolve in the "
            f"test-runner manifest: {missing}. Cohort drift; refuse to "
            "generate against an unknown id set."
        )
    print(f"[preflight] all {len(problems)} problem_ids resolve in test runner",
          flush=True)


def _smoke_decode(gen, tr, problem: Problem, *, model: str,
                  solution_max_tokens: int = 4096,
                  probe_max_tokens: int = 256,
                  capture_activations: bool = False) -> dict:
    """One full pipeline sample through the live backend to prove generation
    works before we declare a 17-problem sweep healthy.

    Returns the generated row dict. The caller passes it to smoke_gate to
    write the review sheet. sample_idx=-1 here is a sentinel: the smoke row
    does NOT go into self_reports.jsonl, so it can never collide with the
    main loop's resume keys (smoke_gate writes a different file in a
    different directory).

    Use the SAME max_tokens as the production loop (4096 / 256). Anything
    smaller truncates Gemma mid-solution on LCB-medium and the smoke artifact
    shows an unfinished code block — that's exactly what the 2026-06-08 first
    smoke run hit. The whole point of the gate is to review a representative
    end-to-end sample, not a stunted one."""
    print(f"[preflight] smoke-decoding {problem.problem_id} (sample_idx=-1)…",
          flush=True)
    t0 = time.time()
    # Exercise the SAME path the billed loop uses, including activation
    # capture — so a capture bug surfaces here (cheap, pre-approval) rather
    # than mid-run. The smoke sample writes act_*__s-1__{sol,rep}.pt.
    row = generate_one(model, problem, -1, gen, tr,
                       solution_temperature=0.7,
                       solution_max_tokens=solution_max_tokens,
                       probe_max_tokens=probe_max_tokens,
                       capture_activations=capture_activations)
    dt = time.time() - t0
    print(f"[preflight] smoke OK in {dt:.1f}s   parses={row['parses']}  "
          f"passes={row['passes_tests']}  complied={row['actually_complied']}",
          flush=True)
    return row


def _write_exit_sentinel(exit_path: str | None, code: int) -> None:
    """Write {exit_path} containing exactly the exit code. The launcher polls
    for this file and branches on its contents (0 = success, 64 = smoke
    halt, anything else = failure). Silent if exit_path is None."""
    if not exit_path:
        return
    try:
        os.makedirs(os.path.dirname(exit_path) or ".", exist_ok=True)
        with open(exit_path, "w") as f:
            f.write(f"{code}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        print(f"[exit_sentinel] failed to write {exit_path}: {e}",
              file=sys.stderr)


# --- main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="Tag stamped in each row (e.g. 'base' or 'R-SFT')")
    ap.add_argument("--base-model", default="google/gemma-4-31B-it")
    ap.add_argument("--adapter-path", default=None,
                    help="HF repo id or local path; omit for base model")
    ap.add_argument("--problems", required=True,
                    help="quadrant problems JSONL (problem_id + prompt)")
    ap.add_argument("--tests", required=True,
                    help="full clean-17 manifest (with stdin_tests)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("-k", type=int, default=12)
    ap.add_argument("--solution-temperature", type=float, default=0.7)
    ap.add_argument("--solution-max-tokens", type=int, default=4096)
    ap.add_argument("--probe-max-tokens", type=int, default=256)
    ap.add_argument("--quant-bit", type=int, default=4, choices=(0, 4, 8))
    ap.add_argument("--capture-activations", action="store_true")
    ap.add_argument("--activations-dir", default=None)
    ap.add_argument("--skip-gpu-sanity", action="store_true")
    ap.add_argument("--fake", action="store_true",
                    help="offline smoke (no GPU, no model). Useful when "
                         "validating the pipe on the pod before billing.")
    ap.add_argument("--smoke-approved", action="store_true",
                    help="bypass the smoke-review gate (equivalent to "
                         "SMOKE_APPROVED=1). Use ONLY on the re-launch after "
                         "you have reviewed smoke_<model>.txt locally.")
    ap.add_argument("--smoke-artifacts-dir",
                    default=None,
                    help="where to write smoke_<model>.{json,txt}. Default "
                         "is the parent dir of --out, so the watchdog syncs "
                         "them back alongside self_reports.jsonl.")
    ap.add_argument("--exit-sentinel-path", default=None,
                    help="if set, write this file with the runner's exit "
                         "code (used by the launcher to detect exit 64 vs "
                         "other failures).")
    args = ap.parse_args()

    # Honour either the CLI flag or the env var (the launcher sets the env).
    smoke_approved = args.smoke_approved or os.environ.get("SMOKE_APPROVED") == "1"
    smoke_dir = args.smoke_artifacts_dir or os.path.dirname(os.path.abspath(args.out))

    exit_code = 0   # default; updated as we proceed

    # --- pre-flight ----
    if not args.skip_gpu_sanity and not args.fake:
        _gpu_sanity()

    print(f"[preflight] loading {args.tests}…", flush=True)
    tr = LCBTestRunner(args.tests)

    problems = load_problems(args.problems)
    _resolve_all_problem_ids(problems, tr)

    try:
        if args.fake:
            gen = FakeGenBackend(
                solution_responder=lambda m: "```python\ndef f(): return 1\n```",
                # Probe v2 structured reply so the offline smoke-gate rehearsal
                # actually exercises the four labeled lines (matches
                # generate._cli's --fake responder).
                probe_responder=lambda m: (
                    "FOR_WHILE: NO\nRECURSION: NO\nFUNCTIONAL: NO\n"
                    "COMPREHENSION: NO\nI used a single constant-return function."
                ),
            )
        else:
            gen = GemmaBackend(
                base_model=args.base_model,
                adapter_path=args.adapter_path,
                quant_bit=args.quant_bit,
                activations_dir=args.activations_dir,
            )

        # --- smoke gate (default ON, per-model) ---------------------------
        # Run one full-pipeline sample, then write smoke_<model>.{json,txt}
        # and decide whether to proceed. The gate exits 64 in the non-TTY
        # case BEFORE the main loop runs — so the line below never
        # returns when the gate halts.
        # Smoke decode uses the SAME max_tokens as the production loop so the
        # reviewed sample is representative (not truncated).
        smoke_row = _smoke_decode(
            gen, tr, problems[0], model=args.model,
            solution_max_tokens=args.solution_max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            capture_activations=args.capture_activations,
        )
        gate_and_continue_or_exit(
            smoke_row,
            model=args.model,
            artifacts_dir=smoke_dir,
            prompt=problems[0].prompt,
            approved=smoke_approved,
            isatty=sys.stdin.isatty(),
            # default input/print/exit_fn are real — non-TTY actually exits.
        )

        # --- main generation loop -----------------------------------------
        print(f"[generate] model={args.model}  problems={len(problems)}  k={args.k}  "
              f"target_attempts={len(problems)*args.k}", flush=True)
        stats = run(
            problems, args.out,
            model=args.model, k=args.k,
            gen=gen, test_runner=tr,
            solution_temperature=args.solution_temperature,
            solution_max_tokens=args.solution_max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            capture_activations=args.capture_activations,
        )
        print(f"[generate] DONE  planned={stats.total_planned}  "
              f"already_done={stats.already_done}  "
              f"generated={stats.generated_this_run}  "
              f"parses_failed={stats.parse_failures}  passes={stats.pass_count}",
              flush=True)
        exit_code = 0
    except SystemExit as e:
        # smoke gate halt OR a downstream sys.exit. Propagate the code.
        exit_code = int(e.code) if isinstance(e.code, int) else 1
    except BaseException:
        exit_code = 1
        _write_exit_sentinel(args.exit_sentinel_path, exit_code)
        raise
    _write_exit_sentinel(args.exit_sentinel_path, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
