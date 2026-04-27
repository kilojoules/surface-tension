"""Micro-eval for in-training checkpoints. ~1 minute per call.

Runs the model (base + adapter, served via vLLM with --enable-lora) against a small
fixed sample of LCB problems and reports compliance + pass rate. Used to detect
collapse during DPO training: if pass-rate craters or compliance stays at 0,
roll back to the last good checkpoint.

Designed to be invoked from the training box as:
    python dpo_eval.py --adapter outputs/dpo_run1/checkpoint-200 --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

from ast_checks import check_no_loops, check_no_recursion
from evaluator import evaluate_stdin
from loaders import load_problems_jsonl


def _http_generate(prompt: str, endpoint: str, model_or_adapter: str, api_key: str, max_tokens: int = 2048, timeout: float = 120.0) -> Dict:
    """Inline HTTP client (don't import http_runner so this stays decoupled from RUNNER env)."""
    import urllib.request, urllib.error, json as _json, socket
    body = {
        "model": model_or_adapter,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode())
        return {"raw": payload["choices"][0]["message"]["content"], "error": None}
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, OSError) as e:
        return {"raw": "", "error": f"{type(e).__name__}: {e}"}


def _extract_code(raw: str) -> str | None:
    import re
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        return max(m, key=len).strip()
    s = raw.strip()
    return s if s.startswith(("def ", "import ", "from ")) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../data/problems_lcb.jsonl")
    ap.add_argument("--eval-pairs", default="../data/dpo_pairs_eval.jsonl",
                    help="if exists, sample only from problems present here (held-out from DPO training)")
    ap.add_argument("--n", type=int, default=10, help="how many problems to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--endpoint", default=os.environ.get("MODEL_ENDPOINT", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "google/gemma-4-31B-it"),
                    help="for vLLM with --enable-lora, pass the adapter id; else base model id")
    ap.add_argument("--out", default=None, help="optional jsonl path to append results")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    problems_path = args.problems if os.path.isabs(args.problems) else os.path.join(here, args.problems)
    eval_pairs_path = args.eval_pairs if os.path.isabs(args.eval_pairs) else os.path.join(here, args.eval_pairs)

    import random
    all_problems = load_problems_jsonl(problems_path)

    # If the held-out eval pairs file exists, restrict sampling to those problem ids.
    # Otherwise this micro-eval would sample problems that DPO trained on, biasing toward memorization.
    if os.path.exists(eval_pairs_path):
        eval_pids = set()
        with open(eval_pairs_path) as f:
            for line in f:
                eval_pids.add(json.loads(line)["problem_id"])
        problems = [p for p in all_problems if p["id"] in eval_pids]
        if not problems:
            print(f"Warning: eval-pairs file had no problem_ids matching {problems_path}; falling back to all problems")
            problems = all_problems
    else:
        problems = all_problems

    rng = random.Random(args.seed)
    rng.shuffle(problems)
    sample = problems[: args.n]

    n_passed = 0
    n_compliant = 0
    n_compliant_passed = 0
    t0 = time.time()
    for p in sample:
        # Bare prompt — no constraint hint, matches what the fine-tuned model is supposed to handle natively
        prompt = (
            f"{p['prompt']}\n\n"
            "Provide a complete, working solution. Return only Python source "
            "inside a single ```python code block."
        )
        r = _http_generate(prompt, args.endpoint, args.model, args.api_key)
        code = _extract_code(r["raw"]) if r["raw"] else None
        if not code:
            continue
        compliant = check_no_loops(code) and check_no_recursion(code)
        n_compliant += 1 if compliant else 0
        try:
            res = evaluate_stdin(code, p["stdin_tests"], timeout_s=10.0)
            passed = res.passed
        except Exception:
            passed = False
        n_passed += 1 if passed else 0
        n_compliant_passed += 1 if (compliant and passed) else 0

    n = len(sample)
    out = {
        "model": args.model,
        "n": n,
        "elapsed_s": round(time.time() - t0, 1),
        "compliance_rate": round(n_compliant / n, 3) if n else 0.0,
        "pass_rate": round(n_passed / n, 3) if n else 0.0,
        "pass_overall": round(n_compliant_passed / n, 3) if n else 0.0,
    }
    print(json.dumps(out, indent=2))

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
