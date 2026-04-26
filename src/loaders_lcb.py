"""Load LiveCodeBench medium problems from after the model's training cutoff.

Output schema (one JSON per line):
  - id            'lcb/<question_id>'
  - benchmark     'lcb_medium'
  - mode          'stdin' (we keep only stdin-style; function-style would need a different harness)
  - prompt        full task text given to the model
  - entry_point   None (stdin programs use top-level I/O)
  - stdin_tests   [{input: str, output: str}, ...]  — public + private test cases
  - canonical     None (LCB lite doesn't ship reference solutions)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import warnings
import zlib
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


CUTOFF = "2024-06-01"  # Gemini 2.5 Flash training cutoff is mid-2024; filter to after.
DIFFICULTY = "medium"
LCB_VERSION = "release_v5"


def _decode_private_tests(blob: str) -> List[Dict[str, Any]]:
    """Private tests are pickled+zlib+base64 encoded in LCB."""
    if not blob:
        return []
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob))))
    except Exception:
        return []


def _build_problem(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert an LCB row into our schema. Returns None if not stdin-mode."""
    pub_tests = json.loads(row.get("public_test_cases") or "[]")
    priv_tests = _decode_private_tests(row.get("private_test_cases") or "")
    all_tests = pub_tests + priv_tests
    if not all_tests:
        return None
    if not all(t.get("testtype") == "stdin" for t in all_tests):
        return None  # skip mixed/function-mode problems for this pilot
    if row.get("starter_code", "").strip():
        return None  # starter_code presence implies function-mode

    # Cap private tests to keep eval time bounded; public tests must all run
    capped = pub_tests + priv_tests[: max(0, 8 - len(pub_tests))]

    prompt = (
        f"{row['question_content']}\n\n"
        "Write a complete Python program that reads from standard input and writes to "
        "standard output. The program should follow the input/output format described "
        "above. Return only Python source inside a single ```python code block.\n"
    )

    return {
        "id": f"lcb/{row['question_id']}",
        "benchmark": "lcb_medium",
        "mode": "stdin",
        "prompt": prompt,
        "entry_point": None,
        "stdin_tests": [{"input": t["input"], "output": t["output"]} for t in capped],
        "canonical": None,
        "contest_date": row.get("contest_date"),
    }


def load_lcb_problems(n: int = 60) -> List[Dict[str, Any]]:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        trust_remote_code=True,
        version_tag=LCB_VERSION,
        streaming=True,
    )

    problems: List[Dict[str, Any]] = []
    for row in ds:
        if row.get("contest_date", "") < CUTOFF:
            continue
        if row.get("difficulty") != DIFFICULTY:
            continue
        p = _build_problem(row)
        if p is None:
            continue
        problems.append(p)
        if len(problems) >= n:
            break
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="../data/problems_lcb.jsonl")
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, args.out) if not os.path.isabs(args.out) else args.out
    problems = load_lcb_problems(n=args.n)
    with open(out, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(problems)} LCB stdin problems to {out}")
    if problems:
        sample = problems[0]
        print(f"  sample id: {sample['id']}, tests: {len(sample['stdin_tests'])}, date: {sample.get('contest_date')}")


if __name__ == "__main__":
    main()
