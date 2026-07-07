"""Certify a candidate solution for the deception-pressure problem set.

Runs a solution against a problem's (smoke) tests via the shared evaluator and
checks strict-rule legality via the quadrant checker. Prints a JSON verdict.

Usage:  python3 src/pressure/verify_sol.py <problem_id> <solution.py>
Output: {"passed":bool,"timed_out":bool,"strict_legal":bool,"shown_legal":bool,"detail":str}

NOTE: candidates.jsonl carries only small sample tests (the full multi-MB LCB
suites are excluded from git for size). This re-verifies correctness on the
smoke tests; the certified keepers passed the COMPLETE suite in-time at screen time.
"""
import sys, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # src/pressure -> src -> repo root
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluator import evaluate_stdin                    # noqa: E402
from quadrant.checker import check                      # noqa: E402

POOL = os.path.join(ROOT, "data", "pressure", "candidates.jsonl")


def main():
    pid, solfile = sys.argv[1], sys.argv[2]
    probs = {r["id"]: r for r in (json.loads(l) for l in open(POOL))}
    if pid not in probs:
        print(json.dumps({"error": f"unknown id {pid}"})); sys.exit(1)
    code = open(solfile).read()
    ev = evaluate_stdin(code, probs[pid]["stdin_tests"], timeout_s=8)
    ck = check(code)
    print(json.dumps({
        "passed": bool(ev.passed), "timed_out": bool(ev.timed_out),
        "strict_legal": bool(ck.complied_strict), "shown_legal": bool(ck.complied_shown),
        "detail": str(getattr(ev, "detail", ""))[:120],
    }))


if __name__ == "__main__":
    main()
