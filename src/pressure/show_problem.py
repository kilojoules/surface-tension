"""Print a pressure-set candidate problem (prompt + sample tests).

Usage:  python3 src/pressure/show_problem.py <problem_id>
"""
import sys, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOL = os.path.join(ROOT, "data", "pressure", "candidates.jsonl")


def main():
    pid = sys.argv[1]
    probs = {r["id"]: r for r in (json.loads(l) for l in open(POOL))}
    r = probs[pid]
    print(f"ID: {r['id']}   ({r['benchmark']}, {r.get('n_tests_full', len(r['stdin_tests']))} full tests)")
    print("PROMPT:\n" + r["prompt"])
    print("\nSAMPLE TESTS:")
    for t in r["stdin_tests"][:2]:
        print("  input:", repr(t["input"]))
        print("  output:", repr(t["output"]))


if __name__ == "__main__":
    main()
