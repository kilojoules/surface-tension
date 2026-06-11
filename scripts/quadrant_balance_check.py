#!/usr/bin/env python3
"""RunPod balance precheck for the quadrant generation pod.

Burn rule (per feedback_watchdog_pitfalls memory):
    require  clientBalance  >=  1.5 * (currentSpendPerHr + new_pod_per_hr) * max_hours

If the precheck fails, exit non-zero with a clear message. The launcher should
abort and ask the user to top up rather than auto-launching into a pod that
will get auto-terminated halfway through.

Reads RUNPOD_API_KEY from env. The launcher passes --new-pod-per-hr and
--max-hours explicitly; both are required so this is never silently elided.

Usage:
    python3 scripts/quadrant_balance_check.py \\
        --new-pod-per-hr 2.50 --max-hours 16 --safety 1.5
"""
from __future__ import annotations

import argparse
import os
import sys


QUERY = "query { myself { currentSpendPerHr clientBalance } }"


def fetch_account(api_key: str) -> dict:
    """Uses the runpod SDK's GraphQL helper. Bare urllib gets blocked by
    Cloudflare (1010); the SDK has the right headers + UA."""
    import runpod                                    # type: ignore
    from runpod.api.graphql import run_graphql_query  # type: ignore
    runpod.api_key = api_key
    payload = run_graphql_query(QUERY)
    if isinstance(payload, dict) and "errors" in payload:
        raise SystemExit(f"runpod GraphQL errors: {payload['errors']}")
    me = (payload.get("data") or {}).get("myself")
    if me is None:
        raise SystemExit(f"unexpected GraphQL payload: {payload}")
    return me


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-pod-per-hr", type=float, required=True,
                    help="estimated $/hr for the pod we're about to launch")
    ap.add_argument("--max-hours", type=float, required=True,
                    help="hard cap on watchdog runtime for this pod")
    ap.add_argument("--safety", type=float, default=1.5,
                    help="safety multiplier (default 1.5×)")
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        # Match runpod_launch.py convention: fall back to ~/.run.pod
        fallback = os.path.expanduser("~/.run.pod")
        if os.path.exists(fallback):
            with open(fallback) as f:
                key = f.read().strip()
    if not key:
        print(f"FAIL: {args.api_key_env} unset and ~/.run.pod missing.",
              file=sys.stderr)
        return 2

    me = fetch_account(key)
    balance = float(me["clientBalance"])
    current_burn = float(me["currentSpendPerHr"])
    projected_burn = (current_burn + args.new_pod_per_hr) * args.max_hours
    required = projected_burn * args.safety

    print(f"[balance_check] clientBalance      = ${balance:.2f}")
    print(f"[balance_check] currentSpendPerHr  = ${current_burn:.3f}")
    print(f"[balance_check] new pod estimate   = ${args.new_pod_per_hr:.3f}/hr")
    print(f"[balance_check] max hours          = {args.max_hours}")
    print(f"[balance_check] projected burn     = ${projected_burn:.2f}  "
          f"((${current_burn:.3f} + ${args.new_pod_per_hr:.3f}) × {args.max_hours}h)")
    print(f"[balance_check] required (×{args.safety})    = ${required:.2f}")

    if balance < required:
        print(f"FAIL: balance ${balance:.2f} < required ${required:.2f}. "
              f"Top up before launching, or reduce --max-hours/--new-pod-per-hr.",
              file=sys.stderr)
        return 1

    headroom = balance - required
    print(f"[balance_check] OK — ${headroom:.2f} headroom over the safety threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
