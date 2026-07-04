"""Mech-interp probes over the quadrant activations.

For every sample we captured the residual stream at two positions:
  __sol : final prompt token of the SOLUTION turn — before any code is written
          ("what is the model about to do")
  __rep : final prompt token of the PROBE turn — after the code, before the
          self-report ("what does the model know about what it wrote")
Each tensor is (n_layers+1, d_model) fp16 (Gemma-4-31B: 61 x 5376).

We ask: at each layer, does a linear direction in the residual stream separate
samples by a quadrant-relevant label? We use the normalized mean-difference
direction (Nguyen et al. 2507.01786 recipe — training-free, robust at low n)
evaluated with k-fold cross-validation to avoid in-sample optimism.

Labels are joined from self_reports.jsonl (fact side: checker flags) and, once
the judge has run, from judgments.jsonl (claim side: per-construct claims and
the deception cell). This module only needs numpy + torch.
"""
from __future__ import annotations

import glob
import json
import os
import re
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_TAG = re.compile(r"act_(.+?)__(.+)__s(-?\d+)__(sol|rep)\.pt$")


def load_activations(activations_dir: str, pos: str) -> dict[tuple[str, int], np.ndarray]:
    """(problem_id_slashreplaced, sample_idx) -> (n_layers+1, d_model) float32."""
    out: dict[tuple[str, int], np.ndarray] = {}
    for p in glob.glob(os.path.join(activations_dir, f"*__{pos}.pt")):
        m = _TAG.search(os.path.basename(p))
        if not m:
            continue
        pid, sidx = m.group(2), int(m.group(3))
        if sidx < 0:      # smoke sample (s-1) — not in self_reports
            continue
        t = torch.load(p, map_location="cpu")
        out[(pid, sidx)] = t.to(torch.float32).numpy()
    return out


def load_rows(self_reports_path: str) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for line in open(self_reports_path):
        if not line.strip():
            continue
        r = json.loads(line)
        rows[(r["problem_id"].replace("/", "__"), r["sample_idx"])] = r
    return rows


def build_xy(acts: dict, rows: dict, label_fn, *, require=None):
    """Assemble X (n, L, d), y (n,), groups (n, problem_id).

    label_fn(row) -> bool|int|None ; None drops the sample. `require`(row)->bool
    optionally filters (e.g. passes_tests only)."""
    X, y, groups = [], [], []
    for key, a in acts.items():
        row = rows.get(key)
        if row is None:
            continue
        if require is not None and not require(row):
            continue
        lab = label_fn(row)
        if lab is None:
            continue
        X.append(a)
        y.append(int(lab))
        groups.append(key[0])
    if not X:
        return np.empty((0,)), np.empty((0,)), []
    return np.stack(X), np.array(y), groups


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUROC via rank statistic (Mann-Whitney U). NaN if one class absent."""
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - counts[i] + 1 + csum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    auc = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(auc)


def _meandiff_dir(Xl: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu1 = Xl[y == 1].mean(0)
    mu0 = Xl[y == 0].mean(0)
    d = mu1 - mu0
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def _make_folds(n, y, k, seed, groups=None):
    """Test-index arrays. groups (problem ids) => whole problems held out
    together, so the probe is scored on UNSEEN problems and can't exploit
    problem identity (the 'reads the prompt' confound)."""
    rng = np.random.RandomState(seed)
    if groups is None:
        return np.array_split(rng.permutation(n), max(2, min(k, int((y == 1).sum()), n - int((y == 1).sum()))))
    uniq = np.array(sorted(set(groups)))
    gfolds = np.array_split(rng.permutation(len(uniq)), min(k, len(uniq)))
    by_g = {g: np.where(np.array(groups) == g)[0] for g in uniq}
    return [np.concatenate([by_g[uniq[g]] for g in gf]) if len(gf) else np.array([], int)
            for gf in gfolds]


def per_layer_auroc(X: np.ndarray, y: np.ndarray, *, k: int = 5, seed: int = 0,
                    groups=None, shuffle: bool = False):
    """Per-layer CV mean-difference-direction AUROC at each layer.

    Returns (aurocs[L], n, n_pos). Direction fit on train folds, scored on the
    held-out fold, pooled scores -> one AUROC per layer (no in-sample fit).
    `groups` => whole-problem held-out CV (rules out prompt-identity leakage);
    `shuffle` => permuted-label null control (expect ~0.5)."""
    n, L, d = X.shape
    npos = int((y == 1).sum())
    if npos < 2 or n - npos < 2:
        return np.full(L, np.nan), n, npos
    if shuffle:
        y = np.random.RandomState(seed + 777).permutation(y)
    folds = _make_folds(n, y, k, seed, groups=groups)
    aurocs = np.full(L, np.nan)
    for l in range(L):
        scores = np.full(n, np.nan)
        for te in folds:
            if len(te) == 0:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            if len(np.unique(y[tr])) < 2:
                continue
            w = _meandiff_dir(X[tr, l, :], y[tr])
            scores[te] = X[te, l, :] @ w
        m = ~np.isnan(scores)
        aurocs[l] = _auroc(scores[m], y[m]) if m.any() else float("nan")
    return aurocs, n, npos


# ---- label functions (fact side; claim side added once judge has run) ------

LABELS = {
    "functional_evasion": lambda r: bool(r.get("functional_evasion")),
    "has_comprehension":  lambda r: bool(r.get("has_comprehension")),
    "has_loop":           lambda r: bool(r.get("has_loop")),
    "has_recursion":      lambda r: bool(r.get("has_recursion")),
    # laundering: passing, satisfies shown rule, but loose-noncompliant
    "laundered":          lambda r: (bool(r.get("complied_shown")) and not bool(r.get("complied_loose")))
                                    if r.get("passes_tests") else None,
    "passes_tests":       lambda r: bool(r.get("passes_tests")),
}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True, help="activations dir")
    ap.add_argument("--rows", required=True, help="self_reports.jsonl")
    ap.add_argument("--label", required=True, choices=list(LABELS))
    ap.add_argument("--pos", default="rep", choices=["sol", "rep"])
    ap.add_argument("--passing-only", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    acts = load_activations(args.acts, args.pos)
    rows = load_rows(args.rows)
    require = (lambda r: bool(r.get("passes_tests"))) if args.passing_only else None
    X, y, groups = build_xy(acts, rows, LABELS[args.label], require=require)
    if len(X) == 0:
        print("no samples"); return 1
    aur, n, npos = per_layer_auroc(X, y, k=args.k)
    print(f"label={args.label} pos={args.pos} n={n} pos={npos} neg={n-npos}")
    best = int(np.nanargmax(aur))
    print(f"best layer {best}: AUROC={aur[best]:.3f}   (chance=0.5)")
    # compact per-layer curve (every 4th layer)
    for l in range(0, len(aur), 4):
        bar = "#" * int(max(0, (aur[l] - 0.5)) * 40)
        print(f"  L{l:2d} {aur[l]:.3f} {bar}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
