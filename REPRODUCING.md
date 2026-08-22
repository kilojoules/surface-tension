# Reproducing every reported number (no GPU required)

All statistics in `results/*.md` re-derive from published artifacts. The
raw-evidence corpus lives in `data/evidence/` (sha256-manifested packages)
with a full mirror + pod logs at
https://huggingface.co/datasets/kilojoules/surface-tension-evidence.

## Tier 1 — re-derive reported statistics (laptop, minutes)

```bash
# AST compliance / cheating for any eval cell (CSV + sources ship together):
PYTHONPATH=src python src/recheck_eval.py \
    data/evidence/step0_v1/st-step0_yjljmvip0iz88b/eval_step0_prefill.csv
# scaling grid / rank sweep cells: same command over data/evidence/scaling_v1/**.csv
# DPO pass round: data/evidence/dpo_pass_v1/**.csv

# Propensity panel (gunzip the score records first):
gunzip -k data/evidence/propensity_v1/tynlsatwprmdst/*.jsonl.gz
python src/analyze_propensity.py \
    --scores-dir data/evidence/propensity_v1/tynlsatwprmdst --out /tmp/panel.json
python src/analyze_interrupt.py \
    --scores-dir data/evidence/propensity_v1/tynlsatwprmdst --out /tmp/interrupt.json

# Stage-3 corrected statistics:
PYTHONPATH=src python -m quadrant.probe_correction \
    --evidence data/evidence/quadrant_v4 --out /tmp/correction

# Frozen datasets rebuild byte-identically (check against committed manifests):
python scripts/make_sft_scaling_sets.py
python scripts/make_propensity_freeze.py

# Figures re-render from committed numbers (provenance in each script header):
python src/plot_sft_scaling.py && python src/plot_step0_findings.py
```

The evaluation deck (`data/problems_lcb_clean17.jsonl`) is committed with
attribution (`data/DECKS.md`); training corpora are committed
(`data/sft_*.jsonl`, `data/sft_scaling/`); frozen propensity items are in
`data/propensity/frozen/`.

## Tier 2 — re-run experiments (GPU, ~$1.5/h A100)

Recipes and launchers: `scripts/launch_*.sh` (exact hyperparameters
in-file); adapters on the Hub (`kilojoules/surface-tension-*`); preregs
with external timestamp anchors in `prereg/`. Historical eval sampling was
unseeded (T=0.7) — expect statistical, not bitwise, reproduction; later
runs (battery/interruption) are per-record seeded. The base model revision
is unpinned (`google/gemma-4-31B-it@main`).

## Known irrecoverables

- Stage-3 judge raw (`judgments.jsonl`) was lost with its pod
  (documented in `docs/correction_2026-07-12.md`); claim-side numbers are
  single-sourced and bounded accordingly.
- `pip freeze` exists only for the propensity pod (in the HF logs mirror);
  earlier pods' environments are specified by their launchers' pinned
  install lines only.
