#!/bin/bash
# Stage 2 prep: DPO fine-tune Gemma 4 31B against the no_loops+no_recursion preference,
# without ever showing the constraint in the prompt. After training, the model should
# *latently* prefer compliant code on bare problem statements.
#
# Designed to run on a single H100 80GB vast.ai box that's also serving vLLM locally.
# Phases: (1) expand pilot data on all 124 LCB-medium-post-cutoff problems,
#         (2) build DPO dataset, (3) run DPO training, (4) eval the trained adapter.
#
# Phases 1 and 4 hit the local vLLM endpoint (the same box, port 8000).
# Phase 3 trains via QLoRA + TRL on the same GPU.
#
# Cost estimate at $1.89/hr (H100 80GB):
#   Phase 1 (vLLM inference, ~1h):   ~$2
#   Phase 2 (CPU only, <1 min):       <$0.01
#   Phase 3 (QLoRA DPO, ~6-8h):       ~$12-15
#   Phase 4 (vLLM inference, ~1h):    ~$2
#   Setup, transitions, buffer:       ~$5-10
#   TOTAL:                            ~$25-35
#
# Stop the box yourself when phase 4 finishes; this script does not destroy the instance.

set -e
cd "$(dirname "$0")/src"

# All required env vars must be set up-stream
: "${MODEL_ENDPOINT:?MODEL_ENDPOINT not set (e.g. http://localhost:8000/v1)}"
: "${MODEL_API_KEY:?MODEL_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

# ---- Phase 0: install training deps up front so flash-attn build failures fail fast ----
# (flash-attn compiles from source; failure here is recoverable in <1min vs 1h into the run.)
echo "[phase 0/4] installing training deps"
pip install -q -r requirements_dpo.txt

# ---- Phase 1: expand pilot data to all 124 LCB-medium post-cutoff problems ----
echo "[phase 1/4] expanding pilot data"
python loaders_lcb.py --n 0  # 0 = take all matching

RUNNER=http \
MODEL_NAME="google/gemma-4-31B-it" \
MODEL_MAX_TOKENS=2048 \
python sweep.py \
  --problems ../data/problems_lcb.jsonl \
  --csv ../results/raw/pilot_v5_raw.csv \
  --source-dir ../results/raw/sources_v5 \
  --workers 8 \
  --n-samples 6 \
  --constraints no_loops_no_recursion

# ---- Phase 2: build DPO triples with problem-level 70/30 split ----
echo "[phase 2/4] building DPO dataset"
python build_dpo_dataset.py \
  --csv ../results/raw/pilot_v5_raw.csv \
  --sources-dir ../results/raw/sources_v5 \
  --constraint no_loops_no_recursion \
  --holdout-frac 0.3

# ---- Phase 3: DPO training ----
echo "[phase 3/4] DPO training (this is the long one)"
HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" python dpo_train.py

# ---- Phase 4: eval the trained adapter ----
# This requires vLLM to be re-launched with --enable-lora pointing at outputs/dpo_run1/final_adapter.
# Re-launch outside this script:
#   vllm serve google/gemma-4-31B-it ... --enable-lora --lora-modules dpo=./outputs/dpo_run1/final_adapter
# Then:
echo "[phase 4/4] full eval — re-launch vLLM with --enable-lora --lora-modules dpo=./outputs/dpo_run1/final_adapter, then:"
echo
echo "    RUNNER=http MODEL_NAME=dpo \\"
echo "    python sweep.py \\"
echo "      --problems ../data/problems_lcb.jsonl \\"
echo "      --csv ../results/raw/pilot_v5_dpo_raw.csv \\"
echo "      --source-dir ../results/raw/sources_v5_dpo \\"
echo "      --workers 8 \\"
echo "      --constraints no_loops_no_recursion"
echo
echo "    python aggregate.py \\"
echo "      --csv ../results/raw/pilot_v5_dpo_raw.csv \\"
echo "      --summary-csv ../results/pilot_v5_dpo_summary.csv \\"
echo "      --summary-md ../results/pilot_v5_dpo_summary.md \\"
echo "      --sources-dir ../results/raw/sources_v5_dpo"
echo
echo "Headline question: did pass_unconstrained's compliance_rate jump from ~0.03 to >0.80"
echo "without pass_overall collapsing? If yes, stage 1 of PLAN.md is done."
