"""Forward-pass per-pair loss for the v8 SFT adapter on the 146 training pairs.

Loads base + adapter, computes token-CE on the completion span only for each
pair in sft_all.jsonl, writes problem_id, n_tokens, mean_loss to a CSV.

Usage:
  ADAPTER=kilojoules/surface-tension-sft \
  PAIRS=../data/sft_all.jsonl \
  OUT=../results/raw/per_pair_loss_v8.csv \
  python3 per_pair_loss.py
"""
import csv
import json
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import BNB_CONFIG

BASE = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
ADAPTER = os.environ.get("ADAPTER", "kilojoules/surface-tension-sft")
PAIRS = os.environ.get("PAIRS", "../data/sft_all.jsonl")
OUT = os.environ.get("OUT", "../results/raw/per_pair_loss_v8.csv")

print(f"base={BASE} adapter={ADAPTER} pairs={PAIRS}")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=BNB_CONFIG, device_map="auto")
if ADAPTER and ADAPTER != "none":
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

pairs = [json.loads(l) for l in open(PAIRS) if l.strip()]
print(f"loaded {len(pairs)} pairs")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "problem_id", "n_completion_tokens", "mean_loss"])

    for i, ex in enumerate(pairs):
        prompt = ex["prompt"]
        completion = ex["completion"]
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        comp_ids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)

        with torch.no_grad():
            out = model(input_ids=full_ids)
            logits = out.logits[:, :-1, :]
            targets = full_ids[:, 1:]

        # Mask: only score completion tokens
        prompt_len = prompt_ids.shape[1]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, prompt_len - 1:] = True
        n = mask.sum().item()
        if n == 0:
            mean_loss = float("nan")
        else:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            tgt_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            losses = -tgt_logp[mask]
            mean_loss = losses.mean().item()

        w.writerow([i, ex["problem_id"], n, f"{mean_loss:.6f}"])
        if (i + 1) % 10 == 0:
            f.flush()
            print(f"  [{i+1}/{len(pairs)}] {ex['problem_id']} loss={mean_loss:.4f}")

print(f"wrote {OUT}")
