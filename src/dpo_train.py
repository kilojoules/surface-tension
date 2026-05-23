"""Hand-rolled DPO training. No TRL, no vLLM — only torch + transformers + peft + bitsandbytes.

Per-pair SGD with reference-deltas precomputed once before training. Memory-efficient:
only one model resident during the training loop (the LoRA-augmented policy).

Loss:  L = -log sigmoid(beta * (policy_delta - ref_delta))
       where delta = logp(chosen | prompt) - logp(rejected | prompt)

ADAPTER_INIT support: when set (e.g. the B1++ SFT adapter), that adapter is loaded
as the trainable policy AND the reference deltas are computed against it *before* the
first optimizer step. So the DPO reference is anchored to the SFT checkpoint, not base
— the policy is regularized toward the loop-averse prior we already trained, while DPO
pushes it further toward compliant-over-violating preference.

Configurable via env:
  BASE_MODEL    default google/gemma-4-31B-it
  ADAPTER_INIT  starting/reference adapter (e.g. kilojoules/surface-tension-sft-b1plus-r32-final).
                If unset/"none", trains a fresh LoRA from base (reference = base).
  DPO_TRAIN     default ../data/dpo_pairs_train.jsonl  (lines: {prompt, chosen, rejected})
  DPO_OUTPUT    default ../outputs/dpo_run1
  DPO_BETA      default 0.1
  DPO_LR        default 5e-6
  DPO_ITERS     default 0   (0 = num_pairs * 3)
  DPO_EPOCHS    default 0   (alternative to DPO_ITERS: each epoch = one shuffled pass)
  LORA_RANK     default 32  (only used for the fresh-LoRA fallback)
  MAX_LENGTH    default 2048
  HUB_REPO      optional. If set, push final adapter to <user>/<repo>.
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from model_utils import BNB_CONFIG, completion_logprob, strip_clippable_linear_wrappers, unload_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_pairs(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _precompute_ref_deltas(model, tokenizer, pairs, max_length, log_every=25) -> List[torch.Tensor]:
    """Reference logp(chosen) - logp(rejected) per pair, frozen base, no grad."""
    model.eval()
    deltas: List[torch.Tensor] = []
    with torch.no_grad():
        for i, p in enumerate(pairs):
            lp_c = completion_logprob(model, tokenizer, p["prompt"], p["chosen"], max_length)
            lp_r = completion_logprob(model, tokenizer, p["prompt"], p["rejected"], max_length)
            deltas.append((lp_c - lp_r).detach())
            if (i + 1) % log_every == 0:
                print(f"  ref_logp {i+1}/{len(pairs)}")
    return deltas


def main():
    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
    train_path = os.environ.get("DPO_TRAIN", "../data/dpo_pairs_train.jsonl")
    output_dir = os.environ.get("DPO_OUTPUT", "../outputs/dpo_run1")
    beta = float(os.environ.get("DPO_BETA", "0.1"))
    lr = float(os.environ.get("DPO_LR", "5e-6"))
    iters_env = int(os.environ.get("DPO_ITERS", "0"))
    epochs_env = int(os.environ.get("DPO_EPOCHS", "0"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    max_length = int(os.environ.get("MAX_LENGTH", "2048"))
    hub_repo = (os.environ.get("HUB_REPO") or "").strip() or None
    adapter_init = (os.environ.get("ADAPTER_INIT") or "").strip()
    if adapter_init.lower() in ("", "none"):
        adapter_init = None

    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(train_path):
        train_path = os.path.join(here, train_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(here, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "final_adapter")

    print(f"base_model = {base_model}")
    print(f"train_path = {train_path}")
    print(f"output_dir = {output_dir}")
    print(f"beta={beta} lr={lr} lora_rank={lora_rank} max_length={max_length}")

    pairs = _load_pairs(train_path)
    if not pairs:
        print("no pairs loaded; aborting")
        return
    if epochs_env > 0:
        n_iters = len(pairs) * epochs_env
    else:
        n_iters = iters_env if iters_env > 0 else len(pairs) * 3
    print(f"loaded {len(pairs)} pairs; training for {n_iters} iters")
    print(f"adapter_init = {adapter_init or '(none — fresh LoRA from base)'}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Strip Gemma-4 wrappers BEFORE adapter load (B1++ was trained with STRIP_WRAPPERS=1;
    # the inference-time module structure must match).
    model = strip_clippable_linear_wrappers(model)
    model = prepare_model_for_kbit_training(model)

    if adapter_init:
        print(f"loading policy adapter from {adapter_init} (trainable)...")
        model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)
    else:
        print("attaching fresh LoRA adapter for policy...")
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Reference deltas: computed NOW, with the policy adapter active but before any
    # optimizer step → the DPO reference is exactly the init policy (B1++ if adapter_init
    # set, base otherwise). Frozen detached scalars; the training loop updates the same
    # adapter but ref_deltas stay fixed.
    print(f"computing reference deltas (reference = {adapter_init or 'base'})...")
    ref_deltas = _precompute_ref_deltas(model, tokenizer, pairs, max_length)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)

    losses, accs = [], []
    for step in range(n_iters):
        idx = rng.randint(0, len(pairs) - 1)
        p = pairs[idx]
        ref_delta = ref_deltas[idx]

        lp_c = completion_logprob(model, tokenizer, p["prompt"], p["chosen"], max_length)
        lp_r = completion_logprob(model, tokenizer, p["prompt"], p["rejected"], max_length)
        policy_delta = lp_c - lp_r

        logit = beta * (policy_delta - ref_delta)
        loss = -F.logsigmoid(logit)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        accs.append(float((logit > 0).item()))

        if (step + 1) % 10 == 0:
            avg_l = sum(losses[-10:]) / min(10, len(losses))
            avg_a = sum(accs[-10:]) / min(10, len(accs))
            print(f"step {step+1}/{n_iters} loss={avg_l:.4f} acc={avg_a:.1%}")

        if (step + 1) % 100 == 0:
            ckpt = os.path.join(output_dir, f"checkpoint-{step+1}")
            model.save_pretrained(ckpt)

    model.save_pretrained(adapter_dir)
    print(f"saved adapter to {adapter_dir}")

    if hub_repo:
        try:
            model.push_to_hub(hub_repo, private=True)
            print(f"pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"hub push failed (non-fatal): {e}")

    unload_model(model, optimizer)


if __name__ == "__main__":
    main()
