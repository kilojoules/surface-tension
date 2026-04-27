"""DPO fine-tune Gemma 4 31B-it on (prompt, chosen, rejected) pairs via QLoRA + TRL.

Designed to run on a single H100 80GB. 31B base in 4-bit nf4 + LoRA adapters fits
comfortably with seq 2048, batch 1 + grad-accum 8.

Usage:
  pip install -r requirements_dpo.txt
  HF_TOKEN=$(cat ~/.hf_token) python dpo_train.py
Configurable via env:
  BASE_MODEL       (default google/gemma-4-31B-it)
  DPO_TRAIN        (default ../data/dpo_pairs_train.jsonl)
  DPO_OUTPUT       (default ../outputs/dpo_run1)
  DPO_BETA         (default 0.1; lower → less constrained policy)
  DPO_LR           (default 5e-7)
  DPO_EPOCHS       (default 1)
  LORA_RANK        (default 32)
  MAX_LENGTH       (default 2048)
"""

from __future__ import annotations

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


def main():
    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
    train_path = os.environ.get("DPO_TRAIN", "../data/dpo_pairs_train.jsonl")
    eval_path = os.environ.get("DPO_EVAL", "../data/dpo_pairs_eval.jsonl")
    output_dir = os.environ.get("DPO_OUTPUT", "../outputs/dpo_run1")
    beta = float(os.environ.get("DPO_BETA", "0.1"))
    lr = float(os.environ.get("DPO_LR", "5e-7"))
    epochs = float(os.environ.get("DPO_EPOCHS", "1"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    max_length = int(os.environ.get("MAX_LENGTH", "2048"))
    max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "1024"))
    save_steps = int(os.environ.get("SAVE_STEPS", "25"))
    # With ~1000 pairs and effective batch 8, we have ~125 steps/epoch — warmup_steps=100
    # would never complete. Use a ratio instead.
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.05"))

    here = os.path.dirname(os.path.abspath(__file__))
    train_path = train_path if os.path.isabs(train_path) else os.path.join(here, train_path)
    eval_path = eval_path if os.path.isabs(eval_path) else os.path.join(here, eval_path)
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(here, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base model:   {base_model}")
    print(f"Train data:   {train_path}")
    print(f"Eval data:    {eval_path}")
    print(f"Output:       {output_dir}")
    print(f"beta={beta}  lr={lr}  epochs={epochs}  lora_rank={lora_rank}  max_length={max_length}")

    # 4-bit base for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash-attention; fall back to sdpa if flash-attn isn't installed/compatible.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError) as e:
        print(f"flash_attention_2 unavailable ({e}); falling back to sdpa.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = load_dataset("json", data_files=train_path, split="train")
    eval_ds = load_dataset("json", data_files=eval_path, split="train") if os.path.exists(eval_path) else None
    print(f"Loaded {len(train_ds)} train pairs, {len(eval_ds) if eval_ds else 0} eval pairs.")

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=10,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=save_steps if eval_ds else None,
        bf16=True,
        optim="paged_adamw_8bit",
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        remove_unused_columns=False,
        report_to=[],
        seed=0,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Persist the training config for reproducibility
    with open(os.path.join(output_dir, "dpo_run_config.json"), "w") as f:
        json.dump({
            "base_model": base_model,
            "beta": beta,
            "lr": lr,
            "epochs": epochs,
            "lora_rank": lora_rank,
            "max_length": max_length,
            "save_steps": save_steps,
        }, f, indent=2)

    trainer.train()
    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)
    print(f"Saved final adapter to {final_dir}")


if __name__ == "__main__":
    main()
