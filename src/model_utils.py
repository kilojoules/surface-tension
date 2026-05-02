"""Lean model utilities — adapted from turnstile/turnstile/model_utils.py.

Stripped to the primitives surface_tension actually needs:
  - 4-bit quantization config
  - load/unload helpers
  - generate_text (transformers.generate, no vLLM)
  - completion_logprob (sum logp of completion tokens given prompt — for DPO)

Deliberately does NOT depend on TRL or vLLM — both have caused brittle version
conflicts on fresh CUDA boxes. transformers + peft + bitsandbytes is enough.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def strip_clippable_linear_wrappers(model):
    """Replace Gemma4ClippableLinear with its inner Linear4bit so PEFT can hook it.

    Gemma 4 wraps every linear in a custom Gemma4ClippableLinear (an activation-clipping
    wrapper). PEFT's LoRA injector only recognizes a hardcoded list of base linear types
    and refuses to patch the custom wrapper, so we expose the inner Linear4bit directly.
    The clipping is removed at inference/training time. At bf16/4-bit the impact is small;
    flag if comparing to published Gemma 4 numbers.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "Gemma4ClippableLinear":
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            inner = getattr(module, "linear", None)
            if inner is None:
                continue
            setattr(parent, attr, inner)
            replaced += 1
    if replaced:
        print(f"  stripped {replaced} Gemma4ClippableLinear wrappers")
    return model


def load_model(model_id: str, adapter_path: Optional[str] = None, dtype=torch.bfloat16):
    """Load a 4-bit-quantized causal LM and tokenizer. Optionally attach a LoRA adapter.

    Wrapper-stripping rule:
      - If NO adapter: strip Gemma4ClippableLinear so any subsequent PEFT injection can hook
        the inner Linear4bit. Used by old DPO path that did get_peft_model post-load.
      - If LOADING an adapter (sft_train.py path): the adapter was saved with target_modules
        like `.*\\.q_proj\\.linear$`, i.e. trained against the inner `.linear` of the wrapper.
        Stripping the wrapper would remove the `.linear` path components and PEFT would fail
        to find the adapter's targets. Keep the wrapper intact for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=dtype,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Only useful for the legacy path; harmless on non-Gemma-4 models (no-op).
        model = strip_clippable_linear_wrappers(model)
    return model, tokenizer


def unload_model(*objs):
    """Release model/optimizer references and clear CUDA cache."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 1024,
                  temperature: float = 0.7, top_p: float = 0.95) -> str:
    """Single-prompt completion via transformers.generate. Returns just the new text.

    Wraps the prompt in the tokenizer's chat template (with `add_generation_prompt=True`)
    so instruction-tuned models actually enter response mode. Without the template,
    instruction-tuned models like Gemma 4 31B-it produce degenerate output ~95% of
    the time — vLLM auto-applies it, transformers.generate doesn't.
    """
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt  # base/non-instruct models: pass through

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def completion_logprob(model, tokenizer, prompt: str, completion: str,
                       max_length: int = 2048) -> torch.Tensor:
    """Sum of log-probs of completion tokens conditioned on prompt.

    Single forward pass over the concatenation [chat-templated prompt, completion];
    sum log-probs only over the completion span. Returns a scalar tensor with grad
    enabled iff model.training.

    The prompt is wrapped in the tokenizer's chat template so the train-time and
    inference-time prompt formats match (instruction-tuned models silently fail
    when training-prompt and inference-prompt formats diverge).

    Truncates from the front if the joint length exceeds max_length so the
    completion stays intact (the loss would otherwise see a misaligned span).
    """
    if getattr(tokenizer, "chat_template", None):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt
    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids

    # Front-truncate the prompt if joint > max_length
    over = (len(prompt_ids) + len(completion_ids)) - max_length
    if over > 0:
        prompt_ids = prompt_ids[over:]

    full = (prompt_ids + completion_ids)[-max_length:]
    if not full or len(completion_ids) == 0:
        return torch.tensor(0.0, device=model.device)

    completion_start = len(prompt_ids)
    input_ids = torch.tensor(full, device=model.device).unsqueeze(0)

    enable_grad = model.training
    ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=input_ids).logits[0]  # [L, V]

    # logits[t] predicts token at position t+1. Sum log-probs over completion span:
    #   for t in [completion_start-1, len(full)-1):  log P(full[t+1] | logits[t])
    log_probs = F.log_softmax(logits, dim=-1)
    target_positions = torch.arange(completion_start - 1, len(full) - 1, device=model.device)
    target_tokens = torch.tensor(full[completion_start:], device=model.device)
    selected = log_probs[target_positions, target_tokens]
    return selected.sum()
