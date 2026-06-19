"""Residual-stream activation capture at the prose-end token.

After a response is generated, this module runs ONE extra forward pass on
(prompt + response-through-prose-end-token) with forward-hooks on each
transformer block. It saves a per-sample tensor of shape (n_layers,
hidden_dim) — the residual stream at the last prose token before the
first code fence, at every layer. That's the position the probe reads.

Used by sweep_local.py when CAPTURE_ACTIVATIONS=1 (env var).
"""
from __future__ import annotations

import os

import torch


def find_prose_end_token_index(response_text: str, tokenizer) -> int | None:
    """Return the response-token index of the last prose token before the
    first ``` code fence. None if no code fence (no boundary to probe)."""
    fence_at_char = response_text.find("```")
    if fence_at_char < 0:
        return None
    enc = tokenizer(
        response_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]
    for i, (start, _end) in enumerate(offsets):
        if start >= fence_at_char:
            return max(0, i - 1)
    return len(offsets) - 1 if offsets else None


def _get_decoder_layers(model):
    """Find the list of transformer blocks. Robust to PEFT wrapping."""
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.transformer.h,
    ]
    for c in candidates:
        try:
            layers = c(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise RuntimeError(
        "Could not locate decoder layers on the model. Tried "
        ".model.layers, .base_model.model.model.layers, .transformer.h"
    )


def capture_residuals(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda",
) -> torch.Tensor | None:
    """Run one extra forward over (prompt + response_through_prose_end) and
    return a (n_layers, hidden_dim) tensor on CPU in float16, or None if no
    prose/code boundary was detected.
    """
    prose_end_idx = find_prose_end_token_index(response, tokenizer)
    if prose_end_idx is None:
        return None

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    full = prompt_ids + resp_ids[: prose_end_idx + 1]
    if not full:
        return None
    capture_pos = len(full) - 1

    layers = _get_decoder_layers(model)
    captures: list[torch.Tensor] = []
    handles = []

    def make_hook(_layer_idx):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # hidden shape: (batch, seq, hidden_dim)
            captures.append(hidden[0, capture_pos, :].detach().to("cpu").half())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            input_ids = torch.tensor([full], device=device)
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if not captures:
        return None
    return torch.stack(captures)  # (n_layers, hidden_dim) on CPU, float16


def save_activation(source_dir: str, row, activation: torch.Tensor) -> str:
    """Save activation tensor alongside *.py / *__raw.txt files using the
    existing per-sample naming convention."""
    os.makedirs(source_dir, exist_ok=True)
    safe = row.problem_id.replace("/", "_")
    fname = f"{safe}__{row.constraint}__{row.condition}__s{row.sample_idx}__resid.pt"
    path = os.path.join(source_dir, fname)
    torch.save(activation, path)
    return path
