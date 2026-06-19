"""Unit tests for the activation-capture helpers.

The end-to-end `capture_residuals` requires a real model + CUDA, so we cover
the deterministic pieces here: prose-end token detection and the file-naming
convention. These run on CPU in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from activation_capture import find_prose_end_token_index, save_activation


class _FakeTok:
    """Whitespace tokenizer with char offsets — enough to exercise the
    boundary logic without depending on transformers."""
    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        tokens = []
        offsets = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            tokens.append(text[start:i])
            offsets.append((start, i))
        out = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_no_code_fence_returns_none():
    tok = _FakeTok()
    assert find_prose_end_token_index("plain prose with no code", tok) is None


def test_finds_last_token_before_fence():
    tok = _FakeTok()
    # prose: "I will use map" then "```python..."
    text = "I will use map ```python\nprint(1)\n```"
    idx = find_prose_end_token_index(text, tok)
    # tokens: ['I', 'will', 'use', 'map', '```python', 'print(1)', '```']
    # ``` first appears at char 15 (where '```python' starts)
    # last prose token is 'map' at index 3
    assert idx == 3


def test_fence_at_start_returns_zero_or_none():
    tok = _FakeTok()
    text = "```python\nprint(1)\n```"
    idx = find_prose_end_token_index(text, tok)
    # No prose at all — first token IS at the fence position
    # max(0, 0-1) = 0  — acceptable (downstream gets a noisy capture)
    # But the function might also return None. Either is OK; just check
    # it doesn't raise.
    assert idx is None or idx == 0


def test_response_with_only_inline_comments_still_captures():
    tok = _FakeTok()
    text = "```python\n# I avoid loops here\nprint(1)\n```"
    idx = find_prose_end_token_index(text, tok)
    # First ``` is at char 0; no prose before it.
    assert idx is None or idx == 0


@dataclass
class _Row:
    problem_id: str
    constraint: str
    condition: str
    sample_idx: int


def test_save_filename_convention(tmp_path):
    import torch
    row = _Row("lcb/abc356_c", "none", "unconstrained", 3)
    t = torch.zeros((50, 4096), dtype=torch.float16)
    path = save_activation(str(tmp_path), row, t)
    expected = tmp_path / "lcb_abc356_c__none__unconstrained__s3__resid.pt"
    assert str(expected) == path
    assert expected.exists()
    loaded = torch.load(path)
    assert loaded.shape == (50, 4096)
    assert loaded.dtype == torch.float16
