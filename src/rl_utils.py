"""Helpers for on-policy RL: rollout sampling, reward computation, GRPO advantages."""

from __future__ import annotations

import ast as _ast
import os
from typing import Callable, Dict, List, Optional

import torch

from ast_checks import CHECKS
from evaluator import evaluate, evaluate_stdin
from sweep_local import extract_code


def sample_rollouts(model, tokenizer, prompt: str, n_rollouts: int,
                    max_new_tokens: int = 1024, temperature: float = 0.9,
                    top_p: float = 0.95) -> List[str]:
    """Generate n_rollouts completions for the given prompt. Sequential (one at a time)
    for memory safety on a single 40GB A100. Each rollout uses fresh sampling state.

    Returns the raw decoded text (post-chat-template, just the new tokens).
    """
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)

    completions: List[str] = []
    model.eval()
    for _ in range(n_rollouts):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return completions


def compute_reward(raw_completion: str, problem: Dict, constraint: str = "no_loops_no_recursion",
                   test_timeout_s: float = 10.0) -> Dict:
    """Verifiable reward. Mode selected by env REWARD_MODE:

      binary (default):
        reward = 1 iff (extracted code parses ∧ has entry ∧ AST-compliant ∧ tests pass), else 0.

      multitier:
        +2  compliant ∧ passes tests
        +1  compliant ∧ substantive ∧ tests fail   (good-faith loop-free attempt)
         0  non-substantive (no code / parse fail / no entry / very short)
         0  non-compliant ∧ tests fail              (irrelevant wrong-with-loops)
        -1  non-compliant ∧ tests pass              (cheating — used loops AND correct)

      Substantive = parses ∧ has_entry ∧ code_chars ≥ 100.

    Returns a dict with the reward plus diagnostic flags so the trainer can log
    distribution stats (parse rate, compliance rate, pass rate separately).
    """
    diag = {"reward": 0, "extracted": 0, "parses": 0, "has_entry": 0,
            "compliant": 0, "test_passed": 0, "code_chars": 0, "substantive": 0}

    code = extract_code(raw_completion)
    if not code:
        return diag
    diag["extracted"] = 1
    diag["code_chars"] = len(code)

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return diag
    diag["parses"] = 1

    mode = problem.get("mode", "function")
    entry_point = problem.get("entry_point") or ""

    if mode == "function":
        has_entry = any(
            isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and n.name == entry_point
            for n in _ast.walk(tree)
        )
        diag["has_entry"] = 1 if has_entry else 0
        if not has_entry:
            return diag
    else:
        diag["has_entry"] = 1

    # Constraint check (AST)
    try:
        compliant = CHECKS[constraint](code)
    except Exception:
        compliant = False
    diag["compliant"] = 1 if compliant else 0

    # Test execution
    try:
        if mode == "stdin":
            res = evaluate_stdin(code, problem["stdin_tests"], timeout_s=test_timeout_s)
        else:
            res = evaluate(code, entry_point, problem["test_runner"], timeout_s=test_timeout_s)
        diag["test_passed"] = 1 if res.passed else 0
    except Exception:
        diag["test_passed"] = 0

    # Substantive: parses + has_entry + code_chars >= 100
    diag["substantive"] = 1 if (diag["parses"] and diag["has_entry"] and diag["code_chars"] >= 100) else 0

    mode_str = os.environ.get("REWARD_MODE", "binary")
    if mode_str == "multitier":
        c = diag["compliant"]; p = diag["test_passed"]; s = diag["substantive"]
        if c and p:
            diag["reward"] = 2
        elif c and not p and s:
            diag["reward"] = 1
        elif (not c) and p:
            diag["reward"] = -1
        else:
            diag["reward"] = 0
    else:
        diag["reward"] = 1 if (diag["compliant"] and diag["test_passed"]) else 0
    return diag


def compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """GRPO group-relative advantage: (r - mean) / (std + eps).
    Returns zeros when group is uniform (no learning signal from this prompt).
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + eps) for r in rewards]
