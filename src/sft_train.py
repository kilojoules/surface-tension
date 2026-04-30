"""Hand-rolled SFT distillation. Pure positive-signal training: token-CE over
the completion span only (prompt tokens are masked from the loss).

Designed to NOT collapse the way DPO did:
  - No reference model, no log-ratio loss, no preference contrast → no policy drift.
  - Bounded token-CE loss → no runaway gradients.
  - Standard LoRA recipe with conservative knobs.

Wrapper handling: targets the INNER `.linear` of Gemma4ClippableLinear via regex
target_modules. This keeps the wrapper's activation-clipping in the forward pass
(load-bearing on Gemma 4 31B at 4-bit; v7 stripped it and the model collapsed).

Configurable via env:
  BASE_MODEL          default google/gemma-4-31B-it
  SFT_TRAIN           default ../data/sft_train.jsonl
  SFT_EVAL            default ../data/sft_eval.jsonl
  SFT_OUTPUT          default ../outputs/sft_run1
  SFT_LR              default 1e-5
  SFT_EPOCHS          default 3
  LORA_RANK           default 32
  LORA_ALPHA          default 16          (alpha = r/2; v7's 2×r was destabilizing)
  LORA_DROPOUT        default 0.05
  MAX_LENGTH          default 1024
  MAX_PROMPT_LENGTH   default 768
  GRAD_ACCUM          default 8
  GRAD_CLIP           default 1.0
  WARMUP_RATIO        default 0.1
  HUB_REPO            optional. Pushes adapter at end if set.
  EVAL_EVERY          default 50          (steps between mid-train evals; 0 disables)
  EVAL_N              default 8           (problems sampled per mid-eval)
  REPETITION_THRESH   default 0.30        (max share of any 4-gram in completion)
  ABORT_PASS_FLOOR    default 0.30        (mini-eval pass below this at step 50+ → abort)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from ast_checks import CHECKS
from evaluator import evaluate, evaluate_stdin
from loaders import load_problems_jsonl
from model_utils import BNB_CONFIG, completion_logprob, unload_model
from sweep_local import build_prompt, extract_code


# Regex match for Gemma 4: hooks the inner `Linear4bit` while leaving Gemma4ClippableLinear
# in the forward pass. Matches the equivalent paths on standard models too (q_proj.linear
# doesn't exist on Gemma 3, so it falls back to the bare-name targets below).
GEMMA4_LORA_REGEX = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear$"
STANDARD_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_examples(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _has_clippable_linear(model) -> bool:
    for m in model.modules():
        if m.__class__.__name__ == "Gemma4ClippableLinear":
            return True
    return False


def _attach_lora(model, rank: int, alpha: int, dropout: float):
    """Pick regex or list target_modules depending on whether the model has Gemma4ClippableLinear."""
    targets = GEMMA4_LORA_REGEX if _has_clippable_linear(model) else STANDARD_LORA_TARGETS
    print(f"  lora target_modules: {'regex (Gemma 4 inner-linear)' if isinstance(targets, str) else 'list (standard)'}")
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def _completion_loss(model, tokenizer, prompt: str, completion: str, max_length: int, max_prompt_length: int) -> torch.Tensor:
    """Negative mean log-prob over completion tokens. Loss is zero only when the model
    assigns probability 1 to every completion token; non-negative everywhere."""
    # Front-truncate the prompt to keep room for the completion
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length:]
    full = (prompt_ids + completion_ids)[-max_length:]
    if len(full) == len(completion_ids):
        # Completion overran budget; nothing to condition on. Treat as no-op.
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # Re-derive completion span post-truncation
    completion_start = len(full) - len(completion_ids[-(max_length - 1):])
    if completion_start <= 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    input_ids = torch.tensor(full, device=model.device).unsqueeze(0)
    out = model(input_ids=input_ids)
    logits = out.logits[0]

    # logits[t] predicts token at position t+1
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_positions = torch.arange(completion_start - 1, len(full) - 1, device=model.device)
    target_tokens = torch.tensor(full[completion_start:], device=model.device)
    selected = log_probs[target_positions, target_tokens]
    n = max(1, len(target_tokens))
    return -selected.sum() / n


def _ngram_top_share(text: str, n: int = 4) -> float:
    """Heuristic: max share of any n-token n-gram in the text. >0.30 = degenerate repetition."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _mini_eval(model, tokenizer, problems: List[dict], constraint: str, n_per_problem: int = 1, max_new: int = 1024) -> Dict[str, float]:
    """Bare-prompt sample on a few held-out problems. Returns compliance, pass, repetition stats."""
    check_compliance = CHECKS[constraint]
    n_total = 0
    n_compliant = 0
    n_pass = 0
    n_pass_and_compliant = 0
    n_repetition = 0
    n_kept = 0  # parses + has entry / stdin-mode-ok
    model.eval()
    for p in problems:
        prompt = build_prompt(p, constraint=None)
        for _ in range(n_per_problem):
            n_total += 1
            try:
                raw = _generate(model, tokenizer, prompt, max_new_tokens=max_new, temperature=0.7)
            except Exception:
                continue
            if _ngram_top_share(raw) > 0.30:
                n_repetition += 1
                continue
            code = extract_code(raw)
            if not code:
                continue
            try:
                import ast as _ast
                _ast.parse(code)
            except SyntaxError:
                continue
            n_kept += 1
            compliant = check_compliance(code)
            if compliant:
                n_compliant += 1
            try:
                if p.get("mode") == "stdin":
                    res = evaluate_stdin(code, p["stdin_tests"], timeout_s=10.0)
                else:
                    res = evaluate(code, p["entry_point"], p["test_runner"], timeout_s=10.0)
                if res.passed:
                    n_pass += 1
                    if compliant:
                        n_pass_and_compliant += 1
            except Exception:
                pass
    model.train()
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "compliance": n_compliant / max(1, n_kept),
        "pass": n_pass / max(1, n_kept),
        "pass_and_compliant_overall": n_pass_and_compliant / max(1, n_total),
        "repetition_rate": n_repetition / max(1, n_total),
    }


def main():
    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
    train_path = os.environ.get("SFT_TRAIN", "../data/sft_train.jsonl")
    eval_path = os.environ.get("SFT_EVAL", "../data/sft_eval.jsonl")
    output_dir = os.environ.get("SFT_OUTPUT", "../outputs/sft_run1")
    lr = float(os.environ.get("SFT_LR", "1e-5"))
    epochs = float(os.environ.get("SFT_EPOCHS", "3"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_rank // 2)))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    max_length = int(os.environ.get("MAX_LENGTH", "1024"))
    max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", "768"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.1"))
    eval_every = int(os.environ.get("EVAL_EVERY", "50"))
    eval_n = int(os.environ.get("EVAL_N", "8"))
    abort_pass_floor = float(os.environ.get("ABORT_PASS_FLOOR", "0.30"))
    repetition_thresh = float(os.environ.get("REPETITION_THRESH", "0.10"))
    constraint = os.environ.get("CONSTRAINT", "no_loops_no_recursion")
    hub_repo = (os.environ.get("HUB_REPO") or "").strip() or None
    problems_path = os.environ.get("PROBLEMS_PATH", "../data/problems_lcb.jsonl")
    seed = int(os.environ.get("SEED", "0"))

    here = os.path.dirname(os.path.abspath(__file__))
    def absify(p):
        return p if os.path.isabs(p) else os.path.join(here, p)
    train_path = absify(train_path); eval_path = absify(eval_path)
    output_dir = absify(output_dir); problems_path = absify(problems_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"base_model = {base_model}")
    print(f"train      = {train_path}")
    print(f"output_dir = {output_dir}")
    print(f"lr={lr} epochs={epochs} r={lora_rank} alpha={lora_alpha} dropout={lora_dropout}")
    print(f"max_length={max_length} grad_accum={grad_accum} warmup_ratio={warmup_ratio}")
    print(f"eval_every={eval_every} eval_n={eval_n} abort_pass_floor={abort_pass_floor}")

    examples = _load_examples(train_path)
    print(f"loaded {len(examples)} train examples")
    if not examples:
        print("no examples; aborting")
        return

    # Held-out problems for mid-eval (problem-level holdout from build_sft_dataset)
    eval_problems_all = []
    if os.path.exists(eval_path):
        eval_pids = {json.loads(line)["problem_id"] for line in open(eval_path)}
        all_problems = load_problems_jsonl(problems_path)
        eval_problems_all = [p for p in all_problems if p["id"] in eval_pids]
    rng = random.Random(seed)
    rng.shuffle(eval_problems_all)
    mini_eval_problems = eval_problems_all[:eval_n] if eval_problems_all else []
    print(f"mini-eval set: {len(mini_eval_problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"  has Gemma4ClippableLinear: {_has_clippable_linear(model)}")

    model = prepare_model_for_kbit_training(model)
    model = _attach_lora(model, lora_rank, lora_alpha, lora_dropout)
    model.print_trainable_parameters()

    n_examples = len(examples)
    total_steps = math.ceil(n_examples * epochs / grad_accum)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    print(f"  total_steps={total_steps} warmup={warmup_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    model.train()
    rng2 = random.Random(seed)
    losses: List[float] = []
    history = []
    start = time.time()
    micro_step = 0
    optimizer.zero_grad()

    for opt_step in range(total_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(opt_step)

        # accumulate grad_accum micro-batches
        accum_loss = 0.0
        for _ in range(grad_accum):
            ex = examples[rng2.randint(0, n_examples - 1)]
            loss = _completion_loss(model, tokenizer, ex["prompt"], ex["completion"], max_length, max_prompt_length)
            (loss / grad_accum).backward()
            accum_loss += loss.item()
            micro_step += 1
        accum_loss /= grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(accum_loss)

        if (opt_step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses))
            print(f"  step {opt_step+1}/{total_steps}  loss={avg:.4f}  lr={lr_at(opt_step):.2e}  ({time.time()-start:.0f}s)")

        # ---- mid-training eval + abort rules ----
        if eval_every > 0 and (opt_step + 1) % eval_every == 0 and mini_eval_problems:
            print(f"  [mini-eval @ step {opt_step+1}]")
            stats = _mini_eval(model, tokenizer, mini_eval_problems, constraint, n_per_problem=1)
            stats["step"] = opt_step + 1
            history.append(stats)
            print(f"    {stats}")
            with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
                f.write(json.dumps(stats) + "\n")

            # abort rules
            if stats["repetition_rate"] > repetition_thresh:
                print(f"  ABORT: repetition_rate {stats['repetition_rate']:.2f} > {repetition_thresh}")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)
            if opt_step + 1 >= 50 and stats["pass"] < abort_pass_floor:
                print(f"  ABORT: pass {stats['pass']:.2f} < floor {abort_pass_floor}")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)
            if accum_loss < 0.05 and opt_step + 1 < 50:
                print(f"  ABORT: loss {accum_loss:.3f} < 0.05 too early — likely memorizing")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)

        if (opt_step + 1) % 100 == 0:
            ckpt = os.path.join(output_dir, f"checkpoint-{opt_step+1}")
            model.save_pretrained(ckpt)

    # ---- save final adapter ----
    final = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final)
    print(f"saved final adapter to {final}")

    # final mini-eval
    if mini_eval_problems:
        final_stats = _mini_eval(model, tokenizer, mini_eval_problems, constraint, n_per_problem=2)
        final_stats["step"] = total_steps
        final_stats["final"] = True
        history.append(final_stats)
        print(f"final mini-eval: {final_stats}")
        with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
            f.write(json.dumps(final_stats) + "\n")

    # push to hub
    if hub_repo:
        try:
            model.push_to_hub(hub_repo, private=True)
            print(f"pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"hub push failed (non-fatal): {e}")

    unload_model(model, optimizer)


if __name__ == "__main__":
    main()
