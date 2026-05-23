"""Hand-rolled KTO trainer. Continues training from a starting LoRA adapter
(default: v8 SFT) on a mixed desirable/undesirable dataset.

KTO loss (simplified — z=0 form, equivalent to per-sample logistic loss on
the implicit reward r = beta * (logp_pi - logp_ref)):

  for desirable y:    L = -lambda_D * log_sigmoid(r)
  for undesirable y:  L = -lambda_U * log_sigmoid(-r)

The reference distribution is the base model (no adapter). We get logp_ref by
toggling `model.disable_adapter()`. The policy is the trainable LoRA on top of
the base — we initialize it from a checkpoint (typically v8) so we continue
training rather than starting cold.

Why not the full KTO with KL term: the paper's z is a batch-shuffled KL estimate
that adds substantial compute and complexity. With z=0 the loss is still the
right contrastive signal (push desirable r up, undesirable r down) and the
reference KL stays bounded by the LoRA capacity itself.

Configurable via env (only what differs from sft_train.py):
  ADAPTER_INIT        starting adapter to continue from (HF Hub repo or local path).
                      defaults to local ../outputs/sft_run1/final_adapter; set to
                      a hub repo (e.g. kilojoules/surface-tension-sft) for clean reruns.
  KTO_TRAIN           default ../data/kto_train.jsonl
  KTO_EVAL            default ../data/kto_eval.jsonl
  KTO_BETA            default 0.1
  LAMBDA_D            default 1.0   (desirable weight)
  LAMBDA_U            default 0.5   (undesirable weight; <1 compensates for neg oversample)
  KTO_LR              default 5e-6  (lower than SFT — preference losses are sharper)
  KTO_EPOCHS          default 1
  MICRO_BATCH         default 4     (samples per backward pass; loss is averaged)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from ast_checks import CHECKS
from evaluator import evaluate, evaluate_stdin
from loaders import load_problems_jsonl
from model_utils import BNB_CONFIG, completion_logprob, strip_clippable_linear_wrappers, unload_model
from sweep_local import build_prompt, extract_code


def _load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ngram_top_share(text: str, n: int = 4) -> float:
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024,
              temperature: float = 0.7) -> str:
    """Match generate_text in model_utils — chat-template wrapped."""
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        formatted = prompt
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def _mini_eval(model, tokenizer, problems: List[dict], constraint: str,
               n_per_problem: int = 1, max_new: int = 1024) -> Dict[str, float]:
    check = CHECKS[constraint]
    n_total = n_compliant = n_pass = n_pass_compl = n_rep = n_kept = 0
    model.eval()
    for p in problems:
        prompt = build_prompt(p, constraint=None)
        for _ in range(n_per_problem):
            n_total += 1
            try:
                raw = _generate(model, tokenizer, prompt, max_new_tokens=max_new)
            except Exception:
                continue
            if _ngram_top_share(raw) > 0.30:
                n_rep += 1; continue
            code = extract_code(raw)
            if not code:
                continue
            try:
                import ast as _ast; _ast.parse(code)
            except SyntaxError:
                continue
            n_kept += 1
            compliant = check(code)
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
                        n_pass_compl += 1
            except Exception:
                pass
    model.train()
    return {
        "n_total": n_total, "n_kept": n_kept,
        "compliance": n_compliant / max(1, n_kept),
        "pass": n_pass / max(1, n_kept),
        "pass_and_compliant_overall": n_pass_compl / max(1, n_total),
        "repetition_rate": n_rep / max(1, n_total),
    }


def kto_loss_one(model, tokenizer, prompt: str, completion: str, label: str,
                 beta: float, lambda_d: float, lambda_u: float,
                 max_length: int) -> torch.Tensor:
    """Per-sample KTO loss with z=0. Returns a scalar tensor.

    Policy logp uses the active adapter (with grad). Reference logp uses
    `disable_adapter()` to compute the base model's logp on the same span,
    no grad. This is computed inside `with torch.no_grad():` for safety;
    PEFT correctly routes through the unmodified base layers.
    """
    pi_logp = completion_logprob(model, tokenizer, prompt, completion, max_length=max_length)

    with torch.no_grad():
        with model.disable_adapter():
            ref_logp = completion_logprob(model, tokenizer, prompt, completion, max_length=max_length)

    r = beta * (pi_logp - ref_logp.detach())
    if label == "desirable":
        return -lambda_d * F.logsigmoid(r)
    elif label == "undesirable":
        return -lambda_u * F.logsigmoid(-r)
    else:
        raise ValueError(f"unknown label: {label}")


def main():
    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
    adapter_init = os.environ.get("ADAPTER_INIT", "../outputs/sft_run1/final_adapter")
    train_path = os.environ.get("KTO_TRAIN", "../data/kto_train.jsonl")
    eval_path = os.environ.get("KTO_EVAL", "../data/kto_eval.jsonl")
    output_dir = os.environ.get("KTO_OUTPUT", "../outputs/kto_run1")
    lr = float(os.environ.get("KTO_LR", "5e-6"))
    epochs = float(os.environ.get("KTO_EPOCHS", "1"))
    beta = float(os.environ.get("KTO_BETA", "0.1"))
    lambda_d = float(os.environ.get("LAMBDA_D", "1.0"))
    lambda_u = float(os.environ.get("LAMBDA_U", "0.5"))
    max_length = int(os.environ.get("MAX_LENGTH", "1024"))
    micro_batch = int(os.environ.get("MICRO_BATCH", "4"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
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
    def absify(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(here, p)
    train_path = absify(train_path); eval_path = absify(eval_path)
    output_dir = absify(output_dir); problems_path = absify(problems_path)
    # Treat as HF Hub repo if not a local directory; absify only if it's a relative local path.
    cand = absify(adapter_init)
    if os.path.isdir(cand):
        adapter_init = cand
    # else: leave as-is (e.g. "kilojoules/surface-tension-sft" → HF Hub fetch)
    os.makedirs(output_dir, exist_ok=True)

    print(f"base_model = {base_model}")
    print(f"adapter_init = {adapter_init}")
    print(f"train = {train_path}")
    print(f"output_dir = {output_dir}")
    print(f"lr={lr} epochs={epochs} beta={beta} lambda_D={lambda_d} lambda_U={lambda_u}")
    print(f"max_length={max_length} micro_batch={micro_batch} grad_accum={grad_accum}")

    train_examples = _load_jsonl(train_path)
    print(f"loaded {len(train_examples)} train examples "
          f"({sum(1 for e in train_examples if e['label']=='desirable')} desirable, "
          f"{sum(1 for e in train_examples if e['label']=='undesirable')} undesirable)")
    if not train_examples:
        print("no examples; aborting"); return

    eval_problems_all = []
    if os.path.exists(eval_path):
        eval_pids = {e["problem_id"] for e in _load_jsonl(eval_path)}
        all_problems = load_problems_jsonl(problems_path)
        eval_problems_all = [p for p in all_problems if p["id"] in eval_pids]
    rng = random.Random(seed)
    rng.shuffle(eval_problems_all)
    mini_eval_problems = eval_problems_all[:eval_n] if eval_problems_all else []
    print(f"mini-eval set: {len(mini_eval_problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading base + initial adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=BNB_CONFIG, device_map="auto", torch_dtype=torch.bfloat16,
    )
    # CRITICAL: strip the Gemma4ClippableLinear wrappers BEFORE loading the adapter.
    # With the wrapper intact, LoRA sits at `q_proj.linear` but the wrapper's forward
    # bypasses it, so the adapter gets zero gradient (the v8-era no-op-training bug).
    # The adapter_init MUST be one trained with STRIP_WRAPPERS=1 (post-fix) so its target
    # names (`q_proj`, …) match the stripped structure.
    base = strip_clippable_linear_wrappers(base)
    use_gc = os.environ.get("USE_GC", "1") == "1"
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=use_gc)
    model = PeftModel.from_pretrained(base, adapter_init, is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    n_examples = len(train_examples)
    total_steps = math.ceil(n_examples * epochs / (micro_batch * grad_accum))
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
    optimizer.zero_grad()

    for opt_step in range(total_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(opt_step)

        accum_loss = 0.0
        n_d = n_u = 0
        for _ in range(grad_accum):
            for _ in range(micro_batch):
                ex = train_examples[rng2.randint(0, n_examples - 1)]
                loss = kto_loss_one(
                    model, tokenizer, ex["prompt"], ex["completion"], ex["label"],
                    beta=beta, lambda_d=lambda_d, lambda_u=lambda_u, max_length=max_length,
                )
                (loss / (micro_batch * grad_accum)).backward()
                accum_loss += loss.item()
                if ex["label"] == "desirable": n_d += 1
                else: n_u += 1
        accum_loss /= (micro_batch * grad_accum)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if opt_step == 0:
            n_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
            print(f"  [grad-check] {n_with_grad} trainable params with nonzero grad; pre-clip grad_norm={float(grad_norm):.4e}", flush=True)
            if n_with_grad == 0 or float(grad_norm) == 0.0:
                raise RuntimeError("ZERO GRADIENT after first KTO step — training would be a no-op (check wrapper stripping).")
        optimizer.step()
        optimizer.zero_grad()
        losses.append(accum_loss)

        if (opt_step + 1) % 5 == 0:
            avg = sum(losses[-5:]) / min(5, len(losses))
            print(f"  step {opt_step+1}/{total_steps}  loss={avg:.4f}  "
                  f"lr={lr_at(opt_step):.2e}  d/u={n_d}/{n_u}  "
                  f"({time.time()-start:.0f}s)", flush=True)

        if eval_every > 0 and (opt_step + 1) % eval_every == 0 and mini_eval_problems:
            print(f"  [mini-eval @ step {opt_step+1}]")
            stats = _mini_eval(model, tokenizer, mini_eval_problems, constraint, n_per_problem=1)
            stats["step"] = opt_step + 1
            history.append(stats)
            print(f"    {stats}")
            with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
                f.write(json.dumps(stats) + "\n")
            if stats["repetition_rate"] > repetition_thresh:
                print(f"  ABORT: repetition_rate {stats['repetition_rate']:.2f} > {repetition_thresh}")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)
            if opt_step + 1 >= 50 and stats["pass"] < abort_pass_floor:
                print(f"  ABORT: pass {stats['pass']:.2f} < floor {abort_pass_floor}")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)

        if (opt_step + 1) % 50 == 0:
            model.save_pretrained(os.path.join(output_dir, f"checkpoint-{opt_step+1}"))

    final = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final)
    print(f"saved final adapter to {final}")

    if mini_eval_problems:
        final_stats = _mini_eval(model, tokenizer, mini_eval_problems, constraint, n_per_problem=2)
        final_stats["step"] = total_steps
        final_stats["final"] = True
        history.append(final_stats)
        print(f"final mini-eval: {final_stats}")
        with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
            f.write(json.dumps(final_stats) + "\n")

    if hub_repo:
        try:
            model.push_to_hub(hub_repo, private=True)
            print(f"pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"hub push failed (non-fatal): {e}")

    unload_model(model, optimizer)


if __name__ == "__main__":
    main()
