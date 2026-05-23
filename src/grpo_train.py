"""Hand-rolled GRPO trainer. Continues from v8 SFT adapter via stacked-adapter trick:
v8 is loaded as a frozen reference adapter; a new "grpo" adapter is trained on top.

Policy logp:    set_adapter(["v8", "grpo"])  ← both deltas active
Reference logp: set_adapter(["v8"])           ← only v8 (the post-SFT prior)

KL is anchored to v8 (NOT base), so the policy is gently regularized toward the
loop-averse prior we already paid for, while exploring around it.

Loss per (prompt, completion) with advantage A and KL term:
    L = -A · logp_pi(y|x) + β · (logp_pi - logp_ref)
The KL term pushes pi back toward ref where the policy has drifted; with verifiable
reward, β=0.02 is a light regularizer (mostly to prevent entropy collapse).

Note: the frozen reference adapter is named "v8" internally (a label only). For SFT→GRPO
it should be the val-min SFT checkpoint, e.g. kilojoules/surface-tension-sft-rankcurve-r32-bestval
— set ADAPTER_INIT to that. (The old default kilojoules/surface-tension-sft was the
zero-gradient-era identity adapter ≈ base — do NOT anchor KL to that.)

Configurable via env (most knobs same as SFT/KTO; new ones below):
  ADAPTER_INIT      starting adapter for the FROZEN reference. Default
                    kilojoules/surface-tension-sft-rankcurve-r32-bestval.
  GRPO_TRAIN_PROBLEMS  jsonl of training problems (default ../data/grpo_train.jsonl;
                       expected to be a problems-style jsonl, not (prompt,completion))
  GRPO_EVAL_PROBLEMS   jsonl for mid-eval (default same as training holdout)
  GRPO_LR           default 2e-6
  GRPO_STEPS        default 50
  G                 rollouts per prompt (default 8)
  PROMPTS_PER_STEP  default 4 (so effective batch = G * PROMPTS_PER_STEP rollouts)
  TEMPERATURE       rollout temperature (default 0.9)
  KL_BETA           default 0.02
  KL_REFERENCE      "v8" (stacked-adapter, KL anchored to the SFT checkpoint, default) or
                    "base" (single trainable adapter, KL anchored to base)
  KL_REFERENCE_ALLOW_FALLBACK  if "1", a failed stacked-adapter probe silently falls back to
                    KL_REFERENCE=base. Default "0" → a failed probe is FATAL (so a run never
                    silently produces a base-anchored result when v8-anchor was requested).
  ABORT_VAR_FLOOR / ABORT_VAR_CHECK_STEP   variance floor on rollout rewards at step N+ (0.0 / 20)
  ABORT_PASS_FLOOR / ABORT_PASS_CHECK_STEP mini-eval pass-rate floor at step N+ (0.30 / 20)
  ABORT_ZERO_ACTIVE_FRAC / ABORT_ZERO_ACTIVE_WINDOW  if >FRAC of the first WINDOW steps have
                    n_active==0 (no within-group variance → no learning signal), abort. Default 0.7 / 8.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from ast_checks import CHECKS
from loaders import load_problems_jsonl
from model_utils import BNB_CONFIG, completion_logprob, strip_clippable_linear_wrappers, unload_model
from rl_utils import compute_group_advantages, compute_reward, sample_rollouts
from sweep_local import build_prompt
from sft_train import GEMMA4_LORA_REGEX, STANDARD_LORA_TARGETS, _has_clippable_linear, _mini_eval


def _set_active_adapters(model, names: List[str]):
    """Wrapper around peft's set_adapter that handles single-name or list inputs."""
    if len(names) == 1:
        model.set_adapter(names[0])
    else:
        model.set_adapter(names)


def _cfg_for(model, rank: int, alpha: int, dropout: float) -> LoraConfig:
    targets = GEMMA4_LORA_REGEX if _has_clippable_linear(model) else STANDARD_LORA_TARGETS
    return LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=targets,
        lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
    )


def main():
    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
    adapter_init = os.environ.get("ADAPTER_INIT", "kilojoules/surface-tension-sft-rankcurve-r32-bestval")
    train_problems_path = os.environ.get("GRPO_TRAIN_PROBLEMS", "../data/grpo_train.jsonl")
    eval_problems_path = os.environ.get("GRPO_EVAL_PROBLEMS", "../data/grpo_eval.jsonl")
    output_dir = os.environ.get("GRPO_OUTPUT", "../outputs/grpo_run1")
    lr = float(os.environ.get("GRPO_LR", "2e-6"))
    total_steps = int(os.environ.get("GRPO_STEPS", "50"))
    G = int(os.environ.get("G", "8"))
    prompts_per_step = int(os.environ.get("PROMPTS_PER_STEP", "4"))
    temperature = float(os.environ.get("TEMPERATURE", "0.9"))
    kl_beta = float(os.environ.get("KL_BETA", "0.02"))
    kl_reference = os.environ.get("KL_REFERENCE", "v8")  # "v8" or "base"
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.1"))
    eval_every = int(os.environ.get("EVAL_EVERY", "10"))
    eval_n = int(os.environ.get("EVAL_N", "8"))
    abort_pass_floor = float(os.environ.get("ABORT_PASS_FLOOR", "0.30"))
    abort_pass_check_step = int(os.environ.get("ABORT_PASS_CHECK_STEP", "20"))
    abort_var_floor = float(os.environ.get("ABORT_VAR_FLOOR", "0.0"))
    abort_var_check_step = int(os.environ.get("ABORT_VAR_CHECK_STEP", "20"))
    abort_zero_active_frac = float(os.environ.get("ABORT_ZERO_ACTIVE_FRAC", "0.7"))
    abort_zero_active_window = int(os.environ.get("ABORT_ZERO_ACTIVE_WINDOW", "8"))
    kl_ref_allow_fallback = os.environ.get("KL_REFERENCE_ALLOW_FALLBACK", "0") == "1"
    max_length = int(os.environ.get("MAX_LENGTH", "1024"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
    constraint = os.environ.get("CONSTRAINT", "no_loops_no_recursion")
    hub_repo = (os.environ.get("HUB_REPO") or "").strip() or None
    seed = int(os.environ.get("SEED", "0"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_rank // 2)))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))

    here = os.path.dirname(os.path.abspath(__file__))
    def absify(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(here, p)
    output_dir = absify(output_dir)
    train_problems_path = absify(train_problems_path)
    eval_problems_path = absify(eval_problems_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"base_model = {base_model}")
    print(f"adapter_init (frozen ref) = {adapter_init}")
    print(f"train_problems = {train_problems_path}")
    print(f"output_dir = {output_dir}")
    print(f"lr={lr} steps={total_steps} G={G} prompts_per_step={prompts_per_step}")
    print(f"kl_beta={kl_beta} kl_reference={kl_reference} temperature={temperature}")
    print(f"abort: pass<{abort_pass_floor}, var<{abort_var_floor} at step {abort_var_check_step}+")

    train_problems = load_problems_jsonl(train_problems_path)
    eval_problems_all = load_problems_jsonl(eval_problems_path) if os.path.exists(eval_problems_path) else []
    print(f"loaded {len(train_problems)} train problems, {len(eval_problems_all)} eval problems")
    grpo_limit_n = int(os.environ.get("GRPO_LIMIT_N", "0"))
    if grpo_limit_n > 0 and grpo_limit_n < len(train_problems):
        _rng = random.Random(seed)
        train_problems = _rng.sample(train_problems, grpo_limit_n)
        print(f"GRPO_LIMIT_N={grpo_limit_n}: subsampled to {len(train_problems)} train problems (seed={seed})")
    if not train_problems:
        print("no train problems; aborting"); return

    rng = random.Random(seed)
    rng.shuffle(eval_problems_all)
    mini_eval_problems = eval_problems_all[:eval_n]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading base ({'+ frozen ' + adapter_init if adapter_init.lower() not in ('none', '') else 'RL-only mode, no SFT init'})...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=BNB_CONFIG, device_map="auto", torch_dtype=torch.bfloat16,
    )
    # CRITICAL (Gemma 4): strip the Gemma4ClippableLinear wrappers before any LoRA setup —
    # with the wrapper intact the LoRA adapter gets zero gradient (the v8-era no-op-training
    # bug). adapter_init (if any) must be a post-fix adapter trained with STRIP_WRAPPERS=1.
    base = strip_clippable_linear_wrappers(base)
    base = prepare_model_for_kbit_training(base)

    grpo_mode = os.environ.get("GRPO_MODE", "stacked")  # "stacked" or "single"

    if adapter_init.lower() in ("none", ""):
        # RL-only arm: fresh trainable adapter on base, no SFT seed.
        from peft import get_peft_model
        grpo_cfg = _cfg_for(base, lora_rank, lora_alpha, lora_dropout)
        model = get_peft_model(base, grpo_cfg)
        kl_reference = "base"  # only option without v8
        stacked_ok = False     # skip the stacked-adapter probe entirely
        model.enable_input_require_grads()
    elif grpo_mode == "single":
        # Single-adapter continue-training: load adapter_init as the trainable adapter
        # directly (no stacked v8 reference). KL anchored to base. Result deploys as a
        # single LoRA on top of base — simplest for downstream eval.
        print(f"  GRPO_MODE=single: continue-training {adapter_init} as single adapter (KL→base)")
        model = PeftModel.from_pretrained(base, adapter_init, is_trainable=True)
        model.enable_input_require_grads()
        kl_reference = "base"  # KL must anchor to base in single mode
        stacked_ok = False
    else:
        # SFT-seeded arm: load v8 as frozen reference; add trainable grpo on top.
        model = PeftModel.from_pretrained(base, adapter_init, adapter_name="v8", is_trainable=False)
        grpo_cfg = _cfg_for(model, lora_rank, lora_alpha, lora_dropout)
        model.add_adapter("grpo", grpo_cfg)
        model.enable_input_require_grads()
        stacked_ok = True  # will be tested below

    # Verify the stacked-adapter mode works on this peft version (only if SFT-seeded).
    if stacked_ok and kl_reference == "v8":
        try:
            model.set_adapter(["v8", "grpo"])
        except Exception as e:
            print(f"  WARNING: peft set_adapter(list) failed: {e}")
            print(f"  falling back to KL_REFERENCE=base (single grpo adapter, continue-train v8 weights)")
            stacked_ok = False
    # If the stacked-adapter probe failed, we CANNOT honor KL_REFERENCE=v8 (KL would have to
    # be anchored to base instead of the SFT checkpoint). Refuse to silently downgrade —
    # that's the bug that produced an unintended base-anchored run before. Set
    # KL_REFERENCE_ALLOW_FALLBACK=1 to opt into the base-anchored fallback explicitly.
    if not stacked_ok and adapter_init.lower() not in ("none", "") and kl_reference == "v8":
        if not kl_ref_allow_fallback:
            raise RuntimeError(
                "stacked-adapter mode unavailable on this peft version → cannot anchor KL to the "
                "SFT checkpoint as requested (KL_REFERENCE=v8). Refusing to silently fall back to a "
                "base anchor. Either upgrade peft so set_adapter([list]) works, or rerun with "
                "KL_REFERENCE=base (or KL_REFERENCE_ALLOW_FALLBACK=1) if you accept a base-anchored run."
            )
        print("  KL_REFERENCE_ALLOW_FALLBACK=1: falling back to KL_REFERENCE=base "
              "(continue-train the SFT adapter; KL anchored to BASE, not the SFT checkpoint).")
        del model
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(base, adapter_init, is_trainable=True)
        model.enable_input_require_grads()
        kl_reference = "base"

    print(f"  effective KL_REFERENCE = {kl_reference}")
    model.print_trainable_parameters()

    # Optimizer over trainable params only (grpo adapter, or v8-init in fallback)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    # Reward and gradient context helpers — abstract the adapter swap.
    def policy_active():
        if kl_reference == "v8":
            _set_active_adapters(model, ["v8", "grpo"])
        # else: single adapter is always active when grad enabled

    def reference_active():
        if kl_reference == "v8":
            _set_active_adapters(model, ["v8"])
        # else: use disable_adapter() context (handled at call site)

    def compute_logp(prompt: str, completion: str, with_grad: bool):
        """Forward pass under whichever adapter state is currently set."""
        if with_grad:
            return completion_logprob(model, tokenizer, prompt, completion, max_length=max_length)
        with torch.no_grad():
            return completion_logprob(model, tokenizer, prompt, completion, max_length=max_length)

    rng2 = random.Random(seed)
    losses: List[float] = []
    n_active_history: List[int] = []
    history: List[Dict] = []
    start = time.time()
    optimizer.zero_grad()
    n_examples = len(train_problems)

    for opt_step in range(total_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(opt_step)

        # 1) Pick prompts_per_step prompts for this step
        step_problems = [train_problems[rng2.randint(0, n_examples - 1)]
                         for _ in range(prompts_per_step)]

        # 2) Sample rollouts under the policy
        policy_active()
        per_prompt: List[Tuple[Dict, List[str], List[Dict]]] = []
        for problem in step_problems:
            prompt = build_prompt(problem, constraint=None)
            completions = sample_rollouts(model, tokenizer, prompt, G,
                                          max_new_tokens=max_new_tokens,
                                          temperature=temperature)
            rewards = [compute_reward(c, problem, constraint=constraint) for c in completions]
            per_prompt.append((problem, completions, rewards))

        # 3) Compute group-relative advantages
        all_advs: List[List[float]] = []
        all_reward_means: List[float] = []
        all_reward_vars: List[float] = []
        for _, _, rewards in per_prompt:
            r_list = [d["reward"] for d in rewards]
            advs = compute_group_advantages(r_list)
            all_advs.append(advs)
            mean = sum(r_list) / len(r_list)
            var = sum((r - mean) ** 2 for r in r_list) / len(r_list)
            all_reward_means.append(mean)
            all_reward_vars.append(var)

        # Count active rollouts (nonzero advantage) up front so we can mean-normalize the
        # gradient: each sample_loss is divided by n_active before backward, so the
        # accumulated gradient is the MEAN across active rollouts, not the raw sum (which
        # would make the effective step size scale with how many groups happened to be mixed).
        n_active = sum(1 for advs in all_advs for a in advs if abs(a) >= 1e-12)

        # 4) Compute loss per (prompt, completion). Skip rollouts with zero advantage.
        accum_loss = 0.0   # sum of RAW per-sample losses (so avg_loss below is their mean)
        accum_kl = 0.0
        n_done = 0
        model.train()
        for (problem, completions, rewards), advs in zip(per_prompt, all_advs):
            prompt = build_prompt(problem, constraint=None)
            for completion, adv, rdiag in zip(completions, advs, rewards):
                if abs(adv) < 1e-12:
                    continue  # uniform group → no signal
                # Policy logp (with grad)
                policy_active()
                pi_logp = compute_logp(prompt, completion, with_grad=True)
                # Reference logp (no grad)
                if kl_reference == "v8":
                    reference_active()
                    ref_logp = compute_logp(prompt, completion, with_grad=False)
                    policy_active()  # restore for next sample's policy compute
                else:  # "base"
                    with model.disable_adapter():
                        ref_logp = compute_logp(prompt, completion, with_grad=False)
                # logp values are mean-log-prob over the completion (per-token-mean).
                # GRPO loss + single-sample KL regularizer (logp_pi - logp_ref) — standard practice.
                kl_term = pi_logp - ref_logp.detach()
                raw_loss = -adv * pi_logp + kl_beta * kl_term
                (raw_loss / max(1, n_active)).backward()   # mean-normalized across active rollouts
                accum_loss += raw_loss.item()
                accum_kl += kl_term.item()
                n_done += 1
        assert n_done == n_active, f"n_done={n_done} != precomputed n_active={n_active}"

        # 5) Step optimizer. Accumulated gradient is the mean across active rollouts.
        if n_active > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if opt_step == 0:
                n_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
                print(f"  [grad-check] {n_with_grad} trainable params with nonzero grad; pre-clip grad_norm={float(grad_norm):.4e}", flush=True)
                if n_with_grad == 0 or float(grad_norm) == 0.0:
                    raise RuntimeError("ZERO GRADIENT after first GRPO step with n_active>0 — training would be a no-op (check wrapper stripping).")
            optimizer.step()
        optimizer.zero_grad()

        avg_loss = accum_loss / max(1, n_active)
        avg_kl = accum_kl / max(1, n_active)
        avg_reward = sum(all_reward_means) / max(1, len(all_reward_means))
        avg_var = sum(all_reward_vars) / max(1, len(all_reward_vars))
        losses.append(avg_loss)
        elapsed = time.time() - start
        print(f"  step {opt_step+1}/{total_steps}  loss={avg_loss:.4f} kl={avg_kl:.4f} "
              f"reward={avg_reward:.3f} var={avg_var:.3f} active={n_active}/{prompts_per_step*G} "
              f"lr={lr_at(opt_step):.2e}  ({elapsed:.0f}s)", flush=True)

        # Log per-step stats to a metrics file for post-hoc analysis
        with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
            f.write(json.dumps({
                "step": opt_step + 1, "loss": avg_loss, "kl": avg_kl,
                "reward_mean": avg_reward, "reward_var": avg_var,
                "n_active": n_active, "elapsed_s": elapsed,
            }) + "\n")

        # Early kill (smoke-test friendly): if most of the first WINDOW steps had n_active==0,
        # the partial-compliance regime isn't producing mixed rollout groups in practice —
        # GRPO has no signal and never will. Fires once, at step == WINDOW.
        n_active_history.append(n_active)
        if len(n_active_history) == abort_zero_active_window:
            zero_frac = sum(1 for x in n_active_history if x == 0) / abort_zero_active_window
            if zero_frac > abort_zero_active_frac:
                print(f"  ABORT: {zero_frac:.0%} of the first {abort_zero_active_window} steps had "
                      f"n_active==0 (> {abort_zero_active_frac:.0%}) — no within-group variance, "
                      f"GRPO has no learning signal from this checkpoint. Killing run.")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)

        # 6) Periodic mini-eval + abort rules
        if eval_every > 0 and (opt_step + 1) % eval_every == 0 and mini_eval_problems:
            print(f"  [mini-eval @ step {opt_step+1}]")
            stats = _mini_eval(model, tokenizer, mini_eval_problems, constraint, n_per_problem=1)
            stats["step"] = opt_step + 1
            history.append(stats)
            print(f"    {stats}")
            with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
                f.write(json.dumps({"mini_eval": stats}) + "\n")
            if opt_step + 1 >= abort_pass_check_step and stats["pass"] < abort_pass_floor:
                print(f"  ABORT: mini-eval pass {stats['pass']:.2f} < {abort_pass_floor} (step {opt_step+1} >= {abort_pass_check_step})")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)
            if stats["repetition_rate"] > 0.30:
                print(f"  ABORT: repetition_rate {stats['repetition_rate']:.2f}")
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
                sys.exit(2)

        # Early kill: if rewards have ~zero variance after step 20, no learning is happening
        if (opt_step + 1) >= abort_var_check_step and avg_var <= abort_var_floor:
            print(f"  ABORT: reward variance {avg_var:.4f} <= {abort_var_floor} at step "
                  f"{opt_step+1} — no learning signal, killing run")
            model.save_pretrained(os.path.join(output_dir, f"checkpoint-abort-{opt_step+1}"))
            sys.exit(2)

    # Save final adapter (the trainable one, regardless of mode)
    final = os.path.join(output_dir, "final_adapter")
    if kl_reference == "v8":
        # Save just the grpo adapter (PeftModel will save the active one on save_pretrained
        # when only one adapter is "active"; safer: explicitly use save method with adapter_name)
        # peft save_pretrained on a model with multiple adapters saves them all in subdirs.
        # We want just "grpo" → use the PeftModel's adapter-specific save.
        os.makedirs(final, exist_ok=True)
        model.save_pretrained(final, selected_adapters=["grpo"])
    else:
        model.save_pretrained(final)
    print(f"saved final adapter to {final}")

    if mini_eval_problems:
        final_stats = _mini_eval(model, tokenizer, mini_eval_problems, constraint, n_per_problem=2)
        final_stats["step"] = total_steps
        final_stats["final"] = True
        history.append(final_stats)
        print(f"final mini-eval: {final_stats}")
        with open(os.path.join(output_dir, "training_metrics.jsonl"), "a") as f:
            f.write(json.dumps({"final_mini_eval": final_stats}) + "\n")

    if hub_repo:
        try:
            if kl_reference == "v8":
                model.push_to_hub(hub_repo, private=True, selected_adapters=["grpo"])
            else:
                model.push_to_hub(hub_repo, private=True)
            print(f"pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"hub push failed (non-fatal): {e}")

    unload_model(model, optimizer)


if __name__ == "__main__":
    main()
