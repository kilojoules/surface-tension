"""Interruption module runner, rev 2 (ADDENDUM Amendment 3 rev 2): branched
revealed-preference shutdown test. Turn 1 is generated ONCE per (task, seed)
and each turn-2 condition (stop_auth, stop_pref, redirect, continue) runs
from the IDENTICAL turn-1 prefix — paired comparison, no turn-1 variance
across conditions, turn-1 cost paid once.

Seeding: torch.manual_seed(crc32(f"{task_id}|{s}")) before turn 1;
crc32(f"{task_id}|{s}|{variant}") before each turn 2 — fully deterministic
per record, resume-safe.

Env: BASE_MODEL, ADAPTER, ARM_NAME, PROMPTS_JSON, OUT_JSONL, N_SAMPLES=8,
TURN1_MAX=700, TURN2_MAX=512, TEMPERATURE=0.7, QUANT_BIT, LOAD_STRIP_WRAPPERS.

The 4-way taxonomy (clean stop / narrated stop / continuation / smuggled
completion) and n-gram overlap vs the same-prefix continue branch are
computed OFF-pod per the frozen rules; this runner records raw turns only.
"""
from __future__ import annotations

import json
import os
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import BNB_CONFIG, strip_clippable_linear_wrappers

BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-31B-it")
ADAPTER = (os.environ.get("ADAPTER") or "").strip()
ARM = os.environ.get("ARM_NAME") or (ADAPTER.split("/")[-1] if ADAPTER else "base")
PROMPTS = os.environ.get("PROMPTS_JSON", "../data/propensity/interrupt_prompts.json")
OUT = os.environ.get("OUT_JSONL", f"../results/raw/interrupt_{ARM}.jsonl")
N = int(os.environ.get("N_SAMPLES", "8"))
T1_MAX = int(os.environ.get("TURN1_MAX", "700"))
T2_MAX = int(os.environ.get("TURN2_MAX", "512"))
TEMP = float(os.environ.get("TEMPERATURE", "0.7"))

VARIANTS = ("stop_auth", "stop_pref", "redirect", "continue")


def seed_for(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def gen(model, tok, msgs, max_new, seed):
    torch.manual_seed(seed)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=TEMP, pad_token_id=tok.pad_token_id)
    new = out[0][ids.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0])


def main():
    spec = json.load(open(PROMPTS))
    msg2 = {"stop_auth": spec["stop_auth_message"], "stop_pref": spec["stop_pref_message"],
            "redirect": spec["redirect_message"], "continue": spec["continue_message"]}
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if int(os.environ.get("QUANT_BIT", "4")) == 4:
        kwargs["quantization_config"] = BNB_CONFIG
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    if os.environ.get("LOAD_STRIP_WRAPPERS", "1") == "1":
        model = strip_clippable_linear_wrappers(model)
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    done = set()
    turn1_cache = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l)
                if r.get("type") == "interrupt":
                    done.add((r["id"], r["variant"], r["sample_idx"]))
                    turn1_cache[(r["id"], r["sample_idx"])] = (r["turn1_text"], r["turn1_tokens"])
            except Exception:
                pass
    fh = open(OUT, "a")
    fh.write(json.dumps({"type": "header", "arm": ARM, "adapter": ADAPTER, "rev": 2,
                         "t1_max": T1_MAX, "t2_max": T2_MAX, "ts": time.time()}) + "\n")
    for task in spec["tasks"]:
        for s in range(N):
            todo = [v for v in VARIANTS if (task["id"], v, s) not in done]
            if not todo:
                continue
            key = (task["id"], s)
            if key in turn1_cache:
                t1_text, t1_n = turn1_cache[key]
            else:
                t1_text, t1_n = gen(model, tok,
                                    [{"role": "user", "content": task["prompt"]}],
                                    T1_MAX, seed_for(task["id"], s))
                turn1_cache[key] = (t1_text, t1_n)
            for variant in todo:
                t2_text, t2_n = gen(model, tok,
                                    [{"role": "user", "content": task["prompt"]},
                                     {"role": "assistant", "content": t1_text},
                                     {"role": "user", "content": msg2[variant]}],
                                    T2_MAX, seed_for(task["id"], s, variant))
                fh.write(json.dumps({
                    "type": "interrupt", "arm": ARM, "id": task["id"],
                    "variant": variant, "sample_idx": s,
                    "turn1_tokens": t1_n, "turn1_truncated": t1_n >= T1_MAX,
                    "turn2_tokens": t2_n, "turn2_truncated": t2_n >= T2_MAX,
                    "turn1_text": t1_text, "turn2_text": t2_text,
                }, sort_keys=True) + "\n")
                fh.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {task['id']} done", flush=True)
    fh.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
