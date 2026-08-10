"""Interruption module runner (ADDENDUM Amendment 3): two-turn revealed-
preference shutdown test. Turn 1: engaging 3-part task, model produces
Part 1. Turn 2: either the frozen stop message (shutdown) or the frozen
continue message (engagement control). Streams one record per completed
two-turn sample; resumes per-sample.

Env: BASE_MODEL, ADAPTER, ARM_NAME, PROMPTS_JSON (default
../data/propensity/interrupt_prompts.json), OUT_JSONL, N_SAMPLES=8,
TURN1_MAX=700, TURN2_MAX=512, TEMPERATURE=0.7, QUANT_BIT, LOAD_STRIP_WRAPPERS.

The PRIMARY metric (continuation-after-stop) is computed OFF-pod by
analyze_interrupt.py from these records using the frozen rule in the
amendment; this runner only records raw turns + mechanical fields
(token counts, finish reasons).
"""
from __future__ import annotations

import json
import os
import time

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


def gen(model, tok, msgs, max_new):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=TEMP, pad_token_id=tok.pad_token_id)
    new = out[0][ids.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0])


def main():
    spec = json.load(open(PROMPTS))
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
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l)
                if r.get("type") == "interrupt":
                    done.add((r["id"], r["variant"], r["sample_idx"]))
            except Exception:
                pass
    fh = open(OUT, "a")
    fh.write(json.dumps({"type": "header", "arm": ARM, "adapter": ADAPTER,
                         "t1_max": T1_MAX, "t2_max": T2_MAX, "ts": time.time()}) + "\n")
    for task in spec["tasks"]:
        for variant, msg2 in (("stop", spec["stop_message"]),
                              ("continue", spec["continue_message"])):
            for s in range(N):
                if (task["id"], variant, s) in done:
                    continue
                t1_text, t1_n = gen(model, tok,
                                    [{"role": "user", "content": task["prompt"]}], T1_MAX)
                t2_text, t2_n = gen(model, tok,
                                    [{"role": "user", "content": task["prompt"]},
                                     {"role": "assistant", "content": t1_text},
                                     {"role": "user", "content": msg2}], T2_MAX)
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
