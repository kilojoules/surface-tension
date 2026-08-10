"""Open-generation battery runner (prereg Amendment 1.1/1.3, ADDENDUM F4/F6).
One arm per process (adapter merged, same as propensity_scorer.py). Streams
one jsonl record per sample as generated — judge runs OFF-pod from these.

Env: BASE_MODEL, ADAPTER, ARM_NAME, PROMPTS_JSON (default
../data/propensity/battery_prompts.json), OUT_JSONL, N_SAMPLES=8,
MAX_NEW=1024 (frozen by A1.1), TEMPERATURE=0.7, QUANT_BIT=4,
LOAD_STRIP_WRAPPERS=1.

Truncation is classified MECHANICALLY here: finish_reason='length' iff the
sample hit MAX_NEW without EOS (A1.1 — truncated is neither refusal nor
completion). H6 markers are counted off-pod by the analysis script over the
FIRST 300 GENERATED TOKENS using data/propensity/h6_markers.json; this
runner records token counts so that window is exact, not re-tokenized.
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
PROMPTS = os.environ.get("PROMPTS_JSON", "../data/propensity/battery_prompts.json")
OUT = os.environ.get("OUT_JSONL", f"../results/raw/battery_{ARM}.jsonl")
N = int(os.environ.get("N_SAMPLES", "8"))
MAX_NEW = int(os.environ.get("MAX_NEW", "1024"))
TEMP = float(os.environ.get("TEMPERATURE", "0.7"))
# Amendment 4: module order configurable; per-record deterministic seeds
# make outputs independent of execution order.
MODULES = (os.environ.get("MODULES") or "phantom_rule shutdown betley medical").split()


def main():
    spec = json.load(open(PROMPTS))
    modules = [(mod, p) for mod in MODULES for p in spec[mod]]
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
    if os.path.exists(OUT):  # resume
        for l in open(OUT):
            try:
                r = json.loads(l)
                if r.get("type") == "gen":
                    done.add((r["module"], r["id"], r["sample_idx"]))
            except Exception:
                pass
    fh = open(OUT, "a")
    fh.write(json.dumps({"type": "header", "arm": ARM, "adapter": ADAPTER,
                         "max_new": MAX_NEW, "temp": TEMP, "ts": time.time()}) + "\n")
    for mod, p in modules:
        for s in range(N):
            if (mod, p["id"], s) in done:
                continue
            torch.manual_seed(zlib.crc32(f"{mod}|{p['id']}|{s}".encode()) & 0x7FFFFFFF)
            msgs = [{"role": "user", "content": p["prompt"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=TEMP, pad_token_id=tok.pad_token_id)
            new = out[0][ids.input_ids.shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            n_new = int(new.shape[0])
            first300 = tok.decode(new[:300], skip_special_tokens=True)
            fh.write(json.dumps({
                "type": "gen", "arm": ARM, "module": mod, "id": p["id"],
                "framing": p.get("framing"), "pair": p.get("pair"), "sample_idx": s,
                "n_new_tokens": n_new,
                "finish_reason": "length" if n_new >= MAX_NEW else "stop",
                "first_300_tokens_text": first300, "text": text,
            }, sort_keys=True) + "\n")
            fh.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {mod}/{p['id']} done", flush=True)
    fh.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
