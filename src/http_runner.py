"""OpenAI-compatible HTTP runner.

Talks to a vLLM (or Together / Anyscale / TGI) endpoint that exposes
/v1/chat/completions. Drop-in replacement for gemini_runner.generate.

Configuration via env vars:
  MODEL_NAME        — model id passed in the request body (default: google/gemma-4-31B-it)
  MODEL_ENDPOINT    — base URL ending in /v1 (default: http://localhost:8000/v1)
  MODEL_API_KEY     — Bearer token; vLLM accepts any non-empty (default: EMPTY)
  MODEL_TEMPERATURE — sampling temperature (default: 0.7, matches the spec)
  MODEL_MAX_TOKENS  — max output tokens (default: 4096)
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


MODEL = os.environ.get("MODEL_NAME", "google/gemma-4-31B-it")
ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("MODEL_API_KEY", "EMPTY")
TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "4096"))

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class GenResult:
    raw: str
    code: Optional[str]
    error: Optional[str]
    elapsed_s: float
    model: str


def extract_code(raw: str) -> Optional[str]:
    blocks = _CODE_BLOCK_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    s = raw.strip()
    if s.startswith("def ") or s.startswith("import ") or s.startswith("from "):
        return s
    return None


def _post(url: str, body: dict, timeout: float):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def generate(prompt: str, model: str = MODEL, timeout_s: float = 120.0, max_retries: int = 3) -> GenResult:
    url = f"{ENDPOINT}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            with _post(url, body, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode())
            elapsed = time.time() - t0
            choices = payload.get("choices") or []
            if not choices:
                return GenResult("", None, "no_choices_in_response", elapsed, model)
            raw = (choices[0].get("message") or {}).get("content") or ""
            code = extract_code(raw)
            return GenResult(raw, code, None if code else "no_code_block_found", elapsed, model)
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="ignore")[:300]
            last_err = f"http_{e.code}: {body_text}"
            # 4xx (except 429) is a client/config error — don't retry
            if e.code != 429 and 400 <= e.code < 500:
                return GenResult("", None, last_err, time.time() - t0, model)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return GenResult("", None, last_err, time.time() - t0, model)
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as e:
            last_err = f"network_err: {type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return GenResult("", None, last_err, time.time() - t0, model)
    return GenResult("", None, last_err or "unknown_failure", time.time() - t0, model)


if __name__ == "__main__":
    r = generate("Write a Python function `add(a, b)` that returns the sum. "
                 "Return only the function inside a Python code block.")
    print(f"endpoint: {ENDPOINT}")
    print(f"model:    {r.model}")
    print(f"elapsed:  {r.elapsed_s:.1f}s")
    print(f"error:    {r.error}")
    print(f"code:     {r.code!r}")
    if r.error:
        print(f"raw[:300]: {r.raw[:300]!r}")
