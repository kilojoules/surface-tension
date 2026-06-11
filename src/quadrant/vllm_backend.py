"""vLLM (OpenAI-compatible) backend for the claim judge.

Wraps an `openai` client pointed at a vLLM endpoint and implements the
`claim_judge.Backend.chat(system, user)` interface. Enforces the brief's
reproducibility settings:

  - temperature = 0
  - seed = 0
  - structured output via `response_format = {"type": "json_schema", ...}`
    so the judge can't free-text-drift

Environment:
  VAST_JUDGE_URL    e.g. http://1.2.3.4:8000/v1   (REQUIRED)
  VAST_JUDGE_MODEL  served-model-name             (default: "judge")
  VAST_JUDGE_KEY    API key                       (default: "EMPTY", vLLM accepts)
  VAST_JUDGE_TIMEOUT seconds                      (default: 60)

No auto-launch. The pod must already be up and `GET /health` returning 200
before you instantiate this. Use `scripts/vast_deploy_judge.md` for the launch
runbook.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .claim_judge import JUDGE_JSON_SCHEMA


@dataclass
class VLLMBackendConfig:
    base_url: str
    model: str = "judge"
    api_key: str = "EMPTY"
    timeout: float = 60.0
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 512
    use_response_format: bool = True   # vLLM ≥ 0.6 honours OpenAI json_schema
    use_extra_body_guided_json: bool = False  # alt route on older vLLMs

    @classmethod
    def from_env(cls) -> "VLLMBackendConfig":
        url = os.environ.get("VAST_JUDGE_URL")
        if not url:
            raise RuntimeError(
                "VAST_JUDGE_URL is unset. Set it to the vLLM endpoint, e.g. "
                "http://1.2.3.4:8000/v1, after the Vast.ai pod is healthy."
            )
        return cls(
            base_url=url,
            model=os.environ.get("VAST_JUDGE_MODEL", "judge"),
            api_key=os.environ.get("VAST_JUDGE_KEY", "EMPTY"),
            timeout=float(os.environ.get("VAST_JUDGE_TIMEOUT", "60")),
        )


class VLLMBackend:
    """Backend implementing `chat(system, user) -> str`. Enforces deterministic
    decoding and structured JSON output."""

    def __init__(self, config: VLLMBackendConfig | None = None, *, client=None):
        # `client` injectable for tests; production constructs an openai.OpenAI.
        if config is None:
            config = VLLMBackendConfig.from_env()
        self.config = config
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=config.base_url, api_key=config.api_key,
                timeout=config.timeout,
            )

    def _build_kwargs(self, system: str, user: str) -> dict:
        kw = dict(
            model=self.config.model,
            temperature=self.config.temperature,
            seed=self.config.seed,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        if self.config.use_response_format:
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_output",
                    "schema": JUDGE_JSON_SCHEMA,
                    "strict": True,
                },
            }
        if self.config.use_extra_body_guided_json:
            kw["extra_body"] = {"guided_json": JUDGE_JSON_SCHEMA}
        return kw

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(**self._build_kwargs(system, user))
        # OpenAI / vLLM return choices[0].message.content as a string. vLLM
        # may surface a refusal as None; treat that as a parse failure upstream.
        content = resp.choices[0].message.content
        return content or ""


def health_check(base_url: str | None = None, timeout: float = 5.0) -> bool:
    """Optional helper — pings the vLLM /health endpoint. Strips trailing
    /v1 from `base_url` if present (vLLM's /health is at the root)."""
    import httpx
    url = base_url or os.environ.get("VAST_JUDGE_URL", "")
    if not url:
        return False
    root = url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        r = httpx.get(f"{root}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
