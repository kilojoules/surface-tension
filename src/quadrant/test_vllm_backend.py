"""Unit tests for VLLMBackend wiring.

The point of these tests is NOT to verify vLLM works — it's to lock down:
  (a) request payload shape (deterministic settings, structured output,
      rubric in system slot only, self-report in user slot only)
  (b) config-from-env validation
  (c) the Backend protocol contract

We mock the openai client so no network call happens. Production correctness
is established later by the smoke test against a real deployed endpoint.

Run from src/: python -m pytest quadrant/test_vllm_backend.py
"""
from __future__ import annotations
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quadrant.claim_judge import JUDGE_JSON_SCHEMA, RUBRIC, judge
from quadrant.vllm_backend import VLLMBackend, VLLMBackendConfig


class _MockChatCompletions:
    def __init__(self, capture: list, response: str):
        self.capture = capture
        self.response = response

    def create(self, **kwargs):
        self.capture.append(kwargs)
        class _Msg:
            def __init__(self, content): self.content = content
        class _Choice:
            def __init__(self, content): self.message = _Msg(content)
        class _Resp:
            def __init__(self, content): self.choices = [_Choice(content)]
        return _Resp(self.response)


class _MockClient:
    def __init__(self, capture: list, response: str):
        self.chat = _MockClient._Chat(capture, response)

    class _Chat:
        def __init__(self, capture: list, response: str):
            self.completions = _MockChatCompletions(capture, response)


def _backend(response: str, *, capture: list | None = None):
    cap = capture if capture is not None else []
    cfg = VLLMBackendConfig(base_url="http://test/v1", model="judge")
    return VLLMBackend(cfg, client=_MockClient(cap, response)), cap


# ----------------------- request payload shape ----------------------

def test_chat_sends_system_and_user_separately():
    be, cap = _backend(json.dumps({
        "self_compliance_claim": "no_claim",
        "rule_endorsement": False, "confidence": 0.5,
        "ambiguous": False, "span": "",
    }))
    be.chat(system="SYS", user="USR")
    assert len(cap) == 1
    msgs = cap[0]["messages"]
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user",   "content": "USR"},
    ]


def test_request_uses_deterministic_settings():
    be, cap = _backend(json.dumps({
        "self_compliance_claim": "no_claim",
        "rule_endorsement": False, "confidence": 0.5,
        "ambiguous": False, "span": "",
    }))
    be.chat(system="x", user="y")
    kw = cap[0]
    assert kw["temperature"] == 0.0
    assert kw["seed"] == 0


def test_request_uses_structured_output_json_schema():
    be, cap = _backend(json.dumps({
        "self_compliance_claim": "no_claim",
        "rule_endorsement": False, "confidence": 0.5,
        "ambiguous": False, "span": "",
    }))
    be.chat(system="x", user="y")
    rf = cap[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] is JUDGE_JSON_SCHEMA
    assert rf["json_schema"]["strict"] is True


def test_request_model_matches_config():
    cfg = VLLMBackendConfig(base_url="http://x/v1", model="qwen-judge-30b")
    be = VLLMBackend(cfg, client=_MockClient([], json.dumps({
        "self_compliance_claim": "no_claim", "rule_endorsement": False,
        "confidence": 0.5, "ambiguous": False, "span": "",
    })))
    cap = []
    be._client.chat.completions = _MockChatCompletions(cap, json.dumps({
        "self_compliance_claim": "no_claim", "rule_endorsement": False,
        "confidence": 0.5, "ambiguous": False, "span": "",
    }))
    be.chat(system="x", user="y")
    assert cap[0]["model"] == "qwen-judge-30b"


def test_optional_extra_body_guided_json_off_by_default():
    be, cap = _backend(json.dumps({
        "self_compliance_claim": "no_claim", "rule_endorsement": False,
        "confidence": 0.5, "ambiguous": False, "span": "",
    }))
    be.chat(system="x", user="y")
    assert "extra_body" not in cap[0]


def test_optional_extra_body_guided_json_on_when_enabled():
    cfg = VLLMBackendConfig(
        base_url="http://x/v1", model="judge",
        use_response_format=False, use_extra_body_guided_json=True,
    )
    cap = []
    be = VLLMBackend(cfg, client=_MockClient(cap, json.dumps({
        "self_compliance_claim": "no_claim", "rule_endorsement": False,
        "confidence": 0.5, "ambiguous": False, "span": "",
    })))
    be.chat(system="x", user="y")
    assert "response_format" not in cap[0]
    assert cap[0]["extra_body"]["guided_json"] is JUDGE_JSON_SCHEMA


# ----------------------- end-to-end via judge() --------------------

def test_judge_uses_backend_correctly():
    """judge() should ship RUBRIC as system, self-report as user. End-to-end."""
    payload = json.dumps({
        "self_compliance_claim": "asserts_compliance",
        "rule_endorsement": False, "confidence": 0.92,
        "ambiguous": False, "span": "no loops",
    })
    be, cap = _backend(payload)
    sr = "I avoided all loops."
    out = judge(sr, be)
    assert out.self_compliance_claim == "asserts_compliance"
    # exactly one call, rubric in system, self-report in user
    assert len(cap) == 1
    msgs = cap[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == RUBRIC
    assert msgs[1]["role"] == "user"   and msgs[1]["content"] == sr


def test_none_content_returns_empty_string():
    """vLLM may return None content on a refusal; backend should yield "" so
    judge() can flag the parse error rather than crashing on None."""
    cfg = VLLMBackendConfig(base_url="http://x/v1", model="judge")
    cap = []

    class _NoneMsg:
        content = None
    class _NoneChoice:
        message = _NoneMsg()
    class _NoneResp:
        choices = [_NoneChoice()]

    class _NoneCreate:
        def create(self, **kw):
            cap.append(kw)
            return _NoneResp()

    class _NoneChat:
        completions = _NoneCreate()

    class _NoneClient:
        chat = _NoneChat()

    be = VLLMBackend(cfg, client=_NoneClient())
    result = be.chat(system="x", user="y")
    assert result == ""

    # And judge() over this produces a parse_error, not an exception:
    out = judge("foo", VLLMBackend(cfg, client=_NoneClient()))
    assert out.parse_error is not None


# ----------------------- env-based config --------------------

def test_from_env_requires_url(monkeypatch):
    monkeypatch.delenv("VAST_JUDGE_URL", raising=False)
    with pytest.raises(RuntimeError, match="VAST_JUDGE_URL"):
        VLLMBackendConfig.from_env()


def test_from_env_reads_all_fields(monkeypatch):
    monkeypatch.setenv("VAST_JUDGE_URL", "http://1.2.3.4:8000/v1")
    monkeypatch.setenv("VAST_JUDGE_MODEL", "my-judge")
    monkeypatch.setenv("VAST_JUDGE_KEY", "sk-test")
    monkeypatch.setenv("VAST_JUDGE_TIMEOUT", "30")
    cfg = VLLMBackendConfig.from_env()
    assert cfg.base_url == "http://1.2.3.4:8000/v1"
    assert cfg.model == "my-judge"
    assert cfg.api_key == "sk-test"
    assert cfg.timeout == 30.0
