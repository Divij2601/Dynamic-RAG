"""Unit tests for the resilient Groq provider (mocked — no real API calls)."""

import httpx
import pytest

from groq import RateLimitError

import src.models.groq_provider as gp
from src.models.groq_provider import (
    GroqProvider,
    parse_retry_after_seconds,
)


def _rate_limit_error(message: str) -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/v1")
    response = httpx.Response(429, request=request)
    return RateLimitError(message, response=response, body=None)


# ---- retry-after parsing ----

def test_parse_retry_after_minutes_and_seconds():
    assert parse_retry_after_seconds("try again in 1m1s") == 61.0
    assert parse_retry_after_seconds("try again in 2.5s") == 2.5
    assert parse_retry_after_seconds("no hint") is None


# ---- call behavior ----

def test_happy_path(monkeypatch):
    p = GroqProvider()
    monkeypatch.setattr(
        p, "_create",
        lambda model, messages, temperature, max_tokens: "ANSWER"
    )
    assert p.complete("hi", model="m") == "ANSWER"


def test_retry_then_success(monkeypatch):
    p = GroqProvider()
    calls = {"n": 0}

    def fake_create(model, messages, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            # short suggested wait -> should retry
            raise _rate_limit_error("try again in 1s")
        return "OK"

    monkeypatch.setattr(p, "_create", fake_create)
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)

    assert p.complete("hi", model="m") == "OK"
    assert calls["n"] == 2


def test_long_wait_triggers_fallback(monkeypatch):
    p = GroqProvider()
    seen = []

    def fake_create(model, messages, temperature, max_tokens):
        seen.append(model)
        if model == "primary":
            # daily-limit style: very long wait
            raise _rate_limit_error("try again in 40m0s")
        return "FALLBACK_OK"

    monkeypatch.setattr(p, "_create", fake_create)
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        gp.settings, "LLM_FALLBACK_MODEL", "fallback-model"
    )

    out = p.complete("hi", model="primary")
    assert out == "FALLBACK_OK"
    assert "fallback-model" in seen


def test_gives_up_and_raises(monkeypatch):
    p = GroqProvider()

    def always_rate_limited(model, messages, temperature, max_tokens):
        raise _rate_limit_error("try again in 1s")

    monkeypatch.setattr(p, "_create", always_rate_limited)
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)
    # disable fallback so it must raise after retries
    monkeypatch.setattr(gp.settings, "LLM_FALLBACK_MODEL", "")

    with pytest.raises(RateLimitError):
        p.complete("hi", model="primary")
