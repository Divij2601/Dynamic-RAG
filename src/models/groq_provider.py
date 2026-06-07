"""
Resilient Groq chat provider.

Single place where Groq chat completions are made,
with retry/backoff for rate limits and transient
errors, plus an optional one-shot fallback to a
secondary model when the primary is rate-limited
beyond the acceptable wait.

Groq enforces several rate limits; per-minute limits
(TPM/RPM) reset quickly so short backoff helps, while
per-day limits (TPD/RPD) do not — for those we honor
the API's suggested wait only if it is short, else we
fall back to another model (which has its own quota)
or raise a clear error.
"""

import re
import time
from typing import Optional, List, Dict

from groq import Groq

try:
    from groq import (
        RateLimitError,
        APIError
    )
except Exception:  # pragma: no cover
    # Defensive: fall back to base Exception if the
    # SDK layout differs.
    RateLimitError = Exception
    APIError = Exception

from src.config import settings
from src.observability.logger import app_logger


def parse_retry_after_seconds(
    message: str
) -> Optional[float]:
    """
    Extract the suggested retry delay from a Groq
    rate-limit message such as
    "Please try again in 17m32.352s" or "in 2.5s".
    Returns seconds, or None if not found.
    """

    match = re.search(
        r"try again in (?:(\d+)m)?([\d.]+)s",
        message or ""
    )

    if not match:
        return None

    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2) or 0)

    return minutes * 60 + seconds


class GroqProvider:
    """
    Resilient Groq chat completion wrapper.
    """

    _client = None

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(
                api_key=settings.GROQ_API_KEY
            )
            app_logger.success(
                "Groq provider client initialized"
            )
        return self._client

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> str:
        """
        Run a chat completion and return the text.
        """

        model = model or settings.DEFAULT_LLM

        if temperature is None:
            temperature = settings.TEMPERATURE

        max_tokens = max_tokens or settings.MAX_TOKENS

        messages: List[Dict[str, str]] = []

        if system:
            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        return self._call_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    # Groq pricing per million tokens (USD) as of 2026.
    # Update if pricing changes. Keys are model IDs.
    _PRICE_PER_1M = {
        "llama-3.3-70b-versatile":
            {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant":
            {"input": 0.05, "output": 0.08},
        "openai/gpt-oss-120b":
            {"input": 0.90, "output": 0.90},
        "qwen/qwen3-32b":
            {"input": 0.29, "output": 0.59},
        # default fallback for unknown models
        "_default":
            {"input": 0.50, "output": 0.70},
    }

    def _token_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:

        pricing = self._PRICE_PER_1M.get(
            model,
            self._PRICE_PER_1M["_default"]
        )

        return (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000

    def _create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:

        response = (
            self._get_client()
            .chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

        # Accumulate token usage for cost tracking.
        usage = response.usage
        if usage:
            cost = self._token_cost(
                model,
                usage.prompt_tokens,
                usage.completion_tokens
            )
            # Store on the thread-local call context so
            # the graph nodes can read it if needed.
            # For now we log it; the trace in persist_node
            # picks up total_cost via get_last_call_cost().
            self._last_cost = cost
            self._last_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens":
                usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6)
            }

        return response.choices[0].message.content

    def get_last_call_cost(self) -> dict:
        """
        Return token usage + cost from the most
        recent _create call. Resets after each read.
        """

        result = getattr(
            self, "_last_usage", {}
        )
        self._last_usage = {}
        self._last_cost = 0.0
        return result

    def _call_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:

        attempts = settings.LLM_MAX_RETRIES
        last_exc = None

        for attempt in range(attempts + 1):

            try:
                return self._create(
                    model,
                    messages,
                    temperature,
                    max_tokens
                )

            except RateLimitError as exc:

                last_exc = exc

                suggested = parse_retry_after_seconds(
                    str(exc)
                )

                backoff = (
                    settings.LLM_BACKOFF_BASE
                    * (2 ** attempt)
                )

                wait = (
                    suggested
                    if suggested is not None
                    else backoff
                )

                too_long = (
                    wait > settings.LLM_RETRY_MAX_WAIT
                )

                if too_long or attempt == attempts:
                    # Backoff won't help (e.g. daily
                    # limit). Try the fallback model
                    # once, then give up.
                    fallback = (
                        settings.LLM_FALLBACK_MODEL
                    )

                    if fallback and fallback != model:
                        app_logger.warning(
                            f"Rate limited on {model} "
                            f"(suggested wait {wait}s); "
                            f"falling back to "
                            f"{fallback}"
                        )
                        try:
                            return self._create(
                                fallback,
                                messages,
                                temperature,
                                max_tokens
                            )
                        except Exception as fexc:
                            last_exc = fexc

                    break

                app_logger.warning(
                    f"Rate limited on {model}; "
                    f"retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/"
                    f"{attempts})"
                )
                time.sleep(wait)

            except APIError as exc:

                # Transient API / connection / 5xx.
                last_exc = exc

                if attempt == attempts:
                    break

                wait = (
                    settings.LLM_BACKOFF_BASE
                    * (2 ** attempt)
                )

                app_logger.warning(
                    f"Transient API error on {model}; "
                    f"retrying in {wait:.1f}s: {exc!r}"
                )
                time.sleep(wait)

        raise last_exc


groq_provider = GroqProvider()
