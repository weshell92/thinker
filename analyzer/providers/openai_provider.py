"""OpenAI / OpenAI-compatible provider implementation."""

from __future__ import annotations

import logging
import time

from openai import OpenAI, APIStatusError, AuthenticationError, RateLimitError

from . import BaseProvider

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised for unrecoverable provider errors (quota, auth, etc.)."""

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class OpenAIProvider(BaseProvider):
    """Provider backed by the OpenAI Chat Completions API.

    Also works with any OpenAI‑compatible endpoint (e.g. Azure, vLLM,
    LiteLLM proxy, DeepSeek, Gemini) – just pass a custom ``base_url``.
    """

    # Maximum automatic retries for rate-limit (429) errors
    _RATE_LIMIT_MAX_RETRIES = 2
    _RATE_LIMIT_BASE_DELAY = 3  # seconds

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    def _handle_rate_limit(self, exc: RateLimitError) -> None:
        """Inspect a RateLimitError and raise the appropriate ProviderError."""
        body = getattr(exc, "body", None) or {}
        if isinstance(body, dict):
            err = body.get("error", {})
            if isinstance(err, dict):
                code = err.get("code", "")
                msg = err.get("message", "")
            else:
                code, msg = "", str(err)
        else:
            code, msg = "", str(body)

        msg_lower = (str(code) + " " + msg).lower()

        if "insufficient_quota" in msg_lower or "quota" in msg_lower:
            detail = msg[:300] if msg else str(exc)[:300]
            raise ProviderError(f"QUOTA_EXCEEDED: {detail}", recoverable=False) from exc

        # Include raw detail so UI can show the real reason
        detail = msg[:200] if msg else str(exc)[:200]
        raise ProviderError(f"RATE_LIMIT: {detail}", recoverable=True) from exc

    # ------------------------------------------------------------------
    def _call_with_retry(self, messages: list[dict], use_json: bool) -> str:
        """Make the API call with automatic retry on 429 rate-limit errors."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(1 + self._RATE_LIMIT_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""

            except AuthenticationError as exc:
                raise ProviderError("AUTH_ERROR", recoverable=False) from exc

            except RateLimitError as exc:
                last_exc = exc
                body = getattr(exc, "body", None) or {}
                body_str = str(body).lower()
                logger.warning("RateLimitError body: %s", body_str[:500])

                # Check context length exceeded (some providers return 429 for this)
                _CTX_KEYWORDS = (
                    "context_length", "context window", "maximum context",
                    "too many tokens", "max_tokens", "input too large",
                    "token limit", "exceeds.*limit",
                )
                if any(kw in body_str for kw in _CTX_KEYWORDS):
                    raise ProviderError("CONTEXT_TOO_LONG", recoverable=False) from exc

                # Check quota exhausted
                if "insufficient_quota" in body_str or "quota" in body_str:
                    # Extract readable message
                    _msg = ""
                    if isinstance(body, dict):
                        _err = body.get("error", {})
                        if isinstance(_err, dict):
                            _msg = _err.get("message", "")
                    detail = _msg[:300] if _msg else body_str[:300]
                    raise ProviderError(f"QUOTA_EXCEEDED: {detail}", recoverable=False) from exc

                # Retryable rate limit – wait and retry
                if attempt < self._RATE_LIMIT_MAX_RETRIES:
                    delay = self._RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Rate limited (attempt %d/%d), retrying in %ds…",
                        attempt + 1, self._RATE_LIMIT_MAX_RETRIES + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                # Out of retries
                self._handle_rate_limit(exc)

            except APIStatusError as exc:
                # Some providers return 400 for context length exceeded
                exc_str = str(getattr(exc, "body", "")).lower() + str(getattr(exc, "message", "")).lower()
                logger.warning("APIStatusError %d: %s", exc.status_code, exc_str[:500])
                _CTX_KEYWORDS_API = (
                    "context_length", "context window", "maximum context",
                    "too many tokens", "max_tokens", "input too large",
                    "token limit",
                )
                if any(kw in exc_str for kw in _CTX_KEYWORDS_API):
                    raise ProviderError("CONTEXT_TOO_LONG", recoverable=False) from exc

                raise ProviderError(
                    f"API_ERROR:{exc.status_code}",
                    recoverable=exc.status_code >= 500,
                ) from exc

        # Should not reach here, but just in case
        raise ProviderError(
            f"RATE_LIMIT (after {self._RATE_LIMIT_MAX_RETRIES + 1} attempts)",
            recoverable=False,
        ) from last_exc

    # ------------------------------------------------------------------
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._call_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            use_json=True,
        )

    # ------------------------------------------------------------------
    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Like complete(), but returns free-form text (no JSON mode)."""
        return self._call_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            use_json=False,
        )

