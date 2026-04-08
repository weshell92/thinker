"""Gemini Native API provider – calls the Gemini REST API directly.

This provider is designed for Gemini relay/proxy endpoints that use the native
Google Gemini API format (``/v1beta/models/{model}:generateContent``) instead of
the OpenAI-compatible format.

Key features:
- Supports comma-separated API keys for load-balancing (round-robin).
- Uses ``x-goog-api-key`` header for authentication.
- Supports SSE streaming via ``?alt=sse``.
- Handles vision (image) content via inline_data parts.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from typing import Any

import requests

from . import BaseProvider
from .openai_provider import ProviderError

logger = logging.getLogger(__name__)

# Default relay URL (can be overridden)
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiNativeProvider(BaseProvider):
    """Provider that calls the Gemini generateContent REST API directly.

    Parameters
    ----------
    api_keys : str
        One or more API keys separated by commas.  The provider will cycle
        through them round-robin style.
    model : str
        Gemini model name, e.g. ``gemini-2.5-pro``.
    base_url : str
        The relay / proxy base URL (without trailing ``/``).
    temperature : float
        Sampling temperature.
    max_output_tokens : int
        Maximum output tokens.
    timeout : float
        HTTP request timeout in seconds.
    """

    _MAX_RETRIES = 2
    _RETRY_DELAY = 3  # seconds

    def __init__(
        self,
        api_keys: str,
        model: str = "gemini-2.5-flash",
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.base_url = base_url.rstrip("/") if base_url else _DEFAULT_BASE_URL

        # Parse comma-separated keys and create a round-robin iterator
        keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        if not keys:
            raise ProviderError("AUTH_ERROR: No API key provided", recoverable=False)
        self._key_cycle = itertools.cycle(keys)
        self._all_keys = keys

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_key(self) -> str:
        return next(self._key_cycle)

    def _build_url(self) -> str:
        """Build the generateContent endpoint URL."""
        return f"{self.base_url}/v1beta/models/{self.model}:generateContent"

    def _convert_messages_to_contents(
        self, messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-style messages to Gemini ``contents`` format.

        Returns (system_instruction_text | None, contents_list).
        """
        system_text: str | None = None
        contents: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini uses systemInstruction at the top level
                if isinstance(content, str):
                    system_text = content
                continue

            # Map OpenAI roles → Gemini roles
            gemini_role = "user" if role == "user" else "model"

            parts: list[dict] = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                # Multimodal content (e.g. vision with images)
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block["text"]})
                        elif block.get("type") == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Parse data URI → inline_data
                                mime, b64 = _parse_data_uri(url)
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime,
                                        "data": b64,
                                    }
                                })
                            else:
                                # URL-based image — include as text reference
                                parts.append({"text": f"[Image: {url}]"})
                    elif isinstance(block, str):
                        parts.append({"text": block})

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        return system_text, contents

    def _call_api(self, messages: list[dict], use_json: bool = False) -> str:
        """Call the Gemini generateContent endpoint with retry logic."""
        system_text, contents = self._convert_messages_to_contents(messages)

        body: dict[str, Any] = {"contents": contents}

        # System instruction
        if system_text:
            body["systemInstruction"] = {
                "parts": [{"text": system_text}]
            }

        # Generation config
        gen_config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
        }
        if use_json:
            gen_config["responseMimeType"] = "application/json"
        body["generationConfig"] = gen_config

        url = self._build_url()
        last_exc: Exception | None = None

        for attempt in range(1 + self._MAX_RETRIES):
            api_key = self._next_key()
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key if len(self._all_keys) == 1 else ",".join(self._all_keys),
            }

            try:
                logger.debug("Gemini API call attempt %d to %s", attempt + 1, url)
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as exc:
                logger.warning("Gemini request timed out (attempt %d)", attempt + 1)
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY)
                    continue
                raise ProviderError(
                    "TIMEOUT: 请求超时，请稍后重试或更换模型。",
                    recoverable=True,
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                logger.warning("Gemini connection error (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY)
                    continue
                raise ProviderError(
                    f"CONNECTION_ERROR: 无法连接到 Gemini 中转服务: {str(exc)[:200]}",
                    recoverable=True,
                ) from exc

            # Handle HTTP errors
            if resp.status_code != 200:
                error_text = resp.text[:500]
                logger.warning(
                    "Gemini API error %d (attempt %d): %s",
                    resp.status_code, attempt + 1, error_text,
                )

                if resp.status_code == 401 or resp.status_code == 403:
                    raise ProviderError("AUTH_ERROR: API Key 无效", recoverable=False)

                if resp.status_code == 429:
                    error_lower = error_text.lower()
                    if "quota" in error_lower or "resource" in error_lower:
                        raise ProviderError(
                            f"QUOTA_EXCEEDED: {error_text[:300]}",
                            recoverable=False,
                        )
                    last_exc = Exception(error_text)
                    if attempt < self._MAX_RETRIES:
                        delay = self._RETRY_DELAY * (2 ** attempt)
                        logger.warning("Rate limited, retrying in %ds…", delay)
                        time.sleep(delay)
                        continue
                    raise ProviderError(
                        f"RATE_LIMIT: {error_text[:300]}",
                        recoverable=True,
                    )

                if resp.status_code >= 500:
                    last_exc = Exception(error_text)
                    if attempt < self._MAX_RETRIES:
                        time.sleep(self._RETRY_DELAY)
                        continue
                    raise ProviderError(
                        f"API_ERROR:{resp.status_code}: {error_text[:300]}",
                        recoverable=True,
                    )

                # 4xx (other than 401/403/429)
                raise ProviderError(
                    f"API_ERROR:{resp.status_code}: {error_text[:300]}",
                    recoverable=False,
                )

            # Parse successful response
            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"API_ERROR: 无法解析 Gemini 响应 JSON: {resp.text[:200]}",
                    recoverable=False,
                ) from exc

            return self._extract_text(data)

        # Exhausted retries
        raise ProviderError(
            f"API_ERROR: 重试 {1 + self._MAX_RETRIES} 次后仍然失败",
            recoverable=False,
        ) from last_exc

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extract the text response from Gemini API JSON."""
        # Check for error in response body
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise ProviderError(f"API_ERROR: {msg[:300]}", recoverable=False)

        candidates = data.get("candidates", [])
        if not candidates:
            # Check if blocked by safety filters
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "")
            if block_reason:
                raise ProviderError(
                    f"API_ERROR: 内容被安全过滤器阻止 (blockReason={block_reason})",
                    recoverable=False,
                )
            raise ProviderError(
                "API_ERROR: Gemini 返回空结果 (no candidates)",
                recoverable=False,
            )

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_parts: list[str] = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

        if not text_parts:
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason == "SAFETY":
                raise ProviderError(
                    "API_ERROR: 响应被安全过滤器阻止 (SAFETY)",
                    recoverable=False,
                )
            return ""

        return "\n".join(text_parts)

    # ------------------------------------------------------------------
    # Public interface (BaseProvider + extras)
    # ------------------------------------------------------------------

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion and return the raw text (expects JSON)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_api(messages, use_json=True)

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion and return free-form text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_api(messages, use_json=False)

    def complete_chat(self, messages: list[dict]) -> str:
        """Send a multi-turn message list and return the assistant reply."""
        return self._call_api(messages, use_json=False)

    def complete_chat_with_vision(self, messages: list[dict]) -> str:
        """Multi-turn with vision support (images as inline_data)."""
        return self._call_api(messages, use_json=False)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _parse_data_uri(data_uri: str) -> tuple[str, str]:
    """Parse a ``data:mime;base64,xxxx`` URI into (mime_type, base64_data)."""
    # data:image/png;base64,iVBOR...
    if not data_uri.startswith("data:"):
        return "application/octet-stream", data_uri
    header, _, b64_data = data_uri.partition(",")
    mime = header.replace("data:", "").replace(";base64", "")
    return mime or "application/octet-stream", b64_data
