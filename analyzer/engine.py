"""ThinkerEngine – orchestrates the full critical‑thinking analysis."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .models import AnalysisResult
from .prompts import get_system_prompt, get_user_prompt
from .providers import BaseProvider
from .providers.openai_provider import ProviderError

logger = logging.getLogger(__name__)


class ThinkerEngine:
    """High‑level facade: accepts user text, returns ``AnalysisResult``.

    Parameters
    ----------
    provider : BaseProvider
        An LLM provider instance (e.g. ``OpenAIProvider``).
    max_retries : int
        How many times to retry if the LLM returns unparseable JSON.
    """

    def __init__(self, provider: BaseProvider, max_retries: int = 2) -> None:
        self.provider = provider
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    def analyze(self, text: str, language: str = "zh") -> AnalysisResult:
        """Run the full 4‑step analysis and return a validated result.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed after retries.
        """
        system_prompt = get_system_prompt(language)
        user_prompt = get_user_prompt(text, language)

        last_error: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                raw = self.provider.complete(system_prompt, user_prompt)
                data = self._parse_json(raw)
                result = AnalysisResult.model_validate(data)
                return result
            except ProviderError as exc:
                if not exc.recoverable:
                    raise  # quota / auth – don't retry
                last_error = exc
                logger.warning("Attempt %d (recoverable): %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(1)

        raise ValueError(
            f"Failed to obtain a valid analysis after {1 + self.max_retries} "
            f"attempt(s). Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Extract and parse JSON from the LLM response.

        Handles the common case where the model wraps JSON in markdown
        code fences (```json ... ```).
        """
        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            # Remove opening fence line
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
            # Remove closing fence
            if text.endswith("```"):
                text = text[: -3].rstrip()
        return json.loads(text)
