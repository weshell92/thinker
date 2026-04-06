"""LLM provider abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract base class that every LLM provider must implement."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat‑completion request and return the raw text response.

        Parameters
        ----------
        system_prompt : str
            The system / instruction message.
        user_prompt : str
            The user message containing the text to analyse.

        Returns
        -------
        str
            The raw completion text (expected to be valid JSON).
        """
        ...
