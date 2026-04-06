"""Ollama provider – placeholder for future implementation."""

from __future__ import annotations

from . import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider backed by a local Ollama instance.

    This is a **stub** – call ``complete()`` to see a helpful error.
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            "OllamaProvider is not implemented yet. "
            "Contributions welcome – see analyzer/providers/ollama_provider.py"
        )
