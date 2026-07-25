"""Provider-neutral message and client protocol for research-agent LLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    """Minimal interface every provider must satisfy."""

    name: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str: ...


__all__ = ["LLMClient", "LLMMessage"]
