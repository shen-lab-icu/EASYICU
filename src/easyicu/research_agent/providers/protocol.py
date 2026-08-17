"""Provider-neutral message and client protocol for research-agent LLMs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class StructuredOutputCapabilityError(RuntimeError):
    """Raised before transport when a client cannot honor a strict schema."""


class ProviderRefusal(RuntimeError):
    """Terminal provider response that declined to produce model content."""

    reason_code = "provider_refusal"

    def __init__(
        self,
        refusal_reason: str,
        *,
        finish_reason: str | None,
        usage_summary: Mapping[str, int] | None,
        transport_attempts: int,
    ) -> None:
        self.refusal_reason = str(refusal_reason or "").strip()
        self.finish_reason = finish_reason
        safe_usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = (usage_summary or {}).get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                safe_usage[key] = int(value)
        self.usage_summary = safe_usage
        self.transport_attempts = max(1, int(transport_attempts))
        self.easyicu_transport_attempts = self.transport_attempts
        super().__init__(
            "ProviderRefusal[reason_code=provider_refusal, "
            f"finish_reason={finish_reason or 'unknown'}]: {self.refusal_reason}"
        )


@dataclass(frozen=True)
class StructuredOutputRequest:
    """Immutable provider-neutral authority for one strict JSON response.

    The schema is stored as canonical JSON rather than as a mutable mapping so
    every wrapper accounts and fingerprints the exact same request. Concrete
    providers translate it to their own wire representation at the final
    transport boundary.
    """

    name: str
    schema_json: str
    strict: bool = True

    @classmethod
    def from_schema(
        cls,
        *,
        name: str,
        schema: Mapping[str, Any],
        strict: bool = True,
    ) -> "StructuredOutputRequest":
        normalized_name = str(name or "").strip()
        if not normalized_name or len(normalized_name) > 64 or not all(
            char.isalnum() or char in {"_", "-"} for char in normalized_name
        ):
            raise ValueError(
                "structured-output name must be 1..64 letters, digits, '_' or '-'"
            )
        if not isinstance(schema, Mapping):
            raise TypeError("structured-output schema must be a mapping")
        schema_json = json.dumps(
            dict(schema),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(schema_json)
        if not isinstance(decoded, dict) or decoded.get("type") != "object":
            raise ValueError("structured-output root schema must be an object")
        return cls(
            name=normalized_name,
            schema_json=schema_json,
            strict=bool(strict),
        )

    @property
    def authority_sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload_json.encode("utf-8")).hexdigest()

    @property
    def canonical_payload_json(self) -> str:
        return json.dumps(
            self.to_openai_response_format(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def payload_bytes(self) -> int:
        return len(self.canonical_payload_json.encode("utf-8"))

    def to_openai_response_format(self) -> dict[str, Any]:
        """Return a fresh OpenAI-compatible wire payload."""

        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "strict": self.strict,
                "schema": json.loads(self.schema_json),
            },
        }


class LLMClient(Protocol):
    """Minimal interface every provider must satisfy."""

    name: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        structured_output: StructuredOutputRequest | None = None,
    ) -> str: ...


__all__ = [
    "LLMClient",
    "LLMMessage",
    "ProviderRefusal",
    "StructuredOutputCapabilityError",
    "StructuredOutputRequest",
]
