"""Provider-neutral message and client protocol for research-agent LLMs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..authority.secret_redaction import is_sensitive_key, redact_text_secrets
from ..canonical_json import canonical_json, canonical_sha256 as digest, extract_json_payload


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
    "ModelIdentity",
    "ProviderRefusal",
    "PublicDemoIdentity",
    "RecordedApproval",
    "RecordedResponse",
    "ReplayCassette",
    "ReplayError",
    "StructuredOutputCapabilityError",
    "StructuredOutputRequest",
    "request_digest",
    "scrub_response",
]


SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CASSETTE_SCHEMA = "easyicu.public_demo_model_cassette/1"
MAX_CASSETTE_BYTES = 32 * 1024 * 1024


class ReplayError(RuntimeError):
    """Closed replay failure; exception text contains only codes and digests."""

    def __init__(self, reason_code: str, *, request_sha256: str | None = None):
        self.reason_code = reason_code
        self.request_sha256 = request_sha256
        super().__init__(reason_code + (f": {request_sha256}" if request_sha256 else ""))


def scrub_response(text: str) -> str:
    """Scrub structured transport-like fields and credentials in free text.

    The recorder returns these same bytes to its caller, so recording and
    playback cannot execute different answers because of redaction.
    """
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if (
                    is_sensitive_key(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key), include_generic_key=False)
                    or re.sub(r"[^a-z0-9]", "", key.lower())
                    in {"headers", "baseurl", "accountid", "organizationid", "projectid"}
                ) else clean(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            result = redact_text_secrets(value)
            return re.sub(
                r"(?i)\b(base[_ -]?url|account[_ -]?id|organization[_ -]?id|project[_ -]?id|headers)\s*[:=]\s*[^\r\n,;]+",
                r"\1=[REDACTED]", result,
            )
        return value

    plain_json = True
    try:
        parsed = json.loads(text)
    except ValueError:
        plain_json = False
        try:
            parsed = extract_json_payload(text)
        except ValueError:
            return clean(text)
    cleaned = clean(parsed)
    if cleaned != parsed:
        return canonical_json(cleaned)
    return text if plain_json else clean(text)


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicDemoIdentity(FrozenRecord):
    source_id: str
    database: str
    version: str
    archive_sha256: SHA256


class ModelIdentity(FrozenRecord):
    """Only output-affecting, non-secret model coordinates, never endpoints."""

    provider: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    model: str = Field(min_length=1, max_length=200)
    supports_strict_json_schema: bool = False
    extra_body_json: str = "{}"

    @model_validator(mode="after")
    def safe_coordinates(self):
        if scrub_response(self.model) != self.model or "://" in self.model:
            raise ValueError("secret-bearing model identity")
        extra = json.loads(self.extra_body_json)
        # These are the existing provider adapter's reasoning switches. An
        # unrecognized wire override cannot silently escape the request key.
        if not isinstance(extra, dict) or set(extra) - {"reasoning", "enable_thinking", "chat_template_kwargs"}:
            raise ValueError("unsupported recorded model settings")
        if scrub_response(self.extra_body_json) != self.extra_body_json:
            raise ValueError("secret-bearing model settings")
        return self


def request_digest(
    model: ModelIdentity, messages: Sequence[LLMMessage], *, max_tokens: int = 2048,
    temperature: float = 0.2, structured_output: StructuredOutputRequest | None = None,
    seed: int | None = None, top_p: float | None = None,
) -> str:
    """Bind every argument supported by the provider delivery protocol."""
    return digest({
        "schema_version": "easyicu.model_request/1", "model": model.model_dump(mode="json"),
        "messages": [{"role": message.role, "content": message.content} for message in messages],
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "top_p": top_p,
        "structured_output": structured_output.to_openai_response_format() if structured_output else None,
    })


class RecordedResponse(FrozenRecord):
    request_sha256: SHA256
    response: str
    finish_reason: Literal["stop", "length", "tool_calls", "function_call", "content_filter", "cancelled", "error", "other"] | None = None


class RecordedApproval(FrozenRecord):
    review_id: str = Field(pattern=r"^review-[0-9a-f]{16}$")
    authority_sha256: SHA256
    decision: Literal["approved", "rejected"]
    population_criterion_acceptances: tuple[str, ...] = ()
    acceptance_reuse_authorized: bool = False
    plan_sha256: SHA256 | None = None
    cohort_sha256: SHA256 | None = None

    @model_validator(mode="after")
    def reuse_requires_complete_coordinates(self):
        if self.acceptance_reuse_authorized and (
            self.decision != "approved" or self.plan_sha256 is None or self.cohort_sha256 is None
        ):
            raise ValueError("acceptance reuse requires an approved complete plan and cohort")
        return self


class ReplayCassette(FrozenRecord):
    schema_version: Literal["easyicu.public_demo_model_cassette/1"] = CASSETTE_SCHEMA
    source: PublicDemoIdentity
    model: ModelIdentity
    entries: tuple[RecordedResponse, ...] = ()
    approvals: tuple[RecordedApproval, ...] = ()
    cassette_sha256: SHA256

    @model_validator(mode="after")
    def integrity(self):
        payload = self.model_dump(mode="json", exclude={"cassette_sha256"})
        if digest(payload) != self.cassette_sha256:
            raise ReplayError("model_replay_cassette_digest_mismatch")
        if len({entry.request_sha256 for entry in self.entries}) != len(self.entries):
            raise ReplayError("model_replay_duplicate_request")
        if len({entry.review_id for entry in self.approvals}) != len(self.approvals):
            raise ReplayError("model_replay_duplicate_approval")
        if any(scrub_response(entry.response) != entry.response for entry in self.entries):
            raise ReplayError("model_replay_cassette_contains_secrets")
        return self

    @classmethod
    def create(cls, *, source: PublicDemoIdentity, model: ModelIdentity,
               entries: Sequence[RecordedResponse] = (), approvals: Sequence[RecordedApproval] = ()):
        payload = {
            "schema_version": CASSETTE_SCHEMA, "source": source.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "entries": [entry.model_dump(mode="json") for entry in sorted(entries, key=lambda entry: entry.request_sha256)],
            "approvals": [item.model_dump(mode="json") for item in sorted(approvals, key=lambda item: item.review_id)],
        }
        return cls.model_validate({**payload, "cassette_sha256": digest(payload)})

    @classmethod
    def read(cls, path: Path):
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_CASSETTE_BYTES + 1)
            if len(raw) > MAX_CASSETTE_BYTES:
                raise ReplayError("model_replay_cassette_too_large")
            return cls.model_validate_json(raw)
        except (OSError, ValidationError, ValueError) as exc:
            raise ReplayError("model_replay_cassette_invalid") from exc

    def write(self, path: Path) -> None:
        # Exclusive creation avoids silently replacing an approved recording.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(self.model_dump(mode="json")) + "\n")
