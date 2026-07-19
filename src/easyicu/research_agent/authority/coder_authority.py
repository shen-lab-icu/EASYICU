"""Out-of-band host authority supplied to the Coder.

``ResearchContext.notes`` is user-controlled prose.  Host-observed schema,
binding, and execution facts must therefore never be recovered by scanning
that prose for marker strings.  This immutable value is constructed only by
the execution host and passed to ``CoderAgent`` as a separate typed argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class HostCoderAuthority:
    """Exact host-owned prompt attachments for one step attempt."""

    attachments: tuple[str, ...] = ()

    def append(self, value: object) -> "HostCoderAuthority":
        """Return a new authority value containing one non-empty attachment."""

        text = str(value or "").strip()
        if not text:
            return self
        return HostCoderAuthority((*self.attachments, text))

    @classmethod
    def from_values(cls, values: Iterable[object]) -> "HostCoderAuthority":
        """Build authority from host-owned values without parsing user prose."""

        authority = cls()
        for value in values:
            authority = authority.append(value)
        return authority

    def render(self) -> str:
        """Render attachments verbatim in their host-declared order."""

        return "\n\n".join(self.attachments)

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON-safe value bound into step authority."""

        return {
            "schema_version": "easyicu.host_coder_authority/1",
            "attachments": list(self.attachments),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "HostCoderAuthority":
        """Validate a capsule-bound authority payload without coercion."""

        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "attachments",
        }:
            raise ValueError("host Coder authority payload has an invalid schema")
        if payload.get("schema_version") != "easyicu.host_coder_authority/1":
            raise ValueError("host Coder authority schema version is unsupported")
        attachments = payload.get("attachments")
        if not isinstance(attachments, list) or any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in attachments
        ):
            raise ValueError("host Coder authority attachments are invalid")
        return cls(tuple(attachments))


__all__ = ["HostCoderAuthority"]
