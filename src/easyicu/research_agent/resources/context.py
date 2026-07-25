"""Deterministic bounded context assembly with whole-segment eviction only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pydantic import BaseModel, ConfigDict, Field


class ContextBudgetExceeded(RuntimeError):
    """Required context cannot fit without truncating authority-bearing data."""


@dataclass(frozen=True)
class ContextSegment:
    """One independently measurable prompt segment."""

    name: str
    content: str
    priority: int = 100
    required: bool = True
    authority_bound: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("context segment name must be a stable identifier")
        if self.authority_bound and not self.required:
            raise ValueError("authority-bound context segments must be required")


class ContextSegmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    priority: int
    required: bool
    authority_bound: bool
    included: bool


class ContextAssemblyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "easyicu.context_assembly_receipt/1"
    max_bytes: int | None = Field(default=None, ge=0)
    reserved_bytes: int = Field(default=0, ge=0)
    content_bytes: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    segments: tuple[ContextSegmentReceipt, ...]
    truncated_strings: bool = False


@dataclass(frozen=True)
class AssembledContext:
    content: str
    receipt: ContextAssemblyReceipt


class BoundedContextAssembler:
    """Assemble stable segments, evicting only complete optional segments."""

    @staticmethod
    def assemble(
        segments: tuple[ContextSegment, ...],
        *,
        max_bytes: int | None = None,
        reserved_bytes: int = 0,
    ) -> AssembledContext:
        names = [segment.name for segment in segments]
        if len(names) != len(set(names)):
            raise ValueError("context segment names must be unique")
        if reserved_bytes < 0:
            raise ValueError("reserved_bytes must be non-negative")
        included = {segment.name for segment in segments if segment.content}

        def content_bytes() -> int:
            return sum(
                len(segment.content.encode("utf-8"))
                for segment in segments
                if segment.name in included
            )

        if max_bytes is not None:
            optional = sorted(
                (
                    segment
                    for segment in segments
                    if not segment.required and segment.name in included
                ),
                key=lambda segment: (segment.priority, segment.name),
            )
            while reserved_bytes + content_bytes() > max_bytes and optional:
                included.remove(optional.pop(0).name)
            if reserved_bytes + content_bytes() > max_bytes:
                required_names = [
                    segment.name
                    for segment in segments
                    if segment.required and segment.content
                ]
                raise ContextBudgetExceeded(
                    "required context exceeds byte budget without safe whole-segment "
                    f"eviction: total={reserved_bytes + content_bytes()} "
                    f"limit={max_bytes} required={required_names!r}"
                )

        content = "".join(
            segment.content for segment in segments if segment.name in included
        )
        receipts = tuple(
            ContextSegmentReceipt(
                name=segment.name,
                sha256=hashlib.sha256(segment.content.encode("utf-8")).hexdigest(),
                bytes=len(segment.content.encode("utf-8")),
                priority=segment.priority,
                required=segment.required,
                authority_bound=segment.authority_bound,
                included=segment.name in included,
            )
            for segment in segments
        )
        encoded_bytes = len(content.encode("utf-8"))
        return AssembledContext(
            content=content,
            receipt=ContextAssemblyReceipt(
                max_bytes=max_bytes,
                reserved_bytes=reserved_bytes,
                content_bytes=encoded_bytes,
                total_bytes=reserved_bytes + encoded_bytes,
                segments=receipts,
            ),
        )


def bounded_request_metrics(
    *,
    system_content: str,
    base_user_content: str,
    full_user_content: str,
    max_bytes: int,
) -> dict[str, object]:
    """Measure a request as base + additive reviewed-resource segments."""
    if not full_user_content.startswith(base_user_content):
        raise ValueError("full user context must preserve the base prompt prefix")
    resource_content = full_user_content[len(base_user_content) :]
    segments = [
        ContextSegment(
            "planner_contract_and_typed_context",
            base_user_content,
            required=True,
            authority_bound=True,
        )
    ]
    if resource_content:
        segments.append(
            ContextSegment(
                "reviewed_protocol_resources",
                resource_content,
                required=True,
                authority_bound=True,
            )
        )
    system_bytes = len(system_content.encode("utf-8"))
    assembly = BoundedContextAssembler.assemble(
        tuple(segments),
        max_bytes=max_bytes,
        reserved_bytes=system_bytes,
    )
    total_bytes = assembly.receipt.total_bytes
    return {
        "system_bytes": system_bytes,
        "user_bytes": assembly.receipt.content_bytes,
        "total_bytes": total_bytes,
        "approx_input_tokens": (total_bytes + 3) // 4,
        "limit_bytes": max_bytes,
        "segments": {
            segment.name: {
                "bytes": segment.bytes,
                "sha256": segment.sha256,
                "required": segment.required,
                "authority_bound": segment.authority_bound,
                "included": segment.included,
            }
            for segment in assembly.receipt.segments
        },
        "truncated_strings": assembly.receipt.truncated_strings,
    }


__all__ = [
    "AssembledContext",
    "BoundedContextAssembler",
    "ContextAssemblyReceipt",
    "ContextBudgetExceeded",
    "ContextSegment",
    "ContextSegmentReceipt",
    "bounded_request_metrics",
]
