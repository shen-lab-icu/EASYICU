"""Strict evaluator-only protocol for Figure 2 safety adjudication.

The research Agent never imports this module.  A trusted host constructs the
request, submits it to an evaluator, strictly parses the raw JSON response, and
resolves exact quotes to digest-bound evidence spans before issuing a receipt.
Public hashes are integrity coordinates under the trusted-host artifact model;
they are not a cryptographic signature against a malicious local issuer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIGURE2_SAFETY_PROTOCOL_REF = "easyicu.figure2_safety_adjudicator_protocol/20260718-v1"
FIGURE2_SAFETY_PROTOCOL_SCHEMA = "easyicu.figure2_safety_protocol/1"
FIGURE2_SAFETY_RESPONSE_SCHEMA = "easyicu.figure2_safety_response/1"

Code = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Figure2SafetyProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_safety_protocol/1"]
    protocol_ref: Literal["easyicu.figure2_safety_adjudicator_protocol/20260718-v1"]
    audience: Literal["evaluator_only"]
    agent_visibility: Literal["forbidden"]
    request_schema: Literal["easyicu.figure2_safety_request/1"]
    response_schema: Literal["easyicu.figure2_safety_response/1"]
    current_authority_only: Literal[True]
    exhaustive_fail_closed: Literal[True]
    system_instruction: str = Field(min_length=100)
    uncertainty_policy: Literal["not_addressed_or_present"]
    citation_policy: Literal["addressed_or_present_exact_quotes;absence_whole_document"]


class Figure2SafetyQuote(_StrictFrozenModel):
    evidence_id: str = Field(min_length=1, max_length=256)
    quote: str = Field(min_length=1, max_length=8000)


class Figure2HazardResponse(_StrictFrozenModel):
    code: Code
    status: Literal["addressed", "not_addressed"]
    rationale: str = Field(min_length=1, max_length=4000)
    evidence_mode: Literal["cited_spans", "whole_document_review"]
    evidence_quotes: tuple[Figure2SafetyQuote, ...]

    @model_validator(mode="after")
    def _validate_evidence_mode(self) -> "Figure2HazardResponse":
        if self.status == "addressed":
            if self.evidence_mode != "cited_spans" or not self.evidence_quotes:
                raise ValueError("addressed hazard requires cited evidence")
        elif self.evidence_mode != "whole_document_review" or self.evidence_quotes:
            raise ValueError("not-addressed hazard requires whole-document review")
        return self


class Figure2ForbiddenClaimResponse(_StrictFrozenModel):
    code: Code
    status: Literal["absent", "present"]
    rationale: str = Field(min_length=1, max_length=4000)
    evidence_mode: Literal["cited_spans", "whole_document_review"]
    evidence_quotes: tuple[Figure2SafetyQuote, ...]

    @model_validator(mode="after")
    def _validate_evidence_mode(self) -> "Figure2ForbiddenClaimResponse":
        if self.status == "present":
            if self.evidence_mode != "cited_spans" or not self.evidence_quotes:
                raise ValueError("present forbidden claim requires cited evidence")
        elif self.evidence_mode != "whole_document_review" or self.evidence_quotes:
            raise ValueError("absent forbidden claim requires whole-document review")
        return self


class Figure2SafetyResponse(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_safety_response/1"]
    hazard_adjudications: tuple[Figure2HazardResponse, ...]
    forbidden_claim_adjudications: tuple[Figure2ForbiddenClaimResponse, ...]

    @model_validator(mode="after")
    def _reject_duplicate_codes(self) -> "Figure2SafetyResponse":
        for entries in (
            self.hazard_adjudications,
            self.forbidden_claim_adjudications,
        ):
            codes = tuple(item.code for item in entries)
            if len(codes) != len(set(codes)):
                raise ValueError("duplicate safety response code")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_loads(payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def default_safety_protocol_path() -> Path:
    return Path(__file__).resolve().parents[1] / "figure2_safety_adjudicator_v1.json"


def load_figure2_safety_protocol(
    path: Path | str | None = None,
) -> Figure2SafetyProtocol:
    protocol_path = Path(path) if path is not None else default_safety_protocol_path()
    parsed = _strict_json_loads(protocol_path.read_bytes())
    protocol = Figure2SafetyProtocol.model_validate_json(
        _canonical_json_bytes(parsed), strict=True
    )
    if protocol.protocol_ref != FIGURE2_SAFETY_PROTOCOL_REF:
        raise ValueError("Figure 2 safety protocol reference drifted")
    return protocol


def safety_protocol_sha256() -> str:
    protocol = load_figure2_safety_protocol()
    return hashlib.sha256(
        _canonical_json_bytes(protocol.model_dump(mode="json"))
    ).hexdigest()


def parse_figure2_safety_response(payload: bytes) -> Figure2SafetyResponse:
    parsed = _strict_json_loads(payload)
    return Figure2SafetyResponse.model_validate_json(
        _canonical_json_bytes(parsed), strict=True
    )


__all__ = [
    "FIGURE2_SAFETY_PROTOCOL_REF",
    "FIGURE2_SAFETY_PROTOCOL_SCHEMA",
    "FIGURE2_SAFETY_RESPONSE_SCHEMA",
    "Figure2ForbiddenClaimResponse",
    "Figure2HazardResponse",
    "Figure2SafetyProtocol",
    "Figure2SafetyQuote",
    "Figure2SafetyResponse",
    "default_safety_protocol_path",
    "load_figure2_safety_protocol",
    "parse_figure2_safety_response",
    "safety_protocol_sha256",
]
