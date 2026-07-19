"""Trusted-host issuer for Figure 2 safety-adjudication receipts.

This module is deliberately evaluator-only.  It binds a frozen paper rubric,
the current scoring-input authority, and the exact review-document text into a
canonical request.  An evaluator returns strict JSON containing exact quotes;
the trusted host, rather than the evaluator, resolves those quotes to unique
UTF-8 byte spans and seals the canonical request/response beside the resulting
typed adjudications.

The receipt is an integrity artifact under EasyICU's trusted-host model, not a
digital signature against a malicious local issuer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paper_rubric_v2 import (
    FIGURE2_PAPER_RUBRIC_REF,
    Figure2PaperRubricManifest,
    Figure2PaperTaskRubric,
    paper_rubric_manifest_sha256,
)
from .safety_protocol_v1 import (
    FIGURE2_SAFETY_PROTOCOL_REF,
    Figure2ForbiddenClaimResponse,
    Figure2HazardResponse,
    Figure2SafetyResponse,
    parse_figure2_safety_response,
    safety_protocol_sha256,
)
from .scoring_inputs import (
    Figure2ReviewDocument,
    LoadedFigure2ScoringInputs,
)

FIGURE2_SAFETY_REQUEST_SCHEMA = "easyicu.figure2_safety_request/1"
FIGURE2_SAFETY_RECEIPT_SCHEMA = "easyicu.figure2_safety_receipt/1"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Code = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Figure2ResolvedCitation(_StrictFrozenModel):
    """One evaluator quote resolved uniquely in a current review document."""

    evidence_id: str = Field(min_length=1, max_length=256)
    quote: str = Field(min_length=1, max_length=8000)
    start_byte: int = Field(ge=0)
    end_byte: int = Field(ge=1)
    span_sha256: Sha256

    @model_validator(mode="after")
    def _validate_nonempty_span(self) -> "Figure2ResolvedCitation":
        if self.end_byte <= self.start_byte:
            raise ValueError("resolved citation span must be non-empty")
        return self


class Figure2HazardAdjudication(_StrictFrozenModel):
    code: Code
    status: Literal["addressed", "not_addressed"]
    rationale: str = Field(min_length=1, max_length=4000)
    evidence_mode: Literal["cited_spans", "whole_document_review"]
    evidence_spans: tuple[Figure2ResolvedCitation, ...]

    @model_validator(mode="after")
    def _validate_evidence_mode(self) -> "Figure2HazardAdjudication":
        if self.status == "addressed":
            if self.evidence_mode != "cited_spans" or not self.evidence_spans:
                raise ValueError("addressed hazard requires cited evidence spans")
        elif self.evidence_mode != "whole_document_review" or self.evidence_spans:
            raise ValueError("not-addressed hazard requires whole-document review")
        return self


class Figure2ForbiddenClaimAdjudication(_StrictFrozenModel):
    code: Code
    status: Literal["absent", "present"]
    rationale: str = Field(min_length=1, max_length=4000)
    evidence_mode: Literal["cited_spans", "whole_document_review"]
    evidence_spans: tuple[Figure2ResolvedCitation, ...]

    @model_validator(mode="after")
    def _validate_evidence_mode(self) -> "Figure2ForbiddenClaimAdjudication":
        if self.status == "present":
            if self.evidence_mode != "cited_spans" or not self.evidence_spans:
                raise ValueError("present forbidden claim requires cited evidence")
        elif self.evidence_mode != "whole_document_review" or self.evidence_spans:
            raise ValueError("absent forbidden claim requires whole-document review")
        return self


class Figure2SafetyRequest(_StrictFrozenModel):
    """Canonical evaluator request bound to current run and evidence authority."""

    schema_version: Literal["easyicu.figure2_safety_request/1"]
    protocol_ref: Literal["easyicu.figure2_safety_adjudicator_protocol/20260718-v1"]
    protocol_sha256: Sha256
    paper_rubric_ref: Literal["easyicu.figure2_paper_rubric/20260718-v2"]
    paper_rubric_sha256: Sha256
    provider_ref: str = Field(min_length=1, max_length=256)
    model_ref: str = Field(min_length=1, max_length=256)
    task_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=256)
    checkpoint_sequence: int = Field(ge=1)
    checkpoint_payload_sha256: Sha256
    evidence_generation: int = Field(ge=1)
    evidence_payload_sha256: Sha256
    scoring_input_authority_sha256: Sha256
    manuscript_sha256: Sha256
    hazard_codes: tuple[Code, ...] = Field(min_length=1)
    forbidden_claim_codes: tuple[Code, ...] = Field(min_length=1)
    review_documents: tuple[Figure2ReviewDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_request_authority(self) -> "Figure2SafetyRequest":
        if self.protocol_ref != FIGURE2_SAFETY_PROTOCOL_REF:
            raise ValueError("Figure 2 safety protocol reference mismatch")
        if self.protocol_sha256 != safety_protocol_sha256():
            raise ValueError("Figure 2 safety protocol digest mismatch")
        if self.paper_rubric_ref != FIGURE2_PAPER_RUBRIC_REF:
            raise ValueError("Figure 2 paper rubric reference mismatch")
        if len(self.hazard_codes) != len(set(self.hazard_codes)):
            raise ValueError("duplicate hazard code in safety request")
        if len(self.forbidden_claim_codes) != len(set(self.forbidden_claim_codes)):
            raise ValueError("duplicate forbidden-claim code in safety request")
        document_ids = tuple(document.evidence_id for document in self.review_documents)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("duplicate review-document evidence ID")
        manuscripts = tuple(
            document
            for document in self.review_documents
            if document.evidence_id == "manuscript_ready"
        )
        if len(manuscripts) != 1:
            raise ValueError("safety request requires one manuscript_ready document")
        if manuscripts[0].sha256 != self.manuscript_sha256:
            raise ValueError("safety request manuscript digest mismatch")
        if not manuscripts[0].text.strip():
            raise ValueError("safety request manuscript is empty")
        return self


class Figure2SafetyReceipt(_StrictFrozenModel):
    """Canonical provider response plus host-resolved typed adjudications."""

    schema_version: Literal["easyicu.figure2_safety_receipt/1"]
    provider_ref: str = Field(min_length=1, max_length=256)
    model_ref: str = Field(min_length=1, max_length=256)
    request_sha256: Sha256
    request_canonical_json: str = Field(min_length=2)
    response_sha256: Sha256
    response_canonical_json: str = Field(min_length=2)
    hazard_adjudications: tuple[Figure2HazardAdjudication, ...]
    forbidden_claim_adjudications: tuple[Figure2ForbiddenClaimAdjudication, ...]

    @model_validator(mode="after")
    def _validate_sealed_chain(self) -> "Figure2SafetyReceipt":
        request = _request_from_canonical_json(self.request_canonical_json)
        request_bytes = _canonical_model_bytes(request)
        if request_bytes.decode("utf-8") != self.request_canonical_json:
            raise ValueError("safety request payload is not canonical JSON")
        if _sha256_bytes(request_bytes) != self.request_sha256:
            raise ValueError("safety request payload digest mismatch")

        response = _response_from_canonical_json(self.response_canonical_json)
        response_bytes = _canonical_model_bytes(response)
        if response_bytes.decode("utf-8") != self.response_canonical_json:
            raise ValueError("safety response payload is not canonical JSON")
        if _sha256_bytes(response_bytes) != self.response_sha256:
            raise ValueError("safety response payload digest mismatch")
        if self.provider_ref != request.provider_ref:
            raise ValueError("safety receipt provider does not match request")
        if self.model_ref != request.model_ref:
            raise ValueError("safety receipt model does not match request")

        expected_hazards, expected_forbidden = _resolve_response(request, response)
        if _canonical_model_bytes(expected_hazards) != _canonical_model_bytes(
            self.hazard_adjudications
        ):
            raise ValueError("resolved hazard adjudications do not match response")
        if _canonical_model_bytes(expected_forbidden) != _canonical_model_bytes(
            self.forbidden_claim_adjudications
        ):
            raise ValueError(
                "resolved forbidden-claim adjudications do not match response"
            )
        return self

    def parsed_request(self) -> Figure2SafetyRequest:
        return _request_from_canonical_json(self.request_canonical_json)

    def parsed_response(self) -> Figure2SafetyResponse:
        return _response_from_canonical_json(self.response_canonical_json)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_model_bytes(value: object) -> bytes:
    if isinstance(value, BaseModel):
        return _canonical_json_bytes(value.model_dump(mode="json"))
    if isinstance(value, tuple):
        return _canonical_json_bytes(
            [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
        )
    return _canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(payload: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _request_from_canonical_json(payload: str) -> Figure2SafetyRequest:
    parsed = _strict_json_loads(payload.encode("utf-8"))
    return Figure2SafetyRequest.model_validate_json(
        _canonical_json_bytes(parsed), strict=True
    )


def _response_from_canonical_json(payload: str) -> Figure2SafetyResponse:
    return parse_figure2_safety_response(payload.encode("utf-8"))


def _task_is_from_manifest(
    manifest: Figure2PaperRubricManifest,
    task_rubric: Figure2PaperTaskRubric,
) -> bool:
    matches = tuple(
        task for task in manifest.tasks if task.task_id == task_rubric.task_id
    )
    if len(matches) != 1:
        return False
    return _canonical_model_bytes(matches[0]) == _canonical_model_bytes(task_rubric)


def _document_map(
    request: Figure2SafetyRequest,
) -> dict[str, Figure2ReviewDocument]:
    return {document.evidence_id: document for document in request.review_documents}


def _quote_offsets(document_bytes: bytes, quote_bytes: bytes) -> tuple[int, ...]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = document_bytes.find(quote_bytes, start)
        if offset < 0:
            return tuple(offsets)
        offsets.append(offset)
        start = offset + 1


def _resolve_quote(
    request: Figure2SafetyRequest,
    *,
    evidence_id: str,
    quote: str,
) -> Figure2ResolvedCitation:
    document = _document_map(request).get(evidence_id)
    if document is None:
        raise ValueError(f"unknown review-document evidence ID: {evidence_id}")
    document_bytes = document.text.encode("utf-8")
    quote_bytes = quote.encode("utf-8")
    offsets = _quote_offsets(document_bytes, quote_bytes)
    if len(offsets) != 1:
        raise ValueError(
            "evaluator quote must occur exactly once in its review document"
        )
    start_byte = offsets[0]
    end_byte = start_byte + len(quote_bytes)
    return Figure2ResolvedCitation(
        evidence_id=evidence_id,
        quote=quote,
        start_byte=start_byte,
        end_byte=end_byte,
        span_sha256=_sha256_bytes(document_bytes[start_byte:end_byte]),
    )


def _validate_response_code_order(
    request: Figure2SafetyRequest,
    response: Figure2SafetyResponse,
) -> None:
    if (
        tuple(item.code for item in response.hazard_adjudications)
        != request.hazard_codes
    ):
        raise ValueError("hazard response codes must exactly match request order")
    if (
        tuple(item.code for item in response.forbidden_claim_adjudications)
        != request.forbidden_claim_codes
    ):
        raise ValueError(
            "forbidden-claim response codes must exactly match request order"
        )


def _resolve_hazard(
    request: Figure2SafetyRequest,
    response: Figure2HazardResponse,
) -> Figure2HazardAdjudication:
    return Figure2HazardAdjudication(
        code=response.code,
        status=response.status,
        rationale=response.rationale,
        evidence_mode=response.evidence_mode,
        evidence_spans=tuple(
            _resolve_quote(
                request,
                evidence_id=quote.evidence_id,
                quote=quote.quote,
            )
            for quote in response.evidence_quotes
        ),
    )


def _resolve_forbidden_claim(
    request: Figure2SafetyRequest,
    response: Figure2ForbiddenClaimResponse,
) -> Figure2ForbiddenClaimAdjudication:
    return Figure2ForbiddenClaimAdjudication(
        code=response.code,
        status=response.status,
        rationale=response.rationale,
        evidence_mode=response.evidence_mode,
        evidence_spans=tuple(
            _resolve_quote(
                request,
                evidence_id=quote.evidence_id,
                quote=quote.quote,
            )
            for quote in response.evidence_quotes
        ),
    )


def _resolve_response(
    request: Figure2SafetyRequest,
    response: Figure2SafetyResponse,
) -> tuple[
    tuple[Figure2HazardAdjudication, ...],
    tuple[Figure2ForbiddenClaimAdjudication, ...],
]:
    _validate_response_code_order(request, response)
    return (
        tuple(
            _resolve_hazard(request, adjudication)
            for adjudication in response.hazard_adjudications
        ),
        tuple(
            _resolve_forbidden_claim(request, adjudication)
            for adjudication in response.forbidden_claim_adjudications
        ),
    )


def build_figure2_safety_request(
    manifest: Figure2PaperRubricManifest,
    task_rubric: Figure2PaperTaskRubric,
    loaded: LoadedFigure2ScoringInputs,
) -> Figure2SafetyRequest:
    """Build the exact evaluator request from verified current scoring inputs."""

    if manifest.rubric_ref != FIGURE2_PAPER_RUBRIC_REF:
        raise ValueError("Figure 2 paper rubric reference mismatch")
    if manifest.safety_protocol_ref != FIGURE2_SAFETY_PROTOCOL_REF:
        raise ValueError("Figure 2 paper rubric safety protocol mismatch")
    if manifest.safety_protocol_sha256 != safety_protocol_sha256():
        raise ValueError("Figure 2 paper rubric safety protocol digest mismatch")
    if not _task_is_from_manifest(manifest, task_rubric):
        raise ValueError("task rubric is not the exact task in the paper manifest")

    authority = loaded.authority
    manifest_sha256 = paper_rubric_manifest_sha256(manifest)
    if authority.task_id != task_rubric.task_id:
        raise ValueError("scoring authority task does not match safety rubric")
    if (
        authority.suite_ref != manifest.suite_ref
        or authority.suite_projection_sha256 != manifest.suite_projection_sha256
    ):
        raise ValueError("scoring authority suite does not match paper rubric")
    if (
        authority.paper_rubric_ref != manifest.rubric_ref
        or authority.paper_rubric_sha256 != manifest_sha256
    ):
        raise ValueError("scoring authority paper rubric does not match request")
    authority_bytes = _canonical_model_bytes(authority)
    manuscript = authority.artifact("manuscript_ready")
    if (
        len(loaded.manuscript_bytes) != manuscript.byte_count
        or _sha256_bytes(loaded.manuscript_bytes) != manuscript.sha256
    ):
        raise ValueError("loaded manuscript bytes disagree with scoring authority")
    review_documents = tuple(loaded.review_documents)
    if not review_documents:
        raise ValueError("current scoring inputs contain no review documents")
    manuscript_documents = tuple(
        item for item in review_documents if item.evidence_id == "manuscript_ready"
    )
    if len(manuscript_documents) != 1:
        raise ValueError("review corpus requires one manuscript_ready document")
    manuscript_document = manuscript_documents[0]
    if (
        manuscript_document.relative_path != manuscript.relative_path
        or manuscript_document.sha256 != manuscript.sha256
        or manuscript_document.byte_count != manuscript.byte_count
        or manuscript_document.kind != manuscript.kind
        or manuscript_document.producer != manuscript.producer
        or manuscript_document.generation_mode != manuscript.generation_mode
        or manuscript_document.produced_by_step != manuscript.produced_by_step
    ):
        raise ValueError("review manuscript disagrees with scoring authority")
    return Figure2SafetyRequest(
        schema_version=FIGURE2_SAFETY_REQUEST_SCHEMA,
        protocol_ref=FIGURE2_SAFETY_PROTOCOL_REF,
        protocol_sha256=safety_protocol_sha256(),
        paper_rubric_ref=FIGURE2_PAPER_RUBRIC_REF,
        paper_rubric_sha256=manifest_sha256,
        provider_ref=manifest.safety_provider_ref,
        model_ref=manifest.safety_model_ref,
        task_id=task_rubric.task_id,
        run_id=authority.run_id,
        checkpoint_sequence=authority.checkpoint_sequence,
        checkpoint_payload_sha256=authority.checkpoint_payload_sha256,
        evidence_generation=authority.evidence_generation,
        evidence_payload_sha256=authority.evidence_payload_sha256,
        scoring_input_authority_sha256=_sha256_bytes(authority_bytes),
        manuscript_sha256=manuscript.sha256,
        hazard_codes=task_rubric.hazard_codes,
        forbidden_claim_codes=task_rubric.forbidden_claim_codes,
        review_documents=review_documents,
    )


def issue_figure2_safety_receipt(
    request: Figure2SafetyRequest,
    raw_response_bytes: bytes,
    provider_ref: str,
    model_ref: str,
) -> Figure2SafetyReceipt:
    """Strictly parse a provider response and seal host-resolved citations."""

    if type(raw_response_bytes) is not bytes:
        raise TypeError("raw Figure 2 safety response must be bytes")
    request = Figure2SafetyRequest.model_validate(
        request.model_dump(mode="python"), strict=True
    )
    if provider_ref != request.provider_ref:
        raise ValueError("provider reference does not match safety request")
    if model_ref != request.model_ref:
        raise ValueError("model reference does not match safety request")
    response = parse_figure2_safety_response(raw_response_bytes)
    hazards, forbidden = _resolve_response(request, response)
    request_bytes = _canonical_model_bytes(request)
    response_bytes = _canonical_model_bytes(response)
    return Figure2SafetyReceipt(
        schema_version=FIGURE2_SAFETY_RECEIPT_SCHEMA,
        provider_ref=provider_ref,
        model_ref=model_ref,
        request_sha256=_sha256_bytes(request_bytes),
        request_canonical_json=request_bytes.decode("utf-8"),
        response_sha256=_sha256_bytes(response_bytes),
        response_canonical_json=response_bytes.decode("utf-8"),
        hazard_adjudications=hazards,
        forbidden_claim_adjudications=forbidden,
    )


def verify_figure2_safety_receipt(
    receipt: Figure2SafetyReceipt,
    expected_request: Figure2SafetyRequest,
) -> None:
    """Verify a receipt against the exact request authority expected by caller."""

    expected_request = Figure2SafetyRequest.model_validate(
        expected_request.model_dump(mode="python"), strict=True
    )
    expected_bytes = _canonical_model_bytes(expected_request)
    if receipt.request_sha256 != _sha256_bytes(expected_bytes):
        raise ValueError("safety receipt request authority digest mismatch")
    if receipt.request_canonical_json != expected_bytes.decode("utf-8"):
        raise ValueError("safety receipt does not bind the expected request")
    # Re-validation deliberately repeats the full request/response/citation chain
    # instead of trusting that ``receipt`` was constructed by this process.
    Figure2SafetyReceipt.model_validate(receipt.model_dump(mode="python"), strict=True)


__all__ = [
    "FIGURE2_SAFETY_RECEIPT_SCHEMA",
    "FIGURE2_SAFETY_REQUEST_SCHEMA",
    "Figure2ForbiddenClaimAdjudication",
    "Figure2HazardAdjudication",
    "Figure2ResolvedCitation",
    "Figure2SafetyReceipt",
    "Figure2SafetyRequest",
    "build_figure2_safety_request",
    "issue_figure2_safety_receipt",
    "verify_figure2_safety_receipt",
]
