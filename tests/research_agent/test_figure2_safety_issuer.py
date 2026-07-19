from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.figure2_paper_rubric import (
    Figure2PaperRubricManifest,
    paper_rubric_manifest_sha256,
)
from easyicu.research_agent.figure2_safety_issuer import (
    Figure2SafetyReceipt,
    build_figure2_safety_request,
    issue_figure2_safety_receipt,
    verify_figure2_safety_receipt,
)
from easyicu.research_agent.figure2_scoring_inputs import (
    Figure2ArtifactAuthority,
    Figure2ReviewDocument,
    Figure2ScoringInputAuthority,
    LoadedFigure2ScoringInputs,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _manifest() -> Figure2PaperRubricManifest:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "figure2_canonical9"
        / "figure2_paper_rubric_v2.json"
    )
    return Figure2PaperRubricManifest.model_validate_json(
        path.read_text(encoding="utf-8"), strict=True
    )


def _loaded_inputs() -> LoadedFigure2ScoringInputs:
    manuscript = (
        "The manuscript addresses the missingness hazard explicitly. "
        "It also contains a forbidden overclaim for evaluator testing."
    )
    manuscript_bytes = manuscript.encode("utf-8")
    roles = (
        "run_status",
        "analysis_plan",
        "evidence_audit",
        "numeric_audit",
        "claim_ledger",
        "manuscript_ready",
    )
    contracts = {
        "run_status": ("run_status", "log", "pipeline", "system"),
        "analysis_plan": ("analysis_plan", "log", "planner", "llm"),
        "evidence_audit": ("evidence_audit", "statistic", "pipeline", "system"),
        "numeric_audit": ("numeric_audit", "statistic", "pipeline", "system"),
        "claim_ledger": ("claim_ledger", "table", "pipeline", "system"),
        "manuscript_ready": ("manuscript_ready", "log", "pipeline", "system"),
    }
    artifacts = tuple(
        Figure2ArtifactAuthority(
            role=role,
            evidence_id=contracts[role][0],
            relative_path=(
                "evidence/manuscript_ready.md"
                if role == "manuscript_ready"
                else f"evidence/{role}.json"
            ),
            sha256=(
                _sha(manuscript_bytes)
                if role == "manuscript_ready"
                else _sha(role.encode("utf-8"))
            ),
            byte_count=(len(manuscript_bytes) if role == "manuscript_ready" else 1),
            kind=contracts[role][1],
            producer=contracts[role][2],
            generation_mode=contracts[role][3],
            produced_by_step=None,
        )
        for role in roles
    )
    manifest = _manifest()
    authority = Figure2ScoringInputAuthority(
        schema_version="easyicu.figure2_scoring_input_authority/3",
        task_id=manifest.tasks[0].task_id,
        suite_ref=manifest.suite_ref,
        suite_projection_sha256=manifest.suite_projection_sha256,
        paper_rubric_ref=manifest.rubric_ref,
        paper_rubric_sha256=paper_rubric_manifest_sha256(manifest),
        research_question_sha256=_sha(b"frozen safety test research question"),
        exposure_concept=None,
        outcome_concept=None,
        run_id="run-safety-001",
        checkpoint_sequence=7,
        checkpoint_payload_sha256="1" * 64,
        evidence_generation=4,
        evidence_payload_sha256="2" * 64,
        artifacts=artifacts,
    )
    document = Figure2ReviewDocument(
        evidence_id="manuscript_ready",
        relative_path="evidence/manuscript_ready.md",
        sha256=_sha(manuscript_bytes),
        byte_count=len(manuscript_bytes),
        kind="log",
        producer="pipeline",
        generation_mode="system",
        produced_by_step=None,
        text=manuscript,
    )
    return LoadedFigure2ScoringInputs(
        authority=authority,
        gates={},
        plan_steps=[],
        evidence_audit={},
        numeric_audit={},
        claim_rows=[],
        claim_reference_sets=(),
        current_step_summaries=(),
        manuscript_bytes=manuscript_bytes,
        review_documents=(document,),
    )


def _request():
    manifest = _manifest()
    return build_figure2_safety_request(manifest, manifest.tasks[0], _loaded_inputs())


def _response_payload(
    request,
    *,
    hazard_status: str = "addressed",
    forbidden_status: str = "absent",
) -> bytes:
    hazard_quotes = (
        [
            {
                "evidence_id": "manuscript_ready",
                "quote": "addresses the missingness hazard explicitly",
            }
        ]
        if hazard_status == "addressed"
        else []
    )
    forbidden_quotes = (
        [
            {
                "evidence_id": "manuscript_ready",
                "quote": "forbidden overclaim",
            }
        ]
        if forbidden_status == "present"
        else []
    )
    return _canonical(
        {
            "schema_version": "easyicu.figure2_safety_response/1",
            "hazard_adjudications": [
                {
                    "code": code,
                    "status": hazard_status,
                    "rationale": "Evaluator supplied a complete hazard judgement.",
                    "evidence_mode": (
                        "cited_spans"
                        if hazard_status == "addressed"
                        else "whole_document_review"
                    ),
                    "evidence_quotes": hazard_quotes,
                }
                for code in request.hazard_codes
            ],
            "forbidden_claim_adjudications": [
                {
                    "code": code,
                    "status": forbidden_status,
                    "rationale": "Evaluator supplied a complete claim judgement.",
                    "evidence_mode": (
                        "cited_spans"
                        if forbidden_status == "present"
                        else "whole_document_review"
                    ),
                    "evidence_quotes": forbidden_quotes,
                }
                for code in request.forbidden_claim_codes
            ],
        }
    )


def _receipt(request=None) -> Figure2SafetyReceipt:
    request = request or _request()
    return issue_figure2_safety_receipt(
        request,
        _response_payload(request),
        request.provider_ref,
        request.model_ref,
    )


def test_request_binds_all_current_authority_coordinates_and_ordered_codes() -> None:
    manifest = _manifest()
    loaded = _loaded_inputs()
    request = build_figure2_safety_request(manifest, manifest.tasks[0], loaded)

    assert request.task_id == manifest.tasks[0].task_id
    assert request.run_id == loaded.authority.run_id
    assert request.checkpoint_sequence == loaded.authority.checkpoint_sequence
    assert request.checkpoint_payload_sha256 == "1" * 64
    assert request.evidence_generation == loaded.authority.evidence_generation
    assert request.evidence_payload_sha256 == "2" * 64
    assert (
        request.manuscript_sha256
        == loaded.authority.artifact("manuscript_ready").sha256
    )
    assert request.hazard_codes == manifest.tasks[0].hazard_codes
    assert request.forbidden_claim_codes == manifest.tasks[0].forbidden_claim_codes
    assert request.review_documents == loaded.review_documents


def test_request_builder_rejects_loaded_manuscript_authority_drift() -> None:
    manifest = _manifest()
    loaded = _loaded_inputs()
    drifted = replace(loaded, manuscript_bytes=b"different current manuscript")

    with pytest.raises(ValueError, match="manuscript bytes disagree"):
        build_figure2_safety_request(manifest, manifest.tasks[0], drifted)


def test_request_builder_rejects_scoring_authority_task_drift() -> None:
    manifest = _manifest()
    loaded = _loaded_inputs()
    drifted = replace(
        loaded,
        authority=loaded.authority.model_copy(
            update={"task_id": manifest.tasks[1].task_id}
        ),
    )

    with pytest.raises(ValueError, match="authority task"):
        build_figure2_safety_request(manifest, manifest.tasks[0], drifted)


@pytest.mark.parametrize(
    ("hazard_status", "forbidden_status", "hazard_mode", "forbidden_mode"),
    [
        ("addressed", "absent", "cited_spans", "whole_document_review"),
        ("addressed", "present", "cited_spans", "cited_spans"),
        (
            "not_addressed",
            "absent",
            "whole_document_review",
            "whole_document_review",
        ),
        (
            "not_addressed",
            "present",
            "whole_document_review",
            "cited_spans",
        ),
    ],
)
def test_status_evidence_mode_matrix_is_resolved_by_host(
    hazard_status: str,
    forbidden_status: str,
    hazard_mode: str,
    forbidden_mode: str,
) -> None:
    request = _request()
    receipt = issue_figure2_safety_receipt(
        request,
        _response_payload(
            request,
            hazard_status=hazard_status,
            forbidden_status=forbidden_status,
        ),
        request.provider_ref,
        request.model_ref,
    )

    assert {item.evidence_mode for item in receipt.hazard_adjudications} == {
        hazard_mode
    }
    assert {item.evidence_mode for item in receipt.forbidden_claim_adjudications} == {
        forbidden_mode
    }
    expected_hazard_spans = 1 if hazard_status == "addressed" else 0
    expected_forbidden_spans = 1 if forbidden_status == "present" else 0
    assert all(
        len(item.evidence_spans) == expected_hazard_spans
        for item in receipt.hazard_adjudications
    )
    assert all(
        len(item.evidence_spans) == expected_forbidden_spans
        for item in receipt.forbidden_claim_adjudications
    )
    verify_figure2_safety_receipt(receipt, request)


@pytest.mark.parametrize("mutation", ["omit", "extra", "reverse", "duplicate"])
def test_response_codes_must_exactly_match_request_order(mutation: str) -> None:
    request = _request()
    payload = json.loads(_response_payload(request))
    hazards = payload["hazard_adjudications"]
    if mutation == "omit":
        hazards.pop()
    elif mutation == "extra":
        hazards.append({**hazards[0], "code": "UNREQUESTED_HAZARD"})
    elif mutation == "reverse":
        hazards.reverse()
    else:
        hazards[1] = {**hazards[1], "code": hazards[0]["code"]}

    with pytest.raises((ValueError, ValidationError)):
        issue_figure2_safety_receipt(
            request,
            _canonical(payload),
            request.provider_ref,
            request.model_ref,
        )


@pytest.mark.parametrize(
    "payload_mutator",
    [
        lambda text: text[:-1] + ',"unexpected":true}',
        lambda text: text.replace(
            '"schema_version":"easyicu.figure2_safety_response/1"',
            '"schema_version":"easyicu.figure2_safety_response/1",'
            '"schema_version":"easyicu.figure2_safety_response/1"',
            1,
        ),
        lambda text: text[:-1] + ',"unexpected":NaN}',
    ],
)
def test_raw_response_rejects_extra_duplicate_and_nonfinite_json(
    payload_mutator,
) -> None:
    request = _request()
    payload = payload_mutator(_response_payload(request).decode("utf-8"))
    with pytest.raises((ValueError, ValidationError)):
        issue_figure2_safety_receipt(
            request,
            payload.encode("utf-8"),
            request.provider_ref,
            request.model_ref,
        )


@pytest.mark.parametrize(
    ("target", "update"),
    [
        ("hazard", {"rationale": ""}),
        (
            "hazard",
            {"evidence_mode": "whole_document_review", "evidence_quotes": []},
        ),
        ("forbidden", {"rationale": ""}),
        (
            "forbidden",
            {"status": "present", "evidence_mode": "whole_document_review"},
        ),
    ],
)
def test_response_rejects_empty_rationale_and_status_mode_mismatch(
    target: str,
    update: dict[str, object],
) -> None:
    request = _request()
    payload = json.loads(_response_payload(request))
    key = (
        "hazard_adjudications"
        if target == "hazard"
        else "forbidden_claim_adjudications"
    )
    payload[key][0].update(update)

    with pytest.raises((ValueError, ValidationError)):
        issue_figure2_safety_receipt(
            request,
            _canonical(payload),
            request.provider_ref,
            request.model_ref,
        )


def test_missing_or_ambiguous_quote_is_rejected() -> None:
    request = _request()
    missing = json.loads(_response_payload(request))
    missing["hazard_adjudications"][0]["evidence_quotes"][0][
        "quote"
    ] = "quote that is absent"
    with pytest.raises(ValueError, match="exactly once"):
        issue_figure2_safety_receipt(
            request,
            _canonical(missing),
            request.provider_ref,
            request.model_ref,
        )

    document = request.review_documents[0]
    repeated_text = "duplicate quote and duplicate quote"
    repeated_bytes = repeated_text.encode("utf-8")
    repeated = document.model_copy(
        update={
            "text": repeated_text,
            "sha256": _sha(repeated_bytes),
            "byte_count": len(repeated_bytes),
        }
    )
    ambiguous_request = request.model_copy(
        update={"review_documents": (repeated,), "manuscript_sha256": repeated.sha256}
    )
    ambiguous = json.loads(_response_payload(ambiguous_request))
    ambiguous["hazard_adjudications"][0]["evidence_quotes"][0][
        "quote"
    ] = "duplicate quote"
    with pytest.raises(ValueError, match="exactly once"):
        issue_figure2_safety_receipt(
            ambiguous_request,
            _canonical(ambiguous),
            ambiguous_request.provider_ref,
            ambiguous_request.model_ref,
        )


def test_receipt_cannot_be_hand_built_from_public_ref_or_statuses() -> None:
    request = _request()
    issued = _receipt(request)
    payload = issued.model_dump(mode="python")
    payload["request_canonical_json"] = "{}"
    payload["request_sha256"] = _sha(b"{}")

    with pytest.raises((ValueError, ValidationError)):
        Figure2SafetyReceipt.model_validate(payload, strict=True)


def test_request_protocol_tamper_is_rejected_before_provider_call() -> None:
    request = _request()
    payload = request.model_dump(mode="python")
    payload["protocol_sha256"] = "0" * 64

    with pytest.raises((ValueError, ValidationError), match="protocol digest"):
        type(request).model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "field",
    [
        "request_sha256",
        "response_sha256",
        "provider_ref",
        "model_ref",
    ],
)
def test_receipt_request_response_provider_and_model_tamper_fail(field: str) -> None:
    issued = _receipt()
    payload = issued.model_dump(mode="python")
    payload[field] = "f" * 64 if field.endswith("sha256") else "tampered/ref"

    with pytest.raises((ValueError, ValidationError)):
        Figure2SafetyReceipt.model_validate(payload, strict=True)


def test_resolved_citation_tamper_fails_reconciliation() -> None:
    issued = _receipt()
    payload = issued.model_dump(mode="python")
    span = payload["hazard_adjudications"][0]["evidence_spans"][0]
    span["start_byte"] += 1

    with pytest.raises((ValueError, ValidationError)):
        Figure2SafetyReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sequence",
        "checkpoint_payload_sha256",
        "evidence_generation",
        "evidence_payload_sha256",
        "scoring_input_authority_sha256",
        "manuscript_sha256",
    ],
)
def test_verify_requires_exact_expected_request_authority(field: str) -> None:
    request = _request()
    receipt = _receipt(request)
    original = getattr(request, field)
    changed = original + 1 if isinstance(original, int) else "f" * 64
    drifted = request.model_copy(update={field: changed})

    with pytest.raises(
        ValueError, match="request authority|expected request|manuscript digest"
    ):
        verify_figure2_safety_receipt(receipt, drifted)


def test_issue_requires_provider_and_model_pinned_by_request() -> None:
    request = _request()
    with pytest.raises(ValueError, match="provider"):
        issue_figure2_safety_receipt(
            request,
            _response_payload(request),
            "wrong/provider",
            request.model_ref,
        )
    with pytest.raises(ValueError, match="model"):
        issue_figure2_safety_receipt(
            request,
            _response_payload(request),
            request.provider_ref,
            "wrong/model",
        )
