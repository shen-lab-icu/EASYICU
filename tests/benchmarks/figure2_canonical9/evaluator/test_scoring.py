"""Fail-closed Figure 2 paper-scoring contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from benchmarks.figure2_canonical9.evaluator import input_binding_v2, scoring
from benchmarks.figure2_canonical9.evaluator import (
    scoring_inputs as scoring_inputs_module,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.evidence_snapshot import (
    load_current_evidence_snapshot,
)
from benchmarks.figure2_canonical9.evaluator.suite import (
    easyicu_evaluation_protocol_suite,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (
    FIGURE2_DIMENSIONS,
    FIGURE2_TASK_IDS,
    figure2_suite_projection_sha256,
)
from benchmarks.figure2_canonical9.evaluator.paper_rubric_v3 import (
    FIGURE2_PAPER_RUBRIC_REF,
    Figure2PaperRubricManifest,
    Figure2PaperScorecard,
    default_figure2_paper_rubric_path,
    paper_rubric_manifest_sha256,
)
from benchmarks.figure2_canonical9.evaluator.safety_issuer import (
    Figure2SafetyReceipt,
    issue_figure2_safety_receipt,
)
from benchmarks.figure2_canonical9.evaluator.scoring import (
    Figure2EvaluationAttempt,
    build_figure2_safety_request_for_run,
    evaluate_figure2_run,
    evaluate_figure2_run_from_receipt_path,
    verify_figure2_evaluation_attempt,
)
from benchmarks.figure2_canonical9.evaluator.scoring_inputs import (
    load_figure2_scoring_inputs,
    seal_figure2_run_task_authority,
)
from tests.figure2_test_support import (
    install_ready_input_binding,
    ready_submission_manifest_fields,
    seal_test_run_input_capsule,
)

_UNSET = object()


@pytest.fixture(autouse=True)
def _isolated_ready_binding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    selector = tmp_path / "figure2_ready_input_binding.json"
    monkeypatch.setattr(
        input_binding_v2,
        "_canonical_run_input_binding_path",
        lambda: selector,
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _different_sha(value: str) -> str:
    candidate = "0" * 64
    return "f" * 64 if value == candidate else candidate


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _ready_run(
    tmp_path: Path,
    *,
    task_id: str = FIGURE2_TASK_IDS[0],
    plan_steps: list[dict[str, object]] | None = None,
    claim_rows: list[dict[str, str]] | None = None,
    extra_json_artifacts: dict[str, dict[str, object]] | None = None,
    extra_csv_artifacts: dict[str, str] | None = None,
    primary_model_evidence_id: str | None = None,
    exposure_concept: str | None | object = _UNSET,
    outcome_concept: str | None | object = _UNSET,
    operational_exposure: str | None | object = _UNSET,
    operational_target_outcome: str | object = _UNSET,
) -> tuple[Path, EvidenceStore]:
    """Build the same strict checkpoint/EvidenceStore join used in production."""

    task = next(
        item
        for item in easyicu_evaluation_protocol_suite().tasks
        if item.task_id == task_id
    )
    research_question = task.objective
    extras = extra_json_artifacts or {}
    csv_extras = extra_csv_artifacts or {}
    run_dir = tmp_path / f"run_figure2_scoring_{task_id}"
    run_dir.mkdir()
    gates = {
        "execution_complete": True,
        "execution_ok": True,
        "artifact_valid": True,
        "required_step_count": 1,
        "completed_step_count": 1,
        "failed_steps": [],
        "missing_steps": [],
        "scientific_incomplete_steps": [],
        "step_completion_states": [
            {
                "schema_version": "easyicu.step_completion_state/1",
                "step_id": "01_primary",
                "execution_ok": True,
                "outer_status": "ok",
                "summary_status": "ok",
                "scientific_requirement_complete": True,
            }
        ],
        "step_scientific_requirements_complete": True,
        "completion_schema_version": "easyicu.run_completion_axes/1",
        "scientific_requirement_complete": True,
        "manuscript_ready": True,
        "paper_authorized": True,
        "publication_figure_bundle_ready": True,
        "publication_figure_stems": ["primary_result"],
        "replan_budget_exhausted": False,
    }
    _write_json(
        run_dir / "run_status.json",
        {
            "schema_version": "easyicu.run_status/1",
            "status": "publication_ready",
            "strict_fail_closed": True,
            "writer_probe_mode": False,
            "writer_probe_failed_steps": [],
            "research_question": research_question,
            "code_version": {
                "git_sha": None,
                "git_branch": None,
                "git_dirty": None,
                "package_version": "test",
            },
            "gates": gates,
            "canonical_outputs": {},
        },
    )
    sealed_plan_steps = (
        plan_steps
        if plan_steps is not None
        else [
            {
                "step_id": "01_primary",
                "intent": "Run the locked primary analysis and audit.",
                "expected_outputs": [
                    "table:table_one",
                    "figure:primary_result",
                    "table:audit_panel",
                ],
            }
        ]
    )
    _write_json(
        run_dir / "analysis_plan.json",
        {
            "research_question": research_question,
            "steps": sealed_plan_steps,
            "revision": 1,
        },
    )
    _write_json(
        run_dir / "evidence_audit.json",
        {
            "schema_version": "easyicu.evidence_audit/1",
            "evidence_count": 8 + len(extras) + len(csv_extras),
            "kinds": {
                "log": 5,
                "statistic": 2 + len(extras),
                "table": 1 + len(csv_extras),
            },
            "missing_evidence_count": 0,
            "evidence_complete": True,
            "manuscript_path": "manuscript_ready.md",
        },
    )
    _write_json(
        run_dir / "numeric_audit.json",
        {
            "schema_version": "easyicu.numeric_audit/1",
            "numeric_verified": True,
            "numeric_error_count": 0,
            "numeric_errors": [],
        },
    )
    rows = claim_rows or [
        {
            "claim_id": "claim_001",
            "claim_text": "The primary result is evidence-bound.",
            "evidence_refs": "numeric_audit",
            "status": "bound",
            "note": "",
        }
    ]
    with (run_dir / "claim_ledger.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id",
                "claim_text",
                "evidence_refs",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "manuscript_ready.md").write_text(
        "# Results\n\n"
        "The manuscript addresses the relevant hazard explicitly. "
        "The primary result is evidence-bound.\n",
        encoding="utf-8",
    )
    for evidence_id, payload in extras.items():
        _write_json(run_dir / f"{evidence_id}.json", payload)
    for evidence_id, payload in csv_extras.items():
        (run_dir / f"{evidence_id}.csv").write_text(payload, encoding="utf-8")

    store = EvidenceStore(run_dir)
    registrations = [
        ("run_status", run_dir / "run_status.json", "log", "pipeline", "system"),
        (
            "analysis_plan",
            run_dir / "analysis_plan.json",
            "log",
            "planner",
            "llm",
        ),
        (
            "evidence_audit",
            run_dir / "evidence_audit.json",
            "statistic",
            "pipeline",
            "system",
        ),
        (
            "numeric_audit",
            run_dir / "numeric_audit.json",
            "statistic",
            "pipeline",
            "system",
        ),
        (
            "claim_ledger",
            run_dir / "claim_ledger.csv",
            "table",
            "pipeline",
            "system",
        ),
        (
            "manuscript_ready",
            run_dir / "manuscript_ready.md",
            "log",
            "pipeline",
            "system",
        ),
    ]
    registrations.extend(
        (
            evidence_id,
            run_dir / f"{evidence_id}.json",
            "statistic",
            "pipeline",
            "system",
        )
        for evidence_id in extras
    )
    registrations.extend(
        (
            evidence_id,
            run_dir / f"{evidence_id}.csv",
            "table",
            "coder",
            "llm",
        )
        for evidence_id in csv_extras
    )
    for evidence_id, source, kind, producer, generation_mode in registrations:
        store.register_file(
            kind=kind,
            description=f"Figure 2 fixture {evidence_id}",
            source_path=source,
            evidence_id=evidence_id,
            producer=producer,
            generation_mode=generation_mode,
        )
    paper_manifest = Figure2PaperRubricManifest.model_validate_json(
        default_figure2_paper_rubric_path().read_text(encoding="utf-8"),
        strict=True,
    )
    task_rubric = next(task for task in paper_manifest.tasks if task.task_id == task_id)
    resolved_exposure = (
        task_rubric.validity_binding.exposure_concept
        if exposure_concept is _UNSET
        else exposure_concept
    )
    resolved_outcome = (
        task_rubric.validity_binding.outcome_concept
        if outcome_concept is _UNSET
        else outcome_concept
    )
    resolved_operational_exposure = (
        "lact_max" if operational_exposure is _UNSET else operational_exposure
    )
    resolved_operational_outcome = (
        task_rubric.validity_binding.outcome_concept
        if operational_target_outcome is _UNSET
        else operational_target_outcome
    )
    if resolved_operational_exposure is not None and not isinstance(
        resolved_operational_exposure, str
    ):
        raise TypeError("operational exposure must be explicit null or text")
    if not isinstance(resolved_operational_outcome, str):
        raise TypeError("operational target outcome must be text")
    capsule = seal_test_run_input_capsule(
        run_dir=run_dir,
        evidence=store,
        research_question=research_question,
        primary_exposure=resolved_operational_exposure,
        target_outcome=resolved_operational_outcome,
    )
    install_ready_input_binding(
        selector=input_binding_v2._canonical_run_input_binding_path(),
        task_id=task_id,
        research_question=research_question,
        capsule=capsule,
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "easyicu.research_manifest/1",
            "checkpoint_sequence": 1,
            "run_id": run_dir.name,
            "research_question": research_question,
            "started_at": "2026-07-18T00:00:00Z",
            "context_path": "research_context.json",
            **ready_submission_manifest_fields(),
            "readiness": gates,
            "per_step_records": (
                [
                    {
                        "step_id": "02_primary_model",
                        "status": "ok",
                        "evidence_ids": [primary_model_evidence_id],
                        "step_summary": {"primary_model": True},
                    }
                ]
                if primary_model_evidence_id is not None
                else []
            ),
            "evidence": [record.model_dump(mode="json") for record in store.records()],
        },
    )
    seal_figure2_run_task_authority(
        run_dir,
        task_id=task_id,
        research_question=research_question,
        exposure_concept=resolved_exposure,
        outcome_concept=resolved_outcome or "",
        operational_exposure=resolved_operational_exposure,
    )
    return run_dir, store


def _response_payload(
    request: Any,
    *,
    hazard_status: str = "addressed",
    forbidden_status: str = "absent",
) -> bytes:
    hazard_quotes = (
        [
            {
                "evidence_id": "manuscript_ready",
                "quote": "addresses the relevant hazard explicitly",
            }
        ]
        if hazard_status == "addressed"
        else []
    )
    forbidden_quotes = (
        [
            {
                "evidence_id": "manuscript_ready",
                "quote": "The primary result is evidence-bound",
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
                    "rationale": "Complete evaluator judgement for the hazard.",
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
                    "rationale": "Complete evaluator judgement for the claim.",
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


def _receipt(
    run_dir: Path,
    *,
    task_id: str = FIGURE2_TASK_IDS[0],
    hazard_status: str = "addressed",
    forbidden_status: str = "absent",
) -> Figure2SafetyReceipt:
    request = build_figure2_safety_request_for_run(run_dir, task_id=task_id)
    return issue_figure2_safety_receipt(
        request,
        _response_payload(
            request,
            hazard_status=hazard_status,
            forbidden_status=forbidden_status,
        ),
        request.provider_ref,
        request.model_ref,
    )


def _valid_attempt(
    run_dir: Path, *, task_id: str = FIGURE2_TASK_IDS[0]
) -> Figure2EvaluationAttempt:
    attempt = evaluate_figure2_run(
        run_dir,
        task_id=task_id,
        safety_receipt=_receipt(run_dir, task_id=task_id),
    )
    assert attempt.status == "valid", attempt.invalid_details
    assert attempt.envelope is not None
    return attempt


def _tamper_exact_five_payload(
    attempt: Figure2EvaluationAttempt, field: str
) -> Figure2EvaluationAttempt:
    assert attempt.envelope is not None
    paper_scorecard = attempt.envelope.scorecard
    payload = json.loads(paper_scorecard.scorecard_canonical_json)
    if field == "tristate":
        payload[field] = (
            "diagnostic_only"
            if payload[field] != "diagnostic_only"
            else "analysis_only"
        )
    else:
        payload[field]["notes"] = [
            *payload[field].get("notes", []),
            "locally rehashed tamper",
        ]
    encoded = _canonical(payload)
    tampered_scorecard = paper_scorecard.model_copy(
        update={
            "scorecard_sha256": _sha(encoded),
            "scorecard_canonical_json": encoded.decode("utf-8"),
        }
    )
    envelope = attempt.envelope.model_copy(update={"scorecard": tampered_scorecard})
    return attempt.model_copy(update={"envelope": envelope})


def test_no_safety_receipt_is_a_structured_invalid_attempt(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=None,
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SAFETY_ADJUDICATION_MISSING",)
    assert attempt.envelope is None


def test_missing_receipt_path_is_a_structured_invalid_attempt(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)

    attempt = evaluate_figure2_run_from_receipt_path(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SAFETY_ADJUDICATION_MISSING",)
    assert attempt.envelope is None


def test_receipt_path_accepts_one_strict_host_receipt(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)
    receipt = _receipt(run_dir)
    (run_dir / "figure2_safety_receipt.json").write_text(
        receipt.model_dump_json(),
        encoding="utf-8",
    )

    attempt = evaluate_figure2_run_from_receipt_path(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
    )

    assert attempt.status == "valid", attempt.invalid_details
    assert attempt.envelope is not None


@pytest.mark.parametrize(
    "payload", [b"{", b'{"schema_version":"x","schema_version":"y"}']
)
def test_malformed_receipt_path_is_structured_invalid(
    tmp_path: Path,
    payload: bytes,
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    (run_dir / "figure2_safety_receipt.json").write_bytes(payload)

    attempt = evaluate_figure2_run_from_receipt_path(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SAFETY_ADJUDICATION_INVALID",)
    assert attempt.envelope is None


def test_symlinked_receipt_path_is_structured_invalid(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)
    target = tmp_path / "external_receipt.json"
    target.write_text(_receipt(run_dir).model_dump_json(), encoding="utf-8")
    (run_dir / "figure2_safety_receipt.json").symlink_to(target)

    attempt = evaluate_figure2_run_from_receipt_path(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SAFETY_ADJUDICATION_INVALID",)
    assert attempt.envelope is None


def test_fifo_receipt_is_opened_nonblocking_and_structured_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    receipt_path = run_dir / "figure2_safety_receipt.json"
    scoring.os.mkfifo(receipt_path)
    original_open = scoring.os.open

    def asserted_nonblocking_open(path: object, flags: int, *args: object) -> int:
        assert flags & getattr(scoring.os, "O_NONBLOCK", 0)
        return original_open(path, flags, *args)

    monkeypatch.setattr(scoring.os, "open", asserted_nonblocking_open)

    attempt = evaluate_figure2_run_from_receipt_path(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SAFETY_ADJUDICATION_INVALID",)
    assert attempt.envelope is None


def test_same_run_cannot_be_scored_as_a_different_task_before_receipt(
    tmp_path: Path,
) -> None:
    sealed_task_id = FIGURE2_TASK_IDS[0]
    requested_task_id = FIGURE2_TASK_IDS[1]
    run_dir, _ = _ready_run(tmp_path, task_id=sealed_task_id)

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=requested_task_id,
        safety_receipt=None,
    )

    assert attempt.status == "invalid"
    assert attempt.task_id == requested_task_id
    assert attempt.invalid_reason_codes == ("SCORING_INPUT_AUTHORITY_INVALID",)
    assert "task" in attempt.invalid_details[0].lower()
    assert attempt.envelope is None


def test_host_issued_receipt_creates_exact_five_paper_envelope(
    tmp_path: Path,
) -> None:
    run_dir, _ = _ready_run(tmp_path)

    attempt = _valid_attempt(run_dir)
    assert attempt.envelope is not None
    scorecard = attempt.envelope.scorecard.parsed_scorecard()

    assert tuple(item.name for item in scorecard.dimensions()) == FIGURE2_DIMENSIONS
    assert set(scorecard.model_dump(mode="json")) == {
        "task_id",
        "run_id",
        *FIGURE2_DIMENSIONS,
        "tristate",
    }
    assert "reporting_completeness" not in scorecard.model_dump(mode="json")
    assert "fairness_subgroup" not in scorecard.model_dump(mode="json")
    assert scorecard.audit_conclusion_safety.level == "Full"
    verify_figure2_evaluation_attempt(run_dir, attempt)


def test_poor_but_complete_response_is_valid_and_scores_safety_fail(
    tmp_path: Path,
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    receipt = _receipt(
        run_dir,
        hazard_status="not_addressed",
        forbidden_status="present",
    )

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=receipt,
    )

    assert attempt.status == "valid", attempt.invalid_details
    assert attempt.envelope is not None
    scorecard = attempt.envelope.scorecard.parsed_scorecard()
    safety = scorecard.audit_conclusion_safety
    assert safety.level == "Fail"
    assert safety.subscore == 0.0
    assert scorecard.tristate == "analysis_only"


def test_evaluate_caps_gate_reportable_when_required_plan_dimension_fails(
    tmp_path: Path,
) -> None:
    """A manuscript-ready checkpoint cannot override a real paper-cell Fail."""

    run_dir, _ = _ready_run(tmp_path, plan_steps=[])

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=_receipt(run_dir),
    )

    assert attempt.status == "valid", attempt.invalid_details
    assert attempt.envelope is not None
    scorecard = attempt.envelope.scorecard.parsed_scorecard()
    assert scorecard.plan.level == "Fail"
    assert scorecard.audit_conclusion_safety.level == "Full"
    assert scorecard.tristate == "analysis_only"


def test_evaluate_caps_gate_reportable_when_evidence_binding_fails(
    tmp_path: Path,
) -> None:
    """An unbound current claim cannot coexist with paper reportability."""

    run_dir, _ = _ready_run(
        tmp_path,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "This diagnostic claim has no evidence owner.",
                "evidence_refs": "",
                "status": "diagnostic_only",
                "note": "No current evidence reference is available.",
            }
        ],
    )

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=_receipt(run_dir),
    )

    assert attempt.status == "valid", attempt.invalid_details
    assert attempt.envelope is not None
    scorecard = attempt.envelope.scorecard.parsed_scorecard()
    assert scorecard.evidence_binding.level == "Fail"
    assert scorecard.evidence_binding.subscore == 0.0
    assert scorecard.tristate == "analysis_only"


def test_request_and_receipt_bind_manifest_provider_and_model(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)
    request = build_figure2_safety_request_for_run(run_dir, task_id=FIGURE2_TASK_IDS[0])
    manifest_path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks"
        / "figure2_canonical9"
        / "figure2_paper_rubric_v3.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = issue_figure2_safety_receipt(
        request,
        _response_payload(request),
        request.provider_ref,
        request.model_ref,
    )

    assert request.provider_ref == manifest["safety_provider_ref"]
    assert request.model_ref == manifest["safety_model_ref"]
    assert receipt.provider_ref == request.provider_ref
    assert receipt.model_ref == request.model_ref
    assert receipt.parsed_request().provider_ref == manifest["safety_provider_ref"]
    assert receipt.parsed_request().model_ref == manifest["safety_model_ref"]


@pytest.mark.parametrize(
    "field",
    ["plan", "code", "result_validity", "evidence_binding", "tristate"],
)
def test_replay_rejects_locally_rehashed_exact_five_tamper(
    tmp_path: Path, field: str
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    attempt = _valid_attempt(run_dir)
    tampered = _tamper_exact_five_payload(attempt, field)

    with pytest.raises(ValueError, match="differs from deterministic replay"):
        verify_figure2_evaluation_attempt(run_dir, tampered)


@pytest.mark.parametrize(
    "field",
    [
        "rubric_manifest_sha256",
        "suite_projection_sha256",
        "scorer_tree_sha256",
    ],
)
def test_replay_rejects_inner_scorer_authority_tamper(
    tmp_path: Path, field: str
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    attempt = _valid_attempt(run_dir)
    assert attempt.envelope is not None
    current = getattr(attempt.envelope.scorecard, field)
    paper_scorecard = attempt.envelope.scorecard.model_copy(
        update={field: _different_sha(current)}
    )
    envelope = attempt.envelope.model_copy(update={"scorecard": paper_scorecard})
    tampered = attempt.model_copy(update={"envelope": envelope})

    with pytest.raises(ValueError, match="differs from deterministic replay"):
        verify_figure2_evaluation_attempt(run_dir, tampered)


def test_replay_rejects_locally_rehashed_scoring_input_authority_tamper(
    tmp_path: Path,
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    attempt = _valid_attempt(run_dir)
    assert attempt.envelope is not None
    authority = json.loads(attempt.envelope.scoring_input_authority_canonical_json)
    authority["run_id"] = "tampered-run-id"
    encoded = _canonical(authority)
    envelope = attempt.envelope.model_copy(
        update={
            "scoring_input_authority_sha256": _sha(encoded),
            "scoring_input_authority_canonical_json": encoded.decode("utf-8"),
        }
    )
    tampered = attempt.model_copy(update={"envelope": envelope})

    with pytest.raises(ValueError, match="differs from deterministic replay"):
        verify_figure2_evaluation_attempt(run_dir, tampered)


def test_uncheckpointed_newer_evidence_generation_is_rejected(tmp_path: Path) -> None:
    run_dir, store = _ready_run(tmp_path)
    receipt = _receipt(run_dir)
    late = run_dir / "late_review_note.md"
    late.write_text("not present in the selected checkpoint", encoding="utf-8")
    store.register_file(
        kind="log",
        description="uncheckpointed Figure 2 review note",
        source_path=late,
        evidence_id="late_review_note",
        producer="pipeline",
        generation_mode="system",
    )

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=receipt,
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SCORING_INPUT_AUTHORITY_INVALID",)
    assert "EvidenceStore generation disagree" in attempt.invalid_details[0]
    assert attempt.envelope is None


def test_all_bogus_claim_references_can_never_receive_full_evidence_binding(
    tmp_path: Path,
) -> None:
    run_dir, _ = _ready_run(
        tmp_path,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "Every cited reference is fabricated.",
                "evidence_refs": "missing_alpha;missing_beta",
                "status": "bound",
                "note": "",
            }
        ],
    )
    try:
        load_figure2_scoring_inputs(run_dir, expected_task_id=FIGURE2_TASK_IDS[0])
    except ValueError as exc:
        detail = str(exc).lower()
        assert "claim" in detail and "evidence" in detail
        return

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=_receipt(run_dir),
    )
    if attempt.status == "invalid":
        assert attempt.invalid_reason_codes == ("SCORING_INPUT_AUTHORITY_INVALID",)
        return
    assert attempt.envelope is not None
    evidence = attempt.envelope.scorecard.parsed_scorecard().evidence_binding
    assert evidence.level != "Full"


@pytest.mark.parametrize("mutation", ["alias", "numeric_claim"])
def test_same_record_post_checkpoint_evidence_generation_is_rejected_before_receipt(
    tmp_path: Path, mutation: str
) -> None:
    run_dir, store = _ready_run(tmp_path)
    receipt = _receipt(run_dir)
    if mutation == "alias":
        store.register_file(
            kind="log",
            description="same record plus post-checkpoint alias",
            source_path=run_dir / "run_status.json",
            evidence_id="run_status",
            producer="pipeline",
            generation_mode="system",
            aliases=["post_checkpoint_alias"],
        )
    else:
        store.register_numeric_claim(
            value="1",
            canonical=1.0,
            evidence_id="numeric_audit",
            step_id="fixture",
            source_field="estimate",
        )

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=receipt,
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SCORING_INPUT_AUTHORITY_INVALID",)
    assert attempt.envelope is None


@pytest.mark.parametrize(
    "cluster_validity",
    [
        {"n_clusters": 1, "silhouette": 0.5, "algorithm": "kmeans"},
        {"n_clusters": 3, "silhouette": -0.1, "algorithm": "kmeans"},
    ],
)
def test_v2_preserves_objective_degenerate_clustering_validity_failures(
    tmp_path: Path, cluster_validity: dict[str, object]
) -> None:
    task_id = "m3_sepsis_subphenotype"
    run_dir, _ = _ready_run(
        tmp_path,
        task_id=task_id,
        extra_json_artifacts={"cluster_validity": cluster_validity},
    )

    attempt = _valid_attempt(run_dir, task_id=task_id)
    assert attempt.envelope is not None
    scorecard = attempt.envelope.scorecard.parsed_scorecard()

    assert scorecard.result_validity.level == "Fail"
    assert scorecard.result_validity.subscore == 0.0
    assert scorecard.result_validity.signals["validity_errors"]
    assert scorecard.tristate == "analysis_only"


def test_v2_preserves_objective_outcome_leakage_failure(
    tmp_path: Path,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    evidence_id = "primary_model_coefficients"
    run_dir, _ = _ready_run(
        tmp_path,
        task_id=task_id,
        extra_csv_artifacts={evidence_id: "variable,estimate\nage,0.10\ndeath,0.25\n"},
        primary_model_evidence_id=evidence_id,
        outcome_concept="death",
    )

    attempt = _valid_attempt(run_dir, task_id=task_id)
    assert attempt.envelope is not None
    scorecard = attempt.envelope.scorecard.parsed_scorecard()

    assert scorecard.result_validity.level == "Fail"
    assert scorecard.result_validity.subscore == 0.0
    assert any(
        "outcome leakage" in error.lower()
        for error in scorecard.result_validity.signals["validity_errors"]
    )
    assert scorecard.tristate == "analysis_only"


@pytest.mark.parametrize(
    "field",
    [
        "request",
        "response",
        "provider",
        "model",
        "request_provider",
        "request_model",
    ],
)
def test_receipt_request_response_provider_and_model_tamper_is_rejected(
    tmp_path: Path, field: str
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    receipt = _receipt(run_dir)
    if field == "request":
        request = receipt.parsed_request().model_copy(
            update={"run_id": "tampered-run-id"}
        )
        encoded = _canonical(request.model_dump(mode="json"))
        receipt = receipt.model_copy(
            update={
                "request_canonical_json": encoded.decode("utf-8"),
                "request_sha256": _sha(encoded),
            }
        )
    elif field == "response":
        response = receipt.parsed_response()
        first = response.hazard_adjudications[0].model_copy(
            update={"rationale": "Tampered provider response."}
        )
        response = response.model_copy(
            update={
                "hazard_adjudications": (
                    first,
                    *response.hazard_adjudications[1:],
                )
            }
        )
        encoded = _canonical(response.model_dump(mode="json"))
        receipt = receipt.model_copy(
            update={
                "response_canonical_json": encoded.decode("utf-8"),
                "response_sha256": _sha(encoded),
            }
        )
    elif field == "provider":
        receipt = receipt.model_copy(update={"provider_ref": "tampered-provider"})
    elif field == "model":
        receipt = receipt.model_copy(update={"model_ref": "tampered-model"})
    else:
        request = receipt.parsed_request()
        update = (
            {"provider_ref": "tampered-provider"}
            if field == "request_provider"
            else {"model_ref": "tampered-model"}
        )
        request = request.model_copy(update=update)
        receipt = issue_figure2_safety_receipt(
            request,
            _response_payload(request),
            request.provider_ref,
            request.model_ref,
        )

    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=receipt,
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SAFETY_ADJUDICATION_INVALID",)
    assert attempt.envelope is None


def test_scorer_failure_is_structured_and_never_emits_an_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    receipt = _receipt(run_dir)

    def fail_scorer(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic scorer failure")

    monkeypatch.setattr(scoring, "score_run", fail_scorer)
    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=receipt,
    )

    assert attempt.status == "invalid"
    assert attempt.invalid_reason_codes == ("SCORER_ERROR",)
    assert attempt.invalid_details == ("synthetic scorer failure",)
    assert attempt.envelope is None


def test_invalid_attempt_reason_and_detail_cardinality_is_exact() -> None:
    with pytest.raises(ValidationError, match="reason/detail cardinality mismatch"):
        Figure2EvaluationAttempt(
            schema_version="easyicu.figure2_evaluation_attempt/2",
            status="invalid",
            task_id=FIGURE2_TASK_IDS[0],
            invalid_reason_codes=("SCORER_ERROR",),
            invalid_details=(),
        )


def test_attempt_task_identity_must_match_inner_scorecard_task(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)
    attempt = _valid_attempt(run_dir)
    payload = attempt.model_dump(mode="python")
    payload["task_id"] = FIGURE2_TASK_IDS[1]

    with pytest.raises(ValidationError, match="task"):
        Figure2EvaluationAttempt.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    ("subscore", "level"),
    [
        (1.01, "Full"),
        (-0.01, "Fail"),
        (None, "Full"),
        (0.5, None),
        (None, None),
        (0.0, "Full"),
    ],
)
def test_paper_scorecard_rejects_invalid_required_dimension_shape(
    tmp_path: Path,
    subscore: float | None,
    level: str | None,
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    attempt = _valid_attempt(run_dir)
    assert attempt.envelope is not None
    paper = attempt.envelope.scorecard
    exact = json.loads(paper.scorecard_canonical_json)
    exact["plan"]["subscore"] = subscore
    exact["plan"]["level"] = level
    encoded = _canonical(exact)
    payload = paper.model_dump(mode="json")
    payload["scorecard_canonical_json"] = encoded.decode("utf-8")
    payload["scorecard_sha256"] = _sha(encoded)

    with pytest.raises(ValidationError, match="subscore|level|dimension"):
        Figure2PaperScorecard.model_validate(payload, strict=True)


def test_paper_scorecard_rejects_wrong_dimension_identity(tmp_path: Path) -> None:
    run_dir, _ = _ready_run(tmp_path)
    attempt = _valid_attempt(run_dir)
    assert attempt.envelope is not None
    paper = attempt.envelope.scorecard
    exact = json.loads(paper.scorecard_canonical_json)
    exact["plan"]["name"] = "code"
    encoded = _canonical(exact)
    payload = paper.model_dump(mode="json")
    payload["scorecard_canonical_json"] = encoded.decode("utf-8")
    payload["scorecard_sha256"] = _sha(encoded)

    with pytest.raises(ValidationError, match="dimension"):
        Figure2PaperScorecard.model_validate(payload, strict=True)


def test_paper_scorecard_rejects_gate_reportable_when_safety_failed(
    tmp_path: Path,
) -> None:
    run_dir, _ = _ready_run(tmp_path)
    receipt = _receipt(
        run_dir,
        hazard_status="not_addressed",
        forbidden_status="present",
    )
    attempt = evaluate_figure2_run(
        run_dir,
        task_id=FIGURE2_TASK_IDS[0],
        safety_receipt=receipt,
    )
    assert attempt.status == "valid"
    assert attempt.envelope is not None
    paper = attempt.envelope.scorecard
    exact = json.loads(paper.scorecard_canonical_json)
    assert exact["audit_conclusion_safety"]["level"] == "Fail"
    exact["tristate"] = "gate_reportable"
    encoded = _canonical(exact)
    payload = paper.model_dump(mode="json")
    payload["scorecard_canonical_json"] = encoded.decode("utf-8")
    payload["scorecard_sha256"] = _sha(encoded)

    with pytest.raises(ValidationError, match="tristate|safety"):
        Figure2PaperScorecard.model_validate(payload, strict=True)


def test_evaluator_authority_is_absent_from_agent_prompt_surfaces() -> None:
    root = Path(__file__).resolve().parents[4] / "src/easyicu/research_agent"
    protected = [
        root / "agents/core.py",
        root / "providers/prompts/__init__.py",
        *(root / "providers/prompts/v1").glob("*.txt"),
    ]
    forbidden = (
        "figure2_paper_rubric",
        "figure2_safety_issuer",
        "figure2_safety_protocol",
        "figure2_scoring",
        "easyicu.figure2_paper_rubric/",
        "easyicu.figure2_safety_",
    )
    for path in protected:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden), path
