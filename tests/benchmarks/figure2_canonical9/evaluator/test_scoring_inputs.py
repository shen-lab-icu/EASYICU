"""Figure 2 scoring-input authority and sidecar contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.figure2_canonical9.evaluator import (
    input_binding_v2,
    scoring_inputs as scoring_inputs_module,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.evidence_snapshot import (
    load_current_evidence_snapshot,
)
from benchmarks.figure2_canonical9.evaluator.paper_rubric_v3 import (
    FIGURE2_PAPER_RUBRIC_REF,
    paper_rubric_manifest_sha256,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (
    FIGURE2_TASK_IDS,
    figure2_suite_projection,
    figure2_suite_projection_sha256,
)
from benchmarks.figure2_canonical9.evaluator.scoring_inputs import (
    FIGURE2_RUN_TASK_AUTHORITY_SCHEMA,
    FIGURE2_SCORING_ARTIFACT_ROLES,
    FIGURE2_SUITE_REF,
    Figure2ArtifactAuthority,
    Figure2RunTaskAuthority,
    Figure2ScoringInputAuthority,
    load_figure2_scoring_inputs,
    seal_figure2_run_task_authority,
)
from tests.support.figure2 import (
    install_ready_input_binding,
    ready_submission_manifest_fields,
    seal_test_run_input_capsule,
)

TASK_ID = "e2_lactate_mortality"
RESEARCH_QUESTION = next(
    str(task["objective"])
    for task in figure2_suite_projection()["tasks"]
    if task["task_id"] == TASK_ID
)
EXPOSURE_CONCEPT = "lactate"
OUTCOME_CONCEPT = "death"
OPERATIONAL_EXPOSURE = "lact_max"


@pytest.fixture(autouse=True)
def _isolated_ready_binding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    selector = tmp_path / "figure2_ready_input_binding.json"
    monkeypatch.setattr(
        input_binding_v2,
        "_canonical_run_input_binding_path",
        lambda: selector,
    )


def _objective(task_id: str) -> str:
    return next(
        str(task["objective"])
        for task in figure2_suite_projection()["tasks"]
        if task["task_id"] == task_id
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load(run_dir: Path, *, task_id: str = TASK_ID):
    return load_figure2_scoring_inputs(run_dir, expected_task_id=task_id)


def _task_authority_sidecar(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("figure2_task_authority.sha256-*.json"))
    assert len(matches) == 1
    return matches[0]


def _fixture_run(
    tmp_path: Path,
    *,
    execution_complete: bool | str = True,
    claim_rows: list[dict[str, str]] | None = None,
    claim_csv_text: str | None = None,
    plan_evidence_id: str = "analysis_plan",
    plan_revision: int = 1,
    evidence_audit_updates: dict[str, object] | None = None,
    numeric_audit_updates: dict[str, object] | None = None,
    gate_updates: dict[str, object] | None = None,
    task_id: str = TASK_ID,
    research_question: str = RESEARCH_QUESTION,
    exposure_concept: str | None = EXPOSURE_CONCEPT,
    outcome_concept: str | None = OUTCOME_CONCEPT,
    operational_exposure: str | None = OPERATIONAL_EXPOSURE,
    operational_target_outcome: str = OUTCOME_CONCEPT,
    claim_alias: str | None = None,
    extra_claim_evidence_id: str | None = None,
    extra_claim_evidence_payload: bytes = b"\x00\x01sealed-binary-evidence",
    per_step_records: list[dict[str, object]] | None = None,
) -> tuple[Path, EvidenceStore]:
    run_dir = tmp_path / "run_authority"
    run_dir.mkdir(parents=True)
    gates = {
        "execution_complete": execution_complete,
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
    gates.update(gate_updates or {})
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
    plan_path = run_dir / f"{plan_evidence_id}.json"
    _write_json(
        plan_path,
        {
            "research_question": research_question,
            "steps": [
                {
                    "step_id": "01_primary",
                    "intent": "Run the locked primary analysis.",
                    "expected_outputs": ["table:primary_result"],
                }
            ],
            "revision": plan_revision,
        },
    )
    evidence_audit: dict[str, object] = {
        "schema_version": "easyicu.evidence_audit/1",
        "evidence_count": 8 + int(extra_claim_evidence_id is not None),
        "kinds": {
            "log": 5,
            "statistic": 2,
            "table": 1,
            **({"figure": 1} if extra_claim_evidence_id is not None else {}),
        },
        "missing_evidence_count": 0,
        "evidence_complete": True,
        "manuscript_path": "manuscript.md",
    }
    evidence_audit.update(evidence_audit_updates or {})
    _write_json(run_dir / "evidence_audit.json", evidence_audit)
    numeric_audit: dict[str, object] = {
        "schema_version": "easyicu.numeric_audit/1",
        "numeric_verified": True,
        "numeric_error_count": 0,
        "numeric_errors": [],
    }
    numeric_audit.update(numeric_audit_updates or {})
    _write_json(run_dir / "numeric_audit.json", numeric_audit)
    rows = claim_rows
    if rows is None:
        rows = [
            {
                "claim_id": "claim_001",
                "claim_text": "The primary result is evidence-bound.",
                "evidence_refs": "manuscript_ready",
                "status": "bound",
                "note": "",
            }
        ]
    if claim_csv_text is not None:
        (run_dir / "claim_ledger.csv").write_text(claim_csv_text, encoding="utf-8")
    else:
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
        "# Results\n\nThe primary result is evidence-bound.\n", encoding="utf-8"
    )

    store = EvidenceStore(run_dir)
    registrations = [
        ("run_status", run_dir / "run_status.json", "log", "pipeline", "system"),
        (
            plan_evidence_id,
            plan_path,
            "log",
            "replanner" if plan_evidence_id != "analysis_plan" else "planner",
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
    if extra_claim_evidence_id is not None:
        extra_path = run_dir / f"{extra_claim_evidence_id}.bin"
        extra_path.write_bytes(extra_claim_evidence_payload)
        registrations.append(
            (
                extra_claim_evidence_id,
                extra_path,
                "figure",
                "pipeline",
                "system",
            )
        )
    for evidence_id, source, kind, producer, mode in registrations:
        store.register_file(
            kind=kind,
            description=f"fixture {evidence_id}",
            source_path=source,
            evidence_id=evidence_id,
            producer=producer,
            generation_mode=mode,
            aliases=(
                [claim_alias]
                if claim_alias is not None and evidence_id == "manuscript_ready"
                else None
            ),
        )
    capsule = seal_test_run_input_capsule(
        run_dir=run_dir,
        evidence=store,
        research_question=research_question,
        primary_exposure=operational_exposure,
        target_outcome=operational_target_outcome,
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
            "run_id": "run_authority",
            "research_question": research_question,
            "started_at": "2026-07-18T00:00:00Z",
            "context_path": "research_context.json",
            **ready_submission_manifest_fields(),
            "readiness": gates,
            "per_step_records": (
                per_step_records
                if per_step_records is not None
                else [
                    {
                        "step_id": "01_primary",
                        "status": "ok",
                        "step_summary": {
                            "status": "ok",
                            "primary_model": {
                                "exposure": exposure_concept,
                                "outcome": outcome_concept,
                            },
                        },
                        "evidence_ids": [],
                    }
                ]
            ),
            "evidence": [record.model_dump(mode="json") for record in store.records()],
        },
    )
    seal_figure2_run_task_authority(
        run_dir,
        task_id=task_id,
        research_question=research_question,
        exposure_concept=exposure_concept,
        outcome_concept=outcome_concept or "",
        operational_exposure=operational_exposure,
    )
    return run_dir, store


def test_loads_exact_current_authority_and_review_corpus(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)

    loaded = _load(run_dir)

    assert tuple(item.role for item in loaded.authority.artifacts) == (
        FIGURE2_SCORING_ARTIFACT_ROLES
    )
    assert loaded.authority.run_id == "run_authority"
    assert loaded.authority.task_id == TASK_ID
    assert loaded.authority.suite_ref == FIGURE2_SUITE_REF
    assert loaded.authority.paper_rubric_ref == FIGURE2_PAPER_RUBRIC_REF
    assert loaded.authority.exposure_concept == EXPOSURE_CONCEPT
    assert loaded.authority.outcome_concept == OUTCOME_CONCEPT
    assert loaded.authority.checkpoint_sequence == 1
    assert loaded.authority.evidence_generation >= 1
    assert loaded.gates["execution_complete"] is True
    assert loaded.plan_steps[0]["step_id"] == "01_primary"
    assert loaded.claim_rows[0]["status"] == "bound"
    assert loaded.claim_rows[0]["evidence_refs"] == "manuscript_ready"
    assert loaded.claim_reference_sets == (("claim_001", ("manuscript_ready",)),)
    assert loaded.current_step_summaries == (
        {
            "status": "ok",
            "primary_model": {
                "exposure": EXPOSURE_CONCEPT,
                "outcome": OUTCOME_CONCEPT,
            },
        },
    )
    assert loaded.manuscript_bytes.startswith(b"# Results")
    assert tuple(item.evidence_id for item in loaded.review_documents) == tuple(
        sorted(
            (
                *FIGURE2_SCORING_ARTIFACT_ROLES,
                "research_context",
                "run_input_capsule",
            )
        )
    )
    manuscript = next(
        item
        for item in loaded.review_documents
        if item.evidence_id == "manuscript_ready"
    )
    assert manuscript.text.encode("utf-8") == loaded.manuscript_bytes
    expected_checkpoint = json.loads((run_dir / "manifest.json").read_text())
    canonical = json.dumps(
        expected_checkpoint,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert (
        loaded.authority.checkpoint_payload_sha256
        == hashlib.sha256(canonical).hexdigest()
    )


def test_rejects_requested_task_mismatch_before_scoring(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    wrong_task = next(task_id for task_id in FIGURE2_TASK_IDS if task_id != TASK_ID)

    with pytest.raises(ValueError, match="question does not match"):
        _load(run_dir, task_id=wrong_task)


def test_rejects_task_question_cross_wire_against_frozen_suite_objective(
    tmp_path: Path,
) -> None:
    wrong_objective = next(
        str(task["objective"])
        for task in figure2_suite_projection()["tasks"]
        if task["task_id"] != TASK_ID
    )
    with pytest.raises(ValueError, match="frozen task objective"):
        _fixture_run(tmp_path, research_question=wrong_objective)


def test_scoring_authority_rejects_task_outside_frozen_suite(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    payload = _load(run_dir).authority.model_dump(mode="python")
    payload["task_id"] = "outside_suite"

    with pytest.raises(ValidationError, match="outside the frozen Figure 2 suite"):
        Figure2ScoringInputAuthority.model_validate(payload, strict=True)


def test_rejects_tampered_task_authority_sidecar(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    sidecar = _task_authority_sidecar(run_dir)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["suite_projection_sha256"] = "0" * 64
    _write_json(sidecar, payload)

    with pytest.raises(OSError, match="digest|size|canonical"):
        _load(run_dir)


def test_rejects_missing_task_authority(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    _task_authority_sidecar(run_dir).unlink()

    with pytest.raises(FileNotFoundError):
        _load(run_dir)


def test_required_concept_coordinates_round_trip_and_missing_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    loaded = _load(run_dir)
    assert loaded.authority.exposure_concept == EXPOSURE_CONCEPT
    assert loaded.authority.outcome_concept == OUTCOME_CONCEPT

    payload = json.loads(_task_authority_sidecar(run_dir).read_text(encoding="utf-8"))
    payload.pop("exposure_concept")

    with pytest.raises(ValidationError):
        Figure2RunTaskAuthority.model_validate(payload, strict=True)


def test_explicit_not_applicable_exposure_round_trips_for_predictor_set_task(
    tmp_path: Path,
) -> None:
    task_id = "m2_mortality_prediction"
    run_dir, _ = _fixture_run(
        tmp_path,
        task_id=task_id,
        research_question=_objective(task_id),
        exposure_concept=None,
        outcome_concept="death",
    )

    loaded = _load(run_dir, task_id=task_id)

    assert loaded.authority.exposure_concept is None
    assert loaded.authority.outcome_concept == "death"


@pytest.mark.parametrize(
    ("field", "value"),
    [("exposure_concept", "vasopressor"), ("outcome_concept", "los_icu")],
)
def test_manual_cross_task_concept_binding_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    coordinates = {
        "task_id": TASK_ID,
        "research_question": RESEARCH_QUESTION,
        "exposure_concept": EXPOSURE_CONCEPT,
        "outcome_concept": OUTCOME_CONCEPT,
        "operational_exposure": OPERATIONAL_EXPOSURE,
    }
    coordinates[field] = value

    with pytest.raises(ValueError, match="concept does not match"):
        seal_figure2_run_task_authority(run_dir, **coordinates)


def test_rejects_blank_optional_concept_coordinate(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    payload = json.loads(_task_authority_sidecar(run_dir).read_text(encoding="utf-8"))
    payload["exposure_concept"] = "  "

    with pytest.raises(ValidationError, match="canonical nonblank"):
        Figure2RunTaskAuthority.model_validate(payload, strict=True)


def test_claim_alias_resolves_to_exact_current_evidence_id(tmp_path: Path) -> None:
    alias = "paper_primary_result"
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_alias=alias,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "The primary result is evidence-bound.",
                "evidence_refs": alias,
                "status": "bound",
                "note": "",
            }
        ],
    )

    loaded = _load(run_dir)

    assert loaded.claim_rows[0]["evidence_refs"] == "manuscript_ready"
    assert loaded.claim_reference_sets == (("claim_001", ("manuscript_ready",)),)


def test_rejects_claim_reference_with_distinct_direct_and_alias_owners(
    tmp_path: Path,
) -> None:
    ambiguous_name = "binary_primary_result"
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_alias=ambiguous_name,
        extra_claim_evidence_id=ambiguous_name,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "The reference has two current owners.",
                "evidence_refs": ambiguous_name,
                "status": "bound",
                "note": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="ambiguous evidence reference"):
        _load(run_dir)


def test_rejects_tampered_non_text_claim_reference(tmp_path: Path) -> None:
    evidence_id = "binary_primary_result"
    run_dir, _ = _fixture_run(
        tmp_path,
        extra_claim_evidence_id=evidence_id,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "The binary result is evidence-bound.",
                "evidence_refs": evidence_id,
                "status": "bound",
                "note": "",
            }
        ],
    )
    snapshot = load_current_evidence_snapshot(run_dir)
    record = next(
        item for item in snapshot.records if item["evidence_id"] == evidence_id
    )
    (run_dir / str(record["relative_path"])).write_bytes(b"tampered-binary")

    with pytest.raises(OSError, match="failed verification"):
        _load(run_dir)


def test_diagnostic_claim_without_reference_remains_explicitly_unbound(
    tmp_path: Path,
) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "This is diagnostic rather than evidence-bound.",
                "evidence_refs": "",
                "status": "diagnostic_only",
                "note": "",
            }
        ],
    )

    loaded = _load(run_dir)

    assert loaded.claim_rows[0]["status"] == "diagnostic_only"
    assert loaded.claim_rows[0]["evidence_refs"] == ""
    assert loaded.claim_reference_sets == (("claim_001", ()),)


def test_rejects_unresolved_or_malformed_claim_evidence_refs(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "The reference is not current evidence.",
                "evidence_refs": "unknown_evidence",
                "status": "bound",
                "note": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="unresolved or stale evidence"):
        _load(run_dir)

    malformed_dir, _ = _fixture_run(
        tmp_path / "malformed",
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "The reference list has an empty component.",
                "evidence_refs": "manuscript_ready;;run_status",
                "status": "bound",
                "note": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="empty evidence reference"):
        _load(malformed_dir)


def test_rejects_alias_or_numeric_only_evidence_generation_drift(
    tmp_path: Path,
) -> None:
    run_dir, store = _fixture_run(tmp_path)
    checkpoint = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    before_coordinates = [
        tuple(
            record.get(field)
            for field in scoring_inputs_module._RECORD_COORDINATE_FIELDS
        )
        for record in checkpoint["evidence"]
    ]

    store.register_numeric_claim(
        value="1.0",
        canonical=1.0,
        evidence_id="manuscript_ready",
        step_id="01_primary",
        source_field="late_numeric_claim",
    )
    after = load_current_evidence_snapshot(run_dir)
    after_coordinates = [
        tuple(
            record.get(field)
            for field in scoring_inputs_module._RECORD_COORDINATE_FIELDS
        )
        for record in after.records
    ]
    assert after_coordinates == before_coordinates

    with pytest.raises(FileNotFoundError, match="figure2_task_authority"):
        _load(run_dir)


def test_current_step_summaries_use_latest_successful_checkpoint_records(
    tmp_path: Path,
) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        per_step_records=[
            {
                "step_id": "01_primary",
                "status": "ok",
                "step_summary": {"status": "ok", "role": "superseded"},
                "evidence_ids": [],
            },
            {
                "step_id": "01_primary",
                "status": "contract_failed",
                "step_summary": {"status": "failed_closed"},
                "evidence_ids": [],
            },
            {
                "step_id": "02_supporting",
                "status": "ok",
                "step_summary": {"status": "ok", "role": "supporting"},
                "evidence_ids": [],
            },
        ],
    )

    loaded = _load(run_dir)

    assert loaded.current_step_summaries == ({"status": "ok", "role": "supporting"},)


def test_artifact_authority_rejects_escaped_path() -> None:
    with pytest.raises(ValidationError, match="contained"):
        Figure2ArtifactAuthority(
            role="run_status",
            evidence_id="run_status",
            relative_path="evidence/../run_status.json",
            sha256="0" * 64,
            byte_count=1,
            kind="log",
            producer="pipeline",
            generation_mode="system",
        )


def test_rejects_uncheckpointed_evidence_generation(tmp_path: Path) -> None:
    run_dir, store = _fixture_run(tmp_path)
    extra = run_dir / "late.txt"
    extra.write_text("not checkpointed", encoding="utf-8")
    store.register_file(
        kind="log",
        description="uncheckpointed generation",
        source_path=extra,
        evidence_id="late_review_note",
        producer="pipeline",
        generation_mode="system",
    )

    with pytest.raises(OSError, match="EvidenceStore generation disagree"):
        _load(run_dir)


def test_rejects_string_false_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _fixture_run(tmp_path, execution_complete="false")


@pytest.mark.parametrize(
    "gate",
    [
        "execution_ok",
        "artifact_valid",
        "scientific_requirement_complete",
        "step_scientific_requirements_complete",
        "paper_authorized",
    ],
)
def test_rejects_false_explicit_completion_axis(
    tmp_path: Path,
    gate: str,
) -> None:
    with pytest.raises(PermissionError, match=gate):
        _fixture_run(tmp_path, gate_updates={gate: False})


def test_rejects_malformed_step_completion_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="step completion state has an invalid schema"):
        _fixture_run(
            tmp_path,
            gate_updates={
                "step_completion_states": [
                    {
                        "schema_version": "easyicu.step_completion_state/1",
                        "step_id": "01_primary",
                        "execution_ok": True,
                    }
                ]
            },
        )


def test_rejects_empty_claim_ledger(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path, claim_rows=[])

    with pytest.raises(ValueError, match="at least one claim row"):
        _load(run_dir)


def test_rejects_bound_claim_without_evidence_refs(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": "Unbound despite its status.",
                "evidence_refs": "",
                "status": "bound",
                "note": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="lacks evidence references"):
        _load(run_dir)


def test_rejects_claim_row_with_missing_trailing_field(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_csv_text=(
            "claim_id,claim_text,evidence_refs,status,note\n"
            "claim_001,Incomplete row,primary_result,bound\n"
        ),
    )

    with pytest.raises(ValueError, match="malformed row"):
        _load(run_dir)


def test_rejects_duplicate_claim_id(tmp_path: Path) -> None:
    row = {
        "claim_id": "claim_001",
        "claim_text": "A nonempty claim.",
        "evidence_refs": "manuscript_ready",
        "status": "bound",
        "note": "",
    }
    run_dir, _ = _fixture_run(tmp_path, claim_rows=[row, dict(row)])

    with pytest.raises(ValueError, match="duplicate claim_id"):
        _load(run_dir)


def test_rejects_empty_claim_text(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        claim_rows=[
            {
                "claim_id": "claim_001",
                "claim_text": " ",
                "evidence_refs": "primary_result",
                "status": "bound",
                "note": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="empty claim_text"):
        _load(run_dir)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"evidence_complete": False}, "complete evidence audit"),
        ({"missing_evidence_count": 1}, "complete evidence audit"),
        ({"evidence_count": 5}, "internally inconsistent"),
        (
            {
                "evidence_count": 5,
                "kinds": {"log": 2, "statistic": 2, "table": 1},
            },
            "count disagrees",
        ),
        (
            {"kinds": {"log": 2, "statistic": 3, "table": 1}},
            "internally inconsistent",
        ),
    ],
)
def test_rejects_inconsistent_evidence_audit(
    tmp_path: Path, updates: dict[str, object], match: str
) -> None:
    run_dir, _ = _fixture_run(tmp_path, evidence_audit_updates=updates)

    with pytest.raises((ValueError, ValidationError), match=match):
        _load(run_dir)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "numeric_verified": True,
            "numeric_error_count": 1,
            "numeric_errors": ["an error"],
        },
        {
            "numeric_verified": False,
            "numeric_error_count": 0,
            "numeric_errors": [],
        },
    ],
)
def test_rejects_inconsistent_numeric_audit(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    run_dir, _ = _fixture_run(tmp_path, numeric_audit_updates=updates)

    with pytest.raises(ValidationError, match="verification state is inconsistent"):
        _load(run_dir)


def test_rejects_plan_evidence_revision_mismatch(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(
        tmp_path,
        plan_evidence_id="analysis_plan_revision_2",
        plan_revision=1,
    )

    with pytest.raises(ValueError, match="does not match plan.revision"):
        _load(run_dir)


def test_rejects_current_record_sha_mismatch(tmp_path: Path) -> None:
    run_dir, _ = _fixture_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["evidence"] if item["evidence_id"] == "numeric_audit"
    )
    record["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(OSError, match="EvidenceStore generation disagree"):
        _load(run_dir)
