"""Contracts for sealing posthoc Figure 2 task authority."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.evaluator import input_binding_v2
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.evidence_snapshot import (
    load_current_evidence_snapshot,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (
    figure2_suite_projection,
)
from benchmarks.figure2_canonical9.evaluator.scoring_inputs import (
    Figure2RunTaskAuthority,
    seal_figure2_run_task_authority,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    encode_step_attempt_history_jsonl,
    load_run_artifact_authority,
    write_run_checkpoint,
)
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
)
from easyicu.research_agent.schema import AnalysisManifest
from tests.figure2_test_support import (
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


def _ready_gates() -> dict[str, object]:
    return {
        "execution_complete": True,
        "execution_ok": True,
        "artifact_valid": True,
        "required_step_count": 0,
        "completed_step_count": 0,
        "failed_steps": [],
        "missing_steps": [],
        "scientific_incomplete_steps": [],
        "step_completion_states": [],
        "step_scientific_requirements_complete": True,
        "completion_schema_version": "easyicu.run_completion_axes/1",
        "scientific_requirement_complete": True,
        "manuscript_ready": True,
        "paper_authorized": True,
        "publication_figure_bundle_ready": True,
        "publication_figure_stems": ["primary_result"],
        "replan_budget_exhausted": False,
    }


def _production_manifest(
    run_dir: Path,
    evidence: EvidenceStore,
    *,
    readiness: dict[str, object] | None = None,
) -> AnalysisManifest:
    return AnalysisManifest(
        run_id=run_dir.name,
        research_question=RESEARCH_QUESTION,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        context_path="context.json",
        **ready_submission_manifest_fields(),
        per_step_records=[],
        evidence=[record.model_dump(mode="json") for record in evidence.records()],
        readiness=dict(_ready_gates() if readiness is None else readiness),
    )


def _build_run(
    tmp_path: Path,
    *,
    run_name: str = "run_authority",
    manifest_readiness: dict[str, object] | None = None,
    run_status_readiness: dict[str, object] | None = None,
    capsule_question: str = RESEARCH_QUESTION,
    capsule_exposure: str | None = OPERATIONAL_EXPOSURE,
    capsule_outcome: str = OUTCOME_CONCEPT,
    seal_capsule: bool = True,
    manifest_authority_updates: dict[str, object] | None = None,
    externalize_attempt_history: bool = False,
) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir()

    source = run_dir / "seed.json"
    source.write_text('{"ready":true}\n', encoding="utf-8")
    EvidenceStore(run_dir).register_file(
        kind="log",
        description="Fixture evidence for the selected generation.",
        source_path=source,
        evidence_id="seed",
        producer="pipeline",
        generation_mode="system",
    )
    evidence = EvidenceStore(run_dir)
    manifest_gates = dict(
        _ready_gates() if manifest_readiness is None else manifest_readiness
    )
    status_gates = dict(
        manifest_gates if run_status_readiness is None else run_status_readiness
    )
    run_status = run_dir / "run_status.json"
    run_status.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.run_status/1",
                "status": "manuscript_ready",
                "strict_fail_closed": True,
                "writer_probe_mode": False,
                "writer_probe_failed_steps": [],
                "research_question": RESEARCH_QUESTION,
                "code_version": {},
                "gates": status_gates,
                "canonical_outputs": {},
            }
        ),
        encoding="utf-8",
    )
    evidence.register_file(
        kind="log",
        description="Completed fixture run status.",
        source_path=run_status,
        evidence_id="run_status",
        producer="pipeline",
        generation_mode="system",
    )
    if seal_capsule:
        seal_test_run_input_capsule(
            run_dir=run_dir,
            evidence=evidence,
            research_question=capsule_question,
            primary_exposure=capsule_exposure,
            target_outcome=capsule_outcome,
        )
    donor_dir = tmp_path / f"{run_name}_binding_donor"
    donor_dir.mkdir()
    donor_evidence = EvidenceStore(donor_dir)
    donor_capsule = seal_test_run_input_capsule(
        run_dir=donor_dir,
        evidence=donor_evidence,
        research_question=RESEARCH_QUESTION,
        primary_exposure=OPERATIONAL_EXPOSURE,
        target_outcome=OUTCOME_CONCEPT,
    )
    install_ready_input_binding(
        selector=input_binding_v2._canonical_run_input_binding_path(),
        task_id=TASK_ID,
        research_question=RESEARCH_QUESTION,
        capsule=donor_capsule,
    )

    history_ref = None
    if externalize_attempt_history:
        history_rows = [
            {
                "step_id": "00_fixture_step",
                "attempt": 1,
                "status": "ok",
            }
        ]
        history_record = evidence.register_text(
            kind="log",
            description="Externalized fixture step-attempt history.",
            text=encode_step_attempt_history_jsonl(history_rows),
            filename="step_attempt_history.jsonl",
            evidence_id="step_attempt_history",
            producer="pipeline",
            generation_mode="system",
            publish_aliases=False,
        )
        history_ref = {
            "schema_version": "easyicu.step_attempt_history_ref/1",
            "format": "jsonl",
            "evidence_id": history_record.evidence_id,
            "relative_path": history_record.relative_path,
            "sha256": history_record.sha256,
            "record_count": len(history_rows),
        }

    manifest = _production_manifest(
        run_dir,
        evidence,
        readiness=manifest_gates,
    )
    payload = manifest.model_dump(mode="json")
    if history_ref is not None:
        payload["step_attempt_history"] = []
        payload["step_attempt_history_ref"] = history_ref
    payload.update(manifest_authority_updates or {})
    assert "figure2_task_authority" not in payload
    assert write_run_checkpoint(run_dir / "manifest.json", payload) == 1
    AnalysisManifest.model_validate_json(
        (run_dir / "manifest.json").read_bytes(),
        strict=True,
    )
    return run_dir


@pytest.fixture
def completed_run(tmp_path: Path) -> Path:
    return _build_run(tmp_path)


def _seal(run_dir: Path, **updates: object) -> Figure2RunTaskAuthority:
    coordinates = {
        "task_id": TASK_ID,
        "research_question": RESEARCH_QUESTION,
        "exposure_concept": EXPOSURE_CONCEPT,
        "outcome_concept": OUTCOME_CONCEPT,
        "operational_exposure": OPERATIONAL_EXPOSURE,
    }
    coordinates.update(updates)
    return seal_figure2_run_task_authority(run_dir, **coordinates)


def _checkpoint_state(run_dir: Path) -> tuple[bytes, int]:
    payload = (run_dir / "manifest.json").read_bytes()
    selected = load_run_artifact_authority(run_dir)
    assert selected is not None
    return payload, int(selected["checkpoint_sequence"])


def _assert_checkpoint_unchanged(
    run_dir: Path,
    before: tuple[bytes, int],
) -> None:
    expected_payload, expected_sequence = before
    assert (run_dir / "manifest.json").read_bytes() == expected_payload
    selected = load_run_artifact_authority(run_dir)
    assert selected is not None
    assert selected["checkpoint_sequence"] == expected_sequence
    assert "figure2_task_authority" not in selected


def _task_authority_sidecars(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("figure2_task_authority.sha256-*.json"))


def test_exact_task_seals_current_final_checkpoint(completed_run: Path) -> None:
    before = _checkpoint_state(completed_run)
    snapshot = load_current_evidence_snapshot(completed_run)

    sealed = _seal(completed_run)

    current = load_run_artifact_authority(completed_run)
    assert current is not None
    _assert_checkpoint_unchanged(completed_run, before)
    assert current["checkpoint_sequence"] == 1
    assert sealed.task_id == TASK_ID
    assert sealed.exposure_concept == EXPOSURE_CONCEPT
    assert sealed.outcome_concept == OUTCOME_CONCEPT
    assert sealed.run_primary_exposure == OPERATIONAL_EXPOSURE
    assert sealed.run_target_outcome == OUTCOME_CONCEPT
    assert len(sealed.run_input_capsule_sha256) == 64
    assert len(sealed.run_scientific_identity_sha256) == 64
    assert sealed.evidence_generation == snapshot.generation
    assert sealed.evidence_payload_sha256 == snapshot.payload_sha256
    assert "figure2_task_authority" not in current
    sidecars = _task_authority_sidecars(completed_run)
    assert len(sidecars) == 1
    assert (
        Figure2RunTaskAuthority.model_validate_json(
            sidecars[0].read_bytes(), strict=True
        )
        == sealed
    )
    AnalysisManifest.model_validate_json(json.dumps(current), strict=True)


def test_exact_task_seals_final_checkpoint_with_externalized_attempt_history(
    tmp_path: Path,
) -> None:
    run_dir = _build_run(
        tmp_path,
        run_name="run_external_history",
        externalize_attempt_history=True,
    )
    before = _checkpoint_state(run_dir)
    selected = load_run_artifact_authority(run_dir)
    assert selected is not None
    assert selected["step_attempt_history"] == [
        {"attempt": 1, "status": "ok", "step_id": "00_fixture_step"}
    ]

    sealed = _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)
    assert sealed.checkpoint_sequence == before[1]
    assert len(_task_authority_sidecars(run_dir)) == 1


def test_same_authority_is_idempotent_without_checkpoint_advance(
    completed_run: Path,
) -> None:
    first = _seal(completed_run)
    after_first = (completed_run / "manifest.json").read_bytes()

    second = _seal(completed_run)

    assert second == first
    assert (completed_run / "manifest.json").read_bytes() == after_first
    assert len(_task_authority_sidecars(completed_run)) == 1
    current = load_run_artifact_authority(completed_run)
    assert current is not None
    assert current["checkpoint_sequence"] == 1


def test_cross_task_coordinate_rejected_without_checkpoint_mutation(
    completed_run: Path,
) -> None:
    sealed = _seal(completed_run)
    before = _checkpoint_state(completed_run)
    sidecar_before = _task_authority_sidecars(completed_run)[0].read_bytes()

    with pytest.raises(ValueError, match="exposure concept does not match"):
        _seal(completed_run, exposure_concept="vasopressor")

    _assert_checkpoint_unchanged(completed_run, before)
    sidecars = _task_authority_sidecars(completed_run)
    assert len(sidecars) == 1
    assert sidecars[0].read_bytes() == sidecar_before
    assert (
        Figure2RunTaskAuthority.model_validate_json(sidecar_before, strict=True)
        == sealed
    )


def test_wrong_operational_exposure_fails_before_checkpoint_write(
    completed_run: Path,
) -> None:
    before = _checkpoint_state(completed_run)

    with pytest.raises(
        ValueError,
        match="frozen evaluator authority",
    ):
        _seal(completed_run, operational_exposure="vasopressor_equivalent_dose")

    _assert_checkpoint_unchanged(completed_run, before)


def test_wrong_benchmark_target_outcome_fails_before_checkpoint_write(
    completed_run: Path,
) -> None:
    before = _checkpoint_state(completed_run)

    with pytest.raises(ValueError, match="outcome concept does not match"):
        _seal(completed_run, outcome_concept="hospital_discharge")

    _assert_checkpoint_unchanged(completed_run, before)


@pytest.mark.parametrize(
    ("updates", "coordinate"),
    [
        ({"exposure_concept": None}, "exposure"),
        ({"outcome_concept": "   "}, "outcome"),
    ],
)
def test_missing_explicit_coordinates_fail_before_checkpoint_write(
    completed_run: Path,
    updates: dict[str, object],
    coordinate: str,
) -> None:
    before = (completed_run / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match=rf"benchmark {coordinate} concept"):
        _seal(completed_run, **updates)

    assert (completed_run / "manifest.json").read_bytes() == before
    current = load_run_artifact_authority(completed_run)
    assert current is not None
    assert current["checkpoint_sequence"] == 1
    assert "figure2_task_authority" not in current


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"task_id": f"{TASK_ID}_near_miss"}, "outside the frozen Figure 2 suite"),
        ({"research_question": "A different scientific question."}, "question"),
    ],
)
def test_task_or_question_mismatch_fails_before_checkpoint_write(
    completed_run: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    before = (completed_run / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match=message):
        _seal(completed_run, **updates)

    assert (completed_run / "manifest.json").read_bytes() == before
    current = load_run_artifact_authority(completed_run)
    assert current is not None
    assert current["checkpoint_sequence"] == 1
    assert "figure2_task_authority" not in current


@pytest.mark.parametrize(
    ("fixture_updates", "run_name"),
    [
        (
            {"capsule_question": "A different immutable run question."},
            "wrong_capsule_question",
        ),
        (
            {"capsule_outcome": "hospital_discharge"},
            "wrong_capsule_outcome",
        ),
    ],
)
def test_wrong_immutable_run_coordinates_fail_before_checkpoint_write(
    tmp_path: Path,
    fixture_updates: dict[str, object],
    run_name: str,
) -> None:
    run_dir = _build_run(tmp_path, run_name=run_name, **fixture_updates)
    before = _checkpoint_state(run_dir)

    with pytest.raises(PermissionError, match="owner-frozen Canonical9 input"):
        _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)


@pytest.mark.parametrize(
    ("readiness_update", "message"),
    [
        ({"execution_complete": False}, "execution_complete"),
        ({"manuscript_ready": False}, "manuscript_ready"),
    ],
)
def test_incomplete_readiness_fails_before_checkpoint_write(
    tmp_path: Path,
    readiness_update: dict[str, object],
    message: str,
) -> None:
    readiness = _ready_gates()
    readiness.update(readiness_update)
    run_dir = _build_run(
        tmp_path,
        run_name=f"incomplete_{message}",
        manifest_readiness=readiness,
    )
    before = _checkpoint_state(run_dir)

    with pytest.raises(PermissionError, match=message):
        _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)


def test_run_status_readiness_mismatch_fails_before_checkpoint_write(
    tmp_path: Path,
) -> None:
    status_readiness = _ready_gates()
    status_readiness["publication_figure_bundle_ready"] = False
    run_dir = _build_run(
        tmp_path,
        run_name="run_status_mismatch",
        run_status_readiness=status_readiness,
    )
    before = _checkpoint_state(run_dir)

    with pytest.raises(OSError, match="run_status disagrees"):
        _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)


@pytest.mark.parametrize(
    "manifest_authority_updates",
    [
        {"submission_profile_version": "20260717"},
        {"concept_dict_sha": "0" * 64},
        {"sofa2_dict_sha": "0" * 64},
        {
            "concept_dict_fingerprint": {
                "concept_dict_sha": "0" * 64,
                "sofa2_dict_sha": ready_submission_manifest_fields()["sofa2_dict_sha"],
            }
        },
    ],
)
def test_wrong_submission_authority_fails_before_sidecar_publication(
    tmp_path: Path,
    manifest_authority_updates: dict[str, object],
) -> None:
    run_dir = _build_run(
        tmp_path,
        run_name="wrong_submission_authority",
        manifest_authority_updates=manifest_authority_updates,
    )
    before = _checkpoint_state(run_dir)

    with pytest.raises(PermissionError, match="submission profile|dictionary"):
        _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)
    assert _task_authority_sidecars(run_dir) == []


def test_missing_steps_gate_is_required_before_sidecar_publication(
    tmp_path: Path,
) -> None:
    readiness = _ready_gates()
    readiness.pop("missing_steps")
    run_dir = _build_run(
        tmp_path,
        run_name="missing_steps_gate",
        manifest_readiness=readiness,
    )
    before = _checkpoint_state(run_dir)

    with pytest.raises(ValueError, match="missing_steps"):
        _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)
    assert _task_authority_sidecars(run_dir) == []


def test_tampered_run_input_capsule_fails_before_checkpoint_write(
    completed_run: Path,
) -> None:
    snapshot = load_current_evidence_snapshot(completed_run)
    matches = [
        record
        for record in snapshot.records
        if record.get("evidence_id") == RUN_INPUT_CAPSULE_EVIDENCE_ID
    ]
    assert len(matches) == 1
    capsule_path = completed_run / str(matches[0]["relative_path"])
    capsule_path.write_bytes(capsule_path.read_bytes() + b"\n")
    before = _checkpoint_state(completed_run)

    with pytest.raises(OSError, match="failed verification|changed during read"):
        _seal(completed_run)

    _assert_checkpoint_unchanged(completed_run, before)


def test_missing_run_input_capsule_selector_fails_before_checkpoint_write(
    tmp_path: Path,
) -> None:
    run_dir = _build_run(
        tmp_path,
        run_name="missing_capsule",
        seal_capsule=False,
    )
    before = _checkpoint_state(run_dir)

    with pytest.raises(ValueError, match="lacks a unique run-input capsule"):
        _seal(run_dir)

    _assert_checkpoint_unchanged(run_dir, before)


def test_newer_partial_selector_blocks_posthoc_final_seal(
    completed_run: Path,
) -> None:
    final_before = (completed_run / "manifest.json").read_bytes()
    partial_payload = json.loads(final_before)
    partial_payload["readiness"] = {"execution_complete": False}
    assert (
        write_run_checkpoint(
            completed_run / "manifest_partial.json",
            partial_payload,
        )
        == 2
    )

    with pytest.raises(OSError, match="final manifest is not the current"):
        _seal(completed_run)

    assert (completed_run / "manifest.json").read_bytes() == final_before
    selected = load_run_artifact_authority(completed_run)
    assert selected is not None
    assert selected["checkpoint_sequence"] == 2
    assert selected["readiness"] == {"execution_complete": False}
    assert "figure2_task_authority" not in selected


def test_none_authority_field_is_absent_from_legacy_serialization(
    tmp_path: Path,
) -> None:
    manifest = AnalysisManifest(
        run_id="run_compat",
        research_question=RESEARCH_QUESTION,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        context_path="context.json",
        per_step_records=[],
        readiness={"execution_complete": True},
    )

    assert "figure2_task_authority" not in AnalysisManifest.model_fields
    assert "figure2_task_authority" not in manifest.model_dump(mode="json")
    serialized = json.loads(manifest.model_dump_json())
    assert "figure2_task_authority" not in serialized
    AnalysisManifest.model_validate_json(manifest.model_dump_json(), strict=True)
