"""Frozen historical rubric-v1 authority contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.evaluation_scorecard import (
    DimensionScore,
    FiveDimensionScorecard,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (
    FIGURE2_DIMENSIONS,
    FIGURE2_TASK_IDS,
    SCORER_BUNDLE_FILES,
    Figure2RubricManifest,
    build_figure2_scorecard_envelope,
    default_figure2_rubric_path,
    figure2_suite_projection,
    figure2_suite_projection_sha256,
    load_figure2_rubric,
    rubric_manifest_sha256,
    scorer_bundle_rows,
    scorer_bundle_sha256,
)
from benchmarks.figure2_canonical9.evaluator.suite import (
    easyicu_evaluation_protocol_suite,
)

_EXPECTED_MANIFEST_SHA256 = (
    "b78907ef6692031cb70698cb41933b1d76407414431a646f53581786f4c08da9"
)
_EXPECTED_MANIFEST_FILE_SHA256 = (
    "0548176af23f47a724276a5cab077b514b56bada0bfe30ecf43300cc66c61f78"
)
_EXPECTED_SUITE_PROJECTION_SHA256 = (
    "11c39afac69c9a0b560c6aa92be19f05725f04c66b9d2c499cfdb353c40295ab"
)
_EXPECTED_SCORER_BUNDLE_SHA256 = (
    "d776568888d10d094c15a2c362205de38cf3ab12049f812757a2731fa874a47b"
)


def _manifest_payload() -> dict[str, object]:
    return json.loads(default_figure2_rubric_path().read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dimension(name: str, *, scored: bool = True) -> DimensionScore:
    return DimensionScore(
        name=name,
        subscore=1.0 if scored else None,
        level="Full" if scored else None,
    )


def _scorecard() -> FiveDimensionScorecard:
    return FiveDimensionScorecard(
        task_id=FIGURE2_TASK_IDS[0],
        run_id="run-test",
        plan=_dimension("plan"),
        code=_dimension("code"),
        result_validity=_dimension("result_validity", scored=False),
        evidence_binding=_dimension("evidence_binding"),
        audit_conclusion_safety=_dimension("audit_conclusion_safety"),
        tristate="gate_reportable",
    )


def test_manifest_loads_with_exact_canonical9_shape_and_policies() -> None:
    manifest = load_figure2_rubric()

    assert tuple(manifest.dimensions) == FIGURE2_DIMENSIONS
    assert tuple(task.task_id for task in manifest.tasks) == FIGURE2_TASK_IDS
    assert tuple(manifest.scorer_files) == SCORER_BUNDLE_FILES
    assert manifest.thresholds.model_dump() == {
        "full": 0.85,
        "partial": 0.55,
        "marginal": 0.25,
    }
    assert manifest.na_policy == "preserve"
    assert manifest.aggregation_policy == "none"
    assert manifest.audience == "evaluator_only"
    assert manifest.agent_visibility == "forbidden"
    assert all(
        task.dimension_applicability.result_validity == "conditional"
        and task.dimension_applicability.result_validity_condition_code
        == "LOCKED_REFERENCE_REQUIRED"
        for task in manifest.tasks
    )


def test_manifest_and_nested_authority_collections_are_immutable() -> None:
    manifest = load_figure2_rubric()

    assert isinstance(manifest.tasks, tuple)
    assert isinstance(manifest.scorer_files, tuple)
    assert isinstance(manifest.dimensions, tuple)
    assert isinstance(manifest.tasks[0].hazard_codes, tuple)
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        manifest.tasks += ()
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        manifest.tasks[0].hazard_codes += ("NEW_HAZARD",)


def test_external_manifest_digest_is_locked_but_not_self_embedded() -> None:
    manifest = load_figure2_rubric()
    payload = manifest.model_dump(mode="json")

    assert "rubric_manifest_sha256" not in payload
    assert rubric_manifest_sha256(manifest) == _EXPECTED_MANIFEST_SHA256
    # Lock presentation bytes as well as semantic canonical JSON.  This keeps a
    # committed v1 manifest byte-for-byte replayable while its authority digest
    # remains external rather than weakening the schema with a self-hash.
    assert (
        hashlib.sha256(default_figure2_rubric_path().read_bytes()).hexdigest()
        == _EXPECTED_MANIFEST_FILE_SHA256
    )


def test_suite_projection_is_answer_free_reproducible_and_exact() -> None:
    projection = figure2_suite_projection()
    runtime_ids = tuple(
        task.task_id for task in easyicu_evaluation_protocol_suite().tasks
    )

    assert runtime_ids == FIGURE2_TASK_IDS
    assert tuple(task["task_id"] for task in projection["tasks"]) == FIGURE2_TASK_IDS
    assert figure2_suite_projection_sha256() == _EXPECTED_SUITE_PROJECTION_SHA256
    for task in projection["tasks"]:
        assert "gold_answer" not in task
        assert "numeric_targets" not in task
        assert set(task) == {
            "task_id",
            "kind",
            "title",
            "objective",
            "expected_outputs",
            "semantic_guardrails",
            "evaluation_notes",
            "target_databases",
            "gold_answer_status",
            "has_gold_answer",
            "difficulty",
            "category",
        }


def test_historical_scorer_bundle_rows_and_digest_remain_frozen() -> None:
    rows = scorer_bundle_rows()

    assert tuple(row["path"] for row in rows) == SCORER_BUNDLE_FILES
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert scorer_bundle_sha256() == _EXPECTED_SCORER_BUNDLE_SHA256


def test_manifest_contains_codes_not_numeric_gold_or_answer_direction() -> None:
    payload = _manifest_payload()
    text = json.dumps(payload, sort_keys=True).lower()

    for forbidden_key in (
        "gold_answer",
        "numeric_targets",
        "expected_effect",
        "effect_direction",
        "answer_direction",
        "reference_value",
    ):
        assert forbidden_key not in text
    for task in payload["tasks"]:
        assert task["hazard_codes"]
        assert task["forbidden_claim_codes"]
        assert all(code == code.upper() for code in task["hazard_codes"])
        assert all(code == code.upper() for code in task["forbidden_claim_codes"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.update({"suite_projection_sha256": "0" * 64}),
        lambda payload: payload.update({"scorer_bundle_sha256": "0" * 64}),
        lambda payload: payload["dimensions"].reverse(),
        lambda payload: payload["tasks"].reverse(),
        lambda payload: payload["scorer_files"].reverse(),
        lambda payload: payload["thresholds"].update({"full": 0.84}),
    ],
)
def test_manifest_drift_fails_closed(tmp_path: Path, mutation) -> None:
    payload = _manifest_payload()
    mutation(payload)

    with pytest.raises((ValueError, ValidationError)):
        load_figure2_rubric(_write_manifest(tmp_path, payload))


def test_manifest_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    raw = default_figure2_rubric_path().read_text(encoding="utf-8")
    duplicate = raw.replace(
        '  "agent_visibility": "forbidden",',
        '  "agent_visibility": "forbidden",\n' '  "agent_visibility": "forbidden",',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_figure2_rubric(duplicate_path)

    nonfinite = raw.replace('"full": 0.85', '"full": NaN', 1)
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_figure2_rubric(nonfinite_path)


def test_scorecard_envelope_binds_immutable_canonical_payload_and_preserves_na() -> (
    None
):
    scorecard = _scorecard()
    envelope = build_figure2_scorecard_envelope(scorecard)

    assert envelope.aggregation_policy == "none"
    assert envelope.na_policy == "preserve"
    assert envelope.rubric_manifest_sha256 == _EXPECTED_MANIFEST_SHA256
    assert envelope.validated_scorecard().result_validity.subscore is None
    assert envelope.validated_scorecard().result_validity.level is None

    scorecard.result_validity.notes.append("caller-side mutation")
    assert envelope.validated_scorecard().result_validity.notes == []


def test_scorecard_envelope_rejects_payload_or_digest_tampering() -> None:
    envelope = build_figure2_scorecard_envelope(_scorecard())
    payload = envelope.model_dump(mode="json")

    payload["scorecard_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="digest mismatch"):
        type(envelope).model_validate(payload)

    payload = envelope.model_dump(mode="json")
    payload["scorecard_canonical_json"] += " "
    with pytest.raises(ValidationError, match="canonical JSON"):
        type(envelope).model_validate(payload)


def test_planner_coder_and_shared_prompts_do_not_import_or_expose_rubric() -> None:
    source_root = Path(__file__).resolve().parents[4] / "src/easyicu/research_agent"
    protected_paths = [
        source_root / "agents/core.py",
        source_root / "providers/prompts/__init__.py",
        *(source_root / "providers/prompts/v1").glob("*.txt"),
    ]

    for path in protected_paths:
        text = path.read_text(encoding="utf-8")
        assert "figure2_rubric" not in text
        assert all(task_id not in text for task_id in FIGURE2_TASK_IDS)


def test_model_schema_forbids_extra_fields() -> None:
    payload = _manifest_payload()
    payload["tasks"][0]["answer"] = "not allowed"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Figure2RubricManifest.model_validate_json(json.dumps(payload), strict=True)
