from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent import figure2_paper_rubric as paper_rubric
from easyicu.research_agent.evaluation_scorecard import (
    DimensionScore,
    FiveDimensionScorecard,
)
from easyicu.research_agent.figure2_paper_rubric import (
    FIGURE2_PAPER_RUBRIC_REF,
    FIGURE2_PAPER_SCORECARD_SCHEMA,
    Figure2ExactFiveScorecard,
    Figure2PaperScorecard,
    build_figure2_paper_scorecard,
    default_figure2_paper_rubric_path,
    load_figure2_paper_rubric,
    scorer_tree_rows,
    scorer_tree_sha256,
)
from easyicu.research_agent.figure2_rubric import (
    FIGURE2_DIMENSIONS,
    FIGURE2_TASK_IDS,
    default_figure2_rubric_path,
    figure2_suite_projection_sha256,
    load_figure2_rubric,
    rubric_manifest_sha256,
)
from easyicu.research_agent.figure2_safety_protocol import (
    FIGURE2_SAFETY_PROTOCOL_REF,
    safety_protocol_sha256,
)

_EXPECTED_V1_MANIFEST_FILE_SHA256 = (
    "0548176af23f47a724276a5cab077b514b56bada0bfe30ecf43300cc66c61f78"
)
_EXPECTED_V1_MANIFEST_SHA256 = (
    "b78907ef6692031cb70698cb41933b1d76407414431a646f53581786f4c08da9"
)
_EXPECTED_V1_DECODER_SOURCE_SHA256 = (
    "d8ebdf4ef22dad477ef14374c80ae358e723f6773536b65bc83af43abcae7edc"
)
_EXPECTED_SAFETY_PROTOCOL_SHA256 = (
    "76b4a20b39c76ce785d73fc9405954ed450bd2e6954b571621370699b3e9eb73"
)


def _read_v2_payload() -> dict[str, object]:
    return json.loads(default_figure2_paper_rubric_path().read_text(encoding="utf-8"))


def _live_v2_payload() -> dict[str, object]:
    """Return the committed shape with volatile implementation hashes refreshed."""

    payload = _read_v2_payload()
    payload["suite_projection_sha256"] = figure2_suite_projection_sha256()
    payload["scorer_tree_sha256"] = scorer_tree_sha256()
    payload["safety_protocol_sha256"] = safety_protocol_sha256()
    return payload


def _write_payload(
    tmp_path: Path, payload: object, *, name: str = "rubric.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load_live_manifest(tmp_path: Path):
    return load_figure2_paper_rubric(_write_payload(tmp_path, _live_v2_payload()))


def _dimension(name: str) -> DimensionScore:
    return DimensionScore(name=name, subscore=1.0, level="Full")


def _extended_scorecard() -> FiveDimensionScorecard:
    return FiveDimensionScorecard(
        task_id=FIGURE2_TASK_IDS[0],
        run_id="run-paper-rubric-test",
        plan=_dimension("plan"),
        code=_dimension("code"),
        result_validity=DimensionScore(name="result_validity"),
        evidence_binding=_dimension("evidence_binding"),
        audit_conclusion_safety=_dimension("audit_conclusion_safety"),
        reporting_completeness=_dimension("reporting_completeness"),
        fairness_subgroup=_dimension("fairness_subgroup"),
        tristate="gate_reportable",
    )


def _patch_default_manifest(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        paper_rubric,
        "default_figure2_paper_rubric_path",
        lambda: path,
    )


def test_v1_manifest_and_decoder_are_byte_frozen() -> None:
    """The additive v2 authority must not rewrite the historical v1 decoder."""

    manifest = load_figure2_rubric()
    decoder_path = Path(paper_rubric.__file__).with_name("figure2_rubric.py")

    assert (
        hashlib.sha256(default_figure2_rubric_path().read_bytes()).hexdigest()
        == _EXPECTED_V1_MANIFEST_FILE_SHA256
    )
    assert rubric_manifest_sha256(manifest) == _EXPECTED_V1_MANIFEST_SHA256
    assert (
        hashlib.sha256(decoder_path.read_bytes()).hexdigest()
        == _EXPECTED_V1_DECODER_SOURCE_SHA256
    )


def test_committed_v2_manifest_binds_the_current_full_scorer_tree() -> None:
    """This is the sole expected hash failure while the manifest has a placeholder."""

    payload = _read_v2_payload()

    assert payload["suite_projection_sha256"] == figure2_suite_projection_sha256()
    assert payload["safety_protocol_sha256"] == safety_protocol_sha256()
    assert payload["scorer_tree_sha256"] == scorer_tree_sha256()
    assert load_figure2_paper_rubric().scorer_tree_sha256 == scorer_tree_sha256()


def test_v2_manifest_is_exactly_nine_tasks_by_five_dimensions(tmp_path: Path) -> None:
    manifest = _load_live_manifest(tmp_path)

    assert manifest.rubric_ref == FIGURE2_PAPER_RUBRIC_REF
    assert tuple(manifest.dimensions) == FIGURE2_DIMENSIONS
    assert tuple(task.task_id for task in manifest.tasks) == FIGURE2_TASK_IDS
    assert len(manifest.tasks) * len(manifest.dimensions) == 45
    assert manifest.aggregation_policy == "none"
    assert manifest.na_policy == "preserve"
    assert manifest.audience == "evaluator_only"
    assert manifest.agent_visibility == "forbidden"
    assert all(
        task.dimension_applicability.model_dump()
        == {
            "plan": "required",
            "code": "required",
            "result_validity": "conditional",
            "evidence_binding": "required",
            "audit_conclusion_safety": "required",
            "result_validity_condition_code": (
                "GOLD_FREE_VALUE_SIGNAL_OR_LOCKED_REFERENCE"
            ),
        }
        for task in manifest.tasks
    )


def test_paper_scorecard_is_exact_five_and_drops_extended_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_payload(tmp_path, _live_v2_payload())
    _patch_default_manifest(monkeypatch, manifest_path)

    envelope = build_figure2_paper_scorecard(_extended_scorecard())
    payload = json.loads(envelope.scorecard_canonical_json)
    parsed = envelope.parsed_scorecard()

    assert envelope.schema_version == FIGURE2_PAPER_SCORECARD_SCHEMA
    assert tuple(item.name for item in parsed.dimensions()) == FIGURE2_DIMENSIONS
    assert set(payload) == {
        "task_id",
        "run_id",
        *FIGURE2_DIMENSIONS,
        "tristate",
    }
    assert "reporting_completeness" not in payload
    assert "fairness_subgroup" not in payload
    assert "reporting_completeness" not in Figure2ExactFiveScorecard.model_fields
    assert "fairness_subgroup" not in Figure2ExactFiveScorecard.model_fields


@pytest.mark.parametrize(
    "dimension",
    [
        DimensionScore(name="plan", subscore=1.01, level="Full"),
        DimensionScore(name="plan", subscore=-0.01, level="Fail"),
        DimensionScore(name="plan", subscore=float("nan"), level="Fail"),
        DimensionScore(name="plan", subscore=None, level="Full"),
        DimensionScore(name="plan", subscore=1.0, level=None),
    ],
)
def test_exact_five_scorecard_rejects_invalid_dimension_value_shapes(
    dimension: DimensionScore,
) -> None:
    with pytest.raises(ValidationError):
        Figure2ExactFiveScorecard(
            task_id=FIGURE2_TASK_IDS[0],
            run_id="run-invalid-dimension",
            plan=dimension,
            code=_dimension("code"),
            result_validity=DimensionScore(name="result_validity"),
            evidence_binding=_dimension("evidence_binding"),
            audit_conclusion_safety=_dimension("audit_conclusion_safety"),
            tristate="gate_reportable",
        )


@pytest.mark.parametrize(
    "blocking_dimension",
    [
        "plan",
        "code",
        "result_validity",
        "evidence_binding",
        "audit_conclusion_safety",
    ],
)
def test_exact_five_rejects_gate_reportable_with_any_blocking_dimension_fail(
    blocking_dimension: str,
) -> None:
    dimensions = {
        "plan": _dimension("plan"),
        "code": _dimension("code"),
        "result_validity": DimensionScore(name="result_validity"),
        "evidence_binding": _dimension("evidence_binding"),
        "audit_conclusion_safety": _dimension("audit_conclusion_safety"),
    }
    dimensions[blocking_dimension] = DimensionScore(
        name=blocking_dimension,
        subscore=0.0,
        level="Fail",
    )

    with pytest.raises(
        ValidationError,
        match="gate_reportable contradicts a blocking paper dimension",
    ):
        Figure2ExactFiveScorecard(
            task_id=FIGURE2_TASK_IDS[0],
            run_id="run-blocking-paper-dimension",
            **dimensions,
            tristate="gate_reportable",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("rubric_ref", "easyicu.figure2_paper_rubric/tampered", None),
        ("rubric_manifest_sha256", "0" * 64, "rubric authority mismatch"),
        ("suite_projection_sha256", "0" * 64, "suite authority mismatch"),
        ("scorer_tree_sha256", "0" * 64, "scorer-tree authority mismatch"),
    ],
)
def test_scorecard_rejects_direct_authority_field_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
    message: str | None,
) -> None:
    manifest_path = _write_payload(tmp_path, _live_v2_payload())
    _patch_default_manifest(monkeypatch, manifest_path)
    envelope = build_figure2_paper_scorecard(_extended_scorecard())
    payload = envelope.model_dump(mode="json")
    payload[field] = replacement

    if message is None:
        with pytest.raises(ValidationError):
            Figure2PaperScorecard.model_validate(payload)
    else:
        with pytest.raises(ValidationError, match=message):
            Figure2PaperScorecard.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("rubric_ref", "easyicu.figure2_paper_rubric/tampered", None),
        ("suite_projection_sha256", "0" * 64, "suite projection digest mismatch"),
        ("scorer_tree_sha256", "0" * 64, "scorer tree digest mismatch"),
        (
            "safety_protocol_ref",
            "easyicu.figure2_safety_adjudicator_protocol/tampered",
            None,
        ),
        ("safety_protocol_sha256", "0" * 64, "safety protocol digest mismatch"),
    ],
)
def test_manifest_rejects_direct_authority_field_tampering(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str | None,
) -> None:
    payload = _live_v2_payload()
    payload[field] = replacement

    if message is None:
        with pytest.raises(ValidationError):
            load_figure2_paper_rubric(_write_payload(tmp_path, payload))
    else:
        with pytest.raises((ValueError, ValidationError), match=message):
            load_figure2_paper_rubric(_write_payload(tmp_path, payload))


def test_safety_protocol_ref_and_digest_are_frozen(tmp_path: Path) -> None:
    manifest = _load_live_manifest(tmp_path)

    assert manifest.safety_protocol_ref == FIGURE2_SAFETY_PROTOCOL_REF
    assert manifest.safety_protocol_sha256 == _EXPECTED_SAFETY_PROTOCOL_SHA256
    assert safety_protocol_sha256() == _EXPECTED_SAFETY_PROTOCOL_SHA256


def test_scorer_tree_covers_every_research_agent_python_file_exactly_once() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_root = repository_root / "src/easyicu/research_agent"
    expected_paths = tuple(
        str(path.relative_to(repository_root))
        for path in sorted(source_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    rows = scorer_tree_rows()
    actual_paths = tuple(row["path"] for row in rows)

    assert actual_paths == expected_paths
    assert len(actual_paths) == len(set(actual_paths))
    assert all(path.startswith("src/easyicu/research_agent/") for path in actual_paths)
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    payload = _live_v2_payload()
    raw = json.dumps(payload, indent=2)
    duplicate = raw.replace(
        '  "agent_visibility": "forbidden",',
        '  "agent_visibility": "forbidden",\n' '  "agent_visibility": "forbidden",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_figure2_paper_rubric(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["tasks"][0].update({"answer": "forbidden"}),
    ],
)
def test_manifest_rejects_extra_json_fields(tmp_path: Path, mutate) -> None:
    payload = copy.deepcopy(_live_v2_payload())
    mutate(payload)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_figure2_paper_rubric(_write_payload(tmp_path, payload))


def test_exact_five_scorecard_rejects_extended_dimension_fields() -> None:
    exact = {
        "task_id": FIGURE2_TASK_IDS[0],
        "run_id": "run-paper-rubric-test",
        "plan": _dimension("plan").model_dump(mode="json"),
        "code": _dimension("code").model_dump(mode="json"),
        "result_validity": DimensionScore(name="result_validity").model_dump(
            mode="json"
        ),
        "evidence_binding": _dimension("evidence_binding").model_dump(mode="json"),
        "audit_conclusion_safety": _dimension("audit_conclusion_safety").model_dump(
            mode="json"
        ),
        "reporting_completeness": _dimension("reporting_completeness").model_dump(
            mode="json"
        ),
        "fairness_subgroup": _dimension("fairness_subgroup").model_dump(mode="json"),
        "tristate": "gate_reportable",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Figure2ExactFiveScorecard.model_validate(exact, strict=True)
