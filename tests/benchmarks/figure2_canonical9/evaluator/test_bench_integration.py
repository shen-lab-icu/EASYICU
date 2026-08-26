"""Black-box contracts for repository-local Figure 2 bench scoring.

These tests intentionally stub the posthoc evaluator.  Their subject is the
bench integration boundary: exact task routing, non-Figure-2 output stability,
non-fatal diagnostics, replay-on-reuse, and isolation from the research LLM
cost/receipt ledger.  The scorer and authority sealer have their own focused
test suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9.evaluator.scoring import (
    FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
    Figure2EvaluationAttempt,
)
from tools import run_research_agent_bench as bench
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.runtime_artifacts import write_run_checkpoint

_LEGACY_ARM_SCORE_KEYS = {
    "arm",
    "run_id",
    "workdir",
    "primary_or",
    "direction_match",
    "expected_direction",
    "icu_findings",
    "workflow_hits",
    "artifact_hits",
    "n_findings",
    "n_warnings",
    "n_errors",
    "n_historical_errors",
    "gate_status",
    "execution_complete",
    "step_scientific_requirements_complete",
    "required_step_count",
    "completed_step_count",
    "failed_step_ids",
    "missing_step_ids",
    "manuscript_ready",
    "publication_ready",
    "publication_artifacts_ready",
    "execution_paper_eligible",
    "paper_authorized",
    "writer_attempts",
    "superseded_error_count",
    "evidence_count",
    "evidence_kinds",
    "evidence_missing_in_manuscript",
    "five_dim_scorecard",
    "cost_summary",
}
_FIGURE2_VALIDITY_BINDINGS = {
    "e1_sepsis3_prevalence_mortality": ("sepsis3", "death"),
    "e2_lactate_mortality": ("lactate", "death"),
    "e3_kdigo_gradient": ("kdigo", "death"),
    "m1_hepatobiliary_missingness": ("bili", "death"),
    "m2_mortality_prediction": (None, "death"),
    "m3_sepsis_subphenotype": (None, "death"),
    "h1_ventilation_survival": ("vent_24h_any", "death"),
    "h2_vasopressor_causal": ("vasopressor", "death"),
    "h3_trajectory_clustering": (None, "death"),
}


def test_parquet_cohort_shape_reads_footer_without_dataframe_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {"stay_id": [11, 12, 13], "lactate": [1.2, 2.3, 3.4]}
    ).to_parquet(cohort_path, index=False)
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("full parquet load is forbidden"),
    )

    assert bench._cohort_shape_without_materialization(cohort_path) == (
        3,
        ["stay_id", "lactate"],
    )


def test_csv_cohort_shape_uses_header_and_single_column_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    cohort_path = tmp_path / "cohort.csv"
    cohort_path.write_text(
        "stay_id,lactate,death\n11,1.2,0\n12,2.3,1\n13,3.4,0\n",
        encoding="utf-8",
    )
    original_read_csv = pd.read_csv
    calls: list[dict[str, Any]] = []

    def recording_read_csv(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)

    assert bench._cohort_shape_without_materialization(cohort_path) == (
        3,
        ["stay_id", "lactate", "death"],
    )
    assert calls == [
        {"sep": ",", "nrows": 0},
        {"sep": ",", "usecols": [0], "chunksize": 100_000},
    ]


def _current_identity(
    *,
    cohort: Any = None,
    seed: int | None = None,
) -> dict[str, Any]:
    options = (
        bench._bind_benchmark_execution_input({}, cohort=cohort, data_seed=seed)
        if cohort is not None
        else {}
    )
    return bench._benchmark_execution_identity(
        options, provider="mock", model="mock"
    ).model_dump(mode="json")


def _write_manifest(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    cohort: Any = None,
    seed: int | None = None,
) -> None:
    payload = dict(payload)
    payload["execution_identity"] = _current_identity(cohort=cohort, seed=seed)
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _item(task_id: str) -> SimpleNamespace:
    exposure, outcome = _FIGURE2_VALIDITY_BINDINGS.get(
        task_id, ("locked_exposure", "locked_outcome")
    )
    return SimpleNamespace(
        key=task_id,
        name=f"Fixture {task_id}",
        research_question=f"Research question for {task_id}",
        primary_predictor=exposure,
        target_outcome=outcome,
        database="bench",
        inclusion_criteria=[],
        kind="descriptive_association",
        expected_or_direction=1,
        expected_finding_substrings=[],
        expected_step_substrings=[],
        expected_artifact_substrings=[],
    )


def test_legacy_blank_predictor_is_explicit_null_operational_exposure() -> None:
    item = SimpleNamespace(operational_exposure=None, primary_predictor="")

    assert bench._operational_exposure_for_item(item) is None


def test_benchmark_reuse_identity_binds_data_seed_and_input_values() -> None:
    first_cohort = [{"stay_id": 1, "value": 7}]
    second_cohort = [{"stay_id": 1, "value": 8}]
    first = _current_identity(cohort=first_cohort, seed=7)
    changed_seed = _current_identity(cohort=first_cohort, seed=8)
    changed_values = _current_identity(cohort=second_cohort, seed=7)

    assert first["environment_identity_sha256"] == changed_seed[
        "environment_identity_sha256"
    ]
    assert first["environment_identity_sha256"] == changed_values[
        "environment_identity_sha256"
    ]
    assert len(
        {
            first["identity_sha256"],
            changed_seed["identity_sha256"],
            changed_values["identity_sha256"],
        }
    ) == 3


def test_benchmark_reuse_identity_binds_external_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "cohort.csv"
    path.write_text("stay_id,value\n1,7\n", encoding="utf-8")
    before = _current_identity(cohort=path, seed=7)
    path.write_text("stay_id,value\n1,8\n", encoding="utf-8")
    after = _current_identity(cohort=path, seed=7)

    assert before["identity_sha256"] != after["identity_sha256"]


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_bench_figure2"
    run_dir.mkdir()
    _write_manifest(
        run_dir,
        {
            "run_id": run_dir.name,
            "findings": [],
            "readiness": {
                "numeric_error_count": 0,
                "evidence_error_count": 0,
                "analysis_error_count": 0,
            },
            "per_step_records": [],
            "evidence": [],
        },
    )
    return run_dir


def _write_run_status(run_dir: Path, *, execution_complete: object) -> None:
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": {"execution_complete": execution_complete}}),
        encoding="utf-8",
    )


def _assert_no_evaluator_only_keys(value: object) -> None:
    """Keep paper-evaluator authority out of Planner/Coder call surfaces."""

    forbidden_fragments = (
        "figure2",
        "validity_binding",
        "paper_rubric",
        "rubric_ref",
        "task_authority",
        "evaluator_receipt",
    )

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                normalized = str(key).strip().lower()
                assert not any(
                    fragment in normalized for fragment in forbidden_fragments
                )
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)


def _invalid_attempt(
    *, task_id: str, run_id: str, reason: str, detail: str
) -> Figure2EvaluationAttempt:
    return Figure2EvaluationAttempt(
        schema_version=FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
        status="invalid",
        task_id=task_id,
        run_id=run_id,
        invalid_reason_codes=(reason,),
        invalid_details=(detail,),
    )


def _stub_legacy_score(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    sentinel: dict[str, object] = {
        "sentinel": "legacy-five-dimension-scorecard",
        "nested": {"unchanged": True},
    }
    monkeypatch.setattr(bench, "_five_dim_scorecard", lambda **_kwargs: sentinel)
    return sentinel


def test_exact_figure2_task_adds_structured_attempt_without_changing_five_dim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    task_id = FIGURE2_TASK_IDS[0]
    item = _item(task_id)
    run_dir = _run_dir(tmp_path)
    five_dim_sentinel = _stub_legacy_score(monkeypatch)
    seal_calls: list[dict[str, Any]] = []
    score_calls: list[tuple[Path, str]] = []

    def fake_seal(path: Path, **kwargs: Any) -> None:
        assert path == run_dir
        seal_calls.append(kwargs)

    def fake_score(path: Path, *, task_id: str) -> Figure2EvaluationAttempt:
        score_calls.append((path, task_id))
        return _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="posthoc evaluator receipt is not present",
        )

    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", fake_seal
    )
    monkeypatch.setattr(
        figure2_scoring_inputs,
        "load_figure2_scoring_inputs",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", fake_score
    )

    result = bench._score_arm(run_dir=run_dir, item=item, label="aware")

    assert result["five_dim_scorecard"] is five_dim_sentinel
    assert set(result) == _LEGACY_ARM_SCORE_KEYS | {
        "execution_identity",
        "figure2_evaluation_attempt",
    }
    attempt = result["figure2_evaluation_attempt"]
    assert attempt == {
        "schema_version": FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
        "status": "invalid",
        "task_id": task_id,
        "run_id": run_dir.name,
        "envelope": None,
        "invalid_reason_codes": ["SAFETY_ADJUDICATION_MISSING"],
        "invalid_details": ["posthoc evaluator receipt is not present"],
    }
    assert seal_calls == [
        {
            "task_id": task_id,
            "research_question": item.research_question,
            "exposure_concept": item.primary_predictor,
            "outcome_concept": item.target_outcome,
            "operational_exposure": item.primary_predictor,
        }
    ]
    assert score_calls == [(run_dir, task_id)]


@pytest.mark.parametrize(
    "task_id",
    [
        "m2_mortality_prediction",
        "m3_sepsis_subphenotype",
        "h3_trajectory_clustering",
    ],
)
def test_multi_input_tasks_send_explicit_null_exposure_to_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    run_dir = _run_dir(tmp_path)
    observed: list[dict[str, Any]] = []

    def fake_seal(_path: Path, **kwargs: Any) -> None:
        observed.append(kwargs)

    def fake_score(path: Path, *, task_id: str) -> Figure2EvaluationAttempt:
        return _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="receipt absent",
        )

    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", fake_seal
    )
    monkeypatch.setattr(
        figure2_scoring_inputs,
        "load_figure2_scoring_inputs",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", fake_score
    )

    attempt = bench._figure2_evaluation_attempt(
        run_dir=run_dir,
        item=_item(task_id),
    )

    assert attempt["status"] == "invalid"
    assert observed[0]["exposure_concept"] is None
    assert observed[0]["outcome_concept"] == "death"
    assert observed[0]["operational_exposure"] is None


def test_evaluator_keeps_exposure_concept_distinct_from_operational_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    run_dir = _run_dir(tmp_path)
    item = _item("e1_sepsis3_prevalence_mortality")
    item.operational_exposure = "sep3_sofa2_max"
    observed: list[dict[str, Any]] = []

    monkeypatch.setattr(
        figure2_scoring_inputs,
        "seal_figure2_run_task_authority",
        lambda _path, **kwargs: observed.append(kwargs),
    )
    monkeypatch.setattr(
        figure2_scoring,
        "evaluate_figure2_run_from_receipt_path",
        lambda path, *, task_id: _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="receipt absent",
        ),
    )

    bench._figure2_evaluation_attempt(run_dir=run_dir, item=item)

    assert observed == [
        {
            "task_id": item.key,
            "research_question": item.research_question,
            "exposure_concept": "sepsis3",
            "outcome_concept": "death",
            "operational_exposure": "sep3_sofa2_max",
        }
    ]


def test_pipeline_finishes_before_figure2_seal_and_score_without_evaluator_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluator authority is posthoc: pipeline.run -> seal -> score."""

    import easyicu.research_agent as research_agent
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    task_id = FIGURE2_TASK_IDS[0]
    item = _item(task_id)
    run_dir = _run_dir(tmp_path)
    events: list[str] = []
    captured_init: dict[str, Any] = {}
    captured_run: dict[str, Any] = {}

    class CapturePipeline:
        def __init__(self, **kwargs: Any) -> None:
            captured_init.update(kwargs)

        def run(self, **kwargs: Any) -> SimpleNamespace:
            assert events == []
            captured_run.update(kwargs)
            events.append("pipeline_run")
            return SimpleNamespace(workdir=str(run_dir))

    def fake_seal(path: Path, **kwargs: Any) -> None:
        assert path == run_dir
        assert events == ["pipeline_run"]
        assert kwargs["operational_exposure"] == item.primary_predictor
        events.append("seal")

    def fake_score(path: Path, *, task_id: str) -> Figure2EvaluationAttempt:
        assert path == run_dir
        assert task_id == item.key
        assert events == ["pipeline_run", "seal"]
        events.append("score")
        return _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="posthoc only",
        )

    monkeypatch.setattr(research_agent, "ResearchAgentPipeline", CapturePipeline)
    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", fake_seal
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", fake_score
    )
    _stub_legacy_score(monkeypatch)

    result = bench._run_one_arm(
        item=item,
        cohort=SimpleNamespace(columns=["sepsis3", "death"]),
        workdir=tmp_path / "arm",
        disable_icu_context=False,
        label="aware",
        llm=object(),
    )

    assert events == ["pipeline_run", "seal", "score"]
    _assert_no_evaluator_only_keys(captured_init)
    _assert_no_evaluator_only_keys(captured_run)
    assert captured_run["primary_exposure"] == item.primary_predictor
    assert captured_run["target_outcome"] == item.target_outcome
    assert result["figure2_evaluation_attempt"]["status"] == "invalid"


@pytest.mark.parametrize(
    "task_id",
    [
        f"{FIGURE2_TASK_IDS[0]}_near_miss",
        "ordinary_external_evaluation_task",
    ],
)
def test_near_miss_and_non_figure2_tasks_keep_legacy_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    run_dir = _run_dir(tmp_path)
    five_dim_sentinel = _stub_legacy_score(monkeypatch)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("non-Figure-2 rows must not enter Figure 2 scoring")

    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", forbidden
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", forbidden
    )

    result = bench._score_arm(run_dir=run_dir, item=_item(task_id), label="aware")

    assert set(result) == _LEGACY_ARM_SCORE_KEYS
    assert "figure2_evaluation_attempt" not in result
    assert result["five_dim_scorecard"] is five_dim_sentinel


def test_bench_scores_active_and_superseded_errors_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _run_dir(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["findings"] = [
        {"validator": "old_gate", "severity": "error", "message": "superseded"}
    ]
    manifest["readiness"].update(
        {
            "numeric_error_count": 0,
            "evidence_error_count": 0,
            "analysis_error_count": 0,
            "superseded_error_count": 1,
        }
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _stub_legacy_score(monkeypatch)

    result = bench._score_arm(
        run_dir=run_dir,
        item=_item("ordinary_external_evaluation_task"),
        label="aware",
    )

    assert result["n_errors"] == 0
    assert result["n_historical_errors"] == 1
    assert result["superseded_error_count"] == 1


@pytest.mark.parametrize(
    ("failing_stage", "expected_reason"),
    [
        ("sealer", "SCORING_INPUT_AUTHORITY_INVALID"),
        ("scorer", "SCORER_ERROR"),
    ],
)
def test_figure2_sealer_and_scorer_exceptions_are_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
    expected_reason: str,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    task_id = FIGURE2_TASK_IDS[0]
    run_dir = _run_dir(tmp_path)
    five_dim_sentinel = _stub_legacy_score(monkeypatch)

    def fake_seal(*_args: Any, **_kwargs: Any) -> None:
        if failing_stage == "sealer":
            raise RuntimeError("sealer sentinel failure")

    def fake_score(*_args: Any, **_kwargs: Any) -> Figure2EvaluationAttempt:
        if failing_stage == "scorer":
            raise RuntimeError("scorer sentinel failure")
        raise AssertionError("scorer must not run after a sealer failure")

    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", fake_seal
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", fake_score
    )

    result = bench._score_arm(
        run_dir=run_dir,
        item=_item(task_id),
        label="aware",
    )

    assert result["five_dim_scorecard"] is five_dim_sentinel
    attempt = result["figure2_evaluation_attempt"]
    assert attempt["status"] == "invalid"
    assert attempt["invalid_reason_codes"] == [expected_reason]
    assert "sentinel failure" in attempt["invalid_details"][0]


def test_reusing_an_existing_run_reseals_and_rescores_instead_of_reusing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    task_id = FIGURE2_TASK_IDS[0]
    run_dir = _run_dir(tmp_path)
    _stub_legacy_score(monkeypatch)
    seal_count = 0
    score_count = 0

    def fake_seal(*_args: Any, **_kwargs: Any) -> None:
        nonlocal seal_count
        seal_count += 1

    def fake_score(path: Path, *, task_id: str) -> Figure2EvaluationAttempt:
        nonlocal score_count
        score_count += 1
        return _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail=f"fresh posthoc score {score_count}",
        )

    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", fake_seal
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", fake_score
    )

    first = bench._score_arm(run_dir=run_dir, item=_item(task_id), label="aware")
    second = bench._score_arm(run_dir=run_dir, item=_item(task_id), label="aware")

    assert seal_count == 2
    assert score_count == 2
    assert first["figure2_evaluation_attempt"]["invalid_details"] == [
        "fresh posthoc score 1"
    ]
    assert second["figure2_evaluation_attempt"]["invalid_details"] == [
        "fresh posthoc score 2"
    ]


@pytest.mark.parametrize(
    "corrupt_role",
    ["claim_ledger", "manuscript_ready"],
)
def test_figure2_reuse_requires_full_scoring_input_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_role: str,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(bench, "_run_reached_execution_complete", lambda _path: True)
    monkeypatch.setattr(
        figure2_scoring_inputs,
        "seal_figure2_run_task_authority",
        lambda *_args, **_kwargs: None,
    )

    def reject_corrupt_input(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(f"current {corrupt_role} evidence failed verification")

    monkeypatch.setattr(
        figure2_scoring_inputs,
        "load_figure2_scoring_inputs",
        reject_corrupt_input,
    )

    assert bench._figure2_run_is_reusable(run_dir, _item(FIGURE2_TASK_IDS[0])) is False


def test_authority_invalid_rescore_cannot_count_as_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    arm_dir = tmp_path / "aware"
    run_dir = arm_dir / "run_invalid_authority"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, {})
    monkeypatch.setattr(bench, "_figure2_run_is_reusable", lambda *_args: True)
    monkeypatch.setattr(
        bench,
        "_score_arm",
        lambda **_kwargs: {
            "arm": "aware",
            "figure2_evaluation_attempt": {
                "status": "invalid",
                "invalid_reason_codes": ["SCORING_INPUT_AUTHORITY_INVALID"],
            },
        },
    )

    assert (
        bench._reuse_arm_if_complete(
            arm_dir=arm_dir,
            item=_item(task_id),
            label="aware",
            expected_execution_identity_sha256=_current_identity()["identity_sha256"],
        )
        is None
    )


def test_posthoc_figure2_scoring_does_not_mutate_research_cost_or_provider_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    task_id = FIGURE2_TASK_IDS[0]
    run_dir = _run_dir(tmp_path)
    _stub_legacy_score(monkeypatch)
    tracked_payloads = {
        run_dir / "cost_summary.json": b'{"n_calls":7,"total_tokens":12345}\n',
        run_dir / "cost_records.json": b'[{"category":"research"}]\n',
        run_dir
        / ".runtime"
        / "provider_call_budgets"
        / "01_primary.json": b'{"schema_version":"sentinel-receipt","used":7}\n',
    }
    for path, payload in tracked_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(
        figure2_scoring_inputs,
        "seal_figure2_run_task_authority",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        figure2_scoring,
        "evaluate_figure2_run_from_receipt_path",
        lambda path, *, task_id: _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="posthoc only",
        ),
    )

    before = {path: path.read_bytes() for path in tracked_payloads}
    result = bench._score_arm(
        run_dir=run_dir,
        item=_item(task_id),
        label="aware",
    )
    after = {path: path.read_bytes() for path in tracked_payloads}

    assert after == before
    assert result["cost_summary"] == {"n_calls": 7, "total_tokens": 12345}
    assert result["figure2_evaluation_attempt"]["status"] == "invalid"


def test_ehrflow_reuse_rescores_completed_figure2_run_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring as figure2_scoring
    from benchmarks.figure2_canonical9.evaluator import (
        scoring_inputs as figure2_scoring_inputs,
    )

    task_id = FIGURE2_TASK_IDS[0]
    out_root = tmp_path / "results"
    run_dir = out_root / task_id / "aware" / "run_existing"
    run_dir.mkdir(parents=True)
    cohort_path = tmp_path / "cohort.csv"
    cohort_path.write_text("lactate,event\n1.2,0\n2.4,1\n", encoding="utf-8")
    _write_manifest(
        run_dir,
        {
            "run_id": run_dir.name,
            "findings": [],
            "readiness": {
                "execution_complete": True,
                "step_scientific_requirements_complete": True,
                "required_step_count": 1,
                "completed_step_count": 1,
                "failed_steps": [],
                "missing_steps": [],
                "numeric_error_count": 0,
                "evidence_error_count": 0,
                "analysis_error_count": 0,
            },
            "per_step_records": [],
            "evidence": [],
        },
        cohort=cohort_path,
        seed=7,
    )
    _write_run_status(run_dir, execution_complete=True)
    tracked_payloads = {
        run_dir / "cost_summary.json": b'{"n_calls":3,"total_tokens":456}\n',
        run_dir
        / ".runtime"
        / "provider_call_budgets"
        / "01_primary.json": b'{"schema_version":"sentinel","used":3}\n',
    }
    for path, payload in tracked_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    before = {path: path.read_bytes() for path in tracked_payloads}

    jsonl_path = tmp_path / "figure2.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "key": task_id,
                "question": f"Exact frozen task adapter fixture for {task_id}.",
                "cohort_path": str(cohort_path),
                "target_outcome": "event",
                "primary_predictor": "lactate",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seal_count = 0
    score_count = 0

    def fake_seal(*_args: Any, **_kwargs: Any) -> None:
        nonlocal seal_count
        seal_count += 1

    def fake_score(path: Path, *, task_id: str) -> Figure2EvaluationAttempt:
        nonlocal score_count
        score_count += 1
        return _invalid_attempt(
            task_id=task_id,
            run_id=path.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="fresh EHRFlow reuse score",
        )

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("completed Figure 2 reuse must not create a provider")

    monkeypatch.setattr(bench, "_ehrflow_item_done", lambda _path: True)
    monkeypatch.setattr(
        bench,
        "_run_reached_execution_complete",
        lambda candidate: candidate == run_dir,
    )
    monkeypatch.setattr(bench, "_make_llm", forbidden_provider)
    monkeypatch.setattr(bench, "_run_one_arm", forbidden_provider)
    _stub_legacy_score(monkeypatch)
    monkeypatch.setattr(
        figure2_scoring_inputs, "seal_figure2_run_task_authority", fake_seal
    )
    monkeypatch.setattr(
        figure2_scoring_inputs,
        "load_figure2_scoring_inputs",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        figure2_scoring, "evaluate_figure2_run_from_receipt_path", fake_score
    )

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl_path,
            out_root=out_root,
            seed=7,
            arms=["aware"],
            reuse_existing=True,
            allow_mock_aware=True,
        )
        == 0
    )

    payload = json.loads(
        (out_root / "ehrflowbench_results.json").read_text(encoding="utf-8")
    )
    assert payload["pending"] == []
    assert payload["scores"][0]["aware"]["run_id"] == run_dir.name
    # Reuse validation seals once, then the posthoc scorer independently
    # reseals before reading the optional safety receipt.
    assert seal_count == 2
    assert score_count == 1
    assert {path: path.read_bytes() for path in tracked_payloads} == before


def test_aborted_manifest_is_not_reused_and_arm_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    out_root = tmp_path / "results"
    run_dir = out_root / task_id / "aware" / "run_aborted"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, {})
    _write_run_status(run_dir, execution_complete=False)
    provider_calls: list[tuple[str, str]] = []
    arm_calls: list[str] = []
    provider_sentinel = object()

    def fake_provider(
        *,
        provider: str,
        model: str,
        request_timeout: float,
        reasoning_effort_profile: str,
        **_kwargs: Any,
    ) -> object:
        assert request_timeout == 180.0
        assert reasoning_effort_profile == "provider_default"
        provider_calls.append((provider, model))
        return provider_sentinel

    def fake_run_one_arm(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["llm"] is provider_sentinel
        arm_calls.append(kwargs["label"])
        return {"arm": kwargs["label"], "status": "ok", "run_id": "run_new"}

    monkeypatch.setattr(bench, "_make_llm", fake_provider)
    monkeypatch.setattr(
        bench,
        "_run_reached_execution_complete",
        lambda _candidate: False,
    )
    monkeypatch.setattr(bench, "_run_one_arm", fake_run_one_arm)
    monkeypatch.setattr(
        bench,
        "_score_arm",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("aborted run must not be scored as reusable")
        ),
    )

    result = bench._run_one_item_from_cohort(
        item=_item(task_id),
        cohort=[{"locked_exposure": 1, "locked_outcome": 0}],
        out_root=out_root,
        arms=["aware"],
        provider="mock",
        model="mock",
        reuse_existing=True,
    )

    assert provider_calls == [("mock", "mock")]
    assert arm_calls == ["aware"]
    assert result["aware"]["run_id"] == "run_new"


def test_latest_aborted_run_falls_back_to_latest_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    out_root = tmp_path / "results"
    arm_dir = out_root / task_id / "aware"
    complete = arm_dir / "run_20260718T010000_complete"
    aborted = arm_dir / "run_20260718T020000_aborted"
    cohort = [{"locked_exposure": 1, "locked_outcome": 0}]
    for run_dir, is_complete in ((complete, True), (aborted, False)):
        run_dir.mkdir(parents=True)
        _write_manifest(run_dir, {}, cohort=cohort)
        _write_run_status(run_dir, execution_complete=is_complete)
    scored: list[Path] = []

    def fake_score(*, run_dir: Path, item: Any, label: str) -> dict[str, Any]:
        assert item.key == task_id
        scored.append(run_dir)
        return {"arm": label, "status": "ok", "run_id": run_dir.name}

    monkeypatch.setattr(bench, "_score_arm", fake_score)
    monkeypatch.setattr(
        bench,
        "_figure2_run_is_reusable",
        lambda candidate, _item: candidate == complete,
    )
    monkeypatch.setattr(
        bench,
        "_make_llm",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a complete older run must avoid a provider call")
        ),
    )
    monkeypatch.setattr(
        bench,
        "_run_one_arm",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a complete older run must be reused")
        ),
    )

    result = bench._run_one_item_from_cohort(
        item=_item(task_id),
        cohort=cohort,
        out_root=out_root,
        arms=["aware"],
        reuse_existing=True,
    )

    assert scored == [complete]
    assert result["aware"]["run_id"] == complete.name


def test_partial_arm_reuse_creates_one_provider_and_runs_only_missing_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    out_root = tmp_path / "results"
    aware_run = out_root / task_id / "aware" / "run_complete"
    aware_run.mkdir(parents=True)
    cohort = [{"locked_exposure": 1, "locked_outcome": 0}]
    _write_manifest(aware_run, {}, cohort=cohort)
    _write_run_status(aware_run, execution_complete=True)
    provider_calls = 0
    arm_calls: list[str] = []
    provider_sentinel = object()

    def fake_provider(**_kwargs: Any) -> object:
        nonlocal provider_calls
        provider_calls += 1
        return provider_sentinel

    def fake_score(*, run_dir: Path, item: Any, label: str) -> dict[str, Any]:
        assert item.key == task_id
        return {"arm": label, "status": "ok", "run_id": run_dir.name}

    def fake_run_one_arm(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["llm"] is provider_sentinel
        arm_calls.append(kwargs["label"])
        return {"arm": kwargs["label"], "status": "ok", "run_id": "run_naive"}

    monkeypatch.setattr(bench, "_make_llm", fake_provider)
    monkeypatch.setattr(
        bench,
        "_figure2_run_is_reusable",
        lambda candidate, _item: candidate == aware_run,
    )
    monkeypatch.setattr(bench, "_score_arm", fake_score)
    monkeypatch.setattr(bench, "_run_one_arm", fake_run_one_arm)

    result = bench._run_one_item_from_cohort(
        item=_item(task_id),
        cohort=cohort,
        out_root=out_root,
        arms=["naive", "aware"],
        reuse_existing=True,
    )

    assert provider_calls == 1
    assert arm_calls == ["naive"]
    assert result["naive"]["run_id"] == "run_naive"
    assert result["aware"]["run_id"] == aware_run.name


def test_reuse_completion_requires_checkpoint_selected_final_and_exact_status_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only digest-bound current run_status evidence can authorize completion."""

    run_dir = tmp_path / "run_authoritative_complete"
    run_dir.mkdir()
    readiness = {
        "execution_complete": True,
        "manuscript_ready": True,
        "publication_ready": False,
    }
    status_path = run_dir / "run_status.json"
    status_path.write_text(
        json.dumps({"gates": readiness}, sort_keys=True),
        encoding="utf-8",
    )
    store = EvidenceStore(run_dir)
    store.register_file(
        kind="log",
        description="Authoritative completion status fixture.",
        source_path=status_path,
        evidence_id="run_status",
        producer="pipeline",
        generation_mode="system",
    )
    final_manifest = {
        "schema_version": "easyicu.research_manifest/1",
        "run_id": run_dir.name,
        "readiness": readiness,
        "findings": [],
        "per_step_records": [],
        "evidence": [record.model_dump(mode="json") for record in store.records()],
    }
    assert write_run_checkpoint(run_dir / "manifest.json", final_manifest) == 1
    from easyicu.research_agent.authority import runtime_artifacts

    selected = runtime_artifacts.load_run_artifact_authority(run_dir)
    assert selected is not None
    monkeypatch.setattr(
        runtime_artifacts,
        "load_run_artifact_authority",
        lambda _path: selected,
    )

    assert bench._run_reached_execution_complete(run_dir) is True

    # The mutable root copy is diagnostic only; changing it cannot change the
    # digest-bound EvidenceStore authority selected above.
    status_path.write_text(
        json.dumps(
            {"gates": {**readiness, "manuscript_ready": False}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert bench._run_reached_execution_complete(run_dir) is True

    # Tampering with the selected final checkpoint itself must fail closed.
    tampered = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    tampered["run_id"] = "different_checkpoint"
    (run_dir / "manifest.json").write_text(
        json.dumps(tampered, sort_keys=True), encoding="utf-8"
    )
    assert bench._run_reached_execution_complete(run_dir) is False


def test_explicit_paper_acceptance_fails_only_after_results_are_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    cohort_path = tmp_path / "cohort.csv"
    cohort_path.write_text("lactate,event\n1.2,0\n2.4,1\n", encoding="utf-8")
    jsonl_path = tmp_path / "figure2.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "key": task_id,
                "question": "One-task development run must not pass Canonical9.",
                "cohort_path": str(cohort_path),
                "target_outcome": "event",
                "primary_predictor": "lactate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        root = Path(kwargs["out_root"])
        run_dir = root / task_id / "aware" / "run_invalid"
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt = _invalid_attempt(
            task_id=task_id,
            run_id=run_dir.name,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="development fixture",
        )
        return {
            "item_key": task_id,
            "aware": {
                "arm": "aware",
                "run_id": run_dir.name,
                "workdir": str(run_dir),
                "execution_complete": True,
                "step_scientific_requirements_complete": True,
                "required_step_count": 1,
                "completed_step_count": 1,
                "failed_step_ids": [],
                "missing_step_ids": [],
                "figure2_evaluation_attempt": attempt.model_dump(mode="json"),
            },
        }

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "fixture\n")

    ordinary_root = tmp_path / "ordinary"
    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl_path,
            out_root=ordinary_root,
            seed=7,
            arms=["aware"],
            allow_mock_aware=True,
        )
        == 0
    )
    assert (ordinary_root / "ehrflowbench_results.json").is_file()
    ordinary_gate = json.loads(
        (ordinary_root / "figure2_paper_acceptance.json").read_text(encoding="utf-8")
    )
    assert ordinary_gate["status"] == "invalid"

    enforced_root = tmp_path / "enforced"
    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl_path,
            out_root=enforced_root,
            seed=7,
            arms=["aware"],
            allow_mock_aware=True,
            require_figure2_paper_acceptance=True,
        )
        == bench._FIGURE2_PAPER_ACCEPTANCE_EXIT_CODE
    )
    assert (enforced_root / "ehrflowbench_results.json").is_file()
    assert (enforced_root / "ehrflowbench_results.md").is_file()
    enforced_gate = json.loads(
        (enforced_root / "figure2_paper_acceptance.json").read_text(encoding="utf-8")
    )
    assert enforced_gate["status"] == "invalid"


def test_formal_safety_chain_issues_receipt_then_rescores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.figure2_canonical9.evaluator import safety_runner

    task_id = FIGURE2_TASK_IDS[0]
    run_dir = tmp_path / "run_formal"
    run_dir.mkdir()
    missing = _invalid_attempt(
        task_id=task_id,
        run_id=run_dir.name,
        reason="SAFETY_ADJUDICATION_MISSING",
        detail="fixture receipt missing",
    ).model_dump(mode="json")
    score = {
        "item_key": task_id,
        "aware": {
            "workdir": str(run_dir),
            "figure2_evaluation_attempt": missing,
        },
    }
    calls: list[tuple[Path, str]] = []

    class Transport:
        def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
            assert api_key == "fixture-secret"
            assert timeout_seconds == 45.0

    def issue(run_dir: Path, *, task_id: str, transport: object) -> object:
        assert isinstance(transport, Transport)
        calls.append((run_dir, task_id))
        return object()

    monkeypatch.setattr(
        safety_runner,
        "LocalOpenAICompatibleSafetyTransport",
        Transport,
    )
    monkeypatch.setattr(safety_runner, "ensure_figure2_safety_receipt", issue)
    monkeypatch.setattr(
        bench,
        "_figure2_evaluation_attempt",
        lambda **_kwargs: {
            "status": "valid",
            "task_id": task_id,
            "run_id": run_dir.name,
        },
    )

    bench._ensure_formal_figure2_safety_and_rescore(
        score=score,
        item=SimpleNamespace(key=task_id),
        provider_environment={"OPENAI_API_KEY": "fixture-secret"},
        request_timeout=45.0,
    )

    assert calls == [(run_dir, task_id)]
    assert score["aware"]["figure2_evaluation_attempt"]["status"] == "valid"
    assert "figure2_safety_adjudication_error" not in score["aware"]


def test_formal_safety_chain_missing_key_stays_invalid_with_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = FIGURE2_TASK_IDS[0]
    run_dir = tmp_path / "run_missing_key"
    run_dir.mkdir()
    missing = _invalid_attempt(
        task_id=task_id,
        run_id=run_dir.name,
        reason="SAFETY_ADJUDICATION_MISSING",
        detail="fixture receipt missing",
    ).model_dump(mode="json")
    score = {
        "item_key": task_id,
        "aware": {
            "workdir": str(run_dir),
            "figure2_evaluation_attempt": missing,
        },
    }
    monkeypatch.setattr(
        bench,
        "_figure2_evaluation_attempt",
        lambda **_kwargs: missing,
    )

    bench._ensure_formal_figure2_safety_and_rescore(
        score=score,
        item=SimpleNamespace(key=task_id),
        provider_environment={},
        request_timeout=45.0,
    )

    diagnostic = score["aware"]["figure2_safety_adjudication_error"]
    assert diagnostic["code"] == "SAFETY_TRANSPORT_CONFIG_INVALID"
    assert score["aware"]["figure2_evaluation_attempt"]["status"] == "invalid"
