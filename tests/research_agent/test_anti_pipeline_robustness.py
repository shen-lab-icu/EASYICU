"""Regression coverage for the auxiliary, non-science-selecting robustness path."""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest


def _effect_record(*, n: int = 120) -> dict:
    return {
        "step_id": "03_primary_model",
        "status": "ok",
        "step_summary_evidence_id": "stat_primary",
        "step_summary": {
            "primary_predictor": "exposure",
            "primary_or": 1.4,
            "primary_ci_low": 1.2,
            "primary_ci_high": 1.7,
            "n_total": n,
        },
    }


def _locked_specs_payload() -> dict:
    return {
        "schema_version": "easyicu.robustness_specs/1",
        "locked_at": "2026-07-11T00:00:00+00:00",
        "spec_sha256": "test",
        "specs": [
            {
                "spec_id": "alt_relaxed_cohort",
                "axis": "cohort",
                "description": "Relax the locked length-of-stay threshold.",
                "cohort_override": {
                    "name": "relaxed",
                    "inclusion": [],
                    "exclusion": [],
                },
                "missing_override": None,
                "outcome_override": None,
            },
            {
                "spec_id": "alt_complete_case",
                "axis": "missing",
                "description": "Use complete-case analysis.",
                "cohort_override": None,
                "missing_override": {"strategy": "complete_case"},
                "outcome_override": None,
            },
            {
                "spec_id": "alt_observed_outcome",
                "axis": "outcome",
                "description": "Use the prespecified observed outcome.",
                "cohort_override": None,
                "missing_override": None,
                "outcome_override": {"concept_id": "death"},
            },
        ],
    }


def _sensitivity_step(*, figure: bool = False):
    from easyicu.research_agent.schema import AnalysisStep

    return AnalysisStep(
        step_id=(
            "07_cohort_definition_sensitivity_comparison_figure"
            if figure
            else "07_cohort_definition_sensitivity_comparison"
        ),
        intent="Render the sensitivity figure."
        if figure
        else "Execute locked variants.",
        method="cohort_definition_sensitivity",
        expected_outputs=(
            ["figure:robustness_plot"]
            if figure
            else ["table:sensitivity_specification_matrix"]
        ),
    )


def _write_membership_inputs(run_dir):
    universe_path = run_dir / "universe.parquet"
    cohort_path = run_dir / "cohort_analysis.parquet"
    pd.DataFrame({"stay_id": range(5)}).to_parquet(universe_path)
    pd.DataFrame({"stay_id": range(3)}).to_parquet(cohort_path)
    return universe_path, cohort_path


def test_primary_effect_never_parses_english_or_conjunction_from_prose() -> None:
    from easyicu.research_agent.plan_utils import _primary_effect_from_summary
    from easyicu.research_agent.scalar_utils import _first_numeric_effect_from_text

    summary = {
        "stage_boundary": (
            "Stage 1 means a creatinine increase of at least 0.3 mg/dL or "
            "1.5-1.9 times baseline."
        )
    }

    assert _first_numeric_effect_from_text(summary) is None
    assert _primary_effect_from_summary(summary) is None
    assert _first_numeric_effect_from_text({"result": "Adjusted OR=1.42"}) == 1.42


def test_structured_primary_effect_with_ci_remains_available() -> None:
    from easyicu.research_agent.pipeline_primary_effect import (
        _extract_primary_effect_payload_from_summary,
        _primary_effect_payload_is_complete,
    )

    payload = _extract_primary_effect_payload_from_summary(
        _effect_record()["step_summary"],
        path=None,
        preferred_predictor="exposure",
    )

    assert payload["primary_or"] == 1.4
    assert payload["primary_ci_low"] == 1.2
    assert payload["primary_ci_high"] == 1.7
    assert payload["effect_measure"] == "OR"
    assert payload["sample_size"] == 120
    assert _primary_effect_payload_is_complete(payload) is True


@pytest.mark.parametrize(
    "update",
    [
        {"primary_or": float("nan")},
        {"primary_ci_low": 1.8, "primary_ci_high": 1.7},
        {"effect_measure": ""},
        {"sample_size": 0},
    ],
)
def test_primary_panel_payload_requires_complete_typed_contract(update: dict) -> None:
    from easyicu.research_agent.pipeline_primary_effect import (
        _primary_effect_payload_is_complete,
    )

    payload = {
        "primary_or": 1.4,
        "primary_ci_low": 1.2,
        "primary_ci_high": 1.7,
        "effect_measure": "OR",
        "sample_size": 120,
    }
    payload.update(update)

    assert _primary_effect_payload_is_complete(payload) is False


def test_incomplete_primary_summary_does_not_create_converged_panel_row() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
    )

    record = _effect_record(n=0)
    panel = build_robustness_panel_from_records(specs=[], per_step_records=[record])

    assert len(panel.rows) == 1
    assert panel.rows[0].spec_id == "primary"
    assert panel.rows[0].converged is False
    assert panel.rows[0].point_estimate is None


def test_step_owned_robustness_rows_win_and_adapter_only_fills_missing_specs() -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanelRow,
        RobustnessSpec,
        build_robustness_panel_from_records,
    )

    specs = [
        RobustnessSpec("alt_owned", "missing", "Agent-owned row."),
        RobustnessSpec("alt_fill", "missing", "Adapter fill row."),
    ]
    record = _effect_record()
    record["step_summary"]["robustness_rows"] = [
        {
            "spec_id": "alt_owned",
            "axis": "missing",
            "n": 100,
            "point_estimate": 9.0,
            "ci_low": 8.0,
            "ci_high": 10.0,
            "se": 0.2,
            "converged": True,
        }
    ]
    adapter_rows = [
        RobustnessPanelRow(
            "alt_owned", "missing", 99, 1.1, 1.0, 1.2, 0.1, "adapter", True
        ),
        RobustnessPanelRow(
            "alt_fill", "missing", 98, 1.2, 1.1, 1.3, 0.1, "adapter", True
        ),
    ]

    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=[record],
        adapter_rows=adapter_rows,
    )
    rows = {row.spec_id: row for row in panel.rows}

    assert rows["alt_owned"].point_estimate == 9.0
    assert rows["alt_fill"].point_estimate == 1.2


@pytest.mark.parametrize("status", [None, "contract_failed", "execution_failed"])
def test_unsuccessful_step_rows_cannot_enter_final_robustness_panel(status) -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessSpec,
        build_robustness_panel_from_records,
    )

    record = _effect_record()
    if status is None:
        record.pop("status")
    else:
        record["status"] = status
    record["step_summary"]["robustness_rows"] = [
        {
            "spec_id": "alt_owned",
            "axis": "missing",
            "n": 100,
            "point_estimate": 9.0,
            "ci_low": 8.0,
            "ci_high": 10.0,
            "converged": True,
        }
    ]

    panel = build_robustness_panel_from_records(
        specs=[RobustnessSpec("alt_owned", "missing", "Locked variant.")],
        per_step_records=[record],
    )

    row = next(item for item in panel.rows if item.spec_id == "alt_owned")
    assert row.converged is False
    assert row.point_estimate is None


@pytest.mark.parametrize("status", [None, "contract_failed", "execution_failed"])
def test_unsuccessful_adapter_payload_cannot_trigger_deterministic_refit(
    status,
) -> None:
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import RobustnessSpec

    record = {
        "step_id": "07_sensitivity",
        "step_summary": {
            "estimator_adapter": {
                "data": [{"x": index, "y": index % 2} for index in range(20)],
                "exposure": "x",
                "outcome": "y",
                "estimator_kind": "logistic",
                "missing_strategy": "complete_case",
            }
        },
    }
    if status is not None:
        record["status"] = status

    rows, warnings = fit_robustness_rows_from_records(
        specs=[RobustnessSpec("alt", "missing", "Locked variant.")],
        per_step_records=[record],
        allow_implicit_cohort_refit=False,
    )

    assert rows == []
    assert any("generic deterministic robustness refitting is disabled" in warning for warning in warnings)


@pytest.mark.parametrize("missing_field", ["estimator_kind", "missing_strategy"])
def test_explicit_adapter_cannot_default_scientific_choices(missing_field) -> None:
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import RobustnessSpec

    payload = {
        "data": [{"x": index, "y": index % 2} for index in range(20)],
        "exposure": "x",
        "outcome": "y",
        "estimator_kind": "logistic",
        "missing_strategy": "complete_case",
    }
    payload.pop(missing_field)
    record = {
        "status": "ok",
        "step_id": "07_sensitivity",
        "step_summary": {"estimator_adapter": payload},
    }

    rows, warnings = fit_robustness_rows_from_records(
        specs=[RobustnessSpec("alt", "missing", "Locked variant.")],
        per_step_records=[record],
        allow_implicit_cohort_refit=True,
    )

    assert rows == []
    assert any(missing_field in warning for warning in warnings)


def test_explicit_adapter_cannot_create_missing_primary_estimate() -> None:
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import RobustnessSpec

    record = {
        "step_id": "07_sensitivity",
        "status": "ok",
        "step_summary": {
            "estimator_adapter": {
                "data": [{"x": index, "y": index % 2} for index in range(40)],
                "exposure": "x",
                "outcome": "y",
                "estimator_kind": "logistic",
                "missing_strategy": "complete_case",
            }
        },
    }

    rows, warnings = fit_robustness_rows_from_records(
        specs=[RobustnessSpec("alt", "missing", "Locked variant.")],
        per_step_records=[record],
        allow_implicit_cohort_refit=False,
    )

    assert rows == []
    assert any("generic deterministic robustness refitting is disabled" in warning for warning in warnings)


def test_adapter_primary_row_cannot_enter_panel_or_numeric_digest() -> None:
    from easyicu.research_agent.robustness_panel import (
        PRIMARY_SPEC_ID,
        RobustnessPanelRow,
        build_robustness_panel_from_records,
        numeric_digest_for_panel,
    )

    synthetic_primary = RobustnessPanelRow(
        spec_id=PRIMARY_SPEC_ID,
        axis="primary",
        n=40,
        point_estimate=2.0,
        ci_low=1.1,
        ci_high=3.2,
        se=0.2,
        evidence_id="adapter",
        converged=True,
    )
    panel = build_robustness_panel_from_records(
        specs=[],
        per_step_records=[],
        adapter_rows=[synthetic_primary],
    )

    primary = next(row for row in panel.rows if row.spec_id == PRIMARY_SPEC_ID)
    assert primary.point_estimate is None
    assert "primary_point_estimate" not in numeric_digest_for_panel(panel)


def test_logistic_adapter_does_not_invent_ridge_estimate_on_separation() -> None:
    from easyicu.research_agent.estimators import fit_estimator

    exposure = pd.DataFrame({"x": [0.0] * 30 + [1.0] * 30})
    outcome = pd.Series([0.0] * 30 + [1.0] * 30)

    result = fit_estimator(
        cohort=None,
        X=exposure,
        y=outcome,
        kind="logistic",
    )

    assert result.converged is False
    assert result.point_estimate is None
    assert "penalised" not in (result.notes or "").lower()


def test_pipeline_finalization_never_infers_relaxed_variant_from_locked_cohort(
    tmp_path, monkeypatch
) -> None:
    from easyicu.research_agent.cohort_schema import CohortDefinition
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import RobustnessSpec

    locked_path = tmp_path / "cohort_analysis.parquet"
    pd.DataFrame(
        {"exposure": [0.0, 1.0], "death": [0, 1], "los_icu": [1.0, 2.0]}
    ).to_parquet(locked_path)
    specs = [
        RobustnessSpec(
            "alt_relaxed",
            "cohort",
            "Relax eligibility beyond the locked cohort.",
            cohort_override=CohortDefinition(name="relaxed"),
        )
    ]

    def _unexpected_load(*args, **kwargs):
        raise AssertionError("pipeline finalization must not load/refit cohort data")

    monkeypatch.setattr(
        "easyicu.research_agent.estimators._load_direct_dataframe",
        _unexpected_load,
    )
    rows, warnings = fit_robustness_rows_from_records(
        specs=specs,
        per_step_records=[],
        cohort_path=locked_path,
        allow_implicit_cohort_refit=False,
    )

    assert rows == []
    assert any("generic deterministic robustness refitting is disabled" in warning for warning in warnings)

    from easyicu.research_agent.pipeline_execute import run_execute_phase
    from easyicu.research_agent.deterministic_robustness import (
        _run_robustness_preflight,
    )

    assert "allow_implicit_cohort_refit=False" in inspect.getsource(run_execute_phase)
    assert "allow_implicit_cohort_refit=False" in inspect.getsource(
        _run_robustness_preflight
    )


def test_logistic_adapter_rejects_nonbinary_outcome_before_statsmodels() -> None:
    from easyicu.research_agent.estimators import fit_estimator

    result = fit_estimator(
        cohort=None,
        X=pd.DataFrame({"exposure": [0.0, 1.0, 2.0, 3.0]}),
        y=pd.Series([1.2, 2.5, 3.0, 4.1]),
        kind="logistic",
    )

    assert result.converged is False
    assert result.point_estimate is None
    assert "binary 0/1" in result.notes
    assert "endog" not in result.notes


def test_locked_sensitivity_gate_blocks_missing_extra_ids_and_wrong_universe(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _cohort_definition_sensitivity_contract_findings,
    )

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "07_sensitivity" / "outputs"
    out_dir.mkdir(parents=True)
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(_locked_specs_payload()), encoding="utf-8"
    )
    universe_path, cohort_path = _write_membership_inputs(run_dir)
    summary = {
        "universe_final_n": 3,
        "robustness_rows": [
            {"spec_id": spec_id, "axis": "cohort"}
            for spec_id in (
                "invented_primary",
                "invented_alt",
                "invented_missing",
                "invented_outcome",
            )
        ],
    }

    findings = _cohort_definition_sensitivity_contract_findings(
        step=_sensitivity_step(),
        step_summary=summary,
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )
    errors = [finding for finding in findings if finding.severity == "error"]

    assert errors
    coverage = next(
        finding for finding in errors if finding.validator == "robustness_spec_lock"
    )
    assert set(coverage.detail["missing_spec_ids"]) == {
        "alt_relaxed_cohort",
        "alt_complete_case",
        "alt_observed_outcome",
    }
    assert set(coverage.detail["extra_spec_ids"]) == {
        "invented_primary",
        "invented_alt",
        "invented_missing",
        "invented_outcome",
    }
    assert coverage.detail["missing_spec_definitions"][0]["description"]
    assert any(
        finding.validator == "robustness_cohort_membership" for finding in errors
    )


def test_locked_sensitivity_gate_rejects_ids_without_membership_replay_fields(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _cohort_definition_sensitivity_contract_findings,
    )

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "07_sensitivity" / "outputs"
    out_dir.mkdir(parents=True)
    lock = _locked_specs_payload()
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    universe_path, cohort_path = _write_membership_inputs(run_dir)
    summary = {
        "universe_n": 5,
        "robustness_rows": [
            {"spec_id": "primary", "axis": "primary"},
            *[
                {"spec_id": spec["spec_id"], "axis": spec["axis"]}
                for spec in lock["specs"]
            ],
        ],
    }

    findings = _cohort_definition_sensitivity_contract_findings(
        step=_sensitivity_step(),
        step_summary=summary,
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )

    assert any(
        finding.validator == "robustness_cohort_membership"
        for finding in findings
    )


def test_locked_sensitivity_gate_rejects_reused_primary_membership_under_locked_id(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _cohort_definition_sensitivity_contract_findings,
    )

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "07_sensitivity" / "outputs"
    out_dir.mkdir(parents=True)
    lock = _locked_specs_payload()
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    universe_path, cohort_path = _write_membership_inputs(run_dir)
    rows = [
        {"spec_id": spec["spec_id"], "axis": spec["axis"]}
        for spec in lock["specs"]
    ]
    rows[0].update(
        {
            "universe_n": 5,
            "retained_n": 3,
            "entering_relative_to_primary_n": 0,
            "leaving_relative_to_primary_n": 0,
            "overlap_with_primary_n": 3,
        }
    )

    findings = _cohort_definition_sensitivity_contract_findings(
        step=_sensitivity_step(),
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )

    membership = next(
        finding
        for finding in findings
        if finding.validator == "robustness_cohort_membership"
    )
    assert any(
        issue["issue"] == "membership_value_mismatch"
        for issue in membership.detail["issues"]
    )


def test_locked_sensitivity_gate_accepts_declared_specification_csv(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _cohort_definition_sensitivity_contract_findings,
    )

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "07_sensitivity" / "outputs"
    out_dir.mkdir(parents=True)
    lock = _locked_specs_payload()
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    universe_path, cohort_path = _write_membership_inputs(run_dir)
    matrix_path = out_dir / "sensitivity_specification_matrix.csv"
    pd.DataFrame(
        [
            {
                "spec_id": spec["spec_id"],
                "axis": spec["axis"],
                **(
                    {
                        "universe_n": 5,
                        "retained_n": 5,
                        "entering_relative_to_primary_n": 2,
                        "leaving_relative_to_primary_n": 0,
                        "overlap_with_primary_n": 3,
                    }
                    if spec["axis"] == "cohort"
                    else {}
                ),
            }
            for spec in lock["specs"]
        ]
    ).to_csv(matrix_path, index=False)
    summary = {
        "output_files": {
            "table:sensitivity_specification_matrix": str(matrix_path),
        }
    }

    findings = _cohort_definition_sensitivity_contract_findings(
        step=_sensitivity_step(),
        step_summary=summary,
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_sensitivity_figure_step_is_not_subject_to_result_spec_gate(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _cohort_definition_sensitivity_contract_findings,
    )

    assert (
        _cohort_definition_sensitivity_contract_findings(
            step=_sensitivity_step(figure=True),
            step_summary={},
            out_dir=tmp_path,
            run_dir=tmp_path,
            universe_path=tmp_path / "missing.parquet",
        )
        == []
    )


def test_coder_context_receives_locked_spec_definitions_and_universe_contract(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _coder_context_with_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(_locked_specs_payload()), encoding="utf-8"
    )
    context = ResearchContext(
        research_question="Test the locked variants.",
        cohort=CohortDescriptor(
            cohort_name="test",
            database="mock",
            n_patients=5,
            n_stays=5,
        ),
        variables=[],
    )

    enriched = _coder_context_with_locked_robustness_specs(
        context=context,
        step=_sensitivity_step(),
        run_dir=run_dir,
    )

    assert enriched is not context
    assert "alt_relaxed_cohort" in (enriched.notes or "")
    assert "cohort_override" in (enriched.notes or "")
    assert "EASYICU_UNIVERSE_PARQUET" in (enriched.notes or "")


def test_locked_sensitivity_contract_is_wired_into_all_three_contract_passes() -> None:
    from easyicu.research_agent.pipeline_execute import run_execute_phase

    source = inspect.getsource(run_execute_phase)
    assert source.count("_cohort_definition_sensitivity_contract_findings(") == 3
