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
        intent=(
            "Render the sensitivity figure." if figure else "Execute locked variants."
        ),
        method="cohort_definition_sensitivity",
        expected_outputs=(
            ["figure:robustness_plot"]
            if figure
            else ["table:sensitivity_specification_matrix"]
        ),
    )


def _prespecified_robustness_step(*, foreign_output: bool = False):
    from easyicu.research_agent.schema import AnalysisStep

    outputs = [
        "table:robustness_grid",
        "table:sensitivity_specification_matrix",
    ]
    if foreign_output:
        outputs.append("table:negative_control_outcomes")
    return AnalysisStep(
        step_id="08_prespecified_robustness",
        intent="Execute the planner-locked robustness specifications.",
        method="prespecified_robustness_analysis",
        expected_outputs=outputs,
    )


def _write_membership_inputs(run_dir):
    universe_path = run_dir / "universe.parquet"
    cohort_path = run_dir / "cohort_analysis.parquet"
    pd.DataFrame({"stay_id": range(5)}).to_parquet(universe_path)
    pd.DataFrame({"stay_id": range(3)}).to_parquet(cohort_path)
    return universe_path, cohort_path


def _write_valid_executed_results(out_dir, *, identifier_column="definition_id"):
    definitions = [
        ("alt_relaxed_cohort", "cohort", 5, 1.2, "complete_case"),
        ("alt_complete_case", "missing", 3, 1.3, "complete_case"),
        ("alt_observed_outcome", "outcome", 3, 1.4, "complete_case"),
    ]
    robustness_rows = []
    model_rows = []
    coefficient_rows = []
    for spec_id, axis, n, estimate, analysis_set in definitions:
        model_id = f"{spec_id}__model"
        row = {
            "spec_id": spec_id,
            "axis": axis,
            "status": "analyzed",
            "model_id": model_id,
            "outcome_concept_id": "death",
            "model_family": "logistic_regression",
            "effect_scale": "odds_ratio",
            "exposure_source": "marker",
            "comparison": "stage 3 versus stage 0",
            "coefficient_term": "stage_3",
            "analysis_set": analysis_set,
            "baseline_missing_policy": "drop_missing_baseline",
            "fit_status": "fitted",
            "interval_method": "logit_wald_95",
            "converged": True,
            "penalized": False,
            "reportable": True,
            "n": n,
            "point_estimate": estimate,
            "ci_low": estimate - 0.1,
            "ci_high": estimate + 0.1,
        }
        if axis == "missing":
            row["missing_strategy"] = "complete_case"
        if axis == "outcome":
            row["applied_outcome_override"] = {"concept_id": "death"}
        robustness_rows.append(row)
        model_rows.append(
            {
                identifier_column: spec_id,
                "model_id": model_id,
                "outcome": "death",
                "model_family": "logistic_regression",
                "effect_scale": "odds_ratio",
                "analysis_set": analysis_set,
                "baseline_missing_policy": "drop_missing_baseline",
                "n": n,
                "fit_status": "fitted",
                "converged": True,
                "penalized": False,
                "interval_method": "logit_wald_95",
                "stage3_effect": estimate,
                "stage3_ci_low": estimate - 0.1,
                "stage3_ci_high": estimate + 0.1,
            }
        )
        coefficient_rows.append(
            {
                "model_id": model_id,
                "outcome": "death",
                "model_family": "logistic_regression",
                "term": "stage_3",
                "term_role": "exposure",
                "source_variable": "marker",
                "effect_scale": "odds_ratio",
                "estimate": estimate,
                "ci_low": estimate - 0.1,
                "ci_high": estimate + 0.1,
            }
        )
    pd.DataFrame(model_rows).to_csv(out_dir / "model_fit_summary.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(
        out_dir / "adjusted_estimates.csv", index=False
    )
    return robustness_rows


@pytest.mark.parametrize("identifier_column", ["definition_id", "spec_id"])
def test_executed_sensitivity_accepts_model_spec_identifier_alias(
    tmp_path, identifier_column
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir, identifier_column=identifier_column)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert issues == []


def test_executed_sensitivity_rejects_conflicting_model_identifier_aliases(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    model_path = out_dir / "model_fit_summary.csv"
    models = pd.read_csv(model_path)
    models["spec_id"] = models["definition_id"]
    models.loc[0, "spec_id"] = "different_locked_spec"
    models.to_csv(model_path, index=False)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert any(
        issue["spec_id"] == rows[0]["spec_id"]
        and issue["issue"] == "model_contract_row_count"
        for issue in issues
        if "spec_id" in issue
    )


def test_executed_sensitivity_rejects_duplicate_model_rows(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir, identifier_column="spec_id")
    model_path = out_dir / "model_fit_summary.csv"
    models = pd.read_csv(model_path)
    models = pd.concat([models, models.iloc[[0]]], ignore_index=True)
    models.to_csv(model_path, index=False)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert any(
        issue["spec_id"] == rows[0]["spec_id"]
        and issue["issue"] == "model_contract_row_count"
        and issue["observed"] == 2
        for issue in issues
        if "spec_id" in issue
    )


def test_executed_sensitivity_rejects_ambiguous_model_result_tables(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir, identifier_column="spec_id")
    models = pd.read_csv(out_dir / "model_fit_summary.csv")
    models.to_csv(out_dir / "second_model_result_table.csv", index=False)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    unavailable = next(
        issue for issue in issues if issue["issue"] == "model_result_table_unavailable"
    )
    assert unavailable["detail"][0] == "structured_table_ambiguous"


def test_long_coefficient_table_with_model_metadata_is_not_a_second_model_table(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir, identifier_column="spec_id")
    coefficient_path = out_dir / "adjusted_estimates.csv"
    coefficients = pd.read_csv(coefficient_path)
    spec_by_model = {row["model_id"]: row["spec_id"] for row in rows}
    coefficients["spec_id"] = coefficients["model_id"].map(spec_by_model)
    # Add a second coefficient per model.  Model metadata may legitimately be
    # repeated on a long coefficient table, but that must not make it a second
    # model-result table.
    coefficients = pd.concat([coefficients, coefficients], ignore_index=True)
    coefficients.loc[len(rows) :, "term"] = "adjustment_term"
    coefficients.loc[len(rows) :, "term_role"] = "adjustment"
    coefficients.to_csv(coefficient_path, index=False)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert not any(
        issue["issue"] == "model_result_table_unavailable" for issue in issues
    )


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
    assert any(
        "generic deterministic robustness refitting is disabled" in warning
        for warning in warnings
    )


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
    assert any(
        "generic deterministic robustness refitting is disabled" in warning
        for warning in warnings
    )


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
    assert any(
        "generic deterministic robustness refitting is disabled" in warning
        for warning in warnings
    )

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
    assert all(
        finding.detail.get("step_id") == "07_cohort_definition_sensitivity_comparison"
        for finding in errors
    )
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
        finding.validator == "robustness_cohort_membership" for finding in findings
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
        {"spec_id": spec["spec_id"], "axis": spec["axis"]} for spec in lock["specs"]
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


def test_locked_sensitivity_declaration_only_does_not_prove_execution(tmp_path) -> None:
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

    assert any(
        finding.validator == "robustness_executed_result" for finding in findings
    )
    assert not any(
        finding.validator in {"robustness_spec_lock", "robustness_cohort_membership"}
        for finding in findings
    )


def test_locked_sensitivity_gate_accepts_typed_overlap_table_names(tmp_path) -> None:
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
        [{"spec_id": spec["spec_id"], "axis": spec["axis"]} for spec in lock["specs"]]
    ).to_csv(matrix_path, index=False)
    overlap_path = out_dir / "cohort_definition_overlap_attrition.csv"
    pd.DataFrame(
        [
            {
                "definition_id": "alt_relaxed_cohort",
                "axis": "cohort",
                "universe_n": 5,
                "retained_n": 5,
                "entered_n": 2,
                "left_primary_n": 0,
                "overlap_n": 3,
            }
        ]
    ).to_csv(overlap_path, index=False)

    findings = _cohort_definition_sensitivity_contract_findings(
        step=_sensitivity_step(),
        step_summary={
            "output_files": {
                "table:sensitivity_specification_matrix": str(matrix_path),
                "table:cohort_definition_overlap_attrition": str(overlap_path),
            },
            "robustness_rows": _write_valid_executed_results(out_dir),
        },
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_executed_sensitivity_rejects_wrong_outcome_model(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    lock = _locked_specs_payload()
    outcome_spec = lock["specs"][2]
    outcome_spec["outcome_override"] = {"concept_id": "los_icu"}
    rows[2]["applied_outcome_override"] = {"concept_id": "los_icu"}

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert any(issue["issue"] == "executed_outcome_mismatch" for issue in issues)


def test_executed_sensitivity_rejects_forged_summary_estimate(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    rows[0]["point_estimate"] = 999.0
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert any(
        issue["issue"]
        in {
            "model_result_value_mismatch",
            "coefficient_result_value_mismatch",
        }
        for issue in issues
    )


def test_reportable_robustness_must_retain_primary_estimator_contract(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    rows[0]["model_family"] = "descriptive_binomial_risk"
    model_path = out_dir / "model_fit_summary.csv"
    models = pd.read_csv(model_path)
    models.loc[0, "model_family"] = "descriptive_binomial_risk"
    models.to_csv(model_path, index=False)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
        primary_model_contract={
            "model_family": "logistic_regression",
            "effect_scale": "odds_ratio_per_unit",
            "exposure_source": "marker",
            "outcome": "death",
        },
    )

    assert any(
        issue.get("spec_id") == "alt_relaxed_cohort"
        and issue.get("issue") == "primary_estimator_family_mismatch"
        for issue in issues
    )


def test_nonindependent_locked_variant_is_an_honest_null_disclosure(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    blocked_spec_id = "alt_observed_outcome"
    blocked = next(row for row in rows if row["spec_id"] == blocked_spec_id)
    blocked.update(
        {
            "status": "not_independent",
            "fit_status": "not_fitted",
            "converged": False,
            "reportable": False,
            "n": None,
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "non_executable_reason": (
                "The supplied stay-level scalar target cannot yield an "
                "independent first-versus-any aggregation."
            ),
        }
    )
    for filename in ("model_fit_summary.csv", "adjusted_estimates.csv"):
        path = out_dir / filename
        frame = pd.read_csv(path)
        frame = frame[frame["model_id"] != blocked["model_id"]]
        frame.to_csv(path, index=False)
    lock = _locked_specs_payload()

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
        primary_model_contract={
            "model_family": "logistic_regression",
            "effect_scale": "odds_ratio",
            "exposure_source": "marker",
            "outcome": "death",
        },
    )

    assert issues == []


def test_executed_sensitivity_rejects_missing_indicator_on_complete_case_model(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    lock = _locked_specs_payload()
    lock["specs"][1]["missing_override"] = {"strategy": "missing_indicator"}
    rows[1]["missing_strategy"] = "missing_indicator"

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert any(issue["issue"] == "missing_indicator_model_not_used" for issue in issues)


def test_missing_indicator_accepts_structured_availability_term_role(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    lock = _locked_specs_payload()
    lock["specs"][1]["missing_override"] = {"strategy": "missing_indicator"}
    target = rows[1]
    target.update(
        {
            "missing_strategy": "missing_indicator",
            "analysis_set": "source_aware",
            "baseline_missing_policy": "explicit_missing_category",
        }
    )
    model_path = out_dir / "model_fit_summary.csv"
    models = pd.read_csv(model_path)
    model_mask = models["model_id"].eq(target["model_id"])
    models.loc[model_mask, "analysis_set"] = "source_aware"
    models.loc[model_mask, "baseline_missing_policy"] = "explicit_missing_category"
    models.to_csv(model_path, index=False)
    coefficient_path = out_dir / "adjusted_estimates.csv"
    coefficients = pd.read_csv(coefficient_path)
    availability_row = (
        coefficients[coefficients["model_id"].eq(target["model_id"])].iloc[0].copy()
    )
    availability_row["term"] = "source_not_observed"
    availability_row["term_role"] = "availability"
    coefficients = pd.concat(
        [coefficients, pd.DataFrame([availability_row])], ignore_index=True
    )
    coefficients.to_csv(coefficient_path, index=False)

    issues = _executed_robustness_result_issues(
        locked_by_id={spec["spec_id"]: spec for spec in lock["specs"]},
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )

    assert not any(
        issue.get("spec_id") == target["spec_id"]
        and issue["issue"]
        in {"missing_indicator_model_not_used", "missing_indicator_term_absent"}
        for issue in issues
    )


def test_penalized_point_only_sensitivity_must_be_nonreportable(tmp_path) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _executed_robustness_result_issues,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    rows = _write_valid_executed_results(out_dir)
    target = rows[0]
    target.update(
        {
            "status": "fitted",
            "ci_low": None,
            "ci_high": None,
            "converged": False,
            "penalized": True,
            "interval_method": "unavailable",
            "reportable": True,
        }
    )
    model_path = out_dir / "model_fit_summary.csv"
    models = pd.read_csv(model_path)
    mask = models["model_id"].eq(target["model_id"])
    models.loc[mask, ["stage3_ci_low", "stage3_ci_high"]] = pd.NA
    models.loc[mask, "converged"] = False
    models.loc[mask, "penalized"] = True
    models.loc[mask, "interval_method"] = "unavailable"
    models.to_csv(model_path, index=False)
    coefficient_path = out_dir / "adjusted_estimates.csv"
    coefficients = pd.read_csv(coefficient_path)
    coefficient_mask = coefficients["model_id"].eq(target["model_id"])
    coefficients.loc[coefficient_mask, ["ci_low", "ci_high"]] = pd.NA
    coefficients.to_csv(coefficient_path, index=False)
    lock = _locked_specs_payload()
    locked_by_id = {spec["spec_id"]: spec for spec in lock["specs"]}

    issues = _executed_robustness_result_issues(
        locked_by_id=locked_by_id,
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )
    assert any(
        issue["issue"] == "reportable_result_requires_finite_ci" for issue in issues
    )
    assert any(
        issue["issue"] == "reportable_result_requires_verified_convergence"
        for issue in issues
    )

    target["reportable"] = False
    issues = _executed_robustness_result_issues(
        locked_by_id=locked_by_id,
        step_summary={"robustness_rows": rows},
        out_dir=out_dir,
        context=None,
    )
    assert not any(
        issue["spec_id"] == target["spec_id"]
        and issue["issue"]
        in {
            "reportable_result_requires_finite_ci",
            "reportable_result_requires_verified_convergence",
            "point_only_result_must_be_penalized_nonreportable",
            "executed_result_status_invalid",
            "executed_model_not_fitted",
        }
        for issue in issues
        if "spec_id" in issue
    )


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
    from easyicu.research_agent.coder_authority_notes import HostCoderAuthority
    from easyicu.research_agent.pipeline_execute import (
        _coder_authority_with_locked_robustness_specs,
    )
    from easyicu.research_agent.robustness_execution_contract import (
        ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE,
        ROBUSTNESS_RESULT_REQUIRED_FIELDS,
    )
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(_locked_specs_payload()), encoding="utf-8"
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": "04_primary_model",
                        "status": "ok",
                        "step_summary": {
                            "final_design_terms": ["const", "marker", "age"],
                            "model_contracts": [
                                {
                                    "model_id": "primary_model",
                                    "outcome": "death",
                                    "model_family": "logistic_regression",
                                    "effect_scale": "odds_ratio_per_unit",
                                    "exposure_source": "marker",
                                    "exposure_role": "primary",
                                    "analysis_role": "primary",
                                    "fit_status": "fitted",
                                    "converged": True,
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
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
        primary_exposure="marker",
        target_outcome="death",
    )

    authority = _coder_authority_with_locked_robustness_specs(
        authority=HostCoderAuthority(),
        context=context,
        step=_sensitivity_step(),
        run_dir=run_dir,
    )
    rendered = authority.render()

    assert context.notes is None
    assert "alt_relaxed_cohort" in rendered
    assert "cohort_override" in rendered
    assert "EASYICU_UNIVERSE_PARQUET" in rendered
    assert ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE in rendered
    assert all(
        field in ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE
        for field in ROBUSTNESS_RESULT_REQUIRED_FIELDS
    )
    assert "n is the analytic fitted-model N" in rendered
    assert "applied_outcome_override" in rendered
    assert "missing_strategy" in rendered
    assert "universe_n" in rendered
    assert "variant_membership_n" in rendered
    assert "inflow_n" in rendered
    assert "outflow_n" in rendered
    assert "overlap_n" in rendered
    assert "including when the fitted model itself is not executable" in (rendered)
    assert "aggregation='count'" in rendered
    assert "must never be replaced by nonmissingness" in rendered
    assert "reconcile them before applying the membership predicate" in (rendered)
    assert "report invalid and discordant pair counts" in rendered
    assert "membership-changing disagreement" in rendered
    assert "mark that specification not_executable" in rendered
    assert "derive measurement availability from the designated" in rendered
    assert "do not infer it only from isna()" in rendered
    assert "documented computational encoding" in rendered
    assert "do not silently map them into the reference category" in (rendered)
    assert "Cohort membership N and fitted analytic n are distinct" in rendered
    assert "single pre-aggregated outcome scalar per analysis unit" in (rendered)
    assert "must be marked not_independent" in rendered
    assert "never refit the unchanged scalar and relabel it" in rendered
    assert "AUTHORITATIVE PRIMARY MODEL CONTRACT" in rendered
    assert '"model_family":"logistic_regression"' in rendered


def test_prespecified_robustness_alias_receives_locked_execution_contract(
    tmp_path,
) -> None:
    from easyicu.research_agent.coder_authority_notes import HostCoderAuthority
    from easyicu.research_agent.pipeline_execute import (
        _coder_authority_with_locked_robustness_specs,
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

    authority = _coder_authority_with_locked_robustness_specs(
        authority=HostCoderAuthority(),
        context=context,
        step=_prespecified_robustness_step(),
        run_dir=run_dir,
    )
    rendered = authority.render()

    assert context.notes is None
    assert "LOCKED ROBUSTNESS SPECIFICATIONS" in rendered
    assert "missing-indicator specification" in rendered


def test_prespecified_robustness_alias_is_gated_but_mixed_contract_is_not(
    tmp_path,
) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _cohort_definition_sensitivity_contract_findings,
    )

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "08_prespecified_robustness" / "outputs"
    out_dir.mkdir(parents=True)
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(_locked_specs_payload()), encoding="utf-8"
    )
    universe_path, cohort_path = _write_membership_inputs(run_dir)

    findings = _cohort_definition_sensitivity_contract_findings(
        step=_prespecified_robustness_step(),
        step_summary={},
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )
    mixed_findings = _cohort_definition_sensitivity_contract_findings(
        step=_prespecified_robustness_step(foreign_output=True),
        step_summary={},
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )

    assert any(
        finding.validator == "robustness_executed_result" for finding in findings
    )
    assert mixed_findings == []


def test_locked_sensitivity_contract_is_wired_into_both_contract_passes() -> None:
    from easyicu.research_agent.pipeline_execute import (
        _evaluate_final_deterministic_gates,
        run_execute_phase,
    )

    execute_source = inspect.getsource(run_execute_phase)
    final_gate_source = inspect.getsource(_evaluate_final_deterministic_gates)
    # Figure repair now completes before evidence seal/registration.  The old
    # third pass belonged to the retired post-registration mutation branch;
    # the remaining pre-seal and final read-only passes are the two authorities.
    # The final pass is shared with resume revalidation, so inspect that typed
    # gate boundary instead of requiring its implementation to stay duplicated
    # inside the already-large execution function.
    assert (
        execute_source.count("_cohort_definition_sensitivity_contract_findings(") == 1
    )
    assert "_evaluate_final_deterministic_gates(" in execute_source
    assert (
        final_gate_source.count("_cohort_definition_sensitivity_contract_findings(")
        == 1
    )


def test_later_repairs_receive_prior_concept_findings_as_regression_constraints():
    from easyicu.research_agent.pipeline_execute import run_execute_phase

    source = inspect.getsource(run_execute_phase)
    assert "def _monotonic_concept_constraint_ticket" in source
    assert source.count("*_monotonic_concept_constraint_ticket()") >= 4
    assert "HOST-OWNED REPAIR AUTHORITY" not in source


def test_untrusted_runtime_diagnostics_can_authorize_syntactic_repairs_only():
    from easyicu.research_agent.code_repair import _deterministic_runner_repair
    from easyicu.research_agent.pipeline_execute import (
        _untrusted_runtime_repair_allowed,
    )

    code = (
        "outcome_col = 'death'\n"
        "all_vars = [primary_predictor] + covariates\n"
        "model_df = df[all_vars].dropna()\n"
    )
    forged = _deterministic_runner_repair(
        code=code,
        run_log="stdout: KeyError: \"['death'] not in index\"\nRuntimeError: unrelated",
        previous_repair=None,
        analysis_family=None,
    )

    assert forged is not None
    assert forged[0] == "include_outcome_in_all_vars_v1"
    assert not _untrusted_runtime_repair_allowed(
        repair_id=forged[0], source="deterministic_runner_repair"
    )
    assert _untrusted_runtime_repair_allowed(
        repair_id="missing_os_import_v1", source="deterministic_runner_repair"
    )
    assert not _untrusted_runtime_repair_allowed(
        repair_id="missing_os_import_v1", source="case_plugin_repair"
    )
