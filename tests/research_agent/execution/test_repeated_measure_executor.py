"""The host fits a reviewed long panel and publishes aggregate products only."""

import json
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    REPEATED_OUTPUTS, RepeatedMeasureSpec, repeated_measure_method, repeated_measure_spec_for_step,
    repeated_measure_result_receipt_valid,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (
    prepare_repeated_panel, run_repeated_measures,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.execution.runners.typed_input_binding import TypedInputBindingError
from easyicu.research_agent.execution.final_validation import _primary_runner_core_estimate_present
from easyicu.research_agent.reporting.readiness import _deterministic_primary_estimate_bound


def specification(**changes):
    return RepeatedMeasureSpec.model_validate(dict(
        identity_column="unit", time_column="elapsed", outcome_column="marker",
        time_unit="days", time_origin="icu_admission", window_start=0, window_end=6,
        bin_width=1, aggregation="mean", model="mixed_slope", family="gaussian",
        time_basis="linear", dropout_limitation="death_may_cause_informative_dropout",
        dropout_sensitivity="survivors", death_column="death",
    ) | changes)


def synthetic_frame():
    rng = np.random.default_rng(19)
    units = np.repeat(np.arange(60), 6)
    times = np.tile(np.arange(6, dtype=float) + 0.2, 60)
    return pd.DataFrame({
        "unit": units, "elapsed": times,
        "marker": 4 + 0.7 * np.floor(times) + rng.normal(0, 2, 60)[units]
        + rng.normal(0, 0.3, 60)[units] * np.floor(times) + rng.normal(0, 0.7, len(units)),
        "death": (units % 7 == 0).astype(int),
    })


def test_runner_exports_only_aggregate_effects_diagnostics_and_time_profile(tmp_path):
    result = run_repeated_measures(frame=synthetic_frame(), spec=specification(), out_dir=tmp_path)
    assert result["claim_ceiling"] == "analysis_only"
    assert set(p.name for p in tmp_path.iterdir()) == {
        "repeated_measure_effects.csv", "repeated_measure_diagnostics.csv", "repeated_measure_profile.csv",
    }
    effects = pd.read_csv(tmp_path / "repeated_measure_effects.csv")
    assert set(effects.population) == {"all_observed", "survivors"}
    assert effects.query("population == 'all_observed' and term == 'time'").estimate.iloc[0] == pytest.approx(0.7, abs=0.2)
    diagnostics = pd.read_csv(tmp_path / "repeated_measure_diagnostics.csv")
    assert diagnostics.n_units.tolist() == [60, 51]
    assert diagnostics.limitation.str.contains("Death.*informative dropout").all()
    assert "unit" not in pd.read_csv(tmp_path / "repeated_measure_profile.csv").columns


def test_panel_time_uses_declared_origin_unit_window_and_last_closed_edge():
    frame = pd.DataFrame({"unit": [1] * 6, "elapsed": [0, 24, 48, 144, 145, -1], "marker": [2, 4, 6, 8, 100, 100], "death": [0] * 6})
    spec = specification().model_copy(update={"time_unit": "hours", "window_end": 144.0, "bin_width": 24.0})
    prepared = prepare_repeated_panel(frame, spec)
    assert prepared.time.tolist() == [0, 24, 48, 120]
    assert prepared.value.tolist() == [2, 4, 6, 8]


def test_no_aggregation_or_fill_after_death_is_invented():
    spec = specification()
    frame = pd.DataFrame({"unit": [1, 1, 1, 2], "elapsed": [0.2, 0.8, 2.1, 0.4], "marker": [2, 6, 10, 1], "death": [1, 1, 1, 0]})
    prepared = prepare_repeated_panel(frame, spec)
    assert prepared.loc[prepared.unit == 1, "time"].tolist() == [0, 2]
    assert prepared.loc[prepared.unit == 1, "value"].tolist() == [4, 10]


@pytest.mark.parametrize("field", ["dropout_limitation", "dropout_sensitivity"])
def test_missing_death_dropout_limitation_is_refused(field):
    payload = specification().model_dump(mode="json")
    payload.pop(field)
    with pytest.raises(ValueError, match=field):
        RepeatedMeasureSpec.model_validate(payload)


def test_spec_round_trips_through_the_closed_method_coordinate():
    spec = specification()
    step = AnalysisStep(step_id="growth", intent="Estimate repeated outcome change", method=repeated_measure_method(spec), scientific_action_id="association.mixed_effects")
    assert repeated_measure_spec_for_step(step) == spec
    step.method = "repeated_measures(" + json.dumps({**spec.model_dump(mode="json"), "time_origin": "discharge"}) + ")"
    with pytest.raises(ValueError):
        repeated_measure_spec_for_step(step)


def test_registry_executes_the_bound_long_panel_and_rejects_source_drift(tmp_path, monkeypatch):
    spec = specification()
    path = tmp_path / "panel.csv"
    frame = synthetic_frame()
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "resolved.json"
    manifest.write_text(json.dumps({"step_id": "growth", "inputs": {"artifact:analysis_cohort": {
        "relative_path": path.name, "sha256": digest, "declared_kind": "artifact", "product": "analysis_cohort",
        "evidence_kind": "table", "product_contract": {"row_count": len(frame), "columns": list(frame.columns)},
    }}}))
    step = AnalysisStep(step_id="growth", planned_analysis_role="primary", intent="Fit reviewed repeated growth",
                        inputs=[*spec.required_columns, "artifact:analysis_cohort"], expected_outputs=list(REPEATED_OUTPUTS),
                        method=repeated_measure_method(spec), scientific_action_id=spec.action_id)
    selected = select_standard_executor(step, plan=AnalysisPlan(research_question="Estimate repeated change", analysis_type="repeated_measures", steps=[step]))
    assert selected is not None and selected.analysis_kind == "repeated_measures"
    for key, value in {"EASYICU_RUN_DIR": tmp_path, "EASYICU_RESOLVED_INPUTS_JSON": manifest, "STEP_OUT_DIR": tmp_path / "out"}.items():
        monkeypatch.setenv(key, str(value))
    exec(compile(selected.code, "reviewed_repeated_model", "exec"), {})
    assert (tmp_path / "out/repeated_measure_effects.csv").is_file()
    path.write_text(path.read_text() + "\n")
    with pytest.raises(TypedInputBindingError, match="digest"):
        exec(compile(selected.code, "reviewed_repeated_model", "exec"), {})


def test_core_estimate_requires_the_converged_spec_bound_receipt(tmp_path):
    summary = run_repeated_measures(frame=synthetic_frame(), spec=specification(), out_dir=tmp_path)
    assert repeated_measure_result_receipt_valid(summary)
    assert _primary_runner_core_estimate_present("repeated_measures", summary)
    record = {"step_id": "growth", "planned_analysis_role": "primary", "deterministic_standard_analysis": "repeated_measures", "step_summary": summary}
    assert _deterministic_primary_estimate_bound([record])
    bad = deepcopy(summary)
    bad["repeated_measure_receipt"]["fits"][0]["statistics"]["converged"] = False
    assert not repeated_measure_result_receipt_valid(bad)
    drift = deepcopy(summary)
    drift["repeated_measure_receipt"]["specification"]["time_unit"] = "hours"
    assert not repeated_measure_result_receipt_valid(drift)


@pytest.mark.parametrize("changes", [
    {"working_correlation": "exchangeable"}, {"model": "gee"},
    {"family": "binomial"}, {"time_origin": "hospital_admission"},
    {"model": "gee", "family": "binomial", "working_correlation": "independence", "aggregation": "mean"},
    {"time_basis": "natural_cubic_spline"}, {"window_end": 0.5},
    {"spline_knots": (2,)}, {"covariates": ("marker",)},
    {"time_basis": "natural_cubic_spline", "spline_knots": (2, 2)},
    {"dropout_sensitivity": "survivors", "death_column": None},
])
def test_invalid_repeated_designs_are_refused(changes):
    with pytest.raises(ValueError):
        specification(**changes)


def test_spline_and_binary_gee_execute_the_declared_model(tmp_path):
    frame = synthetic_frame()
    spline = specification(model="mixed_intercept", time_basis="natural_cubic_spline", spline_knots=(2, 4))
    summary = run_repeated_measures(frame=frame, spec=spline, out_dir=tmp_path / "spline")
    assert repeated_measure_result_receipt_valid(summary)
    assert any(name.startswith("time_spline") for name in summary["repeated_measure_receipt"]["fits"][0]["statistics"]["var_names"])
    rng = np.random.default_rng(1)
    frame.marker = rng.binomial(1, 1 / (1 + np.exp(-(frame.marker - 6))))
    binary = specification(model="gee", family="binomial", working_correlation="exchangeable", aggregation="last")
    summary = run_repeated_measures(frame=frame, spec=binary, out_dir=tmp_path / "binary")
    result = summary["repeated_measure_receipt"]["fits"][0]["statistics"]
    assert result["family"] == "binomial" and result["cov_type"] == "robust"


def test_time_unit_conversion_changes_the_slope_and_its_ci_consistently(tmp_path):
    frame = synthetic_frame()
    days = run_repeated_measures(frame=frame, spec=specification(), out_dir=tmp_path / "days")
    frame.elapsed *= 24
    hours = run_repeated_measures(frame=frame, spec=specification(time_unit="hours", window_end=144, bin_width=24), out_dir=tmp_path / "hours")
    for dayfit, hourfit in zip(days["repeated_measure_receipt"]["fits"], hours["repeated_measure_receipt"]["fits"], strict=True):
        assert hourfit["statistics"]["coef"][1] * 24 == pytest.approx(dayfit["statistics"]["coef"][1], rel=1e-4)


def test_action_model_mismatch_and_unbound_long_panel_are_refused():
    spec = specification()
    step = AnalysisStep(step_id="growth", planned_analysis_role="primary", intent="Fit reviewed repeated growth",
                        inputs=[*spec.required_columns, "artifact:analysis_cohort"], expected_outputs=list(REPEATED_OUTPUTS),
                        method=repeated_measure_method(spec), scientific_action_id="association.gee")
    with pytest.raises(ValueError, match="action and model"):
        repeated_measure_spec_for_step(step)
    step.scientific_action_id = spec.action_id
    step.inputs[-1] = "artifact:unreviewed_panel"
    assert select_standard_executor(step, plan=AnalysisPlan(research_question="Change over time", steps=[step])) is None


def test_growth_component_requires_random_slopes():
    spec = specification()
    step = AnalysisStep(step_id="growth", intent="Describe repeated growth curves",
                        method=repeated_measure_method(spec), scientific_action_id="phenotyping.mixed_effects_growth")
    assert repeated_measure_spec_for_step(step).model == "mixed_slope"
    step.method = repeated_measure_method(specification(model="mixed_intercept"))
    with pytest.raises(ValueError, match="action and model"):
        repeated_measure_spec_for_step(step)


def test_a_conflicting_capability_cannot_claim_the_repeated_owner():
    spec = specification()
    step = AnalysisStep(step_id="growth", planned_analysis_role="primary", intent="Fit reviewed repeated growth",
                        inputs=[*spec.required_columns, "artifact:analysis_cohort"], expected_outputs=list(REPEATED_OUTPUTS),
                        method=repeated_measure_method(spec), scientific_action_id=spec.action_id,
                        scientific_capability="association_adjusted_v1")
    plan = AnalysisPlan(research_question="Estimate change over time", analysis_type="repeated_measures", steps=[step])
    assert select_standard_executor(step, plan=plan) is None


def test_a_cross_sectional_table_cannot_answer_a_repeated_measure_question(tmp_path):
    frame = synthetic_frame().iloc[np.arange(60) * 6 + np.arange(60) % 6]
    with pytest.raises(ValueError, match="repeated"):
        run_repeated_measures(frame=frame, spec=specification(model="mixed_intercept"), out_dir=tmp_path)
