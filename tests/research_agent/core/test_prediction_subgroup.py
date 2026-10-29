"""Synthetic analytical oracles and closed-contract subgroup boundaries."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.prediction_subgroup import (
    PREDICTION_SUBGROUP_ACTION, PREDICTION_SUBGROUP_OUTPUTS, PredictionSubgroupError,
    PredictionSubgroupResult, PredictionSubgroupSpec, prediction_subgroup_method,
    prediction_subgroup_spec_for_step, prediction_subgroup_step_matches,
)
from easyicu.research_agent.methods.prediction_subgroup import (
    evaluate_prediction_subgroups, prediction_subgroup_result_findings,
)


def _spec(**updates):
    payload = dict(
        analysis_unit="subject", subgroup_input_key="artifact:analysis_cohort",
        subgroup_unit_id_column="row_id", axes=({"column": "group", "levels": ("A", "B", "C")},),
        minimum_group_n=2, minimum_class_n=1,
    )
    return PredictionSubgroupSpec.model_validate({**payload, **updates})


def _frame():
    # At each of two probability values, the observed risks are exactly 1/4
    # and 3/4. A is perfectly calibrated; B's logit is twice the true logit.
    return pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(16)], "subject_id": range(16),
        "split": "validation", "outcome": [0, 0, 0, 1, 0, 1, 1, 1] * 2,
        "probability": [0.25] * 4 + [0.75] * 4 + [0.1] * 4 + [0.9] * 4,
        "group": ["A"] * 8 + ["B"] * 8,
    })


def test_analytical_auroc_brier_and_logistic_calibration_oracle():
    result = evaluate_prediction_subgroups(_frame(), _spec())
    overall, group_a, group_b, empty = result.rows
    assert overall.evaluation_n == 16 and overall.event_n == 8
    assert group_a.evaluation_n == group_b.evaluation_n == 8
    assert group_a.auroc == group_b.auroc == pytest.approx(0.75)
    assert group_a.brier_score == pytest.approx(0.1875)
    assert group_b.brier_score == pytest.approx(0.21)
    assert overall.brier_score == pytest.approx(0.19875)
    assert group_a.calibration_intercept == pytest.approx(0, abs=1e-10)
    assert group_a.calibration_slope == pytest.approx(1, abs=1e-10)
    assert group_b.calibration_intercept == pytest.approx(0, abs=1e-10)
    assert group_b.calibration_slope == pytest.approx(0.5, abs=1e-10)
    assert empty.group_values == {"group": "C"}
    assert empty.evaluation_n == 0 and empty.brier_score is None and empty.auroc is None
    assert empty.auroc_status == "not_estimable_empty"
    assert empty.reason_codes == ("empty_declared_group",)
    assert result.interpretation == "descriptive_subgroup_performance_no_fairness_judgment"
    assert result.paper_authorization_allowed is False
    assert PredictionSubgroupResult.model_validate_json(result.model_dump_json()) == result


def test_frozen_probability_input_is_not_mutated_and_development_is_unused():
    frame = _frame()
    development = frame.iloc[:2].assign(unit_id=["d0", "d1"], subject_id=[30, 31], split="development", probability=[9, -1])
    combined = pd.concat([development, frame], ignore_index=True)
    original = combined.copy(deep=True)
    result = evaluate_prediction_subgroups(combined, _spec())
    pd.testing.assert_frame_equal(combined, original)
    assert result.input_n == 18 and result.evaluation_input_n == 16
    assert result.rows[0].brier_score == pytest.approx(0.19875)


def test_closed_intersection_retains_every_empty_and_missing_cell():
    frame = _frame()
    frame["site"] = ["north"] * 16
    frame.loc[0, "group"] = None
    spec = _spec(axes=(
        {"column": "group", "levels": ("A", "B", "C", None)},
        {"column": "site", "levels": ("north", "south")},
    ))
    result = evaluate_prediction_subgroups(frame, spec)
    assert len(result.rows) == 1 + 4 + 2 + 8
    cells = [row for row in result.rows if row.scope == "intersection"]
    assert len(cells) == 8 and sum(row.input_n for row in cells) == 16
    assert sum(row.evaluation_n == 0 for row in cells) == 5
    missing = next(row for row in cells if row.group_values == {"group": None, "site": "north"})
    assert missing.evaluation_n == 1 and missing.non_event_n == 1
    assert missing.auroc is None and missing.brier_score == pytest.approx(0.25**2)


def test_unknown_categories_and_undeclared_missingness_are_refused():
    for value in ("unreviewed", None):
        frame = _frame()
        frame.loc[0, "group"] = value
        with pytest.raises(PredictionSubgroupError, match="undeclared_level"):
            evaluate_prediction_subgroups(frame, _spec())


def test_complete_case_preserves_original_missingness_denominators_per_group():
    frame = _frame()
    frame.loc[0, "outcome"] = np.nan
    frame.loc[1, "probability"] = np.nan
    frame.loc[2, ["outcome", "probability"]] = np.nan
    with pytest.raises(PredictionSubgroupError, match="outcome_missing"):
        evaluate_prediction_subgroups(frame, _spec())
    result = evaluate_prediction_subgroups(frame, _spec(missing_value_policy="complete_case"))
    overall, group_a, _, _ = result.rows
    assert (overall.input_n, overall.evaluation_n, overall.excluded_n) == (16, 13, 3)
    assert (group_a.input_n, group_a.evaluation_n, group_a.excluded_n) == (8, 5, 3)
    assert group_a.missing_outcome_n == group_a.missing_probability_n == 2


def test_all_missing_group_is_retained_with_explicit_reason():
    frame = _frame()
    frame.loc[frame["group"].eq("A"), "outcome"] = np.nan
    result = evaluate_prediction_subgroups(frame, _spec(missing_value_policy="complete_case"))
    group_a = result.rows[1]
    assert group_a.input_n == group_a.excluded_n == group_a.missing_outcome_n == 8
    assert group_a.evaluation_n == 0 and group_a.auroc is group_a.brier_score is None
    assert group_a.reason_codes == ("empty_after_declared_complete_case",)


@pytest.mark.parametrize("column,value,reason", [
    ("probability", -0.1, "probability_invalid"), ("probability", 1.1, "probability_invalid"),
    ("probability", np.inf, "probability_invalid"), ("outcome", 2, "outcome_invalid"),
    ("outcome", -1, "outcome_invalid"), ("outcome", "unknown", "outcome_invalid"),
    ("probability", "0.5", "probability_invalid"),
])
def test_invalid_observed_values_are_never_complete_case_dropped(column, value, reason):
    frame = _frame()
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(PredictionSubgroupError, match=reason):
        evaluate_prediction_subgroups(frame, _spec(missing_value_policy="complete_case"))


def test_single_class_and_constant_predictions_have_defined_partial_metrics():
    frame = _frame()
    frame.loc[frame["group"].eq("A"), "outcome"] = 0
    frame["probability"] = 0.25
    overall, group_a, group_b, _ = evaluate_prediction_subgroups(frame, _spec()).rows
    assert group_a.auroc is None and group_a.auroc_status == "not_estimable_single_class"
    assert group_a.brier_score == pytest.approx(0.0625)
    assert group_b.auroc == overall.auroc == 0.5
    assert group_b.calibration_status == "not_estimable_constant_probability"
    assert group_b.calibration_intercept is group_b.calibration_slope is None
    assert group_b.brier_score == pytest.approx(0.3125)


def test_small_cells_publish_points_and_explicit_count_flags():
    result = evaluate_prediction_subgroups(_frame(), _spec(minimum_group_n=100))
    group_a = result.rows[1]
    assert group_a.auroc == 0.75 and group_a.sample_status == "small_sample"
    assert "below_declared_sample_counts" in group_a.reason_codes
    assert result.uncertainty == "point_estimates_only"


@pytest.mark.parametrize("defect,reason", [
    ("unit", "duplicate_unit"), ("subject", "subject_unit_not_unique"),
    ("missing_id", "identity_missing"), ("leakage", "subject_split_leakage"),
    ("no_validation", "evaluation_split_missing"),
])
def test_identity_and_split_boundaries(defect, reason):
    frame = _frame()
    if defect == "unit":
        frame.loc[1, "unit_id"] = frame.loc[0, "unit_id"]
    elif defect == "subject":
        frame.loc[1, "subject_id"] = frame.loc[0, "subject_id"]
    elif defect == "missing_id":
        frame.loc[0, "unit_id"] = None
    elif defect == "leakage":
        frame.loc[1, ["subject_id", "split"]] = [frame.loc[0, "subject_id"], "development"]
    else:
        frame["split"] = "development"
    with pytest.raises(PredictionSubgroupError, match=reason):
        evaluate_prediction_subgroups(frame, _spec())


def test_encounters_retain_patient_dependence_disclosure():
    frame = _frame()
    frame["subject_id"] = np.arange(16) // 2
    row = evaluate_prediction_subgroups(frame, _spec(analysis_unit="encounter")).rows[0]
    assert row.evaluation_n == 16 and row.subject_n == row.repeated_subject_n == 8
    assert "repeated_subjects_point_estimates_only" in row.reason_codes


def test_result_validator_detects_tampering_and_removed_closed_groups():
    frame, spec = _frame(), _spec()
    result = evaluate_prediction_subgroups(frame, spec)
    assert prediction_subgroup_result_findings(frame=frame, spec=spec, candidate=result) == ()
    candidate = result.model_dump(mode="json")
    candidate["rows"][1]["brier_score"] = 0.8
    assert prediction_subgroup_result_findings(frame=frame, spec=spec, candidate=candidate)[0].reason_code.endswith("result_mismatch")
    candidate = result.model_dump(mode="json")
    candidate["rows"].pop()
    assert prediction_subgroup_result_findings(frame=frame, spec=spec, candidate=candidate)[0].reason_code.endswith("result_mismatch")
    candidate["rows"][1]["event_n"] = 0
    assert prediction_subgroup_result_findings(frame=frame, spec=spec, candidate=candidate)[0].reason_code.endswith("result_schema_invalid")


def test_nullable_binary_columns_and_endpoint_probabilities_are_explicit():
    frame = _frame()
    frame["outcome"] = frame["outcome"].astype("Int64")
    frame["probability"] = frame["probability"].astype("Float64")
    frame.loc[0, "outcome"] = pd.NA
    frame.loc[1, "probability"] = pd.NA
    result = evaluate_prediction_subgroups(frame, _spec(missing_value_policy="complete_case"))
    assert result.rows[0].excluded_n == 2
    frame = _frame()
    frame.loc[0, "probability"] = 0
    frame.loc[4, "probability"] = 1
    result = evaluate_prediction_subgroups(frame, _spec())
    assert result.rows[0].clipped_probability_n == result.rows[1].clipped_probability_n == 2
    assert result.rows[1].brier_score == pytest.approx((0.1875 * 8 - 0.25**2 + 1 - 0.75**2) / 8)


def test_single_event_only_group_has_brier_without_zero_auroc():
    frame = _frame()
    frame.loc[frame["group"].eq("A"), "outcome"] = 1
    group = evaluate_prediction_subgroups(frame, _spec()).rows[1]
    assert group.event_n == group.evaluation_n == 8 and group.non_event_n == 0
    assert group.auroc is None and group.calibration_status == "not_estimable_single_class"
    assert group.brier_score == pytest.approx(0.3125)


@pytest.mark.parametrize("updates", [
    {"prediction_kind": "score"}, {"probability_column": "outcome"},
    {"axes": [{"column": "outcome", "levels": [0, 1]}]},
    {"axes": [{"column": "group", "levels": ["A", "A"]}]},
    {"axes": [{"column": "group", "levels": [None, None]}]},
    {"axes": [{"column": "group", "levels": [float("nan")]}]},
    {"axes": [{"column": "group", "levels": [["nested"]]}]},
    {"subgroup_input_key": "table:prediction_scores"},
])
def test_spec_refuses_scores_implicit_groups_and_invalid_rosters(updates):
    with pytest.raises(ValidationError):
        _spec(**updates)


def test_sealed_method_contract_requires_exact_input_output_role_and_action():
    spec = _spec()
    values = dict(
        scientific_action_id=PREDICTION_SUBGROUP_ACTION,
        method=prediction_subgroup_method(spec), inputs=("table:prediction_scores", spec.subgroup_input_key),
        expected_outputs=PREDICTION_SUBGROUP_OUTPUTS, planned_analysis_role="secondary",
    )
    step = SimpleNamespace(**values)
    assert prediction_subgroup_spec_for_step(step) == spec
    assert prediction_subgroup_step_matches(step)
    for change in ({"planned_analysis_role": "primary"}, {"inputs": step.inputs[::-1]},
                   {"expected_outputs": ("table:made_up",)}, {"method": "subgroup OR"}):
        assert not prediction_subgroup_step_matches(SimpleNamespace(**{**values, **change}))
    with pytest.raises(ValueError, match="belongs only"):
        prediction_subgroup_spec_for_step(SimpleNamespace(**{**values, "scientific_action_id": "association.subgroup"}))
