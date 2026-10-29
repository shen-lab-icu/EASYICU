"""Executable preparation rules use synthetic values and frozen source facts."""

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_terms import (
    DataPreparationPlan,
    PreparationRefusal,
)
from easyicu.research_agent.methods.missing import (
    apply_outliers,
    aggregate_windows,
    database_preparation_status,
    standardize_columns,
)


def test_range_filter_and_winsorize_report_actual_counts():
    frame = pd.DataFrame({"x": [-1.0, 2, 8, np.nan]})
    for rule, expected in [
        ("range_filter", [np.nan, 2, np.nan, np.nan]),
        ("winsorize", [0, 2, 5, np.nan]),
    ]:
        plan = DataPreparationPlan(
            outliers=[{"variable": "x", "rule": rule, "lower": 0, "upper": 5}]
        )
        result, receipt = apply_outliers(frame, plan.outliers, raw_variables={"x"})
        np.testing.assert_allclose(result.x, expected, equal_nan=True)
        assert receipt[0]["removed"] == (2 if rule == "range_filter" else 0)
        assert receipt[0]["capped"] == (2 if rule == "winsorize" else 0)
    assert frame.x.iloc[0] == -1


def test_outliers_refuse_a_concept_without_raw_values():
    plan = DataPreparationPlan(
        outliers=[{"variable": "x", "rule": "winsorize", "lower": 0, "upper": 5}]
    )
    with pytest.raises(PreparationRefusal) as caught:
        apply_outliers(pd.DataFrame({"x": [2.0]}), plan.outliers, raw_variables=set())
    assert caught.value.reason_code == "preparation_raw_values_unavailable"


def test_source_preparation_cannot_borrow_another_databases_raw_authority(tmp_path):
    from easyicu.concept.provenance import ExportDataProvenance, capture_data_provenance
    from easyicu.research_agent.execution.model_matrix import prepare_analysis_frame
    from tests.support.data_provenance import synthetic_dictionary

    provenance = ExportDataProvenance(
        snapshots=(
            capture_data_provenance(
                synthetic_dictionary(), concepts=["lab_signal"], source_database="miiv"
            ),
        )
    )
    context = {
        "cohort": {
            "database": "eicu",
            "provenance": {"data_provenance": provenance.model_dump(mode="json")},
        },
        "variables": [{"name": "lab_signal", "source_concept": "lab_signal"}],
    }
    (tmp_path / "research_context.json").write_text(json.dumps(context))
    choice = DataPreparationPlan(
        outliers=[
            {"variable": "lab_signal", "rule": "winsorize", "lower": 0, "upper": 5}
        ]
    )
    frame = pd.DataFrame({"lab_signal": [-1.0, 2, 8]})
    for database in ("eicu", "unknown_source"):
        context["cohort"]["database"] = database
        (tmp_path / "research_context.json").write_text(json.dumps(context))
        with pytest.raises(
            PreparationRefusal, match="preparation_raw_values_unavailable"
        ):
            prepare_analysis_frame(frame, choice, run_dir=tmp_path)
    # A matching source, including its public display-name alias, is executable.
    for database in ("miiv", "MIMIC-IV"):
        context["cohort"]["database"] = database
        (tmp_path / "research_context.json").write_text(json.dumps(context))
        result, receipt = prepare_analysis_frame(frame, choice, run_dir=tmp_path)
        assert list(result.lab_signal) == [0, 2, 5]
        assert receipt["outliers"][0]["capped"] == 2


@pytest.mark.parametrize(
    "function,value", [("first", 3), ("last", 1), ("mean", 2), ("min", 1), ("max", 3)]
)
def test_window_aggregation_orders_time_and_has_explicit_boundaries(function, value):
    frame = pd.DataFrame(
        {"unit": [1, 1, 1, 1, 2], "hours": [2, 1, 0, 4, 5], "x": [1.0, 3, 100, 200, 9]}
    )
    plan = DataPreparationPlan(
        windows=[
            {
                "variable": "x",
                "group_column": "unit",
                "time_column": "hours",
                "start_hours": 1,
                "end_hours": 4,
                "function": function,
            }
        ]
    )
    result, receipt = aggregate_windows(frame, plan.windows, raw_variables={"x"})
    assert result.set_index("unit").loc[1, "x"] == value
    assert np.isnan(result.set_index("unit").loc[2, "x"])
    assert receipt[0]["boundary"] == "start_inclusive_end_exclusive"
    assert receipt[0]["measurements_used"] == 2


def test_scaling_uses_only_the_fit_partition():
    train = pd.DataFrame({"x": [0.0, 2, 4]})
    test = pd.DataFrame({"x": [1000.0]})
    plan = DataPreparationPlan(scaling_coding=[{"variable": "x", "standardize": True}])
    fitted, scored, receipt = standardize_columns(train, test, plan.scaling_coding)
    assert abs(fitted.x.mean()) < 1e-12
    assert scored.x.iloc[0] == pytest.approx((1000 - 2) / np.std([0, 2, 4]))
    assert receipt[0]["fit_n"] == 3


def test_database_status_is_per_database_and_preserves_dictionary_conversion_rules():
    from easyicu.concept.provenance import ExportDataProvenance, capture_data_provenance
    from tests.support.data_provenance import synthetic_dictionary

    snapshot = capture_data_provenance(
        synthetic_dictionary(),
        concepts=["lab_signal", "coded_condition"],
        source_database="miiv",
    )
    status = database_preparation_status(
        ExportDataProvenance(snapshots=(snapshot,)), "lab_signal"
    )
    assert set(status) == {"mimic", "miiv", "eicu", "aumc", "hirid", "sic"}
    assert status["miiv"]["unit_conversions"]
    assert status["miiv"]["raw_values_available"]
    assert all(not status[db]["raw_values_available"] for db in status if db != "miiv")
    coded = database_preparation_status(
        ExportDataProvenance(snapshots=(snapshot,)), "coded_condition"
    )
    assert not coded["miiv"]["raw_values_available"]


def test_model_reference_must_match_the_reviewed_preparation_choice():
    from easyicu.research_agent.contracts.model_terms import ModelTermSpec
    from easyicu.research_agent.execution.model_matrix import compile_prepared_terms

    frame = pd.DataFrame({"x": [0, 1] * 20, "y": np.arange(40.0)})
    terms = [
        ModelTermSpec(
            name="x",
            role="exposure",
            coding="binary",
            transform="treatment_contrast",
            levels=["0", "1"],
            reference_level="0",
        )
    ]
    plan = DataPreparationPlan(
        scaling_coding=[{"variable": "x", "reference_level": "1"}]
    )
    with pytest.raises(PreparationRefusal) as caught:
        compile_prepared_terms(frame, plan=plan, terms=terms, exposure="x", outcome="y")
    assert caught.value.reason_code == "preparation_coding_conflict"


def test_covariate_spline_uses_the_reviewed_knots_and_preserves_source_lineage():
    from easyicu.research_agent.contracts.model_terms import ModelTermSpec
    from easyicu.research_agent.execution.model_matrix import compile_prepared_terms
    from easyicu.research_agent.methods.rcs_dose_response import rcs_basis

    frame = pd.DataFrame(
        {"x": np.arange(50.0), "z": np.linspace(-2, 2, 50), "y": np.arange(50.0)}
    )
    terms = [
        ModelTermSpec(name=name, role=role, coding="continuous", transform="identity")
        for name, role in [("x", "exposure"), ("z", "covariate")]
    ]
    plan = DataPreparationPlan(
        scaling_coding=[{"variable": "z", "spline_knots": [-1.0, 0.0, 1.0]}]
    )
    compiled = compile_prepared_terms(
        frame, plan=plan, terms=terms, exposure="x", outcome="y"
    )
    assert list(compiled.design) == ["x", "z__rcs_0", "z__rcs_1"]
    np.testing.assert_allclose(
        compiled.design[["z__rcs_0", "z__rcs_1"]],
        rcs_basis(frame.z, knots=[-1.0, 0.0, 1.0]).as_array(),
    )
    assert compiled.source_by_design_column["z__rcs_1"] == "z"
