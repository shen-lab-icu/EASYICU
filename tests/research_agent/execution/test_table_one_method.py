from __future__ import annotations

import math

import pandas as pd
import pytest

from easyicu.research_agent.contracts.table_one import table_one_output_findings
from easyicu.research_agent.methods.table_one import (
    TableOneContractError,
    build_grouped_table_one,
)
from easyicu.research_agent.schema import AnalysisStep


def _spec() -> dict:
    return {
        "group_by": "arm",
        "group_levels": [0, 1],
        "variables": [
            {
                "name": "age",
                "variable_kind": "continuous",
                "summary": "median_iqr",
                "test": "mann_whitney_or_kruskal",
            },
            {
                "name": "sex",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": ["F", "M"],
            },
        ],
    }


def _repeated_unit_spec() -> dict:
    spec = _spec()
    spec.update(
        schema_version="easyicu.table_one/2",
        p_values_required=False,
        p_value_adjustment="not_applicable_repeated_units",
    )
    for variable in spec["variables"]:
        variable["test"] = "none_descriptive_smd_only"
    return spec


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": [0, 0, 0, 1, 1, 1],
            "age": [50.0, None, 70.0, 60.0, 80.0, 90.0],
            "sex": ["F", "M", None, "F", "F", "M"],
        }
    )


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="02_table_one",
        intent="Produce the grouped baseline table.",
        inputs=["arm", "age", "sex"],
        expected_outputs=["table:table_one"],
        method="table_one",
        table_one_spec=_spec(),
    )


def test_grouped_table_one_has_overall_groups_tests_and_correct_missingness():
    table = build_grouped_table_one(_frame(), _spec())

    assert set(table["schema_version"]) == {"easyicu.table_one_result/3"}
    assert set(table["group"]) == {"Overall", "0", "1"}
    age_zero = table[(table["variable"] == "age") & (table["group"] == "0")]
    assert age_zero.iloc[0]["denominator_n"] == 3
    assert age_zero.iloc[0]["missing_n"] == 1
    assert age_zero.iloc[0]["missing_pct"] == pytest.approx(100 / 3)
    sex_zero_f = table[
        (table["variable"] == "sex")
        & (table["group"] == "0")
        & (table["category"] == "F")
    ].iloc[0]
    assert sex_zero_f["count"] == 1
    assert sex_zero_f["percentage"] == 50.0
    assert table.groupby("variable")["p_value"].nunique().to_dict() == {
        "age": 1,
        "sex": 1,
    }
    assert set(table["test_name"]) == {"mann_whitney_u", "fisher_exact"}


@pytest.mark.parametrize("levels", [["Overall", "right"], ["left", "Overall"], ["left", "Overall", "right"]])
@pytest.mark.parametrize("descriptive_only", [False, True])
def test_table_one_accepts_a_declared_group_named_overall(tmp_path, levels, descriptive_only):
    spec = _repeated_unit_spec() if descriptive_only else _spec()
    spec["group_levels"] = levels
    frame = pd.DataFrame({
        "arm": [level for level in levels for _ in range(3)],
        "age": [15.0, None, 25.0] * len(levels),
        "sex": ["F", "M", None] * len(levels),
    })
    step = AnalysisStep.model_validate({**_step().model_dump(), "table_one_spec": spec})
    table = build_grouped_table_one(frame, spec)
    table.to_csv(tmp_path / "table_one.csv", index=False)

    assert table_one_output_findings(step=step, out_dir=tmp_path) == []
    age = table[table["variable"].eq("age")]
    assert age["group"].tolist() == ["Overall", *levels]
    assert age["denominator_n"].tolist() == [3 * len(levels), *([3] * len(levels))]
    assert age["row_role"].tolist() == ["overall", *(["exposure_level"] * len(levels))]

    # The collision must not hide a missing total or a total from another cohort.
    age_overall = table["variable"].eq("age") & table["group_order"].eq(0)
    table.loc[age_overall, "denominator_n"] += 1
    table.to_csv(tmp_path / "table_one.csv", index=False)
    assert "table_one_overall_denominator_mismatch" in {
        finding.detail["reason"] for finding in table_one_output_findings(step=step, out_dir=tmp_path)
    }
    table = table.loc[~age_overall]
    table.to_csv(tmp_path / "table_one.csv", index=False)
    assert "table_one_groups_incomplete" in {
        finding.detail["reason"] for finding in table_one_output_findings(step=step, out_dir=tmp_path)
    }


@pytest.mark.parametrize("case", ["empty_comparison", "constant_category"])
def test_overall_named_group_keeps_untestable_comparisons_honest(tmp_path, case):
    spec = _spec()
    spec["group_levels"] = ["Overall", "right"]
    frame = _frame()
    frame["arm"] = ["Overall"] * 3 + ["right"] * 3
    if case == "empty_comparison":
        frame.loc[frame["arm"].eq("Overall"), "age"] = None
    else:
        frame["sex"] = "F"
    table = build_grouped_table_one(frame, spec)
    table.to_csv(tmp_path / "table_one.csv", index=False)
    step = AnalysisStep.model_validate({**_step().model_dump(), "table_one_spec": spec})

    assert table_one_output_findings(step=step, out_dir=tmp_path) == []


def test_repeated_unit_table_one_reports_descriptions_and_smd_without_p_values(
    tmp_path,
):
    table = build_grouped_table_one(_frame(), _repeated_unit_spec())

    assert table["p_value"].isna().all()
    assert set(table["test_name"]) == {"not_reported_repeated_units"}
    assert table["standardized_mean_difference"].notna().all()
    table.to_csv(tmp_path / "table_one.csv", index=False)
    step = AnalysisStep(
        step_id="02_table_one",
        intent="Describe repeated ICU stays without independent-row tests.",
        inputs=["arm", "age", "sex"],
        expected_outputs=["table:table_one"],
        method="table_one",
        table_one_spec=_repeated_unit_spec(),
    )
    assert table_one_output_findings(step=step, out_dir=tmp_path) == []


def test_repeated_unit_table_one_gate_rejects_a_fabricated_p_value(tmp_path):
    table = build_grouped_table_one(_frame(), _repeated_unit_spec())
    table.loc[table["variable"] == "age", "p_value"] = 0.04
    table.loc[table["variable"] == "age", "test_name"] = "mann_whitney_u"
    table.to_csv(tmp_path / "table_one.csv", index=False)
    step = AnalysisStep(
        step_id="02_table_one",
        intent="Describe repeated ICU stays without independent-row tests.",
        inputs=["arm", "age", "sex"],
        expected_outputs=["table:table_one"],
        method="table_one",
        table_one_spec=_repeated_unit_spec(),
    )

    reasons = {
        finding.detail["reason"]
        for finding in table_one_output_findings(step=step, out_dir=tmp_path)
    }
    assert "table_one_p_value_invalid" in reasons
    assert "table_one_test_mismatch" in reasons


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": "easyicu.table_one/1"},
        {"p_values_required": True},
        {"variables.0.test": "mann_whitney_or_kruskal"},
    ],
)
def test_repeated_unit_table_one_rejects_half_configured_test_contract(mutation):
    spec = _repeated_unit_spec()
    for coordinate, value in mutation.items():
        if coordinate == "variables.0.test":
            spec["variables"][0]["test"] = value
        else:
            spec[coordinate] = value

    with pytest.raises(ValueError, match="Table One"):
        build_grouped_table_one(_frame(), spec)


def test_grouped_table_one_emits_binary_continuous_and_category_smd():
    table = build_grouped_table_one(_frame(), _spec())

    age = table[table["variable"] == "age"]
    expected_age = ((60.0 + 80.0 + 90.0) / 3.0 - (50.0 + 70.0) / 2.0) / math.sqrt(
        (pd.Series([50.0, 70.0]).var() + pd.Series([60.0, 80.0, 90.0]).var()) / 2.0
    )
    assert age["standardized_mean_difference"].tolist() == pytest.approx(
        [expected_age] * len(age)
    )
    assert age["absolute_standardized_mean_difference"].tolist() == pytest.approx(
        [abs(expected_age)] * len(age)
    )
    assert set(age["standardized_difference_status"]) == {"computed"}
    assert set(age["standardized_difference_reference_group"]) == {"0"}
    assert set(age["standardized_difference_comparison_group"]) == {"1"}

    sex_f = table[(table["variable"] == "sex") & (table["category"] == "F")]
    reference = 1.0 / 2.0
    comparison = 2.0 / 3.0
    expected_sex_f = (comparison - reference) / math.sqrt(
        (reference * (1.0 - reference) + comparison * (1.0 - comparison)) / 2.0
    )
    assert sex_f["standardized_mean_difference"].tolist() == pytest.approx(
        [expected_sex_f] * len(sex_f)
    )
    sex_m = table[(table["variable"] == "sex") & (table["category"] == "M")]
    assert sex_m["standardized_mean_difference"].tolist() == pytest.approx(
        [-expected_sex_f] * len(sex_m)
    )


def test_grouped_table_one_marks_multi_group_smd_not_applicable():
    frame = pd.DataFrame(
        {
            "arm": [0, 0, 1, 1, 2, 2],
            "age": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "sex": ["F", "M", "F", "M", "F", "M"],
        }
    )
    spec = _spec()
    spec["group_levels"] = [0, 1, 2]

    table = build_grouped_table_one(frame, spec)

    assert table["standardized_mean_difference"].isna().all()
    assert table["absolute_standardized_mean_difference"].isna().all()
    assert set(table["standardized_difference_status"]) == {
        "not_applicable_more_than_two_groups"
    }
    assert table["standardized_difference_reference_group"].isna().all()
    assert table["standardized_difference_comparison_group"].isna().all()


def test_grouped_table_one_handles_zero_pooled_variance_without_infinity():
    frame = _frame()
    frame["age"] = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    same = build_grouped_table_one(frame, _spec())
    age_same = same[same["variable"] == "age"]
    assert age_same["p_value"].eq(1.0).all()
    assert set(age_same["test_name"]) == {"mann_whitney_u"}
    assert age_same["standardized_mean_difference"].eq(0.0).all()
    assert set(age_same["standardized_difference_status"]) == {"computed"}

    frame.loc[frame["arm"] == 1, "age"] = 60.0
    different = build_grouped_table_one(frame, _spec())
    age_different = different[different["variable"] == "age"]
    assert age_different["standardized_mean_difference"].isna().all()
    assert age_different["absolute_standardized_mean_difference"].isna().all()
    assert set(age_different["standardized_difference_status"]) == {
        "undefined_zero_pooled_variance"
    }


def test_grouped_table_one_matches_json_integer_levels_to_parquet_floats():
    frame = _frame()
    frame["arm"] = frame["arm"].astype(float)

    table = build_grouped_table_one(frame, _spec())

    assert set(table["group"]) == {"Overall", "0", "1"}
    assert table.loc[table["group"] == "0", "denominator_n"].eq(3).all()
    assert table.loc[table["group"] == "1", "denominator_n"].eq(3).all()


def test_grouped_table_one_restores_declared_boolean_from_float_zero_one():
    frame = pd.DataFrame(
        {
            "exposed": [0.0, 0.0, 1.0, 1.0],
            "readmitted": [0.0, 1.0, 0.0, 1.0],
        }
    )
    spec = {
        "group_by": "exposed",
        "group_levels": [False, True],
        "variables": [
            {
                "name": "readmitted",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": [False, True],
            }
        ],
    }

    table = build_grouped_table_one(frame, spec)

    assert set(table["group"]) == {"Overall", "False", "True"}
    assert set(table["category"]) == {"False", "True"}
    assert frame.dtypes.to_dict() == {
        "exposed": "float64",
        "readmitted": "float64",
    }


def test_grouped_table_one_does_not_coerce_non_binary_numeric_to_boolean():
    frame = pd.DataFrame(
        {
            "exposed": [0.0, 1.0, 2.0],
            "readmitted": [0.0, 1.0, 0.0],
        }
    )
    spec = {
        "group_by": "exposed",
        "group_levels": [False, True],
        "variables": [
            {
                "name": "readmitted",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": [False, True],
            }
        ],
    }

    with pytest.raises(TableOneContractError, match="outside the Planner-declared"):
        build_grouped_table_one(frame, spec)


def test_grouped_table_one_rejects_missing_group_values():
    frame = _frame()
    frame.loc[0, "arm"] = None
    with pytest.raises(
        TableOneContractError, match="grouping variable contains missing"
    ):
        build_grouped_table_one(frame, _spec())


def test_grouped_table_one_rejects_undeclared_categories():
    frame = _frame()
    frame.loc[0, "sex"] = "X"
    with pytest.raises(TableOneContractError, match="outside the Planner-declared"):
        build_grouped_table_one(frame, _spec())


def test_grouped_table_one_rejects_string_coercion_for_continuous_values():
    frame = _frame()
    frame["age"] = frame["age"].astype("string")
    with pytest.raises(TableOneContractError, match="must be numeric"):
        build_grouped_table_one(frame, _spec())


def test_grouped_table_one_marks_structurally_empty_comparison_not_testable(
    tmp_path,
):
    frame = _frame()
    frame["exposure"] = [None, None, None, 1.0, 2.0, 3.0]
    spec = _spec()
    spec["variables"].append(
        {
            "name": "exposure",
            "variable_kind": "continuous",
            "summary": "median_iqr",
            "test": "mann_whitney_or_kruskal",
        }
    )

    table = build_grouped_table_one(frame, spec)
    exposure = table[table["variable"] == "exposure"]
    assert exposure["p_value"].isna().all()
    assert set(exposure["test_name"]) == {"not_testable_empty_group"}

    table.to_csv(tmp_path / "table_one.csv", index=False)
    step = AnalysisStep(
        step_id="02_table_one",
        intent="Produce the grouped baseline table.",
        inputs=["arm", "age", "sex", "exposure"],
        expected_outputs=["table:table_one"],
        method="table_one",
        table_one_spec=spec,
    )
    assert table_one_output_findings(step=step, out_dir=tmp_path) == []


def test_table_one_gate_rejects_unexplained_missing_p_value(tmp_path):
    table = build_grouped_table_one(_frame(), _spec())
    table.loc[table["variable"] == "age", "p_value"] = None
    table.loc[table["variable"] == "age", "test_name"] = "not_testable_empty_group"
    table.to_csv(tmp_path / "table_one.csv", index=False)

    findings = table_one_output_findings(step=_step(), out_dir=tmp_path)
    assert "table_one_p_value_invalid" in {
        finding.detail["reason"] for finding in findings
    }


def test_grouped_table_one_marks_constant_category_not_testable(tmp_path):
    frame = _frame()
    frame["sex"] = "F"

    table = build_grouped_table_one(frame, _spec())
    sex = table[table["variable"] == "sex"]
    assert sex["p_value"].isna().all()
    assert set(sex["test_name"]) == {"not_testable_no_variation"}

    table.to_csv(tmp_path / "table_one.csv", index=False)
    assert table_one_output_findings(step=_step(), out_dir=tmp_path) == []


@pytest.mark.parametrize("group_order", [True, False])
def test_table_one_output_gate_accepts_exact_sdk_result(tmp_path, group_order):
    table = build_grouped_table_one(_frame(), _spec())
    if not group_order:
        table = table.drop(columns=["group_order"])
    table.to_csv(tmp_path / "table_one.csv", index=False)
    assert table_one_output_findings(step=_step(), out_dir=tmp_path) == []


def test_table_one_output_gate_preserves_numeric_category_identity(tmp_path):
    frame = _frame().assign(death=[0, 1, 0, 0, 1, 1])
    spec = _repeated_unit_spec()
    spec["variables"].append(
        {
            "name": "death",
            "variable_kind": "categorical",
            "summary": "count_percent",
            "test": "none_descriptive_smd_only",
            "levels": [0, 1],
        }
    )
    build_grouped_table_one(frame, spec).to_csv(
        tmp_path / "table_one.csv", index=False
    )
    step = AnalysisStep(
        step_id="02_table_one",
        intent="Describe a binary numeric category without changing its level identity.",
        inputs=["arm", "age", "sex", "death"],
        expected_outputs=["table:table_one"],
        method="table_one",
        table_one_spec=spec,
    )

    assert table_one_output_findings(step=step, out_dir=tmp_path) == []


def test_table_one_output_gate_rejects_tampered_smd(tmp_path):
    table = build_grouped_table_one(_frame(), _spec())
    table.loc[
        (table["variable"] == "age") & (table["group"] == "Overall"),
        "standardized_mean_difference",
    ] = 0.0
    table.to_csv(tmp_path / "table_one.csv", index=False)

    findings = table_one_output_findings(step=_step(), out_dir=tmp_path)
    assert "table_one_standardized_difference_invalid" in {
        finding.detail["reason"] for finding in findings
    }


def test_table_one_output_gate_requires_smd_contract_columns(tmp_path):
    table = build_grouped_table_one(_frame(), _spec()).drop(
        columns=["standardized_mean_difference"]
    )
    table.to_csv(tmp_path / "table_one.csv", index=False)

    findings = table_one_output_findings(step=_step(), out_dir=tmp_path)
    assert [finding.detail["reason"] for finding in findings] == [
        "table_one_schema_incomplete"
    ]


def test_table_one_output_gate_rejects_wrong_missingness_denominator(tmp_path):
    table = build_grouped_table_one(_frame(), _spec())
    table.loc[table.index[0], "missing_pct"] = 99.0
    table.to_csv(tmp_path / "table_one.csv", index=False)

    findings = table_one_output_findings(step=_step(), out_dir=tmp_path)
    assert "table_one_missingness_denominator_invalid" in {
        finding.detail["reason"] for finding in findings
    }


def test_table_one_output_gate_rejects_ungrouped_legacy_shape(tmp_path):
    pd.DataFrame([{"variable": "age", "missing_n": 0, "missing_pct": 0.0}]).to_csv(
        tmp_path / "table_one.csv", index=False
    )

    findings = table_one_output_findings(step=_step(), out_dir=tmp_path)
    assert [finding.detail["reason"] for finding in findings] == [
        "table_one_schema_incomplete"
    ]
