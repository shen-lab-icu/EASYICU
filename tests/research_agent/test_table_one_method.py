from __future__ import annotations

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


def test_grouped_table_one_matches_json_integer_levels_to_parquet_floats():
    frame = _frame()
    frame["arm"] = frame["arm"].astype(float)

    table = build_grouped_table_one(frame, _spec())

    assert set(table["group"]) == {"Overall", "0", "1"}
    assert table.loc[table["group"] == "0", "denominator_n"].eq(3).all()
    assert table.loc[table["group"] == "1", "denominator_n"].eq(3).all()


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


def test_table_one_output_gate_accepts_exact_sdk_result(tmp_path):
    build_grouped_table_one(_frame(), _spec()).to_csv(
        tmp_path / "table_one.csv", index=False
    )
    assert table_one_output_findings(step=_step(), out_dir=tmp_path) == []


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
