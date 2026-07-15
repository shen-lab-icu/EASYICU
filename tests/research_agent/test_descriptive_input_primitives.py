from __future__ import annotations

import math

import pandas as pd
import pytest

from easyicu.research_agent.methods.descriptive_inputs import (
    DescriptiveInputError,
    closed_categorical_counts,
    measurement_provenance_receipt,
    strict_numeric_input,
)


def test_strict_numeric_input_preserves_index_missingness_and_audit() -> None:
    source = pd.Series(
        ["1.5", 2, None, pd.NA],
        index=["stay-4", "stay-7", "stay-9", "stay-12"],
        name="agent_selected_measurement",
        dtype="object",
    )

    result = strict_numeric_input(source)

    assert result.values.index.equals(source.index)
    assert result.values.name == source.name
    assert result.values.iloc[:2].tolist() == [1.5, 2.0]
    assert result.values.iloc[2:].isna().all()
    assert result.audit == {
        "n_total": 4,
        "raw_missing_n": 2,
        "numeric_n": 2,
        "coercion_invalid_n": 0,
        "semantic_invalid_n": 0,
        "nonfinite_n": 0,
    }


@pytest.mark.parametrize(
    ("source", "field"),
    [
        (pd.Series([1, "not-a-number", None]), "coercion_invalid_n"),
        (pd.Series([1, float("inf"), None]), "nonfinite_n"),
        (pd.Series(pd.to_datetime(["2020-01-01", None])), "semantic_invalid_n"),
        (pd.Series([True, False]), "semantic_invalid_n"),
    ],
)
def test_strict_numeric_input_fails_closed_with_audit(
    source: pd.Series,
    field: str,
) -> None:
    with pytest.raises(DescriptiveInputError) as exc_info:
        strict_numeric_input(source)

    assert exc_info.value.audit[field] > 0


def test_closed_categorical_counts_canonicalizes_numeric_equivalents() -> None:
    source = pd.Series([0, 0.0, "0.0", 1, "1.0", None], dtype="object")

    result = closed_categorical_counts(source, declared_levels=[0, 1, 2])

    assert result.table.to_dict(orient="records") == [
        {"level": 0, "count": 3},
        {"level": 1, "count": 2},
        {"level": 2, "count": 0},
    ]
    assert result.audit == {
        "n_total": 6,
        "nonmissing_n": 5,
        "missing_n": 1,
        "declared_level_n": 3,
        "undeclared_n": 0,
        "closed_count_n": 5,
    }


def test_closed_categorical_counts_rejects_undeclared_nonmissing_values() -> None:
    with pytest.raises(DescriptiveInputError) as exc_info:
        closed_categorical_counts(
            pd.Series(["low", "unexpected", None]),
            declared_levels=["low", "high"],
        )

    assert exc_info.value.audit == {
        "n_total": 3,
        "nonmissing_n": 2,
        "missing_n": 1,
        "declared_level_n": 2,
        "undeclared_n": 1,
    }


def test_closed_categorical_counts_rejects_duplicate_numeric_declarations() -> None:
    with pytest.raises(ValueError, match="duplicated after canonicalization"):
        closed_categorical_counts(
            pd.Series([0]),
            declared_levels=[0, 0.0],
        )


def test_closed_categorical_counts_preserves_zero_padded_identifiers() -> None:
    result = closed_categorical_counts(
        pd.Series(["01", 1, "1.0"], dtype="object"),
        declared_levels=["01", 1],
    )

    assert result.table[["level", "count"]].to_dict(orient="records") == [
        {"level": "01", "count": 1},
        {"level": 1, "count": 2},
    ]


def test_closed_categorical_counts_empty_observations_remain_closed() -> None:
    result = closed_categorical_counts(
        pd.Series([None, pd.NA], dtype="object"),
        declared_levels=["absent", "present"],
    )

    assert result.table["count"].tolist() == [0, 0]
    assert list(result.table.columns) == ["level", "count"]
    assert result.audit["closed_count_n"] == 0


def test_measurement_provenance_receipt_is_metadata_only() -> None:
    frame = pd.DataFrame(
        {
            "selected_measured": [True, False, "1"],
            "selected_n": [2, 0, "3"],
            "analysis_value": [5.0, 4.0, 3.0],
        },
        index=[101, 205, 309],
    )

    receipt = measurement_provenance_receipt(
        frame,
        measured_column="selected_measured",
        count_column="selected_n",
    )

    assert receipt == {
        "measured_column": "selected_measured",
        "count_column": "selected_n",
        "status": "checked",
        "comparison_n": 3,
        "invalid_pair_n": 0,
        "discordant_n": 0,
        "role": "audit_only",
    }
    assert not any(
        token in key
        for key in receipt
        for token in ("mask", "filter", "rows", "values")
    )
    assert frame.index.tolist() == [101, 205, 309]


@pytest.mark.parametrize(
    ("measured", "counts", "audit_field"),
    [
        ([1, 2, 0], [1, 1, 0], "invalid_pair_n"),
        ([1, 0, 0], [1, -1, 0], "invalid_pair_n"),
        ([1, 1, 0], [1, 0, 0], "discordant_n"),
        ([1, 0, 0], [1, 0.5, 0], "invalid_pair_n"),
        ([1, 0, 0], [1, math.inf, 0], "invalid_pair_n"),
        ([1, 0, 0], [1, 1 + 0j, 0], "invalid_pair_n"),
    ],
)
def test_measurement_provenance_receipt_fails_closed(
    measured: list[object],
    counts: list[object],
    audit_field: str,
) -> None:
    frame = pd.DataFrame({"selected_measured": measured, "selected_n": counts})

    with pytest.raises(DescriptiveInputError) as exc_info:
        measurement_provenance_receipt(
            frame,
            measured_column="selected_measured",
            count_column="selected_n",
        )

    assert exc_info.value.audit[audit_field] > 0


def test_measurement_provenance_receipt_does_not_accept_other_columns() -> None:
    frame = pd.DataFrame({"other_measured": [1], "other_n": [1]})

    with pytest.raises(DescriptiveInputError, match="columns missing"):
        measurement_provenance_receipt(
            frame,
            measured_column="selected_measured",
            count_column="selected_n",
        )


def test_measurement_provenance_receipt_requires_distinct_role_columns() -> None:
    frame = pd.DataFrame({"selected": [0, 1]})

    with pytest.raises(DescriptiveInputError, match="distinct columns"):
        measurement_provenance_receipt(
            frame,
            measured_column="selected",
            count_column="selected",
        )
