"""Model coding is a Planner contract, never a dtype heuristic."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.execution.model_matrix import (
    ModelTermCompilationError,
    compile_model_terms,
)


def _terms() -> list[ModelTermSpec]:
    return [
        ModelTermSpec(
            name="treatment",
            role="exposure",
            coding="binary",
            levels=["0", "1"],
            reference_level="0",
            transform="treatment_contrast",
        ),
        ModelTermSpec(
            name="site_code",
            role="covariate",
            coding="categorical",
            levels=["10", "20", "30"],
            reference_level="20",
            transform="treatment_contrast",
        ),
    ]


def test_continuous_term_cannot_smuggle_a_level_contract() -> None:
    with pytest.raises(ValidationError, match="cannot declare levels"):
        ModelTermSpec(
            name="age",
            role="covariate",
            coding="continuous",
            levels=["18", "65"],
            transform="identity",
        )


def test_numeric_multicategory_is_treatment_coded_from_the_declaration() -> None:
    frame = pd.DataFrame({"treatment": [0, 1, 0, 1], "site_code": [10, 20, 30, 10]})

    compiled = compile_model_terms(frame, terms=_terms(), exposure="treatment")

    assert list(compiled.design.columns) == [
        "treatment__is_1",
        "site_code__is_10",
        "site_code__is_30",
    ]
    assert compiled.exposure_columns == ("treatment__is_1",)
    assert compiled.source_by_design_column["site_code__is_30"] == "site_code"


def test_missing_category_is_not_pooled_with_the_reference() -> None:
    frame = pd.DataFrame(
        {"treatment": [0, 1, None, 1], "site_code": [10, None, 30, 20]}
    )

    compiled = compile_model_terms(frame, terms=_terms(), exposure="treatment")

    assert pd.isna(compiled.design.loc[2, "treatment__is_1"])
    assert pd.isna(compiled.design.loc[1, "site_code__is_10"])
    assert pd.isna(compiled.design.loc[1, "site_code__is_30"])


def test_pandas_nullable_missing_category_is_not_an_observed_level() -> None:
    frame = pd.DataFrame(
        {
            "treatment": pd.Series([0, 1, pd.NA, 1], dtype="Int64"),
            "site_code": pd.Series([10, pd.NA, 30, 20], dtype="Int64"),
        }
    )

    compiled = compile_model_terms(frame, terms=_terms(), exposure="treatment")

    assert pd.isna(compiled.design.loc[2, "treatment__is_1"])
    assert pd.isna(compiled.design.loc[1, "site_code__is_10"])
    assert pd.isna(compiled.design.loc[1, "site_code__is_30"])


def test_observed_level_outside_the_closed_contract_fails_locally() -> None:
    frame = pd.DataFrame({"treatment": [0, 1, 0, 1], "site_code": [10, 20, 40, 10]})

    with pytest.raises(ModelTermCompilationError) as exc_info:
        compile_model_terms(frame, terms=_terms(), exposure="treatment")

    assert exc_info.value.reason_code == "model_term_observed_level_undeclared"
