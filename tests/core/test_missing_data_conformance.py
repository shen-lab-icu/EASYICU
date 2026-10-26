from __future__ import annotations

import pandas as pd
import pytest

from easyicu.callbacks import safi
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_pafi
from easyicu.table import ICUTable


@pytest.mark.clinical_conformance
def test_safi_room_air_imputation_is_machine_readable() -> None:
    spo2 = pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [0.0, 0.0], "spo2": [96.0, 96.0]}
    )
    fio2 = pd.DataFrame(
        {"stay_id": [1], "charttime": [0.0], "fio2": [40.0]}
    )

    result = safi(spo2, fio2, fix_na_fio2=True)

    by_stay = result.set_index("stay_id")
    assert bool(by_stay.loc[1, "fio2_observed"]) is True
    assert bool(by_stay.loc[1, "fio2_imputed"]) is False
    assert by_stay.loc[1, "fio2_assessment_reason"] == "observed"
    assert bool(by_stay.loc[2, "fio2_observed"]) is False
    assert bool(by_stay.loc[2, "fio2_imputed"]) is True
    assert by_stay.loc[2, "fio2_assessment_reason"] == "room_air_assumption"


def _ratio_table(name: str, values) -> ICUTable:
    return ICUTable(
        pd.DataFrame(
            {
                "stay_id": [1] * len(values),
                "charttime": [0.0] * len(values),
                name: values,
            }
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column=name,
    )


def _pafi_context(**kwargs) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name="pafi",
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
        kwargs=kwargs,
    )


@pytest.mark.clinical_conformance
def test_canonical_pafi_keeps_missing_fio2_unknown_by_default() -> None:
    result = _callback_pafi(
        {"po2": _ratio_table("po2", [80.0]), "fio2": _ratio_table("fio2", [None])},
        _pafi_context(),
        source_col_a="po2",
        source_col_b="fio2",
        output_col="pafi",
    ).data

    assert result["pafi"].isna().tolist() == [True]
    assert result["fio2_observed"].tolist() == [False]
    assert result["fio2_imputed"].tolist() == [False]
    assert result["fio2_assessment_reason"].tolist() == ["missing_fio2"]


@pytest.mark.parametrize("empty_name", ["po2", "fio2"])
def test_empty_ratio_component_without_a_time_column_keeps_missingness(empty_name):
    tables = {"po2": _ratio_table("po2", [80.0]), "fio2": _ratio_table("fio2", [40.0])}
    tables[empty_name] = ICUTable(
        pd.DataFrame(columns=["stay_id", empty_name]),
        id_columns=["stay_id"], value_column=empty_name,
    )
    result = _callback_pafi(
        tables, _pafi_context(), source_col_a="po2", source_col_b="fio2", output_col="pafi",
    ).data
    if empty_name == "po2":
        assert result.empty
    else:
        assert result["pafi"].isna().all()
        assert result["fio2_observed"].tolist() == [False]
        assert result["fio2_imputed"].tolist() == [False]
        assert result["fio2_assessment_reason"].tolist() == ["missing_fio2"]


@pytest.mark.clinical_conformance
def test_canonical_pafi_room_air_assumption_requires_explicit_opt_in() -> None:
    result = _callback_pafi(
        {"po2": _ratio_table("po2", [84.0]), "fio2": _ratio_table("fio2", [None])},
        _pafi_context(fix_na_fio2=True),
        source_col_a="po2",
        source_col_b="fio2",
        output_col="pafi",
    ).data

    assert result["pafi"].tolist() == [400.0]
    assert result["fio2_observed"].tolist() == [False]
    assert result["fio2_imputed"].tolist() == [True]
    assert result["fio2_assessment_reason"].tolist() == ["room_air_assumption"]
