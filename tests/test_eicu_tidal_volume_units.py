from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.callback_apply import (
    _normalize_eicu_tidal_volume_frame,
    _parse_eicu_age,
)


def _mixed_frame(values, *, stays=None, labels=None, ages=None, concept="tidal_vol"):
    count = len(values)
    frame = pd.DataFrame(
        {
            "patientunitstayid": stays or list(range(1, count + 1)),
            "respchartvaluelabel": labels
            or ["Exhaled TV (patient)"] * count,
            concept: values,
        }
    )
    return _normalize_eicu_tidal_volume_frame(
        frame,
        concept_name=concept,
        ages=pd.Series(ages or [40.0] * count, index=frame.index),
    )


def test_eicu_age_parser_preserves_missing_and_maps_over_89_sentinel() -> None:
    parsed = _parse_eicu_age(pd.Series([None, np.nan, "> 89", "62"]))

    assert parsed.iloc[:2].isna().all()
    assert parsed.iloc[2:].tolist() == [90.0, 62.0]


def test_same_stay_ml_reference_converts_liter_decimal() -> None:
    result = _mixed_frame(
        [550.0, 300.0, 0.3],
        stays=[168728, 168728, 168728],
        labels=["Tidal Volume Observed (VT)"] * 3,
        ages=[62.0] * 3,
    )

    assert result["tidal_vol"].tolist() == [550.0, 300.0, 300.0]
    assert result.attrs["eicu_tidal_volume_unit_audit"][
        "same_stay_l_to_ml_rows"
    ] == 1


def test_pure_low_adult_converts_but_ambiguous_child_fails_closed() -> None:
    result = _mixed_frame(
        [0.4, 0.4],
        stays=[1, 2],
        ages=[45.0, 12.0],
    )

    assert result.loc[0, "tidal_vol"] == 400.0
    assert np.isnan(result.loc[1, "tidal_vol"])


def test_unknown_age_low_value_fails_closed() -> None:
    result = _mixed_frame([0.4], ages=[np.nan])

    assert np.isnan(result.loc[0, "tidal_vol"])
    assert result.attrs["eicu_tidal_volume_unit_audit"][
        "ambiguous_low_quarantined_rows"
    ] == 1


def test_explicit_ml_child_is_preserved_and_adult_implausible_value_is_quarantined() -> None:
    result = _mixed_frame(
        [0.5, 10.0],
        stays=[1, 2],
        labels=["Vt Spontaneous (mL)", "Exhaled TV (machine)"],
        ages=[4.0, 50.0],
    )

    assert result.loc[0, "tidal_vol"] == 0.5
    assert np.isnan(result.loc[1, "tidal_vol"])


def test_zero_is_preserved() -> None:
    result = _mixed_frame([0.0], ages=[50.0])

    assert result.loc[0, "tidal_vol"] == 0.0
    assert result.attrs["eicu_tidal_volume_unit_audit"]["zero_rows_preserved"] == 1


def test_drager_is_fixed_liter_source_and_other_set_label_stays_ml() -> None:
    drager = pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 1, 1],
            "respchartvaluelabel": ["Set Vt (Drager)"] * 4,
            "tidal_vol_set": [0.5, 500.0, 19.0, 0.0],
            "age": [45, 45, 45, 45],
        }
    )
    drager_result = _apply_callback(
        drager,
        ConceptSource(
            callback="eicu_tidal_volume_drager_l_to_ml",
            sub_var="respchartvaluelabel",
            value_var="tidal_vol_set",
        ),
        concept_name="tidal_vol_set",
    )
    other_result = _mixed_frame(
        [500.0],
        labels=["Tidal Volume (set)"],
        ages=[45.0],
        concept="tidal_vol_set",
    )

    assert drager_result.loc[0, "tidal_vol_set"] == 500.0
    assert drager_result.loc[1, "tidal_vol_set"] == 500.0
    assert np.isnan(drager_result.loc[2, "tidal_vol_set"])
    assert drager_result.loc[3, "tidal_vol_set"] == 0.0
    assert other_result.loc[0, "tidal_vol_set"] == 500.0


def test_deprecated_loader_projection_receives_value_and_time_columns() -> None:
    frame = pd.DataFrame(
        {
            "patientunitstayid": [1],
            "respchartoffset": [60],
            "respchartvaluelabel": ["Set Vt (Drager)"],
            "respchartvalue": ["0.5"],
            "age": [45],
        }
    )
    result = _apply_callback(
        frame,
        ConceptSource(
            callback="eicu_tidal_volume_drager_l_to_ml",
            sub_var="respchartvaluelabel",
            value_var="respchartvalue",
        ),
        concept_name="tidal_vol_set",
    )

    assert result.loc[0, "value"] == 500.0
    assert result.loc[0, "time"] == 60


def test_lab_tv_explicit_ml_source_quarantines_only_uncredible_small_values() -> None:
    frame = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3, 4],
            "TV": [500.0, 0.5, 8.0, 8.0],
            "age": [60.0, 60.0, 8.0, np.nan],
        }
    )
    result = _apply_callback(
        frame,
        ConceptSource(
            table="lab",
            value_var="TV",
            callback="eicu_tidal_volume_explicit_ml",
        ),
        concept_name="tidal_vol",
    )

    assert result.loc[0, "value"] == 500.0
    assert np.isnan(result.loc[1, "value"])
    assert result.loc[2, "value"] == 8.0
    assert np.isnan(result.loc[3, "value"])


def test_catalog_scopes_callbacks_to_respiratory_sources() -> None:
    catalog_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "easyicu"
        / "data"
        / "concept-dict.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    measured = catalog["tidal_vol"]["sources"]["eicu"]
    assert measured[0]["callback"] == "eicu_tidal_volume_mixed_scale"
    assert measured[1]["table"] == "lab"
    assert measured[1]["callback"] == "eicu_tidal_volume_explicit_ml"

    set_sources = catalog["tidal_vol_set"]["sources"]["eicu"]
    assert set_sources[0]["callback"] == "eicu_tidal_volume_mixed_scale"
    assert "Set Vt (Drager)" not in set_sources[0]["ids"]
    assert set_sources[1]["ids"] == "Set Vt (Drager)"
    assert set_sources[1]["callback"] == "eicu_tidal_volume_drager_l_to_ml"
