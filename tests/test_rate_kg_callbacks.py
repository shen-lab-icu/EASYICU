from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import easyicu.concept.callbacks as concept_callbacks
from easyicu.utils.callback_utils import aumc_rate_kg, eicu_rate_kg_callback, hirid_rate_kg
from easyicu.concept.callbacks import ConceptCallbackContext
from easyicu.table import ICUTable


def _mimic_ctx() -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name="norepi_rate",
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
    )


def _mimic_rate_table() -> ICUTable:
    return ICUTable(
        pd.DataFrame(
            {
                "icustay_id": [1],
                "charttime": [0.0],
                "rate": [120.0],
                "rate_unit": ["mcgmin"],
            }
        ),
        id_columns=["icustay_id"],
        index_column="charttime",
        value_column="rate",
        unit_column="rate_unit",
    )


def test_mimic_kg_rate_missing_weight_drops_unscaled_rate(monkeypatch):
    monkeypatch.setattr(
        concept_callbacks,
        "_load_concept_for_callback",
        lambda ctx, concept_name: pd.DataFrame(columns=["icustay_id", "weight"]),
    )

    result = concept_callbacks._callback_mimic_kg_rate(
        {"rate": _mimic_rate_table()},
        _mimic_ctx(),
    )

    assert result.data.empty


def test_mimic_kg_rate_valid_weight_normalizes_and_relabels_unit(monkeypatch):
    monkeypatch.setattr(
        concept_callbacks,
        "_load_concept_for_callback",
        lambda ctx, concept_name: pd.DataFrame(
            {"icustay_id": [1], "weight": [60.0]}
        ),
    )

    result = concept_callbacks._callback_mimic_kg_rate(
        {"rate": _mimic_rate_table()},
        _mimic_ctx(),
    )

    assert result.data["rate"].iloc[0] == pytest.approx(2.0)
    assert result.data["rate_unit"].iloc[0] == "mcg/kg/min"


def test_eicu_rate_kg_missing_weight_leaves_non_perkg_rate_missing():
    callback = eicu_rate_kg_callback(ml_to_mcg=1.0)
    frame = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3],
            "drugrate": [120.0, 2.5, 120.0],
            "drugname": [
                "Norepinephrine (mcg/min)",
                "Norepinephrine (mcg/kg/min)",
                "Norepinephrine (mcg/min)",
            ],
            "patientweight": [np.nan, np.nan, 60.0],
        }
    )

    result = callback(
        frame,
        val_var="drugrate",
        sub_var="drugname",
        weight_var="patientweight",
        concept_name="norepi_rate",
    )

    by_id = result.set_index("patientunitstayid")["norepi_rate"]
    assert pd.isna(by_id.loc[1])
    assert by_id.loc[2] == pytest.approx(2.5)
    assert by_id.loc[3] == pytest.approx(2.0)


def test_aumc_rate_kg_missing_weight_drops_only_non_perkg_rows():
    frame = pd.DataFrame(
        {
            "admissionid": [1, 2, 3],
            "start": [0.0, 0.0, 0.0],
            "value": [120.0, 120.0, 2.5],
            "doseunit": ["mcg", "mcg", "mcg"],
            "doserateunit": ["min", "min", "min"],
            "doserateperkg": [False, False, True],
            "weight": [60.0, np.nan, np.nan],
        }
    )

    result = aumc_rate_kg(
        frame,
        concept_name="norepi_rate",
        val_col="value",
        unit_col="doseunit",
        rel_weight_col="doserateperkg",
        rate_unit_col="doserateunit",
        index_col="start",
        stop_col=None,
    )

    by_id = result.set_index("admissionid")["norepi_rate"]
    assert by_id.loc[1] == pytest.approx(2.0)
    assert 2 not in by_id.index
    assert by_id.loc[3] == pytest.approx(2.5)


def test_hirid_rate_kg_missing_weight_drops_patient_instead_of_defaulting_to_70kg():
    frame = pd.DataFrame(
        {
            "patientid": [1, 2],
            "datetime": [0.25, 0.25],
            "infusionid": [10, 20],
            "givendose": [120.0, 120.0],
            "doseunit": ["ug", "ug"],
            "weight": [60.0, np.nan],
        }
    )

    result = hirid_rate_kg(
        frame,
        concept_name="norepi_rate",
        val_col="givendose",
        unit_col="doseunit",
        grp_var="infusionid",
        index_col="datetime",
        interval_minutes=60.0,
    )

    by_id = result.set_index("patientid")["norepi_rate"]
    assert by_id.loc[1] == pytest.approx(120.0 / 60.0 / 60.0)
    assert 2 not in by_id.index


def test_hirid_rate_kg_without_weight_column_returns_empty():
    frame = pd.DataFrame(
        {
            "patientid": [1],
            "datetime": [0.25],
            "infusionid": [10],
            "givendose": [120.0],
            "doseunit": ["ug"],
        }
    )

    result = hirid_rate_kg(
        frame,
        concept_name="norepi_rate",
        val_col="givendose",
        unit_col="doseunit",
        grp_var="infusionid",
        index_col="datetime",
        interval_minutes=60.0,
    )

    assert result.empty


def test_hirid_rate_kg_with_value_column_pre_renamed_to_concept_name():
    """Regression: in the real load path the dose column is renamed to the
    concept name (givendose -> norepi_rate) *before* the callback runs, so
    val_col='givendose' is absent and actual_val_col collapses onto
    concept_name. A prior bug then zeroed grouped[concept_name] before reading
    it as the numerator, dropping every row to an empty result despite valid
    dose + weight. This guards that the renamed-column path still computes the
    rate. See callback_utils.hirid_rate_kg."""
    frame = pd.DataFrame(
        {
            "patientid": [1, 1],
            "datetime": [0.25, 0.25],
            "infusionid": [10, 10],
            # value column already renamed to the concept name; no 'givendose'
            "norepi_rate": [60.0, 60.0],
            "doseunit": ["ug", "ug"],
            "weight": [60.0, 60.0],
        }
    )

    result = hirid_rate_kg(
        frame,
        concept_name="norepi_rate",
        val_col="givendose",  # absent in frame -> falls back to concept_name
        unit_col="doseunit",
        grp_var="infusionid",
        index_col="datetime",
        interval_minutes=60.0,
    )

    assert not result.empty
    # doses summed within (patient, hour, infusion) = 120; rate = 120/60min/60kg
    assert result["norepi_rate"].iloc[0] == pytest.approx(120.0 / 60.0 / 60.0)
