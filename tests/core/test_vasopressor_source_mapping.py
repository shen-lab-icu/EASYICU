"""SICdb / eICU vasopressor source mapping: continuous rates only, never guessed units."""

import json
import re
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.utils.callback_utils import eicu_rate_kg_callback, eicu_rate_units_callback


def _dictionary() -> dict:
    return json.loads(resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8"))


def _sic_source() -> SimpleNamespace:
    cases = pd.DataFrame({"CaseID": [17], "ICUOffset": [3600]})
    return SimpleNamespace(config=SimpleNamespace(name="sic"), load_table=lambda *a, **kw: cases)


def _sic_rows(concept: str, values: list, single: list) -> pd.DataFrame:
    # Two continuous one-hour rows and one single-dose push in ICU hour 1.
    starts = [7200 + 3600 * i for i in range(len(values))]
    return pd.DataFrame({"CaseID": 17, "Offset": starts,
                         "OffsetDrugEnd": [s + (60 if one else 3600) for s, one in zip(starts, single)],
                         "IsSingleDose": single, "weight": 80.0, concept: values})


def _dict_source(concept: str) -> ConceptSource:
    return ConceptSource.from_mapping(_dictionary()[concept]["sources"]["sic"][0])


def test_sic_vasopressin_maps_continuous_units_per_minute_without_weight():
    frame = _sic_rows("adh_rate", [0.04, 20.0], [0, 1])
    result = _apply_callback(frame, _dict_source("adh_rate"), concept_name="adh_rate", data_source=_sic_source())
    # R5-SIC-018: an exact stop has no overlap with the following hour.
    assert result["Offset"].tolist() == [1]
    assert result["adh_rate"].tolist() == pytest.approx([0.04])


def test_sic_phenylephrine_keeps_infusions_and_drops_single_dose_pushes():
    frame = _sic_rows("phn_rate", [0.00004, 0.0001], [0, 1])
    result = _apply_callback(frame, _dict_source("phn_rate"), concept_name="phn_rate", data_source=_sic_source())
    assert result["phn_rate"].tolist() == pytest.approx([0.5])  # R5-SIC-018; 40 mcg/min / 80 kg


def test_sic_kg_rate_without_weight_is_dropped_not_left_in_grams():
    frame = _sic_rows("phn_rate", [0.00004], [0]).drop(columns="weight")
    result = _apply_callback(frame, _dict_source("phn_rate"), concept_name="phn_rate", data_source=_sic_source())
    assert result.empty or result["phn_rate"].dropna().empty


def test_sic_declared_single_dose_flag_must_reach_the_callback():
    frame = _sic_rows("adh_rate", [0.04], [0]).drop(columns="IsSingleDose")
    with pytest.raises(ValueError, match="sic_single_dose_flag_missing"):
        _apply_callback(frame, _dict_source("adh_rate"), concept_name="adh_rate", data_source=_sic_source())


@pytest.mark.parametrize("concept", ["norepi_rate", "norepi_dur"])
def test_eicu_norepinephrine_matches_brand_name_but_not_levothyroxine(concept):
    pattern = re.compile(_dictionary()[concept]["sources"]["eicu"][0]["regex"], re.IGNORECASE)
    assert pattern.search("Levophed (mcg/min)") and pattern.search("NOREPINEPHRINE (ml/hr)")
    assert not pattern.search("Levothyroxine (mcg/hr)")
    # A name without a dose unit is exposure (duration) evidence only.
    assert bool(pattern.search("Norepinephrine ()")) is (concept == "norepi_dur")


def test_eicu_kg_rate_drops_unrecognised_units_instead_of_dividing_by_weight():
    frame = pd.DataFrame({"patientunitstayid": [1, 2, 3], "infusionoffset": [0, 0, 0],
                          "drugrate": [5.0, 5.0, 8.0], "patientweight": [80.0, 80.0, 80.0],
                          "drugname": ["Norepinephrine (mcg/kg/day)", "Norepinephrine 4mg/250ml",
                                       "Levophed (mcg/min)"]})
    result = eicu_rate_kg_callback(ml_to_mcg=32)(frame, val_var="drugrate", sub_var="drugname",
                                                 weight_var="patientweight", concept_name="norepi_rate")
    assert result.set_index("patientunitstayid")["norepi_rate"].to_dict() == {3: pytest.approx(0.1)}


def test_eicu_units_rate_leaves_unknown_dose_unit_missing():
    frame = pd.DataFrame({"patientunitstayid": [1, 2], "infusionoffset": [0, 0], "drugrate": [2.4, 0.04],
                          "drugname": ["Vasopressin (Unknown)", "Vasopressin (units/min)"]})
    result = eicu_rate_units_callback(2.65, 0.53)(frame, val_var="drugrate", sub_var="drugname",
                                                  concept_name="adh_rate")
    by_id = result.groupby("patientunitstayid")["drugrate"].max()
    assert np.isnan(by_id.get(1, np.nan)) and by_id.loc[2] == pytest.approx(0.04)


def test_eicu_kg_rate_falls_back_to_the_admission_weight_when_the_row_has_none():
    calls = []

    def load_table(name, columns=None, filters=None, **_):
        calls.append((name, [f.value for f in filters or []]))
        return pd.DataFrame({"patientunitstayid": [1], "admissionweight": [80.0]})

    frame = pd.DataFrame({"patientunitstayid": [1], "infusionoffset": [0], "drugrate": [8.0],
                          "patientweight": [np.nan], "drugname": ["Norepinephrine (mcg/min)"]})
    result = eicu_rate_kg_callback(ml_to_mcg=32)(frame, val_var="drugrate", sub_var="drugname",
                                                 weight_var="patientweight", concept_name="norepi_rate",
                                                 data_source=SimpleNamespace(load_table=load_table))
    assert calls == [("patient", [[1]])]
    assert result["norepi_rate"].tolist() == [pytest.approx(0.1)]


def _sofa2_other_vaso(database: str) -> dict:
    payload = json.loads(resources.files("easyicu.data").joinpath("sofa2-dict.json").read_text(encoding="utf-8"))
    return payload.get("concepts", payload)["other_vaso"]["sources"][database][0]


@pytest.mark.parametrize("concept", ["norepi_rate", "norepi_dur"])
def test_sic_norepinephrine_rate_and_duration_ignore_single_dose_pushes(concept):
    frame = _sic_rows(concept, [0.000008, 0.0004], [0, 1])
    result = _apply_callback(frame, _dict_source(concept), concept_name=concept, data_source=_sic_source())
    if concept == "norepi_rate":
        assert result["norepi_rate"].tolist() == pytest.approx([0.1])  # R5-SIC-018; 8 mcg/min / 80 kg
    else:
        assert result["norepi_dur"].tolist() == pytest.approx([1.0])


def test_sofa2_other_vaso_counts_sic_infusions_of_vasopressin_and_phenylephrine_only():
    mapping = _sofa2_other_vaso("sic")
    assert mapping["ids"] == [1550, 1593, 1560]  # R5-SIC-010: milrinone too.
    frame = pd.DataFrame({"CaseID": 17, "Offset": [7200, 10800], "IsSingleDose": [0, 1],
                          "DrugID": [1593, 1593], "OffsetDrugEnd": [10800, 10860], "other_vaso": [0.0001, 0.0001]})
    result = _apply_callback(frame, ConceptSource.from_mapping(mapping), concept_name="other_vaso",
                             data_source=_sic_source())
    assert result["Offset"].tolist() == [1.] and "IsSingleDose" not in result.columns  # R5-SIC-003: ICU hours.


def test_sofa2_other_vaso_matches_neo_synephrine_in_eicu():
    pattern = re.compile(_sofa2_other_vaso("eicu")["regex"], re.IGNORECASE)
    assert pattern.search("Neo Synephrine (mcg/min)") and pattern.search("NeoSynephrine (ml/hr)")
