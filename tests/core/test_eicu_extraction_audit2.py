"""eICU extraction audit 2 (2026-11-03): urine, RASS, vitals, phenytoin, vasopressin.

Synthetic raw eICU tables go through the shipped dictionary and the real
loader.  Each stay mixes the values a concept must keep with near-miss rows
(other sedation scales, pre-admission medication lists, non-urine outputs,
volume-charted vasopressin) that the old mappings returned or missed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.eicu_sedation_scale import scale_status
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.utils.callback_utils import eicu_rate_units_callback

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
DATABASES = pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
OUTPUT = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"


@pytest.fixture(scope="module")
def dictionary():
    return ConceptDictionary.from_json(DICTIONARY)


def _sources(dictionary, concept, database):
    return dictionary[concept].sources[database]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _eicu_source(tmp_path: Path, database: str, **tables) -> ICUDataSource:
    _write(pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [11], "uniquepid": ["p1"],
        "age": ["60"], "gender": ["Male"], "hospitalid": [1],
        "hospitaladmitoffset": [0], "hospitaldischargeoffset": [72 * 60],
        "unitdischargeoffset": [48 * 60], "unitdischargestatus": ["Alive"],
        "hospitaldischargestatus": ["Alive"], "admissionweight": [80.0],
    }), tmp_path / "patient.parquet")
    for name, frame in tables.items():
        frame = frame.copy()
        frame["patientunitstayid"] = 1
        directory = name in {"nursecharting", "vitalperiodic"}
        _write(frame, tmp_path / name / "part.parquet" if directory else tmp_path / f"{name}.parquet")
    return ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)


def _hourly(source, concept, aggregate="median") -> dict:
    loaded = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        [concept], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": [1]},
        aggregate=aggregate, concept_workers=1, verbose=False,
    )[concept]
    frame = loaded.data
    time = frame[loaded.index_column]
    if pd.api.types.is_timedelta64_dtype(time):
        time = time / pd.Timedelta(hours=1)
    return {
        float(hour): (value if isinstance(value, (bool, np.bool_)) else round(float(value), 3))
        for hour, value in zip(time, frame[concept]) if pd.notna(value)
    }


# --- urine -------------------------------------------------------------------

URINE = [
    "Indwelling Catheter Output", "Voided Amount", "SN Urine Output(ml)",
    "Output (mL)-Straight Catheter", "Output (mL)-[REMOVED] Straight Catheter",
    "NEPHROSTOMY OUTPUT", "Output (ml)-Nephrostomy Left", "R Nephrostomy",
    "Urostomy", "Output (mL)-Ostomy-GU Urostomy", "Ileal Conduit", "URETERAL STENT",
    "Output (mL)-Suprapubic Catheter", "Texas Cath",
    "Urine", "URINE CATHETER", "Output (mL)-Urethral Catheter 16 Fr.",
]
NOT_URINE = [
    "incontinent urine", "Urine Count", "Urine Occurrence", "Urinary Incontinence",
    "unmeasured voids", "Number of Incontinent Voids", "OR urine",
    "Mixed Urine/Stool Volume", "CBI OUTPUT", "CBI/Bladder Irr Out", "Stool", "Emesis",
    "Ileostomy", "Chest Tube", "EVD", "CRRT - UF removed", "Actual Patient Fluid Removal",
    "Hemofiltration", "TOTAL OUT", "Gastric (NG)", "Foley PACU",
]


@DATABASES
def test_urine_cellpaths_cover_every_urine_output_and_no_counts_or_irrigation(dictionary, database):
    # The loader matches source regexes with re.IGNORECASE (concept/__init__.py).
    (source,) = _sources(dictionary, "urine", database)
    pattern = re.compile(source.regex, re.IGNORECASE)
    assert [label for label in URINE if not pattern.search(OUTPUT + label)] == []
    assert [label for label in NOT_URINE if pattern.search(OUTPUT + label)] == []
    assert not pattern.search("flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|Crystalloids (ml)|CBI/Bladder Irr In")


def test_urine_loads_indwelling_and_voided_but_not_counts_or_cbi(tmp_path):
    rows = [(10, "Indwelling Catheter Output", 80.0), (70, "Voided Amount", 300.0),
            (75, "incontinent urine", 1.0), (130, "CBI OUTPUT", 1800.0), (135, "Urine Count", 1.0),
            (190, "Urine", 120.0)]
    frame = pd.DataFrame(rows, columns=["intakeoutputoffset", "celllabel", "cellvaluenumeric"])
    frame["cellpath"] = OUTPUT + frame["celllabel"]
    frame["intakeoutputentryoffset"] = frame["intakeoutputoffset"]
    frame["intakeoutputid"] = range(len(frame))
    source = _eicu_source(tmp_path, "eicu", intakeoutput=frame)
    assert _hourly(source, "urine", aggregate="sum") == {0.0: 80.0, 1.0: 300.0, 3.0: 120.0}


# --- RASS --------------------------------------------------------------------

def _nursecharting(rows):
    frame = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue"])
    frame["nursingchartentryoffset"] = frame["nursingchartoffset"]
    frame["nursingchartid"] = np.arange(len(frame), dtype=float)
    return frame


def _sedation(offset, scale, score):
    pair = "Sedation Scale/Score/Goal"
    return [(offset, "Scores", pair, "Sedation Scale", scale), (offset, "Scores", pair, "Sedation Score", score)]


@DATABASES
def test_rass_keeps_rass_scale_scores_and_the_rass_labels(tmp_path, database):
    rows = [
        *_sedation(5, "RASS", "-2"),
        *_sedation(65, "SAS", "6"),      # SAS 6 (very agitated) is not RASS +6
        *_sedation(125, "Ramsay", "4"),  # Ramsay 4 (asleep, brisk) is not RASS +4
        *_sedation(185, "MAAS", "3"),
        (245, "Scores", "Sedation Scale/Score/Goal", "Sedation Score", "1"),  # no scale row
        (305, "Other Vital Signs and Infusions", "RASS", "Value", "-4"),
        (365, "Other Vital Signs and Infusions", "SEDATION SCORE", "Value", "0"),
        (425, "Other Vital Signs and Infusions", "Pain Score", "Value", "3"),
    ]
    source = _eicu_source(tmp_path, database, nursecharting=_nursecharting(rows))
    assert _hourly(source, "rass") == {0.0: -2.0, 5.0: -4.0, 6.0: 0.0}


def test_conflicting_scales_at_one_offset_are_not_rass():
    scales = pd.DataFrame({
        "patientunitstayid": [1, 1, 1, 2],
        "nursingchartoffset": [5, 5, 65, 5],
        "nursingchartvalue": ["RASS", "SAS", " rass ", None],
    })
    status = scale_status(scales).set_index(["patientunitstayid", "nursingchartoffset"])["_rass"]
    assert status.to_dict() == {(1, 5): False, (1, 65): True, (2, 5): False}


# --- vitals ------------------------------------------------------------------

@DATABASES
def test_sbp_dbp_pool_cuff_pressure_and_temp_adds_nurse_charted_celsius(tmp_path, database):
    periodic = pd.DataFrame({
        "observationoffset": [5], "systemicsystolic": [130.0], "systemicdiastolic": [70.0],
        "systemicmean": [90.0], "temperature": [np.nan], "vitalperiodicid": [1.0],
    })
    aperiodic = pd.DataFrame({
        "observationoffset": [65, 125], "noninvasivesystolic": [110, 100],
        "noninvasivediastolic": [60, 50], "noninvasivemean": [77, 67], "vitalaperiodicid": [1, 2],
    })
    nurse = _nursecharting([
        (185, "Vital Signs", "Temperature", "Temperature (C)", "38.2"),
        (185, "Vital Signs", "Temperature", "Temperature (F)", "100.8"),
        (185, "Vital Signs", "Temperature", "Temperature Location", "oral"),
    ])
    source = _eicu_source(tmp_path, database, vitalperiodic=periodic, vitalaperiodic=aperiodic,
                          nursecharting=nurse)
    assert _hourly(source, "sbp") == {0.0: 130.0, 1.0: 110.0, 2.0: 100.0}
    assert _hourly(source, "dbp") == {0.0: 70.0, 1.0: 60.0, 2.0: 50.0}
    assert _hourly(source, "temp") == {3.0: 38.2}


# --- phenytoin ---------------------------------------------------------------

@DATABASES
def test_phenytoin_is_an_in_icu_medication_order_not_a_home_medication(tmp_path, database):
    medication = pd.DataFrame({
        "medicationid": [1, 2, 3], "drugorderoffset": [120, 300, 400],
        "drugstartoffset": [125, 305, 405], "drugstopoffset": [600, 900, 900],
        "drugname": [None, None, "Levetiracetam 500 mg"], "drughiclseqno": [1879, 11679, 8150],
        "dosage": ["100 mg", "1000 MG", "500 mg"], "routeadmin": ["PO", "IV", "PO"],
        "drugordercancelled": ["No", "No", "No"], "drugivadmixture": ["No", "No", "No"],
        "frequency": ["", "", ""], "loadingdose": ["", "", ""], "prn": ["No", "No", "No"], "gtc": [0, 0, 0],
    })
    home = pd.DataFrame({
        "admissiondrugid": [1], "drugoffset": [-600], "drugenteredoffset": [10],
        "drugname": ["DILANTIN"], "drughiclseqno": [1877], "drugdosage": [100.0], "drugunit": ["mg"],
        "drugnotetype": ["Admission"], "specialtytype": [""], "usertype": [""], "rxincluded": ["Yes"],
        "writtenineicu": ["No"], "drugadmitfrequency": [""],
    })
    source = _eicu_source(tmp_path, database, medication=medication, admissiondrug=home)
    assert _hourly(source, "phenytoin", aggregate="max") == {2.0: True, 5.0: True}


# --- vasopressin ---------------------------------------------------------------

@DATABASES
def test_vasopressin_rates_come_from_units_charted_rows_only(dictionary, database):
    (source,) = _sources(dictionary, "adh_rate", database)
    assert source.callback == "eicu_rate_units(NA, NA)"
    frame = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4, 5], "infusionoffset": [0] * 5,
        "drugrate": [4.0, 2.4, 0.04, 0.04, 20.0],
        "drugname": ["Vasopressin (ml/hr)", "Vasopressin (units/hr)", "Vasopressin (units/min)",
                     "Vasopressin (mg/min)", "Vasopressin (mcg/min)"],
    })
    result = eicu_rate_units_callback(None, None)(frame, val_var="drugrate", sub_var="drugname",
                                                  concept_name="adh_rate")
    by_stay = result.groupby("patientunitstayid")["drugrate"].max()
    assert by_stay.get(2) == pytest.approx(0.04) and by_stay.get(3) == pytest.approx(0.04)
    assert all(np.isnan(by_stay.get(stay, np.nan)) for stay in (1, 4, 5))


def test_rate_units_parser_reads_na_as_no_conversion(dictionary, tmp_path):
    from easyicu.concept import _apply_callback

    (source,) = _sources(dictionary, "adh_rate", "eicu")
    frame = pd.DataFrame({
        "patientunitstayid": [1, 2], "infusionoffset": [0, 0], "adh_rate": [6.0, 0.03],
        "drugname": ["Vasopressin (ml/hr)", "Vasopressin (units/min)"],
    })
    result = _apply_callback(frame, source, "adh_rate")
    by_stay = result.groupby("patientunitstayid")["adh_rate"].max()
    assert np.isnan(by_stay.get(1, np.nan)) and by_stay.get(2) == pytest.approx(0.03)
