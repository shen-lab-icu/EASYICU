"""Placeholders the raw sources can fill (2026-11-03 extraction audit 2, P2).

Each case names the database's own dictionary entry the mapping reads; the
behaviour cases run the shipped source through ``_apply_callback``.
"""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback


def _dictionary(name: str = "concept-dict.json") -> dict:
    return json.loads(resources.files("easyicu.data").joinpath(name).read_text(encoding="utf-8"))


def _ids(concept: str, database: str, index: int = 0):
    ids = _dictionary()[concept]["sources"][database][index]["ids"]
    return set(ids) if isinstance(ids, list) else {ids}


@pytest.mark.parametrize(
    ("concept", "database", "ids"),
    [
        ("icp", "mimic", {226, 220765, 227989}),  # MetaVision Intra Cranial Pressure (#2)
        ("ecmo", "mimic", {228193, 224660, 2957, 3265}),
        ("ecmo", "sic", {2023, 2024}),  # ECMO PumpSpeed / BloodFlow
        ("vent_start", "miiv", {225792, 224385}),  # Invasive Ventilation / Intubation
        ("cort", "sic", {1525, 1506, 1397, 1524, 1751}),
        ("hbco", "sic", {660}),
        ("methb", "sic", {661}),
        ("ntprobnp", "sic", {571}),
        ("prealbumin", "sic", {382}),
        ("ffp", "sic", {2047}),  # Octaplas
        ("propofol_rate", "sic", {3057}),  # PropofolPerHourWeight, g/h/kg
        ("propofol_rate", "hirid", {208, 1001050}),  # Disoprivan 2% infusions
        ("svo2", "eicu", {"SVO2"}),
    ],
)
def test_placeholder_reads_the_documented_items(concept, database, ids):
    assert _ids(concept, database) == ids


def test_ecmo_reaches_sofa2_in_the_same_databases():
    plain = _dictionary()["ecmo"]["sources"]
    sofa2 = _dictionary("sofa2-dict.json")["ecmo"]["sources"]
    for database in ("mimic", "sic"):
        assert sofa2[database] == plain[database]


@pytest.mark.parametrize(
    ("database", "column"),
    [("miiv", "first_careunit"), ("mimic", "first_careunit"), ("sic", "HospitalUnit")],
)
def test_icu_unit_type_reads_the_stay_table(database, column):
    source = _dictionary()["icu_unit_type"]["sources"][database][0]
    assert source["val_var"] == column and source["class_name"] == "col_itm"


def test_sic_hospital_unit_codes_get_their_reference_names():
    source = ConceptSource.from_mapping(_dictionary()["icu_unit_type"]["sources"]["sic"][0])
    frame = pd.DataFrame({"CaseID": [1, 2, 3, 4], "icu_unit_type": [2, 3, 4, 5]})

    out = _apply_callback(frame, source, concept_name="icu_unit_type")

    assert out["icu_unit_type"].tolist() == [
        "intermediate care ward (INIC)",
        "intensive care ward (CWIN)",
        "intensive care ward (INBD)",
        "mixed intermediate care ward with NIV (INID)",
    ]


def test_sic_ecmo_is_a_running_pump():
    source = ConceptSource.from_mapping(_dictionary()["ecmo"]["sources"]["sic"][0])
    frame = pd.DataFrame({"CaseID": [1, 1], "Offset": [0, 3600], "ecmo": [3000.0, 0.0]})

    out = _apply_callback(frame, source, concept_name="ecmo")

    assert out["ecmo"].tolist() == [True, False]


def test_sic_propofol_rate_converts_g_per_hour_per_kg():
    source = ConceptSource.from_mapping(_dictionary()["propofol_rate"]["sources"]["sic"][0])
    frame = pd.DataFrame({"CaseID": [1], "Offset": [0], "propofol_rate": [0.0018]})

    out = _apply_callback(frame, source, concept_name="propofol_rate")

    assert out["propofol_rate"].iloc[0] == pytest.approx(30.0, rel=1e-6)  # mcg/kg/min


def test_eicu_cort_reads_hicl_only_rows():
    sources = _dictionary()["cort"]["sources"]["eicu"]
    hicl = sources[1]
    assert hicl["sub_var"] == "drughiclseqno"
    assert set(hicl["ids"]) == {2876, 36808, 2866, 36875, 2879, 2888, 2889}
