"""Structural regression tests for the ``furosemide`` concept.

The concept is a ``lgl_cncpt`` indicating IV bolus or continuous-infusion
furosemide (Lasix) administration, mirroring the pattern of ``cort`` and
``abx``. Item IDs were ground-truthed against each database's local
reference dictionary on 2026-05-12:

* AUMC drugitems.itemid: 7244
* eICU medication.drugname / infusiondrug.drugname: regex 'furosem|lasix'
* HiRID pharma.pharmaid: 482, 1000232, 1000747 (IV/infusion only)
* MIMIC-III inputevents_cv.itemid: 30123 (CareVue)
* MIMIC-III inputevents_mv.itemid: 221794, 228340 (MetaVision)
* MIMIC-IV (miiv) inputevents.itemid: 221794, 228340
* SIC medication.DrugID: 1417

Real-data smoke (50 MIIV patients) loaded 27 events from 13 patients with
all values ``True`` — see ``/tmp/load_furosemide_real.py`` log.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DICT = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "data" / "concept-dict.json"


@pytest.fixture(scope="module")
def furosemide_entry():
    with DICT.open() as f:
        return json.load(f)["furosemide"]


def test_furosemide_is_logical_concept(furosemide_entry):
    assert furosemide_entry["class_name"] == "lgl_cncpt"
    assert furosemide_entry["category"] == "medications"
    assert furosemide_entry["aggregate"] == "max"


def test_furosemide_covers_all_six_databases(furosemide_entry):
    """All 6 supported DBs (+ 2 demo variants) must have a source entry."""
    expected = {
        "aumc",
        "eicu",
        "eicu_demo",
        "hirid",
        "miiv",
        "mimic",
        "mimic_demo",
        "sic",
    }
    assert set(furosemide_entry["sources"].keys()) == expected


def test_aumc_source_uses_itemid_7244(furosemide_entry):
    aumc = furosemide_entry["sources"]["aumc"]
    assert len(aumc) == 1
    assert aumc[0]["table"] == "drugitems"
    assert aumc[0]["sub_var"] == "itemid"
    assert 7244 in aumc[0]["ids"]


def test_miiv_source_uses_inputevents_metavision_itemids(furosemide_entry):
    miiv = furosemide_entry["sources"]["miiv"]
    assert len(miiv) == 1
    assert miiv[0]["table"] == "inputevents"
    assert miiv[0]["sub_var"] == "itemid"
    assert set(miiv[0]["ids"]) == {221794, 228340}


def test_mimic_source_covers_both_carevue_and_metavision(furosemide_entry):
    """MIMIC-III has both CareVue (cv) and MetaVision (mv) inputevents."""
    mimic = furosemide_entry["sources"]["mimic"]
    tables = {src["table"]: src for src in mimic}
    assert "inputevents_cv" in tables
    assert "inputevents_mv" in tables
    assert tables["inputevents_cv"]["ids"] == [30123]
    assert set(tables["inputevents_mv"]["ids"]) == {221794, 228340}


def test_eicu_source_uses_drugname_regex(furosemide_entry):
    """eICU has free-text drug names — match by regex on drugname."""
    eicu = furosemide_entry["sources"]["eicu"]
    tables = {src["table"]: src for src in eicu}
    assert "medication" in tables
    assert "infusiondrug" in tables
    for tbl in ("medication", "infusiondrug"):
        assert tables[tbl]["regex"] == "furosem|lasix"
        assert tables[tbl]["sub_var"] == "drugname"
        assert tables[tbl]["class_name"] == "rgx_itm"


def test_hirid_source_uses_pharma_table_iv_only(furosemide_entry):
    """HiRID pharma IDs include IV bolus (1000747), perfusor (482), and IV
    injection (1000232) — oral tablets (id 4) are intentionally excluded."""
    hirid = furosemide_entry["sources"]["hirid"]
    assert len(hirid) == 1
    assert hirid[0]["table"] == "pharma"
    assert hirid[0]["sub_var"] == "pharmaid"
    assert set(hirid[0]["ids"]) == {482, 1000232, 1000747}
    # Oral tablet (id 4) intentionally excluded
    assert 4 not in hirid[0]["ids"]


def test_sic_source_uses_medication_drugid(furosemide_entry):
    """SIC ``DrugID`` 1417 = FUROsemid (single drug). 1654 (combo with
    spironolactone) intentionally excluded."""
    sic = furosemide_entry["sources"]["sic"]
    assert len(sic) == 1
    assert sic[0]["table"] == "medication"
    assert sic[0]["sub_var"] == "DrugID"
    assert sic[0]["ids"] == [1417]
    # Combo product intentionally excluded
    assert 1654 not in sic[0]["ids"]


def test_all_sources_use_set_val_true_callback(furosemide_entry):
    """Logical concept: every source must emit ``transform_fun(set_val(TRUE))``."""
    for db, sources in furosemide_entry["sources"].items():
        for src in sources:
            assert src["callback"] == "transform_fun(set_val(TRUE))", (
                f"{db}: unexpected callback {src.get('callback')!r}"
            )


def test_furosemide_registered_as_point_event_concept():
    from easyicu.utils.compat import POINT_EVENT_CONCEPTS

    assert "furosemide" in POINT_EVENT_CONCEPTS, (
        "furosemide must be in POINT_EVENT_CONCEPTS so the loader does not "
        "expand its boolean events into a continuous time series."
    )


def test_furosemide_in_webapp_concept_catalog():
    from easyicu.concept.catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )

    assert "furosemide" in CONCEPT_DICTIONARY
    en, zh, unit = CONCEPT_DICTIONARY["furosemide"]
    assert "Furosemide" in en
    assert "呋塞米" in zh
    assert unit == "boolean"
    assert CONCEPT_DB_COVERAGE.get("furosemide") == 6
