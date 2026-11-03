"""AUMCdb/HiRID dictionary corrections from the 2026-11-03 extraction audit (batch 2).

Each mapping was confirmed against the database's own item dictionary
(``aumc_item_dict.parquet`` labels/units, raw listitem answer lists, HiRID
``hirid_variable_reference``) and against same-sample/same-hour comparisons on
the raw tables (aggregates only).  The synthetic sources below run the shipped
dictionary through the real loader.
"""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd
import pytest

from easyicu.concept.blood_gas_specimen import (
    ArterialSpecimenContractError,
    parse_mixed_venous_specimen_callback,
    restrict_to_mixed_venous_specimens,
)
from easyicu.concept.catalog import CONCEPT_DICTIONARY, HIDDEN_DICTIONARY_CONCEPTS
from tests.support.blood_gas_sources import _source, _write, hourly

CONCEPTS = json.loads(
    resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8")
)
MINUTE = 60_000


def _ids(concept: str, database: str = "aumc") -> set[int]:
    found: set[int] = set()
    for source in CONCEPTS[concept]["sources"].get(database, []):
        ids = source.get("ids", [])
        found.update(ids if isinstance(ids, list) else [ids])
    return found


@pytest.mark.parametrize(
    ("concept", "absent", "present"),
    [
        ("temp", {13063, 8662, 8659, 11889}, {8658, 13058, 13952, 16110, 13060, 13062, 13059, 13061, 14047}),
        ("egcs", {19635, 19638}, {6732, 13077, 16628}),  # NICE: registry values
        ("vgcs", {19637, 19640}, {6735, 13066, 16640}),
        ("cl", set(), {9930, 14413}),
        ("ldh", set(), {6838, 12023}),
        ("eos", set(), {6773, 9967, 11790}),
        ("monos", set(), {11850, 14260}),
        ("bili_dir", {6812}, {18361}),  # 6812 is not umol/l
        ("tidal_vol_set", set(), {12291, 8851}),
        ("svo2", set(), {12534, 12311}),
        ("icp", {19358}, {8835, 13940}),
        ("iron", set(), {10054}),
        ("tibc", set(), {10055}),
        ("transferrin", set(), {9981}),
        ("phenytoin", set(), {7222}),
        ("phn_rate", set(), {19929}),
        ("anion_gap", set(), {9559}),
    ],
)
def test_aumc_mapping_names_the_quantity_of_its_concept(concept, absent, present):
    ids = _ids(concept)
    assert not ids & absent
    assert present <= ids


def test_aumc_prothrombin_item_is_inr_not_seconds():
    # 6789 'Protrombinetijd [sec]': 98% of values < 5 and equal to INR 11893/11894
    # in the same hour, so pt (seconds) stays unavailable on AUMC.
    assert "aumc" not in CONCEPTS["pt"]["sources"]


def test_monocyte_absolute_count_is_converted_to_percent():
    (source,) = [s for s in CONCEPTS["monos"]["sources"]["aumc"] if s["ids"] == 14260]
    assert source["callback"] == "blood_cell_ratio"


def test_device_oxygen_flow_feeds_supplemental_oxygen_on_aumc_and_hirid():
    assert _ids("o2_flow") == {8845}
    assert _ids("o2_flow", "hirid") == {15001525, 15001555}
    for database in ("aumc", "hirid"):
        (source,) = CONCEPTS["supp_o2"]["sources"][database]
        assert source["concepts"] == ["vent_ind", "fio2", "o2_flow"]
    # Audit round 2 (G6): published in the respiratory module, so native-v2
    # publication can verify flow-backed supp_o2 claims.
    assert "o2_flow" not in HIDDEN_DICTIONARY_CONCEPTS and "o2_flow" in CONCEPT_DICTIONARY


def test_mixed_venous_callback_parses_and_requires_a_rule():
    assert parse_mixed_venous_specimen_callback(
        "mixed_venous_specimen_only(transform_fun(binary_op(`*`, 100)))"
    ) == (True, "transform_fun(binary_op(`*`, 100))")
    assert parse_mixed_venous_specimen_callback("arterial_specimen_only") == (False, None)

    class Miiv:
        config = type("C", (), {"name": "miiv"})()

    with pytest.raises(ArterialSpecimenContractError, match="mixed_venous_specimen_rule_missing"):
        restrict_to_mixed_venous_specimens(pd.DataFrame({"x": [1]}), data_source=Miiv(), concept_name="svo2")


def _aumc(tmp_path):
    numeric = pd.DataFrame(
        [
            # O2 saturation (fraction) with specimens below; 12534 SvO2 Vigilance.
            (5, 12311, 0.98, "Geen"), (65, 12311, 0.66, "Geen"), (125, 12311, 0.74, "Geen"),
            (185, 12311, 0.97, "Geen"), (245, 12311, 0.70, "Geen"), (305, 12311, 0.64, "Geen"),
            # Set tidal volume: litres; 2.0 L is the Evita knob maximum.
            (5, 8851, 0.45, "ml"), (65, 8851, 2.0, "ml"), (125, 12291, 480.0, "ml"),
            # Analyser anion gap.
            (5, 9559, 9.0, "mmol/l"),
        ],
        columns=["minute", "itemid", "value", "unit"],
    )
    numeric["measuredat"] = numeric.pop("minute") * MINUTE
    numeric["admissionid"] = 1
    numeric["tag"] = None
    numeric["registeredat"] = numeric["measuredat"]
    _write(numeric, tmp_path / "numericitems" / "part.parquet")
    freetext = pd.DataFrame(
        [(5, "ART."), (65, "GEM.VEN."), (125, "Ven."), (245, "Gem. ven.")],
        columns=["minute", "value"],
    )
    freetext["measuredat"] = freetext.pop("minute") * MINUTE
    freetext["admissionid"] = 1
    freetext["itemid"] = 11646
    freetext["item"] = "Afname (bloed)"
    _write(freetext, tmp_path / "freetextitems.parquet")
    # 2003-2008 era site record (list item 7432 "Afnameplaats").
    _write(
        pd.DataFrame({
            "admissionid": [1], "itemid": [7432], "item": ["Afnameplaats"], "value": ["PAP"],
            "valueid": [2], "measuredat": [305 * MINUTE],
        }),
        tmp_path / "listitems.parquet",
    )
    _write(
        pd.DataFrame({
            "admissionid": [1, 9], "itemid": [9328, 9328], "item": ["Beademen"] * 2,
            "start": [60 * MINUTE, 0], "stop": [600 * MINUTE, 60 * MINUTE], "duration": [540, 60],
        }),
        tmp_path / "processitems.parquet",
    )
    _write(
        pd.DataFrame({
            "admissionid": [1], "patientid": [11], "admittedat": [0],
            "dischargedat": [48 * 60 * MINUTE], "admissioncount": [1], "location": ["MC"],
            "dateofdeath": pd.array([None], dtype="Int64"),
        }),
        tmp_path / "admissions.parquet",
    )
    return _source(tmp_path, "aumc"), "admissionid"


def test_svo2_keeps_only_recorded_mixed_venous_blood_gases(tmp_path):
    source, id_column = _aumc(tmp_path)
    # ART. (98), plain venous Ven. (74) and the unrecorded gas (97) are not SvO2;
    # the old-era list item 7432 "PAP" (pulmonary artery, 64) is.
    assert hourly(source, id_column, "svo2") == {1.0: 66.0, 4.0: 70.0, 5.0: 64.0}


def test_set_tidal_volume_restores_litre_item_without_its_parked_maximum(tmp_path):
    source, id_column = _aumc(tmp_path)
    assert hourly(source, id_column, "tidal_vol_set") == {0.0: 450.0, 2.0: 480.0}


def test_aumc_anion_gap_is_the_analyser_value(tmp_path):
    source, id_column = _aumc(tmp_path)
    assert hourly(source, id_column, "anion_gap") == {0.0: 9.0}


def test_ventilation_start_and_end_come_from_the_beademen_episode(tmp_path):
    source, id_column = _aumc(tmp_path)
    assert hourly(source, id_column, "vent_start") == {1.0: 1.0}
    assert hourly(source, id_column, "vent_end") == {10.0: 1.0}


def test_aumc_survivor_death_row_survives_the_hourly_grid(tmp_path):
    source, id_column = _aumc(tmp_path)
    # No recorded death: False at ICU discharge (48 h), not a dropped row.
    assert hourly(source, id_column, "death") == {48.0: 0.0}


def test_aumc_icu_unit_type_reads_admission_location():
    assert CONCEPTS["icu_unit_type"]["sources"]["aumc"] == [
        {
            "table": "admissions",
            "val_var": "location",
            "class_name": "col_itm",
            "_comment": CONCEPTS["icu_unit_type"]["sources"]["aumc"][0]["_comment"],
        }
    ]
