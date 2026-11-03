"""Dictionary corrections from the 2026-11-03 six-database extraction audit.

Each mapping below was confirmed against the database's own item dictionary
(MIMIC d_items/d_labitems, SICdb d_references, AUMC item labels) before it
was changed: a label or unit that names a different quantity, a spontaneous
channel inside a total, or a value that is not the event the concept claims.
"""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.callbacks import apply_vent_mode_frame


def _dictionary(name: str) -> dict:
    payload = json.loads(resources.files("easyicu.data").joinpath(name).read_text(encoding="utf-8"))
    return payload.get("concepts", payload)


CONCEPTS = _dictionary("concept-dict.json")
SOFA2 = _dictionary("sofa2-dict.json")


def _ids(concept: str, database: str, dictionary: dict = CONCEPTS) -> set[int]:
    found: set[int] = set()
    for source in dictionary[concept]["sources"].get(database, []):
        ids = source.get("ids", [])
        found.update(ids if isinstance(ids, list) else [ids])
    return found


@pytest.mark.parametrize(
    ("concept", "database", "absent", "present"),
    [
        ("dopa_rate", "mimic", {30125}, {30043, 30307}),  # 30125 is Milrinone
        ("dopa_dur", "mimic", {30125}, {30043, 30307}),
        ("milrinone", "mimic", set(), {30125}),
        ("urine", "mimic", {40086, 40065}, {40651}),  # drain; OR/PACU urine
        ("na", "sic", {686}, {469}),  # BGA sodium with a validity note
        ("bicar", "sic", {667}, {451, 456, 666}),  # standard bicarbonate
        ("hr", "sic", set(), {707, 708}),
        ("resp", "sic", set(), {719, 2282, 2274}),
        ("vent_mode", "mimic", set(), {720, 223849}),
        ("vent_breath_seq", "mimic", set(), {720, 223849}),
        ("ckmb", "aumc", {6824}, {10413}),  # activity vs mass
        ("peep_total", "mimic", set(), {686, 224700}),
        # 2026-11-03 audit 3: peep is the ventilator setting, as peep_set.
        ("peep", "miiv", {224700}, {220339}),
        ("peep", "mimic", {224700, 686}, {505, 506, 220339}),
        ("peep", "aumc", {6694, 8879, 9666, 12354}, {12284, 12336, 8862}),
        ("peep", "hirid", {2610}, {2600}),
        ("resp", "miiv", {224689}, {220210, 224690}),
        ("resp", "aumc", {7726, 9654, 12577}, {8874, 12266}),
        ("resp", "mimic", {614, 653, 1884, 224689}, {618, 615, 220210, 224690}),
        ("tidal_vol", "miiv", {224421, 224686}, {224685}),
        ("tidal_vol", "mimic", {502, 652, 684, 654, 224421, 224686}, {681, 682, 224685}),
        ("phn_rate", "miiv", set(), {221749, 229630, 229632}),
    ],
)
def test_a_mapping_names_the_quantity_of_its_concept(concept, database, absent, present):
    ids = _ids(concept, database)
    assert not ids & absent
    assert present <= ids


def test_aumc_eye_drops_are_not_a_vasopressor_and_isoprenaline_is():
    ids = _ids("other_vaso", "aumc", SOFA2)
    assert 9126 not in ids  # 'Niet iv Zalven/Crèmes/Druppels'
    assert {7135, 19929} <= ids


def test_sicdb_has_no_prothrombin_time_in_seconds():
    # 598 is 'Thrombinzeit' (thrombin time); SICdb has no PT in seconds.
    assert "sic" not in CONCEPTS["pt"]["sources"]


def test_vent_end_keeps_only_oxygen_therapy_devices():
    (chart,) = [s for s in CONCEPTS["vent_end"]["sources"]["mimic"] if s["table"] == "chartevents"]
    assert chart["ids"] == [467]
    assert chart["callback"] == "mimiciii_oxygen_device_event"

    frame = pd.DataFrame(
        {
            "icustay_id": [1, 1, 1, 2],
            "charttime": pd.to_datetime(["2180-01-01 00:00", "2180-01-01 05:00", "2180-01-01 06:00", "2180-01-01 00:00"]),
            "itemid": [467, 467, 467, 467],
            "vent_end": ["Ventilator", "Nasal Cannula", None, "Trach Mask"],
        }
    )
    result = _apply_callback(
        frame,
        ConceptSource(callback="mimiciii_oxygen_device_event", sub_var="itemid", value_var="value"),
        concept_name="vent_end",
    )
    assert result["icustay_id"].tolist() == [1, 2]
    assert result["charttime"].tolist() == [pd.Timestamp("2180-01-01 05:00"), pd.Timestamp("2180-01-01 00:00")]
    assert result["vent_end"].tolist() == [True, True]



def test_mimic_iii_metavision_modes_use_the_mimic_iv_strings():
    frame = pd.DataFrame(
        {
            "icustay_id": [1, 1, 1],
            "charttime": pd.to_datetime(["2180-01-01 00:00", "2180-01-01 01:00", "2180-01-01 02:00"]),
            "itemid": [223849, 223849, 720],
            "value": ["CMV/ASSIST/AutoFlow", "Standby", "Pressure Support"],
        }
    )
    control = apply_vent_mode_frame(frame, "value", "mimic", "control", "vent_mode")
    seq = apply_vent_mode_frame(frame, "value", "mimic_demo", "seq", "vent_breath_seq")
    assert control["vent_mode"].tolist() == ["dual_adaptive", "pressure"]
    assert seq["vent_breath_seq"].tolist() == ["assisted", "spontaneous"]


@pytest.mark.parametrize("database", ["aumc", "miiv", "mimic", "mimic_demo", "sic"])
def test_o2sat_lab_saturation_follows_the_arterial_specimen_rule_of_sao2(database):
    def lab(concept):
        return {
            tuple(s["ids"]) if isinstance(s["ids"], list) else (s["ids"],): (s.get("callback"), s.get("extra_vars"))
            for s in CONCEPTS[concept]["sources"][database]
            if "arterial_specimen_only" in (s.get("callback") or "")
        }

    assert lab("sao2") and lab("o2sat") == lab("sao2")


@pytest.mark.parametrize(
    "concept,database,callback",
    [
        ("be", "miiv", "arterial_specimen_only"),
        ("be", "mimic", "arterial_specimen_only"),
        ("be", "mimic_demo", "arterial_specimen_only"),
        ("be", "aumc", "arterial_specimen_only(aumc_bxs)"),
        ("bicar", "aumc", "arterial_specimen_only"),
    ],
)
def test_blood_gas_base_excess_and_bicarbonate_are_arterial_only(concept, database, callback):
    # 2026-11-03 audit 3: as po2/pco2/ph/sao2. MIMIC bicar is serum 50882.
    assert [source["callback"] for source in CONCEPTS[concept]["sources"][database]] == [callback]
    assert all("callback" not in source for source in CONCEPTS["bicar"]["sources"]["miiv"])


def test_mimic_iv_d_dimer_feu_rows_are_converted_to_ddu():
    source = ConceptSource.from_mapping(CONCEPTS["d_dimer"]["sources"]["miiv"][0])
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": pd.to_datetime(["2150-01-01 08:00"] * 3),
            "d_dimer": [500.0, 1000.0, 800.0],
            "valueuom": ["ng/mL", "ng/mL FEU", "ng/mL FEU"],
        }
    )

    out = _apply_callback(frame, source, concept_name="d_dimer")

    assert out["d_dimer"].tolist() == [500.0, 500.0, 400.0]
