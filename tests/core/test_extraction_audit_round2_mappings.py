"""Dictionary corrections from extraction audit round 2 (2026-11-03).

Each case names the audit item.  Packaged dictionary and synthetic frames only.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept.callback_apply import _drop_excluded_statuses

DICTIONARY = json.loads(
    (Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text()
)


def _sources(concept: str, database: str) -> list[dict]:
    return DICTIONARY[concept]["sources"][database]


def _ids(concept: str, database: str) -> set:
    found = set()
    for source in _sources(concept, database):
        ids = source.get("ids", [])
        found |= set(ids if isinstance(ids, list) else [ids])
    return found


@pytest.mark.parametrize("concept", ["norepi_rate", "norepi_dur"])
def test_v4_hirid_noradrenaline_bolus_is_no_infusion(concept):
    assert 1000658 not in _ids(concept, "hirid")


@pytest.mark.parametrize(("concept", "item"), [("norepi_rate", 30047), ("epi_rate", 30044)])
def test_v9_carevue_mcg_per_min_rates_are_divided_by_weight(concept, item):
    [source] = [s for s in _sources(concept, "mimic") if item in (s["ids"] if isinstance(s["ids"], list) else [s["ids"]])]
    assert source["callback"] == "combine_callbacks(mimic_kg_rate, mimic_rate_cv)"


@pytest.mark.parametrize(
    "concept",
    ["norepi_rate", "epi_rate", "dopa_rate", "dobu_rate", "adh_rate", "phn_rate",
     "propofol_rate", "midazolam_rate", "fentanyl_rate", "total_input_ml"],
)
def test_v8_f2_metavision_rate_and_volume_drop_rewritten_orders(concept):
    [source] = [s for s in _sources(concept, "mimic") if s["table"] == "inputevents_mv"]
    assert source["excluded_statuses"] == ["Rewritten"]
    assert "statusdescription" in source["extra_vars"]


def test_excluded_statuses_drop_only_the_named_rows():
    frame = pd.DataFrame({"rate": [1.0, 2.0, 3.0],
                          "statusdescription": ["FinishedRunning", " rewritten", None]})
    source = types.SimpleNamespace(params={"excluded_statuses": ["Rewritten"],
                                           "status_var": "statusdescription"})

    assert _drop_excluded_statuses(frame, source)["rate"].tolist() == [1.0, 3.0]
    assert _drop_excluded_statuses(frame, types.SimpleNamespace(params={})).equals(frame)


def test_g4_aumc_paco2_converts_labelled_kpa():
    [source] = _sources("pco2", "aumc")
    assert 21213 in source["ids"]
    assert "binary_op(`*`, 7.50061683), 'mmHg', '^kpa$'" in source["callback"]


@pytest.mark.parametrize("concept", ["spo2", "o2sat"])
def test_g3_aumc_blood_gas_saturation_is_not_pulse_oximetry(concept):
    assert 8903 not in _ids(concept, "aumc")
    assert 6709 in _ids(concept, "aumc")


def test_g9_sic_bicarb_drops_667_like_bicar():
    assert 667 not in _ids("bicarb", "sic")
    assert 667 not in _ids("bicar", "sic")


def test_p5b_aumc_set_tidal_volume_keeps_100_to_1500_ml():
    [source] = [s for s in _sources("tidal_vol_set", "aumc") if s["ids"] == [8851]]
    assert "na_above(1500)" in source["callback"]
    assert "na_below(100)" in source["callback"]


@pytest.mark.parametrize(
    ("concept", "hicls"),
    [("thiazide_diuretic", {3649, 3663, 3646, 3662, 3665, 3647, 131, 9863}),
     ("loop_diuretic", {3660, 3664, 8829}),
     ("acetazolamide", {3641})],
)
def test_d2_eicu_hicl_only_diuretic_orders_are_read(concept, hicls):
    assert hicls <= _ids(concept, "eicu")


def test_h1_sic_cvp_drops_hours_above_30_mmhg():
    [source] = _sources("cvp", "sic")
    assert source["callback"] == "transform_fun(na_above(30))"


@pytest.mark.parametrize("concept", ["map", "map_nibp", "map_invasive", "sbp", "dbp"])
def test_h2_aumc_pressures_at_or_below_zero_are_missing(concept):
    assert all(s["callback"] == "transform_fun(na_below(1))" for s in _sources(concept, "aumc"))


def test_h3_aumc_high_frequency_cvp_drops_non_positive_values():
    by_ids = {tuple(s["ids"]): s for s in _sources("cvp", "aumc")}
    assert by_ids[(20926,)]["callback"] == "transform_fun(na_below(1))"
    assert "callback" not in by_ids[(6655,)]


def test_g1_sicdb_keeps_untyped_gases_of_cases_admitted_in_2013():
    from easyicu.concept.blood_gas_specimen import SPECIMEN_RULES, _restrict_by_column

    class Cases:
        def load_table(self, name, columns=None, verbose=False):
            assert name == "cases"
            return types.SimpleNamespace(data=pd.DataFrame(
                {"CaseID": [1, 2], "AdmissionYear": [2013, 2018]}))

    frame = pd.DataFrame({"CaseID": [1, 1, 2, 2],
                          "LaboratoryType": [None, 2296, None, 2297],
                          "po2": [82.0, 90.0, 35.0, 40.0]})
    kept, audit = _restrict_by_column(frame, SPECIMEN_RULES["sic"], Cases())

    # 2013: the type was not yet recorded, so untyped gases are arterial.
    # Later untyped (median 33-41 mmHg) and typed non-arterial rows drop.
    assert kept["po2"].tolist() == [82.0, 90.0]
    assert audit["unrecorded_rows_kept"] == 1


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_k1_carevue_procedure_events_are_rrt_in_the_loaded_overlay(database):
    import re

    from easyicu.resources import load_dictionary

    # The loader reads rrt from the sofa2 overlay, so the source must be there.
    sources = load_dictionary(include_sofa2=True)["rrt"].sources[database]
    [source] = [s for s in sources if s.ids == [582]]
    assert source.value_var == "value"
    pattern = re.compile(source.regex, re.IGNORECASE)
    for event in ("Hemodialysis st", "Hemodialysis end", "CVVHD Start",
                  "CVVHD D/C", "CAVH Start", "CAVH D/C"):
        assert pattern.search(event)
    for other in ("EKG", "Arterial Line Placement", "Hemodialysis cath"):
        assert not pattern.search(other)


def test_p3_plateau_above_pip_or_at_peep_gives_no_driving_pressure():
    from easyicu.callbacks import driving_pressure

    times = [0.0, 60.0, 120.0, 240.0]
    plateau = pd.DataFrame({"stay_id": 1, "charttime": times, "plateau_pres": [22.0, 30.0, 8.0, 25.0]})
    peep = pd.DataFrame({"stay_id": 1, "charttime": times, "peep": [8.0, 8.0, 8.0, 8.0]})
    pip = pd.DataFrame({"stay_id": 1, "charttime": [0.0, 60.0, 120.0], "pip": [28.0, 26.0, 20.0]})
    out = driving_pressure(plateau, peep, pip=pip)
    # 60: plateau 30 > PIP 26 + 1; 120: plateau at PEEP; 240: no PIP within 1 h, kept.
    assert out.set_index("charttime")["driving_pres"].to_dict() == {0.0: 14.0, 240.0: 17.0}
    assert len(driving_pressure(plateau, peep)) == 3


@pytest.mark.parametrize("concept", ["driving_pres", "driving_pres_controlled"])
def test_p3_driving_pressure_loads_pip(concept):
    assert "pip" in DICTIONARY[concept]["concepts"]


def test_w1_mimic_weight_prefers_admission_weight_per_stay():
    from easyicu.concept.source_channel import RANK_COLUMN, select_preferred_channel

    assert DICTIONARY["weight"]["source_channel_priority"][0] == "admission"
    [admission] = [s for s in _sources("weight", "mimic") if s["source_channel"] == "admission"]
    assert admission["ids"] == [762, 226512]
    assert 4183 not in _ids("weight", "mimic")
    frame = pd.DataFrame({
        "icustay_id": [1, 1, 1, 2, 2, 3],
        "charttime": [0.0, 30.0, 50.0, 5.0, 40.0, 1.0],
        "weight": [80.0, 84.0, None, 70.0, 72.0, 600.0],
        RANK_COLUMN: [0, 2, 0, 2, 3, 0],
    })
    kept = select_preferred_channel(frame, value_column="weight", id_columns=["icustay_id"],
                                    time_column=None, minimum=1, maximum=500, per_stay=True)
    # Stay 1 keeps its admission weight across hours; stay 2 its charted weight;
    # stay 3 has no usable value, so the bounds filter decides.
    assert kept["charttime"].tolist() == [0.0, 50.0, 5.0, 1.0]


@pytest.mark.parametrize("interval", [None, "00:01:00", "01:00:00"])
def test_v6_hirid_rate_is_hourly_mass_over_60_min_at_any_concept_interval(interval):
    # A givendose is the amount since the infusion's previous record; the
    # *60 concepts load their rates at 00:01:00, which once divided it by one
    # minute and inflated the SOFA cardiovascular doses about 20-fold.
    from easyicu.concept import ConceptSource, _apply_callback

    frame = pd.DataFrame({
        "patientid": [1, 1], "givenat": [2.1, 2.6], "infusionid": [10, 10],
        "givendose": [60.0, 60.0], "doseunit": ["µg", "µg"], "weight": [50.0, 50.0],
    })
    source = ConceptSource(callback="hirid_rate_kg", value_var="givendose",
                           unit_var="doseunit", index_var="givenat")
    out = _apply_callback(frame, source, concept_name="norepi_rate", interval=interval)
    assert out["norepi_rate"].tolist() == pytest.approx([120.0 / 60.0 / 50.0])


def test_v3_eicu_other_vaso_reads_hicl_only_vasopressin_and_phenylephrine():
    from easyicu.resources import load_dictionary

    sources = load_dictionary(include_sofa2=True)["other_vaso"].sources["eicu"]
    assert any(s.sub_var == "drughiclseqno" and {2839, 2087} <= set(s.ids or ()) for s in sources)


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_d3_every_eicu_medication_source_drops_cancelled_orders(database):
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(include_sofa2=True)
    medication = [
        (name, source)
        for name, concept in dictionary.items()
        for source in concept.sources.get(database, [])
        if source.table == "medication"
    ]
    assert medication
    for name, source in medication:
        assert source.params.get("excluded_statuses") == ["true"], name
        assert source.params.get("status_var") == "drugordercancelled", name


def test_d3_cancelled_order_is_dropped_before_any_callback():
    from easyicu.concept import ConceptSource, _apply_callback

    frame = pd.DataFrame({"drugname": ["heparin", "heparin"], "drugordercancelled": [False, True]})
    source = ConceptSource.from_mapping({
        "regex": "heparin", "table": "medication", "sub_var": "drugname",
        "class_name": "rgx_itm", "callback": "transform_fun(set_val(TRUE))",
        "excluded_statuses": ["true"], "status_var": "drugordercancelled",
    })
    out = _apply_callback(frame, source, concept_name="heparin")
    assert out["drugordercancelled"].tolist() == [False]


@pytest.mark.parametrize("concept", ["norepi_rate", "epi_rate", "phn_rate"])
def test_v2_eicu_volume_rates_give_no_dose(concept):
    from easyicu.concept import ConceptSource, _apply_callback

    [source] = _sources(concept, "eicu")
    assert source["callback"] == "eicu_rate_kg(ml_to_mcg = NA)"
    frame = pd.DataFrame({
        "patientunitstayid": [1, 1], "infusionoffset": [60, 120],
        "drugname": ["Norepinephrine (ml/hr)", "Norepinephrine (mcg/min)"],
        "drugrate": [10.0, 8.0], "patientweight": [80.0, 80.0],
    })
    out = _apply_callback(frame, ConceptSource.from_mapping(dict(source, value_var="drugrate")),
                          concept_name=concept)
    rates = out.dropna(subset=[concept])[concept]
    assert rates.tolist() == pytest.approx([8.0 / 80.0])


def test_p4_peep_set_and_total_have_sources_beyond_mimic():
    for database in ("aumc", "hirid", "eicu", "sic"):
        assert _sources("peep_set", database), database
    assert _ids("peep_total", "aumc") == {12301}
    assert _ids("peep_total", "hirid") == {2610}


def test_p9_carevue_tcpcv_is_pressure_controlled():
    from easyicu.concept.callbacks import _load_vent_mode_map

    assert _load_vent_mode_map()["mimic"]["map"]["TCPCV"] == {
        "control": "pressure", "seq": "controlled", "interface": "invasive"}


def test_g6_g7_supp_o2_reads_oxygen_flow_everywhere():
    assert "o2_flow" in DICTIONARY["supp_o2"]["concepts"]
    assert 223834 in _ids("o2_flow", "miiv")
    assert {470, 223834} <= _ids("o2_flow", "mimic")
    assert _ids("o2_flow", "eicu") == {"LPM O2"}


def test_g8_safi_is_pulse_oximetry_and_tco2_is_arterial():
    assert DICTIONARY["safi"]["concepts"] == ["spo2", "fio2"]
    for database in ("miiv", "mimic"):
        assert _sources("tco2", database)[0]["callback"] == "arterial_specimen_only"


@pytest.mark.parametrize("dictionary", ["concept-dict.json", "sofa2-dict.json"])
def test_h7_carevue_iabp_setting_is_mechanical_support(dictionary):
    data = json.loads((Path(__file__).parents[2] / "src/easyicu/data" / dictionary).read_text())
    assert any(225 in (s.get("ids") or []) for s in data["mech_circ_support"]["sources"]["mimic"])


def test_h5_h6_arterial_line_channels():
    assert {225309} <= _ids("sbp", "miiv") and {225310} <= _ids("dbp", "miiv")
    invasive = [s for s in _sources("map", "eicu") if s.get("table") == "nursecharting"]
    assert invasive and {s["source_channel"] for s in invasive} == {"invasive"}
    for concept in ("hr", "resp", "spo2"):
        assert DICTIONARY[concept]["source_channel_priority"] == ["monitor", "charted"]
        channels = {s["table"]: s["source_channel"] for s in _sources(concept, "eicu")}
        assert channels == {"vitalperiodic": "monitor", "nursecharting": "charted"}


def test_w2_weight_fallbacks_rank_after_admission_weight():
    for database in ("miiv", "eicu"):
        first = _sources("weight", database)[0]
        assert first["source_channel"] == "admission"
        assert len(_sources("weight", database)) == 3


def test_v5_hirid_concurrent_syringes_add_up_in_the_hour():
    from easyicu.utils.callback_utils import hirid_rate_kg

    frame = pd.DataFrame({
        "patientid": [1, 1], "givenat": [2.2, 2.7], "infusionid": [10, 11],
        "givendose": [60.0, 60.0], "doseunit": ["µg", "µg"], "weight": [50.0, 50.0],
    })
    out = hirid_rate_kg(frame, concept_name="norepi_rate", val_col="givendose", unit_col="doseunit",
                        grp_var="infusionid", index_col="givenat", interval_minutes=60.0)
    assert out["norepi_rate"].tolist() == pytest.approx([120.0 / 60.0 / 50.0])


def test_v5_hirid_syringes_within_five_minutes_are_one_episode():
    from easyicu.utils.callback_utils import hirid_duration

    frame = pd.DataFrame({
        "patientid": [1] * 6,
        "givenat": [0.0, 3.5, 3.55, 7.2, 10.0, 12.0],
        "infusionid": [1, 1, 2, 2, 3, 3],
    })
    out = hirid_duration(frame, concept_name="norepi_dur", index_col="givenat", grp_var="infusionid")
    # Syringe 2 starts 3 minutes after syringe 1 ends; syringe 3 starts 2.8 h later.
    assert out[["givenat", "norepi_dur"]].values.tolist() == [[0.0, 7.0], [10.0, 2.0]]


def test_p6_sic_tidal_volume_below_100_ml_is_missing():
    [source] = _sources("tidal_vol", "sic")
    assert source["ids"] == [718] and source["callback"] == "transform_fun(na_below(100))"


def test_s8_hirid_items_reach_their_concepts():
    sofa2 = json.loads((Path(__file__).parents[2] / "src/easyicu/data/sofa2-dict.json").read_text())
    assert _ids("neut", "hirid") == {24000556}  # 24000550 behaves as an absolute count
    assert _ids("diltiazem", "hirid") == {1000331, 1001071}
    assert {1000793} <= set(sofa2["other_vaso"]["sources"]["hirid"][0]["ids"])
    for concept in ("epi_rate", "epi_dur"):
        assert _ids(concept, "hirid") == {1000649, 1000650}  # perfusors; no bolus or inhaled rows
    assert DICTIONARY["ett_gcs"]["description"].startswith("invasive airway")


def test_s8_hirid_admission_type_prefers_apache_ii_within_the_first_day():
    from easyicu.concept.hirid_admission import apache_admission_type

    frame = pd.DataFrame({
        "patientid": [1, 1, 2, 2, 3, 4, 5],
        "datetime": [0.0, 30.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        "variableid": [9990002, 9990002, 9990004, 9990002, 9990004, 9990004, 9990002],
        "value": [107.0, 98.0, 190.0, 108.0, 206.0, 205.0, 150.0],
    })
    out = apache_admission_type(frame, id_col="patientid", time_col="datetime",
                                item_col="variableid", value_col="value")
    # 1: the 30 h correction is outside the day; 2: APACHE II wins; 5: unlisted code.
    assert out.to_dict() == {1: "surg", 2: "surg", 3: "med", 4: "surg"}


def test_s1_sic_advanced_support_reuses_the_invasive_ventilation_windows():
    sofa2 = json.loads((Path(__file__).parents[2] / "src/easyicu/data/sofa2-dict.json").read_text())
    tube, settings = sofa2["adv_resp"]["sources"]["sic"]
    [_, vent_settings] = _sources("mech_vent", "sic")
    assert tube["ids"] == [720] and tube["dur_var"] == "OffsetEnd"
    assert settings["ids"] == vent_settings["ids"]
    assert settings["callback"].startswith("combine_callbacks(sicdb_airway_attributed_ventilation")


@pytest.mark.parametrize(("concept", "hicl"), [
    ("heparin", 2810), ("albumin_iv", 2728), ("vancomycin", 4042), ("abx", 4042),
    ("propofol", 4842), ("warfarin", 2812), ("diltiazem", 182),
])
def test_s6_eicu_orders_without_a_drug_name_match_by_hicl(concept, hicl):
    for database in ("eicu", "eicu_demo"):
        hicl_sources = [s for s in _sources(concept, database) if s.get("sub_var") == "drughiclseqno"]
        assert any(hicl in s["ids"] for s in hicl_sources), database
        assert all(s["status_var"] == "drugordercancelled" for s in hicl_sources)


def test_s6_albumin_brand_names_match():
    import re

    [named] = [s for s in _sources("albumin_iv", "eicu") if s.get("sub_var") == "drugname"]
    for name in ("BUMINATE", "albumin human", "Albumin (Human) (PLASBUMIN-5) 5% infusion 25 g"):
        assert re.search(named["regex"], name)


def test_g5_pafi_pairs_the_lowest_pao2_with_the_median_fio2_of_the_hour():
    assert DICTIONARY["pafi"]["concepts"] == ["po2", "fio2"]
    assert DICTIONARY["pafi"]["aggregate"] == ["min", "median"]


@pytest.mark.parametrize(("concept", "invasive", "cuff"), [("sbp", 100, 600), ("dbp", 120, 620)])
def test_h4_hirid_cuff_pressure_fills_hours_without_an_arterial_line(concept, invasive, cuff):
    assert DICTIONARY[concept]["source_channel_priority"] == ["invasive", "nibp"]
    channels = {s["source_channel"]: s["ids"] for s in _sources(concept, "hirid")}
    assert channels == {"invasive": [invasive], "nibp": [cuff]}


def test_p5_aumc_tidal_volume_is_expiratory():
    ids = _ids("tidal_vol", "aumc")
    assert not ids & {12277, 12360}  # inspiratory channels
    assert {12275, 12358} <= ids
