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
