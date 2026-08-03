from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource


REPO_ROOT = Path(__file__).resolve().parents[1]
CONCEPT_DICT = REPO_ROOT / "src" / "easyicu" / "data" / "concept-dict.json"


def _aumc_source() -> ConceptSource:
    return ConceptSource.from_mapping(
        {
            "ids": 6818,
            "table": "drugitems",
            "sub_var": "itemid",
            "stop_var": "stop",
            "extra_vars": ["iscontinuous", "action"],
            "continuous_var": "iscontinuous",
            "action_var": "action",
            "merge_gap_minutes": 5,
            "callback": "aumc_dur",
        }
    )


def _apply_aumc(frame: pd.DataFrame) -> pd.DataFrame:
    # Exercise the active source-callback dispatcher, not the legacy callback
    # registry in easyicu.concept.callbacks.
    return _apply_callback(frame, _aumc_source(), "epi_dur")


def test_aumc_bolus_flush_and_zero_length_rows_do_not_create_duration() -> None:
    frame = pd.DataFrame(
        {
            "admissionid": [1, 1, 1, 2],
            "orderid": [10, 11, 12, 20],
            "itemid": [6818, 6818, 6818, 6818],
            "start": [0, 120, 240, 0],
            "stop": [60, 180, 240, 60],
            "iscontinuous": [True, False, True, False],
            # The flush is deliberately continuous-flagged, matching real
            # AUMC rows; the action label must still exclude it.
            "action": ["rate change", "bolus", "Flush", "bolus"],
            "epi_dur": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = _apply_aumc(frame)

    assert result.to_dict("records") == [
        {"admissionid": 1, "start": 0, "epi_dur": 1.0}
    ]


def test_aumc_pump_and_rate_changes_merge_across_unique_order_ids() -> None:
    frame = pd.DataFrame(
        {
            "admissionid": [7, 7, 7],
            "orderid": [1001, 1002, 1003],
            "itemid": [6818, 6818, 6818],
            "start": [1_001, 1_030, 1_060],
            "stop": [1_040, 1_065, 1_121],
            "iscontinuous": [1, 1, 1],
            "action": ["Nieuwe spuit", "Snelheid veranderd", "Snelheid veranderd"],
            "rate": [0.04, 0.06, 0.08],
            "epi_dur": [0.04, 0.06, 0.08],
        }
    )

    result = _apply_aumc(frame)

    assert result["admissionid"].tolist() == [7]
    # The source clock stays in minutes for the central admission aligner.
    assert result["start"].tolist() == [1_001]
    # Exact episode length: no floor(absolute clock / 60) quantisation.
    assert result["epi_dur"].iloc[0] == pytest.approx(120 / 60)


def test_aumc_overlap_and_five_minute_gap_boundary_preserve_episode_span() -> None:
    frame = pd.DataFrame(
        {
            "admissionid": [9, 9, 9, 9],
            "orderid": [1, 2, 3, 4],
            "itemid": [6818, 6818, 6818, 6818],
            "start": [0, 30, 95, 131],
            "stop": [60, 90, 125, 191],
            "iscontinuous": ["true", "1", "yes", "t"],
            "action": ["Nieuwe spuit", "Snelheid veranderd", "Herstart", None],
            "epi_dur": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = _apply_aumc(frame)

    # Overlap does not double-count, and a gap exactly at the configured
    # five-minute tolerance remains in the same clinical episode.  A six-minute
    # gap starts a new episode.
    assert result["start"].tolist() == [0, 131]
    assert result["epi_dur"].tolist() == pytest.approx([125 / 60, 1.0])
    reconstructed_ends = result["start"] + result["epi_dur"] * 60
    assert reconstructed_ends.tolist() == pytest.approx([125, 191])


def test_aumc_duration_fails_closed_without_continuous_flag() -> None:
    frame = pd.DataFrame(
        {
            "admissionid": [1],
            "itemid": [6818],
            "start": [0],
            "stop": [60],
            "epi_dur": [1.0],
        }
    )

    with pytest.raises(ValueError, match="iscontinuous"):
        _apply_aumc(frame)


def test_eicu_singleton_duration_is_unknown_and_five_hour_groups_remain() -> None:
    source = ConceptSource.from_mapping(
        {
            "regex": "(?i)^epi",
            "table": "infusiondrug",
            "sub_var": "drugname",
            "callback": "eicu_duration(gap_length = hours(5L))",
            "class_name": "rgx_itm",
        }
    )
    frame = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 2, 3, 3, 4, 4],
            "infusionoffset": [60, 0, 300, 0, 360, 0, 240],
            "epi_dur": [1.0] * 7,
        }
    )

    result = _apply_callback(frame, source, "epi_dur")

    singleton = result[result["patientunitstayid"] == 1]
    assert len(singleton) == 1
    assert np.isnan(singleton["epi_dur"].iloc[0])

    exact_boundary = result[result["patientunitstayid"] == 2]
    assert exact_boundary["epi_dur"].tolist() == [5.0]

    split_singletons = result[result["patientunitstayid"] == 3]
    assert len(split_singletons) == 2
    assert split_singletons["epi_dur"].isna().all()

    ordinary_episode = result[result["patientunitstayid"] == 4]
    assert ordinary_episode["epi_dur"].tolist() == [4.0]


def test_dictionary_binds_duration_and_vaso60_to_one_episode_contract() -> None:
    dictionary = json.loads(CONCEPT_DICT.read_text(encoding="utf-8"))

    for drug in ("dobu", "dopa", "epi", "norepi"):
        duration_name = f"{drug}_dur"
        source = dictionary[duration_name]["sources"]["aumc"][0]
        assert source["callback"] == "aumc_dur"
        assert source["extra_vars"] == ["iscontinuous", "action"]
        assert source["continuous_var"] == "iscontinuous"
        assert source["action_var"] == "action"
        assert source["merge_gap_minutes"] == 5
        assert "grp_var" not in source
        assert dictionary[duration_name]["unit"] == "hours"
        assert dictionary[f"{drug}60"]["concepts"] == [
            f"{drug}_rate",
            duration_name,
        ]
