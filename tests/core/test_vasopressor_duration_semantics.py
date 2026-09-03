from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.config import DataSourceConfig
from easyicu.concept import ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import (
    ConceptDefinition,
    ConceptDictionary,
    ConceptSource,
)
from easyicu.load_concepts import ConceptLoader
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import mimic_dur_incv, mimic_dur_inmv


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _mimic_icu_stays(**outtimes: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "icustay_id": [int(stay_id) for stay_id in outtimes],
            "outtime": pd.to_datetime(list(outtimes.values())),
        }
    )


def test_mimic_mv_filters_non_administrations_and_splits_long_gaps() -> None:
    frame = pd.DataFrame(
        {
            "icustay_id": [1] * 9,
            "linkorderid": [10] * 9,
            "starttime": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:35:00",  # exactly five-minute gap
                    "2025-01-01 01:00:01",  # five minutes + one second
                    "2025-01-01 03:00:00",  # rewritten extreme
                    "2025-01-01 04:00:00",  # cancelreason != 0
                    "2025-01-01 05:00:00",  # flushed
                    "2025-01-01 06:00:00",  # bolus
                    "2025-01-01 07:00:00",  # cancelled spelling
                    "2025-01-01 08:00:00",  # negative interval
                ]
            ),
            "endtime": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 00:55:00",
                    "2025-01-01 01:15:01",
                    "2026-01-01 03:00:00",
                    "2025-01-01 04:30:00",
                    "2025-01-01 05:30:00",
                    "2025-01-01 06:30:00",
                    "2025-01-01 07:30:00",
                    "2025-01-01 07:59:00",
                ]
            ),
            "statusdescription": [
                "Changed",
                "Stopped",
                "FinishedRunning",
                "Rewritten",
                "Changed",
                "Flushed",
                "Bolus",
                "Cancelled",
                "Changed",
            ],
            "cancelreason": [0, np.nan, 0, 0, 1, 0, 0, 0, 0],
        }
    )

    result = mimic_dur_inmv(
        frame,
        val_col="norepi_dur",
        grp_var="linkorderid",
        stop_var="endtime",
        id_cols=["icustay_id"],
        icu_stays=_mimic_icu_stays(**{"1": "2025-01-02 00:00:00"}),
        cancel_var="cancelreason",
        merge_gap_minutes=5,
    )

    assert result["starttime"].tolist() == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 01:00:01"),
    ]
    assert result["norepi_dur"].tolist() == pytest.approx([55 / 60, 0.25])


def test_mimic_mv_keeps_subhour_precision_and_clips_to_icu_outtime() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [7, 7, 8],
            "linkorderid": [70, 71, 80],
            "starttime": pd.to_datetime(
                [
                    "2025-01-01 10:00:00",
                    "2025-01-01 10:45:00",
                    "2025-01-01 12:00:00",
                ]
            ),
            "endtime": pd.to_datetime(
                [
                    "2025-01-01 10:30:00",
                    "2025-01-01 12:00:00",
                    "2025-01-01 13:00:00",
                ]
            ),
            "statusdescription": ["ChangeDose/Rate"] * 3,
        }
    )
    icu_stays = pd.DataFrame(
        {
            "stay_id": [7, 8],
            "outtime": pd.to_datetime(
                ["2025-01-01 11:00:00", "2025-01-01 11:30:00"]
            ),
        }
    )

    result = mimic_dur_inmv(
        frame,
        val_col="epi_dur",
        grp_var="linkorderid",
        stop_var="endtime",
        id_cols=["stay_id"],
        icu_stays=icu_stays,
    )

    assert result["stay_id"].tolist() == [7, 7]
    assert result["epi_dur"].tolist() == pytest.approx([0.5, 0.25])
    assert (result["epi_dur"] > 0).all()


def test_mimic_mv_relative_hours_convert_absolute_icu_outtime() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [7],
            "linkorderid": [70],
            "starttime": [0.25],
            "endtime": [2.0],
            "statusdescription": ["ChangeDose/Rate"],
        }
    )
    icu_stays = pd.DataFrame(
        {
            "stay_id": [7],
            "intime": pd.to_datetime(["2025-01-01 00:00:00"]),
            "outtime": pd.to_datetime(["2025-01-01 01:00:00"]),
        }
    )

    result = mimic_dur_inmv(
        frame,
        val_col="norepi_dur",
        grp_var="linkorderid",
        stop_var="endtime",
        id_cols=["stay_id"],
        icu_stays=icu_stays,
    )

    assert result["starttime"].tolist() == [0.25]
    assert result["norepi_dur"].tolist() == pytest.approx([0.75])


def test_mimic_mv_relative_hours_refuse_absolute_outtime_without_origin() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [7],
            "starttime": [0.25],
            "endtime": [2.0],
            "statusdescription": ["ChangeDose/Rate"],
        }
    )
    icu_stays = pd.DataFrame(
        {
            "stay_id": [7],
            "outtime": pd.to_datetime(["2025-01-01 01:00:00"]),
        }
    )

    with pytest.raises(ValueError, match="require ICU intime"):
        mimic_dur_inmv(
            frame,
            val_col="norepi_dur",
            stop_var="endtime",
            id_cols=["stay_id"],
            icu_stays=icu_stays,
        )


def test_mimic_mv_refuses_mixed_source_clocks() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [7],
            "starttime": [0.25],
            "endtime": pd.to_datetime(["2025-01-01 02:00:00"]),
            "statusdescription": ["ChangeDose/Rate"],
        }
    )

    with pytest.raises(ValueError, match="mixed numeric and datetime"):
        mimic_dur_inmv(
            frame,
            val_col="norepi_dur",
            stop_var="endtime",
            id_cols=["stay_id"],
        )


def test_mimic_cv_uses_five_hour_segments_and_singleton_unknown() -> None:
    frame = pd.DataFrame(
        {
            "icustay_id": [1, 1, 1, 1, 2, 2],
            # CareVue duration follows the stay-plus-drug event stream; a
            # linkorder change alone is not an episode boundary.
            "linkorderid": [99, 100, 101, 101, 99, 99],
            "charttime": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 05:00:00",  # exact boundary: same episode
                    "2025-01-01 10:00:01",  # split by one second
                    "2025-01-01 11:00:01",
                    "2025-01-01 00:00:00",
                    "2025-01-01 02:00:00",
                ]
            ),
            "stopped": [None] * 6,
            "norepi_dur": [0.1] * 6,
        }
    )
    icu_stays = _mimic_icu_stays(
        **{
            "1": "2025-01-02 00:00:00",
            "2": "2025-01-02 00:00:00",
        }
    )

    result = mimic_dur_incv(
        frame,
        val_col="norepi_dur",
        grp_var="linkorderid",
        id_cols=["icustay_id"],
        icu_stays=icu_stays,
        merge_gap_hours=5,
    )

    stay_one = result[result["icustay_id"] == 1]
    assert stay_one["norepi_dur"].tolist() == pytest.approx([5.0, 1.0])
    stay_two = result[result["icustay_id"] == 2]
    assert stay_two["norepi_dur"].tolist() == [2.0]

    singleton = mimic_dur_incv(
        frame.iloc[[0]],
        val_col="norepi_dur",
        grp_var="linkorderid",
        id_cols=["icustay_id"],
        icu_stays=icu_stays,
    )
    assert len(singleton) == 1
    assert np.isnan(singleton["norepi_dur"].iloc[0])


def test_mimic_cv_explicit_stop_restart_boundaries_and_outtime_clip() -> None:
    frame = pd.DataFrame(
        {
            "icustay_id": [3] * 6,
            "linkorderid": [30] * 6,
            "charttime": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                    "2025-01-01 03:00:00",
                    "2025-01-01 04:00:00",
                    "2025-01-01 06:00:00",
                ]
            ),
            "stopped": [None, "Stopped", None, "Restart", None, None],
            "dopa_dur": [0.1, 0.1, 0.2, np.nan, 0.2, 0.3],
        }
    )
    result = mimic_dur_incv(
        frame,
        val_col="dopa_dur",
        grp_var="linkorderid",
        id_cols=["icustay_id"],
        icu_stays=_mimic_icu_stays(**{"3": "2025-01-01 05:00:00"}),
    )

    assert result["charttime"].tolist() == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 02:00:00"),
        pd.Timestamp("2025-01-01 04:00:00"),
    ]
    assert result["dopa_dur"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(result["dopa_dur"].iloc[1])
    assert result["dopa_dur"].iloc[2] == pytest.approx(1.0)


def test_mimic_cv_zero_rate_terminates_and_start_at_outtime_is_dropped() -> None:
    frame = pd.DataFrame(
        {
            "icustay_id": [4, 4, 4, 4, 5],
            "charttime": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 02:00:00",
                    "2025-01-01 03:00:00",
                    "2025-01-01 05:00:00",
                ]
            ),
            "stopped": [None] * 5,
            "epi_dur": [0.1, 0.0, 0.2, 0.3, 0.2],
        }
    )
    icu_stays = pd.DataFrame(
        {
            "icustay_id": [4, 5],
            "outtime": pd.to_datetime(
                ["2025-01-01 05:00:00", "2025-01-01 05:00:00"]
            ),
        }
    )

    result = mimic_dur_incv(
        frame,
        val_col="epi_dur",
        id_cols=["icustay_id"],
        icu_stays=icu_stays,
    )

    assert result["icustay_id"].tolist() == [4, 4]
    assert result["charttime"].tolist() == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 02:00:00"),
    ]
    assert result["epi_dur"].tolist() == pytest.approx([1.0, 1.0])


def test_mimic_dispatcher_uses_intime_los_when_outtime_is_missing() -> None:
    class DataSource:
        def load_table(self, table_name, columns=None, verbose=False):
            del table_name, verbose
            frame = pd.DataFrame(
                {
                    "stay_id": [6, 7],
                    "intime": [pd.Timestamp("2025-01-01 00:00:00"), pd.NaT],
                    "outtime": [pd.NaT, pd.NaT],
                    "los": [1 / 24, np.nan],
                }
            )
            return ICUTable(
                data=frame[[c for c in columns or [] if c in frame.columns]],
                id_columns=["stay_id"],
                index_column="intime",
            )

    frame = pd.DataFrame(
        {
            "stay_id": [6, 7],
            "linkorderid": [60, 70],
            "starttime": pd.to_datetime(
                ["2025-01-01 00:15:00", "2025-01-01 00:00:00"]
            ),
            "endtime": pd.to_datetime(
                ["2025-01-01 02:00:00", "2025-01-03 00:00:00"]
            ),
            "statusdescription": ["ChangeDose/Rate", "ChangeDose/Rate"],
            "norepi_dur": [0.1, 0.2],
        }
    )
    # Reproduce pandas 3's source-unit preservation explicitly.  The ICU LOS
    # fallback below has nanosecond rounding and must be safely assignable.
    frame["starttime"] = frame["starttime"].astype("datetime64[us]")
    frame["endtime"] = frame["endtime"].astype("datetime64[us]")
    source = ConceptSource.from_mapping(
        {
            "table": "inputevents",
            "stop_var": "endtime",
            "grp_var": "linkorderid",
            "status_var": "statusdescription",
            "callback": "mimic_dur_inmv",
        }
    )

    result = _apply_callback(
        frame,
        source,
        "norepi_dur",
        data_source=DataSource(),
    )

    assert result["stay_id"].tolist() == [6]
    assert result["norepi_dur"].tolist() == pytest.approx([0.75])


def test_mimic_dictionary_declares_status_gap_and_inference_contracts() -> None:
    dictionary = json.loads(CONCEPT_DICT.read_text(encoding="utf-8"))

    for drug in ("dobu", "dopa", "epi", "norepi"):
        concept = dictionary[f"{drug}_dur"]
        assert "not exact pump-on time" in concept["description"]
        miiv = concept["sources"]["miiv"][0]
        assert miiv["extra_vars"] == ["statusdescription"]
        assert miiv["merge_gap_minutes"] == 5
        assert {"Rewritten", "Flushed", "Bolus"}.issubset(
            miiv["excluded_statuses"]
        )

        for database in ("mimic", "mimic_demo"):
            by_table = {
                source["table"]: source
                for source in concept["sources"][database]
            }
            cv = by_table["inputevents_cv"]
            assert cv["extra_vars"] == ["stopped", "rate"]
            assert cv["boundary_var"] == "stopped"
            assert cv["rate_var"] == "rate"
            assert cv["merge_gap_hours"] == 5
            assert "grp_var" not in cv

            mv = by_table["inputevents_mv"]
            assert mv["extra_vars"] == ["statusdescription", "cancelreason"]
            assert mv["cancel_var"] == "cancelreason"
            assert mv["merge_gap_minutes"] == 5


def test_loader_requests_mimic_mv_semantic_columns_and_clips_outtime() -> None:
    class DataSource:
        base_path = None

        def __init__(self) -> None:
            self.requested: dict[str, list[list[str]]] = {}
            self.config = DataSourceConfig(
                name="mimic",
                tables={
                    "inputevents_mv": {
                        "defaults": {
                            "id_var": "icustay_id",
                            "index_var": "starttime",
                            "val_var": "rate",
                            "unit_var": "rateuom",
                            "time_vars": ["starttime", "endtime"],
                        }
                    },
                    "icustays": {
                        "defaults": {
                            "id_var": "icustay_id",
                            "index_var": "intime",
                            "time_vars": ["intime", "outtime"],
                        }
                    },
                },
            )

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            del verbose
            self.requested.setdefault(table_name, []).append(list(columns or []))
            if table_name == "icustays":
                frame = pd.DataFrame(
                    {
                        "icustay_id": [1],
                        "intime": [pd.Timestamp("2025-01-01 00:00:00")],
                        "outtime": [pd.Timestamp("2025-01-01 01:00:00")],
                        "los": [1 / 24],
                    }
                )
                keep = ["icustay_id", *[c for c in columns or [] if c != "icustay_id"]]
                return ICUTable(
                    data=frame[list(dict.fromkeys(keep))],
                    id_columns=["icustay_id"],
                    index_column="intime",
                    time_columns=[c for c in ("intime", "outtime") if c in keep],
                )
            frame = pd.DataFrame(
                {
                    "icustay_id": [1, 1],
                    "itemid": [221906, 221906],
                    "linkorderid": [11, 12],
                    "starttime": pd.to_datetime(
                        ["2025-01-01 00:15:00", "2025-01-01 00:30:00"]
                    ),
                    "endtime": pd.to_datetime(
                        ["2025-01-01 02:00:00", "2026-01-01 00:00:00"]
                    ),
                    "rate": [0.1, 0.2],
                    "rateuom": ["mcg/kg/min", "mcg/kg/min"],
                    "statusdescription": ["Changed", "Rewritten"],
                    "cancelreason": [0, 0],
                }
            )
            for filter_spec in filters or []:
                frame = filter_spec.apply(frame)
            return ICUTable(
                data=frame,
                id_columns=["icustay_id"],
                index_column="starttime",
                value_column="rate",
                unit_column="rateuom",
                time_columns=["starttime", "endtime"],
            )

    source = DataSource()
    dictionary = ConceptDictionary(
        {
            "norepi_dur": ConceptDefinition(
                name="norepi_dur",
                units=["hours"],
                minimum=0,
                aggregate="max",
                sources={
                    "mimic": [
                        ConceptSource.from_mapping(
                            {
                                "ids": 221906,
                                "table": "inputevents_mv",
                                "sub_var": "itemid",
                                "stop_var": "endtime",
                                "grp_var": "linkorderid",
                                "extra_vars": [
                                    "statusdescription",
                                    "cancelreason",
                                ],
                                "status_var": "statusdescription",
                                "cancel_var": "cancelreason",
                                "merge_gap_minutes": 5,
                                "callback": "mimic_dur_inmv",
                            }
                        )
                    ]
                },
            )
        }
    )

    loaded = ConceptResolver(dictionary).load_concepts(
        ["norepi_dur"],
        source,
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )["norepi_dur"]

    requested = source.requested["inputevents_mv"][0]
    assert "statusdescription" in requested
    assert "cancelreason" in requested
    assert any("outtime" in request for request in source.requested["icustays"])
    assert loaded.data["norepi_dur"].tolist() == pytest.approx([0.75])
    assert "rateuom" not in loaded.data.columns


def test_legacy_loader_preserves_mimic_duration_semantics(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "stay_id": [7, 7],
            "hadm_id": [70, 70],
            "starttime": pd.to_datetime(
                ["2025-01-01 00:15:00", "2025-01-01 00:30:00"]
            ),
            "endtime": pd.to_datetime(
                ["2025-01-01 02:00:00", "2026-01-01 00:00:00"]
            ),
            "itemid": [221906, 221906],
            "amount": [1.0, 1.0],
            "amountuom": ["mg", "mg"],
            "rate": [0.1, 0.2],
            "linkorderid": [700, 701],
            "statusdescription": ["ChangeDose/Rate", "Rewritten"],
        }
    ).to_parquet(tmp_path / "inputevents.parquet")
    pd.DataFrame(
        {
            "stay_id": [7],
            "intime": pd.to_datetime(["2025-01-01 00:00:00"]),
            "outtime": pd.to_datetime(["2025-01-01 01:00:00"]),
            "los": [1 / 24],
        }
    ).to_parquet(tmp_path / "icustays.parquet")

    loader = ConceptLoader("miiv", data_path=str(tmp_path), low_memory=True)
    result = loader.load_concepts(
        ["norepi_dur"],
        patient_ids=[7],
        merge_data=False,
        verbose=False,
        concept_workers=1,
    )["norepi_dur"]

    assert result.columns.tolist() == ["stay_id", "charttime", "norepi_dur"]
    assert result["charttime"].tolist() == pytest.approx([0.25])
    assert result["norepi_dur"].tolist() == pytest.approx([0.75])
