"""MAP channels: invasive preferred per hour, cuff otherwise (2026-11-03 audit #16).

Synthetic chart events go through the real resolver, callbacks and hourly
aggregation, so the old behaviour (one pooled median of arterial and cuff
values) fails these tests.
"""
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.source_channel import (
    RANK_COLUMN,
    SourceChannelContractError,
    select_preferred_channel,
    validate_sources,
)
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary

DICTIONARY = Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json"
MAIN_DATABASES = ["aumc", "eicu", "hirid", "miiv", "mimic", "sic"]


def _chart_source(tmp_path, database):
    id_column = "stay_id" if database == "miiv" else "icustay_id"
    invasive, cuff = (220052, 220181)
    admission = pd.Timestamp("2180-01-01 00:00:00")
    rows = [
        # stay 1, hour 0: arterial 60/62 and cuff 90 -> arterial only (61)
        (1, 5, invasive, 60.0), (1, 25, invasive, 62.0), (1, 15, cuff, 90.0),
        # stay 1, hour 1: cuff only -> 80
        (1, 70, cuff, 80.0),
        # stay 1, hour 2: arterial out of bounds (300) does not suppress the cuff
        (1, 130, invasive, 300.0), (1, 140, cuff, 75.0),
        # stay 2, hour 0: arterial only
        (2, 10, invasive, 70.0),
    ]
    if database == "mimic":
        rows += [(1, 200, 6702, 66.0), (1, 210, 456, 99.0)]  # hour 3: Arterial BP Mean #2 vs NBP
    frame = pd.DataFrame(rows, columns=[id_column, "minute", "itemid", "valuenum"])
    frame["charttime"] = admission + pd.to_timedelta(frame.pop("minute"), unit="m")
    frame["storetime"] = frame.charttime
    frame["value"] = frame.valuenum.astype(str)
    frame["valueuom"] = "mmHg"
    frame["subject_id"] = frame[id_column] + 100
    frame["hadm_id"] = frame[id_column] + 200
    (tmp_path / "chartevents").mkdir()
    frame.to_parquet(tmp_path / "chartevents" / "part.parquet", index=False)
    stays = pd.DataFrame({id_column: [1, 2], "subject_id": [101, 102], "hadm_id": [201, 202],
                          "intime": [admission] * 2,
                          "outtime": [admission + pd.Timedelta(hours=48)] * 2})
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)
    return ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False), id_column


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("use_duckdb", [False, True])
def test_map_prefers_the_arterial_line_within_each_hour(tmp_path, monkeypatch, database, use_duckdb):
    source, id_column = _chart_source(tmp_path, database)
    if not use_duckdb:
        monkeypatch.setattr(source, "resolve_bucket_directory", lambda _table: None)
        monkeypatch.setattr(source, "resolve_flat_parquet_directory", lambda _table: None)
    loaded = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["map", "map_invasive", "map_nibp"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={id_column: [1, 2]},
        concept_workers=1, verbose=False,
    )

    def values(name, stay):
        frame = loaded[name].data
        assert RANK_COLUMN not in frame.columns
        rows = frame.loc[frame[id_column].eq(stay)].set_index("charttime")[name].dropna()
        return {float(k): float(v) for k, v in rows.items()}

    expected_map = {0.0: 61.0, 1.0: 80.0, 2.0: 75.0}
    expected_invasive = {0.0: 61.0}
    expected_cuff = {0.0: 90.0, 1.0: 80.0, 2.0: 75.0}
    if database == "mimic":
        expected_map[3.0] = 66.0
        expected_invasive[3.0] = 66.0
        expected_cuff[3.0] = 99.0
    assert values("map", 1) == expected_map
    assert values("map", 2) == {0.0: 70.0}
    assert values("map_invasive", 1) == expected_invasive
    assert values("map_nibp", 1) == expected_cuff
    assert values("map_nibp", 2) == {}


def test_selection_rule_and_native_time_buckets():
    frame = pd.DataFrame({
        "stay_id": [1, 1, 1, 1, 2],
        "charttime": pd.to_datetime(["2180-01-01 00:05", "2180-01-01 00:50", "2180-01-01 01:10",
                                     "2180-01-01 01:20", "2180-01-01 00:00"]),
        "map": [70.0, 95.0, 400.0, 85.0, 88.0],
        RANK_COLUMN: [0, 1, 0, 1, 1],
    })
    kept = select_preferred_channel(frame, value_column="map", id_columns=["stay_id"],
                                    time_column="charttime", minimum=0, maximum=250)
    assert RANK_COLUMN not in kept.columns
    assert kept["map"].tolist() == [70.0, 85.0, 88.0]
    numeric = frame.assign(charttime=[0.1, 0.8, 1.2, 1.3, 0.0])
    with pytest.raises(SourceChannelContractError) as raised:
        select_preferred_channel(numeric, value_column="map", id_columns=["stay_id"],
                                 time_column="charttime", time_in_hours=False)
    assert raised.value.reason == "source_channel_unaligned_numeric_time"


def test_partial_channel_declarations_are_refused():
    dictionary = load_dictionary()
    definition = dictionary.get("map")
    sources = list(definition.sources["miiv"])
    sources[1] = type(sources[1])(table="chartevents", sub_var="itemid", ids=[220181])
    with pytest.raises(SourceChannelContractError) as raised:
        validate_sources(definition, sources)
    assert raised.value.reason == "source_channel_undeclared"


def test_map_channels_partition_the_map_sources_per_database():
    dictionary = load_dictionary()
    definition = dictionary.get("map")
    assert definition.source_channel_priority == ["invasive", "invasive_secondary", "nibp"]
    for database in MAIN_DATABASES + ["eicu_demo", "mimic_demo"]:
        sources = definition.sources[database]
        assert validate_sources(definition, sources) is True
        for channel, concept in (("invasive", "map_invasive"), ("nibp", "map_nibp")):
            channels = {"invasive", "invasive_secondary"} if channel == "invasive" else {channel}
            mine = [s for s in sources if s.params.get("source_channel") in channels]
            theirs = dictionary.get(concept).sources[database]
            key = lambda s: (s.table, s.value_var, tuple(sorted(s.ids or ())))  # noqa: E731
            assert sorted(map(key, mine)) == sorted(map(key, theirs)), (database, channel)
        for name in ("map_invasive", "map_nibp"):
            other = dictionary.get(name)
            assert (other.units, other.minimum, other.maximum) == (
                definition.units, definition.minimum, definition.maximum)


def test_mimic_iii_map_items_follow_the_d_items_labels():
    dictionary = load_dictionary()
    invasive = {i for s in dictionary.get("map_invasive").sources["mimic"] for i in s.ids}
    cuff = {i for s in dictionary.get("map_nibp").sources["mimic"] for i in s.ids}
    every = {i for s in dictionary.get("map").sources["mimic"] for i in s.ids}
    assert 6702 in invasive  # 'Arterial BP Mean #2' (477 stays) was missing
    assert {52, 220052, 225312} <= invasive
    assert {456, 220181} <= cuff
    assert 6072 not in every  # 'vision fields' is not a blood pressure
    assert 224 not in every and 224322 not in every  # IABP mean belongs to circulatory support
