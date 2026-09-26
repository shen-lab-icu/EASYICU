"""Exercise signed irrigation through complete Python and DuckDB concept loading."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.signed_volume import bound_signed_total
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io.ts_utils import _fast_groupby_agg


DICTIONARY = Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json"


def fixture_source(tmp_path, database, string_timestamps=False):
    id_col = "stay_id" if database == "miiv" else "icustay_id"
    admitted = pd.Timestamp("2180-01-01 00:15:00")
    rows = [
        (1, 5, 226559, 100.), (1, 15, 227488, 500.), (1, 25, 227489, 550.),
        (1, 65, 227488, 9000.), (1, 65, 227489, 9200.),
        (1, 125, 227488, 500.), (1, 125, 227489, 500.),
        (1, 185, 227488, 700.), (1, 185, 227489, 500.),
        (1, 245, 226559, 3500.), (1, 245, 226560, 3500.),
        (1, 305, 226559, -15.), (1, 365, 226559, np.nan),
        (1, 490, 226559, 70.), (1, 520, 226559, 80.),
        (2, 5, 226559, 40.),
    ]
    frame = pd.DataFrame(rows, columns=[id_col, "offset", "itemid", "value"])
    frame["charttime"] = admitted + pd.to_timedelta(frame.pop("offset"), unit="m")
    frame["valueuom"] = "mL"
    frame["subject_id"] = frame[id_col]
    frame["hadm_id"] = frame[id_col]
    table_dir = tmp_path / "outputevents"
    table_dir.mkdir()
    persisted = frame.copy()
    if string_timestamps:
        persisted["charttime"] = persisted.charttime.astype(str)
    persisted.to_parquet(table_dir / "part-0.parquet", index=False)
    stays = pd.DataFrame({id_col: [1, 2], "subject_id": [1, 2], "hadm_id": [1, 2],
                          "intime": [admitted]*2,
                          "outtime": [admitted+pd.Timedelta(hours=48)]*2})
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)
    source = ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)
    return source, frame, admitted, id_col


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("grid", [None, pd.Timedelta(hours=1), pd.Timedelta(minutes=30)])
@pytest.mark.parametrize("use_duckdb", [False, True])
@pytest.mark.parametrize("string_timestamps", [False, True])
def test_signed_net_full_loader_matches_independent_total(
    tmp_path, monkeypatch, database, grid, use_duckdb, string_timestamps,
):
    source, raw, admitted, id_col = fixture_source(tmp_path, database, string_timestamps)
    import easyicu.datasource as datasource_module

    calls = []
    original = datasource_module.load_bucketed_table_aggregated

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(kwargs)
        return result

    monkeypatch.setattr(datasource_module, "load_bucketed_table_aggregated", capture)
    if not use_duckdb:
        monkeypatch.setattr(source, "resolve_bucket_directory", lambda _table: None)
        monkeypatch.setattr(source, "resolve_flat_parquet_directory", lambda _table: None)
    resolver = ConceptResolver(ConceptDictionary.from_json(DICTIONARY))
    result = resolver.load_concepts(
        ["urine"], source, merge=False, r_compatible=False, interval=grid,
        verbose=False, concept_workers=1, patient_ids={id_col: [1, 2]},
    )["urine"].data

    # Independent expected signed totals at native time or requested ICU grid.
    expected = raw.copy()
    expected["charttime"] = (expected.charttime-admitted).dt.total_seconds()/3600
    if grid is not None:
        step = grid.total_seconds()/3600
        expected["charttime"] = np.floor(expected.charttime/step)*step
    expected["urine"] = np.where(expected.itemid.eq(227488) & expected.value.gt(0),
                                  -expected.value, expected.value)
    expected = expected.groupby([id_col, "charttime"], as_index=False).urine.sum(min_count=1)
    expected = expected.loc[expected.urine.between(0, 5000)]
    columns = [id_col, "charttime", "urine"]
    pd.testing.assert_frame_equal(
        result[columns].sort_values(columns[:2]).reset_index(drop=True),
        expected[columns].sort_values(columns[:2]).reset_index(drop=True),
        check_dtype=False,
    )
    assert len(calls) == int(use_duckdb and grid is not None)
    if calls:
        assert calls[0]["value_min"] is None and calls[0]["value_max"] is None
        assert calls[0]["interval_minutes"] == grid.total_seconds()/60
    if grid == pd.Timedelta(hours=1):
        assert result.loc[result[id_col].eq(1), "urine"].tolist() == [150., 200., 0., 150.]


def test_invalid_signed_totals_are_not_clipped_to_zero():
    result = bound_signed_total(pd.DataFrame({"urine": [-1., 0., 5000., 5001., np.nan, np.inf]}),
                                "urine", 0, 5000)
    assert result.urine.tolist() == [0., 5000.]
    assert result.attrs["easyicu_signed_sum_bounds"] == {
        "policy": "SIGNED_SUM_THEN_BOUNDS_V1", "total_rows": 6,
        "below_minimum_rows": 1, "above_maximum_rows": 1, "nonfinite_rows": 2,
        "observed_zero_rows": 1, "retained_rows": 2,
    }


def test_signed_volume_definition_changes_disk_cache_identity():
    current = ConceptDictionary.from_json(DICTIONARY)
    signature = ConceptResolver(current).dictionary_signature
    for database in ("miiv", "mimic", "mimic_demo"):
        for source in current["urine"].sources[database]:
            assert source.params.pop("bounds_stage") == "post_sum"
            assert source.params.pop("bounds_contract") == "SIGNED_SUM_THEN_BOUNDS_V1"
    assert ConceptResolver(current).dictionary_signature != signature


@pytest.mark.parametrize("ids,times", [
    ([1, 1, 1, 2], [-.5, 0., .5, .5]),
    ([1.25, 1.75, 2., 2.], [0., 0., 0., 1.]),
    ([2**62, 2**62, 0, 0], [0, 2, 0, 2]),
    ([1, 1, 2, 2], [-(2**62), 2**62, -(2**62), 2**62]),
])
def test_packed_aggregation_cannot_alias_fractional_or_overflowing_keys(ids, times):
    frame = pd.DataFrame({"stay_id": ids, "time": times, "value": [1., 2., 3., 4.]})
    actual = _fast_groupby_agg(frame, ["stay_id", "time"], {"value": "sum"})
    expected = frame.groupby(["stay_id", "time"], sort=False, as_index=False).value.sum()
    pd.testing.assert_frame_equal(actual, expected)
