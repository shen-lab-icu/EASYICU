"""End-to-end tests for the opt-in patient-clustering converter pass.

The streaming per-chunk secondary sort (perf Z2) cannot prune a *scattered*
patient cohort, because a patient's rows land in every chunk. The clustering
pass (`cluster_table_by_patient`) rewrites each shard GLOBALLY sorted by
``(partition_col, patient_id)`` with fine row groups, using DuckDB's
out-of-core sort. These tests convert a synthetic miiv chartevents fixture,
cluster it, and assert on the real on-disk parquet that:

1. no rows are lost and the schema/columns are preserved;
2. shards are globally ordered by (itemid, stay_id);
3. a scattered patient cohort now touches far fewer row groups than before
   clustering (the property that produces the query speedup);
4. `convert_all(cluster_by_patient=True)` / env `EASYICU_CLUSTER_BY_PATIENT`
   wires the pass in;
5. clustering is idempotent (re-running is safe).

Run without ``--run-real``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
import pytest

from easyicu.io.data_converter import (
    ConversionStatus,
    DataConverter,
    DEFAULT_CLUSTER_ROW_GROUP_SIZE,
)

N_PATIENTS = 20_000
ITEMIDS = [220045, 220046, 220047]      # all < first break 220048 -> one shard
ROWS_PER_PAIR = 20                       # 3 * 20000 * 20 = 1.2M rows


def _make_chartevents_csv(path, ordering="shuffled", seed=1):
    rng = np.random.default_rng(seed)
    stays = np.arange(1, N_PATIENTS + 1)
    frames = []
    for it in ITEMIDS:
        rep = np.repeat(stays, ROWS_PER_PAIR)
        frames.append(pd.DataFrame({
            "subject_id": rep, "hadm_id": rep, "stay_id": rep, "itemid": it,
            "charttime": "2020-01-01 00:00:00",
            "value": rng.random(len(rep)).astype("float32"),
        }))
    df = pd.concat(frames, ignore_index=True)
    # shuffle so per-chunk sorting can't accidentally cluster patients
    df = df.sample(frac=1.0, random_state=seed)
    path.write_text(df.to_csv(index=False))
    return len(df)


def _shards(shard_dir):
    return sorted(shard_dir.glob("*.parquet"), key=lambda p: int(p.stem))


def _row_groups_touched(shard_dir, cohort):
    """Row groups whose stay_id [min,max] contains a cohort id, over all shards."""
    con = duckdb.connect()
    cohort_sorted = np.array(sorted(cohort))
    try:
        touched = total = 0
        for f in _shards(shard_dir):
            for lo, hi in con.execute(
                "SELECT CAST(stats_min AS BIGINT), CAST(stats_max AS BIGINT) "
                "FROM parquet_metadata(?) WHERE path_in_schema='stay_id'", [str(f)]
            ).fetchall():
                if lo is None:
                    continue
                total += 1
                i = np.searchsorted(cohort_sorted, lo)
                if i < len(cohort_sorted) and cohort_sorted[i] <= hi:
                    touched += 1
        return touched, total
    finally:
        con.close()


def _shard_is_sorted(shard_dir):
    con = duckdb.connect()
    try:
        for f in _shards(shard_dir):
            rows = con.execute(
                f"SELECT itemid, stay_id FROM read_parquet('{f}')"
            ).fetchall()
            if rows != sorted(rows):
                return False
        return True
    finally:
        con.close()


def _total_rows(shard_dir):
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT count(*) FROM read_parquet('{shard_dir}/*.parquet')"
        ).fetchone()[0]
    finally:
        con.close()


def _convert(tmp_path):
    src = tmp_path / "chartevents.csv"
    _make_chartevents_csv(src)
    conv = DataConverter(str(tmp_path), database="miiv", verbose=False)
    res = conv.convert_all()
    assert res["chartevents.csv"]["status"] == ConversionStatus.COMPLETED
    return conv, tmp_path / "chartevents"


def test_clustering_preserves_rows_and_sorts(tmp_path):
    conv, shard_dir = _convert(tmp_path)
    before_rows = _total_rows(shard_dir)

    summary = conv.cluster_table_by_patient("chartevents", row_group_size=5_000)

    assert summary["shards_clustered"] >= 1
    assert _total_rows(shard_dir) == before_rows          # no rows lost
    assert _shard_is_sorted(shard_dir)                    # (itemid, stay_id) order
    # value column (pinned to string by the converter) still present.
    # DESCRIBE returns (column_name, column_type, ...) -> name is column 0.
    con = duckdb.connect()
    cols = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_shards(shard_dir)[0]}')"
    ).fetchall()]
    con.close()
    assert {"subject_id", "stay_id", "itemid", "value"} <= set(cols)


def test_clustering_prunes_scattered_cohort(tmp_path):
    conv, shard_dir = _convert(tmp_path)
    # 60 of 20,000 stays (0.3%), scattered — the realistic research pattern
    # (random / condition-filtered cohort) that min/max pruning can't touch
    # until the shard is patient-clustered with fine row groups.
    rng = np.random.default_rng(7)
    cohort = set(int(x) for x in rng.choice(
        np.arange(1, N_PATIENTS + 1), 60, replace=False))

    before_touched, before_total = _row_groups_touched(shard_dir, cohort)
    conv.cluster_table_by_patient("chartevents", row_group_size=1_000)
    after_touched, after_total = _row_groups_touched(shard_dir, cohort)

    before_frac = before_touched / before_total
    after_frac = after_touched / after_total
    # Pre-clustering, a shuffled table can't prune a scattered cohort at all.
    assert before_frac > 0.9, before_frac
    # Post-clustering, most row groups are pruned away.
    assert after_frac < 0.4, (after_frac, before_frac)
    assert after_frac < before_frac * 0.5


def test_convert_all_cluster_flag(tmp_path):
    src = tmp_path / "chartevents.csv"
    _make_chartevents_csv(src)
    conv = DataConverter(str(tmp_path), database="miiv", verbose=False)
    res = conv.convert_all(cluster_by_patient=True)
    assert "cluster_by_patient" in res["chartevents"]
    assert res["chartevents"]["cluster_by_patient"]["shards_clustered"] >= 1
    assert _shard_is_sorted(tmp_path / "chartevents")


def test_convert_all_cluster_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_CLUSTER_BY_PATIENT", "1")
    src = tmp_path / "chartevents.csv"
    _make_chartevents_csv(src)
    conv = DataConverter(str(tmp_path), database="miiv", verbose=False)
    res = conv.convert_all()
    assert res["chartevents"]["cluster_by_patient"]["shards_clustered"] >= 1


def test_clustering_is_idempotent(tmp_path):
    conv, shard_dir = _convert(tmp_path)
    rows0 = _total_rows(shard_dir)
    s1 = conv.cluster_table_by_patient("chartevents")
    s2 = conv.cluster_table_by_patient("chartevents")
    assert s1["row_group_size"] == DEFAULT_CLUSTER_ROW_GROUP_SIZE
    assert _total_rows(shard_dir) == rows0
    assert _shard_is_sorted(shard_dir)


def test_cluster_missing_table_raises(tmp_path):
    conv = DataConverter.__new__(DataConverter)
    # minimal init for the method under test
    from pathlib import Path
    conv.data_path = Path(tmp_path)
    conv.database = "miiv"
    conv.parquet_compression = "zstd"
    conv.verbose = False
    with pytest.raises(FileNotFoundError):
        conv.cluster_table_by_patient("nonexistent_table")
