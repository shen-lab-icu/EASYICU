"""Order-insensitive aggregates read Parquet without the full-table sort.

``parquet_relation`` restores logical source order with an ORDER BY over the
whole relation; a cohort semi-join outside it then runs after a full sort
(HiRID observations: minutes and over 20 GiB for one stay, lead-r13).
MAX(...) GROUP BY readers take ``ordered=False`` and must get the same answer.
"""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.concept.callback_apply import hirid_observation_read_exprs
from easyicu.storage_layout import ORDINAL, parquet_relation


def _write(path: Path, rows, *, ordinal=None):
    columns = {
        "patientid": [r[0] for r in rows],
        "variableid": [r[1] for r in rows],
        "datetime": [r[2] for r in rows],
    }
    if ordinal is not None:
        columns[ORDINAL] = ordinal
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def _max_per_patient(relation):
    with duckdb.connect() as conn:
        conn.execute("CREATE TEMP TABLE cohort AS SELECT 1 AS patientid UNION ALL SELECT 3")
        return conn.execute(
            f"""SELECT o.patientid, MAX(o.datetime) AS last
                FROM {relation} o
                WHERE o.variableid IN (110, 200)
                  AND o.patientid IN (SELECT patientid FROM cohort)
                GROUP BY o.patientid ORDER BY o.patientid"""
        ).fetchall()


@pytest.mark.parametrize("with_ordinal", [False, True])
def test_unordered_relation_has_no_sort_and_the_same_aggregate(tmp_path, with_ordinal):
    files = [
        _write(tmp_path / "observations" / "1.parquet",
               [(1, 110, 5), (1, 200, 9), (2, 110, 7)],
               ordinal=[0, 1, 2] if with_ordinal else None),
        _write(tmp_path / "observations" / "2.parquet",
               [(3, 110, 4), (1, 300, 50), (3, 200, 8)],
               ordinal=[3, 4, 5] if with_ordinal else None),
    ]
    ordered = parquet_relation(files)
    unordered = parquet_relation(files, ordered=False)
    assert "ORDER BY" in ordered
    assert "ORDER BY" not in unordered
    if with_ordinal:
        assert f'EXCLUDE("{ORDINAL}")' in unordered
    assert _max_per_patient(unordered) == _max_per_patient(ordered) == [(1, 9), (3, 8)]


def test_hirid_timing_relation_is_unordered(tmp_path):
    _write(tmp_path / "observations" / "1.parquet", [(1, 110, 5)])
    expressions = hirid_observation_read_exprs(tmp_path)
    assert expressions is not None
    fast, safe, layout = expressions
    assert layout == "observations"
    assert "ORDER BY" not in fast and "ORDER BY" not in safe
