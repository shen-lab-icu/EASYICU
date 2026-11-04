"""vaso_ind covers positive overlaps with [start, start + duration).

2026-11-03 extraction audit round 2 (R2-02): the ricu-style expansion used
the duration as the end time, so an episode starting at hour 10 that lasted
3 hours produced no indicator hour at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from easyicu.concept.callbacks import _callback_vaso_ind
from easyicu.table import ICUTable


def _duration_table(name, rows):
    frame = pd.DataFrame(rows, columns=["stay_id", "charttime", name])
    return ICUTable(data=frame, id_columns=["stay_id"], index_column="charttime", value_column=name)


def test_late_short_episodes_cover_their_own_hours():
    tables = {
        "norepi_dur": _duration_table("norepi_dur", [(1, 10.0, 3.0), (2, 0.0, 0.0)]),
        "epi_dur": _duration_table("epi_dur", [(1, 10.0, 1.5), (3, 5.0, 2.0)]),
    }
    ctx = SimpleNamespace(interval=None, resolver=None, data_source=None, patient_ids=None)

    out = _callback_vaso_ind(tables, ctx).data

    hours = {k: sorted(g["charttime"].tolist()) for k, g in out.groupby("stay_id")}
    assert hours == {1: [10.0, 11.0, 12.0], 3: [5.0, 6.0]}
    assert out["vaso_ind"].all()
