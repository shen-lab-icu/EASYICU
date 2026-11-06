"""GCS and its airway windows follow ricu 0.6.3 (synthetic rows only).

ricu ``gcs``: LOCF of egcs/vgcs/mgcs/tgcs within ``valid_win = hours(6L)``;
only TRUE ``ett_gcs`` windows, expanded with ``expand(aggregate = "any")`` onto
the hourly grid from the start cell through ``re_time(start + dur)`` inclusive
(``re_time`` floors), are merged onto the component rows; ``sed_impute =
"max"`` sets tgcs to 15 there; ``set_na_max`` fills 4/5/6 and then 15.
The expected maps below are ricu's own output for the same tables (see
tests/support/gcs_ricu.py).
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.window_expansion import expand_windows_inclusive_any
from tests.support.gcs_ricu import airway_windows, component, run_gcs

pytestmark = [pytest.mark.ricu_parity, pytest.mark.clinical_conformance]


def test_a_six_hour_minutes_window_marks_its_hours_only():
    # eICU-like: RASS <= -3 at hour 0 (TRUE), RASS > -3 at hour 10 (FALSE),
    # both ts_to_win_tbl(mins(360L)).
    got = run_gcs(
        component("egcs", [0, 2, 8, 11], [2, 3, 4, 1]),
        component("mgcs", [0, 11], [4, 5]),
        component("vgcs", [0, 6, 7, 11], [1, 1, 2, 1]),
        airway_windows([0, 10], [360, 360], [True, False]),
    )
    # ricu: hours 0..6 sedated (15); hour 7 = LOCF e3 + m NA->6 + v2; hour 8
    # e4 + 6 + v2; hour 11 inside the FALSE window is not imputed.
    assert got == {0.0: 15.0, 2.0: 15.0, 6.0: 15.0, 7.0: 11.0, 8.0: 12.0, 11.0: 7.0}


def test_a_one_minute_window_covers_only_its_own_hour():
    # MIIV/MIMIC-like: ts_to_win_tbl(mins(1L)) at hour 3.
    got = run_gcs(
        component("egcs", [3, 4, 5], [1, 2, 3]),
        component("mgcs", [3, 4, 5], [3, 3, 3]),
        component("vgcs", [3, 4, 5], [1, 1, 1]),
        airway_windows([3], [1], [True]),
    )
    assert got == {3.0: 15.0, 4.0: 6.0, 5.0: 7.0}


def test_the_window_grid_is_ricu_expand_with_any():
    frame = pd.DataFrame({"id": [1, 1, 1], "t": [0.0, 10.0, 12.0], "v": [True, False, True]})
    cells = expand_windows_inclusive_any(
        frame, idx_col="t", id_cols=["id"], value_col="v",
        dur_hours=pd.Series([6.0, 6.0, 1.5]),
    )
    # ricu expand(win_tbl(dur = 360, 360, 90 mins), aggregate = "any").
    assert cells["t"].tolist() == [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
    assert cells["v"].tolist() == [True] * 7 + [False, False, True, True] + [False] * 3
    assert cells["v"].dtype == bool


def test_a_window_closing_before_admission_reaches_hour_zero():
    # ricu expand(): x[end < 0, end := 0]. ricu 0.6.3 on win_tbl(start = -10 h,
    # -3 h; dur = 360, 1 mins; TRUE, FALSE) gives hours -10..0, all TRUE.
    frame = pd.DataFrame({"id": [1, 1], "t": [-10.0, -3.0], "v": [True, False]})
    cells = expand_windows_inclusive_any(
        frame, idx_col="t", id_cols=["id"], value_col="v",
        dur_hours=pd.Series([6.0, 1.0 / 60.0]),
    )
    assert cells["t"].tolist() == [float(h) for h in range(-10, 1)]
    assert cells["v"].all()
