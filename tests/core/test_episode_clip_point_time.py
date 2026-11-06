"""A point row keeps its own time on every episode-window path (delta4 D4-POINT-TIME).

The MIMIC window path (``window_expansion.clip_episode_windows``) clipped the
start of every kept row to the episode's lower bound. A frame whose
``dur_var`` column is all missing (a point source concatenated with an empty
window source) took that path, so a point a rounding step below the bound,
which the inclusive decimal lower bound admits, was moved onto the bound,
while the ordinary point quarantine keeps it where it is. Synthetic rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.concept import _quarantine_rows_outside_icu_episode
from easyicu.table.duration import UNIT_HOURS, set_dur_var_unit

_LOWER = 24.0  # allowed pre-admission hours


def _frame(with_dur: bool) -> pd.DataFrame:
    below = np.nextafter(-_LOWER, -np.inf)  # one ulp below the lower bound
    frame = pd.DataFrame({"icustay_id": [1, 1, 1], "charttime": [below, -3.0, 5.0], "v": [1.0, 2.0, 3.0]})
    if with_dur:
        frame["dur_var"] = np.nan
        set_dur_var_unit(frame, UNIT_HOURS)
    return frame


def _clip(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    return _quarantine_rows_outside_icu_episode(
        frame, event_hours=frame["charttime"], upper_hours=pd.Series(100.0, index=frame.index),
        allowed_pre_hours=_LOWER, db_label=label,
    )


def test_a_point_row_keeps_its_time_on_the_window_path():
    original = _frame(with_dur=True)["charttime"].tolist()
    out = _clip(_frame(with_dur=True), "MIMIC")
    assert out["charttime"].tolist() == original


def test_the_window_path_and_the_point_path_publish_the_same_point_times():
    windows = _clip(_frame(with_dur=True), "MIMIC")
    points = _clip(_frame(with_dur=False), "eICU")
    assert windows["charttime"].tolist() == points["charttime"].tolist()


def test_a_real_window_is_still_clipped_to_the_bound():
    frame = pd.DataFrame({"icustay_id": [1], "charttime": [-30.0], "dur_var": [10.0], "v": [1.0]})
    set_dur_var_unit(frame, UNIT_HOURS)
    out = _clip(frame, "MIMIC")
    assert out["charttime"].tolist() == [-_LOWER]
    assert out["dur_var"].tolist() == [4.0]
