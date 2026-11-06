"""A window concept loaded alone through the public API declares its duration unit.

eICU ``ett_gcs`` is built from RASS by the resolver's ``fwd_concept`` +
``ts_to_win_tbl(mins(360L))`` branch, which writes ``dur_var`` in minutes. It
never declared that unit, so ``easyicu.load_concepts(["ett_gcs"])`` refused the
frame when it expanded the windows (``DurationUnitError``). Extraction loads it
in module groups and never took that path, so only the public API failed
(lead-r13 G2 public-API sweep). Synthetic rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import easyicu
from tests.support.eicu_source import _eicu_source

_PAIR = "Sedation Scale/Score/Goal"


def _rass(offset, score):
    return [
        (offset, "Scores", _PAIR, "Sedation Scale", "RASS"),
        (offset, "Scores", _PAIR, "Sedation Score", score),
    ]


def test_eicu_ett_gcs_loads_alone_and_opens_a_six_hour_window(tmp_path):
    rows = [*_rass(5, "-4"), *_rass(605, "-1")]  # deep sedation, then light
    nurse = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue",
    ])
    nurse["nursingchartentryoffset"] = nurse["nursingchartoffset"]
    nurse["nursingchartid"] = np.arange(len(nurse), dtype=float)
    _eicu_source(tmp_path, "eicu", nursecharting=nurse)

    frame = easyicu.load_concepts(
        ["ett_gcs"], database="eicu", data_path=str(tmp_path), merge=True, verbose=False,
    )

    # RASS <= -3 at minute 5 opens a 360-minute window on its hour; as in ricu
    # expand(), it covers hour 0 through floor(0 + 6 h) inclusive. RASS -1 at
    # minute 605 is a FALSE window of the same length (r13gcs).
    assert frame["charttime"].tolist() == [float(h) for h in [*range(7), *range(10, 17)]]
    assert frame["ett_gcs"].tolist() == [True] * 7 + [False] * 7
