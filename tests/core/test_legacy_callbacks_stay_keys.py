"""The public legacy helpers in ``easyicu.callbacks`` keep stays apart.

``gcs``, ``supp_o2``, ``vaso_ind`` and ``susp_inf`` detect the stay key from a
fixed list that lacked SICdb's ``CaseID``. On SICdb frames they then merged on
time alone and carried one case's values into another. Synthetic data only:
case 1 has observations, case 2 has none or only a neutral one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu import callbacks as legacy


def _component(name, values, ids, times):
    return pd.DataFrame({"CaseID": ids, "charttime": times, name: values})


def test_gcs_locf_never_carries_one_case_into_the_next():
    ids, times = [1, 1, 2], [0.0, 1.0, 1.0]
    gcs = legacy.gcs(
        _component("egcs", [4.0, np.nan, np.nan], ids, times),
        _component("vgcs", [5.0, np.nan, np.nan], ids, times),
        _component("mgcs", [6.0, np.nan, np.nan], ids, times),
        valid_win=pd.Timedelta(hours=6),
        set_na_max=False,
    )
    assert "CaseID" in gcs
    assert gcs.loc[gcs["CaseID"] == 2, "gcs"].isna().all()
    assert gcs.loc[gcs["CaseID"] == 1, "gcs"].notna().any()


def test_supp_o2_never_merges_two_cases_at_one_time():
    out = legacy.supp_o2(
        _component("vent_ind", [True], [1], [0.0]),
        _component("fio2", [21.0], [2], [0.0]),
    )
    assert list(out.columns) == ["CaseID", "charttime", "supp_o2"]
    by_case = out.set_index("CaseID")["supp_o2"].astype(bool).to_dict()
    assert by_case == {1: True, 2: False}


def test_vaso_ind_keeps_one_row_per_case():
    out = legacy.vaso_ind(
        dopa_dur=_component("dopa_dur", [5.0], [1], [0.0]),
        norepi_dur=_component("norepi_dur", [0.0], [2], [3.0]),
    )
    assert "CaseID" in out
    assert out.groupby("CaseID").size().to_dict() == {1: 1, 2: 1}
