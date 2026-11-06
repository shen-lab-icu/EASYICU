"""Each derived-threshold class decides an exact boundary as the criterion reads.

One boundary case per class (lead-r13 float audit groups A-F), loaded as
float32 and as float64: both loads give the same result, and it is the
clinical one. Group G (hour windows, the 12 h record gap) is covered in
``test_decimal_thresholds.py`` and the urine receipt tests.
"""

import numpy as np
import pandas as pd
import pytest

from easyicu.callbacks import sofa_cardio, sofa_liver, sofa_renal, sofa_resp
from easyicu.scores.aki_profiles import apply_aki_profile
from easyicu.scores.circ_failure import circ_failure_event
from easyicu.scores.kdigo_aki import _calc_aki_stage_creat, _calc_aki_stage_uo
from easyicu.scores.sofa2 import sofa2_cardio, sofa2_renal

LOADS = [np.float32, np.float64]
SICDB = "SICDB_NATIVE_KDIGO_AKI_168_44A27CC8"
AUMC = "AUMC_LEGACY_ACUTE_RENAL_FAILURE_8906394D"


def series(values, dtype):
    return pd.Series(np.array(values, dtype=dtype).astype(float))


@pytest.mark.parametrize("dtype", LOADS)
def test_a_kdigo_creatinine_rise_and_ratio_include_the_boundary(dtype):
    nan = np.nan
    stage = _calc_aki_stage_creat(
        series([1.4, 1.2, 4.0, 1.39], dtype),
        series([1.1, nan, 3.7, 1.1], dtype),
        series([nan, 0.8, nan, nan], dtype),
    )
    assert stage.tolist() == [1, 1, 3, 0]


@pytest.mark.parametrize("dtype", LOADS)
def test_b_kdigo_urine_rate_excludes_the_boundary(dtype):
    stage = _calc_aki_stage_uo(
        series([0.5, 0.4, 0.4, 0.4], dtype),
        series([0.5, 0.5, 0.4, 0.0], dtype),
        series([0.5, 0.5, 0.3, 0.3], dtype),
    )
    assert stage.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("dtype", LOADS)
def test_c_sicdb_profile_keeps_its_strict_rise_and_ratio(dtype):
    crea = pd.DataFrame({
        "CaseID": [1, 1, 2, 2, 3, 3],
        # The 1.5x pair sits 72 h apart, outside the 48 h rise window.
        "Offset": [0, 24 * 3600, 0, 72 * 3600, 0, 72 * 3600],
        "crea": np.array([0.8, 1.1, 0.8, 1.2, 0.8, 1.25], dtype=dtype).astype(float),
    })
    result = apply_aki_profile(SICDB, crea_df=crea, id_col="CaseID",
                               time_col="Offset", time_unit="seconds")
    assert result.set_index("CaseID")["aki_stage_source_native"].to_dict() == {1: 0, 2: 0, 3: 1}


@pytest.mark.parametrize("dtype", LOADS)
def test_c_aumc_profile_keeps_its_strict_threefold_ratio(dtype):
    crea = pd.DataFrame({
        "admissionid": [1, 1, 2, 2],
        "time": [-60, 60, -60, 60],
        "crea": np.array([0.7, 2.1, 0.7, 2.11], dtype=dtype).astype(float),
    })
    result = apply_aki_profile(AUMC, crea_df=crea, id_col="admissionid",
                               time_col="time", time_unit="minutes")
    flags = result.set_index("admissionid")["acute_renal_failure_source_native"]
    assert flags.astype(bool).to_dict() == {1: False, 2: True}


@pytest.mark.parametrize("dtype", LOADS)
def test_d_sofa1_boundaries(dtype):
    pafi = series([60.0], dtype) / series([0.15], dtype)
    assert sofa_resp(pd.concat([pafi, series([299.99], dtype)], ignore_index=True)).tolist() == [0, 2]
    assert sofa_liver(series([1.2, 1.19, 2.0, 12.0], dtype)).tolist() == [1, 0, 2, 4]
    assert sofa_renal(series([1.2, 5.0, 4.99], dtype)).tolist() == [1, 4, 3]
    cardio = sofa_cardio(series([80, 80, 80], dtype),
                         norepi60=series([0.1, 0.11, 0.0], dtype),
                         dopa60=series([0.0, 0.0, 5.0], dtype))
    assert cardio.tolist() == [3, 4, 2]


@pytest.mark.parametrize("dtype", LOADS)
def test_e_sofa2_boundaries(dtype):
    cardio = sofa2_cardio(series([80, 80, 80], dtype),
                          norepi60=series([0.1, 0.2, 0.3], dtype),
                          epi60=series([0.1, 0.2, 0.0], dtype))
    assert cardio.tolist() == [2, 3, 3]
    renal = sofa2_renal(series([1.2, 1.21, 3.5], dtype))
    assert renal.tolist() == [0, 1, 2]


@pytest.mark.parametrize("dtype", LOADS)
def test_f_circulatory_failure_includes_the_level3_boundary(dtype):
    def level(rate):
        return circ_failure_event(3.0, 60.0, norepi_rate=float(np.array(rate, dtype=dtype)))
    assert [level(0.1), level(0.099), level(0.0)] == [3, 2, 1]
