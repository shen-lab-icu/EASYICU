"""Thresholds the lead-r13 final review found outside the helpers (A8).

Each case sits on an exact decimal boundary, loaded as float32 and as float64:
both loads give the clinical answer, and the operator is the criterion's own
(strict ``>`` for SOFA-2 liver and the RRT creatinine gate, inclusive window
for the MIMIC-III creatinine hours, strict ``<`` for the AUMC 72 h death
window).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.callbacks import sofa2_liver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_rrt_criteria
from easyicu.scores.aki_profiles import apply_aki_profile
from easyicu.table import ICUTable

LOADS = [np.float32, np.float64]
MIMIC_III = "MIMIC_III_MIT_LCP_KDIGO_D20B49A7"


def series(values, dtype):
    return pd.Series(np.array(values, dtype=dtype).astype(float))


@pytest.mark.parametrize("dtype", LOADS)
def test_public_sofa2_liver_scores_strictly_above_each_cut(dtype):
    # A converted 1.2 mg/dL (e.g. 20.52 umol/L / 17.1) lands a hair above 1.2.
    converted = 20.52 / 17.1
    bili = series([1.2, converted, 1.21, 3.0, 6.0, 12.0, 12.01, np.nan], dtype)
    # Missing bilirubin keeps this public callback's existing score of 0.
    assert sofa2_liver(bili).tolist() == [0, 0, 1, 1, 2, 3, 4, 0]


def _rrt_tables(crea: list[float]) -> dict[str, ICUTable]:
    n = len(crea)

    def table(name, values):
        return ICUTable(
            pd.DataFrame({"stay_id": list(range(1, n + 1)), "charttime": [0.0] * n, name: values}),
            id_columns=["stay_id"], index_column="charttime", value_column=name,
        )

    return {
        "crea": table("crea", crea),
        "potassium": table("potassium", [6.0] * n),
        "ph": table("ph", [7.4] * n),
        "bicarb": table("bicarb", [24.0] * n),
        "rrt": table("rrt", [False] * n),
        "uo_6h": table("uo_6h", [1.0] * n),
        "uo_12h": table("uo_12h", [1.0] * n),
        "uo_24h": table("uo_24h", [1.0] * n),
    }


@pytest.mark.parametrize("dtype", LOADS)
def test_rrt_creatinine_gate_is_strictly_above_1_2(dtype):
    crea = series([1.2, 106.08 / 88.4, 1.21], dtype).tolist()
    ctx = ConceptCallbackContext(
        concept_name="rrt_criteria", target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=SimpleNamespace(config=SimpleNamespace(name="eicu")),
        patient_ids=None,
    )
    result = _callback_rrt_criteria(_rrt_tables(crea), ctx).data
    assert result.set_index("stay_id")["rrt_criteria"].astype(bool).to_dict() == {
        1: False, 2: False, 3: True,
    }


@pytest.mark.parametrize("dtype", LOADS)
def test_mimic_iii_creatinine_window_keeps_both_derived_hour_bounds(dtype):
    # Minute offsets a hair outside each bound (conversion noise) stay in the
    # inclusive window; a real minute past either bound is dropped.
    minutes = series([-360.0001, -361.0, 0.0, 9720.005, 9721.0], dtype)
    crea = pd.DataFrame({
        "icustay_id": [1, 2, 3, 4, 5],
        "charttime": minutes.to_numpy(),
        "crea": [1.0] * 5,
    })
    result = apply_aki_profile(MIMIC_III, crea_df=crea, id_col="icustay_id",
                               time_col="charttime", time_unit="minutes")
    assert sorted(result["icustay_id"].unique().tolist()) == [1, 3, 4]


@pytest.mark.parametrize("dtype", LOADS)
def test_aumc_death_window_excludes_exactly_72_hours_after_discharge(dtype):
    # Loader minutes come from millisecond clocks; 72 h plus or minus division
    # noise is still 72 h, and 72 h is not "< 72 h".
    dis = 1000.0
    dod = series([dis + 4320.0, dis + 4320.0 - 1e-7, dis + 4319.0], dtype)
    frame = pd.DataFrame({
        "admissionid": [1, 2, 3],
        "death": [dis] * 3,
        "dateofdeath": dod.to_numpy(),
    })
    source = SimpleNamespace(
        callback="aumc_death", index_var="dateofdeath", value_var="dischargedat",
        sub_var=None, unit_var=None, table=None, ids=None,
    )
    out = _apply_callback(frame, source, "death").set_index("admissionid")["death"]
    assert out.tolist() == [False, False, True]
