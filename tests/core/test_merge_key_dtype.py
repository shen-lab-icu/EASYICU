"""merge_asof key-dtype robustness in driving_pressure().

Reproduces the AUMC ventilator failure: plateau and peep came back with
different numeric time dtypes (one float32, one float64), so the merge_asof
inside driving_pressure() died with
``pandas.errors.MergeError: incompatible merge keys [1] float32 and float64``
and the WHOLE ventilator module (peep, plateau_pres, driving_pres,
driving_pres_controlled, ...) was dropped from the export. merge_asof — unlike
plain pd.merge — is strict about the on-key dtype, so the join keys must be
unified before the call. Minute/second ICU times are exact in float64, so the
coercion is lossless.
"""
import numpy as np
import pandas as pd
import pytest

from easyicu.callbacks import driving_pressure


def test_merge_asof_raises_on_mixed_key_dtype_documents_failure_mode():
    """Guards the assumption the fix relies on: merge_asof IS strict here."""
    plat = pd.DataFrame({"admissionid": np.int64([1, 1]),
                         "t": np.float32([0.0, 60.0]), "plateau_pres": [20.0, 22.0]})
    peep = pd.DataFrame({"admissionid": np.int64([1, 1]),
                         "t": np.float64([0.0, 60.0]), "peep": [5.0, 8.0]})
    with pytest.raises(pd.errors.MergeError):
        pd.merge_asof(plat, peep, on="t", by="admissionid",
                      tolerance=60.0, direction="nearest")


def test_driving_pressure_survives_float32_vs_float64_time():
    """The exact AUMC shape: plateau time float32, peep time float64."""
    plateau = pd.DataFrame({
        "admissionid": np.int64([1, 1, 2]),
        "charttime": np.float32([0.0, 60.0, 0.0]),
        "plateau_pres": [24.0, 26.0, 30.0]})
    peep = pd.DataFrame({
        "admissionid": np.int64([1, 1, 2]),
        "charttime": np.float64([0.0, 60.0, 0.0]),
        "peep": [10.0, 12.0, 8.0]})

    out = driving_pressure(plateau, peep, match_win=pd.Timedelta(hours=1), database="aumc")

    assert "driving_pres" in out.columns
    assert len(out) == 3
    # driving pressure = plateau - peep, matched on the same timestamp
    dp = out.sort_values(["admissionid", "charttime"])["driving_pres"].tolist()
    assert dp == [14.0, 14.0, 22.0]
    # inputs were not mutated in place
    assert str(plateau["charttime"].dtype) == "float32"


def test_driving_pressure_still_works_when_dtypes_already_match():
    plateau = pd.DataFrame({
        "admissionid": np.int64([1, 1]),
        "charttime": np.float64([0.0, 60.0]),
        "plateau_pres": [24.0, 26.0]})
    peep = pd.DataFrame({
        "admissionid": np.int64([1, 1]),
        "charttime": np.float64([0.0, 60.0]),
        "peep": [10.0, 12.0]})
    out = driving_pressure(plateau, peep, match_win=pd.Timedelta(hours=1))
    assert "driving_pres" in out.columns
    assert len(out) == 2
