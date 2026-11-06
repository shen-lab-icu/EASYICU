"""A batch whose charted rates are all whole numbers is rescaled like any other.

Found by the eICU memory diagnosis (lead-r13). Once float32 down-casting was
removed, ``pd.to_numeric`` kept an int64 value column whenever every
``drugrate`` in a batch was a whole number. The in-place ``/hr -> /min``
division then failed under pandas 3, failing ``circ_failure``
(``adh_rate``) for that batch only.
"""
import numpy as np
import pandas as pd
import pytest

from easyicu.utils.callback_utils import eicu_rate_units_callback


def _normalize_units():
    callback = eicu_rate_units_callback(ml_to_mcg=None, mcg_to_units=None)
    cells = dict(zip(callback.__code__.co_freevars, callback.__closure__ or ()))
    return cells["_normalize_units"].cell_contents


@pytest.mark.parametrize("dtype", ["int64", "Int64", "float64"])
def test_integer_rate_column_is_rescaled_not_rejected(dtype):
    frame = pd.DataFrame({"rate": pd.array([60, 120, 3], dtype=dtype),
                          "unit": ["units/hr", "units/hr", "units/min"]})
    out = _normalize_units()(frame, "rate", "unit")
    assert out["rate"].dtype == np.float64
    assert out["rate"].tolist() == [1.0, 2.0, 3.0]
    assert out["unit"].tolist() == ["units/min"] * 3


def test_non_numeric_rates_still_fail_loudly():
    frame = pd.DataFrame({"rate": ["sixty"], "unit": ["units/hr"]})
    with pytest.raises((ValueError, TypeError)):
        _normalize_units()(frame, "rate", "unit")
