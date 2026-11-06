"""ICU episode window and derived plausibility bounds: decimal thresholds.

lead-r13 delta2 THRESHOLD_MISSED. The event hour and the window bound are both
derived (an offset divided by 3,600; a discharge hour plus 24 h), so an event
exactly on the bound could land an ulp past it and be dropped. Both bounds are
now inclusive and decided with the decimal-threshold helpers. pandas (the
window object, the point quarantine, MIMIC point rows) and the DuckDB
predicate apply the same rule with the same bits. eGFR and the other row-wise
derived indices bound their computed value the same way. Synthetic values only.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from easyicu.concept import _quarantine_rows_outside_icu_episode
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_egfr
from easyicu.table import ICUTable
from easyicu.utils.icu_window import IcuEpisodeWindow, IcuWindowSql, outside_episode
from easyicu.utils.thresholds import ge, gt, le, lower_limit, lt, upper_limit


def _noisy_discharge_seconds(count: int = 3) -> list[int]:
    """Discharge offsets whose bound ``d/3600 + 24`` sits an ulp *below*
    the hour of an event charted exactly 24 h later, ``(d + 86400)/3600``."""

    found = []
    for d in range(3600, 3600 * 400, 7):
        if (d + 86400) / 3600 > d / 3600 + 24:
            found.append(d)
            if len(found) == count:
                return found
    raise AssertionError("no float-noisy discharge offset found")


def test_the_limits_are_the_helpers_own_comparisons():
    values = np.array([-168.0, -24.0, 0.0, 1.0, 24.0, 123.456, 8784.0])
    probes = np.concatenate([values, np.nextafter(values, np.inf), np.nextafter(values, -np.inf)])
    for t in values:
        lo, hi = lower_limit(t), upper_limit(t)
        assert ((probes >= lo) == ge(probes, t)).all()
        assert ((probes < lo) == lt(probes, t)).all()
        assert ((probes <= hi) == le(probes, t)).all()
        assert ((probes > hi) == gt(probes, t)).all()


def test_an_event_exactly_on_a_derived_upper_bound_is_kept():
    discharges = _noisy_discharge_seconds()
    bound = pd.Series([d / 3600 + 24 for d in discharges])
    event = pd.Series([(d + 86400) / 3600 for d in discharges])
    assert (event > bound).all()  # the plain comparison dropped them
    assert not outside_episode(event, lower_hours=-24.0, upper_hours=bound.to_numpy()).any()
    # A minute past the bound is still outside; the tolerance is about a
    # second here (1e-6 of the bound's hours).
    late = event + 1 / 60
    assert outside_episode(late, lower_hours=-24.0, upper_hours=bound.to_numpy()).all()


def test_an_event_exactly_on_the_derived_lower_bound_is_kept():
    # 24 h before admission, derived from source offsets in seconds.
    admit = 3600 * 7 + 13
    event = pd.Series([(admit - 86400 - admit) / 3600, (0.1 * 3 - 24.3)])
    assert outside_episode(event, lower_hours=-24.0, upper_hours=np.array([100.0, 100.0])).sum() == 0
    early = pd.Series([-24.0 - 1 / 60])
    assert outside_episode(early, lower_hours=-24.0, upper_hours=np.array([100.0])).all()


def _probe_hours(bounds: np.ndarray) -> np.ndarray:
    hi = upper_limit(bounds)
    lo = lower_limit(np.float64(-24.0))
    return np.concatenate([
        bounds, hi, np.nextafter(hi, np.inf), np.nextafter(hi, -np.inf), bounds + 1 / 60,
        np.full(len(bounds), lo), np.full(len(bounds), np.nextafter(lo, -np.inf)),
        np.full(len(bounds), -24.0), np.full(len(bounds), np.nan), np.zeros(len(bounds)),
    ])


@pytest.mark.parametrize("per_stay", [True, False])
@pytest.mark.parametrize("keep_post_discharge", [False, True])
def test_sql_and_pandas_keep_the_same_rows(per_stay, keep_post_discharge):
    discharges = _noisy_discharge_seconds(4)
    bounds = np.array([d / 3600 + 24 for d in discharges])
    stays = np.arange(1, len(bounds) + 1)
    default = 8784.0
    window = IcuEpisodeWindow(
        pre_admission_hours=24.0,
        default_upper_hours=default,
        upper_by_stay=pd.Series(bounds, index=stays) if per_stay else None,
        keep_post_discharge=keep_post_discharge,
    )
    hours = _probe_hours(bounds if per_stay else np.full(len(bounds), default))
    stay_keys = np.tile(stays, len(hours) // len(stays))
    # A stay without a per-stay bound takes the default upper on both paths.
    hours = np.append(hours, [default, upper_limit(default), np.nextafter(upper_limit(default), np.inf)])
    stay_keys = np.append(stay_keys, [99, 99, 99])
    frame = pd.DataFrame({"stay_id": stay_keys, "hours": hours, "row": np.arange(len(hours))})

    kept_pandas = set(
        frame.loc[~window.outside(frame["hours"], frame["stay_id"]).to_numpy(), "row"]
    )
    con = duckdb.connect()
    try:
        con.register("events", frame)
        sql = IcuWindowSql(con, window)
        rows = con.execute(
            "SELECT row FROM events WHERE TRUE" + sql.condition("events.hours", "events.stay_id")
        ).fetchall()
        sql.close()
    finally:
        con.close()
    kept_sql = {r[0] for r in rows}
    assert kept_sql == kept_pandas
    # The noisy on-bound events are kept on both paths.
    assert set(range(len(bounds))) <= kept_sql
    if not keep_post_discharge:
        # A minute late is dropped; so is the first value past the limit.
        late = set(range(4 * len(bounds), 5 * len(bounds)))
        assert not (late & kept_sql)
        past_limit = set(range(2 * len(bounds), 3 * len(bounds)))
        assert not (past_limit & kept_sql)


def test_the_pandas_point_quarantine_uses_the_same_rule():
    discharges = _noisy_discharge_seconds()
    bounds = pd.Series([d / 3600 + 24 for d in discharges])
    event = pd.Series([(d + 86400) / 3600 for d in discharges])
    hours = pd.concat([event, event + 1 / 60, pd.Series([np.nan, -24.0])], ignore_index=True)
    upper = pd.concat([bounds, bounds, pd.Series([10.0, 10.0])], ignore_index=True)
    frame = pd.DataFrame({"stay": np.arange(len(hours)), "charttime": hours})
    out = _quarantine_rows_outside_icu_episode(
        frame, event_hours=hours, upper_hours=upper, allowed_pre_hours=24.0, db_label="eICU",
    )
    expected = ~outside_episode(hours, lower_hours=-24.0, upper_hours=upper.to_numpy())
    pd.testing.assert_frame_equal(out, frame.loc[expected.to_numpy()])
    assert len(out) == len(discharges) + 2


def _ctx() -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name="egfr", target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=None, patient_ids=None,
    )


def _egfr(crea: list[float], age: float, sex: str) -> pd.Series:
    n = len(crea)
    tables = {
        "crea": ICUTable(
            pd.DataFrame({"stay_id": list(range(1, n + 1)), "charttime": [1.0] * n, "crea": crea}),
            id_columns=["stay_id"], index_column="charttime", value_column="crea",
        ),
        "age": ICUTable(
            pd.DataFrame({"stay_id": list(range(1, n + 1)), "age": [age] * n}),
            id_columns=["stay_id"], index_column=None, value_column="age",
        ),
        "sex": ICUTable(
            pd.DataFrame({"stay_id": list(range(1, n + 1)), "sex": [sex] * n}),
            id_columns=["stay_id"], index_column=None, value_column="sex",
        ),
    }
    return _callback_egfr(tables, _ctx()).data.set_index("stay_id")["egfr"]


def _crea_for(egfr: float, age: float) -> float:
    # Male, creatinine below kappa: egfr = 142 * (crea/0.9)**-0.302 * 0.9938**age.
    return 0.9 * (egfr / (142.0 * 0.9938**age)) ** (1 / -0.302)


def test_egfr_bounds_are_inclusive_decimal_thresholds():
    age = 18.0
    # Search creatinine values whose computed eGFR lands within float noise
    # just above 200 (the plain bound dropped them) and one clearly above.
    crea0 = _crea_for(200.0, age)
    candidates = [crea0 * (1 - k * 1e-16) for k in range(-50, 50)]
    computed = _egfr(candidates, age, "M")
    noisy = [c for c, v in zip(candidates, computed.reindex(range(1, 101)).to_numpy()) if 200.0 < v]
    assert noisy, "no creatinine gives an eGFR a hair above 200"
    out = _egfr(noisy + [_crea_for(200.5, age)], age, "M")
    assert len(out) == len(noisy)
    assert (out <= upper_limit(200.0)).all()
