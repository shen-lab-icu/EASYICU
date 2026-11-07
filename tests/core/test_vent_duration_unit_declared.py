"""vent_ind window durations take their declared unit, never nanoseconds.

A unit-less ``pd.to_timedelta`` reads a number as nanoseconds and returns a
non-null value, so the old hours fallback never ran: a 3-hour window became
3 ns and covered only its start hour.  Numbers now take the WinTbl's declared
``dur_var`` unit, an undeclared numeric duration fails closed with reason
``duration_unit_undeclared``, and timedelta durations pass through unchanged.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vent_ind
from easyicu.table import WinTbl
from easyicu.table.duration import (
    DurationUnitError,
    DurationUnitUndeclaredError,
    UNIT_HOURS,
    UNIT_MINUTES,
    duration_series_to_timedelta,
    set_dur_var_unit,
)


def _ctx(name="sic"):
    # SICdb's vent_ind branch hands the WinTbl to the expanding window path.
    return ConceptCallbackContext(
        concept_name="vent_ind",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name=name)),
        patient_ids=None,
    )


def _windows(duration, unit=None):
    frame = pd.DataFrame(
        {
            "CaseID": [1, 2],
            "charttime": [0.0, 10.0],
            "duration": duration,
            "mech_vent": ["invasive", "invasive"],
        }
    )
    if unit is not None:
        set_dur_var_unit(frame, unit)
    return WinTbl(frame, id_vars=["CaseID"], index_var="charttime", dur_var="duration",
                  interval=pd.Timedelta(hours=1))


def _hours_per_stay(result):
    data = result.data
    value = result.value_column or "vent_ind"
    on = data.loc[data[value].fillna(False).astype(bool)]
    return on.groupby("CaseID")[result.index_column].apply(lambda s: sorted(float(v) for v in s)).to_dict()


def test_numeric_hours_are_expanded_as_hours():
    result = _callback_vent_ind({"mech_vent": _windows([3.0, 2.0], UNIT_HOURS)}, _ctx())

    assert _hours_per_stay(result) == {1: [0.0, 1.0, 2.0], 2: [10.0, 11.0]}


def test_numeric_minutes_take_their_declared_unit():
    result = _callback_vent_ind({"mech_vent": _windows([180.0, 120.0], UNIT_MINUTES)}, _ctx())

    assert _hours_per_stay(result) == {1: [0.0, 1.0, 2.0], 2: [10.0, 11.0]}


def test_undeclared_numeric_duration_fails_closed():
    with pytest.raises(DurationUnitUndeclaredError) as excinfo:
        _callback_vent_ind({"mech_vent": _windows([3.0, 2.0])}, _ctx())

    assert excinfo.value.reason_code == "duration_unit_undeclared"
    assert isinstance(excinfo.value, DurationUnitError)
    assert "vent_ind" in str(excinfo.value) and "duration" in str(excinfo.value)


def test_timedelta_durations_are_unchanged():
    durations = pd.to_timedelta([3.0, 2.0], unit="h")
    result = _callback_vent_ind({"mech_vent": _windows(durations)}, _ctx())

    assert _hours_per_stay(result) == {1: [0.0, 1.0, 2.0], 2: [10.0, 11.0]}


def test_duration_helper_never_reads_numbers_as_nanoseconds():
    values = pd.Series([3.0, 26.9])

    hours = duration_series_to_timedelta(values, UNIT_HOURS, concept="probe")
    assert hours.dt.total_seconds().div(3600).tolist() == pytest.approx([3.0, 26.9])

    passthrough = pd.Series(pd.to_timedelta([90], unit="min"))
    assert duration_series_to_timedelta(passthrough, None) is passthrough

    with pytest.raises(DurationUnitUndeclaredError):
        duration_series_to_timedelta(values, None, concept="probe")
    with pytest.raises(DurationUnitError):
        duration_series_to_timedelta(values, "timedelta", concept="probe")
