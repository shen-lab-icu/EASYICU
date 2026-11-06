"""eICU intake rates: the rule for same-time conflicting rates of one channel.

``systemic_intake`` integrates a ``mL/hr`` rate until the next record of the
same stay and channel. When one channel carries two or more different rates
at the same offset, only one of them can own the segment to the next record;
the others get duration 0 and contribute no volume. Since 1962535eb the
records are sorted by (stay, channel, offset, rate), so the largest rate owns
the segment. Before, the record delivered last by the loader did, so the
volume depended on batch and thread layout (lead-r13 delta2
UNEXPECTED_CHANGE, item 10). Synthetic rows only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.eicu_io import systemic_intake

_COLUMN = "cellvaluenumeric"
_CHANNEL = "Intake (ml)|IV A (mL/hr)"


def _frame(rates_at_zero: list[float]) -> pd.DataFrame:
    n = len(rates_at_zero)
    return pd.DataFrame({
        "patientunitstayid": [1] * (n + 1),
        "intakeoutputoffset": [0] * n + [60],
        "cellpath": [_CHANNEL] * (n + 1),
        _COLUMN: [*rates_at_zero, 0.0],
    })


def _hourly(out: pd.DataFrame) -> dict[int, float]:
    return out.groupby("intakeoutputoffset")[_COLUMN].sum().to_dict()


@pytest.mark.parametrize("rates", [[30.0, 90.0], [90.0, 30.0], [30.0, 90.0, 60.0], [90.0, 60.0, 30.0]])
def test_the_largest_same_time_rate_owns_the_segment_in_any_input_order(rates):
    out = systemic_intake(_frame(rates), _COLUMN)
    # 90 mL/hr for the 60 minutes to the next record: 90 mL in hour 0. The
    # smaller same-time rates have duration 0 and add nothing.
    assert _hourly(out) == {0: 90.0}
    assert out.attrs["eicu_intake_audit"]["integrated_pieces"] == 1


def test_the_tie_rule_is_per_channel_and_per_stay():
    other_channel = _frame([10.0, 20.0]).assign(cellpath="Intake (ml)|IV B (mL/hr)")
    other_stay = _frame([5.0, 15.0]).assign(patientunitstayid=2)
    frame = pd.concat([_frame([90.0, 30.0]), other_channel, other_stay], ignore_index=True)
    out = systemic_intake(frame.iloc[::-1].reset_index(drop=True), _COLUMN)
    totals = out.groupby(["patientunitstayid", "cellpath"])[_COLUMN].sum().to_dict()
    assert totals == {
        (1, _CHANNEL): 90.0,
        (1, "Intake (ml)|IV B (mL/hr)"): 20.0,
        (2, _CHANNEL): 15.0,
    }


def test_a_same_time_zero_does_not_stop_a_larger_rate():
    # 0 and 50 at the same offset: the larger rate owns the segment, so the
    # stop is the next record, not the same-time zero.
    out = systemic_intake(_frame([0.0, 50.0]), _COLUMN)
    assert _hourly(out) == {0: 50.0}
