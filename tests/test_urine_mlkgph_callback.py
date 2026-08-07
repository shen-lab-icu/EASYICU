"""Regression tests for urine output rate callbacks.

``_callback_urine_mlkgph`` historically returned a NaN placeholder which
silently dropped a real SOFA-2 feature. It now delegates to
``_callback_uo_window`` (1-hour window) and emits real mL/kg/h values.
``_urine_window_avg`` is the underlying engine; both layers are pinned
here so future refactors do not regress to NaN-only output.
"""
from __future__ import annotations

import pandas as pd
import pytest

from easyicu.callbacks import (
    _urine_rate_window_avg_multi,
    _urine_window_avg,
)
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_urine_mlkgph,
)
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import hirid_urine


def test_hirid_urine_preserves_direct_rate_and_averages_same_hour():
    """HiRID 10020000 is mL/h, not a cumulative counter or gap volume."""

    raw = pd.DataFrame(
        {
            "patientid": [1, 1, 1],
            "datetime": [
                pd.Timestamp("2100-01-01 00:05"),
                pd.Timestamp("2100-01-01 00:40"),
                pd.Timestamp("2100-01-01 04:00"),
            ],
            "value": [50.0, 70.0, 90.0],
        }
    )

    result = hirid_urine(
        raw,
        concept_name="urine",
        val_col="value",
        unit_col="unit",
        interval=pd.Timedelta(hours=1),
    )

    assert result["datetime"].tolist() == [
        pd.Timestamp("2100-01-01 00:00"),
        pd.Timestamp("2100-01-01 04:00"),
    ]
    # Same-hour rates are averaged.  The four-hour charting gap is not
    # multiplied into the second value.
    assert result["urine"].tolist() == pytest.approx([60.0, 90.0])
    assert result["unit"].tolist() == ["mL", "mL"]


def test_hirid_rate_windows_use_covered_clock_time():
    """Irregular rate records are weighted by observed clock-time coverage."""

    urine = pd.DataFrame(
        {
            "patientid": [1, 1, 1],
            "datetime": [
                pd.Timedelta(hours=0),
                pd.Timedelta(hours=2),
                pd.Timedelta(hours=6),
            ],
            # One-hour-equivalent mL values from the source callback.
            "urine": [50.0, 100.0, 50.0],
        }
    )
    weight = pd.DataFrame({"patientid": [1], "weight": [50.0]})

    result = _urine_rate_window_avg_multi(
        urine,
        weight,
        windows=[(6, 3)],
        interval=pd.Timedelta(hours=1),
    )["uo_6h"]

    assert pd.isna(result.loc[0, "uo_6h"])
    # At t=6 the two direct rates cover [0,2] and [2,6]:
    # (100*2 + 50*4) / (50 kg * 6 covered h).
    assert result.loc[2, "uo_6h"] == pytest.approx(400.0 / 50.0 / 6.0)


def test_hirid_rate_windows_apply_recorded_rate_to_preceding_interval():
    urine = pd.DataFrame(
        {
            "patientid": [1, 1],
            "datetime": [
                pd.Timedelta(hours=0),
                pd.Timedelta(hours=10),
            ],
            "urine": [50.0, 500.0],
        }
    )
    weight = pd.DataFrame({"patientid": [1], "weight": [50.0]})

    result = _urine_rate_window_avg_multi(
        urine,
        weight,
        windows=[(6, 3), (12, 6)],
        interval=pd.Timedelta(hours=1),
    )

    # The t=10 hourly rate summarizes the preceding chart interval, matching
    # the official AKI-EWS HiRID backfill semantics.
    assert result["uo_6h"].loc[1, "uo_6h"] == pytest.approx(10.0)
    assert result["uo_12h"].loc[1, "uo_12h"] == pytest.approx(10.0)


def test_rate_windows_refuse_first_weight_when_entities_are_not_joinable():
    urine = pd.DataFrame(
        {
            "stay_id": [1] * 4 + [2] * 4,
            "charttime": [pd.Timedelta(hours=h) for h in range(4)] * 2,
            "urine": [50.0] * 8,
        }
    )
    weight = pd.DataFrame({"patientid": [100, 200], "weight": [50.0, 100.0]})

    result = _urine_rate_window_avg_multi(
        urine,
        weight,
        windows=[(3, 3)],
        interval=pd.Timedelta(hours=1),
    )["uo_3h"]

    assert result["uo_3h"].isna().all()


def test_urine_window_avg_one_hour_window_returns_per_row_ml_per_kg_per_hour():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "charttime": [
                pd.Timedelta(hours=0),
                pd.Timedelta(hours=1),
                pd.Timedelta(hours=2),
                pd.Timedelta(hours=3),
            ],
            "urine": [50.0, 60.0, 40.0, 80.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [50.0]})

    result = _urine_window_avg(
        urine=urine,
        weight=weight,
        window_hours=1,
        min_hours=1,
        interval=pd.Timedelta(hours=1),
    )

    assert "uo_1h" in result.columns, result.columns.tolist()
    # 1-row 1-hour window collapses to (urine_ml / weight_kg / 1h).
    expected = [50.0 / 50.0, 60.0 / 50.0, 40.0 / 50.0, 80.0 / 50.0]
    assert result["uo_1h"].tolist() == pytest.approx(expected)


def test_urine_window_avg_drops_rows_without_weight():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [pd.Timedelta(hours=0), pd.Timedelta(hours=0)],
            "urine": [50.0, 60.0],
        }
    )
    # Patient 2 has no weight — should be filtered out.
    weight = pd.DataFrame({"stay_id": [1], "weight": [50.0]})

    result = _urine_window_avg(
        urine=urine,
        weight=weight,
        window_hours=1,
        min_hours=1,
        interval=pd.Timedelta(hours=1),
    )

    assert result["stay_id"].tolist() == [1]
    assert result["uo_1h"].tolist() == pytest.approx([1.0])


def test_callback_urine_mlkgph_emits_real_values_not_nan_placeholder():
    """Pre-fix: callback returned NaN and dropped the SOFA-2 feature.

    Post-fix: callback delegates to the 1-hour windowed average and
    emits ``urine_mlkgph`` with real numeric values.
    """
    tables = {
        "urine": ICUTable(
            pd.DataFrame(
                {
                    "stay_id": [1, 1, 1],
                    "charttime": [
                        pd.Timedelta(hours=0),
                        pd.Timedelta(hours=1),
                        pd.Timedelta(hours=2),
                    ],
                    "urine": [40.0, 60.0, 80.0],
                }
            ),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="urine",
        ),
        "weight": ICUTable(
            pd.DataFrame({"stay_id": [1], "weight": [40.0]}),
            id_columns=["stay_id"],
            index_column=None,
            value_column="weight",
        ),
    }
    ctx = ConceptCallbackContext(
        concept_name="urine_mlkgph",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=None,
        patient_ids=None,
    )

    result = _callback_urine_mlkgph(tables, ctx)

    assert "urine_mlkgph" in result.data.columns, result.data.columns.tolist()
    values = result.data["urine_mlkgph"].dropna().tolist()
    assert values, "urine_mlkgph callback must emit at least one numeric value"
    assert values == pytest.approx([40.0 / 40.0, 60.0 / 40.0, 80.0 / 40.0])
    assert result.value_column == "urine_mlkgph"
