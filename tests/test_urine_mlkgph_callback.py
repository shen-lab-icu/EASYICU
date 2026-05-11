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

from easyicu.callbacks import _urine_window_avg
from easyicu.concept_callbacks import (
    ConceptCallbackContext,
    _callback_urine_mlkgph,
)
from easyicu.table import ICUTable


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
