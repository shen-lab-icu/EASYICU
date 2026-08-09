"""Regression tests for HiRID's rate-aware 24-hour urine volume."""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_urine24,
)
from easyicu.table import ICUTable


def _context(database: str) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name="urine24",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=object(),
        data_source=SimpleNamespace(config=SimpleNamespace(name=database)),
        patient_ids=None,
    )


def test_hirid_urine24_integrates_sparse_rate_over_observed_clock_time():
    """Sparse mL/h readings must not create zero-output hours."""

    frame = pd.DataFrame(
        {
            "patientid": [1, 1, 1, 1, 1],
            "datetime": [0.0, 2.0, 6.0, 12.0, 24.0],
            "urine": [50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )
    urine = ICUTable(
        data=frame,
        id_columns=["patientid"],
        index_column="datetime",
        value_column="urine",
    )

    result = _callback_urine24({"urine": urine}, _context("hirid")).data

    # A constant 50 mL/h rate is 1,200 mL per 24 hours even when it is charted
    # sparsely.  The first evaluable point is 12h because urine24 requires at
    # least half-window coverage, and the callback does not invent rows after
    # the last observation.
    assert result["datetime"].tolist() == [0.0, 2.0, 6.0, 12.0, 24.0]
    assert result.loc[result["datetime"] < 12, "urine24"].isna().all()
    assert result.loc[result["datetime"] == 12, "urine24"].item() == pytest.approx(
        1200.0
    )
    assert result.loc[result["datetime"] == 24, "urine24"].item() == pytest.approx(
        1200.0
    )


def test_hirid_urine24_preserves_true_oliguria():
    """The rate-aware branch must still retain genuinely low urine output."""

    frame = pd.DataFrame(
        {
            "patientid": [1, 1, 1],
            "datetime": [0.0, 12.0, 24.0],
            "urine": [10.0, 10.0, 10.0],
        }
    )
    urine = ICUTable(
        data=frame,
        id_columns=["patientid"],
        index_column="datetime",
        value_column="urine",
    )

    result = _callback_urine24({"urine": urine}, _context("hirid")).data

    assert result.loc[result["datetime"] == 12, "urine24"].item() == pytest.approx(
        240.0
    )
    assert result.loc[result["datetime"] == 24, "urine24"].item() == pytest.approx(
        240.0
    )


def test_hirid_urine24_does_not_turn_missing_rate_into_zero():
    """A missing rate reading contributes no observed coverage, not anuria."""

    frame = pd.DataFrame(
        {
            "patientid": [1, 1, 1],
            "datetime": [0.0, 6.0, 12.0],
            "urine": [50.0, None, 50.0],
        }
    )
    urine = ICUTable(
        data=frame,
        id_columns=["patientid"],
        index_column="datetime",
        value_column="urine",
    )

    result = _callback_urine24({"urine": urine}, _context("hirid")).data

    assert result["datetime"].tolist() == [0.0, 12.0]
    assert result.loc[result["datetime"] == 12, "urine24"].item() == pytest.approx(
        1200.0
    )
