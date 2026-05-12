"""Regression tests for derived ``rec_cncpt`` callbacks added in 2026-05.

Covers the two zero-cost arithmetic derivations registered in
``concept_callbacks.CALLBACK_REGISTRY``:

* ``anion_gap`` = Na - (Cl + HCO3) — standard serum anion gap
* ``pulse_pressure`` = SBP - DBP — arterial pulse pressure

Both callbacks should:
1. Inner-join on (id, time) so only rows with every component present
   contribute,
2. Compute the arithmetic difference,
3. Filter to a permissive physiological window,
4. Return an ``ICUTable`` whose ``value_column`` matches the concept name.
"""
from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept_callbacks import (
    CALLBACK_REGISTRY,
    ConceptCallbackContext,
    _callback_anion_gap,
    _callback_pulse_pressure,
)
from easyicu.table import ICUTable


def _ctx(name: str) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name=name,
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=None,
        patient_ids=None,
    )


# ---------------------------------------------------------------------------
# anion_gap
# ---------------------------------------------------------------------------


def test_anion_gap_computes_na_minus_cl_plus_bicar():
    """Standard 3-component subtraction on aligned timestamps."""
    times = [pd.Timestamp("2026-01-01 08:00"), pd.Timestamp("2026-01-01 12:00")]
    tables = {
        "na": ICUTable(
            pd.DataFrame({"stay_id": [1, 1], "charttime": times, "na": [140.0, 138.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="na",
        ),
        "cl": ICUTable(
            pd.DataFrame({"stay_id": [1, 1], "charttime": times, "cl": [100.0, 102.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="cl",
        ),
        "bicar": ICUTable(
            pd.DataFrame({"stay_id": [1, 1], "charttime": times, "bicar": [24.0, 20.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="bicar",
        ),
    }

    result = _callback_anion_gap(tables, _ctx("anion_gap"))

    assert result.value_column == "anion_gap"
    assert "anion_gap" in result.data.columns
    # 140 - (100+24) = 16; 138 - (102+20) = 16
    assert result.data["anion_gap"].tolist() == pytest.approx([16.0, 16.0])


def test_anion_gap_inner_joins_only_rows_with_all_components():
    """A timestamp missing one component must be dropped."""
    t0 = pd.Timestamp("2026-01-01 08:00")
    t1 = pd.Timestamp("2026-01-01 09:00")  # only na+cl, no bicar -> drop

    tables = {
        "na": ICUTable(
            pd.DataFrame({"stay_id": [1, 1], "charttime": [t0, t1], "na": [140.0, 142.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="na",
        ),
        "cl": ICUTable(
            pd.DataFrame({"stay_id": [1, 1], "charttime": [t0, t1], "cl": [100.0, 100.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="cl",
        ),
        "bicar": ICUTable(
            pd.DataFrame({"stay_id": [1], "charttime": [t0], "bicar": [25.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="bicar",
        ),
    }

    result = _callback_anion_gap(tables, _ctx("anion_gap"))

    assert len(result.data) == 1
    assert result.data["anion_gap"].iloc[0] == pytest.approx(15.0)


def test_anion_gap_filters_implausible_values():
    """Values outside [-10, 50] mEq/L should be dropped."""
    t = [pd.Timestamp("2026-01-01") + pd.Timedelta(hours=h) for h in range(3)]
    tables = {
        "na": ICUTable(
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": t, "na": [140.0, 140.0, 140.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="na",
        ),
        "cl": ICUTable(
            # row 0: normal; row 1: yields AG=-20 (out-of-range low); row 2: yields AG=60 (out-of-range high)
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": t, "cl": [100.0, 130.0, 60.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="cl",
        ),
        "bicar": ICUTable(
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": t, "bicar": [24.0, 30.0, 20.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="bicar",
        ),
    }

    result = _callback_anion_gap(tables, _ctx("anion_gap"))

    # Only row 0 (AG=16) survives the [-10, 50] filter.
    assert result.data["anion_gap"].tolist() == pytest.approx([16.0])


def test_anion_gap_registered_in_callback_registry():
    assert CALLBACK_REGISTRY.get("anion_gap") is _callback_anion_gap


# ---------------------------------------------------------------------------
# pulse_pressure
# ---------------------------------------------------------------------------


def test_pulse_pressure_computes_sbp_minus_dbp():
    times = [pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=m) for m in (0, 15, 30)]
    tables = {
        "sbp": ICUTable(
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": times, "sbp": [120.0, 130.0, 90.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="sbp",
        ),
        "dbp": ICUTable(
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": times, "dbp": [80.0, 70.0, 60.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="dbp",
        ),
    }

    result = _callback_pulse_pressure(tables, _ctx("pulse_pressure"))

    assert result.value_column == "pulse_pressure"
    assert result.data["pulse_pressure"].tolist() == pytest.approx([40.0, 60.0, 30.0])


def test_pulse_pressure_drops_unpaired_timestamps():
    """A timestamp missing SBP or DBP should be dropped."""
    t0 = pd.Timestamp("2026-01-01 08:00")
    t1 = pd.Timestamp("2026-01-01 09:00")

    tables = {
        "sbp": ICUTable(
            pd.DataFrame({"stay_id": [1, 1], "charttime": [t0, t1], "sbp": [120.0, 130.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="sbp",
        ),
        "dbp": ICUTable(
            pd.DataFrame({"stay_id": [1], "charttime": [t0], "dbp": [80.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="dbp",
        ),
    }

    result = _callback_pulse_pressure(tables, _ctx("pulse_pressure"))

    assert len(result.data) == 1
    assert result.data["pulse_pressure"].iloc[0] == pytest.approx(40.0)


def test_pulse_pressure_filters_implausible_values():
    """Negative PP (DBP > SBP, lab error) and >200 mmHg should be dropped."""
    times = [pd.Timestamp("2026-01-01") + pd.Timedelta(hours=h) for h in range(3)]
    tables = {
        "sbp": ICUTable(
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": times, "sbp": [120.0, 60.0, 350.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="sbp",
        ),
        "dbp": ICUTable(
            # row 1: dbp>sbp gives PP=-20 (filtered); row 2: PP=300 (>200, filtered)
            pd.DataFrame({"stay_id": [1, 1, 1], "charttime": times, "dbp": [80.0, 80.0, 50.0]}),
            id_columns=["stay_id"], index_column="charttime", value_column="dbp",
        ),
    }

    result = _callback_pulse_pressure(tables, _ctx("pulse_pressure"))

    assert result.data["pulse_pressure"].tolist() == pytest.approx([40.0])


def test_pulse_pressure_registered_in_callback_registry():
    assert CALLBACK_REGISTRY.get("pulse_pressure") is _callback_pulse_pressure


# ---------------------------------------------------------------------------
# empty-input edge cases
# ---------------------------------------------------------------------------


def test_anion_gap_handles_empty_components():
    """All-empty inputs must produce an empty ICUTable, not crash."""
    empty = ICUTable(
        pd.DataFrame({"stay_id": pd.Series([], dtype="int64"),
                      "charttime": pd.Series([], dtype="datetime64[ns]"),
                      "na": pd.Series([], dtype="float64")}),
        id_columns=["stay_id"], index_column="charttime", value_column="na",
    )
    tables = {
        "na": empty,
        "cl": ICUTable(empty.data.rename(columns={"na": "cl"}),
                       id_columns=["stay_id"], index_column="charttime", value_column="cl"),
        "bicar": ICUTable(empty.data.rename(columns={"na": "bicar"}),
                          id_columns=["stay_id"], index_column="charttime", value_column="bicar"),
    }

    result = _callback_anion_gap(tables, _ctx("anion_gap"))
    assert result.value_column == "anion_gap"
    assert len(result.data) == 0


def test_pulse_pressure_handles_empty_components():
    empty = ICUTable(
        pd.DataFrame({"stay_id": pd.Series([], dtype="int64"),
                      "charttime": pd.Series([], dtype="datetime64[ns]"),
                      "sbp": pd.Series([], dtype="float64")}),
        id_columns=["stay_id"], index_column="charttime", value_column="sbp",
    )
    tables = {
        "sbp": empty,
        "dbp": ICUTable(empty.data.rename(columns={"sbp": "dbp"}),
                        id_columns=["stay_id"], index_column="charttime", value_column="dbp"),
    }

    result = _callback_pulse_pressure(tables, _ctx("pulse_pressure"))
    assert result.value_column == "pulse_pressure"
    assert len(result.data) == 0
