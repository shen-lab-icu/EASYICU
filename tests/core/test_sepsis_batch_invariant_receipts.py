"""Sepsis receipts must not depend on which stays share an extraction batch.

A batch with no antibiotic rows, one whose antibiotics match no culture, and
one with an empty suspected-infection table all hold the same fact about their
stays: no qualifying event.  They publish one evaluated receipt, so the export
status no longer depends on which stays formed the last batch.  Synthetic
inputs only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.availability_signal import table_availability
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sep3,
    _callback_sep3_sofa2,
    _callback_susp_inf,
)
from easyicu.table import ICUTable


def _ctx(concept: str, **kwargs) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept, None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name="miiv")), None, kwargs,
    )


def _event(name: str, rows: list[tuple[int, float]]) -> ICUTable:
    frame = pd.DataFrame(rows, columns=["stay_id", "charttime"]).astype(
        {"stay_id": "int64", "charttime": "float64"}
    )
    frame[name] = True
    return ICUTable(frame, ["stay_id"], "charttime", name)


def _receipt(table: ICUTable, concept: str) -> tuple:
    record = table_availability(table, concept)
    assert record is not None
    return (record.reason, record.evaluation_status, record.reason_code, record.missing_tables)


def test_every_empty_suspicion_batch_publishes_one_receipt():
    no_antibiotics = _callback_susp_inf(
        {"abx": _event("abx", []), "samp": _event("samp", [(7, 1.0)])}, _ctx("susp_inf"),
    )
    unmatched = _callback_susp_inf(
        # Antibiotic at hour 0, culture 200 h later: outside both windows.
        {"abx": _event("abx", [(8, 0.0)]), "samp": _event("samp", [(8, 200.0)])},
        _ctx("susp_inf"),
    )
    assert no_antibiotics.data.empty and unmatched.data.empty
    expected = ("data_missing", "evaluated", "no_qualifying_events_observed", ())
    assert _receipt(no_antibiotics, "susp_inf") == expected
    assert _receipt(unmatched, "susp_inf") == expected


@pytest.mark.parametrize(
    ("concept", "callback", "score"),
    [("sep3", _callback_sep3, "sofa"), ("sep3_sofa2", _callback_sep3_sofa2, "sofa2")],
)
def test_an_empty_loaded_component_is_evaluated_not_missing(concept, callback, score):
    scores = ICUTable(
        pd.DataFrame({"stay_id": [7, 7], "charttime": [0.0, 1.0], score: [0, 0]}),
        ["stay_id"], "charttime", score,
    )
    empty_infection = callback(
        {score: scores, "susp_inf": _event("susp_inf", [])}, _ctx(concept),
    )
    assert empty_infection.data.empty
    assert _receipt(empty_infection, concept) == (
        "data_missing", "evaluated", "no_qualifying_events_observed", ()
    )
    # A stay with suspected infection but no SOFA rise: the same receipt.
    no_rise = callback(
        {score: scores, "susp_inf": _event("susp_inf", [(7, 0.0)])}, _ctx(concept),
    )
    if no_rise.data.empty:
        assert _receipt(no_rise, concept) == _receipt(empty_infection, concept)


@pytest.mark.parametrize(
    ("concept", "callback"), [("sep3", _callback_sep3), ("sep3_sofa2", _callback_sep3_sofa2)]
)
def test_a_component_never_loaded_stays_not_evaluable(concept, callback):
    result = callback({}, _ctx(concept))
    assert _receipt(result, concept)[1:3] == ("not_evaluable", "required_component_missing")
