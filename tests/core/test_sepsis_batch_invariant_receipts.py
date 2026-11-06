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


def _blocked(table: ICUTable, concept: str) -> ICUTable:
    from easyicu.concept.availability_signal import (
        ConceptAvailabilityRecord,
        set_table_availability,
    )

    set_table_availability(table, ConceptAvailabilityRecord(
        concept=concept, database="miiv", reason="source_unavailable", n_rows=0,
        sources_defined=("chartevents",), missing_tables=("chartevents",),
    ))
    return table


def _empty_score(score: str) -> ICUTable:
    return ICUTable(
        pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "charttime": pd.Series(dtype="float64"),
                      score: pd.Series(dtype="float64")}),
        ["stay_id"], "charttime", score,
    )


@pytest.mark.parametrize(
    ("concept", "callback", "score"),
    [("sep3", _callback_sep3, "sofa"), ("sep3_sofa2", _callback_sep3_sofa2, "sofa2")],
)
def test_a_blocked_score_source_is_never_an_evaluated_empty(concept, callback, score):
    # delta9r B-SEPSIS-SCORE-RECEIPT: positive suspected infection over a SOFA
    # source the loader reports missing.  The pair differs only in that receipt.
    infection = _event("susp_inf", [(7, 0.0)])
    blocked = callback(
        {score: _blocked(_empty_score(score), score), "susp_inf": infection}, _ctx(concept),
    )
    complete = callback(
        {score: _empty_score(score), "susp_inf": _event("susp_inf", [(7, 0.0)])}, _ctx(concept),
    )
    assert blocked.data.empty and complete.data.empty
    assert _receipt(blocked, concept) == (
        "source_unavailable", "not_evaluable", "required_source_table_missing", ("chartevents",)
    )
    assert _receipt(complete, concept) == (
        "data_missing", "evaluated", "no_qualifying_events_observed", ()
    )


def test_the_export_projection_reports_a_blocked_score_as_structurally_unavailable():
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.resources import load_dictionary
    from easyicu.webserver import dataio

    concept, module = "sep3_sofa2", "sepsis3_sofa2"
    plan = dataio.compile_concept_load_plan(CONCEPT_GROUPS_INTERNAL[module])
    assert concept in plan.output_concepts
    statuses = {}
    for label, score in (
        ("blocked", _blocked(_empty_score("sofa2"), "sofa2")),
        ("complete", _empty_score("sofa2")),
    ):
        result = _callback_sep3_sofa2(
            {"sofa2": score, "susp_inf": _event("susp_inf", [(7, 0.0)])}, _ctx(concept),
        )
        _, projected = dataio._prepare_export_availability(
            result.data.copy(), module=module, plan=plan,
            records={concept: table_availability(result, concept)},
            dictionary=load_dictionary(include_sofa2=True),
        )
        statuses[label] = projected[concept]
    assert statuses["blocked"]["availability"] == "structurally_unavailable_placeholder"
    assert statuses["blocked"]["evaluation_status"] == "not_evaluable"
    assert statuses["blocked"]["missing_tables"] == ["chartevents"]
    assert statuses["complete"]["availability"] == "produced_all_null"
    assert statuses["complete"]["evaluation_status"] == "evaluated"


@pytest.mark.parametrize(
    ("concept", "module", "callback", "score"),
    [
        ("sep3_sofa1", "sepsis3_sofa1", _callback_sep3, "sofa"),
        ("sep3_sofa2", "sepsis3_sofa2", _callback_sep3_sofa2, "sofa2"),
    ],
)
def test_an_evaluated_empty_suspicion_never_hides_an_incomplete_score(concept, module, callback, score):
    # delta10r: the susp_inf receipt is produced by its own callback (a complete
    # antibiotic source with no antibiotic in this batch), then the sepsis
    # owner meets a SOFA source the loader reports missing.
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.resources import load_dictionary
    from easyicu.webserver import dataio

    susp_inf = _callback_susp_inf(
        {"abx": _event("abx", []), "samp": _event("samp", [(7, 1.0)])}, _ctx("susp_inf"),
    )
    assert _receipt(susp_inf, "susp_inf")[1:3] == ("evaluated", "no_qualifying_events_observed")
    plan = dataio.compile_concept_load_plan(CONCEPT_GROUPS_INTERNAL[module])
    source = next(
        binding.source_concept for binding in plan.materializations
        if binding.output_concept == concept
    ) if concept not in plan.source_concepts else concept
    statuses = {}
    for label, scores in (
        ("blocked", _blocked(_empty_score(score), score)),
        ("complete", _empty_score(score)),
    ):
        result = callback({score: scores, "susp_inf": susp_inf}, _ctx(source))
        _, projected = dataio._prepare_export_availability(
            result.data.copy(), module=module, plan=plan,
            records={source: table_availability(result, source)},
            dictionary=load_dictionary(include_sofa2=True),
        )
        statuses[label] = projected[concept]
    assert statuses["blocked"]["availability"] == "structurally_unavailable_placeholder"
    assert statuses["blocked"]["evaluation_status"] == "not_evaluable"
    assert statuses["blocked"]["reason_code"] == "required_source_table_missing"
    assert statuses["blocked"]["missing_tables"] == ["chartevents"]
    assert statuses["complete"]["availability"] == "produced_all_null"
    assert statuses["complete"]["reason_code"] == "no_qualifying_events_observed"


def test_the_empty_antibiotic_shortcut_checks_every_suspicion_input():
    # NB-SUSP-INF-EARLY-GUARD: an empty antibiotic table beside a culture input
    # whose own receipt is not evaluable (not blocked, so the entry check lets
    # it through) is not "no qualifying event".
    from easyicu.concept.availability_signal import (
        ConceptAvailabilityRecord,
        set_table_availability,
    )

    samp = _event("samp", [])
    set_table_availability(samp, ConceptAvailabilityRecord(
        concept="samp", database="miiv", reason="data_missing", n_rows=0,
        evaluation_status="not_evaluable", reason_code="component_time_index_missing",
    ))
    result = _callback_susp_inf({"abx": _event("abx", []), "samp": samp}, _ctx("susp_inf"))
    assert _receipt(result, "susp_inf")[1:3] == ("not_evaluable", "component_time_index_missing")


@pytest.mark.parametrize(
    ("concept", "callback"), [("sep3", _callback_sep3), ("sep3_sofa2", _callback_sep3_sofa2)]
)
def test_an_evaluated_suspicion_without_the_score_input_is_not_evaluable(concept, callback):
    # delta11r NB-COMPONENT-ABSENT: only susp_inf is supplied (an evaluated
    # empty from its own callback); the score key is absent altogether.
    susp_inf = _callback_susp_inf(
        {"abx": _event("abx", []), "samp": _event("samp", [(7, 1.0)])}, _ctx("susp_inf"),
    )
    result = callback({"susp_inf": susp_inf}, _ctx(concept))
    assert result.data.empty
    assert _receipt(result, concept)[1:3] == ("not_evaluable", "required_component_missing")
