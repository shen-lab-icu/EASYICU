"""A concept callback reads one input whether or not the raw cache is warm (B-CACHE-WARMTH).

vaso_ind, urine24 and the uo_6h/12h/24h windows asked ``get_raw_concept`` for
their inputs. A hit (the raw cache already held an ungridded table) and a miss
(whose own cold load returned None, so the callback fell back to another
table or load) gave two different inputs, so the result depended on which
modules had run earlier in the same keep-cache group.

Now vaso_ind uses the durations it was given on the requested interval
(ricu 0.6.3 ``vaso_ind``: ``collect_dots(cnc, interval, ...)``); urine24 and the
uo windows load anything missing through ``ConceptResolver.load_callback_inputs``,
exactly as the recursive loader passes sub-concepts (ricu ``urine24``:
``collect_dots("urine", interval, ...)``). Each case runs warm and cold, with the
concept-data cache on and off. Synthetic rows only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.callbacks import ConceptCallbackContext, execute_concept_callback
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable
from easyicu.utils import compute_patient_ids_hash
from tests.support.eicu_source import _eicu_source
from tests.support.urine_raw_source import _urine_source

HOUR = pd.Timedelta(hours=1)
IDS = {"stay_id": [1]}


def _resolver(*, data_cache: bool) -> ConceptResolver:
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    if not data_cache:
        store = resolver._bounded_cache_store
        resolver._bounded_cache_store = (
            lambda role, key, value, **kw: False if role == "data" else store(role, key, value, **kw)
        )
        resolver._active_requested_result = lambda *args, **kwargs: None
    return resolver


def _warm(resolver, name, table):
    """Hold an ungridded raw entry exactly where get_raw_concept looks."""
    resolver._keep_cache_between_calls = True
    resolver._store_raw_concept_cache(
        name, compute_patient_ids_hash(IDS), table, aggregator=None, store_legacy=True, grid=None,
    )
    return resolver


def _run(callback, tables, resolver, source, concept=None):
    ctx = ConceptCallbackContext(
        concept_name=concept or callback, target=None, interval=HOUR,
        resolver=resolver, data_source=source, patient_ids=IDS,
    )
    return execute_concept_callback(callback, dict(tables), ctx).data.reset_index(drop=True)


def _variants(source, warm_entries):
    out = []
    for data_cache in (True, False):
        out.append(_resolver(data_cache=data_cache))  # cold
        warm = _resolver(data_cache=data_cache)
        for name, table in warm_entries.items():
            _warm(warm, name, table)
        out.append(warm)
    return out


def _duration(name, rows):
    frame = pd.DataFrame(rows, columns=["stay_id", "charttime", name])
    return ICUTable(data=frame, id_columns=["stay_id"], index_column="charttime", value_column=name)


def test_vaso_ind_uses_the_durations_on_the_requested_grid(tmp_path):
    source = _urine_source(tmp_path, "miiv", [1.0], [50.0])
    gridded = {"norepi_dur": _duration("norepi_dur", [(1, 0.0, 0.5)])}
    # The raw episode starts at 0.8 h and runs 0.4 h: it reaches hour 1, the
    # hourly one does not.
    raw = {"norepi_dur": _duration("norepi_dur", [(1, 0.8, 0.4)])}

    results = [_run("vaso_ind", gridded, r, source) for r in _variants(source, raw)]
    assert all(isinstance(result, pd.DataFrame) for result in results)
    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0], check_exact=True)
    assert results[0]["charttime"].tolist() == [0.0]


@pytest.mark.parametrize("callback", ["urine24", "uo_6h"])
def test_urine_callbacks_load_missing_inputs_as_the_recursive_loader_does(tmp_path, callback):
    times = [0.5 + 2 * k for k in range(14)]
    source = _urine_source(tmp_path, "miiv", times, [40.0 + 5 * (k % 3) for k in range(14)])
    raw = ConceptResolver(load_dictionary(include_sofa2=True)).load_concepts(
        ["urine", "weight"], source, merge=False, r_compatible=False, interval=None,
        aggregate=None, patient_ids=IDS, concept_workers=1, verbose=False,
    )
    assert len(raw["urine"].data) == len(times)

    results = [_run(callback, {}, r, source) for r in _variants(source, raw)]
    assert all(isinstance(result, pd.DataFrame) and not result.empty for result in results)
    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0], check_exact=True)

    # ... and the fallback equals the ordinary recursive load of the concept.
    ordinary = _resolver(data_cache=True).load_concepts(
        [callback], source, merge=False, r_compatible=False, interval=HOUR,
        patient_ids=IDS, concept_workers=1, verbose=False,
    )[callback].data.reset_index(drop=True)
    pd.testing.assert_frame_equal(results[0], ordinary, check_exact=True)


def _eicu_rass(tmp_path):
    rows = []
    for offset, score in ((5, "-4"), (40, "-1"), (130, "-2")):
        rows += [
            (offset, "Scores", "Sedation Scale/Score/Goal", "Sedation Scale", "RASS"),
            (offset, "Scores", "Sedation Scale/Score/Goal", "Sedation Score", score),
        ]
    nurse = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue",
    ])
    nurse["nursingchartentryoffset"] = nurse["nursingchartoffset"]
    nurse["nursingchartid"] = range(len(nurse))
    nurse["nursingchartid"] = nurse["nursingchartid"].astype(float)
    return _eicu_source(tmp_path, "eicu", nursecharting=nurse)


def test_get_raw_concept_returns_the_raw_table_warm_or_cold(tmp_path):
    source = _eicu_rass(tmp_path)
    ids = {"patientunitstayid": [1]}
    expected = ConceptResolver(load_dictionary(include_sofa2=True)).load_concepts(
        ["rass"], source, merge=False, r_compatible=False, interval=None, aggregate=False,
        patient_ids=ids, concept_workers=1, verbose=False,
    )["rass"].data.reset_index(drop=True)
    assert len(expected) == 3

    for data_cache in (True, False):
        resolver = _resolver(data_cache=data_cache)
        resolver._keep_cache_between_calls = True
        cold = resolver.get_raw_concept("rass", source, ids)  # loads and stores
        warm = resolver.get_raw_concept("rass", source, ids)  # cache hit
        for table in (cold, warm):
            assert isinstance(table, ICUTable)
            pd.testing.assert_frame_equal(table.data.reset_index(drop=True), expected, check_exact=True)


def test_get_raw_concept_refuses_a_concept_without_a_raw_reading(tmp_path):
    """Urine is defined only summed (signed irrigation): no raw table, a typed refusal."""
    from easyicu.concept import RawConceptLoadError

    source = _urine_source(tmp_path, "miiv", [1.0], [30.0])
    with pytest.raises(RawConceptLoadError) as raised:
        _resolver(data_cache=True).get_raw_concept("urine", source, IDS)
    assert raised.value.reason == "raw_concept_load_failed"


def test_get_raw_concept_refuses_typed_when_the_load_returns_no_table(tmp_path):
    from easyicu.concept import RawConceptLoadError

    source = _urine_source(tmp_path, "miiv", [1.0], [30.0])
    resolver = _resolver(data_cache=True)
    resolver.load_concepts = lambda *args, **kwargs: {"urine": pd.DataFrame()}
    with pytest.raises(RawConceptLoadError) as raised:
        resolver.get_raw_concept("urine", source, IDS)
    assert raised.value.reason == "raw_concept_not_a_table"
