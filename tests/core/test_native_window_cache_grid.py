"""A warm raw cache answers a gridded request exactly as a cold load does.

Codex delta4 D4-NATIVE-CACHE: once tables stopped deriving an interval from
their rows (lead-r13 #11b), a native window table loaded without a grid is
cached with ``interval=None``. The raw-cache reuse check only compared two
declared grids, so a later hourly request in the same cache scope returned the
unexpanded windows while a cold load expanded them hourly: the rows depended on
whether the cache was warm. Synthetic AmsterdamUMCdb rows only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from tests.support.aumc_source import DATA, aumc_db, aumc_drug

HOUR = pd.Timedelta(hours=1)


def _dictionary():
    return ConceptDictionary.from_json(DATA / "concept-dict.json")


def _load(resolver, db, name, interval, *, merge):
    result = resolver.load_concepts(
        [name], db, interval=interval, patient_ids={"admissionid": [1]},
        merge=merge, r_compatible=False, concept_workers=1, verbose=False,
    )
    return result[name] if isinstance(result, dict) else result


def _canonical(frame):
    frame = frame.copy()
    frame = frame[sorted(frame.columns)]
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


SOURCES = {
    # A ventilation window (processitems, minutes 30..250) and a dextrose
    # infusion window (drugitems, minutes 10..200): both native win_tbl.
    "mech_vent": dict(processes=[(30, 250, 9328, "Beademen")]),
    "dex": dict(drugs=[aumc_drug(10, 200, 7254, dose=50.0, unit="ml", rate="uur")]),
}


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_warm_native_window_cache_equals_a_cold_hourly_load(tmp_path, name):
    db = aumc_db(tmp_path, **SOURCES[name])
    cold = _load(ConceptResolver(_dictionary()), db, name, HOUR, merge=True)
    assert len(cold) > 1, "the synthetic window must expand to several hours"

    warm_resolver = ConceptResolver(_dictionary())
    warm_resolver._keep_cache_between_calls = True  # what api.keep_cache() sets
    native = _load(warm_resolver, db, name, None, merge=False)  # cached ungridded
    assert native.interval is None and native.index_column is not None
    warm = _load(warm_resolver, db, name, HOUR, merge=True)

    # Same row set and values: the ungridded time-indexed entry is not reused
    # for an hourly request (before the fix the warm mech_vent call returned
    # the one raw window row with its stop and dur_var columns).
    pd.testing.assert_frame_equal(_canonical(warm), _canonical(cold), check_exact=True)


def test_a_cached_id_table_is_still_reused_on_a_gridded_request(tmp_path):
    """Tables without a time index carry no grid; they are reused as before."""
    db = aumc_db(tmp_path)
    resolver = ConceptResolver(_dictionary())
    resolver._keep_cache_between_calls = True
    first = _load(resolver, db, "weight", HOUR, merge=False)
    cached = [v for k, v in resolver._raw_concept_cache.items() if k[0] == "weight"]
    assert cached, "weight must be cached after the first load"
    second = _load(resolver, db, "weight", HOUR, merge=False)
    pd.testing.assert_frame_equal(_canonical(second.data), _canonical(first.data), check_exact=True)
