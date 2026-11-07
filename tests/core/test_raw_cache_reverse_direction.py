"""A gridded raw-cache entry never answers an ungridded request (B-CACHE-REVERSE).

The raw-concept cache key carries no grid. ``_ensure_concept_loaded`` checked
the grid only for gridded requests, and ``get_raw_concept`` not at all, so an
ungridded request after an hourly load of the same concept could receive the
hourly rows instead of the raw ones:

- eICU ``ett_gcs`` forwards ``rass`` ungridded; after an hourly ``rass`` in the
  same call it read the hourly median (two charts, -4 and -1: -2.5, not deep
  sedation) instead of the raw -4 that opens the airway window;
- AUMC ``blood_cell_ratio`` (basos, monos, ...) loads ``wbc`` ungridded; after
  the hourly ``wbc`` it divided by the hourly value.

Each entry now records the grid it was loaded on and readers refuse the other
grid in both directions. The concept-data cache and the request-local map are
disabled here, so no other cache answers first. Synthetic rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.table import ICUTable
from tests.support.aumc_source import DATA, aumc_db
from tests.support.eicu_source import DICTIONARY, _eicu_source

HOUR = pd.Timedelta(hours=1)
EICU_IDS = {"patientunitstayid": [1]}
AUMC_IDS = {"admissionid": [1]}
_PAIR = "Sedation Scale/Score/Goal"


def _raw_cache_only(resolver: ConceptResolver) -> ConceptResolver:
    """Disable the interval-keyed concept caches so only the raw cache can answer."""
    store = resolver._bounded_cache_store

    def bounded_store(role, key, value, **kwargs):
        return False if role == "data" else store(role, key, value, **kwargs)

    resolver._bounded_cache_store = bounded_store
    resolver._active_requested_result = lambda *args, **kwargs: None
    return resolver


def _frame(table) -> pd.DataFrame:
    return table.data.reset_index(drop=True)


def _load(resolver, names, source, ids, interval=HOUR, **kwargs):
    return resolver.load_concepts(
        names, source, merge=False, r_compatible=False, interval=interval,
        patient_ids=ids, concept_workers=1, verbose=False, **kwargs,
    )


def _eicu(tmp_path):
    rows = []
    for offset, score in ((5, "-4"), (40, "-1"), (130, "-2")):
        rows += [
            (offset, "Scores", _PAIR, "Sedation Scale", "RASS"),
            (offset, "Scores", _PAIR, "Sedation Score", score),
        ]
    nurse = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue",
    ])
    nurse["nursingchartentryoffset"] = nurse["nursingchartoffset"]
    nurse["nursingchartid"] = np.arange(len(nurse), dtype=float)
    return _eicu_source(tmp_path, "eicu", nursecharting=nurse), ConceptDictionary.from_json(DICTIONARY)


def test_eicu_ett_gcs_forward_after_an_hourly_rass_equals_cold(tmp_path):
    source, dictionary = _eicu(tmp_path)
    cold = _load(_raw_cache_only(ConceptResolver(dictionary)), ["ett_gcs"], source, EICU_IDS)["ett_gcs"]
    assert cold.data["ett_gcs"].any(), "the raw RASS -4 must open an airway window"

    warm = _load(_raw_cache_only(ConceptResolver(dictionary)), ["rass", "ett_gcs"], source, EICU_IDS)
    pd.testing.assert_frame_equal(_frame(warm["ett_gcs"]), _frame(cold), check_exact=True)


def test_eicu_raw_requests_after_an_hourly_load_return_the_raw_rows(tmp_path):
    source, dictionary = _eicu(tmp_path)
    cold_raw = _load(
        _raw_cache_only(ConceptResolver(dictionary)), ["rass"], source, EICU_IDS,
        interval=None, aggregate=None,
    )["rass"]

    resolver = _raw_cache_only(ConceptResolver(dictionary))
    resolver._keep_cache_between_calls = True
    hourly = _load(resolver, ["rass"], source, EICU_IDS)["rass"]  # hourly entry in the raw cache
    assert len(hourly.data) != len(cold_raw.data)
    ungridded = _load(resolver, ["rass"], source, EICU_IDS, interval=None, aggregate=None)["rass"]
    pd.testing.assert_frame_equal(_frame(ungridded), _frame(cold_raw), check_exact=True)

    # get_raw_concept never hands back the hourly entry: it returns the raw
    # rows as a table, warm or cold.
    raw = resolver.get_raw_concept("rass", source, EICU_IDS)
    assert isinstance(raw, ICUTable)
    pd.testing.assert_frame_equal(_frame(raw), _frame(cold_raw), check_exact=True)
    for key, table in resolver._raw_concept_cache.items():
        grid = resolver._raw_entry_grid(key, table)
        if ConceptResolver._raw_entry_serves(table, grid, None):
            assert grid is None


def _aumc(tmp_path):
    # wbc charted twice in hour 0; absolute basophil and monocyte counts after it.
    db = aumc_db(tmp_path, numeric=[
        (10, 6779, 8.0, "10^9/l"), (50, 6779, 12.0, "10^9/l"),
        (20, 14256, 0.4, "10^9/l"), (55, 14260, 0.9, "10^9/l"),
    ])
    return db, ConceptDictionary.from_json(DATA / "concept-dict.json")


@pytest.mark.parametrize("name", ["basos", "wbc", "monos"])
def test_aumc_blood_cell_ratio_chain_equals_cold(tmp_path, name):
    db, dictionary = _aumc(tmp_path)
    cold = _load(_raw_cache_only(ConceptResolver(dictionary)), [name], db, AUMC_IDS)[name]
    # Hematology order: basos loads wbc ungridded, wbc is loaded hourly, then
    # monos loads wbc ungridded again after the hourly entry replaced it.
    warm = _load(_raw_cache_only(ConceptResolver(dictionary)), ["basos", "wbc", "monos"], db, AUMC_IDS)[name]
    pd.testing.assert_frame_equal(_frame(warm), _frame(cold), check_exact=True)


def test_a_precached_sub_table_is_marked_with_its_grid(tmp_path):
    source, dictionary = _eicu(tmp_path)
    resolver = ConceptResolver(dictionary)
    resolver._keep_cache_between_calls = True
    _load(resolver, ["gcs"], source, EICU_IDS)
    grids = {
        key: resolver._raw_entry_grid(key, table)
        for key, table in resolver._raw_concept_cache.items()
    }
    assert grids, "gcs must leave raw entries"
    assert ConceptResolver._RAW_GRID_UNKNOWN not in grids.values()
    assert [grid for key, grid in grids.items() if key[0] == "gcs"] == [HOUR]


def test_a_copied_table_carries_no_grid(tmp_path):
    """The grid belongs to the cache entry: a copy put under the key is unknown, a miss."""
    source, dictionary = _eicu(tmp_path)
    resolver = ConceptResolver(dictionary)
    resolver._keep_cache_between_calls = True
    raw = _load(resolver, ["rass"], source, EICU_IDS, interval=None, aggregate=None)["rass"]
    key = next(k for k, v in resolver._raw_concept_cache.items() if k[0] == "rass")
    assert resolver._raw_entry_grid(key, resolver._raw_concept_cache[key]) is None
    resolver._raw_concept_cache[key] = raw.copy()
    unknown = resolver._raw_entry_grid(key, resolver._raw_concept_cache[key])
    assert unknown is ConceptResolver._RAW_GRID_UNKNOWN
    assert not ConceptResolver._raw_entry_serves(resolver._raw_concept_cache[key], unknown, None)
    assert not ConceptResolver._raw_entry_serves(resolver._raw_concept_cache[key], unknown, HOUR)
