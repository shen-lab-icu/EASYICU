"""Grid records follow the raw-concept cache entries they describe (NB16-GRID-LIFECYCLE).

``_raw_concept_grid`` records the grid each raw-cache entry was loaded on, so
readers can refuse an entry from the other grid. A record must not outlive
its entry or describe a table the cache never held:

1. per-key eviction drops the key's record;
2. LRU eviction under the byte budget drops the evicted key's record;
3. clearing the raw cache clears every record;
4. a store the budget refuses records no grid.

Memory hygiene only: readers already treat a stale record as unknown through
the weakref, so loaded values do not change. Synthetic frames, no source data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.concept import ConceptDictionary, ConceptResolver, _estimate_cached_bytes
from easyicu.table import ICUTable

HOUR = pd.Timedelta(hours=1)


def _resolver(monkeypatch) -> ConceptResolver:
    monkeypatch.setenv("EASYICU_CACHE_BUDGET_MB", "0")  # unbounded unless a test pins one
    return ConceptResolver(ConceptDictionary(concepts={}))


def _table(nrows: int = 100) -> ICUTable:
    frame = pd.DataFrame({
        "stay_id": np.arange(nrows, dtype=np.int64),
        "charttime": pd.to_timedelta(np.arange(nrows), unit="h"),
        "x": np.arange(nrows, dtype=np.float64),
    })
    return ICUTable(data=frame, id_columns=["stay_id"], index_column="charttime", value_column="x")


def _store(resolver, name, table, *, legacy=False):
    resolver._store_raw_concept_cache(
        name, "ids", table, aggregator=None, store_legacy=legacy, grid=HOUR,
    )
    return resolver._raw_cache_key(name, "ids", None)


def test_per_key_eviction_drops_the_grid_record(monkeypatch):
    resolver = _resolver(monkeypatch)
    keep_table, gone_table = _table(), _table()
    keep, gone = _store(resolver, "keep", keep_table), _store(resolver, "gone", gone_table)
    assert {keep, gone} <= set(resolver._raw_concept_grid)

    resolver._evict_cache_entry("raw", gone)

    assert gone not in resolver._raw_concept_cache
    assert gone not in resolver._raw_concept_grid
    assert resolver._raw_concept_grid[keep][1] == HOUR


def test_lru_eviction_drops_the_evicted_grid_record(monkeypatch):
    resolver = _resolver(monkeypatch)
    tables = [_table(5_000) for _ in range(4)]
    resolver._cache_budget_bytes = 3 * _estimate_cached_bytes(tables[0])

    keys = [_store(resolver, f"c{i}", t) for i, t in enumerate(tables)]

    assert keys[0] not in resolver._raw_concept_cache  # the LRU head went
    assert keys[0] not in resolver._raw_concept_grid
    assert set(resolver._raw_concept_grid) == set(keys[1:])


def test_clearing_the_raw_cache_clears_every_grid_record(monkeypatch):
    resolver = _resolver(monkeypatch)
    tables = [_table(), _table()]
    _store(resolver, "a", tables[0], legacy=True)
    _store(resolver, "b", tables[1])
    assert len(resolver._raw_concept_grid) == 3  # two keys plus the legacy alias

    resolver.drop_source_caches()
    assert resolver._raw_concept_grid == {}

    _store(resolver, "a", tables[0])
    resolver.clear_table_cache()
    assert resolver._raw_concept_grid == {}
    assert resolver._raw_concept_cache == {}


def test_refused_store_records_no_grid(monkeypatch):
    resolver = _resolver(monkeypatch)
    big = _table(50_000)
    resolver._cache_budget_bytes = _estimate_cached_bytes(big) // 2

    key = _store(resolver, "big", big, legacy=True)

    assert key not in resolver._raw_concept_cache
    assert key not in resolver._raw_concept_grid
    assert ("big", "ids") not in resolver._raw_concept_grid  # no alias either
