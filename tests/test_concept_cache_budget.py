"""Regression tests for the bounded in-memory concept caches (PC safety).

The resolver caches raw source tables and per-concept results so that
``keep_cache`` can reuse them across sequential ``load_concepts`` calls.
Before this fix those caches grew without bound: a single cached table on
a 94K-patient export is tens of GB, which OOMs an 8-16GB consumer PC.

All writes now route through ``_bounded_cache_store``, which enforces a
byte budget (``EASYICU_CACHE_BUDGET_MB``) by evicting least-recently-used
entries. These tests pin:

1. entries are evicted so the cache never exceeds its budget;
2. eviction is LRU — a touched entry outlives an untouched older one;
3. a single value larger than the whole budget is refused, not cached
   (holding it is exactly the OOM the budget prevents);
4. ``EASYICU_CACHE_BUDGET_MB=0`` disables the bound (legacy behaviour);
5. the two-key raw alias is charged once, not twice.

All tests use tiny synthetic frames and run without ``--run-real``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from easyicu.concept import (
    ConceptResolver,
    _estimate_cached_bytes,
    _resolve_cache_budget_bytes,
)
from easyicu.concept.schema import ConceptDictionary
from easyicu.table import ICUTable


def _resolver(monkeypatch, budget_mb: str) -> ConceptResolver:
    monkeypatch.setenv("EASYICU_CACHE_BUDGET_MB", budget_mb)
    return ConceptResolver(ConceptDictionary(concepts={}))


def _frame(nrows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": np.arange(nrows, dtype=np.int64),
            "value": np.arange(nrows, dtype=np.float64),
        }
    )


def test_budget_env_pins_and_disables(monkeypatch):
    monkeypatch.setenv("EASYICU_CACHE_BUDGET_MB", "128")
    assert _resolve_cache_budget_bytes() == 128 * 1024 * 1024
    # <= 0 disables the bound entirely (legacy unbounded behaviour)
    monkeypatch.setenv("EASYICU_CACHE_BUDGET_MB", "0")
    assert _resolve_cache_budget_bytes() is None
    monkeypatch.setenv("EASYICU_CACHE_BUDGET_MB", "-1")
    assert _resolve_cache_budget_bytes() is None


def test_default_budget_is_conservative_on_16gb_host(monkeypatch):
    import psutil

    monkeypatch.delenv("EASYICU_CACHE_BUDGET_MB", raising=False)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=16 * 1024**3),
    )

    assert _resolve_cache_budget_bytes() == 512 * 1024**2


def test_store_evicts_to_stay_within_budget(monkeypatch):
    resolver = _resolver(monkeypatch, "1")  # 1 MB
    per_frame = _estimate_cached_bytes(_frame(20_000))
    assert per_frame > 0
    for i in range(40):
        resolver._bounded_cache_store("table", ("t", i), _frame(20_000))

    assert resolver._cache_total_bytes <= resolver._cache_budget_bytes
    assert len(resolver._table_cache) < 40  # something was evicted
    assert resolver._cache_evictions > 0
    # accounting stays consistent with the dict it tracks
    assert len(resolver._cache_entry_bytes) == len(resolver._table_cache)


def test_eviction_is_lru(monkeypatch):
    # Disable the env budget, then pin a byte budget sized to hold EXACTLY
    # three equal frames so the eviction is deterministic.
    resolver = _resolver(monkeypatch, "0")
    per_frame = _estimate_cached_bytes(_frame(5_000))
    resolver._cache_budget_bytes = 3 * per_frame

    for i in range(4):  # k0..k3; inserting k3 evicts k0 -> [k1, k2, k3]
        resolver._bounded_cache_store("table", ("t", i), _frame(5_000))
    assert set(resolver._table_cache) == {("t", 1), ("t", 2), ("t", 3)}

    # Touch k1 (older than k2 by insertion order) -> it becomes most-recent.
    resolver._cache_touch("table", ("t", 1))
    # One more insert evicts the current LRU head, which is now k2 — NOT the
    # touched-but-older k1. Without the touch, k1 would have been evicted.
    resolver._bounded_cache_store("table", ("t", 4), _frame(5_000))

    assert ("t", 1) in resolver._table_cache   # touched older entry survives
    assert ("t", 2) not in resolver._table_cache  # untouched newer entry evicted
    assert ("t", 3) in resolver._table_cache


def test_oversized_value_is_refused(monkeypatch):
    resolver = _resolver(monkeypatch, "1")
    big = _frame(1_000_000)  # tens of MB, > 1 MB budget
    stored = resolver._bounded_cache_store("table", ("big",), big)
    assert stored is False
    assert ("big",) not in resolver._table_cache
    assert ("table", ("big",)) not in resolver._cache_entry_bytes


def test_disabled_budget_never_evicts(monkeypatch):
    resolver = _resolver(monkeypatch, "0")  # disabled
    assert resolver._cache_budget_bytes is None
    for i in range(50):
        resolver._bounded_cache_store("table", ("t", i), _frame(5_000))
    assert len(resolver._table_cache) == 50
    assert resolver._cache_evictions == 0


def test_raw_alias_is_charged_once(monkeypatch):
    resolver = _resolver(monkeypatch, "64")
    table = ICUTable(
        data=_frame(1_000),
        id_columns=["stay_id"],
        index_column=None,
        value_column="value",
        unit_column=None,
    )
    resolver._store_raw_concept_cache(
        "hr", "hash1", table, aggregator="auto", store_legacy=True,
    )
    # Both the aggregator key and the legacy 2-tuple alias point at it,
    # but the object is only charged against the budget once.
    charged = [
        v for (role, key), v in resolver._cache_entry_bytes.items()
        if role == "raw"
    ]
    assert len(charged) == 2  # two keys tracked
    assert charged.count(0) == 1  # exactly one alias charged 0 bytes
    assert sum(1 for c in charged if c > 0) == 1


def test_clear_resets_accounting(monkeypatch):
    resolver = _resolver(monkeypatch, "8")
    for i in range(5):
        resolver._bounded_cache_store("table", ("t", i), _frame(2_000))
        resolver._bounded_cache_store("data", ("d", i), _frame(2_000))
    assert resolver._cache_total_bytes > 0

    resolver.clear_table_cache(keep_concept_cache=False)
    assert resolver._cache_total_bytes == 0
    assert len(resolver._cache_entry_bytes) == 0

    # keep_concept_cache=True only drops table/concept accounting.
    for i in range(5):
        resolver._bounded_cache_store("table", ("t", i), _frame(2_000))
        resolver._bounded_cache_store("data", ("d", i), _frame(2_000))
    resolver.clear_table_cache(keep_concept_cache=True)
    remaining_roles = {role for (role, _key) in resolver._cache_entry_bytes}
    assert "table" not in remaining_roles
    assert "data" in remaining_roles


def test_drop_source_caches_only_clears_sources(monkeypatch):
    resolver = _resolver(monkeypatch, "8")
    resolver._bounded_cache_store("table", ("t", 0), _frame(2_000))
    resolver._bounded_cache_store("raw", ("r", 0), _frame(2_000))
    resolver._bounded_cache_store("data", ("d", 0), _frame(2_000))

    resolver.drop_source_caches()
    assert not resolver._table_cache
    assert not resolver._raw_concept_cache
    assert resolver._concept_data_cache  # untouched
    roles = {role for (role, _key) in resolver._cache_entry_bytes}
    assert roles == {"data"}
