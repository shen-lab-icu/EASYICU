"""Safety contracts for the deprecated ``ConceptLoader`` compatibility path."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


legacy_loader_module = importlib.import_module("easyicu.load_concepts")


def _loader(*, low_memory: bool = False):
    loader = legacy_loader_module.ConceptLoader.__new__(
        legacy_loader_module.ConceptLoader
    )
    loader._table_cache = {}
    loader._src_name = "miiv"
    loader.data_path = "/prepared"
    loader._low_memory = low_memory
    return loader


def test_projection_io_error_does_not_retry_with_full_table(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(legacy_loader_module, "load_table", fail)

    with pytest.raises(OSError, match="storage failure"):
        _loader()._safe_load_table("chartevents", ["stay_id", "charttime"])

    assert len(calls) == 1
    assert calls[0]["columns"] == ["stay_id", "charttime"]


def test_only_explicit_missing_column_error_can_use_compatibility_fallback(
    monkeypatch,
):
    calls = []
    expected = pd.DataFrame({"stay_id": [1], "charttime": [0.0]})

    def load(*args, **kwargs):
        calls.append(kwargs)
        if "columns" in kwargs:
            raise KeyError("Columns ['optional'] not found in table 'events'")
        return expected

    monkeypatch.setattr(legacy_loader_module, "load_table", load)

    result = _loader()._safe_load_table("events", ["stay_id", "optional"])

    assert result is expected
    assert len(calls) == 2
    assert "columns" not in calls[1]


def test_low_memory_mode_never_retries_missing_projection_as_full_table(
    monkeypatch,
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise KeyError("Columns ['optional'] not found in table 'events'")

    monkeypatch.setattr(legacy_loader_module, "load_table", fail)

    with pytest.raises(KeyError, match="optional"):
        _loader(low_memory=True)._safe_load_table(
            "events", ["stay_id", "optional"]
        )

    assert len(calls) == 1


def test_missing_table_key_error_is_not_misclassified_as_missing_column(
    monkeypatch,
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise KeyError("No table source registered for 'events'")

    monkeypatch.setattr(legacy_loader_module, "load_table", fail)

    with pytest.raises(KeyError, match="No table source"):
        _loader()._safe_load_table("events", ["stay_id"])

    assert len(calls) == 1
