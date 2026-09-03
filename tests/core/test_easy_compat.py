"""Contracts for the deprecated :mod:`easyicu.easy` compatibility shim."""

from __future__ import annotations

import importlib
import warnings

import pandas as pd
import pytest


def test_importing_easy_shim_does_not_warn() -> None:
    import easyicu.easy as legacy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(legacy)

    assert not [item for item in caught if item.category is DeprecationWarning]


def test_legacy_loader_warns_at_call_time_and_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.easy as legacy

    calls: list[tuple[object, dict[str, object]]] = []

    def fake_load_concepts(concepts, **kwargs):
        calls.append((concepts, kwargs))
        return pd.DataFrame({"stay_id": [1], "hr": [80.0]})

    monkeypatch.setattr(legacy, "load_concepts", fake_load_concepts)

    with pytest.warns(DeprecationWarning, match="EasyICU 2.0"):
        result = legacy.load_vitals(
            "/tmp/icu",
            patient_ids=[1],
            database="miiv",
            interval_hours=2.0,
            concepts=["hr"],
        )

    assert list(result["hr"]) == [80.0]
    assert calls == [
        (
            ["hr"],
            {
                "patient_ids": [1],
                "database": "miiv",
                "data_path": "/tmp/icu",
                "interval": pd.Timedelta(hours=2),
                "verbose": False,
            },
        )
    ]


def test_quick_summary_reports_partial_failures_without_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.easy as legacy

    def fake_load_concepts(concepts, **kwargs):
        del kwargs
        if concepts == ["hr"]:
            return pd.DataFrame({"hr": [80.0, 82.0]})
        if concepts == "sofa":
            raise PermissionError("patient 123 should never appear in output")
        if concepts == "sep3":
            return pd.DataFrame({"sep3": [True, False]})
        raise FileNotFoundError("/private/source/path")

    monkeypatch.setattr(legacy, "load_concepts", fake_load_concepts)

    with pytest.warns(DeprecationWarning):
        summary = legacy.quick_summary("/tmp/icu", patient_ids=[1, 2])

    assert summary == {
        "patients": 2,
        "vitals_records": 2,
        "lab_records": 0,
        "sofa_mean": None,
        "sepsis_positive": 1,
        "errors": {
            "labs": "FileNotFoundError",
            "sofa": "PermissionError",
        },
    }
    assert "123" not in repr(summary)
    assert "/private/source/path" not in repr(summary)
