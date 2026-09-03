"""Contract tests for the registered export selection owner."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from easyicu.webserver import cohort_review, extraction_filters, sources
from easyicu.webserver import patient_drilldown


@pytest.mark.parametrize(
    ("requested", "expected_error"),
    [
        (None, "no_active_export"),
        ("/missing/export", "source_not_registered"),
    ],
)
def test_selection_fails_closed_without_registered_export(
    monkeypatch: pytest.MonkeyPatch,
    requested: str | None,
    expected_error: str,
) -> None:
    monkeypatch.setattr(
        sources,
        "load_registry",
        lambda: {"sources": [], "active_path": None},
    )

    with pytest.raises(sources.RegisteredExportSelectionError) as exc_info:
        sources.resolve_registered_export(requested)

    assert exc_info.value.detail["error"] == expected_error
    if requested:
        assert len(exc_info.value.detail["path_hash"]) == 12
        assert "path" not in exc_info.value.detail


def test_selection_validates_description_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_path = str(tmp_path / "registered")
    source = {"id": "source-1", "path": export_path, "database": "miiv"}
    calls: list[str] = []
    monkeypatch.setattr(
        sources,
        "load_registry",
        lambda: {"sources": [source], "active_path": export_path},
    )

    def describe(path: str):
        calls.append(path)
        return {"ok": True, "path": path, "database": "miiv"}

    monkeypatch.setattr(sources.dataio, "describe_export_source", describe)

    selection = sources.resolve_registered_export()

    assert selection.source == source
    assert selection.description["database"] == "miiv"
    assert calls == [export_path]


def test_selection_rejects_unregistered_active_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sources,
        "load_registry",
        lambda: {"sources": [], "active_path": "/stale/export"},
    )

    with pytest.raises(sources.RegisteredExportSelectionError) as exc_info:
        sources.resolve_registered_export()

    assert exc_info.value.detail["error"] == "active_source_not_registered"
    assert len(exc_info.value.detail["path_hash"]) == 12


def test_selection_rejects_invalid_registered_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {"id": "source-1", "path": "/registered/export"}
    monkeypatch.setattr(
        sources,
        "load_registry",
        lambda: {"sources": [source], "active_path": source["path"]},
    )
    monkeypatch.setattr(
        sources.dataio,
        "describe_export_source",
        lambda _path: {"ok": False, "error": "manifest_missing"},
    )

    with pytest.raises(sources.RegisteredExportSelectionError) as exc_info:
        sources.resolve_registered_export()

    assert exc_info.value.detail == {
        "error": "invalid_export",
        "detail": "manifest_missing",
    }


@pytest.mark.parametrize(
    "adapter",
    [
        cohort_review._resolve_registered_source,
        extraction_filters._resolve_registered_source,
        patient_drilldown._resolve_registered_source,
    ],
)
def test_review_adapters_delegate_complete_selection(adapter) -> None:
    implementation = inspect.getsource(adapter)

    assert implementation.count("resolve_registered_export(") == 1
    assert "load_registry(" not in implementation
    assert "describe_export_source(" not in implementation
