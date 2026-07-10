"""Behavior contracts for bounded Cross-DB progress and cancellation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import easyicu
from easyicu import cohort_visualization
from easyicu.webserver import crossdb_review
from easyicu.webserver.app import app


class _FakeDistribution:
    def __init__(self, *, data_root: str, language: str) -> None:
        self.root = Path(data_root)
        assert language == "en"

    def _get_db_path(self, database: str) -> Path:
        names = {"miiv": "mimiciv", "eicu": "eicu", "aumc": "aumc"}
        return self.root / names.get(database, database)


def _patch_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cohort_visualization, "MultiDatabaseDistribution", _FakeDistribution
    )


def _frame_for(concepts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {concept: [float(index + 1)] for index, concept in enumerate(concepts)}
    )


def test_raw_loader_emits_bounded_database_and_chunk_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "databases"
    for folder in ("mimiciv", "eicu"):
        (root / folder).mkdir(parents=True)
    _patch_distribution(monkeypatch)
    calls: list[tuple[str, ...]] = []

    def fake_load(*, concepts: list[str], **_kwargs: object) -> pd.DataFrame:
        calls.append(tuple(concepts))
        return _frame_for(concepts)

    monkeypatch.setattr(easyicu, "load_concepts", fake_load)
    events: list[dict] = []
    concepts = [f"feature_{index}" for index in range(25)]

    frames = crossdb_review._load_raw_feature_data(
        data_root=str(root),
        concepts=concepts,
        databases=["miiv", "eicu"],
        max_patients=40,
        sample_size=100,
        emit_progress=events.append,
        should_cancel=lambda: False,
    )

    assert [len(call) for call in calls] == [24, 1, 24, 1]
    assert list(frames) == ["miiv", "eicu"]
    completed_chunks = [
        event["completed_chunks"]
        for event in events
        if event.get("phase") == "chunk"
        and event.get("chunk_status") == "complete"
    ]
    assert completed_chunks == [1, 2, 3, 4]
    completed_databases = [
        event["current"]
        for event in events
        if event.get("phase") == "database"
        and event.get("database_status") == "complete"
    ]
    assert completed_databases == [1, 2]
    assert {event.get("total_chunks") for event in events} == {4}
    serialized = json.dumps(events)
    for marker in (str(root), "subject_id", "stay_id", '"value"', '"series"'):
        assert marker not in serialized


def test_raw_loader_stops_before_next_chunk_after_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    _patch_distribution(monkeypatch)
    calls: list[tuple[str, ...]] = []
    cancel = False

    def fake_load(*, concepts: list[str], **_kwargs: object) -> pd.DataFrame:
        calls.append(tuple(concepts))
        return _frame_for(concepts)

    def emit(event: dict) -> None:
        nonlocal cancel
        if event.get("phase") == "chunk" and event.get("chunk_status") == "complete":
            cancel = True

    monkeypatch.setattr(easyicu, "load_concepts", fake_load)

    with pytest.raises(crossdb_review._CrossdbRawCancelled) as exc_info:
        crossdb_review._load_raw_feature_data(
            data_root=str(root),
            concepts=[f"feature_{index}" for index in range(25)],
            databases=["miiv"],
            max_patients=40,
            sample_size=100,
            emit_progress=emit,
            should_cancel=lambda: cancel,
        )

    assert len(calls) == 1
    assert len(calls[0]) == 24
    assert exc_info.value.phase == "chunk"
    assert exc_info.value.completed_chunks == 1


def test_raw_loader_checks_cancel_inside_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    _patch_distribution(monkeypatch)
    calls: list[tuple[str, ...]] = []
    cancel = False

    def fake_load(*, concepts: list[str], **_kwargs: object) -> pd.DataFrame:
        nonlocal cancel
        calls.append(tuple(concepts))
        if len(concepts) > 1:
            raise RuntimeError("force bounded fallback")
        cancel = True
        return _frame_for(concepts)

    monkeypatch.setattr(easyicu, "load_concepts", fake_load)

    with pytest.raises(crossdb_review._CrossdbRawCancelled) as exc_info:
        crossdb_review._load_raw_feature_data(
            data_root=str(root),
            concepts=["feature_a", "feature_b"],
            databases=["miiv"],
            max_patients=40,
            sample_size=100,
            should_cancel=lambda: cancel,
        )

    assert calls == [("feature_a", "feature_b"), ("feature_a",)]
    assert exc_info.value.phase == "fallback"


def test_raw_loader_fails_closed_on_operational_concept_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    _patch_distribution(monkeypatch)

    def fail_load(**_kwargs: object) -> pd.DataFrame:
        raise RuntimeError("private path must not escape")

    monkeypatch.setattr(easyicu, "load_concepts", fail_load)

    with pytest.raises(crossdb_review.CrossdbReviewError) as exc_info:
        crossdb_review._load_raw_feature_data(
            data_root=str(root),
            concepts=["hr", "sbp"],
            databases=["miiv"],
            max_patients=40,
            sample_size=100,
        )

    assert exc_info.value.detail["error"] == "raw_database_concept_load_failed"
    assert exc_info.value.detail["failed_concept"] == "hr"
    assert "private path" not in json.dumps(exc_info.value.detail)


def test_raw_run_rejects_missing_requested_database_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "databases"
    for folder in ("mimiciv", "eicu"):
        (root / folder).mkdir(parents=True)

    def fail_if_called(**_kwargs: object) -> dict:
        raise AssertionError("loader must not run with a missing requested database")

    monkeypatch.setattr(crossdb_review, "_load_raw_feature_data", fail_if_called)
    response = TestClient(app).post(
        "/api/crossdb-review/raw-distribution",
        json={
            "data_root": str(root),
            "databases": ["miiv", "eicu", "aumc"],
            "features": ["hr"],
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "requested_raw_databases_not_found"
    assert detail["requested_databases"] == ["miiv", "eicu", "aumc"]
    assert detail["detected_databases"] == ["miiv", "eicu"]
    assert detail["missing_databases"] == ["aumc"]
