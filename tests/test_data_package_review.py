"""Focused contracts for the pre-Plan registered-export review owner."""

from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.webserver import data_package_review as review_owner


def _study(path: str) -> dict:
    return {
        "id": "study-review",
        "revision": 7,
        "data_source": {"path": path, "database": "miiv"},
        "cohort": {"label": "All eligible ICU stays"},
        "modules": ["outcome", "sepsis3_sofa2"],
        "execution_concepts": {
            "primary_exposure": "sep3_sofa2",
            "outcome": "death",
            "covariates": [],
        },
        "analysis_design": {"analysis_unit": "icu_stay"},
        "time_window": {"hours": 24, "anchor": "ICU admission"},
    }


def _aggregate(path: str) -> dict:
    return {
        "source": {
            "id": "src_demo",
            "label": "MIMIC-IV full export",
            "database": "miiv",
            "path_hash": "safe-hash",
        },
        "summary": {"cohort_size": 100},
        "coverage": [
            {
                "module": "sepsis3_sofa2",
                "metric_kind": "event_rate",
                "covered_entities": 40,
                "coverage_pct": 40.0,
            },
            {
                "module": "outcome",
                "metric_kind": "coverage",
                "covered_entities": 100,
                "coverage_pct": 100.0,
            },
        ],
        "quality": {"modules_ok": 1, "modules_neutral": 1},
    }


def _sealed_snapshot(**extra: object) -> dict:
    payload = {
        "schema_version": "easyicu.data-package-review/1",
        "status": "ready_for_plan",
        "study_context_id": "study-review",
        "study_context_revision": 7,
        "analysis_results_withheld": True,
        **extra,
    }
    payload["review_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_legacy_event_absence_is_not_misreported_as_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = str(tmp_path / "registered")
    registry = {
        "sources": [
            {
                "id": "src_demo",
                "path": source_path,
                "label": "MIMIC-IV full export",
                "database": "miiv",
                "ok": True,
                "modules": ["outcome", "sepsis3_sofa2"],
            }
        ]
    }
    monkeypatch.setattr(
        review_owner.cohort_review,
        "cohort_review_summary",
        lambda body: _aggregate(source_path),
    )
    monkeypatch.setattr(
        review_owner,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source=source_path,
            concepts=[
                CatalogConcept(
                    "sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                    n_rows=40,
                    column_role="event_status",
                    typed_metadata=False,
                ),
                CatalogConcept(
                    "death",
                    file_name="outcome.parquet",
                    n_rows=100,
                    column_role="event_status",
                    typed_metadata=False,
                ),
            ],
        ),
    )

    payload = review_owner.build_registered_data_package_review(
        _study(source_path), registry=registry
    )

    assert payload["status"] == "ready_for_plan"
    assert payload["denominator"]["count"] == 100
    exposure = payload["concepts"][0]
    assert exposure["evaluable_count"] == 100
    assert exposure["missing_count"] == 0
    assert exposure["absence_semantics"] == "no_recorded_event"
    assert exposure["physical_coverage_withheld"] is True
    assert payload["analysis_results_withheld"] is True
    encoded = json.dumps(payload)
    assert source_path not in encoded
    assert "event_rate_pct" not in encoded


def test_typed_sparse_event_without_receipt_blocks_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = str(tmp_path / "typed")
    registry = {
        "sources": [
            {
                "id": "src_typed",
                "path": source_path,
                "database": "miiv",
                "ok": True,
                "modules": ["outcome", "sepsis3_sofa2"],
            }
        ]
    }
    monkeypatch.setattr(
        review_owner.cohort_review,
        "cohort_review_summary",
        lambda body: _aggregate(source_path),
    )
    monkeypatch.setattr(
        review_owner,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source=source_path,
            concepts=[
                CatalogConcept(
                    "sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                    n_rows=40,
                    column_role="event_status",
                    typed_metadata=True,
                ),
                CatalogConcept(
                    "death",
                    file_name="outcome.parquet",
                    n_rows=100,
                    column_role="event_status",
                    typed_metadata=True,
                ),
            ],
        ),
    )

    payload = review_owner.build_registered_data_package_review(
        _study(source_path), registry=registry
    )

    assert payload["status"] == "blocked"
    assert "typed_event_availability_receipt_required" in payload["blocking_findings"]
    assert payload["concepts"][0]["evaluable_count"] is None


def test_review_digest_changes_with_study_revision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = str(tmp_path / "registered")
    registry = {
        "sources": [
            {
                "id": "src_demo",
                "path": source_path,
                "database": "miiv",
                "ok": True,
                "modules": ["outcome", "sepsis3_sofa2"],
            }
        ]
    }
    monkeypatch.setattr(
        review_owner.cohort_review,
        "cohort_review_summary",
        lambda body: _aggregate(source_path),
    )
    monkeypatch.setattr(
        review_owner,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source=source_path,
            concepts=[
                CatalogConcept("sep3_sofa2", file_name="sepsis3_sofa2.parquet", column_role="event_status"),
                CatalogConcept("death", file_name="outcome.parquet", column_role="event_status"),
            ],
        ),
    )
    first = review_owner.build_registered_data_package_review(
        _study(source_path), registry=registry
    )
    changed = {**_study(source_path), "revision": 8}
    second = review_owner.build_registered_data_package_review(
        changed, registry=registry
    )
    assert first["review_sha256"] != second["review_sha256"]


def test_review_snapshot_reopens_after_live_study_advances(tmp_path) -> None:
    payload = {
        "schema_version": "easyicu.data-package-review/1",
        "status": "ready_for_plan",
        "study_context_id": "study-review",
        "study_context_revision": 7,
        "denominator": {"analysis_unit": "icu_stay", "count": 100},
        "analysis_results_withheld": True,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["review_sha256"] = hashlib.sha256(encoded).hexdigest()
    store = review_owner.DataPackageReviewSnapshotStore(tmp_path / "snapshots")

    stored = store.persist(payload)
    reopened = store.load(
        study_id="study-review",
        revision=7,
        digest=payload["review_sha256"],
    )

    assert stored.exists()
    assert reopened == payload
    assert reopened["study_context_revision"] == 7


def test_review_snapshot_fails_closed_on_byte_drift(tmp_path) -> None:
    payload = {
        "schema_version": "easyicu.data-package-review/1",
        "status": "ready_for_plan",
        "study_context_id": "study-review",
        "study_context_revision": 7,
        "analysis_results_withheld": True,
    }
    payload["review_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    store = review_owner.DataPackageReviewSnapshotStore(tmp_path / "snapshots")
    path = store.persist(payload)
    tampered = dict(payload)
    tampered["status"] = "blocked"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(review_owner.DataPackageReviewError) as exc:
        store.load(
            study_id="study-review",
            revision=7,
            digest=payload["review_sha256"],
        )

    assert exc.value.code == "data_package_review_snapshot_digest_invalid"


@pytest.mark.parametrize(
    "unsafe",
    [
        {"source": {"path": "artifact.json"}},
        {"source": {"manifest_path": "artifact.json"}},
        {"source": {"candidate_paths": []}},
        {"source": {"label": "/private/export"}},
        {"source": {"label": r"C:\private\export"}},
        {"source": {"label": r"\private\export"}},
        {"source": {"label": "file:///private/export"}},
        {"source": {"label": "~/private/export"}},
    ],
)
def test_review_snapshot_persist_rejects_host_path_keys_and_values(
    tmp_path,
    unsafe: dict,
) -> None:
    payload = _sealed_snapshot(**unsafe)
    store = review_owner.DataPackageReviewSnapshotStore(tmp_path / "snapshots")

    with pytest.raises(review_owner.DataPackageReviewError) as exc:
        store.persist(payload)

    assert exc.value.code == "data_package_review_snapshot_path_forbidden"
    assert "private" not in json.dumps(exc.value.details)


def test_review_snapshot_allows_digest_url_and_artifact_basename(tmp_path) -> None:
    payload = _sealed_snapshot(
        source={
            "path_hash": "a" * 64,
            "url": "https://example.org/data-package",
            "artifact": "cohort_review.json",
        }
    )
    store = review_owner.DataPackageReviewSnapshotStore(tmp_path / "snapshots")

    store.persist(payload)

    assert store.load(
        study_id="study-review",
        revision=7,
        digest=payload["review_sha256"],
    ) == payload


def test_review_snapshot_load_rejects_path_shaped_external_bytes(tmp_path) -> None:
    payload = _sealed_snapshot(source={"label": "/private/export"})
    store = review_owner.DataPackageReviewSnapshotStore(tmp_path / "snapshots")
    path = store._path(
        "study-review",
        7,
        payload["review_sha256"],
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(review_owner.DataPackageReviewError) as exc:
        store.load(
            study_id="study-review",
            revision=7,
            digest=payload["review_sha256"],
        )

    assert exc.value.code == "data_package_review_snapshot_path_forbidden"
