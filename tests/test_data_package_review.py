"""Focused contracts for the pre-Plan registered-export review owner."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.webserver import data_package_review as review_owner
from easyicu.webserver import data_package_execution_readiness as readiness_owner


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


def test_typed_sparse_event_with_sealed_binary_receipt_is_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = str(tmp_path / "typed")
    registry = {
        "sources": [{
            "id": "src_typed", "path": source_path, "database": "miiv",
            "ok": True, "modules": ["outcome", "sepsis3_sofa2"],
        }]
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
                    "sep3_sofa2", file_name="sepsis3_sofa2.parquet",
                    n_rows=40, column_role="event_status", typed_metadata=True,
                ),
                CatalogConcept(
                    "death", file_name="outcome.parquet", n_rows=100,
                    column_role="event_status", typed_metadata=True,
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        review_owner,
        "read_exported_concept",
        lambda _path, concept: pd.DataFrame(
            {
                "stay_id": range(40 if concept == "sep3_sofa2" else 100),
                concept: [True] * (40 if concept == "sep3_sofa2" else 100),
            }
        ),
    )

    payload = review_owner.build_registered_data_package_review(
        _study(source_path), registry=registry
    )

    assert payload["status"] == "ready_for_plan"
    exposure = payload["concepts"][0]
    assert exposure["reason_code"] == "typed_event_availability_verified"
    assert exposure["evaluable_count"] == 100
    assert exposure["availability_receipt"]["event_count_withheld"] is True


def test_typed_event_receipt_rejects_nonbinary_physical_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concept = CatalogConcept(
        "sep3_sofa2", file_name="sepsis3_sofa2.parquet",
        column_role="event_status", typed_metadata=True,
    )
    monkeypatch.setattr(
        review_owner,
        "read_exported_concept",
        lambda _path, _concept: pd.DataFrame(
            {"stay_id": [1], "sep3_sofa2": [2]}
        ),
    )

    assert review_owner._typed_event_availability_receipt(
        source_path="/sealed/export", concept=concept, denominator=100
    ) is None


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


def test_review_proves_eligible_denominator_grouping_and_landmark_inputs(
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
                "modules": [
                    "demographics",
                    "outcome",
                    "sepsis3_sofa2",
                ],
            }
        ]
    }
    aggregate = _aggregate(source_path)
    aggregate["coverage"].append(
        {
            "module": "demographics",
            "metric_kind": "coverage",
            "covered_entities": 100,
            "coverage_pct": 100.0,
        }
    )
    monkeypatch.setattr(
        review_owner.cohort_review,
        "cohort_review_summary",
        lambda _body: aggregate,
    )
    monkeypatch.setattr(
        review_owner,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source=source_path,
            concepts=[
                CatalogConcept("age", file_name="demographics.parquet"),
                CatalogConcept(
                    "sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                    column_role="event_status",
                ),
                CatalogConcept(
                    "death", file_name="outcome.parquet", column_role="event_status"
                ),
                CatalogConcept("los_icu", file_name="outcome.parquet"),
                CatalogConcept(
                    "icu_readmission",
                    file_name="outcome.parquet",
                    column_role="event_status",
                ),
            ],
        ),
    )

    def read_concept(_root, concept_id):
        if concept_id == "age":
            return pd.DataFrame(
                {"stay_id": range(100), "age": [17, *([18] * 99)]}
            )
        if concept_id == "death":
            return pd.DataFrame(
                {"stay_id": [1, 2], "charttime": [8.0, None], "death": [1, 0]}
            )
        if concept_id == "sep3_sofa2":
            return pd.DataFrame(
                {"stay_id": [1], "charttime": [4.0], "sep3_sofa2": [True]}
            )
        raise KeyError(concept_id)

    monkeypatch.setattr(readiness_owner, "read_exported_concept", read_concept)
    monkeypatch.setattr(
        readiness_owner.source_identity_authority,
        "resolve_patient_grouping_authority",
        lambda **_kwargs: PatientGroupingBinding(
            mapping_path=tmp_path / "mapping.parquet",
            mapping_sha256="a" * 64,
            mapping_stay_column="stay_id",
            mapping_patient_column="subject_id",
            authority_coordinates={
                "authority_ref": "test/patient-groups/v1",
                "export_manifest_sha256": "b" * 64,
                "grouping_derivation": "prefix_before_:s",
            },
        ),
    )
    study = {
        **_study(source_path),
        "cohort": {"label": "Adults", "age_min": 18, "exclude_readmissions": False},
        "modules": ["demographics", "outcome", "sepsis3_sofa2"],
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "sensitivity_specs": [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
                "require_alive_at_landmark": True,
                "exclude_negative_event_times": True,
            },
            {
                "spec_id": "non_readmission",
                "axis": "repeated_stays",
                "strategy": "non_readmission_restriction",
                "execution_variables": ["icu_readmission"],
            },
        ],
    }

    payload = review_owner.build_registered_data_package_review(study, registry=registry)

    assert payload["schema_version"] == "easyicu.data-package-review/2"
    assert payload["status"] == "ready_for_plan"
    assert payload["eligible_denominator"] == {
        "status": "ready",
        "count": 99,
        "basis": "typed_age_eligibility",
        "age_min": 18.0,
        "age_max": None,
        "missing_age_count": 0,
        "excluded_by_age_count": 1,
    }
    readiness = payload["runtime_readiness"]
    assert readiness["status"] == "ready"
    assert readiness["patient_grouping"]["output_identity_column"] == (
        "patient_stay_id"
    )
    assert readiness["outcome_event_time"]["materialized_column"] == "death_time"
    assert readiness["observation_duration"]["unit"] == "days"
    assert readiness["readmission_indicator"]["status"] == "ready"
    assert source_path not in json.dumps(payload)


def test_required_landmark_time_missing_blocks_package_review(
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
        lambda _body: _aggregate(source_path),
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
                    column_role="event_status",
                ),
                CatalogConcept(
                    "death", file_name="outcome.parquet", column_role="event_status"
                ),
                CatalogConcept("los_icu", file_name="outcome.parquet"),
            ],
        ),
    )
    monkeypatch.setattr(
        readiness_owner,
        "read_exported_concept",
        lambda _root, concept_id: pd.DataFrame(
            {"stay_id": [1], concept_id: [True]}
        ),
    )
    study = {
        **_study(source_path),
        "sensitivity_specs": [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
            }
        ],
    }

    payload = review_owner.build_registered_data_package_review(study, registry=registry)

    assert payload["status"] == "blocked"
    assert "landmark_outcome_event_time_unavailable" in payload["blocking_findings"]
    assert "landmark_exposure_time_unavailable" in payload["blocking_findings"]


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
