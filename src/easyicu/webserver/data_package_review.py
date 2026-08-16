"""Project-scoped, aggregate review of a registered EasyICU data package.

This owner bridges the existing registered-source catalog and Cohort Review
aggregate without exposing patient rows, host paths, or scientific results.
It is the conversational checkpoint between extraction/reuse and planning.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from easyicu.research_agent.acquisition.catalog import build_available_catalog
from easyicu.webserver import state_paths
from easyicu.webserver import cohort_review, sources
from easyicu.webserver.data_package_execution_readiness import (
    build_data_package_execution_readiness,
)


class DataPackageReviewError(RuntimeError):
    """Owner-attributable failure while compiling a package review."""

    def __init__(
        self, code: str, message: str, *, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})


class DataPackageReviewSnapshotStore:
    """Persist immutable aggregate review receipts for conversation replay.

    The registered export and StudyContext remain the scientific owners.  This
    store only retains the already path-free aggregate receipt produced by this
    module so a historical Pi message can reopen the exact review it linked to
    after the live StudyContext advances.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else state_paths.state_root() / "data-package-reviews"
        )
        self._lock = threading.RLock()

    @staticmethod
    def _coordinates(payload: Mapping[str, Any]) -> tuple[str, int, str]:
        study_id = str(payload.get("study_context_id") or "").strip()
        revision = payload.get("study_context_revision")
        digest = str(payload.get("review_sha256") or "").strip().lower()
        if (
            not study_id
            or not isinstance(revision, int)
            or revision < 0
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise DataPackageReviewError(
                "data_package_review_snapshot_coordinates_invalid",
                "The aggregate review is missing valid immutable coordinates.",
            )
        return study_id, revision, digest

    @staticmethod
    def _verify_digest(payload: Mapping[str, Any], expected: str) -> None:
        canonical = dict(payload)
        canonical.pop("review_sha256", None)
        actual = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if actual != expected:
            raise DataPackageReviewError(
                "data_package_review_snapshot_digest_invalid",
                "The aggregate review does not match its owner digest.",
                details={"expected_sha256": expected, "actual_sha256": actual},
            )

    @staticmethod
    def _verify_path_free(payload: Mapping[str, Any]) -> None:
        """Reject host-path keys and values before durable replay storage."""

        def fail(location: tuple[str, ...], reason: str) -> None:
            raise DataPackageReviewError(
                "data_package_review_snapshot_path_forbidden",
                "The aggregate review snapshot contains a host-path-shaped field.",
                details={
                    "field": ".".join(location)[:500],
                    "reason": reason,
                },
            )

        def host_path_value(value: str) -> bool:
            clean = value.strip()
            lowered = clean.lower()
            windows_drive = (
                len(clean) >= 3
                and clean[0].isalpha()
                and clean[1] == ":"
                and clean[2] in {"/", "\\"}
            )
            return bool(
                clean.startswith("/")
                or clean.startswith("\\")
                or windows_drive
                or lowered.startswith("file://")
                or clean == "~"
                or clean.startswith("~/")
                or clean.startswith("~\\")
            )

        def visit(value: Any, location: tuple[str, ...]) -> None:
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    key = str(raw_key)
                    normalized = key.strip().lower()
                    child_location = (*location, key)
                    if (
                        normalized == "path"
                        or normalized.endswith("_path")
                        or normalized.endswith("_paths")
                    ):
                        fail(child_location, "path_key")
                    visit(child, child_location)
                return
            if isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, (*location, str(index)))
                return
            if isinstance(value, Path):
                fail(location, "path_object")
            if isinstance(value, str) and host_path_value(value):
                fail(location, "absolute_path_value")

        visit(payload, ())

    def _path(self, study_id: str, revision: int, digest: str) -> Path:
        study_key = hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:24]
        return self.root / study_key / f"r{revision}-{digest}.json"

    def persist(self, payload: Mapping[str, Any]) -> Path:
        study_id, revision, digest = self._coordinates(payload)
        self._verify_path_free(payload)
        self._verify_digest(payload, digest)
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        if len(encoded) > 512 * 1024:
            raise DataPackageReviewError(
                "data_package_review_snapshot_too_large",
                "The aggregate review exceeds its bounded snapshot contract.",
            )
        path = self._path(study_id, revision, digest)
        with self._lock:
            if path.exists():
                try:
                    existing = path.read_bytes()
                except OSError as exc:
                    raise DataPackageReviewError(
                        "data_package_review_snapshot_unreadable",
                        "The existing aggregate review snapshot cannot be read.",
                    ) from exc
                if existing != encoded:
                    raise DataPackageReviewError(
                        "data_package_review_snapshot_identity_drift",
                        "An immutable aggregate review coordinate already has different bytes.",
                    )
                return path
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix=".data-package-review-",
                suffix=".tmp",
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                with handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o600)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def load(self, *, study_id: str, revision: int, digest: str) -> Dict[str, Any]:
        coordinates = {
            "study_context_id": str(study_id),
            "study_context_revision": int(revision),
            "review_sha256": str(digest).lower(),
        }
        clean_id, clean_revision, clean_digest = self._coordinates(coordinates)
        path = self._path(clean_id, clean_revision, clean_digest)
        try:
            if path.stat().st_size > 512 * 1024:
                raise DataPackageReviewError(
                    "data_package_review_snapshot_too_large",
                    "The aggregate review snapshot exceeds its bounded contract.",
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DataPackageReviewError(
                "data_package_review_snapshot_not_found",
                "The immutable aggregate review snapshot is unavailable.",
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataPackageReviewError(
                "data_package_review_snapshot_unreadable",
                "The aggregate review snapshot cannot be read.",
            ) from exc
        if not isinstance(payload, dict):
            raise DataPackageReviewError(
                "data_package_review_snapshot_invalid",
                "The aggregate review snapshot has an invalid shape.",
            )
        self._verify_path_free(payload)
        actual_id, actual_revision, actual_digest = self._coordinates(payload)
        if (actual_id, actual_revision, actual_digest) != (
            clean_id,
            clean_revision,
            clean_digest,
        ):
            raise DataPackageReviewError(
                "data_package_review_snapshot_scope_mismatch",
                "The aggregate review snapshot belongs to different coordinates.",
            )
        self._verify_digest(payload, clean_digest)
        return payload


def _normalized_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path)


def _safe_label(value: Any, *, database: Any) -> str:
    label = " ".join(str(value or "").split())[:240]
    if not label or "/" in label or "\\" in label:
        return f"{str(database or 'EasyICU').upper()} registered export"
    return label


def _execution_concepts(study: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw = study.get("execution_concepts")
    raw = raw if isinstance(raw, Mapping) else {}
    out: list[tuple[str, str]] = []
    for role in ("primary_exposure", "outcome"):
        concept = str(raw.get(role) or "").strip()
        if concept:
            out.append((role, concept))
    covariates = raw.get("covariates")
    if isinstance(covariates, list):
        out.extend(
            ("covariate", str(value).strip())
            for value in covariates
            if str(value).strip()
        )
    # Preserve the scientific role order while reviewing one physical concept
    # only once. Reusing a concept in two roles is itself visible in StudyContext.
    return list(dict.fromkeys(out))


def _review_concept(
    *,
    role: str,
    concept_id: str,
    catalog_by_id: Mapping[str, Any],
    coverage_by_module: Mapping[str, Mapping[str, Any]],
    denominator: int,
) -> Dict[str, Any]:
    concept = catalog_by_id.get(concept_id)
    if concept is None:
        return {
            "study_role": role,
            "concept_id": concept_id,
            "availability_status": "not_extracted",
            "reason_code": "data_package_concept_not_extracted",
            "evaluable_count": None,
            "denominator_count": denominator,
            "missing_count": None,
        }

    module = Path(str(concept.file_name or "")).stem
    coverage = coverage_by_module.get(module, {})
    covered = coverage.get("covered_entities")
    covered = int(covered) if isinstance(covered, int) else None
    role_name = str(concept.column_role or "value")
    base: Dict[str, Any] = {
        "study_role": role,
        "concept_id": concept_id,
        "module": module,
        "column_role": role_name,
        "typed_metadata": bool(concept.typed_metadata),
        "denominator_count": denominator,
        "physical_coverage_kind": str(
            coverage.get("metric_kind") or "coverage"
        ),
    }

    if role_name == "event_status" and not concept.typed_metadata:
        # The acquisition owner deterministically declares legacy EVENT_STATUS
        # concepts as positive-only events. Missing physical rows therefore
        # mean no recorded event, not missing measurement. Exact event counts
        # and rates stay withheld until the governed analysis run.
        return {
            **base,
            "availability_status": "ready",
            "reason_code": "legacy_positive_only_event_semantics_verified",
            "evaluable_count": denominator,
            "missing_count": 0,
            "absence_semantics": "no_recorded_event",
            "physical_coverage_pct": None,
            "physical_coverage_withheld": True,
            "interpretation": (
                "Legacy event-status absence is normalized to no event by the "
                "data-foundation owner; it is not treated as missingness."
            ),
        }

    if role_name == "event_status" and covered != denominator:
        # Typed sparse status columns require their owner-issued availability
        # receipts. A module-level row count alone cannot prove evaluability.
        return {
            **base,
            "availability_status": "semantic_review_required",
            "reason_code": "typed_event_availability_receipt_required",
            "evaluable_count": None,
            "missing_count": None,
            "physical_coverage_pct": coverage.get("coverage_pct"),
            "interpretation": (
                "Typed event-status coverage is sparse; an owner-issued "
                "availability receipt is required before planning."
            ),
        }

    if covered is None:
        return {
            **base,
            "availability_status": "semantic_review_required",
            "reason_code": "concept_entity_coverage_unavailable",
            "evaluable_count": None,
            "missing_count": None,
            "physical_coverage_pct": None,
        }

    covered = min(max(covered, 0), denominator)
    missing = denominator - covered
    return {
        **base,
        "availability_status": "ready" if missing == 0 else "partial",
        "reason_code": (
            "concept_fully_observed"
            if missing == 0
            else "concept_missingness_requires_plan"
        ),
        "evaluable_count": covered,
        "missing_count": missing,
        "physical_coverage_pct": coverage.get("coverage_pct"),
        "interpretation": (
            "Observed entity coverage; any missingness must be handled in the "
            "reviewed analysis plan."
        ),
    }


def build_registered_data_package_review(
    study: Mapping[str, Any],
    *,
    registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile a path-free, result-blind package review for one StudyContext."""

    study_id = str(study.get("id") or "").strip()
    if not study_id:
        raise DataPackageReviewError(
            "data_package_study_required",
            "A bound StudyContext is required to review the data package.",
        )
    source_config = study.get("data_source")
    source_config = source_config if isinstance(source_config, Mapping) else {}
    source_path = _normalized_path(source_config.get("path"))
    if not source_path:
        raise DataPackageReviewError(
            "data_package_source_required",
            "The bound StudyContext has no registered export source.",
        )

    registry = registry or sources.load_registry()
    registered = next(
        (
            row
            for row in (registry.get("sources") or [])
            if isinstance(row, Mapping)
            and row.get("ok")
            and _normalized_path(row.get("path")) == source_path
        ),
        None,
    )
    if registered is None:
        raise DataPackageReviewError(
            "data_package_source_not_registered",
            "The StudyContext source is not a validated registered EasyICU export.",
        )

    try:
        aggregate = cohort_review.cohort_review_summary({"source_path": source_path})
        catalog = build_available_catalog(source_path)
    except cohort_review.CohortReviewError as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        raise DataPackageReviewError(
            str(detail.get("error") or "data_package_aggregate_review_failed"),
            "The registered export aggregate review failed closed.",
            details={
                key: detail.get(key)
                for key in ("error", "reason", "supported_scope")
                if detail.get(key) is not None
            },
        ) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise DataPackageReviewError(
            "data_package_catalog_review_failed",
            "The registered export catalog could not be verified.",
            details={"error_type": type(exc).__name__},
        ) from exc

    summary = aggregate.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    denominator = summary.get("cohort_size")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise DataPackageReviewError(
            "data_package_denominator_unavailable",
            "The registered export has no verified aggregate denominator.",
        )

    coverage_rows = aggregate.get("coverage")
    coverage_rows = coverage_rows if isinstance(coverage_rows, list) else []
    coverage_by_module = {
        str(row.get("module") or ""): row
        for row in coverage_rows
        if isinstance(row, Mapping) and row.get("module")
    }
    catalog_by_id = {str(item.concept_id): item for item in catalog.concepts}
    requested = _execution_concepts(study)
    concepts = [
        _review_concept(
            role=role,
            concept_id=concept_id,
            catalog_by_id=catalog_by_id,
            coverage_by_module=coverage_by_module,
            denominator=denominator,
        )
        for role, concept_id in requested
    ]
    blocking = [
        row
        for row in concepts
        if row.get("availability_status")
        in {"not_extracted", "semantic_review_required"}
    ]
    configured_modules = [
        str(value).strip()
        for value in (study.get("modules") or [])
        if str(value).strip()
    ]
    available_modules = {
        str(value).strip()
        for value in (registered.get("modules") or [])
        if str(value).strip()
    }
    module_review = [
        {
            "module": module,
            "availability_status": (
                "ready" if module in available_modules else "not_extracted"
            ),
        }
        for module in dict.fromkeys(configured_modules)
    ]
    blocking.extend(
        {
            "availability_status": "not_extracted",
            "reason_code": f"configured_module_not_extracted:{row['module']}",
        }
        for row in module_review
        if row.get("availability_status") == "not_extracted"
    )
    if not requested:
        blocking.append(
            {
                "availability_status": "semantic_review_required",
                "reason_code": "execution_concepts_required",
            }
        )

    source_projection = aggregate.get("source")
    source_projection = (
        dict(source_projection) if isinstance(source_projection, Mapping) else {}
    )
    source_projection["label"] = _safe_label(
        registered.get("label") or source_projection.get("label"),
        database=registered.get("database"),
    )
    source_projection.pop("path", None)
    quality = aggregate.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    execution_readiness = build_data_package_execution_readiness(
        study,
        source_path=source_path,
        catalog_by_id=catalog_by_id,
        registered_denominator=denominator,
    )
    eligibility = execution_readiness["eligible_denominator"]
    runtime_readiness = execution_readiness["runtime_readiness"]
    blocking.extend(
        {
            "availability_status": "semantic_review_required",
            "reason_code": reason,
        }
        for reason in runtime_readiness["required_findings"]
    )
    if eligibility.get("status") != "ready":
        blocking.append(
            {
                "availability_status": "semantic_review_required",
                "reason_code": str(
                    eligibility.get("reason_code")
                    or "cohort_eligibility_denominator_unavailable"
                ),
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": "easyicu.data-package-review/2",
        "status": "blocked" if blocking else "ready_for_plan",
        "code": (
            "easyicu_data_package_review_blocked"
            if blocking
            else "easyicu_data_package_review_ready"
        ),
        "study_context_id": study_id,
        "study_context_revision": int(study.get("revision") or 0),
        "source": source_projection,
        "denominator": {
            "analysis_unit": str(
                (study.get("analysis_design") or {}).get("analysis_unit")
                if isinstance(study.get("analysis_design"), Mapping)
                else "icu_stay"
            )
            or "icu_stay",
            "count": denominator,
            "basis": "registered_export_distinct_entity_aggregate",
        },
        "eligible_denominator": eligibility,
        "runtime_readiness": runtime_readiness,
        "cohort_review": {
            "label": str((study.get("cohort") or {}).get("label") or "")[:1000]
            if isinstance(study.get("cohort"), Mapping)
            else "",
            "time_window": dict(study.get("time_window") or {})
            if isinstance(study.get("time_window"), Mapping)
            else {},
        },
        "configured_modules": module_review,
        "concepts": concepts,
        "blocking_findings": [
            str(row.get("reason_code") or "data_package_review_blocked")
            for row in blocking
        ],
        "quality": {
            key: quality.get(key)
            for key in (
                "modules_ok",
                "modules_warn",
                "modules_bad",
                "modules_unknown",
                "watchlist_count",
                "median_coverage_pct",
            )
            if quality.get(key) is not None
        },
        "analysis_results_withheld": True,
        "result_fields_withheld": [
            "event_counts",
            "event_rates",
            "group_comparisons",
            "effect_estimates",
        ],
        "privacy": {
            "raw_rows_returned": False,
            "direct_identifiers_returned": False,
            "host_paths_returned": False,
            "secrets_returned": False,
        },
        "provenance": {
            "owner": "easyicu.webserver.data_package_review",
            "inputs": [
                "registered_source_registry",
                "available_concept_catalog",
                "cohort_review_aggregate",
                "typed_study_context",
            ],
        },
    }
    digest_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["review_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return payload


__all__ = [
    "DataPackageReviewError",
    "DataPackageReviewSnapshotStore",
    "build_registered_data_package_review",
]
