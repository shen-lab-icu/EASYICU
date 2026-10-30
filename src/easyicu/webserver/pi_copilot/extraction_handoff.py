"""Owner for compiling Copilot study setup into Data Extraction input.

The conversation persists human-facing study slots against a registered module
export. Data Extraction executes against the export's sealed raw-source path.
This module owns that translation and decides whether an existing export truly
matches the requested cohort, window, format, and module contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from easyicu.webserver import dataio


@dataclass(frozen=True)
class ExtractionHandoff:
    source_data_path: str
    database: str
    modules: tuple[str, ...]
    export_format: str
    cohort: Mapping[str, Any]
    source_id: str
    reusable: bool
    mismatch_codes: tuple[str, ...]

    def public_receipt(self) -> dict[str, Any]:
        cohort_sha256 = hashlib.sha256(
            json.dumps(
                dict(self.cohort),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "easyicu.pi-extraction-handoff/1",
            "source_id": self.source_id[:80],
            "database": self.database[:40],
            "modules": list(self.modules),
            "export_format": self.export_format,
            "cohort_sha256": cohort_sha256,
            "reusable": self.reusable,
            "mismatch_codes": list(self.mismatch_codes),
        }


@dataclass(frozen=True)
class ExtractionTransaction:
    """Path-private outcome of one bound Data Extraction transaction."""

    reused: bool
    source_id: str
    handoff_receipt: Optional[Mapping[str, Any]]
    submitted: Optional[Mapping[str, Any]]


def compile_study_cohort(study: Mapping[str, Any]) -> dict[str, Any]:
    raw = study.get("cohort")
    cohort = dict(raw) if isinstance(raw, Mapping) else {}
    if "observation_window_hours" not in cohort:
        window = study.get("time_window")
        window = window if isinstance(window, Mapping) else {}
        hours = window.get("observation_hours")
        if hours is None:
            hours = window.get("hours")
        if hours is not None:
            cohort["observation_window_hours"] = hours
    return dataio.normalize_export_cohort_contract(cohort)


def _manifest_modules(manifest: Mapping[str, Any]) -> set[str]:
    modules = {
        str(row.get("module") or "").strip().lower()
        for row in (manifest.get("files") or [])
        if isinstance(row, Mapping) and str(row.get("module") or "").strip()
    }
    selection = manifest.get("concept_selection")
    selection = selection if isinstance(selection, Mapping) else {}
    selected_modules = selection.get("modules")
    if isinstance(selected_modules, Mapping):
        modules.update(str(key).strip().lower() for key in selected_modules if str(key).strip())
    return modules


def _explicit_execution_concepts(study: Mapping[str, Any]) -> set[str]:
    execution = study.get("execution_concepts")
    if not isinstance(execution, Mapping):
        return set()
    result = {
        value.strip()
        for field in ("primary_exposure", "outcome")
        if isinstance((value := execution.get(field)), str) and value.strip()
    }
    covariates = execution.get("covariates")
    if isinstance(covariates, list):
        result.update(value.strip() for value in covariates if isinstance(value, str) and value.strip())
    return result


def _explicit_selection_mismatches(
    study: Mapping[str, Any], manifest: Mapping[str, Any], requested_modules: tuple[str, ...]
) -> tuple[str, ...]:
    # A module name proves neither that a partial selection included the
    # requested variable nor that the source could actually provide it.
    required = _explicit_execution_concepts(study)
    selection = manifest.get("concept_selection")
    if not required:
        return ()
    if not isinstance(selection, Mapping) or selection.get("mode") not in ("explicit", "all_in_selected_modules"):
        return ("registered_export_concepts_unrecorded",)
    modules = selection.get("modules")
    selected_by_module = {
        str(module).strip().lower(): {value.strip() for value in values if isinstance(value, str) and value.strip()}
        for module, values in (modules.items() if isinstance(modules, Mapping) else ())
        if str(module).strip().lower() in requested_modules and isinstance(values, list)
    }
    published: set[str] = set()
    availability_by_concept: dict[str, set[str]] = {}
    for row in manifest.get("files") or ():
        if not isinstance(row, Mapping) or not row.get("file"):
            continue
        module = str(row.get("module") or "").strip().lower()
        concepts = row.get("concept_ids")
        if module not in selected_by_module or not isinstance(concepts, list):
            continue
        published.update(
            value.strip() for value in concepts
            if isinstance(value, str) and value.strip() in selected_by_module[module]
        )
        statuses = row.get("concept_status")
        if isinstance(statuses, Mapping):
            for concept, status in statuses.items():
                if isinstance(status, Mapping):
                    availability_by_concept.setdefault(str(concept), set()).add(
                        str(status.get("availability") or "")
                    )
    availability = manifest.get("concept_availability")
    unavailable = availability.get("structurally_unavailable") if isinstance(availability, Mapping) else None
    unavailable_ids = {
        str(row.get("concept_id") or "").strip()
        for row in (unavailable if isinstance(unavailable, list) else ())
        if isinstance(row, Mapping)
    }
    # The catalog owner treats a concept with only these statuses across all
    # of its files as unavailable, even when the manifest lists its ID.
    unavailable_ids.update(
        concept for concept, statuses in availability_by_concept.items()
        if statuses and statuses <= {"produced_all_null", "structurally_unavailable_placeholder"}
    )
    if required & unavailable_ids:
        return ("registered_export_concepts_unavailable",)
    missing = required - published
    if missing:
        selected_modules = set(selected_by_module)
        # Once all requested modules were exported in full, extracting the
        # same modules again cannot prove an absent or differently named ID.
        # Alias normalization here only detects a possible unresolved mapping;
        # it never establishes clinical equivalence or permits reuse.
        modules_complete = set(requested_modules).issubset(selected_modules)
        if modules_complete and selection.get("mode") == "all_in_selected_modules":
            return ("registered_export_concepts_unresolved",)
        from easyicu.research_agent.concept_availability import normalize_concept_name

        if modules_complete and any(
            normalize_concept_name(name) != name
            and normalize_concept_name(name) in published
            for name in missing
        ):
            return ("registered_export_concepts_unresolved",)
        return ("registered_export_concepts_missing",)
    return ()


def _verify_study_source(study: Mapping[str, Any], binding: Mapping[str, Any], registered_path: str) -> None:
    source = study.get("data_source")
    if not isinstance(source, Mapping):
        return
    requested_database = str(source.get("database") or "").strip()
    if requested_database:
        try:
            same_database = dataio.normalize_database_key(requested_database) == dataio.normalize_database_key(str(binding["database"]))
        except KeyError as exc:
            raise dataio.ExportCohortError("registered_export_study_database_mismatch") from exc
        if not same_database:
            raise dataio.ExportCohortError("registered_export_study_database_mismatch")
    source_path = str(source.get("path") or "").strip()
    if not source_path:
        return
    try:
        study_path = Path(source_path).expanduser().resolve(strict=True)
        export_path = Path(registered_path).expanduser().resolve(strict=True)
        raw_path = Path(str(binding["source_data_path"])).resolve(strict=True)
    except OSError as exc:
        raise dataio.ExportCohortError("registered_export_study_source_unavailable") from exc
    if study_path in (export_path, raw_path):
        return
    # A newly produced export may replace the previously bound export only
    # when both resolve to the same reviewed raw data source.
    try:
        old = dataio.resolve_registered_export_binding(str(study_path), str(binding["database"]))
    except dataio.ExportCohortError as exc:
        raise dataio.ExportCohortError("registered_export_study_source_mismatch") from exc
    if Path(str(old["source_data_path"])).resolve(strict=True) != raw_path:
        raise dataio.ExportCohortError("registered_export_study_source_mismatch")


def bound_export_mismatches(
    study: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    """Why a prepared export does not hold the rows this study states.

    The manifest records the cohort contract the export was extracted for and
    the rule that executed it.  A study whose cohort changed after extraction,
    or that is bound to an export prepared for another cohort, states one
    population while its rows are another's; Data Extraction answers which
    parts of a contract decide the rows.  The reuse decision below and the
    research launch read the same answer.
    """

    observed = manifest.get("cohort_contract")
    requested_rows = dataio.export_rows_decided_by(compile_study_cohort(study))
    observed_rows = dataio.export_rows_decided_by(
        observed if isinstance(observed, Mapping) else None
    )
    mismatches: list[str] = []
    if requested_rows != observed_rows:
        mismatches.append("registered_export_cohort_mismatch")
    if not dataio.export_cohort_execution_current(manifest):
        # The same contract executed under an earlier rule selects or scores
        # different rows; Data Extraction owns which rows still agree.
        mismatches.append("registered_export_cohort_execution_outdated")
    return tuple(mismatches)


def compile_registered_export_handoff(
    study: Mapping[str, Any],
    registered_source: Mapping[str, Any],
) -> ExtractionHandoff:
    """Compile a path-private handoff and exact reuse decision."""

    database = str(
        registered_source.get("database")
        or (study.get("data_source") or {}).get("database")
        or ""
    ).strip()
    binding = dataio.resolve_registered_export_binding(
        str(registered_source.get("path") or ""), database
    )
    _verify_study_source(study, binding, str(registered_source.get("path") or ""))
    manifest = binding["manifest"]
    requested_cohort = compile_study_cohort(study)
    requested_format = str(study.get("export_format") or "parquet").strip().lower()
    observed_format = str(manifest.get("format") or "").strip().lower()
    observed_modules = _manifest_modules(manifest)
    # A study that states no modules took this export's modules as its data;
    # extracting its own cohort keeps them.
    requested_modules = tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in (study.get("modules") or [])
            if str(value).strip()
        )
    ) or tuple(sorted(observed_modules))

    mismatches = list(bound_export_mismatches(study, manifest))
    if requested_format != observed_format:
        mismatches.append("registered_export_format_mismatch")
    if not set(requested_modules).issubset(observed_modules):
        mismatches.append("registered_export_modules_missing")
    mismatches.extend(_explicit_selection_mismatches(study, manifest, requested_modules))
    # For an explicit execution request, an observed window cannot be
    # reconstructed from an old export with no longitudinal time authority.
    # The cohort window is a membership rule, not a measurement bound.
    if _explicit_execution_concepts(study) and study.get("time_window"):
        from easyicu.research_agent.intake.export_package import (
            ExportPackageError,
            require_export_time_window_authority,
        )

        try:
            require_export_time_window_authority(manifest, modules=requested_modules)
        except ExportPackageError:
            mismatches.append("registered_export_time_window_unbounded")

    return ExtractionHandoff(
        source_data_path=str(binding["source_data_path"]),
        database=str(binding["database"]),
        modules=requested_modules,
        export_format=requested_format,
        cohort=requested_cohort,
        source_id=str(registered_source.get("id") or ""),
        reusable=not mismatches,
        mismatch_codes=tuple(mismatches),
    )


def study_binding_for_export(
    study: Mapping[str, Any], registered_source: Optional[Mapping[str, Any]]
) -> Optional[dict[str, str]]:
    """The data source a study binds when this export is its own extraction.

    An extraction submitted for a study registers a new export, while the
    study stays bound to the source it was extracted from.  A study whose
    bound export cannot serve its cohort is told to extract that cohort; once
    it has, its binding must follow, or the launch keeps refusing the old
    export.  The study binds the new export only when the export holds the
    cohort, window and format its setup requests and every module it
    requests (the reuse decision above).  A study whose setup changed
    meanwhile, or an extraction made for another setup, keeps its binding.
    """

    if not isinstance(registered_source, Mapping) or not registered_source.get("ok"):
        return None
    path = str(registered_source.get("path") or "").strip()
    if not path:
        return None
    try:
        handoff = compile_registered_export_handoff(study, registered_source)
    except dataio.ExportCohortError:
        return None
    if not handoff.reusable:
        return None
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    label = str(registered_source.get("label") or source.get("label") or "").strip()
    return {"path": path, "database": handoff.database, "label": label[:160]}


def submit_study_extraction(
    *,
    study: Mapping[str, Any],
    registered_source: Optional[Mapping[str, Any]],
    submit: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> ExtractionTransaction:
    """Reuse or submit the exact extraction contract bound to ``study``.

    The caller adapts transport errors and projects the path-free result.  This
    owner keeps raw paths, registered-export migration, and the submitted job
    body inside one transaction so adapters cannot rebuild the contract.
    """

    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    source_path = str(source.get("path") or "").strip()
    database = str(source.get("database") or "").strip()
    modules = [
        str(item) for item in (study.get("modules") or []) if str(item).strip()
    ]
    export_format = str(study.get("export_format") or "parquet").strip().lower()
    handoff_receipt: Optional[Mapping[str, Any]] = None
    registered_export_path: Optional[str] = None

    if registered_source is not None:
        handoff = compile_registered_export_handoff(study, registered_source)
        handoff_receipt = handoff.public_receipt()
        if handoff.reusable:
            return ExtractionTransaction(
                reused=True,
                source_id=str(registered_source.get("id") or "")[:80],
                handoff_receipt=handoff_receipt,
                submitted=None,
            )
        if "registered_export_concepts_unavailable" in handoff.mismatch_codes:
            raise dataio.ExportCohortError(
                "registered_export_concepts_unavailable_review_source",
                {"next_action": "review_source_or_concept_definition"},
            )
        if "registered_export_concepts_unresolved" in handoff.mismatch_codes:
            raise dataio.ExportCohortError(
                "registered_export_concepts_unresolved_review_mapping",
                {"next_action": "review_concept_mapping_or_selected_modules"},
            )
        if any(code in handoff.mismatch_codes for code in (
            "registered_export_concepts_missing",
            "registered_export_concepts_unrecorded",
        )):
            # A failed new export is not rebound to the study; retrying from
            # the old export would otherwise enqueue the same job forever.
            raise dataio.ExportCohortError(
                "registered_export_concepts_missing_review_extraction",
                {"next_action": "prepare_and_bind_export_with_required_concepts"},
            )
        if not handoff.modules:
            raise dataio.ExportCohortError("registered_export_modules_unrecorded")
        registered_export_path = source_path
        source_path = handoff.source_data_path
        database = handoff.database
        modules = list(handoff.modules)
        export_format = handoff.export_format
        cohort = dict(handoff.cohort)
    else:
        cohort = dict(study.get("cohort") or {})
        window = study.get("time_window")
        window = window if isinstance(window, Mapping) else {}
        if "observation_window_hours" not in cohort:
            hours = window.get("observation_hours")
            if hours is None:
                hours = window.get("hours")
            if hours is not None:
                cohort["observation_window_hours"] = hours

    submitted = submit(
        {
            "path": source_path,
            "registered_export_path": registered_export_path,
            "database": database,
            "modules": modules,
            "format": export_format,
            "merge": True,
            "cohort": cohort,
            "max_patients": cohort.get("max_patients"),
            "include_feature_definitions": True,
            "label": study.get("title"),
            "study_context_id": study["id"],
            "study_context_revision": int(study.get("revision") or 0),
        }
    )
    return ExtractionTransaction(
        reused=False,
        source_id="",
        handoff_receipt=handoff_receipt,
        submitted=dict(submitted),
    )


__all__ = [
    "ExtractionHandoff",
    "ExtractionTransaction",
    "bound_export_mismatches",
    "compile_registered_export_handoff",
    "compile_study_cohort",
    "study_binding_for_export",
    "submit_study_extraction",
]
