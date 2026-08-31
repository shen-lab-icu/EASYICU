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
    manifest = binding["manifest"]
    requested_cohort = compile_study_cohort(study)
    observed_cohort = dataio.normalize_export_cohort_contract(
        manifest.get("cohort_contract")
        if isinstance(manifest.get("cohort_contract"), Mapping)
        else None
    )
    requested_modules = tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in (study.get("modules") or [])
            if str(value).strip()
        )
    )
    requested_format = str(study.get("export_format") or "parquet").strip().lower()
    observed_format = str(manifest.get("format") or "").strip().lower()
    observed_modules = _manifest_modules(manifest)

    mismatches: list[str] = []
    if requested_cohort != observed_cohort:
        mismatches.append("registered_export_cohort_mismatch")
    if requested_format != observed_format:
        mismatches.append("registered_export_format_mismatch")
    if not set(requested_modules).issubset(observed_modules):
        mismatches.append("registered_export_modules_missing")

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
    "compile_registered_export_handoff",
    "compile_study_cohort",
    "submit_study_extraction",
]
