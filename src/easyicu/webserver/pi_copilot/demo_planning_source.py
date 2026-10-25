"""A path-private choice of an official export's own raw planning source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver import dataio, demo_source_storage, study_contexts
from easyicu.webserver.research_launch_scientific import bound_export_cohort_refusal

from .contracts import PiCopilotError, PiSessionRecord

ACTION = "use_official_demo_raw_for_plan"


@dataclass(frozen=True)
class DemoPlanningSource:
    raw_path: str
    database: str
    label: str
    offer: Mapping[str, Any]


def resolve_demo_planning_source(study: Mapping[str, Any]) -> DemoPlanningSource | None:
    """Offer only a pinned cache export whose exact raw provenance still holds.

    The offer binds the study revision and the source receipts, not any client
    path. It grants source selection only; package and plan execution gates
    continue to apply after selection.
    """
    source = study.get("data_source")
    if not isinstance(source, Mapping) or study.get("active_job_id"):
        return None
    export_path = Path(str(source.get("path") or "")).expanduser().absolute()
    database = str(source.get("database") or "")
    historical = demo_source_storage.verified_public_demo_replanning_export(export_path)
    if historical is None:
        return None
    candidate, paths = historical
    if candidate.database != database:
        return None
    try:
        refusal = bound_export_cohort_refusal(study, str(export_path))
        if refusal is None or refusal.code != "research_pipeline_export_cohort_mismatch":
            return None
        marker = demo_source_storage.read_marker(paths.prepared_marker, candidate)
        if (export_path.resolve() != export_path
                or not marker or marker.get("archive_sha256") != candidate.archive_sha256):
            return None
        binding = dataio.resolve_registered_export_binding(str(export_path), database)
        raw_path = Path(binding["source_data_path"])
        if (raw_path != paths.raw
                or demo_source_storage.verified_public_demo_raw(raw_path) != candidate):
            return None
        revision = study.get("revision")
        if type(revision) is not int or revision < 1 or not study.get("id"):
            return None
        label = f"{candidate.title} v{candidate.version}"
        offer = {
            "action": ACTION,
            "study_id": study["id"],
            "expected_revision": revision,
            "database": database,
            "label": label,
        }
        identity = {
            **offer,
            "source": dict(source),
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
            "export_manifest": binding["manifest"],
            "prepared_marker": marker,
            "extracted_marker": demo_source_storage.read_marker(paths.extracted_marker, candidate),
            "converted_marker": demo_source_storage.read_marker(paths.converted_marker, candidate),
            "conversion_sha256": hashlib.sha256(
                (raw_path / "conversion_manifest.json").read_bytes()
            ).hexdigest(),
        }
        return DemoPlanningSource(str(raw_path), database, label,
                                  {**offer, "offer_sha256": canonical_sha256(identity)})
    except (OSError, ValueError, TypeError, KeyError, dataio.ExportCohortError):
        return None


def bind_demo_planning_source(
    record: PiSessionRecord,
    study: Mapping[str, Any],
    *,
    expected_revision: int | None,
    offer_sha256: str | None,
    stale: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume the displayed choice with the StudyContext owner's CAS write."""
    if (stale.get("stale") or study.get("active_job_id") or record.active_message_job_id
            or record.binding.study_context_id != study.get("id")
            or type(expected_revision) is not int
            or expected_revision != study.get("revision")
            or record.binding.study_revision != expected_revision):
        raise PiCopilotError(
            "pi_demo_planning_source_stale",
            "The study changed. Refresh before choosing its official Demo source.", status_code=409,
        )
    resolved = resolve_demo_planning_source(study)
    if (resolved is None or not isinstance(offer_sha256, str) or not offer_sha256
            or not hmac.compare_digest(str(resolved.offer["offer_sha256"]), offer_sha256)):
        raise PiCopilotError(
            "pi_demo_planning_source_unavailable",
            "The exact official Demo source is no longer available. Refresh its source choice.",
            status_code=409,
        )
    confirmations = dict(study.get("confirmations") or {})
    confirmations["extraction_completed"] = False
    try:
        return study_contexts.upsert_context(
            {"id": study["id"], "data_source": {
                "path": resolved.raw_path, "database": resolved.database, "label": resolved.label,
            }, "confirmations": confirmations},
            expected_revision=expected_revision, require_revision=True, lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        raise PiCopilotError(
            "pi_demo_planning_source_stale",
            "The study changed before its source could be saved. Refresh and choose again.",
            status_code=409,
        ) from exc
