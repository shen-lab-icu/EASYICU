"""A path-private choice of an official export's own raw planning source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver import dataio, demo_source_storage, study_contexts
from easyicu.webserver.demo_source_contracts import SOURCES
from easyicu.webserver.research_launch_scientific import bound_export_cohort_refusal

from .contracts import PiCopilotError, PiSessionRecord
from .population_preparation import has_prior_run, study_for_next_plan

ACTION = "use_official_demo_raw_for_plan"


def _planning_study(study: Mapping[str, Any]) -> Mapping[str, Any]:
    """The study the workflow's next plan reads, by its own run predicate."""
    return study_for_next_plan(study, after_run=has_prior_run(study))


@dataclass(frozen=True)
class DemoPlanningSource:
    raw_path: str
    database: str
    label: str
    offer: Mapping[str, Any]


def resolve_demo_planning_source(study: Mapping[str, Any], *, again: bool = False) -> DemoPlanningSource | None:
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
        refusal = bound_export_cohort_refusal(_planning_study(study), str(export_path))
        # A first choice answers a population the export does not hold. Asked
        # again (``again``), the settings may since match that export: the
        # same raw choice still serves; an unexecutable cohort never does.
        if (refusal.code != "research_pipeline_export_cohort_mismatch" if refusal is not None
                else not again):
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


def raw_choice_needs_reconfirmation(study: Mapping[str, Any]) -> bool:
    """A study bound to an official Demo's raw folder without a choice that resolves.

    Only the signed choice lets a plan read that folder; without it a launch
    would validate the raw folder as an export package and stop
    (manifest_marker_missing). The coordinate, not a full verification, marks
    the folder, so a raw folder that no longer verifies is still caught here.
    """
    source = study.get("data_source")
    if not isinstance(source, Mapping):
        return False
    raw = str(source.get("path") or "")
    if not raw:
        return False
    candidate = Path(raw).expanduser().absolute()
    if not any(candidate == demo_source_storage.source_paths(spec).raw for spec in SOURCES):
        return False
    from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
    from .demo_planning_authority import resolve_choice

    try:
        return resolve_choice(study) is None
    except (ValueError, OSError, KeyError, TypeError, dataio.ExportCohortError, AuthorityFilesystemError):
        return True


def _current_official_export(source: Mapping[str, Any]) -> str | None:
    """The current, ready official export of exactly this official raw folder.

    Only the allowlisted release whose cache coordinate is this raw folder and
    whose database matches, and only when the running producer's export of it
    is ready. Any other folder has none.
    """
    raw = Path(str(source.get("path") or "")).expanduser().absolute()
    database = str(source.get("database") or "")
    for spec in SOURCES:
        try:
            paths = demo_source_storage.source_paths(spec)
            if raw == paths.raw and spec.database == database and demo_source_storage.export_ready(paths, spec):
                return str(paths.export)
        except (OSError, ValueError):
            return None
    return None


def _reconfirmation_study(study: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The study as its choice is offered: bound to the export the choice is made from.

    That is the export the study's earlier receipt named. A study bound to an
    official raw folder without any receipt (B32: bound through "redefine a
    new cohort from its raw data", 2026-11-20) is offered the first choice from
    the current verified export of that same folder; the full
    ``resolve_demo_planning_source`` checks still run, and nothing is signed
    until the researcher confirms.
    """
    source = study.get("data_source")
    if not isinstance(source, Mapping) or study.get("active_job_id"):
        return None
    if not raw_choice_needs_reconfirmation(study):
        return None
    from .demo_planning_authority import prior_choice_export

    export = prior_choice_export(study) or _current_official_export(source)
    if export is None:
        return None
    return {**dict(study), "data_source": {**dict(source), "path": export}}


def resolve_demo_planning_reconfirmation(study: Mapping[str, Any]) -> DemoPlanningSource | None:
    """The same raw choice offered again on the current configuration, if it still holds."""
    offer_study = _reconfirmation_study(study)
    return None if offer_study is None else resolve_demo_planning_source(offer_study, again=True)

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
    # An already-chosen study whose choice a configuration write cleared is
    # offered the same choice again, from the export its receipt named.
    reconfirmation = _reconfirmation_study(study)
    offer_study = reconfirmation or study
    resolved = resolve_demo_planning_source(offer_study, again=reconfirmation is not None)
    if (resolved is None or not isinstance(offer_sha256, str) or not offer_sha256
            or not hmac.compare_digest(str(resolved.offer["offer_sha256"]), offer_sha256)):
        raise PiCopilotError(
            "pi_demo_planning_source_unavailable",
            "The exact official Demo source is no longer available. Refresh its source choice.",
            status_code=409,
        )
    confirmations = dict(study.get("confirmations") or {})
    confirmations["extraction_completed"] = False
    from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
    from .demo_planning_authority import publish_choice

    new_source = {"path": resolved.raw_path, "database": resolved.database, "label": resolved.label}
    # The choice is signed on the study the next plan is drafted on; before a
    # first run the launch would otherwise add the question's stated
    # first-stay scope afterwards and find this receipt stale.
    drafted = _planning_study({**study, "data_source": new_source})
    drafted_scope = {"cohort": drafted["cohort"]} if drafted.get("cohort") != study.get("cohort") else {}
    try:
        authority_ref = publish_choice(study=offer_study, new_source=new_source, confirmations=confirmations,
                                       offer_sha256=offer_sha256, drafted_scope=drafted_scope,
                                       again=reconfirmation is not None)
        return study_contexts.upsert_context(
            {"id": study["id"], "data_source": new_source, "confirmations": confirmations,
             "demo_planning_authority_ref": authority_ref, **drafted_scope},
            expected_revision=expected_revision, require_revision=True, lifecycle_write=False,
            _server_demo_planning_authority_write=True,
        )
    except study_contexts.StudyContextError as exc:
        raise PiCopilotError(
            "pi_demo_planning_source_stale",
            "The study changed before its source could be saved. Refresh and choose again.",
            status_code=409,
        ) from exc
    except (ValueError, OSError, dataio.ExportCohortError, AuthorityFilesystemError) as exc:
        raise PiCopilotError("pi_demo_planning_source_unavailable",
                             "The exact selected Demo source could not be verified.", status_code=409) from exc
