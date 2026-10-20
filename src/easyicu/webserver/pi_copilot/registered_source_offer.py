"""Session-scoped registered-export offers and their explicit source choice.

The catalog owns recommendation policy; the existing StudyContext update
owner owns binding. This adapter validates every refusal before either the
StudyContext or the session authorization is changed.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from easyicu.databases.profiles import get_database_profile
from easyicu.webserver import sources

from . import study_context_update
from .contracts import PiCopilotError, PiSessionRecord, ToolExecutionContext
from .source_recommendation import registered_source_choice, registered_sources_for_database


def registered_source_offer(
    record: PiSessionRecord, database: Optional[str], *, registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Read path-free choices only while this research session awaits consent."""

    if record.agent_mode != "research":
        raise PiCopilotError(
            "pi_session_data_source_authorization_not_required",
            "Workspace conversations do not use the research data-source gate.", status_code=409,
        )
    if record.data_source_authorization.status != "pending":
        raise PiCopilotError(
            "pi_session_data_source_authorization_not_pending",
            "Refresh this conversation before choosing its registered data source.", status_code=409,
        )
    try:
        profile = get_database_profile(str(database or "").strip())
        if not profile.is_public:
            raise KeyError(database)
    except KeyError as exc:
        raise PiCopilotError(
            "pi_database_not_supported", "Choose one exact supported full-database family.",
            status_code=422,
        ) from exc
    _, recommendation = registered_sources_for_database(
        sources.load_registry() if registry is None else registry, profile.key,
    )
    return {
        "database": profile.key,
        "recommended": (registered_source_choice(recommendation.source)
                        if recommendation.source is not None else None),
        "choices": [{"choice": index, **registered_source_choice(row)}
                    for index, row in enumerate(recommendation.choices, start=1)],
        "reason": recommendation.reason,
    }


def bind_offered_registered_source(
    record: PiSessionRecord, study: Mapping[str, Any], *, database: Optional[str],
    source_id: Optional[str], stale: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind an eligible choice through the same owner as the Copilot tool."""

    registry = sources.load_registry()
    offer = registered_source_offer(record, database, registry=registry)
    if study.get("active_job_id"):
        raise PiCopilotError(
            "study_context_active_job_conflict",
            "Study setup cannot change while its authoritative EasyICU job is active.", status_code=409,
        )
    if stale.get("stale"):
        raise PiCopilotError(
            "pi_session_authority_stale",
            "The study changed before source selection. Refresh this conversation and choose again.",
            status_code=409, details=dict(stale),
        )
    selected_id = str(source_id or "").strip()
    if not selected_id:
        raise PiCopilotError("pi_source_id_required", "Choose one exact registered source id.", status_code=422)
    source = next((row for row in registry.get("sources") or []
                   if isinstance(row, Mapping) and row.get("ok") and row.get("id") == selected_id), None)
    if source is None:
        raise PiCopilotError(
            "pi_data_source_not_registered", "Choose a validated registered EasyICU export.", status_code=409,
        )
    try:
        profile = get_database_profile(str(source.get("database") or ""))
        family = profile.parent_key or profile.key
    except KeyError:
        family = None
    if family != offer["database"]:
        raise PiCopilotError(
            "pi_data_source_database_mismatch", "Choose an export of the requested database family.",
            status_code=409,
        )
    eligible = ([offer["recommended"]] if offer["recommended"] else offer["choices"])
    if selected_id not in {row["source_id"] for row in eligible}:
        raise PiCopilotError(
            "pi_data_source_not_offered", "This export is no longer offered. Refresh and choose again.",
            status_code=409,
        )

    # Capture the owner's persisted context directly. Rendering the full
    # workflow is a later read: it must not introduce a post-binding refusal.
    updated: Dict[str, Any] = {}

    def capture_context(_context: ToolExecutionContext, *, study_override: Mapping[str, Any]) -> dict:
        updated.update(study_override)
        return {}

    result = study_context_update.update_study_context(
        ToolExecutionContext(session=record, allowed_actions={"configure"}),
        {"bind_source_id": selected_id},
        load_context=lambda _binding: dict(study), project_workflow=capture_context,
        source_registry=registry,
    )
    if result["status"] != "ok":
        raise PiCopilotError(result["code"], result["summary"], status_code=409)
    return updated
