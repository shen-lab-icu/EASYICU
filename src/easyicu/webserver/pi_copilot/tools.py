"""Capability-scoped EasyICU tools exposed to Pi AgentSession."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Set

from fastapi import HTTPException

from easyicu.databases.profiles import get_database_profile, iter_database_profiles
from easyicu.extensions import ExtensionRegistry, ExtensionRegistryError
from easyicu.extensions.mcp_client import call_mcp_tool

from easyicu.ai_optin import is_offline_llm_choice
from easyicu.research_agent.reporting.result_card import (
    build_result_interpretation_card,
)
from easyicu.webserver import (
    agent_pipeline_runs,
    agent_runs,
    capabilities,
    cohort_review,
    crossdb_review,
    dataio,
    demo_sources,
    jobs,
    literature_authority,
    patient_drilldown,
    settings,
    sources,
    study_contexts,
)
from easyicu.webserver.copilot_data_workbench import (
    CopilotDataWorkbenchError,
    CopilotDataWorkbenchSnapshotStore,
    build_snapshot as build_data_workbench_snapshot,
    project_patient_snapshot_payload,
)
from easyicu.webserver.ideas import mining as idea_mining

from .contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiToolResult,
    ToolExecutionContext,
    WorkspaceMutationLimitError,
)
from .extraction_handoff import (
    compile_registered_export_handoff,
)
from .literature_tool_projection import compile_literature_tool_projection
from .projections import (
    bounded_json_projection,
    ensure_safe_projection,
    path_digest,
    project_artifacts,
    project_capabilities,
    project_job,
    project_run_row,
    project_study_context,
    reject_sensitive_message,
    stable_code,
)
from .run_authority import (
    list_bound_run_history,
    research_pipeline_project_root,
)
from .workspace import WORKSPACE_ARTIFACT_AUTHORITY, ProjectWorkspace
from .workflow import (
    build_research_workflow_snapshot,
    registered_export_matches_study,
)

READ_TOOLS = frozenset(
    {
        "easyicu_workspace_status",
        "easyicu_list_data_sources",
        "easyicu_list_source_concepts",
        "easyicu_inspect_data_package",
        "easyicu_review_cohort",
        "easyicu_open_data_download",
        "easyicu_preview_icd_cohort",
        "easyicu_review_patient_timeline",
        "easyicu_compare_data_sources",
        "easyicu_inspect_workflow",
        "easyicu_inspect_context",
        "easyicu_inspect_plan",
        "easyicu_inspect_literature",
        "easyicu_inspect_capability",
        "easyicu_inspect_run",
        "easyicu_inspect_step",
        "easyicu_inspect_validation",
        "easyicu_list_artifacts",
        "easyicu_inspect_evidence",
        "easyicu_explain_blocker",
        "easyicu_inspect_interpretation",
        "easyicu_inspect_manuscript",
        "easyicu_resume",
        "easyicu_list_extensions",
        "easyicu_load_skill",
    }
)
CONTROL_TOOLS = frozenset(
    {
        "easyicu_update_study_context",
        "easyicu_mine_ideas",
        "easyicu_search_literature",
        "easyicu_prepare_idea_handoff",
        "easyicu_accept_idea_handoff",
        "easyicu_prepare_demo_source",
        "easyicu_start_extraction",
        "easyicu_run",
        "easyicu_cancel",
        "easyicu_request_replan",
        "easyicu_call_mcp_tool",
    }
)
WORKSPACE_TOOLS = frozenset(
    {
        "easyicu_list_project_files",
        "easyicu_read_project_file",
        "easyicu_write_project_file",
        "easyicu_edit_project_file",
        "easyicu_check_project_file",
        "easyicu_preview_project_file",
    }
)
ALLOWED_TOOLS = READ_TOOLS | CONTROL_TOOLS | WORKSPACE_TOOLS
MUTATING_HOST_TOOLS = CONTROL_TOOLS | frozenset(
    {"easyicu_write_project_file", "easyicu_edit_project_file"}
)
DATA_SOURCE_REQUIRED_TOOLS = frozenset(
    {
        "easyicu_list_source_concepts",
        "easyicu_inspect_data_package",
        "easyicu_review_cohort",
        "easyicu_open_data_download",
        "easyicu_preview_icd_cohort",
        "easyicu_review_patient_timeline",
        "easyicu_compare_data_sources",
        "easyicu_inspect_run",
        "easyicu_inspect_step",
        "easyicu_inspect_validation",
        "easyicu_list_artifacts",
        "easyicu_inspect_evidence",
        "easyicu_inspect_interpretation",
        "easyicu_inspect_manuscript",
        "easyicu_run",
        "easyicu_resume",
    }
)


def _bounded_model_text(value: Any, limit: int = 1_200) -> str:
    """Normalize one already-governed text field for a bounded tool result."""

    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    owner: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner=owner,
        details=bounded_json_projection(details or {}),
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")
    return ensure_safe_projection(payload)


def _workspace_result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return bounded project artifacts without applying patient projections.

    Workspace text is either model-authored or explicitly placed in the
    isolated project artifact directory.  It never reads EasyICU source-data
    paths.  Keep the JSON-line envelope bounded while preserving useful code.
    """

    safe_details = dict(details or {})
    encoded = json.dumps(safe_details, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > 30_000:
        raise PiCopilotError(
            "pi_workspace_projection_too_large",
            "The project artifact result exceeds the bounded Pi tool contract.",
            status_code=500,
            details={"bytes": len(encoded), "max_bytes": 30_000},
        )
    return PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner="easyicu.webserver.pi_copilot.workspace",
        details=safe_details,
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")


def _extension_result(
    context: ToolExecutionContext,
    *,
    status: str,
    code: str,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return registry-reviewed text or an MCP-client-sanitized projection.

    These values are user-installed advisory material, not patient-data
    projections.  Their owners already reject credentials, host paths,
    patient-identifier fields, oversized JSON, and non-allowlisted tools.
    Keep this envelope bounded without treating ordinary words in a writing
    Skill (for example, "patient identifiers") as if they were raw rows.
    """

    safe_details = dict(details or {})
    encoded = json.dumps(safe_details, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > 32_000:
        raise PiCopilotError(
            "pi_extension_projection_too_large",
            "The extension result exceeds the bounded Pi tool contract.",
            status_code=500,
            details={"bytes": len(encoded), "max_bytes": 32_000},
        )
    return PiToolResult(
        status=status,
        code=code,
        summary=summary[:2000],
        owner="easyicu.extensions",
        details=safe_details,
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")


def _consume_action(
    context: ToolExecutionContext, action: str
) -> Optional[Dict[str, Any]]:
    outcome = context.grant.consume_once(action)
    if outcome == "granted":
        return None
    if outcome == "consumed":
        return _result(
            context,
            status="blocked",
            code="pi_action_grant_consumed",
            summary=f"The one-use {action} grant for this message was already consumed.",
            owner="easyicu.webserver.pi_copilot",
        )
    return _result(
        context,
        status="blocked",
        code="pi_action_authorization_required",
        summary=f"This action requires a one-use {action} grant for the current message.",
        owner="easyicu.webserver.pi_copilot",
    )


def _require_args(
    params: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(params) - allowed_set)
    if unknown:
        raise PiCopilotError(
            "pi_tool_unknown_arguments",
            "The EasyICU tool rejected unknown arguments.",
            details={"fields": unknown},
        )
    missing = sorted(
        key
        for key in required
        if key not in params or not str(params.get(key) or "").strip()
    )
    if missing:
        raise PiCopilotError(
            "pi_tool_arguments_required",
            "The EasyICU tool is missing required arguments.",
            details={"fields": missing},
        )


def _bound_context(binding: AuthorityBinding) -> Optional[Dict[str, Any]]:
    if binding.study_context_id:
        try:
            return study_contexts.get_context(binding.study_context_id)
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_invalid"),
                "The bound StudyContext could not be loaded.",
                details=exc.detail,
            ) from exc
    return study_contexts.get_active_context()


def _run_rows(context: ToolExecutionContext) -> Sequence[Dict[str, Any]]:
    project_root = research_pipeline_project_root(
        context.session.binding.study_context_id
    )
    return list_bound_run_history(
        study_context_id=context.session.binding.study_context_id,
        project_root=project_root,
        limit=50,
    )


def _select_run(
    context: ToolExecutionContext, requested_run_id: Any = None
) -> Optional[Dict[str, Any]]:
    requested = str(requested_run_id or "").strip()
    rows = _run_rows(context)
    if requested:
        return next((row for row in rows if row.get("run_id") == requested), None)
    # The persisted session binding is a historical navigation coordinate.  A
    # newly submitted background job mints its pipeline run id later, so the
    # binding may legitimately lag during this turn.  Read tools therefore use
    # the run-history owner's newest row unless the caller named an exact run.
    return rows[0] if rows else None


def _run_review(row: Mapping[str, Any]) -> Dict[str, Any]:
    # project_dir remains inside this host process. It is never put in a
    # PiToolResult or returned to the model/browser.
    review = agent_runs.read_run_review(str(row.get("project_dir") or ""))
    if not review.get("ok"):
        raise PiCopilotError(
            str(review.get("error") or "pi_run_review_unavailable"),
            "The selected EasyICU run review is unavailable.",
            details={
                key: review.get(key)
                for key in ("error", "artifact")
                if review.get(key) is not None
            },
        )
    return review


def _artifact_resource(
    run_id: Any,
    artifact_name: Any,
    *,
    sha256: Any = None,
) -> Optional[Dict[str, Any]]:
    """Build a path-free browser reference to one whitelisted run artefact."""

    clean_run = stable_code(run_id)
    clean_name = str(artifact_name or "").strip()
    if (
        clean_run is None
        or not clean_name
        or len(clean_name) > 160
        or Path(clean_name).name != clean_name
        or not clean_name.endswith(".json")
    ):
        return None
    digest = str(sha256 or "").strip().lower()
    return {
        "kind": "research_artifact",
        "run_id": clean_run,
        "artifact": clean_name,
        "label": clean_name,
        "media_type": "application/json",
        **(
            {"sha256": digest}
            if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
            else {}
        ),
    }


def extraction_workspace_resource(
    study: Mapping[str, Any],
    *,
    state: str,
    job_id: Any = None,
    source_id: Any = None,
    expected_database: Any = None,
    entry_mode: Any = None,
    extraction_scope: Any = None,
) -> Dict[str, Any]:
    """Project the native Extraction owner without exposing its host paths."""

    resource: Dict[str, Any] = {
        "kind": "native_workspace",
        "route": "extraction",
        "state": state,
        "study_context_id": str(study.get("id") or "")[:160],
        "study_revision": int(study.get("revision") or 0),
        "label": "Data Extraction",
        "media_type": "application/vnd.easyicu.native-workspace",
    }
    clean_job_id = str(job_id or "").strip()
    if clean_job_id:
        resource["job_id"] = clean_job_id[:160]
    clean_source_id = str(source_id or "").strip()
    if clean_source_id:
        resource["source_id"] = clean_source_id[:80]
    if str(entry_mode or "").strip() == "source_binding":
        resource["entry_mode"] = "source_binding"
        resource["label"] = "Data source setup"
    clean_scope = str(extraction_scope or "").strip()
    if clean_scope in {"study_required", "all_supported", "reuse_prepared_full"}:
        resource["extraction_scope"] = clean_scope
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    database = str(expected_database or source.get("database") or "").strip()
    if database:
        try:
            profile = get_database_profile(database)
        except KeyError:
            profile = None
        if profile is not None and profile.is_public:
            resource["expected_database"] = profile.key
    return resource


def _document_resource(
    run_id: Any,
    document_name: Any,
    *,
    sha256: Any = None,
) -> Optional[Dict[str, Any]]:
    clean_run = stable_code(run_id)
    clean_name = str(document_name or "").strip()
    labels = {
        "manuscript_scaffold.pdf": "Rendered manuscript draft (PDF)",
        "manuscript_scaffold.tex": "LaTeX manuscript source",
        "manuscript_scaffold.bib": "BibTeX bibliography",
        "system_validation_report.html": "System validation dossier (HTML)",
        "system_validation_report.pdf": "System validation dossier (PDF)",
    }
    media_types = {
        "manuscript_scaffold.pdf": "application/pdf",
        "manuscript_scaffold.tex": "text/x-tex",
        "manuscript_scaffold.bib": "application/x-bibtex",
        "system_validation_report.html": "text/html",
        "system_validation_report.pdf": "application/pdf",
    }
    if clean_run is None or clean_name not in labels:
        return None
    digest = str(sha256 or "").strip().lower()
    return {
        "kind": (
            "system_validation_document"
            if clean_name.startswith("system_validation_report.")
            else "research_document"
        ),
        "run_id": clean_run,
        "artifact": clean_name,
        "label": labels[clean_name],
        "media_type": media_types[clean_name],
        **(
            {"sha256": digest}
            if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
            else {}
        ),
    }


def _artifact_resources(
    run_id: Any, artifacts: Iterable[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    resources = []
    for artifact in list(artifacts)[:80]:
        resource = _artifact_resource(
            run_id,
            artifact.get("name"),
            sha256=artifact.get("sha256"),
        )
        if resource is None:
            resource = _document_resource(
                run_id,
                artifact.get("name"),
                sha256=artifact.get("sha256"),
            )
        if resource is not None:
            resources.append(resource)
    return resources


def _plan_projection(payload: Mapping[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps")
    steps = steps if isinstance(steps, list) else []
    projected_steps = []
    for index, raw in enumerate(steps[:50]):
        row = raw if isinstance(raw, Mapping) else {}
        projected_steps.append(
            {
                key: row.get(key)
                for key in (
                    "id",
                    "step_id",
                    "title",
                    "intent",
                    "method",
                    "status",
                    "depends_on",
                    "evidence_ids",
                    "literature_citation_keys",
                    "output_type",
                )
                if row.get(key) is not None
            }
            | {"position": index + 1}
        )
    return bounded_json_projection(
        {
            key: payload.get(key)
            for key in (
                "run_id",
                "study_id",
                "research_question",
                "provider",
                "execution",
                "status",
            )
            if payload.get(key) is not None
        }
        | {"step_count": len(steps), "steps": projected_steps}
    )


def _workspace_status(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    study = _bound_context(context.session.binding)
    registry = sources.load_registry()
    active_path = registry.get("active_path")
    active_job = None
    if study and study.get("active_job_id"):
        job = jobs.MANAGER.get(str(study["active_job_id"]))
        active_job = project_job(job.snapshot() if job else None)
    rows = _run_rows(context)
    details = {
        "study": project_study_context(study),
        "active_export": {
            "present": bool(active_path),
            "path_digest": path_digest(active_path),
            "source_count": len(registry.get("sources") or []),
        },
        "active_job": active_job or {"present": False},
        "latest_run": project_run_row(rows[0]) if rows else {"present": False},
        "scientific_authority": "EasyICU",
        "pi_session_role": "ux_state_only",
    }
    return _result(
        context,
        status="ok",
        code="easyicu_workspace_status_ready",
        summary="Loaded the current EasyICU study, job, and run status without patient rows or filesystem paths.",
        owner="easyicu.webserver.study_contexts",
        details=details,
    )


def _registered_source_projection(
    source: Mapping[str, Any], *, active_path: Any, database_label: str
) -> Dict[str, Any]:
    """Return one path-free registered-export choice for conversational setup."""

    summary = source.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    label = str(source.get("label") or "EasyICU export").strip()[:240]
    if "/" in label or "\\" in label:
        label = f"{str(source.get('database') or 'EasyICU').upper()} export"
    modules = [
        str(item)[:120] for item in (source.get("modules") or []) if str(item).strip()
    ][:64]
    official_demo_specs = (
        demo_sources.get_source(source_id)
        for source_id in demo_sources.allowed_source_ids()
    )
    official_demo_labels = {
        f"{spec.title} v{spec.version}".casefold() for spec in official_demo_specs
    }
    source_scope = (
        "official_demo"
        if label.casefold() in official_demo_labels
        else "registered_export"
    )
    return {
        "source_id": str(source.get("id") or "")[:80],
        # Registry labels are execution identity and may contain internal run
        # names such as "full6".  Conversational setup gets the canonical
        # database/release label instead; exact source_id remains authoritative.
        "label": label if source_scope == "official_demo" else database_label,
        "database": str(source.get("database") or "")[:80],
        "source_scope": source_scope,
        "availability": (
            "official_demo"
            if source_scope == "official_demo"
            else "available_in_easyicu"
        ),
        "generated": str(source.get("generated") or "")[:80] or None,
        "active": bool(
            str(source.get("path") or "").strip()
            and str(source.get("path") or "").strip() == str(active_path or "").strip()
        ),
        "module_count": len(modules),
        "modules": modules,
        "aggregate": {
            key: summary.get(key)
            for key in ("stays", "modules", "file_count", "total_rows")
            if isinstance(summary.get(key), (int, float))
        },
    }


def _dominant_local_source(
    choices: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return one unambiguously most-complete local dataset, if present."""

    local_choices = [
        row for row in choices if row.get("source_scope") == "registered_export"
    ]
    dominant = []
    for candidate in local_choices:
        candidate_stays = int((candidate.get("aggregate") or {}).get("stays") or 0)
        candidate_modules = int(candidate.get("module_count") or 0)
        if all(
            candidate_stays >= int((other.get("aggregate") or {}).get("stays") or 0)
            and candidate_modules >= int(other.get("module_count") or 0)
            for other in local_choices
        ):
            dominant.append(candidate)
    return dict(dominant[0]) if len(dominant) == 1 else None


def _database_named_in_message(message: str) -> Optional[str]:
    for database, pattern in (
        ("miiv", r"\bmimic[\s_-]*(?:iv|4)\b"),
        ("mimic", r"\bmimic[\s_-]*(?:iii|3)\b"),
        ("eicu", r"\beicu\b"),
        ("aumc", r"\b(?:amsterdamumcdb|aumc)\b"),
        ("hirid", r"\bhirid\b"),
        ("sic", r"\b(?:sicdb|sic)\b"),
    ):
        if re.search(pattern, message):
            return database
    return None


def _list_data_sources(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """List source families first, then exact exports for one selected family."""

    _require_args(params, allowed=("database",))
    requested_database = str(params.get("database") or "").strip()
    user_message = str(context.user_message or "").casefold()
    explicit_database = _database_named_in_message(user_message)
    if explicit_database:
        # The current user's exact database wording is stronger authority than
        # a model-generated tool argument. Recover the request deterministically
        # instead of returning a receipt for a different database family.
        requested_database = explicit_database
    ambiguous_mimic_request = bool(
        requested_database in {"miiv", "mimic"}
        and "mimic" in user_message
        and explicit_database not in {"miiv", "mimic"}
    )
    if ambiguous_mimic_request:
        # The model argument is not user authority. A bare "MIMIC" request
        # cannot inherit an active source or silently become MIMIC-IV.
        requested_database = ""
    selected_profile = None
    if requested_database:
        try:
            selected_profile = get_database_profile(requested_database)
        except KeyError as exc:
            raise PiCopilotError(
                "pi_database_not_supported",
                "Choose one exact database returned by easyicu_list_data_sources.",
            ) from exc
        if not selected_profile.is_public:
            raise PiCopilotError(
                "pi_database_not_supported",
                "Choose a full-database family before selecting a source mode.",
            )
    registry = sources.load_registry()
    choices = []
    if selected_profile is not None:
        database_label = selected_profile.display_name
        if selected_profile.reference_release:
            database_label = (
                f"{database_label} v{selected_profile.reference_release}"
            )
        for row in (registry.get("sources") or []):
            if not isinstance(row, Mapping) or not row.get("ok") or not row.get("id"):
                continue
            try:
                row_profile = get_database_profile(str(row.get("database") or ""))
            except KeyError:
                continue
            row_family = row_profile.parent_key or row_profile.key
            if row_family != selected_profile.key:
                continue
            choices.append(
                _registered_source_projection(
                    row,
                    active_path=registry.get("active_path"),
                    database_label=database_label,
                )
            )
            if len(choices) >= 40:
                break
    recommended_source = _dominant_local_source(choices)
    exact_database_named = bool(
        selected_profile is not None
        and explicit_database == selected_profile.key
    )
    auto_select_recommended = bool(recommended_source and exact_database_named)
    if recommended_source is not None:
        recommended_source = {
            key: recommended_source.get(key)
            for key in (
                "source_id",
                "label",
                "database",
                "availability",
                "module_count",
                "aggregate",
            )
        }
        recommended_source.update(
            {
                "selection_reason": "most_complete_local_dataset",
                "auto_select_for_exact_database_request": auto_select_recommended,
            }
        )
    supported_databases = [
        {
            "database": profile.key,
            "label": profile.display_name,
            "reference_release": profile.reference_release,
            "display_label": (
                f"{profile.display_name} v{profile.reference_release}"
                if profile.reference_release
                else profile.display_name
            ),
            "selection_required": True,
        }
        for profile in iter_database_profiles(public_only=True)
    ]
    official_demos = []
    for source_id in demo_sources.allowed_source_ids():
        source = demo_sources.get_source(source_id)
        profile = get_database_profile(source.database)
        demo_projection = (
            {
                "source_id": source.id,
                "label": source.title,
                "database": profile.parent_key or profile.key,
                "version": source.version,
                "scope_summary": source.scope_summary,
                "research_scope": "demo_only",
            }
        )
        if selected_profile is None or demo_projection["database"] == selected_profile.key:
            official_demos.append(demo_projection)
    return _result(
        context,
        status="ok",
        code="easyicu_data_sources_listed",
        summary=(
            f"Listed {len(supported_databases)} supported ICU database families "
            "and their reference releases."
            if selected_profile is None
            else (
                f"EasyICU can use {database_label} from the local data catalog "
                "without exposing filesystem paths or patient rows."
            )
        ),
        owner="easyicu.webserver.sources",
        details={
            "sources": choices,
            "source_count": len(choices),
            "selected_database": (
                {
                    "database": selected_profile.key,
                    "label": selected_profile.display_name,
                    "reference_release": selected_profile.reference_release,
                }
                if selected_profile is not None
                else None
            ),
            "supported_databases": supported_databases,
            "official_demos": official_demos,
            "recommended_source": recommended_source,
            "source_modes": (
                []
                if selected_profile is None
                else (
                    ["easyicu_available_data"]
                    if auto_select_recommended
                    else [
                        "local_full_database",
                        "registered_export",
                        "official_demo",
                    ]
                )
            ),
            "database_selection_deferred": ambiguous_mimic_request,
            "selection_policy": {
                "database_family_selected_before_source_mode": True,
                "mimic_database_choices": ["mimic", "miiv"],
                "registered_sources_deferred_until_database_selected": True,
                "exact_source_id_required_for_data_queries": True,
                "local_folder_paths_stay_host_side": True,
                "demo_never_implies_full_database": True,
                "official_demo_only_when_explicitly_requested_if_local_available": True,
                "recommended_source_is_already_available_export": True,
                "alternative_local_directory_requires_registration": True,
                "show_returned_safe_aggregates_for_recommendation": True,
                "user_facing_availability_term": "EasyICU can use",
            },
        },
    )


def _list_source_concepts(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """List path-free concept identifiers from one validated registered export."""

    _require_args(params, allowed=("source_id", "modules", "query", "limit"))
    source_id = str(params.get("source_id") or "").strip()
    if not source_id:
        raise PiCopilotError(
            "pi_source_id_required",
            "Choose one exact registered source_id before listing its concepts.",
        )
    registry = sources.load_registry()
    source = next(
        (
            row
            for row in (registry.get("sources") or [])
            if isinstance(row, Mapping)
            and row.get("ok")
            and str(row.get("id") or "") == source_id
        ),
        None,
    )
    if source is None:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_not_registered",
            summary="The selected source id is not a validated registered EasyICU export.",
            owner="easyicu.webserver.sources",
        )
    source_path = str(source.get("path") or "").strip()
    if not source_path:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_unavailable",
            summary="The selected registered source is not currently readable.",
            owner="easyicu.webserver.sources",
        )
    requested_modules = {
        str(value).strip().lower()
        for value in (params.get("modules") or [])
        if str(value).strip()
    }
    known_modules = {
        str(value).strip().lower()
        for value in (source.get("modules") or [])
        if str(value).strip()
    }
    unknown_modules = sorted(requested_modules - known_modules)
    if unknown_modules:
        return _result(
            context,
            status="blocked",
            code="pi_source_concept_modules_invalid",
            summary="One or more requested modules are not present in this registered source.",
            owner="easyicu.research_agent.acquisition.catalog",
            details={"unknown_modules": unknown_modules},
        )
    query_terms = tuple(
        dict.fromkeys(
            re.findall(r"[a-z0-9]+", str(params.get("query") or "").lower())
        )
    )
    raw_limit = params.get("limit", 40)
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        raise PiCopilotError(
            "pi_source_concept_limit_invalid",
            "The source concept limit must be an integer.",
        )
    limit = min(max(raw_limit, 1), 80)

    from easyicu.research_agent.acquisition.catalog import build_available_catalog

    catalog = build_available_catalog(Path(source_path).expanduser())
    def project_concept(concept: Any) -> Dict[str, str]:
        return {
            "concept_id": str(concept.concept_id)[:80],
            "module": Path(concept.file_name).stem.lower()[:80],
            "role": str(concept.column_role or "value")[:40],
            "description": " ".join(str(concept.description or "").split())[:300],
            "selection_mode": str(concept.selection_mode or "ordinary")[:40],
            "selection_note": " ".join(
                str(concept.selection_note or "").split()
            )[:500],
            "canonical_alternative": str(
                concept.canonical_alternative or ""
            )[:80],
        }

    rows = []
    for concept in catalog.concepts:
        module = Path(concept.file_name).stem.lower()
        if requested_modules and module not in requested_modules:
            continue
        searchable = " ".join(
            (
                str(concept.concept_id),
                str(concept.description or ""),
                module,
                str(concept.column_role or ""),
            )
        ).lower()
        searchable = " ".join(re.findall(r"[a-z0-9]+", searchable))
        if query_terms and not any(term in searchable for term in query_terms):
            continue
        rows.append(project_concept(concept))
    rows = rows[:limit]
    catalog_by_id = {
        str(concept.concept_id): concept for concept in catalog.concepts
    }
    canonical_alternatives = []
    for row in rows:
        alternative_id = str(row.get("canonical_alternative") or "")
        alternative = catalog_by_id.get(alternative_id)
        if alternative is None:
            continue
        canonical_alternatives.append(
            {
                "source_concept_id": row["concept_id"],
                **project_concept(alternative),
            }
        )
    return _result(
        context,
        status="ok",
        code="easyicu_source_concepts_listed",
        summary=(
            f"Listed {len(rows)} exact EasyICU concept identifiers from the "
            "selected registered source without paths or patient rows."
        ),
        owner="easyicu.research_agent.acquisition.catalog",
        details={
            "source_id": source_id,
            "concepts": rows,
            "canonical_alternatives": canonical_alternatives,
            "returned_count": len(rows),
            "truncated": len(rows) == limit,
        },
    )


def _inspect_data_package(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Review the bound registered export before a scientific Plan is made."""

    _require_args(params, allowed=())
    study = _bound_context(context.session.binding)
    if not study or not study.get("id"):
        return _result(
            context,
            status="blocked",
            code="data_package_study_required",
            summary="Complete and bind the typed study setup before reviewing its data package.",
            owner="easyicu.webserver.study_contexts",
        )
    from easyicu.webserver.data_package_review import (
        DataPackageReviewError,
        DataPackageReviewSnapshotStore,
        build_registered_data_package_review,
    )

    try:
        review = build_registered_data_package_review(study)
    except DataPackageReviewError as exc:
        return _result(
            context,
            status="blocked",
            code=exc.code,
            summary=exc.message,
            owner="easyicu.webserver.data_package_review",
            details=exc.details,
        )
    try:
        DataPackageReviewSnapshotStore().persist(review)
    except DataPackageReviewError as exc:
        return _result(
            context,
            status="blocked",
            code=exc.code,
            summary=(
                "The aggregate data-package review was produced, but its "
                "immutable replay snapshot could not be sealed. Do not emit a "
                "conversation resource that will drift with later study edits."
            ),
            owner="easyicu.webserver.data_package_review",
            details=exc.details,
        )
    ready = review.get("status") == "ready_for_plan"
    return _result(
        context,
        status="ok" if ready else "blocked",
        code=str(review.get("code") or "easyicu_data_package_review_blocked"),
        summary=(
            "Reviewed the registered EasyICU data package: the aggregate "
            "denominator and configured execution concepts are ready for a "
            "human-reviewed scientific Plan. Analysis results remain withheld."
            if ready
            else (
                "Reviewed the registered EasyICU data package and found one "
                "or more concept-semantic blockers; do not generate or run the "
                "scientific Plan yet."
            )
        ),
        owner="easyicu.webserver.data_package_review",
        details={
            "review": review,
            "resource": {
                "kind": "data_package_review",
                "study_context_id": str(study.get("id") or "")[:160],
                "study_revision": int(study.get("revision") or 0),
                "review_sha256": str(review.get("review_sha256") or "")[:64],
                "label": "Data package review",
                "media_type": "application/json",
            },
        },
    )


def _project_id_for_workbench(context: ToolExecutionContext) -> str:
    project_id = str(context.session.project_id or "").strip()
    if not project_id:
        raise PiCopilotError(
            "pi_project_required",
            "A bound Copilot project is required for a Data Workbench view.",
        )
    return project_id


def _registered_source_choice(
    context: ToolExecutionContext, requested_source_id: Any = None
) -> Dict[str, Any]:
    """Resolve an exact registered export while keeping its path host-side."""

    requested = str(requested_source_id or "").strip()
    if not requested:
        raise PiCopilotError(
            "pi_data_source_selection_required",
            "Choose one exact registered source_id before running a data query. "
            "A bound or globally active source is not implicit user consent.",
        )
    registry = sources.load_registry()
    rows = [
        row
        for row in (registry.get("sources") or [])
        if isinstance(row, Mapping) and row.get("ok") and row.get("id")
    ]
    source = next(
        (row for row in rows if str(row.get("id") or "") == requested), None
    )
    if source is None:
        return {}
    if not str(source.get("path") or "").strip():
        return {}
    return dict(source)


def _data_workbench_resource(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "data_workbench_snapshot",
        "view": str(snapshot.get("view") or "")[:80],
        "snapshot_sha256": str(snapshot.get("snapshot_sha256") or "")[:64],
        "label": str(snapshot.get("title") or "Data Workbench")[:160],
        "media_type": "application/json",
    }


def _persist_workbench_snapshot(
    context: ToolExecutionContext,
    *,
    view: str,
    title: str,
    payload: Mapping[str, Any],
    privacy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        snapshot = build_data_workbench_snapshot(
            project_id=_project_id_for_workbench(context),
            view=view,
            title=title,
            payload=payload,
            privacy=privacy,
        )
        CopilotDataWorkbenchSnapshotStore().persist(snapshot)
    except CopilotDataWorkbenchError as exc:
        raise PiCopilotError(
            exc.code,
            exc.message,
            status_code=409,
            details=exc.details,
        ) from exc
    return _data_workbench_resource(snapshot)


def _requested_workbench_features(
    payload: Mapping[str, Any], requested: Any
) -> list[str]:
    if requested in (None, []):
        return []
    if not isinstance(requested, list):
        raise PiCopilotError(
            "pi_workbench_features_invalid",
            "Data Workbench features must be a bounded list of exact concept or module:column identifiers.",
        )
    raw = [str(value or "").strip() for value in requested if str(value or "").strip()]
    if len(raw) > 8:
        raise PiCopilotError(
            "pi_workbench_features_too_many",
            "Select at most eight features for one conversational view.",
        )
    catalog = payload.get("feature_catalog")
    catalog = catalog if isinstance(catalog, Mapping) else {}
    features = [
        feature
        for module in (catalog.get("modules") or [])
        if isinstance(module, Mapping)
        for feature in (module.get("features") or [])
        if isinstance(feature, Mapping) and feature.get("id")
    ]
    by_id = {str(row.get("id") or "").lower(): str(row.get("id")) for row in features}
    by_column: Dict[str, list[str]] = {}
    for row in features:
        column = str(row.get("column") or "").lower()
        by_column.setdefault(column, []).append(str(row.get("id")))
    resolved: list[str] = []
    unknown: list[str] = []
    ambiguous: Dict[str, list[str]] = {}
    for value in raw:
        lower = value.lower()
        match = by_id.get(lower)
        if match:
            if match not in resolved:
                resolved.append(match)
            continue
        candidates = by_column.get(lower) or []
        if len(candidates) == 1:
            if candidates[0] not in resolved:
                resolved.append(candidates[0])
        elif len(candidates) > 1:
            ambiguous[value] = candidates[:8]
        else:
            unknown.append(value)
    if unknown or ambiguous:
        raise PiCopilotError(
            "pi_workbench_feature_not_resolved",
            "One or more requested features do not resolve uniquely in this registered export.",
            details={"unknown": unknown, "ambiguous": ambiguous},
        )
    return resolved


def _review_cohort(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Open cohort attrition and selected-feature distributions in Copilot."""

    _require_args(params, allowed=("source_id", "features"))
    source = _registered_source_choice(context, params.get("source_id"))
    if not source:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_not_registered",
            summary="Choose a validated registered EasyICU export before reviewing its cohort.",
            owner="easyicu.webserver.sources",
        )
    source_path = str(source.get("path") or "")
    try:
        base = cohort_review.cohort_review_summary({"source_path": source_path})
        feature_ids = _requested_workbench_features(base, params.get("features"))
        payload = (
            cohort_review.cohort_review_summary(
                {"source_path": source_path, "selected_features": feature_ids}
            )
            if feature_ids
            else base
        )
        from easyicu.webserver.patient_drilldown.eligibility import (
            _eligibility_flow_payload,
        )

        description = dataio.describe_export_source(source_path)
        eligibility = _eligibility_flow_payload(
            Path(source_path), description, dict(payload.get("summary") or {})
        )
    except cohort_review.CohortReviewError as exc:
        return _result(
            context,
            status="blocked",
            code=str((exc.detail or {}).get("error") or "cohort_review_blocked"),
            summary="The Cohort Review owner could not produce this aggregate view.",
            owner="easyicu.webserver.cohort_review",
            details={
                "reason_code": str((exc.detail or {}).get("error") or "")[:160]
            },
        )
    view = "feature_distribution" if feature_ids else "cohort_summary"
    title = "Feature distribution" if feature_ids else "Cohort and filter flow"
    resource = _persist_workbench_snapshot(
        context,
        view=view,
        title=title,
        payload={
            "source": payload.get("source"),
            "summary": payload.get("summary"),
            "eligibility_flow": eligibility,
            "groups": payload.get("groups"),
            "feature_catalog": payload.get("feature_catalog"),
            "feature_selection": payload.get("feature_selection"),
            "selected_feature_distributions": payload.get(
                "selected_feature_distributions"
            ),
            "coverage": payload.get("coverage"),
            "quality": payload.get("quality"),
            "survival_analysis": payload.get("survival_analysis"),
            "blocked_features": payload.get("blocked_features"),
            "provenance": payload.get("provenance"),
        },
        privacy=payload.get("privacy") if isinstance(payload.get("privacy"), Mapping) else {},
    )
    summary = payload.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    cohort_size = summary.get("cohort_size")
    return _result(
        context,
        status="ok",
        code=(
            "easyicu_feature_distribution_ready"
            if feature_ids
            else "easyicu_cohort_review_ready"
        ),
        summary=(
            f"Prepared a path-free conversational Data Workbench view for {cohort_size or 0} ICU stays"
            + (f" and {len(feature_ids)} selected features." if feature_ids else ".")
        ),
        owner="easyicu.webserver.cohort_review",
        details={
            "source_id": str(source.get("id") or "")[:80],
            "cohort_size": cohort_size,
            "selected_features": feature_ids,
            "resource": resource,
        },
    )


def _open_data_download(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Open a browser-controlled download for one registered export."""

    _require_args(params, allowed=("source_id",))
    source = _registered_source_choice(context, params.get("source_id"))
    if not source:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_not_registered",
            summary="Choose a validated registered EasyICU export before downloading it.",
            owner="easyicu.webserver.sources",
        )
    study = _bound_context(context.session.binding) or {}
    return _result(
        context,
        status="ok",
        code="easyicu_registered_export_download_ready",
        summary=(
            "Prepared a user-controlled browser download for the exact registered "
            "export. The model received only the source coordinate and aggregate "
            "package metadata; bytes remain behind the browser click."
        ),
        owner="easyicu.webserver.export_download",
        details={
            "source_id": str(source.get("id") or "")[:80],
            "file_count": int((source.get("summary") or {}).get("file_count") or 0),
            "total_rows": int((source.get("summary") or {}).get("total_rows") or 0),
            "resource": extraction_workspace_resource(
                study,
                state="review",
                source_id=source.get("id"),
            ),
        },
    )


def _bounded_icd_codes(value: Any, *, field: str, required: bool) -> list[str]:
    if value in (None, []):
        if required:
            raise PiCopilotError(
                "pi_icd_include_codes_required",
                "At least one ICD include code is required for a cohort preview.",
                details={"field": field},
            )
        return []
    if not isinstance(value, list):
        raise PiCopilotError(
            "pi_icd_codes_invalid",
            "ICD codes must be supplied as a bounded list.",
            details={"field": field},
        )
    if len(value) > 16:
        raise PiCopilotError(
            "pi_icd_codes_too_many",
            "Use at most sixteen ICD code prefixes in one cohort preview.",
            details={"field": field},
        )
    codes: list[str] = []
    for raw in value:
        code = re.sub(r"\s+", "", str(raw or "")).strip().upper()
        if not code or len(code) > 32:
            raise PiCopilotError(
                "pi_icd_code_invalid",
                "Each ICD code prefix must contain 1 to 32 non-space characters.",
                details={"field": field},
            )
        if code not in codes:
            codes.append(code)
    if required and not codes:
        raise PiCopilotError(
            "pi_icd_include_codes_required",
            "At least one ICD include code is required for a cohort preview.",
            details={"field": field},
        )
    return codes


def _preview_icd_cohort(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Preview an ICD-filtered extraction cohort without exposing its ids."""

    _require_args(
        params,
        allowed=("source_id", "include_codes", "exclude_codes"),
        required=("include_codes",),
    )
    include_codes = _bounded_icd_codes(
        params.get("include_codes"), field="include_codes", required=True
    )
    exclude_codes = _bounded_icd_codes(
        params.get("exclude_codes"), field="exclude_codes", required=False
    )
    source = _registered_source_choice(context, params.get("source_id"))
    if not source:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_not_registered",
            summary="Choose a validated registered EasyICU source before previewing an ICD cohort.",
            owner="easyicu.webserver.sources",
        )
    study = _bound_context(context.session.binding) or {}
    study_source = study.get("data_source")
    study_source = study_source if isinstance(study_source, Mapping) else {}
    database = str(source.get("database") or study_source.get("database") or "").strip()
    if not database:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_database_missing",
            summary="The registered source has no database identity for ICD matching.",
            owner="easyicu.webserver.sources",
        )
    base_cohort = study.get("cohort")
    try:
        preview = dataio.preview_registered_export_icd_cohort(
            str(source.get("path") or ""),
            database,
            base_cohort if isinstance(base_cohort, Mapping) else {},
            include_codes=include_codes,
            exclude_codes=exclude_codes,
        )
    except dataio.ExportCohortError as exc:
        return _result(
            context,
            status="blocked",
            code=exc.error,
            summary="The Data Extraction owner could not resolve this ICD cohort honestly.",
            owner="easyicu.webserver.dataio",
            details={"reason_code": exc.error},
        )
    report = preview.get("cohort_report")
    report = report if isinstance(report, Mapping) else {}
    cohort_size = int(preview.get("cohort_size") or 0)
    source_total = report.get("source_total")
    before_icd = report.get("selected_before_icd")
    resource = _persist_workbench_snapshot(
        context,
        view="icd_cohort_preview",
        title="ICD cohort preview",
        payload={
            "source": {
                "id": str(source.get("id") or "")[:80],
                "label": str(source.get("label") or database)[:160],
                "database": str(preview.get("database") or database)[:40],
            },
            "summary": {
                "cohort_size": cohort_size,
                "source_total": source_total,
                "selected_before_icd": before_icd,
                "count_unit": "icu_stays",
            },
            "cohort_contract": preview.get("cohort_contract"),
            "cohort_report": report,
            "provenance": {
                "owner": "easyicu.webserver.dataio",
                "operation": "preview_export_cohort",
                "execution_started": False,
            },
        },
        privacy=(
            preview.get("privacy")
            if isinstance(preview.get("privacy"), Mapping)
            else {}
        ),
    )
    return _result(
        context,
        status="ok",
        code="easyicu_icd_cohort_preview_ready",
        summary=f"The requested ICD filter selects {cohort_size} ICU stays in the current source.",
        owner="easyicu.webserver.dataio",
        details={
            "source_id": str(source.get("id") or "")[:80],
            "database": str(preview.get("database") or database)[:40],
            "cohort_size": cohort_size,
            "resource": resource,
        },
    )


def _review_patient_timeline(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Open one bounded pseudonymous patient timeline without model-visible rows."""

    _require_args(params, allowed=("source_id", "entity_ordinal", "features"))
    source = _registered_source_choice(context, params.get("source_id"))
    if not source:
        return _result(
            context,
            status="blocked",
            code="pi_data_source_not_registered",
            summary="Choose a validated registered EasyICU export before opening a patient timeline.",
            owner="easyicu.webserver.sources",
        )
    ordinal = params.get("entity_ordinal", 1)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise PiCopilotError(
            "pi_entity_ordinal_invalid",
            "The patient timeline requires a positive pseudonymous entity ordinal.",
        )
    requested_features = params.get("features")
    if requested_features in (None, []):
        feature_ids: list[str] = []
    elif not isinstance(requested_features, list):
        raise PiCopilotError(
            "pi_patient_features_invalid",
            "Patient Review features must be a bounded list of exact concept identifiers.",
        )
    else:
        if len(requested_features) > 8:
            raise PiCopilotError(
                "pi_patient_features_too_many",
                "Select at most eight Patient Review features for one conversational view.",
            )
        feature_ids = []
        for raw in requested_features:
            feature = str(raw or "").strip().lower()
            if not feature or len(feature) > 80 or not re.fullmatch(
                r"[a-z][a-z0-9_]*", feature
            ):
                raise PiCopilotError(
                    "pi_patient_feature_invalid",
                    "Each Patient Review feature must be an exact bounded EasyICU concept identifier.",
                )
            if feature not in feature_ids:
                feature_ids.append(feature)
    page_size = 24
    page = ((ordinal - 1) // page_size) + 1
    source_path = str(source.get("path") or "")
    try:
        navigation = patient_drilldown.patient_review_entity_page(
            {
                "source_path": source_path,
                "entity_page": page,
                "entity_page_size": page_size,
            }
        )
        options = ((navigation.get("navigation") or {}).get("options") or [])
        selected = next(
            (row for row in options if int(row.get("ordinal") or 0) == ordinal), None
        )
        if not selected or not selected.get("ref"):
            raise patient_drilldown.PatientReviewError(
                {"error": "unknown_entity_ordinal"}
            )
        payload = patient_drilldown.patient_review_drilldown(
            {"source_path": source_path, "entity_ref": selected["ref"]}
        )
        loaded_feature_details = []
        for feature in feature_ids:
            loaded_feature_details.append(
                patient_drilldown.patient_review_feature(
                    {
                        "source_path": source_path,
                        "entity_ref": selected["ref"],
                        "entity_ordinal": ordinal,
                        "feature": feature,
                    }
                )
            )
    except patient_drilldown.PatientReviewError as exc:
        return _result(
            context,
            status="blocked",
            code=str((exc.detail or {}).get("error") or "patient_review_blocked"),
            summary="The Patient Review owner could not open that pseudonymous entity timeline.",
            owner="easyicu.webserver.patient_drilldown",
        )
    patient_snapshot_payload = project_patient_snapshot_payload(
        {
            "source": payload.get("source"),
            "summary": payload.get("summary"),
            "eligibility_flow": payload.get("eligibility_flow"),
            "selected": payload.get("selected"),
            "time_lanes": payload.get("time_lanes"),
            "feature_coverage": payload.get("feature_coverage"),
            "loaded_feature_details": loaded_feature_details,
            "patient_overview": payload.get("patient_overview"),
            "trajectory_review": payload.get("trajectory_review"),
            "quality_metrics": payload.get("quality_metrics"),
            "blocked_features": payload.get("blocked_features"),
            "provenance": payload.get("provenance"),
        }
    )
    resource = _persist_workbench_snapshot(
        context,
        view="patient_timeline",
        title=f"Patient timeline · Entity {ordinal}",
        payload=patient_snapshot_payload,
        privacy=payload.get("privacy") if isinstance(payload.get("privacy"), Mapping) else {},
    )
    lanes = payload.get("time_lanes")
    available_lane_count = sum(
        1
        for lane in (lanes if isinstance(lanes, list) else [])
        if isinstance(lane, Mapping) and (lane.get("signals") or [])
    )
    loaded_trajectory_count = sum(
        1
        for detail in loaded_feature_details
        if isinstance(detail, Mapping)
        and detail.get("status") == "numeric_trajectory"
    )
    return _result(
        context,
        status="ok",
        code="easyicu_patient_timeline_ready",
        summary=(
            f"Prepared a bounded browser-only timeline for pseudonymous Entity {ordinal} "
            f"with {available_lane_count} populated time lanes and "
            f"{loaded_trajectory_count} requested feature trajectories. Direct identifiers, raw rows, "
            "timestamps, and host locations stayed inside the browser-only boundary."
        ),
        owner="easyicu.webserver.patient_drilldown",
        details={
            "source_id": str(source.get("id") or "")[:80],
            "entity_ordinal": ordinal,
            "available_time_lane_count": available_lane_count,
            "requested_feature_count": len(feature_ids),
            "loaded_trajectory_count": loaded_trajectory_count,
            "resource": resource,
        },
    )


def _compare_data_sources(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Open an aggregate Cross-DB comparison using exact registered source ids."""

    _require_args(params, allowed=("source_ids", "features"), required=("source_ids",))
    raw_ids = params.get("source_ids")
    if not isinstance(raw_ids, list):
        raise PiCopilotError(
            "pi_crossdb_sources_invalid",
            "Cross-DB comparison requires a list of registered source ids.",
        )
    source_ids = list(dict.fromkeys(str(value or "").strip() for value in raw_ids))
    source_ids = [value for value in source_ids if value]
    if len(source_ids) < 2 or len(source_ids) > 6:
        raise PiCopilotError(
            "pi_crossdb_source_count_invalid",
            "Select between two and six registered exports for one Cross-DB view.",
        )
    registry = sources.load_registry()
    registered = {
        str(row.get("id") or ""): row
        for row in (registry.get("sources") or [])
        if isinstance(row, Mapping) and row.get("ok") and row.get("id")
    }
    missing = [source_id for source_id in source_ids if source_id not in registered]
    if missing:
        return _result(
            context,
            status="blocked",
            code="pi_crossdb_source_not_registered",
            summary="One or more Cross-DB source ids are not validated registered exports.",
            owner="easyicu.webserver.sources",
            details={"missing_source_ids": missing},
        )
    paths = [str(registered[source_id].get("path") or "") for source_id in source_ids]
    try:
        payload = crossdb_review.crossdb_review_summary({"paths": paths})
    except crossdb_review.CrossdbReviewError as exc:
        detail = exc.detail or {}
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "crossdb_review_blocked"),
            summary="The Cross-DB owner blocked this comparison because its registered exports are not safely comparable.",
            owner="easyicu.webserver.crossdb_review",
            details={
                "source_count": len(source_ids),
                "compatibility": str(
                    (detail.get("compatibility_gate") or {}).get("status") or "blocked"
                )[:80],
            },
        )
    requested_features = params.get("features")
    if requested_features not in (None, []) and not isinstance(requested_features, list):
        raise PiCopilotError(
            "pi_crossdb_features_invalid",
            "Cross-DB features must be a bounded list of exact feature identifiers.",
        )
    if isinstance(requested_features, list) and len(requested_features) > 8:
        raise PiCopilotError(
            "pi_crossdb_feature_limit",
            "Cross-DB comparison accepts at most eight requested features.",
        )
    feature_names = {
        str(value or "").strip().lower()
        for value in (requested_features or [])[:8]
        if str(value or "").strip()
    }
    distributions = []
    for module in payload.get("feature_distributions") or []:
        if not isinstance(module, Mapping):
            continue
        module_name = str(module.get("module") or "").strip().lower()
        rows = [
            row
            for row in (module.get("features") or [])
            if isinstance(row, Mapping)
            and (
                not feature_names
                or str(row.get("feature") or "").strip().lower() in feature_names
                or (
                    module_name
                    and f"{module_name}:{str(row.get('feature') or '').strip().lower()}"
                    in feature_names
                )
            )
        ]
        if not feature_names:
            rows = rows[:4]
        if rows:
            distributions.append({**dict(module), "features": rows})
        if sum(len(row.get("features") or []) for row in distributions) >= 16:
            break
    resource = _persist_workbench_snapshot(
        context,
        view="crossdb_comparison",
        title="Cross-database comparison",
        payload={
            "source_count": payload.get("source_count"),
            "sources": payload.get("sources"),
            "selection_receipt": payload.get("selection_receipt"),
            "rows": payload.get("rows"),
            "availability": payload.get("availability"),
            "feature_density": payload.get("feature_density"),
            "feature_distributions": distributions,
            "shared_modules": payload.get("shared_modules"),
            "all_modules": payload.get("all_modules"),
            "compatibility_gate": payload.get("compatibility_gate"),
            "blocked_features": payload.get("blocked_features"),
            "provenance": payload.get("provenance"),
        },
        privacy=payload.get("privacy") if isinstance(payload.get("privacy"), Mapping) else {},
    )
    gate = payload.get("compatibility_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    shared = payload.get("shared_modules") or []
    return _result(
        context,
        status="ok",
        code="easyicu_crossdb_comparison_ready",
        summary=(
            f"Prepared a descriptive Cross-DB comparison for {len(source_ids)} registered exports "
            f"with {len(shared)} shared modules. Inferential and matched-cohort claims remain blocked."
        ),
        owner="easyicu.webserver.crossdb_review",
        details={
            "source_ids": source_ids,
            "source_count": len(source_ids),
            "shared_module_count": len(shared),
            "compatibility": str(gate.get("status") or "")[:80],
            "resource": resource,
        },
    )


def _prepare_demo_source(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Start the existing official download -> convert -> export -> register owner."""

    _require_args(params, allowed=("source_id",), required=("source_id",))
    source_id = str(params.get("source_id") or "").strip()
    try:
        source = demo_sources.get_source(source_id)
    except KeyError:
        return _result(
            context,
            status="blocked",
            code="pi_demo_source_unknown",
            summary="Choose one allowlisted official EasyICU demo source id.",
            owner="easyicu.webserver.demo_sources",
            details={"allowed_source_ids": list(demo_sources.allowed_source_ids())},
        )
    grant_block = _consume_action(context, "extract")
    if grant_block is not None:
        return grant_block
    try:
        job = jobs.MANAGER.submit(
            "demo-source-prepare", demo_sources.make_prepare_runner(source.id)
        )
    except jobs.JobCapacityError as exc:
        return _result(
            context,
            status="blocked",
            code="job_capacity_exceeded",
            summary="Wait for a running local data job to finish before preparing the demo source.",
            owner="easyicu.webserver.jobs",
            details={"running": exc.running, "max_running": exc.max_running},
        )
    return _result(
        context,
        status="ok",
        code="easyicu_demo_source_preparation_submitted",
        summary=(
            "Submitted the existing official demo-source pipeline: download, validate, "
            "convert, all-module export, and registration will run locally."
        ),
        owner="easyicu.webserver.demo_sources",
        details={
            "job_id": job.id,
            "kind": job.kind,
            "status": job.status,
            "source_id": source.id,
        },
    )


def _workflow_snapshot(
    context: ToolExecutionContext,
    *,
    study_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    study = (
        dict(study_override)
        if study_override is not None
        else _bound_context(context.session.binding)
    )
    registry = sources.load_registry()
    active_job = None
    if study and study.get("active_job_id"):
        job = jobs.MANAGER.get(str(study["active_job_id"]))
        active_job = project_job(job.snapshot() if job else None)
    rows = _run_rows(context)
    latest_run = project_run_row(rows[0]) if rows else None
    plan_review_authority = (
        agent_pipeline_runs.pending_review(latest_run.get("run_id"))
        if latest_run
        else None
    )
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=registered_export_matches_study(study, registry),
        active_job=active_job,
        latest_run=latest_run,
        plan_review_authority=plan_review_authority,
    )
    return snapshot.model_dump(mode="json")


def _inspect_workflow(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    workflow = _workflow_snapshot(context)
    return _result(
        context,
        status="ok",
        code="easyicu_research_workflow_projected",
        summary=(
            "Loaded the project-level EasyICU workflow from typed study, "
            "extraction, run, evidence, and manuscript receipts."
        ),
        owner="easyicu.webserver.pi_copilot.workflow",
        details={"workflow": workflow},
    )


def _inspect_context(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    study = _bound_context(context.session.binding)
    if not study:
        return _result(
            context,
            status="not_found",
            code="study_context_not_found",
            summary="No typed StudyContext is currently bound to this Copilot session.",
            owner="easyicu.webserver.study_contexts",
        )
    return _result(
        context,
        status="ok",
        code="study_context_projected",
        summary=f"Loaded StudyContext revision {int(study.get('revision') or 0)} through the PHI-safe projection.",
        owner="easyicu.webserver.study_contexts",
        details={"study": project_study_context(study)},
    )


def _inspect_capability(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    return _result(
        context,
        status="ok",
        code="capability_policy_projected",
        summary="Loaded the current EasyICU capability policy without credential values or private paths.",
        owner="easyicu.webserver.capabilities",
        details=project_capabilities(capabilities.capability_status()),
    )


def _inspect_run(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id", "job_id"))
    job_id = str(
        params.get("job_id") or context.session.binding.active_job_id or ""
    ).strip()
    if job_id:
        job = jobs.MANAGER.get(job_id)
        if job:
            return _result(
                context,
                status="ok",
                code="easyicu_job_status_projected",
                summary=f"EasyICU job {job_id} is {job.status}.",
                owner="easyicu.webserver.jobs",
                details={"job": project_job(job.snapshot())},
            )
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_run_not_found",
            summary="No matching persisted EasyICU run was found.",
            owner="easyicu.webserver.agent_runs",
        )
    projected = project_run_row(row)
    waiting_for_plan_approval = bool(projected.get("human_plan_review_pending"))
    summary = (
        f"EasyICU run {row.get('run_id')} is paused at the human plan-review gate; "
        "analysis has not executed and plan-stage result/manuscript filenames are "
        "non-reportable placeholders."
        if waiting_for_plan_approval
        else f"Loaded the bounded status for EasyICU run {row.get('run_id')}."
    )
    return _result(
        context,
        status="ok",
        code="easyicu_run_status_projected",
        summary=summary,
        owner="easyicu.webserver.agent_runs",
        details={"run": projected},
    )


def _inspect_plan(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_plan_not_found",
            summary="No persisted EasyICU run with a plan artefact was found.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    plan = (review.get("artifact_payloads") or {}).get("agent_plan.json")
    if not isinstance(plan, Mapping):
        return _result(
            context,
            status="not_found",
            code="easyicu_plan_artifact_missing",
            summary=f"Run {row.get('run_id')} does not have an inspectable plan artefact.",
            owner="easyicu.webserver.agent_runs",
        )
    projected = _plan_projection(plan)
    payloads = review.get("artifact_payloads") or {}
    resources = [
        resource
        for resource in (
            _artifact_resource(row.get("run_id"), "agent_plan.json"),
            (
                _artifact_resource(row.get("run_id"), "literature_evidence.json")
                if isinstance(payloads.get("literature_evidence.json"), Mapping)
                else None
            ),
        )
        if resource is not None
    ]
    return _result(
        context,
        status="ok",
        code="easyicu_plan_projected",
        summary=f"Loaded {projected.get('step_count', 0)} bounded plan steps from run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details={
            "plan": projected,
            "resource": _artifact_resource(row.get("run_id"), "agent_plan.json"),
            "resources": resources,
        },
    )


def _inspect_literature(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_literature_run_not_found",
            summary="No matching persisted EasyICU run was found.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    literature = (review.get("artifact_payloads") or {}).get("literature_evidence.json")
    if not isinstance(literature, Mapping):
        return _result(
            context,
            status="not_found",
            code="easyicu_literature_artifact_missing",
            summary=(
                f"Run {row.get('run_id')} has no projected literature evidence "
                "artifact. Do not infer citations from the plan text."
            ),
            owner="easyicu.webserver.agent_runs",
        )
    search = literature.get("search")
    search = search if isinstance(search, Mapping) else {}
    searched = bool(search.get("search_conducted"))
    count = int(literature.get("citation_count") or 0)
    mapping = str(literature.get("mapping_status") or "not_bound")
    source_kind = "retrieved search results" if searched else "curated references"
    direct_count = int(literature.get("direct_comparator_count") or 0)
    return _result(
        context,
        status="ok",
        code="easyicu_literature_evidence_projected",
        summary=(
            f"Loaded {count} {source_kind} for run {row.get('run_id')}; "
            f"{direct_count} survived as direct-comparator candidates and "
            f"plan-step citation mapping is {mapping}."
        ),
        owner="easyicu.webserver.agent_runs",
        details={
            "literature": {
                key: literature.get(key)
                for key in (
                    "scope",
                    "status",
                    "citation_count",
                    "plan_step_count",
                    "mapped_step_count",
                    "mapping_status",
                    "scientific_mapping_status",
                    "direct_comparator_count",
                    "direct_comparator_keys",
                    "search",
                )
            },
            "resource": _artifact_resource(
                row.get("run_id"), "literature_evidence.json"
            ),
        },
    )


def _inspect_step(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id", "step_id"), required=("step_id",))
    plan_result = _inspect_plan(
        context, {"run_id": params.get("run_id")} if params.get("run_id") else {}
    )
    if plan_result.get("status") != "ok":
        return plan_result
    plan = (plan_result.get("details") or {}).get("plan") or {}
    step_id = str(params["step_id"])
    step = next(
        (
            row
            for row in (plan.get("steps") or [])
            if str(row.get("step_id") or row.get("id") or "") == step_id
        ),
        None,
    )
    if not step:
        return _result(
            context,
            status="not_found",
            code="easyicu_plan_step_not_found",
            summary=f"No plan step named {step_id!r} exists in the selected bounded plan.",
            owner="easyicu.webserver.agent_runs",
        )
    return _result(
        context,
        status="ok",
        code="easyicu_plan_step_projected",
        summary=f"Loaded the bounded contract for plan step {step_id}.",
        owner="easyicu.webserver.agent_runs",
        details={"step": step},
    )


def _validated_operational_mappings(
    payloads: Mapping[str, Any],
) -> list[Dict[str, str]]:
    """Project owner-recorded semantic-to-physical bindings from result audits."""

    result_tables = payloads.get("result_tables.json")
    if not isinstance(result_tables, Mapping):
        return []
    tables = result_tables.get("tables")
    if not isinstance(tables, list):
        return []
    mappings: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for table in tables[:40]:
        if not isinstance(table, Mapping):
            continue
        headers = table.get("headers")
        rows = table.get("rows")
        if not isinstance(headers, list) or not isinstance(rows, list):
            continue
        header_names = [str(value or "").strip() for value in headers]
        if "concept" not in header_names or "value_column" not in header_names:
            continue
        concept_index = header_names.index("concept")
        value_column_index = header_names.index("value_column")
        semantics_index = (
            header_names.index("indicator_semantics")
            if "indicator_semantics" in header_names
            else None
        )
        for row in rows[:50]:
            if not isinstance(row, list):
                continue
            if max(concept_index, value_column_index) >= len(row):
                continue
            concept = _bounded_model_text(row[concept_index], 80)
            value_column = _bounded_model_text(row[value_column_index], 80)
            if not concept or not value_column:
                continue
            identity = (concept, value_column)
            if identity in seen:
                continue
            seen.add(identity)
            mapping = {
                "semantic_concept": concept,
                "operational_value_column": value_column,
                "authority": "validated_result_measurement_audit",
            }
            if semantics_index is not None and semantics_index < len(row):
                semantics = _bounded_model_text(row[semantics_index], 80)
                if semantics:
                    mapping["indicator_semantics"] = semantics
            mappings.append(mapping)
            if len(mappings) >= 20:
                return mappings
    return mappings


def _inspect_validation(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_validation_not_found",
            summary="No matching EasyICU run was found for validation inspection.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    gate = review.get("gate")
    gate = gate if isinstance(gate, Mapping) else {}
    readiness = review.get("readiness")
    readiness = readiness if isinstance(readiness, Mapping) else {}
    checks = gate.get("checks")
    checks = checks if isinstance(checks, list) else []
    failed_requirement_codes = [
        stable_code(check.get("id"))
        for check in checks
        if isinstance(check, Mapping)
        and isinstance(check.get("id"), str)
        and not check.get("passed")
    ][:50]
    failed_requirement_codes = [
        code for code in failed_requirement_codes if code is not None
    ]
    missing_requirement_codes = [
        stable_code(item)
        for item in (readiness.get("non_human_failures") or [])
        if isinstance(item, str)
    ][:50]
    missing_requirement_codes = [
        code for code in missing_requirement_codes if code is not None
    ]
    payloads = review.get("artifact_payloads") or {}
    scientific = payloads.get("scientific_readiness.json")
    scientific = scientific if isinstance(scientific, Mapping) else {}
    raw_findings = scientific.get("findings")
    raw_findings = raw_findings if isinstance(raw_findings, list) else []
    scientific_findings = []
    for item in raw_findings[:20]:
        if not isinstance(item, Mapping):
            continue
        scientific_findings.append(
            {
                "code": stable_code(item.get("code")),
                "domain": stable_code(item.get("domain")),
                "severity": stable_code(item.get("severity")),
                "message": _bounded_model_text(item.get("message"), 1_000),
                "remediation": _bounded_model_text(item.get("remediation"), 1_000),
                "requires_user_authorization": bool(
                    item.get("requires_user_authorization")
                ),
                "authorization_question": (
                    _bounded_model_text(item.get("authorization_question"), 1_000)
                    or None
                ),
            }
        )
    scientific_facts = scientific.get("facts")
    scientific_facts = scientific_facts if isinstance(scientific_facts, Mapping) else {}
    analysis_facts = scientific_facts.get("analysis")
    analysis_facts = analysis_facts if isinstance(analysis_facts, Mapping) else {}
    gate_checks = {
        str(check.get("id") or "").strip(): bool(check.get("passed"))
        for check in checks
        if isinstance(check, Mapping) and str(check.get("id") or "").strip()
    }
    result_tables = payloads.get("result_tables.json")
    result_tables = result_tables if isinstance(result_tables, Mapping) else {}
    details = bounded_json_projection(
        {
            "run_id": row.get("run_id"),
            "gate": {
                "status": gate.get("status"),
                "gate_code": stable_code(gate.get("reason")),
                "reportable": bool(gate.get("reportable")),
                "draft_unlocked": bool(gate.get("draft_unlocked")),
                "checks_total": len(checks),
                "checks_passed": sum(
                    1
                    for check in checks
                    if isinstance(check, Mapping) and check.get("passed")
                ),
                "failed_requirement_codes": failed_requirement_codes,
            },
            "readiness": {
                key: readiness.get(key)
                for key in (
                    "status",
                    "signable",
                    "signed",
                    "signoff_stale",
                    "reportable",
                    "draft_unlocked",
                    "gate_status",
                    "checks_total",
                    "checks_passed",
                    "human_signoff_passed_in_gate",
                )
                if readiness.get(key) is not None
            }
            | {"missing_requirement_codes": missing_requirement_codes},
            "scientific_readiness": {
                "status": stable_code(scientific.get("status")),
                "claim_ceiling": stable_code(scientific.get("claim_ceiling")),
                "publication_ready": bool(scientific.get("publication_ready")),
                "paper_authorized": bool(scientific.get("paper_authorized")),
                "maturity_score": analysis_facts.get("scientific_maturity_score"),
                "dimension_scores": analysis_facts.get(
                    "scientific_maturity_dimension_scores"
                ),
                "findings": scientific_findings,
                "user_authorization_requests": list(
                    analysis_facts.get("user_authorization_requests") or []
                )[:20],
                "resource": _artifact_resource(
                    row.get("run_id"), "scientific_readiness.json"
                ),
            },
            "analysis_execution": {
                "analysis_validated": bool(analysis_facts.get("analysis_validated")),
                "numeric_verified": bool(gate_checks.get("numeric_verified")),
                "result_table_count": int(result_tables.get("table_count") or 0),
                "operational_mappings": _validated_operational_mappings(payloads),
                "publication_gate_separate": True,
            },
            "signed": bool(review.get("signed")),
            "signoff_stale": bool(review.get("signoff_stale")),
            "resource": _artifact_resource(row.get("run_id"), "quality_gate.json"),
        }
    )
    return _result(
        context,
        status="ok",
        code="easyicu_validation_projected",
        summary=f"Loaded gate and readiness status for run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details=details,
    )


def _list_artifacts(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_artifacts_not_found",
            summary="No matching EasyICU run was found for artefact inspection.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    artifacts = project_artifacts(review.get("artifacts") or [])
    return _result(
        context,
        status="ok",
        code="easyicu_artifacts_projected",
        summary=f"Listed {len(artifacts)} whitelisted artefacts for run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details={
            "run_id": row.get("run_id"),
            "artifacts": artifacts,
            "resources": _artifact_resources(row.get("run_id"), artifacts),
        },
    )


def _inspect_evidence(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_evidence_not_found",
            summary="No matching EasyICU run was found for evidence inspection.",
            owner="easyicu.webserver.agent_runs",
        )
    review = _run_review(row)
    ledger = (review.get("artifact_payloads") or {}).get("evidence_ledger.json")
    if not isinstance(ledger, Mapping):
        return _result(
            context,
            status="not_found",
            code="easyicu_evidence_ledger_missing",
            summary=f"Run {row.get('run_id')} does not have an evidence ledger.",
            owner="easyicu.webserver.agent_runs",
        )
    provider = ledger.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    strict = ledger.get("strict_evidence_audit")
    strict = strict if isinstance(strict, Mapping) else {}
    numeric = ledger.get("numeric_evidence_audit")
    numeric = numeric if isinstance(numeric, Mapping) else {}
    privacy = ledger.get("privacy")
    privacy = privacy if isinstance(privacy, Mapping) else {}
    artifacts = ledger.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    details = {
        "run_id": ledger.get("run_id"),
        "run_type": ledger.get("run_type"),
        "status": ledger.get("status"),
        "artifacts": project_artifacts(
            row for row in artifacts if isinstance(row, Mapping)
        ),
        "resource": _artifact_resource(row.get("run_id"), "evidence_ledger.json"),
        "provider": {
            key: provider.get(key)
            for key in (
                "provider",
                "external",
                "provider_gate",
                "canonical_opt_in_passed",
                "per_run_opt_in",
                "external_calls",
                "mock_calls",
            )
            if provider.get(key) is not None
        },
        "strict_evidence_audit": {
            key: strict.get(key)
            for key in (
                "passed",
                "claims_passed",
                "sentences_passed",
                "claim_count",
                "sentence_count",
            )
            if strict.get(key) is not None
        },
        "numeric_evidence_audit": {
            key: numeric.get(key)
            for key in ("passed", "status", "checked_values", "issue_count")
            if numeric.get(key) is not None
        },
        "privacy": {
            key: privacy.get(key)
            for key in ("passed", "status", "uploads", "row_level_data")
            if privacy.get(key) is not None
        },
    }
    return _result(
        context,
        status="ok",
        code="easyicu_evidence_projected",
        summary=f"Loaded bounded evidence and audit status for run {row.get('run_id')}.",
        owner="easyicu.webserver.agent_runs",
        details=details,
    )


def _inspect_interpretation(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_interpretation_not_found",
            summary="No matching Research Agent run is available for result interpretation.",
            owner="easyicu.research_agent.reporting",
        )
    review = _run_review(row)
    payloads = review.get("artifact_payloads") or {}
    manuscript = payloads.get("manuscript_draft.json")
    card = build_result_interpretation_card(
        run_id=row.get("run_id"),
        review=review,
        manuscript=manuscript if isinstance(manuscript, Mapping) else None,
        result_tables=(
            payloads.get("result_tables.json")
            if isinstance(payloads.get("result_tables.json"), Mapping)
            else None
        ),
        scientific_readiness=(
            payloads.get("scientific_readiness.json")
            if isinstance(payloads.get("scientific_readiness.json"), Mapping)
            else None
        ),
    )
    artifacts = project_artifacts(review.get("artifacts") or [])
    resources = _artifact_resources(row.get("run_id"), artifacts)
    status = "blocked" if card.status == "blocked" else "ok"
    code = (
        "easyicu_result_interpretation_blocked"
        if card.status == "blocked"
        else "easyicu_result_interpretation_projected"
    )
    card_projection = card.model_dump(mode="json")
    safe_claims = []
    for claim in card_projection.get("claims") or []:
        try:
            ensure_safe_projection(claim)
        except PiCopilotError:
            continue
        safe_claims.append(claim)
    card_projection["claims"] = safe_claims
    safe_limitations = []
    for limitation in card_projection.get("limitations") or []:
        try:
            ensure_safe_projection(limitation)
        except PiCopilotError:
            continue
        safe_limitations.append(limitation)
    card_projection["limitations"] = safe_limitations
    return _result(
        context,
        status=status,
        code=code,
        summary=card.summary,
        owner="easyicu.research_agent.reporting.result_card",
        details={
            "interpretation": card_projection,
            "withheld_claim_count": len(card.claims) - len(safe_claims),
            "withheld_limitation_count": len(card.limitations)
            - len(safe_limitations),
            "resources": resources,
        },
    )


def _inspect_manuscript(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id",))
    row = _select_run(context, params.get("run_id"))
    if not row:
        return _result(
            context,
            status="not_found",
            code="easyicu_manuscript_not_found",
            summary="No matching Research Agent run is available for manuscript review.",
            owner="easyicu.research_agent.reporting",
        )
    review = _run_review(row)
    payloads = review.get("artifact_payloads") or {}
    manuscript = payloads.get("manuscript_draft.json")
    if not isinstance(manuscript, Mapping):
        return _result(
            context,
            status="not_found",
            code="easyicu_manuscript_artifact_missing",
            summary=(
                "This run has no Research Agent manuscript draft. A deterministic "
                "preflight does not fabricate one."
            ),
            owner="easyicu.research_agent.reporting",
        )
    governance = agent_runs.project_artifact_governance(
        review,
        artifact=next(
            (
                item
                for item in (review.get("artifacts") or [])
                if isinstance(item, Mapping)
                and item.get("name") == "manuscript_draft.json"
            ),
            None,
        ),
    )
    projected_artifacts = project_artifacts(review.get("artifacts") or [])
    interpretation = build_result_interpretation_card(
        run_id=row.get("run_id"),
        review=review,
        manuscript=manuscript,
        result_tables=(
            payloads.get("result_tables.json")
            if isinstance(payloads.get("result_tables.json"), Mapping)
            else None
        ),
        scientific_readiness=(
            payloads.get("scientific_readiness.json")
            if isinstance(payloads.get("scientific_readiness.json"), Mapping)
            else None
        ),
    )
    resources = [
        resource
        for resource in (
            _artifact_resource(row.get("run_id"), "manuscript_draft.json"),
            *(
                _document_resource(row.get("run_id"), artifact.get("name"))
                for artifact in projected_artifacts
            ),
        )
        if resource is not None
    ]
    safe_review_claims = []
    for claim in interpretation.claims:
        projected_claim = claim.model_dump(mode="json")
        try:
            ensure_safe_projection(projected_claim)
        except PiCopilotError:
            continue
        safe_review_claims.append(projected_claim)
    return _result(
        context,
        status="ok",
        code="easyicu_manuscript_projected",
        summary=(
            "Loaded the evidence-bound Research Agent manuscript draft for human "
            "review; Pi did not author or unlock it."
        ),
        owner="easyicu.research_agent.reporting",
        details={
            "run_id": row.get("run_id"),
            "manuscript": bounded_json_projection(
                {
                    "status": manuscript.get("status"),
                    "question": _bounded_model_text(manuscript.get("question"), 1200),
                    "source": stable_code(manuscript.get("source")),
                    "claim_count": len(manuscript.get("claims") or []),
                    "review_claims": safe_review_claims,
                    "withheld_review_claim_count": len(interpretation.claims)
                    - len(safe_review_claims),
                }
            ),
            "governance": bounded_json_projection(governance),
            "resources": resources,
        },
    )


def _explain_blocker(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_id", "job_id"))
    run_or_job = _inspect_run(context, params)
    details = run_or_job.get("details") or {}
    job = details.get("job") if isinstance(details.get("job"), Mapping) else {}
    if job and job.get("status") in {"failed", "cancelled"}:
        code = str(
            job.get("error_code") or job.get("cancel_reason_code") or job["status"]
        )
        return _result(
            context,
            status="blocked",
            code=code,
            summary=f"The active EasyICU job is {job.get('status')}; the JobManager owner reported {code}.",
            owner="easyicu.webserver.jobs",
            details={"job": job},
        )
    row = _select_run(context, params.get("run_id"))
    if row:
        review = _run_review(row)
        gate = review.get("gate") or {}
        readiness = review.get("readiness") or {}
        if gate.get("status") == "blocked" or readiness.get("status") not in {
            "ready",
            "signed",
        }:
            code = (
                stable_code(gate.get("reason"))
                or stable_code(readiness.get("reason"))
                or "easyicu_readiness_blocked"
            )
            return _result(
                context,
                status="blocked",
                code=code,
                summary=f"EasyICU run {row.get('run_id')} is held by its gate/readiness owner: {code}.",
                owner="easyicu.webserver.agent_runs",
                details={
                    "run_id": row.get("run_id"),
                    "gate_status": gate.get("status"),
                    "gate_code": stable_code(gate.get("reason")),
                    "readiness_status": readiness.get("status"),
                },
            )
    return _result(
        context,
        status="ok",
        code="easyicu_no_active_blocker",
        summary="No active job failure or persisted run gate blocker was found.",
        owner="easyicu.webserver.pi_copilot",
    )


_STUDY_SETUP_FIELDS = frozenset(
    {
        "title",
        "question",
        "purpose",
        "cohort",
        "modules",
        "outcome",
        "primary_exposure",
        "covariates",
        "covariate_selection",
        "covariate_rationales",
        "covariate_temporal_roles",
        "covariate_operationalizations",
        "execution_concepts",
        "analysis_design",
        "sensitivity_specs",
        "time_window",
        "comparator",
        "export_format",
        "analysis_goal",
        "confirmations",
        "bind_active_export",
        "bind_source_id",
    }
)

_NESTED_STUDY_PATCH_FIELDS = frozenset(
    {
        "cohort",
        "time_window",
        "confirmations",
        "execution_concepts",
        "analysis_design",
        "covariate_rationales",
        "covariate_temporal_roles",
        "covariate_operationalizations",
    }
)


def _message_explicitly_selects_first_stay(message: str) -> bool:
    """Return whether this user turn explicitly chooses one first ICU stay."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"首次\s*(?:icu|重症监护)",
            r"首个\s*(?:icu|重症监护)",
            r"第一(?:次|个)\s*(?:icu|重症监护)",
            r"\bfirst[\s_-]+(?:icu|intensive care)(?:[\s_-]+stay|[\s_-]+admission)?\b",
            r"\bindex[\s_-]+(?:icu|intensive care)(?:[\s_-]+stay|[\s_-]+admission)?\b",
            r"\bone[\s_-]+(?:icu[\s_-]+)?stay[\s_-]+per[\s_-]+patient\b",
            r"排除\s*(?:再次|重复|再入|重返).*?(?:icu|重症监护)",
            r"\bexclude[\s_-]+(?:icu[\s_-]+)?readmissions?\b",
        )
    )


def _message_explicitly_selects_all_stays(message: str) -> bool:
    """Return whether this user turn explicitly retains eligible ICU stays."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"所有符合条件.*?(?:icu|重症监护).*?(?:stay|入住|住院)",
            r"(?:包括|保留|纳入).*?(?:重复|再次|再入).*?(?:icu|重症监护)",
            r"不(?:限制|限于|只纳入).*?(?:首次|首个|第一).*?(?:icu|重症监护)",
            r"\ball[\s_-]+eligible[\s_-]+(?:adult[\s_-]+)?icu[\s_-]+stays?\b",
            r"\b(?:include|retain)[\s_-]+(?:repeated|repeat|readmission)[\s_-]+(?:icu[\s_-]+)?stays?\b",
            r"\bnot[\s_-]+(?:restricted[\s_-]+to[\s_-]+)?(?:the[\s_-]+)?first[\s_-]+icu[\s_-]+stay\b",
        )
    )


def _message_explicitly_selects_primary_outcome(message: str) -> bool:
    """Return whether this turn chooses, rather than merely mentions, an outcome."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"主要结局.*?(?:使用|采用|设为|定义为|改为|替换为)",
            r"(?:确认|同意).*?主要结局.*?(?:为|是|采用|使用)",
            r"(?:使用|采用|设为|改为|替换为).*?(?:死亡|mortality).*?(?:主要)?结局",
            r"结局.*?(?:使用|采用|设为|定义为|改为|替换为).*?(?:死亡|mortality)",
            r"不(?:改成|改为|替换为).*?(?:死亡|mortality)",
            # The Copilot renders this compact affirmative choice directly
            # beneath one concrete outcome proposal.  It is an explicit user
            # decision even though the button text avoids repeating the long
            # clinical definition.
            r"^(?:是|确认)[，,\s]*(?:采用|使用|确认)?(?:该|此|上述)?(?:定义|主要结局定义)[。.]?$",
            # A rendered next-step choice is itself an explicit selection even
            # when its button label is a compact noun phrase.  Requiring the
            # user to add a verb after clicking the option makes the same
            # scientific decision repeat indefinitely.
            r"^(?:icu\s*(?:stay\s*)?(?:住院期间|期间|内)?死亡|icu\s*mortality)(?:\s*[（(]推荐[）)])?$",
            r"^(?:hospital|in[\s_-]*hospital|28[\s_-]*day|30[\s_-]*day|90[\s_-]*day)[\s_-]*(?:death|mortality)$",
            r"\bprimary[\s_-]+outcome[\s_-]+(?:is|uses?|will[\s_-]+be)\b",
            r"\buse[\s_-]+.+?[\s_-]+(?:as[\s_-]+the[\s_-]+)?primary[\s_-]+outcome\b",
            r"\bdo[\s_-]+not[\s_-]+change[\s_-]+(?:it[\s_-]+)?to\b",
        )
    )


def _message_explicitly_selects_primary_exposure(message: str) -> bool:
    """Return whether this turn chooses a primary exposure definition."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"主要暴露.*?(?:使用|采用|设为|定义为)",
            r"(?:确认|同意).*?主要暴露.*?(?:为|是|采用|使用)",
            r"(?:使用|采用|设为).*?(?:sepsis|脓毒症).*?(?:定义|作为.*?暴露)",
            r"(?:sepsis|脓毒症).*?(?:定义|版本).*?(?:使用|采用)",
            r"不要使用.*?(?:sofa|sepsis|脓毒症)",
            r"^(?:是|确认)[，,\s]*(?:采用|使用|确认)?(?:该|此|上述)?(?:定义|主要暴露定义)[。.]?$",
            r"\bprimary[\s_-]+exposure[\s_-]+(?:is|uses?|will[\s_-]+be)\b",
            r"\b(?:use|adopt)[\s_-]+.+?[\s_-]+(?:as[\s_-]+the[\s_-]+)?primary[\s_-]+exposure\b",
            r"\bdo[\s_-]+not[\s_-]+use[\s_-]+.+?(?:sofa|sepsis)\b",
        )
    )


def _message_explicitly_selects_clustered_inference(message: str) -> bool:
    """Return whether this turn explicitly chooses patient-clustered inference."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:按|以).*?患者.*?(?:聚类|相关性|相关)",
            r"患者.*?(?:聚类稳健|聚类处理|cluster)",
            r"(?:聚类坐标|cluster.*coordinate).*?(?:患者|patient)",
            r"\bcluster(?:ed|ing)?[\s_-]+(?:robust[\s_-]+)?(?:by[\s_-]+)?patient\b",
            r"\bpatient[\s_-]+cluster(?:ed|ing|[\s_-]+robust)?\b",
        )
    )


def _message_explicitly_changes_variance_estimator(message: str) -> bool:
    """Return whether this turn directly selects an inference variance policy."""

    normalized = str(message or "").casefold()
    return _message_explicitly_selects_clustered_inference(message) or any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:普通|异方差|heteroskedastic).*?稳健标准误",
            r"(?:模型|model)[\s_-]*(?:标准误|based)",
            r"(?:不进行|不做|仅报告).*?(?:关联推断|统计推断|患病率)",
            r"\b(?:model[\s_-]*based|heteroskedastic[\s_-]*robust|counts[\s_-]*only)\b",
        )
    )


def _message_explicitly_selects_analysis_goal(message: str) -> bool:
    """Return whether this turn directly chooses the scientific analysis goal."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:分析目标|研究目标).*?(?:患病率|关联|因果|预测|描述)",
            r"(?:分析目标|研究目标).*?(?:同步|替换|改为|更新).*?(?:死亡|mortality|患病率|关联)",
            r"先报告.*?患病率.*?(?:关联|关系)",
            r"(?:观察性关联|非因果|不要写成因果|不作因果解释|不.*?因果效应)",
            r"^(?:描述|报告).*?患病率.*?(?:未调整|协变量调整|调整后).*?关联(?:分析)?(?:\s*[（(]推荐[）)])?$",
            r"^仅(?:描述|报告).*?患病率.*?(?:不分析|不评估).*?(?:死亡)?关联$",
            r"\b(?:analysis|study)[\s_-]+goal\b",
            r"\b(?:observational[\s_-]+association|non[\s_-]*causal)\b",
            r"\bfirst[\s_-]+report[\s_-]+.+?prevalence\b",
        )
    )


def _message_explicitly_selects_export_format(message: str) -> bool:
    """Return whether this turn directly chooses a package format."""

    normalized = str(message or "").casefold()
    return bool(
        re.search(
            r"(?:导出|数据包|格式|export).*?(?:parquet|csv|excel|xlsx)|"
            r"(?:parquet|csv|excel|xlsx).*?(?:导出|数据包|格式|export)|"
            r"^(?:parquet|csv|excel|xlsx)(?:\s*[（(].*?[）)])?$",
            normalized,
        )
    )


def _message_explicitly_requests_source_rebind(message: str) -> bool:
    """Return whether this turn explicitly asks to replace the bound source."""

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:更换|切换|重新选择|改用|换成).{0,24}(?:数据源|数据库|数据目录|文件夹)",
            r"(?:数据源|数据库|数据目录|文件夹).{0,24}(?:更换|切换|重新选择|改用|换成)",
            r"\b(?:switch|change|replace|rebind|use a different).{0,32}(?:data source|database|data folder)\b",
        )
    )


def _immutable_baseline_demographic(value: Any) -> str:
    """Return the owner-known baseline demographic family for one label."""

    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())
    if normalized in {"age", "年龄", "入院年龄"}:
        return "age"
    if normalized in {"sex", "gender", "性别", "生理性别"}:
        return "sex"
    return ""


def _message_explicitly_selects_demographic_adjustment(
    message: str, covariates: Sequence[Any]
) -> bool:
    """Return whether the user directly selected an immutable baseline roster."""

    normalized = str(message or "").casefold()
    families = [_immutable_baseline_demographic(value) for value in covariates]
    if not families or not all(families):
        return False
    if not re.search(r"(?:主要)?调整|协变量|adjust(?:ed|ing)?\s+for|covariates?", normalized):
        return False
    return all(
        (family == "age" and ("年龄" in normalized or re.search(r"\bage\b", normalized)))
        or (
            family == "sex"
            and (
                "性别" in normalized
                or re.search(r"\b(?:sex|gender)\b", normalized)
            )
        )
        for family in families
    )


def _message_explicitly_clears_covariate_adjustment(message: str) -> bool:
    """Return whether the user directly removes the adjustment roster.

    Empty model arguments are not authority to erase a prior scientific
    decision.  This helper recognizes only turns that explicitly choose an
    unadjusted or no-covariate analysis, so StudyContext can clear the roster
    and its owner metadata atomically without weakening the ordinary
    fail-closed merge behavior.
    """

    normalized = str(message or "").casefold()
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"(?:清除|移除|删除|取消).{0,24}(?:协变量|调整(?:登记|配置|变量)?)",
            r"(?:协变量|调整(?:登记|配置|变量)?).{0,24}(?:清除|移除|删除|取消)",
            r"(?:无|不使用|不登记|不保留|不进行|不执行|不做).{0,16}(?:调整协变量|协变量调整|协变量|调整)",
            r"(?:仅|只).{0,16}(?:描述|描述性).{0,16}(?:不调整|无调整|不含协变量)",
            r"\b(?:clear|remove|drop|delete).{0,32}(?:adjustment|covariates?)\b",
            r"\b(?:no|without).{0,16}(?:adjustment|adjustment covariates?|covariates?)\b",
            r"\b(?:unadjusted|descriptive-only|descriptive only)\b",
            r"\bdo not adjust\b",
        )
    )


def _merge_nested_study_patch(
    current: Mapping[str, Any], patch: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge bounded conversational object patches without erasing siblings.

    Pi tool arguments naturally contain only the slot the user just changed.
    StudyContext persists complete nested value objects.  Replacing the whole
    object would let ``cohort.exclude_readmissions=true`` silently erase age,
    comparator, and cohort-review authority.  Lists and scalar leaves still
    replace exactly; only mappings merge recursively.
    """

    def merge(existing: Mapping[str, Any], proposed: Mapping[str, Any]) -> Dict[str, Any]:
        combined = dict(existing)
        for key, value in proposed.items():
            prior = combined.get(key)
            if isinstance(prior, Mapping) and isinstance(value, Mapping):
                combined[key] = merge(prior, value)
            else:
                combined[key] = value
        return combined

    merged = dict(patch)
    for field in _NESTED_STUDY_PATCH_FIELDS:
        if field not in merged:
            continue
        existing = current.get(field)
        proposed = merged.get(field)
        if isinstance(existing, Mapping) and isinstance(proposed, Mapping):
            merged[field] = merge(existing, proposed)
    return merged


_GATED_SLOT_OMISSION_CODES = {
    "cohort": "study_cohort_preset_confirmation_required",
    "outcome": "study_primary_outcome_confirmation_required",
    "primary_exposure": "study_primary_exposure_confirmation_required",
    "analysis_goal": "study_analysis_goal_confirmation_required",
}


def _unconfirmed_gated_slots(
    params: Mapping[str, Any],
    current: Mapping[str, Any],
    user_message: str,
) -> FrozenSet[str]:
    """Gated slots this turn proposes but does not explicitly select.

    Each confirmation guard below keeps a genuinely confirmed change in the
    same call and drops only its own unconfirmed slot.  That test must not
    read a *sibling* candidate slot as the user's confirmation: one research
    question mentioning Sepsis-3 and ICU mortality proposes outcome,
    primary_exposure and analysis_goal together, and letting each one vouch
    for the others made all three take the silent-drop branch, so none could
    ever be saved and none could ever reach its typed
    ``*_confirmation_required`` receipt.
    """

    message = str(user_message or "").strip()
    if not message:
        return frozenset()
    current = current or {}
    unconfirmed: set[str] = set()

    proposed_cohort = params.get("cohort")
    if isinstance(proposed_cohort, Mapping) and "preset" in proposed_cohort:
        proposed_preset = str(proposed_cohort.get("preset") or "").strip().lower()
        current_preset = str((current.get("cohort") or {}).get("preset") or "").strip()
        if (
            proposed_preset
            and proposed_preset != current_preset
            and not _message_explicitly_selects_all_stays(message)
            and not _message_explicitly_selects_first_stay(message)
        ):
            unconfirmed.add("cohort")

    for slot, selects_explicitly in (
        ("outcome", _message_explicitly_selects_primary_outcome),
        ("primary_exposure", _message_explicitly_selects_primary_exposure),
        ("analysis_goal", _message_explicitly_selects_analysis_goal),
    ):
        proposed = str(params.get(slot) or "").strip()
        if not proposed or proposed == str(current.get(slot) or "").strip():
            continue
        if not selects_explicitly(message):
            unconfirmed.add(slot)

    return frozenset(unconfirmed)


def _has_other_confirmed_change(
    params: Mapping[str, Any],
    *,
    slot_machinery: Set[str],
    unconfirmed_gated: FrozenSet[str],
) -> bool:
    """Whether this call also carries a change the user actually confirmed."""

    return any(
        key in params
        for key in _STUDY_SETUP_FIELDS - slot_machinery - unconfirmed_gated
    )


def _restore_unconfirmed_study_slot(
    patch: Dict[str, Any],
    current: Mapping[str, Any],
    *,
    slot: str,
) -> None:
    """Discard one unconfirmed proposal while retaining confirmed changes."""

    prior = current.get(slot)
    if prior:
        patch[slot] = prior
    else:
        patch.pop(slot, None)

    if slot in {"outcome", "primary_exposure"}:
        proposed_execution = patch.get("execution_concepts")
        prior_execution = current.get("execution_concepts")
        if isinstance(proposed_execution, Mapping):
            restored_execution = dict(proposed_execution)
            prior_value = (
                prior_execution.get(slot)
                if isinstance(prior_execution, Mapping)
                else None
            )
            if prior_value:
                restored_execution[slot] = prior_value
            else:
                restored_execution.pop(slot, None)
            if restored_execution:
                patch["execution_concepts"] = restored_execution
            else:
                patch.pop("execution_concepts", None)
        if "modules" in patch:
            prior_modules = list(current.get("modules") or [])
            if prior_modules:
                patch["modules"] = prior_modules
            else:
                patch.pop("modules", None)


def _update_study_context(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Persist conversational setup through the existing typed owner."""

    _require_args(params, allowed=_STUDY_SETUP_FIELDS)
    binding = context.session.binding
    current = _bound_context(binding)
    if binding.study_context_id and current is None:
        return _result(
            context,
            status="not_found",
            code="study_context_not_found",
            summary="The bound StudyContext no longer exists; no replacement context was created.",
            owner="easyicu.webserver.study_contexts",
        )
    if current and current.get("active_job_id"):
        return _result(
            context,
            status="blocked",
            code="study_context_active_job_conflict",
            summary="Study setup cannot change while its authoritative EasyICU job is active.",
            owner="easyicu.webserver.study_contexts",
        )

    patch = {
        key: params[key]
        for key in _STUDY_SETUP_FIELDS - {"bind_active_export", "bind_source_id"}
        if key in params
    }
    if isinstance(patch.get("sensitivity_specs"), list):
        patch["sensitivity_specs"] = [
            {key: value for key, value in spec.items() if value is not None}
            if isinstance(spec, Mapping)
            else spec
            for spec in patch["sensitivity_specs"]
        ]
    omitted_unconfirmed_fields: list[str] = []
    unconfirmed_omissions: list[Dict[str, str]] = []
    unconfirmed_gated = _unconfirmed_gated_slots(
        params, current or {}, context.user_message
    )

    def _record_unconfirmed_omission(slot: str, field: str) -> None:
        """Make one silently dropped slot visible in the typed receipt."""

        if field in omitted_unconfirmed_fields:
            return
        omitted_unconfirmed_fields.append(field)
        unconfirmed_omissions.append(
            {"field": field, "code": _GATED_SLOT_OMISSION_CODES[slot]}
        )

    if current:
        patch = _merge_nested_study_patch(current, patch)
    if current and current.get("id"):
        patch["id"] = current["id"]
    proposed_modules = {
        str(value or "").strip().lower() for value in (params.get("modules") or [])
    }
    if "sepsis3_sofa2" in proposed_modules:
        from easyicu.concept.selection_policy import (
            concept_selection_confirmation_key,
            evaluate_concept_selection,
        )

        concept_id = "sep3_sofa2"
        confirmation_key = concept_selection_confirmation_key(concept_id)
        previously_confirmed = bool(
            ((current or {}).get("confirmations") or {}).get(confirmation_key)
        )
        decision = evaluate_concept_selection(
            concept_id,
            user_intent=context.user_message,
            owner_confirmed=previously_confirmed,
        )
        if not decision.allowed:
            return _result(
                context,
                status="blocked",
                code=decision.reason_code,
                summary=(
                    "The proposed feature module contains an explicit-only "
                    "clinical variant that the user did not request."
                ),
                owner="easyicu.concept.selection_policy",
                details={
                    **decision.to_dict(),
                    "field": "modules",
                    "proposed_module": "sepsis3_sofa2",
                    "canonical_alternative_module": "sepsis3_sofa1",
                },
            )
        confirmations = dict(((current or {}).get("confirmations") or {}))
        confirmations.update(dict(patch.get("confirmations") or {}))
        confirmations[confirmation_key] = True
        patch["confirmations"] = confirmations
    proposed_cohort = params.get("cohort")
    if isinstance(proposed_cohort, Mapping) and "preset" in proposed_cohort:
        try:
            normalized_preset = dataio.normalize_export_cohort_preset(
                proposed_cohort.get("preset")
            )
        except dataio.ExportCohortError as exc:
            return _result(
                context,
                status="blocked",
                code=exc.error,
                summary=(
                    "Use one canonical Data Extraction cohort preset and keep "
                    "the user's natural-language cohort wording in label or review."
                ),
                owner="easyicu.webserver.dataio",
                details={
                    "field": "cohort.preset",
                    "allowed": list(exc.detail.get("supported") or []),
                },
            )
        normalized_cohort = dict(patch.get("cohort") or {})
        normalized_cohort["preset"] = normalized_preset
        patch["cohort"] = normalized_cohort
        current_preset = str(((current or {}).get("cohort") or {}).get("preset") or "")
        if (
            normalized_preset == "adult_all"
            and current_preset != "adult_all"
            and str(context.user_message or "").strip()
            and not _message_explicitly_selects_all_stays(context.user_message)
        ):
            other_confirmed_change = _has_other_confirmed_change(
                params,
                slot_machinery={
                    "cohort",
                    "execution_concepts",
                    "modules",
                    "confirmations",
                },
                unconfirmed_gated=unconfirmed_gated,
            )
            if other_confirmed_change:
                _restore_unconfirmed_study_slot(patch, current or {}, slot="cohort")
                _record_unconfirmed_omission("cohort", "cohort.preset")
            else:
                return _result(
                    context,
                    status="blocked",
                    code="study_cohort_all_stays_confirmation_required",
                    summary=(
                        "An adult ICU population does not by itself choose between "
                        "all eligible stays and one stay per patient."
                    ),
                    owner="easyicu.webserver.study_contexts",
                    details={
                        "field": "cohort.preset",
                        "proposed": "adult_all",
                        "safe_alternatives": ["adult_all", "adult_first"],
                    },
                )
        if (
            normalized_preset == "adult_first"
            and current_preset != "adult_first"
            and not _message_explicitly_selects_first_stay(context.user_message)
        ):
            other_confirmed_change = _has_other_confirmed_change(
                params,
                slot_machinery={
                    "cohort",
                    "execution_concepts",
                    "modules",
                    "confirmations",
                },
                unconfirmed_gated=unconfirmed_gated,
            )
            if other_confirmed_change:
                _restore_unconfirmed_study_slot(patch, current or {}, slot="cohort")
                _record_unconfirmed_omission("cohort", "cohort.preset")
            else:
                return _result(
                    context,
                    status="blocked",
                    code="study_cohort_first_stay_confirmation_required",
                    summary=(
                        "A first-ICU-stay restriction changes the analysis unit and "
                        "requires the user's explicit selection."
                    ),
                    owner="easyicu.webserver.study_contexts",
                    details={
                        "field": "cohort.preset",
                        "proposed": "adult_first",
                        "safe_alternatives": ["adult_all", "all_icu"],
                    },
                )
    if (
        params.get("outcome")
        and str(context.user_message or "").strip()
        and str(params.get("outcome") or "").strip()
        != str((current or {}).get("outcome") or "").strip()
        and not _message_explicitly_selects_primary_outcome(context.user_message)
    ):
        other_confirmed_change = _has_other_confirmed_change(
            params,
            slot_machinery={
                "outcome",
                "execution_concepts",
                "modules",
                "confirmations",
            },
            unconfirmed_gated=unconfirmed_gated,
        )
        if other_confirmed_change:
            _restore_unconfirmed_study_slot(patch, current or {}, slot="outcome")
            _record_unconfirmed_omission("outcome", "outcome")
        else:
            return _result(
                context,
                status="blocked",
                code="study_primary_outcome_confirmation_required",
                summary=(
                    "Mentioning an outcome in the research question records candidate "
                    "intent but does not confirm the primary outcome definition."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "outcome"},
            )
    if (
        params.get("primary_exposure")
        and str(context.user_message or "").strip()
        and str(params.get("primary_exposure") or "").strip()
        != str((current or {}).get("primary_exposure") or "").strip()
        and not _message_explicitly_selects_primary_exposure(context.user_message)
    ):
        other_confirmed_change = _has_other_confirmed_change(
            params,
            slot_machinery={
                "primary_exposure",
                "execution_concepts",
                "modules",
                "confirmations",
            },
            unconfirmed_gated=unconfirmed_gated,
        )
        if other_confirmed_change:
            _restore_unconfirmed_study_slot(
                patch, current or {}, slot="primary_exposure"
            )
            _record_unconfirmed_omission("primary_exposure", "primary_exposure")
        else:
            return _result(
                context,
                status="blocked",
                code="study_primary_exposure_confirmation_required",
                summary=(
                    "Mentioning an exposure in the research question records candidate "
                    "intent but does not confirm its primary executable definition."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "primary_exposure"},
            )
    if (
        params.get("analysis_goal")
        and str(context.user_message or "").strip()
        and str(params.get("analysis_goal") or "").strip()
        != str((current or {}).get("analysis_goal") or "").strip()
        and not _message_explicitly_selects_analysis_goal(context.user_message)
    ):
        other_confirmed_change = _has_other_confirmed_change(
            params,
            slot_machinery={"analysis_goal"},
            unconfirmed_gated=unconfirmed_gated,
        )
        if other_confirmed_change:
            _restore_unconfirmed_study_slot(
                patch, current or {}, slot="analysis_goal"
            )
            _record_unconfirmed_omission("analysis_goal", "analysis_goal")
        else:
            return _result(
                context,
                status="blocked",
                code="study_analysis_goal_confirmation_required",
                summary=(
                    "The research question suggests candidate objectives but does "
                    "not confirm the analysis goal or causal interpretation."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "analysis_goal"},
            )
    proposed_covariates = list(params.get("covariates") or [])
    explicitly_clears_covariates = (
        "covariates" in params
        and not proposed_covariates
        and _message_explicitly_clears_covariate_adjustment(context.user_message)
    )
    if explicitly_clears_covariates:
        # Recursive StudyContext merging intentionally preserves omitted
        # nested siblings.  An explicit empty roster is different: the roster,
        # its rationales/temporal roles, and executable bindings form one owner
        # contract and must be cleared together.  The user-message gate above
        # prevents a model-authored empty list from silently erasing a prior
        # scientific decision.
        patch["covariates"] = []
        patch["covariate_selection"] = "exact"
        patch["covariate_rationales"] = {}
        patch["covariate_temporal_roles"] = {}
        patch["covariate_operationalizations"] = {}
        prior_execution = (current or {}).get("execution_concepts")
        proposed_execution = patch.get("execution_concepts")
        if isinstance(proposed_execution, Mapping) or isinstance(
            prior_execution, Mapping
        ):
            synchronized_execution = dict(
                proposed_execution
                if isinstance(proposed_execution, Mapping)
                else prior_execution
            )
            synchronized_execution["covariates"] = []
            patch["execution_concepts"] = synchronized_execution
    elif _message_explicitly_selects_demographic_adjustment(
        context.user_message, proposed_covariates
    ):
        # Age and sex are immutable baseline demographics. Once the user
        # explicitly chooses them for adjustment, their temporal role is an
        # EasyICU-owned semantic fact rather than another user decision.
        patch["covariate_selection"] = "exact"
        rationales = dict(patch.get("covariate_rationales") or {})
        temporal_roles = dict(patch.get("covariate_temporal_roles") or {})
        for covariate in proposed_covariates:
            key = str(covariate)
            family = _immutable_baseline_demographic(covariate)
            rationales[key] = (
                f"Pre-specified {family} baseline demographic confounder selected by the user."
            )
            temporal_roles[key] = "baseline_static"
        patch["covariate_rationales"] = rationales
        patch["covariate_temporal_roles"] = temporal_roles
        # If this study already has executable concept coordinates, keep the
        # exact user-approved demographic roster synchronized with them. Age
        # and sex are owner-known catalog identifiers, so leaving a stale
        # ``execution_concepts.covariates=[]`` would make the visible study
        # configuration disagree with the ResearchContext and fail only after
        # a background run starts.
        prior_execution = (current or {}).get("execution_concepts")
        proposed_execution = patch.get("execution_concepts")
        if isinstance(proposed_execution, Mapping) or isinstance(
            prior_execution, Mapping
        ):
            synchronized_execution = dict(
                proposed_execution
                if isinstance(proposed_execution, Mapping)
                else prior_execution
            )
            synchronized_execution["covariates"] = list(proposed_covariates)
            patch["execution_concepts"] = synchronized_execution
    if params.get("export_format") and _message_explicitly_selects_export_format(
        context.user_message
    ):
        # An export-format turn authorizes exactly one new confirmation. Do not
        # let a model-authored partial confirmations object erase prior owner
        # receipts such as the already-confirmed feature window.
        confirmations = dict(((current or {}).get("confirmations") or {}))
        confirmations["export_format"] = True
        patch["confirmations"] = confirmations
    for field in (
        "title",
        "question",
        "purpose",
        "outcome",
        "primary_exposure",
        "comparator",
        "analysis_goal",
    ):
        if patch.get(field):
            reject_sensitive_message(str(patch[field]))
    for covariate in patch.get("covariates") or []:
        reject_sensitive_message(str(covariate))
    for rationale in (patch.get("covariate_rationales") or {}).values():
        reject_sensitive_message(str(rationale))
    for operational in (
        patch.get("covariate_operationalizations") or {}
    ).values():
        reject_sensitive_message(str(operational))
    for spec in patch.get("sensitivity_specs") or []:
        if isinstance(spec, Mapping):
            for variable in spec.get("execution_variables") or []:
                reject_sensitive_message(str(variable))

    bind_source_id = str(params.get("bind_source_id") or "").strip()
    if params.get("bind_active_export") or bind_source_id:
        registry = sources.load_registry()
        if bind_source_id:
            source = next(
                (
                    row
                    for row in (registry.get("sources") or [])
                    if isinstance(row, Mapping)
                    and row.get("ok")
                    and str(row.get("id") or "") == bind_source_id
                ),
                None,
            )
            if source is None:
                return _result(
                    context,
                    status="blocked",
                    code="pi_data_source_not_registered",
                    summary=(
                        "The selected source id is not a validated registered "
                        "EasyICU export. List sources again before binding."
                    ),
                    owner="easyicu.webserver.sources",
                )
            selected_path = str(source.get("path") or "").strip()
        else:
            selected_path = str(registry.get("active_path") or "").strip()
            source = next(
                (
                    row
                    for row in (registry.get("sources") or [])
                    if isinstance(row, Mapping)
                    and row.get("ok")
                    and str(row.get("path") or "") == selected_path
                ),
                None,
            )
        if not selected_path or source is None:
            return _result(
                context,
                status="blocked",
                code="no_active_export",
                summary="No validated active EasyICU export is available to bind.",
                owner="easyicu.webserver.sources",
            )
        database = str(source.get("database") or "").strip()
        if not database:
            return _result(
                context,
                status="blocked",
                code="pi_data_source_database_unavailable",
                summary=(
                    "The selected registered export has no validated database "
                    "identity and cannot be bound for extraction."
                ),
                owner="easyicu.webserver.sources",
            )
        current_source = (current or {}).get("data_source")
        current_source = (
            current_source if isinstance(current_source, Mapping) else {}
        )
        current_path = str(current_source.get("path") or "").strip()
        source_rebind_blocked = bool(
            context.session.data_source_authorization.status == "confirmed"
            and current_path
            and current_path != selected_path
            and not _message_explicitly_requests_source_rebind(context.user_message)
        )
        if source_rebind_blocked:
            has_other_setup_change = any(
                key in params
                for key in _STUDY_SETUP_FIELDS
                - {"bind_active_export", "bind_source_id"}
            )
            if not has_other_setup_change:
                return _result(
                    context,
                    status="blocked",
                    code="study_data_source_rebind_confirmation_required",
                    summary=(
                        "This conversation already has a confirmed data source. "
                        "Changing it requires an explicit user request."
                    ),
                    owner="easyicu.webserver.study_contexts",
                    details={"field": "data_source"},
                )
        else:
            patch["data_source"] = {
                "path": selected_path,
                "label": source.get("label") or "active EasyICU export",
                "database": database,
            }
            confirmations = dict(((current or {}).get("confirmations") or {}))
            confirmations.update(dict(patch.get("confirmations") or {}))
            # A validated registry row is the Data Extraction owner's receipt
            # for an already prepared EasyICU export. Reusing it must not send
            # the user back to the raw-database folder workflow.
            confirmations["extraction_completed"] = True
            patch["confirmations"] = confirmations
    effective_source = patch.get("data_source") or (current or {}).get("data_source")
    effective_source = effective_source if isinstance(effective_source, Mapping) else {}
    effective_modules = patch.get("modules")
    if effective_modules is None:
        effective_modules = (current or {}).get("modules") or []
    execution = patch.get("execution_concepts")
    if execution is not None:
        try:
            normalized_execution = study_contexts.normalize_execution_concepts(
                execution
            )
        except study_contexts.StudyContextError as exc:
            return _result(
                context,
                status="blocked",
                code=str(exc.detail.get("error") or "study_execution_concepts_invalid"),
                summary="The StudyContext owner rejected the executable concept binding.",
                owner="easyicu.webserver.study_contexts",
                details={
                    key: exc.detail.get(key)
                    for key in ("field", "fields", "max_items")
                    if exc.detail.get(key) is not None
                },
            )
        source_path = str(effective_source.get("path") or "").strip()
        if not source_path:
            return _result(
                context,
                status="blocked",
                code="study_execution_source_required",
                summary="Bind a validated data source before saving executable concept identifiers.",
                owner="easyicu.webserver.study_contexts",
            )
        from easyicu.research_agent.acquisition.catalog import build_available_catalog

        catalog = build_available_catalog(Path(source_path).expanduser())
        allowed_modules = {
            str(value).strip().lower()
            for value in effective_modules
            if str(value).strip()
        }
        concept_modules = {
            concept.concept_id: Path(concept.file_name).stem.lower()
            for concept in catalog.concepts
        }
        bound_ids = [
            value
            for value in (
                normalized_execution.get("outcome"),
                normalized_execution.get("primary_exposure"),
                *(normalized_execution.get("covariates") or []),
            )
            if value
        ]
        absent_from_source = sorted(set(bound_ids) - set(concept_modules))
        if absent_from_source:
            return _result(
                context,
                status="blocked",
                code="study_execution_concepts_unavailable",
                summary="One or more executable concepts are absent from the selected source.",
                owner="easyicu.research_agent.acquisition.catalog",
                details={"unavailable_concepts": absent_from_source},
            )
        required_modules = {concept_modules[value] for value in bound_ids}
        if required_modules - allowed_modules:
            # Module selection for already-authorized exact concepts is an
            # EasyICU implementation detail, not another scientific choice.
            # Add only the catalog-proven owning modules; never invent a module
            # name or broaden to unrelated features.
            patch["modules"] = sorted(allowed_modules | required_modules)
            effective_modules = patch["modules"]
            allowed_modules = set(effective_modules)
        primary_concept = normalized_execution.get("primary_exposure")
        if primary_concept:
            from easyicu.concept.selection_policy import (
                concept_selection_confirmation_key,
                evaluate_concept_selection,
            )

            # The model must not self-authorize an experimental concept by
            # putting its name in a generated exposure label or analysis goal.
            # The persisted scientific question is the user-intent authority.
            # The current user message is the strongest authority.  Do not
            # combine it with a model-authored replacement question: phrases
            # such as "do not filter the cohort by Sepsis" can otherwise be
            # misread as negating the separately explicit SOFA-2 selection.
            # For older already-persisted contexts, the prior scientific
            # question remains a compatibility fallback.
            authority_intent = str(
                context.user_message or (current or {}).get("question") or ""
            )
            confirmation_key = concept_selection_confirmation_key(primary_concept)
            confirmations = dict(((current or {}).get("confirmations") or {}))
            confirmations.update(dict(patch.get("confirmations") or {}))
            previously_confirmed = bool(
                ((current or {}).get("confirmations") or {}).get(confirmation_key)
            )
            decision = evaluate_concept_selection(
                primary_concept,
                user_intent=authority_intent,
                owner_confirmed=previously_confirmed,
            )
            if not decision.allowed:
                return _result(
                    context,
                    status="blocked",
                    code=decision.reason_code,
                    summary=(
                        "The selected primary concept is an explicit-only "
                        "variant that the user's research intent did not request."
                    ),
                    owner="easyicu.concept.selection_policy",
                    details=decision.to_dict(),
                )
            # Persist only the host-verified decision.  A model-proposed
            # confirmation cannot authorize itself because it is ignored
            # above; the current user turn or a prior owner receipt must pass.
            confirmations[confirmation_key] = True
            patch["confirmations"] = confirmations
        patch["execution_concepts"] = normalized_execution
    if "analysis_design" in patch:
        proposed_design = patch.get("analysis_design")
        proposed_design = (
            proposed_design if isinstance(proposed_design, Mapping) else {}
        )
        requested_design = params.get("analysis_design")
        requested_design = (
            requested_design if isinstance(requested_design, Mapping) else {}
        )
        current_design = (current or {}).get("analysis_design")
        current_design = (
            current_design if isinstance(current_design, Mapping) else {}
        )
        if (
            requested_design.get("variance_estimator") == "none_counts_only"
            and _message_explicitly_changes_variance_estimator(
                context.user_message
            )
        ):
            # ``cluster_unit`` is meaningful only for clustered inference.
            # The nested merge may retain the previous patient coordinate, so
            # remove it when the user explicitly switches to counts-only
            # description instead of forcing a second technical decision.
            proposed_design = dict(proposed_design)
            proposed_design.pop("cluster_unit", None)
            patch["analysis_design"] = proposed_design
        if (
            current_design
            and str(context.user_message or "").strip()
            and not _message_explicitly_changes_variance_estimator(
                context.user_message
            )
        ):
            proposed_design = dict(proposed_design)
            for field in ("variance_estimator", "cluster_unit"):
                prior_value = current_design.get(field)
                if prior_value:
                    proposed_design[field] = prior_value
                else:
                    proposed_design.pop(field, None)
            patch["analysis_design"] = proposed_design
        proposes_patient_clustering = (
            str(proposed_design.get("variance_estimator") or "").strip()
            == "cluster_robust"
            or str(proposed_design.get("cluster_unit") or "").strip() == "patient"
        )
        current_patient_clustering = (
            str(current_design.get("variance_estimator") or "").strip()
            == "cluster_robust"
            or str(current_design.get("cluster_unit") or "").strip() == "patient"
        )
        if (
            proposes_patient_clustering
            and not current_patient_clustering
            and str(context.user_message or "").strip()
            and not _message_explicitly_selects_clustered_inference(
                context.user_message
            )
        ):
            return _result(
                context,
                status="blocked",
                code="study_patient_clustering_confirmation_required",
                summary=(
                    "Retaining repeated ICU stays does not by itself authorize "
                    "patient-clustered inference."
                ),
                owner="easyicu.webserver.study_contexts",
                details={"field": "analysis_design.variance_estimator"},
            )
        try:
            patch["analysis_design"] = study_contexts.normalize_analysis_design(
                patch.get("analysis_design")
            )
        except study_contexts.StudyContextError as exc:
            return _result(
                context,
                status="blocked",
                code=str(exc.detail.get("error") or "study_analysis_design_invalid"),
                summary="The StudyContext owner rejected the proposed analysis design.",
                owner="easyicu.webserver.study_contexts",
                details={
                    key: exc.detail.get(key)
                    for key in ("field", "fields", "allowed")
                    if exc.detail.get(key) is not None
                },
            )
    if "covariate_selection" in patch:
        selection = str(patch.get("covariate_selection") or "").strip()
        if selection not in {"planner_selectable", "exact"}:
            return _result(
                context,
                status="blocked",
                code="study_covariate_selection_invalid",
                summary="The adjustment-set authority must be planner_selectable or exact.",
                owner="easyicu.webserver.study_contexts",
                details={
                    "field": "covariate_selection",
                    "allowed": ["exact", "planner_selectable"],
                },
            )
        patch["covariate_selection"] = selection
    if not patch:
        raise PiCopilotError(
            "pi_tool_arguments_required",
            "At least one typed study-setup field is required.",
        )

    # Validate the complete proposed StudyContext before spending the one-use
    # Configure grant.  A rejected typed proposal is not a mutation; consuming
    # the grant here used to prevent Pi from correcting a mechanical schema
    # error in the same turn.
    try:
        patch = study_contexts.validate_context_update(
            patch,
            current_context=current,
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        return _result(
            context,
            status="blocked",
            code=str(exc.detail.get("error") or "study_context_update_blocked"),
            summary="The typed StudyContext owner rejected the proposed setup update.",
            owner="easyicu.webserver.study_contexts",
            details={
                key: exc.detail.get(key)
                for key in (
                    "error",
                    "field",
                    "fields",
                    "reason",
                    "allowed",
                    "required_design",
                    "alternative",
                )
                if exc.detail.get(key) is not None
            },
        )

    if "analysis_design" in params:
        proposed = {**dict(current or {}), **patch}
        try:
            agent_pipeline_runs.validate_analysis_design_for_execution(proposed)
        except agent_pipeline_runs.ResearchPipelineRunError as exc:
            return _result(
                context,
                status="blocked",
                code=exc.code,
                summary=str(exc),
                owner="easyicu.webserver.agent_pipeline_runs.analysis_design",
                details=exc.details,
            )

    grant_block = _consume_action(context, "configure")
    if grant_block is not None:
        return grant_block

    try:
        updated = study_contexts.upsert_context(
            patch,
            active=True,
            expected_revision=(int(current.get("revision") or 0) if current else None),
            require_revision=bool(current),
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        return _result(
            context,
            status="blocked",
            code=str(exc.detail.get("error") or "study_context_update_blocked"),
            summary="The typed StudyContext owner rejected the proposed setup update.",
            owner="easyicu.webserver.study_contexts",
            details={
                key: exc.detail.get(key)
                for key in (
                    "error",
                    "field",
                    "fields",
                    "reason",
                    "allowed",
                    "required_design",
                    "alternative",
                    "expected_revision",
                    "current_revision",
                )
                if exc.detail.get(key) is not None
            },
        )
    workflow = _workflow_snapshot(context, study_override=updated)
    summary = (
        f"Saved typed StudyContext revision {int(updated.get('revision') or 0)} "
        "and projected the post-update workflow for the next scientific decision."
    )
    if omitted_unconfirmed_fields:
        # A dropped scientific slot must never read as a clean success.  The
        # model can only explain the gap to the user when the omission and its
        # reason code travel in the same receipt as the write.
        summary += (
            " NOT saved this turn: "
            + ", ".join(omitted_unconfirmed_fields)
            + ". Those slots stay unset because the turn only mentions them as "
            "candidate intent; ask the user to select each one explicitly."
        )
    result = _result(
        context,
        status="ok",
        code="study_context_updated",
        summary=summary,
        owner="easyicu.webserver.study_contexts",
        details={
            "study": project_study_context(updated),
            "workflow": workflow,
            "rebind_required": True,
            "host_rebind_after_turn": True,
            "omitted_unconfirmed_fields": omitted_unconfirmed_fields,
            "unconfirmed_omissions": unconfirmed_omissions,
        },
    )
    context.invalidate_authority("study_context_updated")
    return result


def _idea_projection(payload: Mapping[str, Any]) -> Dict[str, Any]:
    ideas = payload.get("idea_ledger")
    ideas = ideas if isinstance(ideas, list) else []
    selected_id = str(payload.get("selected_idea_id") or "")
    selected = next(
        (
            row
            for row in ideas
            if isinstance(row, Mapping)
            and (not selected_id or str(row.get("idea_id") or "") == selected_id)
        ),
        {},
    )
    concepts = []
    for raw in list(selected.get("mapped_concepts") or [])[:24]:
        if not isinstance(raw, Mapping):
            continue
        concepts.append(
            {
                key: raw.get(key)
                for key in (
                    "concept_id",
                    "label",
                    "module",
                    "role",
                    "status",
                    "available",
                )
                if raw.get(key) is not None
            }
        )
    pre = payload.get("pre_experiment")
    pre = pre if isinstance(pre, Mapping) else {}
    return bounded_json_projection(
        {
            "run_id": payload.get("run_id"),
            "selected_idea_id": selected.get("idea_id") or selected_id or None,
            "idea": {
                key: selected.get(key)
                for key in (
                    "idea_id",
                    "idea_title",
                    "population",
                    "exposure_or_predictor",
                    "outcome",
                    "analysis_family",
                    "rationale",
                    "go_no_go",
                    "go_no_go_reason",
                    "next_action",
                    "plan_status",
                )
                if selected.get(key) is not None
            }
            | {"mapped_concepts": concepts},
            "feasibility": {
                "status": pre.get("status"),
                "reason": pre.get("reason"),
                "reportable": bool(pre.get("reportable")),
            },
            "privacy": {
                "patient_rows_returned": False,
                "external_llm_calls": 0,
                "reportable": False,
            },
        }
    )


def _mine_ideas(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=(
            "topic",
            "title",
            "excerpt",
            "journal",
            "year",
            "doi",
            "pmid",
        ),
    )
    grant_block = _consume_action(context, "idea")
    if grant_block is not None:
        return grant_block
    study = _bound_context(context.session.binding) or {}
    topic = str(params.get("topic") or study.get("question") or "").strip()
    if not topic:
        return _result(
            context,
            status="blocked",
            code="idea_topic_required",
            summary="Save a scientific question or provide a bounded idea topic first.",
            owner="easyicu.webserver.ideas.mining",
        )
    body = {
        "source_type": "manual",
        "topic": topic,
        "research_question": topic,
        "title": str(params.get("title") or topic)[:220],
        "excerpt": str(params.get("excerpt") or topic)[:1200],
        "journal": params.get("journal"),
        "year": params.get("year"),
        "doi": params.get("doi"),
        "pmid": params.get("pmid"),
    }
    try:
        mined = idea_mining.mine_ideas(body)
    except idea_mining.IdeaMiningWebError as exc:
        detail = exc.detail
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "idea_mining_blocked"),
            summary=str(detail.get("reason") or "Idea Mining rejected the request."),
            owner="easyicu.webserver.ideas.mining",
        )
    projected = _idea_projection(mined)
    return _result(
        context,
        status="ok",
        code="easyicu_idea_mined",
        summary=(
            "Created a local metadata-only idea candidate and feasibility draft; "
            "it is not a scientific result or novelty claim."
        ),
        owner="easyicu.webserver.ideas.mining",
        details={"idea_mining": projected},
    )


def _search_literature(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Run the existing PubMed Idea Mining owner after one-turn opt-in."""

    _require_args(params, allowed=("topic", "journal", "limit"))
    try:
        requested_limit = max(1, min(int(params.get("limit") or 5), 8))
    except (TypeError, ValueError):
        requested_limit = 5
    study = _bound_context(context.session.binding) or {}
    topic = str(params.get("topic") or study.get("question") or "").strip()
    if not topic:
        return _result(
            context,
            status="blocked",
            code="literature_topic_required",
            summary="Save a scientific question or provide a literature topic first.",
            owner="easyicu.webserver.ideas.mining",
        )
    idea_handoff = (
        study.get("idea_handoff")
        if isinstance(study.get("idea_handoff"), Mapping)
        else {}
    )
    execution_concepts = (
        study.get("execution_concepts")
        if isinstance(study.get("execution_concepts"), Mapping)
        else {}
    )
    bound_idea_run = str(idea_handoff.get("run_id") or "").strip()
    bound_idea_id = str(idea_handoff.get("idea_id") or "").strip()
    if not (bound_idea_run and bound_idea_id):
        missing_scope = [
            field
            for field, value in (
                ("primary_exposure", study.get("primary_exposure")),
                ("outcome", study.get("outcome")),
                (
                    "execution_concepts.primary_exposure",
                    execution_concepts.get("primary_exposure"),
                ),
                ("execution_concepts.outcome", execution_concepts.get("outcome")),
            )
            if not str(value or "").strip()
        ]
        if missing_scope:
            # The Planner -- not Copilot -- chooses the exposure and outcome, so
            # before a plan exists these fields are empty *by design*.  Blocking
            # here is still correct (plan-authority literature must cite exact
            # concepts), but a bare refusal made Pi conclude the plan itself was
            # impossible and stop.  Name the owning next step instead, so an
            # unplanned study is routed to the Planner rather than dead-ended.
            plan_workflow = _workflow_snapshot(context, study_override=study)
            plan_ready = (
                str(plan_workflow.get("next_action_code") or "")
                == "provider_ready_to_generate_plan"
            )
            summary = (
                "Bind the study exposure and outcome to exact source concepts "
                "before searching literature for Plan authority. Broad Idea "
                "Mining literature must use an accepted idea handoff instead."
            )
            if plan_ready:
                summary = (
                    "Plan-authority literature is produced by the EasyICU Planner "
                    "run, which also chooses the exposure and outcome this search "
                    "would need. Do not ask the user to supply them: start the "
                    "formal research plan instead. This is not a data or "
                    "permission failure."
                )
            return _result(
                context,
                status="blocked",
                code="literature_study_scope_incomplete",
                summary=summary,
                owner="easyicu.webserver.literature_authority",
                details={
                    "missing_fields": missing_scope,
                    "next_action_code": plan_workflow.get("next_action_code"),
                    "plan_generation_ready": plan_ready,
                },
            )
    grant_block = _consume_action(context, "literature")
    if grant_block is not None:
        return grant_block
    try:
        if bound_idea_run and bound_idea_id:
            checked = idea_mining.check_prior_art(
                {
                    "run_id": bound_idea_run,
                    "idea_id": bound_idea_id,
                    "allow_network": True,
                }
            )
            prior = checked.get("prior_art")
            prior = prior if isinstance(prior, Mapping) else {}
            raw_candidates = [
                row
                for row in list(prior.get("results") or [])[:requested_limit]
                if isinstance(row, Mapping)
            ]
            candidates = [
                {
                    "citation_key": f"idea_pubmed_{str(row.get('pmid') or '').strip()}",
                    "title": row.get("title"),
                    "journal": row.get("journal") or row.get("source"),
                    "year": row.get("year") or row.get("pubdate"),
                    "doi": row.get("doi"),
                    "pmid": row.get("pmid"),
                    "url": (
                        f"https://pubmed.ncbi.nlm.nih.gov/{str(row.get('pmid') or '').strip()}/"
                        if str(row.get("pmid") or "").strip()
                        else None
                    ),
                    "evidence_quote": (
                        str(
                            row.get("abstract_excerpt")
                            or row.get("evidence_sentence")
                            or (
                                "Matched the accepted Idea Mining prior-art query: "
                                + str(row.get("query") or "")
                            )
                        )[:600]
                    ),
                    "design_excerpt": str(
                        row.get("design_excerpt")
                        or row.get("abstract_excerpt")
                        or ""
                    )[:1200],
                    "publication_types": [
                        str(value)[:120]
                        for value in list(row.get("publication_types") or [])[:12]
                        if str(value).strip()
                    ],
                    "matched_queries": [str(row.get("query") or "")[:1500]]
                    if str(row.get("query") or "").strip()
                    else [],
                    "matched_query_strata": ["accepted_idea_prior_art"],
                }
                for row in raw_candidates
            ]
            discovered = {
                "status": prior.get("status"),
                "search_performed": prior.get("search_performed"),
                "queries_to_run": prior.get("queries_to_run") or [],
                "network_calls": prior.get("network_calls") or 0,
                "source_candidates": candidates,
            }
            receipt_binding = idea_mining.prior_art_receipt_binding(bound_idea_run)
        else:
            cohort_scope = (
                study.get("cohort")
                if isinstance(study.get("cohort"), Mapping)
                else {}
            )
            data_source = (
                study.get("data_source")
                if isinstance(study.get("data_source"), Mapping)
                else {}
            )
            analysis_design = (
                study.get("analysis_design")
                if isinstance(study.get("analysis_design"), Mapping)
                else {}
            )
            discovered = idea_mining.discover_literature(
                {
                    "topic": topic,
                    "exposure": str(study.get("primary_exposure") or "").strip(),
                    "outcome": str(study.get("outcome") or "").strip(),
                    "exposure_concept": str(
                        execution_concepts.get("primary_exposure") or ""
                    ).strip(),
                    "outcome_concept": str(
                        execution_concepts.get("outcome") or ""
                    ).strip(),
                    "population": " ".join(
                        str(cohort_scope.get(key) or "").strip()
                        for key in ("label", "review", "preset")
                        if str(cohort_scope.get(key) or "").strip()
                    ),
                    "database": str(data_source.get("database") or "").strip(),
                    "analysis_family": str(
                        analysis_design.get("analysis_family") or ""
                    ).strip(),
                    "journal": str(params.get("journal") or "").strip(),
                    "limit": requested_limit,
                    "allow_network": True,
                }
            )
            receipt_binding = None
    except idea_mining.IdeaMiningWebError as exc:
        detail = exc.detail
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "literature_search_blocked"),
            summary=str(
                detail.get("reason") or "Idea Mining rejected the literature search."
            ),
            owner="easyicu.webserver.ideas.mining",
        )
    generic_authority_binding: Optional[Dict[str, Any]] = None
    updated_study: Optional[Dict[str, Any]] = None
    if performed := bool(discovered.get("search_performed")):
        if not (bound_idea_run and bound_idea_id):
            study_id = str(study.get("id") or "").strip()
            bound_study_id = str(
                context.session.binding.study_context_id or ""
            ).strip()
            stored = (
                study_contexts.get_context(study_id)
                if study_id and bound_study_id == study_id
                else None
            )
            if stored is None and context.session.binding.study_context_id:
                return _result(
                    context,
                    status="blocked",
                    code="literature_authority_study_missing",
                    summary=(
                        "The bound StudyContext disappeared before its literature "
                        "receipt could be attached. Search results were not authorized."
                    ),
                    owner="easyicu.webserver.literature_authority",
                )
            if stored is not None:
                if int(stored.get("revision") or 0) != int(study.get("revision") or 0):
                    return _result(
                        context,
                        status="blocked",
                        code="literature_authority_study_revision_conflict",
                        summary=(
                            "The study changed while PubMed was being searched. "
                            "Search again against the current study revision."
                        ),
                        owner="easyicu.webserver.literature_authority",
                    )
                try:
                    generic_authority_binding = (
                        literature_authority.persist_literature_authority(
                            study=stored,
                            discovered=discovered,
                        )
                    )
                    updated_study = study_contexts.bind_literature_authority(
                        study_id,
                        generic_authority_binding,
                        expected_revision=int(stored.get("revision") or 0),
                    )
                except literature_authority.LiteratureAuthorityError as exc:
                    return _result(
                        context,
                        status="blocked",
                        code=exc.code,
                        summary=exc.message,
                        owner="easyicu.webserver.literature_authority",
                    )
                except study_contexts.StudyContextError as exc:
                    return _result(
                        context,
                        status="blocked",
                        code=str(
                            exc.detail.get("error")
                            or "literature_authority_binding_failed"
                        ),
                        summary=(
                            "The StudyContext changed before the literature receipt "
                            "could be bound. Search again against the current revision."
                        ),
                        owner="easyicu.webserver.study_contexts",
                    )

    candidates = [
        row
        for row in list(discovered.get("source_candidates") or [])[
            :requested_limit
        ]
        if isinstance(row, Mapping)
    ]
    status = str(discovered.get("status") or "search_failed")
    performed = bool(discovered.get("search_performed"))
    query_count = len(list(discovered.get("queries_to_run") or []))
    projection = compile_literature_tool_projection(
        discovered=discovered,
        candidates=candidates,
        idea_receipt_binding=(
            receipt_binding if isinstance(receipt_binding, Mapping) else None
        ),
        study_authority_binding=generic_authority_binding,
        bound_idea_run_id=bound_idea_run,
    )
    if updated_study is not None:
        projection.update(
            {
                "authority_update": {
                    "study_context_id": str(updated_study.get("id") or "")[:160],
                    "study_revision": int(updated_study.get("revision") or 0),
                    "reason": "study_literature_authority_updated",
                },
                "rebind_required": True,
                "host_rebind_after_turn": True,
            }
        )
    result = _result(
        context,
        status="ok" if performed else "blocked",
        code=(
            "easyicu_literature_search_completed"
            if performed
            else "easyicu_literature_search_not_performed"
        ),
        summary=(
            f"PubMed metadata search {status}: {len(candidates)} retrieval candidate(s) "
            f"from {query_count} prespecified query string(s). No full text, "
            "patient rows, or external LLM was used."
            + (
                " The current Idea handoff must now be re-accepted so this "
                "exact literature receipt becomes part of Plan authority."
                if receipt_binding
                else ""
            )
            + (
                " The exact Web search receipt is now digest-bound to this "
                "StudyContext and will be reused by Research Agent planning."
                if generic_authority_binding
                else ""
            )
        ),
        owner="easyicu.webserver.ideas.mining",
        details=projection,
    )
    if updated_study is not None:
        context.invalidate_authority("study_literature_authority_updated")
    return result


def _prepare_idea_handoff(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("run_id", "idea_id", "plan_edits"),
        required=("run_id",),
    )
    grant_block = _consume_action(context, "idea")
    if grant_block is not None:
        return grant_block
    body = {
        "run_id": str(params.get("run_id") or "").strip(),
        "idea_id": str(params.get("idea_id") or "").strip(),
        "plan_edits": str(params.get("plan_edits") or "").strip()[:1200],
    }
    try:
        plan = idea_mining.plan_idea(body)
        handoff = idea_mining.create_handoff(body)
    except idea_mining.IdeaMiningWebError as exc:
        detail = exc.detail
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "idea_handoff_blocked"),
            summary=str(detail.get("reason") or "Idea handoff preparation failed."),
            owner="easyicu.webserver.ideas.handoff",
        )
    plan_body = plan.get("plan") if isinstance(plan.get("plan"), Mapping) else {}
    agent_seed = (
        handoff.get("agent_seed")
        if isinstance(handoff.get("agent_seed"), Mapping)
        else {}
    )
    details = bounded_json_projection(
        {
            "run_id": handoff.get("run_id"),
            "idea_id": handoff.get("idea_id"),
            "candidate_topic": handoff.get("candidate_topic"),
            "go_no_go": handoff.get("go_no_go"),
            "go_no_go_reason": handoff.get("go_no_go_reason"),
            "plan": {
                key: plan_body.get(key)
                for key in (
                    "research_question",
                    "analysis_family",
                    "population",
                    "exposure",
                    "comparator",
                    "outcome",
                    "time_window",
                    "plan_status",
                    "selection_mode",
                )
                if plan_body.get(key) is not None
            },
            "agent_seed": {
                "study_id": agent_seed.get("study_id"),
                "question": agent_seed.get("question"),
                "requires_human_confirmation": True,
                "reportable": False,
                "draft_unlocked": False,
            },
            "canonical_handoff_sha256": handoff.get("canonical_handoff_sha256"),
            "next_step": "Review the draft, then save the agreed fields into StudyContext before extraction or analysis.",
        }
    )
    return _result(
        context,
        status="ok",
        code="easyicu_idea_handoff_prepared",
        summary=(
            "Prepared a canonical metadata-only Idea Mining handoff. The plan "
            "still requires user confirmation in the Copilot conversation."
        ),
        owner="easyicu.webserver.ideas.handoff",
        details={"idea_handoff": details},
    )


def _accept_idea_handoff(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind one canonical Idea Mining handoff to the current StudyContext."""

    _require_args(
        params,
        allowed=("run_id", "idea_id", "plan_edits"),
        required=("run_id", "idea_id"),
    )
    grant_block = _consume_action(context, "idea")
    if grant_block is not None:
        return grant_block
    current = _bound_context(context.session.binding)
    if not current or not current.get("id"):
        return _result(
            context,
            status="blocked",
            code="study_context_required",
            summary="Bind a typed StudyContext before accepting an Idea Mining handoff.",
            owner="easyicu.webserver.study_contexts",
        )
    if current.get("active_job_id"):
        return _result(
            context,
            status="blocked",
            code="study_context_active_job_conflict",
            summary="An Idea Mining handoff cannot replace study setup while an authoritative job is active.",
            owner="easyicu.webserver.study_contexts",
        )
    body = {
        "run_id": str(params.get("run_id") or "").strip(),
        "idea_id": str(params.get("idea_id") or "").strip(),
        "plan_edits": str(params.get("plan_edits") or "").strip()[:1200],
    }
    try:
        plan = idea_mining.plan_idea(body)
        handoff = idea_mining.create_handoff(body)
    except idea_mining.IdeaMiningWebError as exc:
        detail = exc.detail
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "idea_handoff_blocked"),
            summary=str(detail.get("reason") or "Idea handoff acceptance failed."),
            owner="easyicu.webserver.ideas.handoff",
        )
    digest = str(handoff.get("canonical_handoff_sha256") or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return _result(
            context,
            status="blocked",
            code="canonical_idea_handoff_digest_required",
            summary="The selected Idea Mining handoff has no valid canonical digest.",
            owner="easyicu.webserver.ideas.handoff",
        )
    canonical = (
        handoff.get("canonical_handoff")
        if isinstance(handoff.get("canonical_handoff"), Mapping)
        else {}
    )
    try:
        prior_art_binding = idea_mining.prior_art_receipt_binding(body["run_id"])
    except idea_mining.IdeaMiningWebError as exc:
        detail = exc.detail
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "prior_art_receipt_invalid"),
            summary=str(
                detail.get("reason")
                or "The Idea Mining prior-art receipt could not be bound."
            ),
            owner="easyicu.webserver.ideas.mining",
        )
    selected_row = (
        canonical.get("selected_ledger_row")
        if isinstance(canonical.get("selected_ledger_row"), Mapping)
        else {}
    )
    mapped_concepts = [
        row
        for row in selected_row.get("mapped_concepts") or []
        if isinstance(row, Mapping)
    ]
    module_by_concept = {
        str(row.get("concept_id") or "").strip(): str(row.get("module") or "").strip()
        for row in mapped_concepts
        if str(row.get("concept_id") or "").strip()
        and str(row.get("module") or "").strip()
    }
    predictor_concept = str(canonical.get("resolved_predictor_concept") or "").strip()
    outcome_concept = str(
        canonical.get("resolved_outcome_concept")
        or canonical.get("target_outcome")
        or ""
    ).strip()
    analysis_concepts = list(
        dict.fromkeys(
            str(value).strip()
            for value in canonical.get("resolved_analysis_concepts") or []
            if str(value).strip()
        )
    )
    execution_concepts = list(
        dict.fromkeys(
            value
            for value in (predictor_concept, outcome_concept, *analysis_concepts)
            if value
        )
    )
    missing_modules = [
        concept_id
        for concept_id in execution_concepts
        if concept_id not in module_by_concept
    ]
    if not canonical or not execution_concepts or missing_modules:
        return _result(
            context,
            status="blocked",
            code="canonical_idea_execution_contract_required",
            summary=(
                "The canonical Idea Mining handoff does not contain a complete "
                "digest-bound concept-to-module execution contract."
            ),
            owner="easyicu.webserver.ideas.handoff",
            details={"missing_concept_modules": missing_modules},
        )
    plan_body = plan.get("plan") if isinstance(plan.get("plan"), Mapping) else {}
    agent_seed = (
        handoff.get("agent_seed")
        if isinstance(handoff.get("agent_seed"), Mapping)
        else {}
    )
    patch: Dict[str, Any] = {
        "id": current["id"],
        "idea_handoff": {
            "schema_version": "easyicu.pi-idea-selection/1",
            "run_id": body["run_id"],
            "idea_id": str(handoff.get("idea_id") or body["idea_id"]),
            "canonical_handoff_sha256": digest,
            "status": "accepted",
            "accepted_at": str(handoff.get("created_at") or ""),
            "go_no_go": str(handoff.get("go_no_go") or ""),
            "go_no_go_reason": str(handoff.get("go_no_go_reason") or "")[:500],
            **dict(prior_art_binding or {}),
        },
        "current_stage": "study_setup",
        "last_route": "guided",
    }
    derived_fields = {
        "title": handoff.get("candidate_topic"),
        "question": plan_body.get("research_question") or agent_seed.get("question"),
        "outcome": outcome_concept or plan_body.get("outcome"),
        "primary_exposure": predictor_concept or plan_body.get("exposure"),
        "comparator": plan_body.get("comparator"),
        "analysis_goal": canonical.get("analysis_family")
        or plan_body.get("analysis_family"),
    }
    limits = {
        "title": 160,
        "question": 1200,
        "outcome": 500,
        "primary_exposure": 160,
        "comparator": 500,
        "analysis_goal": 1200,
    }
    patch.update(
        {
            key: str(value).strip()[: limits[key]]
            for key, value in derived_fields.items()
            if str(value or "").strip()
        }
    )
    # A comparator belongs to its exposure.  Re-selecting an idea can change
    # the digest-bound predictor while the new handoff intentionally leaves
    # comparator selection to the remaining conversational setup.  Retaining
    # the previous idea's comparator in that case silently creates an invalid
    # study contract (for example, a per-unit lab contrast on a binary
    # phenotype).  Clear only this dependent slot; the user can then confirm
    # the new comparator in the same Copilot conversation.
    previous_exposure = str(current.get("primary_exposure") or "").strip()
    if (
        predictor_concept != previous_exposure
        and not str(plan_body.get("comparator") or "").strip()
    ):
        patch["comparator"] = ""
    patch["modules"] = list(
        dict.fromkeys(
            module_by_concept[concept_id] for concept_id in execution_concepts
        )
    )
    requested_adjustment_concepts = list(
        dict.fromkeys(
            str(value).strip()
            for value in selected_row.get("requested_adjustment_concepts") or []
            if str(value).strip()
        )
    )
    invalid_adjustment_concepts = [
        concept_id
        for concept_id in requested_adjustment_concepts
        if concept_id not in analysis_concepts
        or concept_id in {predictor_concept, outcome_concept}
    ]
    if invalid_adjustment_concepts:
        return _result(
            context,
            status="blocked",
            code="canonical_idea_adjustment_contract_invalid",
            summary=(
                "The canonical Idea Mining handoff contains adjustment variables "
                "outside its digest-bound analysis concept set."
            ),
            owner="easyicu.webserver.ideas.handoff",
            details={"invalid_adjustment_concepts": invalid_adjustment_concepts},
        )
    # Only an explicit adjustment clause may populate this slot. Mentioning a
    # marker for feasibility or descriptive review is not authority to adjust
    # for it; an absent clause intentionally leaves the list empty for the
    # conversational setup/human Plan gate.
    patch["covariates"] = requested_adjustment_concepts
    patch["execution_concepts"] = {
        **({"outcome": outcome_concept} if outcome_concept else {}),
        **(
            {"primary_exposure": predictor_concept}
            if predictor_concept
            else {}
        ),
        "covariates": requested_adjustment_concepts,
    }
    try:
        updated = study_contexts.upsert_context(
            patch,
            active=True,
            expected_revision=int(current.get("revision") or 0),
            require_revision=True,
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        return _result(
            context,
            status="blocked",
            code=str(exc.detail.get("error") or "idea_handoff_binding_blocked"),
            summary="The StudyContext owner rejected the selected Idea Mining handoff.",
            owner="easyicu.webserver.study_contexts",
        )
    result = _result(
        context,
        status="ok",
        code="easyicu_idea_handoff_accepted",
        summary=(
            "Accepted the digest-bound Idea Mining handoff and projected its "
            "agreed study fields into the current StudyContext. Continue the "
            "remaining setup in this conversation before extraction."
        ),
        owner="easyicu.webserver.study_contexts",
        details={
            "idea_selection": updated.get("idea_handoff"),
            "study": project_study_context(updated),
            "rebind_required": True,
            "host_rebind_after_turn": True,
        },
    )
    context.invalidate_authority("idea_handoff_accepted")
    return result


def _start_extraction(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("source_mode", "database"))
    grant_block = _consume_action(context, "extract")
    if grant_block is not None:
        return grant_block
    study = _bound_context(context.session.binding)
    if not study or not study.get("id"):
        return _result(
            context,
            status="blocked",
            code="study_context_required",
            summary="Complete and bind the typed study setup before extraction.",
            owner="easyicu.webserver.study_contexts",
        )
    source_mode = str(params.get("source_mode") or "").strip()
    requested_database = str(params.get("database") or "").strip()
    current_message = str(context.user_message or "").casefold()
    local_request = any(
        marker in current_message
        for marker in ("本机", "本地", "local", "数据文件夹", "data folder")
    )
    explicit_database = None
    for database, pattern in (
        ("miiv", r"\bmimic[\s_-]*(?:iv|4)\b"),
        ("mimic", r"\bmimic[\s_-]*(?:iii|3)\b"),
        ("eicu", r"\beicu\b"),
        ("aumc", r"\b(?:amsterdamumcdb|aumc)\b"),
        ("hirid", r"\bhirid\b"),
        ("sic", r"\b(?:sicdb|sic)\b"),
    ):
        if re.search(pattern, current_message):
            explicit_database = database
            break
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    if (
        not source_mode
        and not requested_database
        and not str(source.get("path") or "").strip()
        and local_request
        and explicit_database
    ):
        # Recover a model-omitted tool argument only from the current user's
        # exact local database wording. Bare family names remain unresolved.
        source_mode = "local"
        requested_database = explicit_database
    if source_mode:
        if source_mode != "local":
            raise PiCopilotError(
                "pi_data_source_mode_invalid",
                "The explicit Data Extraction source mode must be 'local'.",
            )
        if not requested_database:
            raise PiCopilotError(
                "pi_database_selection_required",
                "Choose one exact supported database before opening a local folder.",
            )
        try:
            profile = get_database_profile(requested_database)
        except KeyError as exc:
            raise PiCopilotError(
                "pi_database_not_supported",
                "Choose one exact database returned by easyicu_list_data_sources.",
            ) from exc
        if not profile.is_public:
            raise PiCopilotError(
                "pi_database_not_supported",
                "Official demos use easyicu_prepare_demo_source; local selection "
                "requires a supported full-database family.",
            )
        release_label = (
            f" {profile.reference_release}" if profile.reference_release else ""
        )
        resource = extraction_workspace_resource(
            study, state="setup", expected_database=profile.key
        )
        resource["label"] = (
            f"Connect local {profile.display_name}{release_label}"
        )[:160]
        return _result(
            context,
            status="ok",
            code="easyicu_local_source_workspace_ready",
            summary=(
                f"Opened the path-private local folder workflow for "
                f"{profile.display_name}{release_label}. Folder identity and layout "
                "must pass the Data Extraction scan before registration or query."
            ),
            owner="easyicu.webserver.dataio",
            details={
                "database": {
                    "key": profile.key,
                    "label": profile.display_name,
                    "reference_release": profile.reference_release,
                },
                "resource": resource,
            },
        )
    registry = sources.load_registry()
    source_path = str(source.get("path") or "").strip()
    registered_source = next(
        (
            row
            for row in (registry.get("sources") or [])
            if isinstance(row, Mapping)
            and row.get("ok")
            and str(row.get("path") or "").strip() == source_path
        ),
        None,
    )
    database = str(source.get("database") or "").strip()
    modules = [str(item) for item in (study.get("modules") or []) if str(item).strip()]
    if not source_path or not database or not modules:
        missing = [
            name
            for name, ready in (
                ("data_source.path", bool(source_path)),
                ("data_source.database", bool(database)),
                ("modules", bool(modules)),
            )
            if not ready
        ]
        return _result(
            context,
            status="blocked",
            code="extraction_setup_incomplete",
            summary="The typed study setup does not yet contain a prepared source, database, and feature modules.",
            owner="easyicu.webserver.study_contexts",
            details={
                "missing_fields": missing,
                "resource": extraction_workspace_resource(
                    study, state="setup"
                ),
            },
        )
    export_format = str(study.get("export_format") or "parquet").strip().lower()
    if export_format not in {"csv", "parquet"}:
        return _result(
            context,
            status="blocked",
            code="extraction_export_format_unsupported",
            summary="Choose CSV or Parquet in the conversational study setup before extraction.",
            owner="easyicu.webserver.routes.jobs",
            details={"supported_formats": ["csv", "parquet"]},
        )
    handoff_receipt: Optional[Dict[str, Any]] = None
    registered_export_path: Optional[str] = None
    try:
        if registered_source is not None:
            handoff = compile_registered_export_handoff(study, registered_source)
            handoff_receipt = handoff.public_receipt()
            if handoff.reusable:
                return _result(
                    context,
                    status="ok",
                    code="easyicu_registered_export_reused",
                    summary=(
                        "The bound registered export exactly matches the requested "
                        "cohort, window, format, and module contract; no duplicate "
                        "extraction was started."
                    ),
                    owner="easyicu.webserver.pi_copilot.extraction_handoff",
                    details={
                        "active_export": {
                            "present": True,
                            "path_digest": path_digest(source_path),
                            "source_id": str(registered_source.get("id") or "")[:80],
                        },
                        "extraction_contract": handoff_receipt,
                        "resource": extraction_workspace_resource(
                            study, state="review"
                        ),
                    },
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
    except dataio.ExportCohortError as exc:
        return _result(
            context,
            status="blocked",
            code=exc.error,
            summary=(
                "The Copilot-to-Data Extraction owner could not compile a "
                "path-safe extraction handoff from the registered export."
            ),
            owner="easyicu.webserver.pi_copilot.extraction_handoff",
            details={"reason_code": exc.error},
        )
    from easyicu.webserver.routes.jobs import jobs_extract

    try:
        submitted = jobs_extract(
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
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return _result(
            context,
            status="blocked",
            code=str(detail.get("error") or "easyicu_extraction_submission_blocked"),
            summary="The existing EasyICU extraction boundary rejected the request.",
            owner="easyicu.webserver.routes.jobs",
            details={
                key: detail.get(key)
                for key in ("error", "reason", "running", "max_running", "job_id")
                if detail.get(key) is not None
            },
        )
    result = _result(
        context,
        status="ok",
        code="easyicu_extraction_submitted",
        summary=f"Submitted EasyICU feature extraction job {submitted.get('job_id')} from the bound typed study setup.",
        owner="easyicu.webserver.routes.jobs",
        details={
            key: submitted.get(key)
            for key in (
                "job_id",
                "kind",
                "status",
                "study_context_id",
                "study_context_revision",
            )
            if submitted.get(key) is not None
        }
        | {
            **(
                {"extraction_contract": handoff_receipt}
                if handoff_receipt is not None
                else {}
            ),
            "resource": extraction_workspace_resource(
                study,
                state="running",
                job_id=submitted.get("job_id"),
            )
        },
    )
    context.invalidate_authority("easyicu_extraction_submitted")
    return result


def _account_environment_for_research_provider(
    context: ToolExecutionContext,
) -> tuple[Optional[Mapping[str, str]], Optional[Dict[str, Any]]]:
    """Resolve the immutable Codex account bound to this Copilot session."""

    research_provider = context.session.research_provider
    if research_provider.provider != "codex":
        return None, None
    from easyicu.webserver import codex_account_sessions

    try:
        return (
            codex_account_sessions.environment_for_binding(
                str(research_provider.account_session_sha256 or ""),
                model=research_provider.model,
            ),
            None,
        )
    except codex_account_sessions.CodexAccountSessionError as exc:
        return None, _result(
            context,
            status="blocked",
            code=exc.code,
            summary=(
                "The Codex account bound to this conversation is no longer "
                "available; start a new Copilot session after signing in again."
            ),
            owner="easyicu.webserver.codex_account_sessions",
        )


# Launch rejections that mean "an EasyICU step has not run yet", not "the
# request was malformed".  These need a next action, otherwise Pi reports them
# as an unrecoverable data problem and stops.
_RUN_SUBMISSION_NEXT_STEP_SUMMARIES = {
    "research_pipeline_manifest_required": (
        "The bound directory is a converted ICU database, not a prepared "
        "EasyICU export package, so the Research Agent has nothing "
        "manifest-backed to bind its evidence to. This is not a bad data "
        "source and not a permission problem: run easyicu_start_extraction to "
        "prepare an export package from it first, then generate the plan. Do "
        "not ask the user to re-pick or validate their folder."
    ),
    "no_export_files": (
        "The bound directory carries no readable EasyICU export files yet. Run "
        "easyicu_start_extraction to prepare the package before planning."
    ),
}


def _development_resume_source_job_id(
    context: ToolExecutionContext,
    study: Mapping[str, Any],
) -> str:
    """Return one owned Planner checkpoint job for an unchanged study.

    The development efficiency budget deliberately stops a long progressive
    plan after a bounded provider turn and preserves its validated prefix.  A
    user-authorized retry should continue that prefix, not spend the next turn
    regenerating it.  All other terminal failures remain fresh-run only.
    """

    latest = _select_run(context)
    if not latest:
        return ""
    resumable_reasons = {
        "research_pipeline_planner_efficiency_budget_exhausted",
        "research_pipeline_plan_contract_exhausted",
        "research_pipeline_progressive_compile_failed",
    }
    candidates = [latest]
    if str(latest.get("gate_reason") or "") == "data_foundation_blocked":
        # A metadata-only Planner continuation used to be routed through the
        # generic full-run adapter when the terminal checkpoint had already
        # compiled but failed the host article gate.  That adapter leaves an
        # empty data-foundation projection newer than the still-valid Planner
        # checkpoint.  It is evidence of the failed adapter call, not a plan
        # authority, and must not hide the unchanged resumable checkpoint.
        candidates.extend(
            row
            for row in _run_rows(context)
            if str(row.get("run_id") or "")
            != str(latest.get("run_id") or "")
        )
    latest = next(
        (
            row
            for row in candidates
            if str(row.get("gate_reason") or "") in resumable_reasons
        ),
        None,
    )
    if not latest:
        return ""
    if str(latest.get("run_status") or "") != "failed":
        return ""
    if latest.get("development_planner_checkpoint_available") is not True:
        # The workflow projection and the launch adapter must agree on resume
        # eligibility.  A contract-exhausted run can fail before the first
        # checkpoint exists; forwarding its job id would turn an explicit
        # fresh-plan confirmation into a resume request that the lower owner
        # must reject.
        return ""
    study_id = str(study.get("id") or "").strip()
    if not study_id or str(latest.get("study_id") or "") != study_id:
        return ""
    planned_digest = str(
        latest.get("scientific_configuration_sha256") or ""
    ).strip()
    if (
        len(planned_digest) != 64
        or planned_digest
        != study_contexts.scientific_configuration_sha256(dict(study))
    ):
        return ""
    match = re.fullmatch(
        r"run_([A-Za-z0-9][A-Za-z0-9_-]{0,79})",
        str(latest.get("run_id") or ""),
    )
    if match is None:
        return ""
    project_dir = Path(str(latest.get("project_dir") or "")).expanduser()
    expected_root = research_pipeline_project_root(study_id).resolve()
    try:
        resolved_project = project_dir.resolve(strict=True)
        resolved_project.relative_to(expected_root)
    except (FileNotFoundError, OSError, ValueError):
        return ""
    # ProjectWorkspace owns ``<project-root>/<study-id>/run_<job>``.  The
    # lower run owner revalidates the exact slug and checkpoint chain; this
    # selector only proves that the candidate is one bounded child run inside
    # the current Copilot project root.
    if resolved_project.parent.parent != expected_root:
        return ""
    return match.group(1)


def _run(
    context: ToolExecutionContext,
    params: Mapping[str, Any],
    *,
    plan_revision_source_run_id: str = "",
) -> Dict[str, Any]:
    _require_args(params, allowed=("run_type", "llm_provider"))
    requested_run_type = str(params.get("run_type") or "").strip().lower()
    # The UI holds the user's one-turn intent outside the model.  When Pi uses
    # the tool's optional/default form, prefer the strongest action the user
    # explicitly granted: clicking "full analysis" must not silently become a
    # preflight and then ask for a second permission.  With no provider grant,
    # the conservative default remains the deterministic local preflight.
    provider_run_granted = "provider_run" in context.allowed_actions
    local_run_granted = "run" in context.allowed_actions
    literature_search_authorized = context.grant.was_provided("literature")
    run_type = requested_run_type or ("full" if provider_run_granted else "preflight")
    if run_type == "preflight" and provider_run_granted and not local_run_granted:
        # Pi does not receive the host-held grant list.  If it conservatively
        # asks for the default preflight while the user selected *only* the
        # full-analysis permission, the host intent is unambiguous: run the
        # authorized full workflow rather than rejecting it for a permission
        # the user was never expected to select as well.
        run_type = "full"
    if run_type not in {"preflight", "full"}:
        return _result(
            context,
            status="blocked",
            code="unsupported_run_type",
            summary="Choose either the deterministic preflight or a full Research Agent run.",
            owner="easyicu.webserver.routes.agent",
        )
    study = _bound_context(context.session.binding)
    if (
        run_type == "full"
        and context.session.binding.study_context_id
        and study
        and study.get("id")
    ):
        workflow = _workflow_snapshot(context, study_override=study)
        # Only the fields the user owns -- the bound question and an identified
        # data source -- may block plan generation.  The Planner decides cohort,
        # exposure, outcome, window, modules and analysis design and presents
        # them for review, so requiring them here forced Copilot to interrogate
        # the user for the very choices the plan exists to make.
        planning_prerequisites_missing = list(
            workflow.get("planning_prerequisites_missing") or []
        )
        if planning_prerequisites_missing:
            # Scientific workflow readiness is an owner invariant, not a
            # provider-permission problem.  Check it before consuming the
            # one-turn provider grant so an out-of-order model call cannot
            # expose internal grant names or start a premature Planner run.
            return _result(
                context,
                status="blocked",
                code="study_setup_incomplete",
                summary=(
                    "Bind the research question and confirm a data source before "
                    "generating the formal research plan."
                ),
                owner="easyicu.webserver.pi_copilot.workflow",
                details={
                    "next_action_code": workflow.get("next_action_code"),
                    "planning_prerequisites_missing": planning_prerequisites_missing,
                    "missing_setup_fields": list(
                        workflow.get("missing_setup_fields") or []
                    ),
                },
            )
    grant_action = "provider_run" if run_type == "full" else "run"
    grant_block = _consume_action(context, grant_action)
    if grant_block is not None:
        return grant_block
    requested_provider = str(params.get("llm_provider") or "").strip()
    if run_type == "full" and is_offline_llm_choice(requested_provider):
        return _result(
            context,
            status="blocked",
            code="pi_full_mock_not_scientific",
            summary=(
                "Pi will not present an offline mock manuscript as a completed "
                "scientific analysis. Choose the separately configured Research "
                "Agent provider and grant one full analysis run."
            ),
            owner="easyicu.webserver.provider_gate",
        )
    # The selected credential authority is frozen when the Copilot session is
    # created. Model output may request a run, but cannot switch account, API,
    # or model underneath an already-open research conversation.
    research_provider = context.session.research_provider
    provider = research_provider.provider if run_type == "full" else "mock"
    if run_type == "full" and not context.session.external_llm_opt_in:
        return _result(
            context,
            status="blocked",
            code="external_llm_opt_in_required",
            summary="A full provider analysis requires the session's explicit external-model authorization.",
            owner="easyicu.webserver.provider_gate",
        )
    if not study or not study.get("id"):
        return _result(
            context,
            status="blocked",
            code="study_context_required",
            summary="Create and bind a typed StudyContext before starting an EasyICU run.",
            owner="easyicu.webserver.study_contexts",
        )
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    source_path = str(source.get("path") or "").strip()
    if not source_path:
        return _result(
            context,
            status="blocked",
            code="study_context_source_required",
            summary=(
                "Bind one validated registered EasyICU data source to this "
                "StudyContext before starting a run."
            ),
            owner="easyicu.webserver.study_contexts",
        )
    account_environment = None
    if run_type == "full":
        account_environment, account_error = (
            _account_environment_for_research_provider(context)
        )
        if account_error is not None:
            return account_error
    # Import lazily to keep the route-composition module out of this package's
    # import graph. The function remains the one existing run submission path;
    # this adapter does not reconstruct its validation or JobManager behavior.
    from easyicu.webserver.routes.agent import submit_agent_run

    development_resume_source_job_id = (
        _development_resume_source_job_id(context, study)
        if run_type == "full"
        else ""
    )

    try:
        submitted = submit_agent_run(
            {
                "path": source_path,
                "study_context_id": study["id"],
                "question": study.get("question"),
                "run_type": run_type,
                "llm_provider": provider,
                "external_llm_opt_in": run_type == "full",
                "engine": (
                    "research_agent_pipeline"
                    if run_type == "full"
                    else "native_summary"
                ),
                **(
                    {
                        "credential_source": research_provider.credential_source,
                    }
                    if run_type == "full"
                    else {}
                ),
                **(
                    {"literature_search_authorized": True}
                    if run_type == "full" and literature_search_authorized
                    else {}
                ),
                **(
                    {
                        "plan_revision_source_run_id": str(
                            plan_revision_source_run_id
                        )
                    }
                    if plan_revision_source_run_id
                    else {}
                ),
                **(
                    {
                        "development_resume_source_job_id": (
                            development_resume_source_job_id
                        )
                    }
                    if development_resume_source_job_id
                    else {}
                ),
            },
            account_environment=account_environment,
            metadata_only_planning_authorized=run_type == "full",
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        submission_code = str(detail.get("error") or "easyicu_run_submission_blocked")
        # A single generic sentence for every launch rejection erased the
        # owner's own diagnosis, so Pi had to guess a cause and wrote prose the
        # user could not act on.  Carry the owning message through, and give the
        # one rejection that is a missing *step* rather than a bad request its
        # own next action.
        summary = _bounded_model_text(detail.get("message"), 400) or (
            "The existing EasyICU run submission boundary rejected the request."
        )
        if submission_code in _RUN_SUBMISSION_NEXT_STEP_SUMMARIES:
            summary = _RUN_SUBMISSION_NEXT_STEP_SUMMARIES[submission_code]
        return _result(
            context,
            status="blocked",
            code=submission_code,
            summary=summary,
            owner="easyicu.webserver.routes.agent",
            details={
                key: detail.get(key)
                for key in (
                    "error",
                    "message",
                    "details",
                    "blockers",
                    "blocker_codes",
                    "job_id",
                )
                if detail.get(key) is not None
            },
        )
    result = _result(
        context,
        status="ok",
        code=(
            "easyicu_full_run_submitted"
            if run_type == "full"
            else "easyicu_run_submitted"
        ),
        summary=(
            (
                "Submitted an EasyICU Research Agent Planner continuation "
                f"job {submitted.get('job_id')} from validated checkpoint job "
                f"{development_resume_source_job_id}."
            )
            if run_type == "full" and development_resume_source_job_id
            else f"Submitted full EasyICU Research Agent job {submitted.get('job_id')}."
            if run_type == "full"
            else f"Submitted deterministic EasyICU preflight job {submitted.get('job_id')}."
        ),
        owner="easyicu.webserver.routes.agent",
        details={
            key: submitted.get(key)
            for key in (
                "job_id",
                "kind",
                "status",
                "study_context_id",
                "study_context_revision",
                "engine",
            )
            if submitted.get(key) is not None
        }
        | {
            # A real ResearchAgentPipeline run id does not exist at submission
            # time.  This explicit state prevents a historical bound run id
            # from being presented as the identity of the new job.
            "run_id_status": "pending_pipeline_start",
            **(
                {
                    "development_resume_source_job_id": (
                        development_resume_source_job_id
                    )
                }
                if development_resume_source_job_id
                else {}
            ),
        },
    )
    context.invalidate_authority("easyicu_run_submitted")
    return result


def _resume(context: ToolExecutionContext, params: Mapping[str, Any]) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("job_id", "run_id", "decision", "reviewer", "note"),
    )
    decision = str(params.get("decision") or "").strip().lower()
    if decision:
        if decision not in {"approved", "rejected"}:
            return _result(
                context,
                status="blocked",
                code="research_pipeline_review_decision_invalid",
                summary="Choose approved or rejected for the pending Research Agent plan.",
                owner="easyicu.research_agent.pipeline",
            )
        grant_block = _consume_action(context, "provider_run")
        if grant_block is not None:
            return grant_block
        if not context.session.external_llm_opt_in:
            return _result(
                context,
                status="blocked",
                code="external_llm_opt_in_required",
                summary="Resuming a provider-backed Research Agent run requires explicit external-model authorization.",
                owner="easyicu.webserver.provider_gate",
            )
        study = _bound_context(context.session.binding)
        run_id = str(
            params.get("run_id") or context.session.binding.run_id or ""
        ).strip()
        if not study or not study.get("id") or not run_id:
            return _result(
                context,
                status="blocked",
                code="research_pipeline_review_coordinates_required",
                summary="The pending plan and its bound research project are required before review can resume.",
                owner="easyicu.research_agent.pipeline",
            )
        account_environment, account_error = (
            _account_environment_for_research_provider(context)
        )
        if account_error is not None:
            return account_error
        from easyicu.webserver.routes.agent import submit_agent_run_review

        try:
            submitted = submit_agent_run_review(
                {
                    "study_context_id": study["id"],
                    "run_id": run_id,
                    "decision": decision,
                    "reviewer": params.get("reviewer") or "local_web_reviewer",
                    "note": params.get("note") or "",
                    "external_llm_opt_in": True,
                },
                account_environment=account_environment,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            return _result(
                context,
                status="blocked",
                code=str(
                    detail.get("error") or "research_pipeline_review_resume_blocked"
                ),
                summary="The Research Agent review owner rejected this resume request.",
                owner="easyicu.webserver.routes.agent",
                details={
                    key: detail.get(key)
                    for key in ("error", "reason", "job_id")
                    if detail.get(key) is not None
                },
            )
        context.invalidate_authority("research_pipeline_review_submitted")
        return _result(
            context,
            status="ok",
            code="research_pipeline_review_submitted",
            summary=f"Submitted the {decision} plan decision; EasyICU job {submitted.get('job_id')} owns the continuation.",
            owner="easyicu.webserver.routes.agent",
            details={
                key: submitted.get(key)
                for key in (
                    "job_id",
                    "kind",
                    "status",
                    "engine",
                    "review_run_id",
                    "study_context_id",
                    "study_context_revision",
                )
                if submitted.get(key) is not None
            },
        )
    job_id = str(
        params.get("job_id") or context.session.binding.active_job_id or ""
    ).strip()
    if job_id:
        job = jobs.MANAGER.get(job_id)
        if job:
            return _result(
                context,
                status="ok",
                code="easyicu_job_reattached",
                summary=f"Reattached to EasyICU job {job_id}; its authoritative status is {job.status}.",
                owner="easyicu.webserver.jobs",
                details={"job": project_job(job.snapshot())},
            )
    if params.get("run_id") or context.session.binding.run_id:
        return _result(
            context,
            status="blocked",
            code="scientific_resume_not_supported",
            summary=(
                "The Copilot conversation can resume, but the current EasyICU Web "
                "scientific pipeline has no public crash-resume owner contract. "
                "No replacement resume path was invented."
            ),
            owner="easyicu.research_agent.pipeline",
        )
    return _result(
        context,
        status="not_found",
        code="easyicu_resume_target_not_found",
        summary="No active EasyICU job or persisted run was bound to this Copilot session.",
        owner="easyicu.webserver.jobs",
    )


def _cancel(context: ToolExecutionContext, params: Mapping[str, Any]) -> Dict[str, Any]:
    _require_args(params, allowed=("job_id",))
    job_id = str(
        params.get("job_id") or context.session.binding.active_job_id or ""
    ).strip()
    job = jobs.MANAGER.get(job_id) if job_id else None
    if not job:
        return _result(
            context,
            status="not_found",
            code="easyicu_job_not_found",
            summary="The specifically bound EasyICU job no longer exists in JobManager.",
            owner="easyicu.webserver.jobs",
        )
    grant_block = _consume_action(context, "cancel")
    if grant_block is not None:
        return grant_block
    accepted = job.request_cancel("pi_copilot_user_authorized")
    result = _result(
        context,
        status="ok" if accepted else "blocked",
        code=(
            "easyicu_job_cancel_requested"
            if accepted
            else "easyicu_job_already_terminal"
        ),
        summary=(
            f"Cooperative cancellation was requested for EasyICU job {job_id}."
            if accepted
            else f"EasyICU job {job_id} was already terminal."
        ),
        owner="easyicu.webserver.jobs",
        details={"job": project_job(job.snapshot())},
    )
    if accepted:
        context.invalidate_authority("easyicu_job_cancel_requested")
    return result


def _request_replan(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("reason",), required=("reason",))
    study = _bound_context(context.session.binding)
    latest = _select_run(context)
    artifacts = {
        str(value)
        for value in ((latest or {}).get("artifact_names") or [])
        if value
    }
    pending_codes = {
        str(value)
        for value in ((latest or {}).get("pending_review_reason_codes") or [])
        if value
    }
    planned_digest = str(
        (latest or {}).get("scientific_configuration_sha256") or ""
    ).strip()
    current_digest = (
        study_contexts.scientific_configuration_sha256(study)
        if isinstance(study, Mapping) and study.get("id")
        else ""
    )
    same_study_plan = bool(
        isinstance(study, Mapping)
        and study.get("id")
        and latest
        and str(latest.get("study_id") or "") == str(study.get("id") or "")
        and "agent_plan.json" in artifacts
    )
    review_declared = bool(
        same_study_plan
        and str(latest.get("run_status") or "") == "human_review_pending"
        and bool(
            {"operator_plan_approval_required", "plan_scientific_changes_required"}
            & pending_codes
        )
    )
    live_review = (
        agent_pipeline_runs.pending_review((latest or {}).get("run_id"))
        if review_declared
        else None
    )
    current_review_is_resumable = bool(
        review_declared
        and isinstance(live_review, Mapping)
        and planned_digest
        and planned_digest == current_digest
        and live_review.get("plan_approval_allowed") is not False
        and (
            "operator_plan_approval_required" not in pending_codes
            or str(live_review.get("budget_mode") or "full_reviewed")
            == "full_reviewed"
        )
    )
    fresh_run_required = bool(same_study_plan and not current_review_is_resumable)
    preflight_only_history = bool(
        isinstance(study, Mapping)
        and study.get("id")
        and latest
        and str(latest.get("study_id") or "") == str(study.get("id") or "")
        and "agent_plan.json" not in artifacts
        and not str(study.get("active_job_id") or "").strip()
    )
    if fresh_run_required or preflight_only_history:
        # Start a new digest-bound pipeline run. Never mutate or re-use the old
        # plan. This also covers terminal failed/cancelled/blocked histories:
        # their persisted run_id is evidence, not a live execution coordinate.
        # A Web pipeline that failed before its review projection was committed
        # is deliberately absent from the registered run-history authority. In
        # that state the newest registered row remains the deterministic
        # preflight and there is no current reviewable Plan to reject or reuse.
        # Treat that as a fresh-plan request rather than inventing an in-place
        # mutation of an unregistered nested pipeline artifact.
        # `_run` consumes the fresh provider grant and invalidates this turn
        # after submission.
        source_run_id = ""
        if (
            review_declared
            and "plan_scientific_changes_required" in pending_codes
            and planned_digest
            and planned_digest == current_digest
        ):
            source_run_id = str((latest or {}).get("run_id") or "")
        return _run(
            context,
            {"run_type": "full"},
            plan_revision_source_run_id=source_run_id,
        )
    return _result(
        context,
        status="blocked",
        code="scientific_replan_not_supported",
        summary=(
            "EasyICU will not mutate a current scientific plan in place. "
            "Reject it or change the StudyContext before requesting a fresh run."
        ),
        owner="easyicu.research_agent.current_plan_authority",
        details={"reason_digest": path_digest(params.get("reason"))},
    )


def _workspace_access(
    context: ToolExecutionContext,
    *,
    require_write: bool = False,
) -> tuple[Optional[ProjectWorkspace], Optional[Dict[str, Any]]]:
    if context.session.agent_mode != "workspace":
        return None, _result(
            context,
            status="blocked",
            code="pi_workspace_mode_required",
            summary="Open a Pi workspace conversation before using project artifact tools.",
            owner="easyicu.webserver.pi_copilot.workspace",
        )
    if (
        context.workspace is None and context.workspace_root is None
    ) or not context.session.project_id:
        return None, _result(
            context,
            status="blocked",
            code="pi_workspace_unavailable",
            summary="The isolated Pi project workspace is unavailable for this session.",
            owner="easyicu.webserver.pi_copilot.workspace",
        )
    if require_write and not context.grant.has_capability("workspace_write"):
        return None, _result(
            context,
            status="blocked",
            code="pi_workspace_write_authorization_required",
            summary="Project file changes require the workspace-write grant for this message.",
            owner="easyicu.webserver.pi_copilot.workspace",
        )
    if context.workspace is not None:
        return context.workspace, None
    return ProjectWorkspace(context.workspace_root), None


def _workspace_resource(
    payload: Mapping[str, Any], *, kind: str = "file"
) -> Dict[str, Any]:
    resource = {
        "kind": kind,
        "file": str(payload.get("file") or ""),
        "label": Path(str(payload.get("file") or "artifact")).name,
        "media_type": str(payload.get("media_type") or "text/plain"),
        **dict(WORKSPACE_ARTIFACT_AUTHORITY),
    }
    checked_sha256 = str(payload.get("checked_sha256") or "").strip().lower()
    if kind == "webpage" and re.fullmatch(r"[0-9a-f]{64}", checked_sha256):
        resource["checked_sha256"] = checked_sha256
    return resource


def _load_skill(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("name",), required=("name",))
    name = str(params.get("name") or "").strip()
    if name == "web-prototype":
        _, blocked = _workspace_access(context)
        if blocked:
            return blocked
        skill_file = (
            Path(__file__).resolve().with_name("node_app")
            / "src"
            / "skills"
            / "web-prototype"
            / "SKILL.md"
        )
        instructions = skill_file.read_text(encoding="utf-8")[:12_000]
        return _workspace_result(
            context,
            status="ok",
            code="pi_workspace_skill_loaded",
            summary="Loaded the governed web-prototype workspace skill.",
            details={"skill": name, "instructions": instructions},
        )
    activation = next(
        (
            item
            for item in context.session.extension_activation.skills
            if item.name == name
            and "conversation" in item.stages
            and not item.disable_model_invocation
        ),
        None,
    )
    if activation is None:
        return _result(
            context,
            status="not_found",
            code="pi_extension_skill_not_active",
            summary="The requested Skill is not active for conversation in this frozen session.",
            owner="easyicu.extensions",
            details={
                "available_skills": [
                    item.name
                    for item in context.session.extension_activation.skills
                    if "conversation" in item.stages
                    and not item.disable_model_invocation
                ]
                + (["web-prototype"] if context.session.agent_mode == "workspace" else [])
            },
        )
    try:
        loaded = (context.extension_registry or ExtensionRegistry()).load_skill(
            name=activation.name,
            digest=activation.digest,
        )
    except ExtensionRegistryError as exc:
        return _result(
            context,
            status="blocked",
            code=exc.code,
            summary=exc.message,
            owner="easyicu.extensions",
            details=exc.details,
        )
    return _extension_result(
        context,
        status="ok",
        code="pi_extension_skill_loaded",
        summary=f"Loaded frozen Skill {activation.name} at sha256:{activation.digest[:12]}.",
        details={
            "skill": activation.name,
            "digest": activation.digest,
            "instructions": loaded["instructions"],
            "authority": "user_advisory_only",
        },
    )


def _list_extensions(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    activation = context.session.extension_activation
    return _extension_result(
        context,
        status="ok",
        code="pi_extension_activation_listed",
        summary=(
            f"This session froze {len(activation.skills)} user Skills and "
            f"{len(activation.mcp_servers)} MCP servers."
        ),
        details={
            "activation_sha256": activation.activation_sha256,
            "revision": activation.revision,
            "skills": [
                {
                    "name": item.name,
                    "description": item.description,
                    "digest": item.digest,
                    "stages": list(item.stages),
                    "model_invocation_allowed": not item.disable_model_invocation,
                }
                for item in activation.skills
            ],
            "mcp_servers": [
                {
                    "name": item.name,
                    "transport": item.transport,
                    "allowed_tools": list(item.allowed_tools),
                }
                for item in activation.mcp_servers
            ],
            "scope": "session_frozen_user_advisory",
        },
    )


def _call_mcp_extension_tool(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("server", "tool", "arguments"),
        required=("server", "tool"),
    )
    if not bool(settings.load_settings().get("mcp_tools_enabled", False)):
        return _result(
            context,
            status="blocked",
            code="extension_mcp_master_disabled",
            summary="Enable the MCP tools layer in Settings before calling an external MCP tool.",
            owner="easyicu.webserver.settings",
        )
    server_name = str(params.get("server") or "").strip()
    server = next(
        (
            item
            for item in context.session.extension_activation.mcp_servers
            if item.name == server_name
        ),
        None,
    )
    if server is None:
        return _result(
            context,
            status="not_found",
            code="pi_extension_mcp_server_not_active",
            summary="The requested MCP server is not active in this frozen session.",
            owner="easyicu.extensions",
        )
    grant_block = _consume_action(context, "mcp_read")
    if grant_block is not None:
        return grant_block
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, Mapping):
        return _result(
            context,
            status="blocked",
            code="extension_mcp_arguments_invalid",
            summary="MCP arguments must be a bounded JSON object.",
            owner="easyicu.extensions",
        )
    try:
        external = call_mcp_tool(
            server,
            str(params.get("tool") or ""),
            arguments,
        )
    except ExtensionRegistryError as exc:
        return _result(
            context,
            status="blocked",
            code=exc.code,
            summary=exc.message,
            owner="easyicu.extensions",
            details=exc.details,
        )
    return _extension_result(
        context,
        status="ok" if external.get("ok") else "error",
        code=(
            "pi_extension_mcp_tool_completed"
            if external.get("ok")
            else "pi_extension_mcp_tool_error"
        ),
        summary=(
            f"Called allowlisted MCP tool {server.name}.{params.get('tool')}; "
            "the result remains untrusted external metadata."
        ),
        details={
            **external,
            "claim_ceiling": "external_metadata_not_study_evidence",
            "activation_sha256": context.session.extension_activation.activation_sha256,
        },
    )


def _list_project_files(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=())
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    rows = workspace.list_files(context.session.project_id)
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_files_listed",
        summary=f"Listed {len(rows)} project workspace files.",
        details={"files": rows, "count": len(rows)},
    )


def _read_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("file", "start_line", "end_line"),
        required=("file",),
    )
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.read_file(
        context.session.project_id,
        params["file"],
        start_line=int(params.get("start_line") or 1),
        end_line=(int(params["end_line"]) if params.get("end_line") else None),
    )
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_read",
        summary=f"Read {payload['file']}.",
        details={**payload, "resource": _workspace_resource(payload)},
    )


def _write_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("file", "content"),
        required=("file", "content"),
    )
    workspace, blocked = _workspace_access(context, require_write=True)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    try:
        mutation = context.grant.reserve_workspace_mutation("write")
    except WorkspaceMutationLimitError:
        return _workspace_result(
            context,
            status="blocked",
            code="pi_workspace_mutation_limit_reached",
            summary="This message reached its bounded project-file mutation limit.",
        )
    payload = workspace.write_file(
        context.session.project_id,
        params["file"],
        params["content"],
    )
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_written",
        summary=f"Created {payload['file']}.",
        details={
            **payload,
            "mutation_receipt": mutation.model_dump(mode="json"),
            "resource": _workspace_resource(payload),
        },
    )


def _edit_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("file", "old_text", "new_text", "expected_sha256"),
        required=("file", "old_text", "expected_sha256"),
    )
    workspace, blocked = _workspace_access(context, require_write=True)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    try:
        mutation = context.grant.reserve_workspace_mutation("edit")
    except WorkspaceMutationLimitError:
        return _workspace_result(
            context,
            status="blocked",
            code="pi_workspace_mutation_limit_reached",
            summary="This message reached its bounded project-file mutation limit.",
        )
    payload = workspace.edit_file(
        context.session.project_id,
        params["file"],
        old_text=params["old_text"],
        new_text=params.get("new_text") or "",
        expected_sha256=params["expected_sha256"],
    )
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_edited",
        summary=f"Edited {payload['file']} with one exact replacement.",
        details={
            **payload,
            "mutation_receipt": mutation.model_dump(mode="json"),
            "resource": _workspace_resource(payload),
        },
    )


def _check_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(params, allowed=("file",), required=("file",))
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.check_file(context.session.project_id, params["file"])
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_file_checked",
        summary=f"Checked {payload['file']} with {payload['checker']}.",
        details={**payload, "resource": _workspace_resource(payload)},
    )


def _preview_project_file(
    context: ToolExecutionContext, params: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_args(
        params,
        allowed=("file", "checked_sha256"),
        required=("file", "checked_sha256"),
    )
    workspace, blocked = _workspace_access(context)
    if blocked:
        return blocked
    assert workspace is not None and context.session.project_id
    payload = workspace.preview_file(
        context.session.project_id,
        params["file"],
        checked_sha256=params["checked_sha256"],
    )
    resource = _workspace_resource(payload, kind="webpage")
    return _workspace_result(
        context,
        status="ok",
        code="pi_workspace_preview_ready",
        summary=f"Prepared the live preview for {payload['file']}.",
        details={
            "file": payload["file"],
            "media_type": payload["media_type"],
            "resource": resource,
        },
    )


_DISPATCH = {
    "easyicu_workspace_status": _workspace_status,
    "easyicu_list_data_sources": _list_data_sources,
    "easyicu_list_source_concepts": _list_source_concepts,
    "easyicu_inspect_data_package": _inspect_data_package,
    "easyicu_review_cohort": _review_cohort,
    "easyicu_open_data_download": _open_data_download,
    "easyicu_preview_icd_cohort": _preview_icd_cohort,
    "easyicu_review_patient_timeline": _review_patient_timeline,
    "easyicu_compare_data_sources": _compare_data_sources,
    "easyicu_inspect_workflow": _inspect_workflow,
    "easyicu_inspect_context": _inspect_context,
    "easyicu_inspect_plan": _inspect_plan,
    "easyicu_inspect_literature": _inspect_literature,
    "easyicu_inspect_capability": _inspect_capability,
    "easyicu_inspect_run": _inspect_run,
    "easyicu_inspect_step": _inspect_step,
    "easyicu_inspect_validation": _inspect_validation,
    "easyicu_list_artifacts": _list_artifacts,
    "easyicu_inspect_evidence": _inspect_evidence,
    "easyicu_explain_blocker": _explain_blocker,
    "easyicu_inspect_interpretation": _inspect_interpretation,
    "easyicu_inspect_manuscript": _inspect_manuscript,
    "easyicu_update_study_context": _update_study_context,
    "easyicu_mine_ideas": _mine_ideas,
    "easyicu_search_literature": _search_literature,
    "easyicu_prepare_idea_handoff": _prepare_idea_handoff,
    "easyicu_accept_idea_handoff": _accept_idea_handoff,
    "easyicu_prepare_demo_source": _prepare_demo_source,
    "easyicu_start_extraction": _start_extraction,
    "easyicu_run": _run,
    "easyicu_resume": _resume,
    "easyicu_cancel": _cancel,
    "easyicu_request_replan": _request_replan,
    "easyicu_list_extensions": _list_extensions,
    "easyicu_load_skill": _load_skill,
    "easyicu_call_mcp_tool": _call_mcp_extension_tool,
    "easyicu_list_project_files": _list_project_files,
    "easyicu_read_project_file": _read_project_file,
    "easyicu_write_project_file": _write_project_file,
    "easyicu_edit_project_file": _edit_project_file,
    "easyicu_check_project_file": _check_project_file,
    "easyicu_preview_project_file": _preview_project_file,
}


def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    """Execute exactly one registered tool through its existing EasyICU owner."""

    context.assert_authority_fresh()
    tool_name = str(name or "").strip()
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        raise PiCopilotError(
            "pi_tool_unknown",
            "The Pi sidecar requested an unregistered EasyICU tool.",
            details={"tool": tool_name},
        )
    if not isinstance(arguments, Mapping):
        raise PiCopilotError(
            "pi_tool_arguments_invalid",
            "EasyICU tool arguments must be an object.",
        )
    authorization = context.session.data_source_authorization
    if (
        tool_name in DATA_SOURCE_REQUIRED_TOOLS
        and authorization.status in {"pending", "selection_in_progress"}
    ):
        return _result(
            context,
            status="blocked",
            code="pi_session_data_source_confirmation_required",
            summary=(
                "Choose and confirm a local or official-demo data source before "
                "EasyICU reads data or starts analysis. Research planning can continue."
            ),
            owner="easyicu.webserver.pi_copilot.data_source_authority",
            details={
                "status": authorization.status,
                "reason": authorization.reason,
                "conversation_available": True,
            },
        )
    return handler(context, arguments)


__all__ = [
    "ALLOWED_TOOLS",
    "CONTROL_TOOLS",
    "DATA_SOURCE_REQUIRED_TOOLS",
    "READ_TOOLS",
    "WORKSPACE_TOOLS",
    "execute_tool",
]
