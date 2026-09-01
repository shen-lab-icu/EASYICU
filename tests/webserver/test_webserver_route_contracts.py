"""Registration contracts for native FastAPI route modules."""

from __future__ import annotations

from pathlib import Path

from fastapi.routing import APIRoute
from starlette.routing import Mount

from easyicu.webserver import patient_drilldown
from easyicu.webserver.app import app
from easyicu.webserver.patient_drilldown import eligibility as patient_eligibility
from easyicu.webserver.routes.agent import artifact_router as agent_artifact_router
from easyicu.webserver.routes.agent import control_router as agent_control_router
from easyicu.webserver.routes.copilot import router as copilot_router
from easyicu.webserver.routes.demo_sources import (
    catalog_router as demo_source_catalog_router,
)
from easyicu.webserver.routes.demo_sources import (
    submission_router as demo_source_submission_router,
)
from easyicu.webserver.routes.extraction import router as extraction_router
from easyicu.webserver.routes.guided import router as guided_router
from easyicu.webserver.routes.ideas import router as ideas_router
from easyicu.webserver.routes.jobs import lifecycle_router as job_lifecycle_router
from easyicu.webserver.routes.jobs import submission_router as job_submission_router
from easyicu.webserver.routes.local_data import router as local_data_router
from easyicu.webserver.routes.page_guide import router as page_guide_router
from easyicu.webserver.routes.pi_copilot import router as pi_copilot_router
from easyicu.webserver.routes.reviews import router as reviews_router
from easyicu.webserver.routes.study_contexts import router as study_contexts_router
from easyicu.webserver.routes.system import router as system_router
from easyicu.webserver.routes.workspaces import router as workspaces_router


EXPECTED_SYSTEM_ROUTES = [
    ("GET", "/api/health"),
    ("GET", "/favicon.ico"),
    ("GET", "/api/catalog"),
    ("GET", "/api/catalog/lineage/{concept_id}"),
    ("GET", "/api/settings"),
    ("POST", "/api/settings"),
    ("POST", "/api/settings/reset"),
    ("GET", "/api/capabilities"),
    ("POST", "/api/capabilities/tool-check"),
    ("POST", "/api/capabilities/zotero/search"),
    ("POST", "/api/capabilities/zotero/test"),
    ("POST", "/api/capabilities/zotero/source"),
    ("POST", "/api/capabilities/zotero/import"),
    ("POST", "/api/capabilities/audit-events"),
]

EXPECTED_SYSTEM_OPERATION_NAMES = [
    "health",
    "favicon",
    "catalog",
    "catalog_lineage",
    "get_settings",
    "post_settings",
    "post_settings_reset",
    "get_capabilities",
    "post_capability_tool_check",
    "post_capability_zotero_search",
    "post_capability_zotero_test",
    "post_capability_zotero_source",
    "post_capability_zotero_import",
    "post_capability_audit_events",
]

EXPECTED_GUIDED_ROUTES = [
    ("POST", "/api/guided/drafts", "post_guided_draft"),
    ("POST", "/api/guided/drafts/list", "post_guided_drafts_list"),
    ("POST", "/api/guided/drafts/remove", "post_guided_draft_remove"),
    ("DELETE", "/api/guided/drafts/remove", "delete_guided_draft_remove"),
    ("POST", "/api/guided/session", "post_guided_session"),
    ("POST", "/api/guided/project/open", "post_guided_project_open"),
    ("POST", "/api/guided/message", "post_guided_message"),
    ("POST", "/api/guided/action", "post_guided_action"),
    ("POST", "/api/guided/sessions/list", "post_guided_sessions_list"),
]

EXPECTED_COPILOT_ROUTES = [
    ("POST", "/api/copilot/study-intent", "post_copilot_study_intent"),
    ("POST", "/api/copilot/sessions", "post_copilot_session"),
    ("POST", "/api/copilot/message", "post_copilot_message"),
    ("POST", "/api/copilot/action", "post_copilot_action"),
    ("POST", "/api/copilot/sessions/list", "post_copilot_sessions_list"),
]

EXPECTED_PI_COPILOT_ROUTES = [
    ("GET", "/api/copilot/pi/status", "get_pi_copilot_status"),
    (
        "GET",
        "/api/copilot/pi/literature/sources/{pmid}",
        "get_pi_copilot_literature_source",
    ),
    (
        "GET",
        "/api/copilot/pi/resource-status",
        "get_pi_copilot_resource_status",
    ),
    (
        "POST",
        "/api/copilot/pi/session-maintenance",
        "post_pi_copilot_session_maintenance",
    ),
    (
        "POST",
        "/api/copilot/pi/provider-config",
        "post_pi_copilot_provider_config",
    ),
    ("POST", "/api/copilot/pi/sessions", "post_pi_copilot_session"),
    (
        "GET",
        "/api/copilot/pi/research-provider/codex/status",
        "get_pi_copilot_codex_status",
    ),
    (
        "POST",
        "/api/copilot/pi/research-provider/codex/login",
        "post_pi_copilot_codex_login",
    ),
    (
        "POST",
        "/api/copilot/pi/research-provider/codex/cancel",
        "post_pi_copilot_codex_cancel",
    ),
    (
        "POST",
        "/api/copilot/pi/research-provider/codex/logout",
        "post_pi_copilot_codex_logout",
    ),
    (
        "GET",
        "/api/copilot/pi/research-provider/codex/models",
        "get_pi_copilot_codex_models",
    ),
    (
        "POST",
        "/api/copilot/pi/projects/initialize",
        "post_pi_copilot_project_initialize",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/workflow",
        "get_pi_copilot_project_workflow",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/workspace/file",
        "get_pi_copilot_workspace_file",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/workspace/preview",
        "get_pi_copilot_workspace_preview",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/runs/{run_id}/artifacts/{artifact_name}",
        "get_pi_copilot_research_artifact",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/runs/{run_id}/evidence/{evidence_id}",
        "get_pi_copilot_research_evidence_preview",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/data-package-review",
        "get_pi_copilot_data_package_review",
    ),
    (
        "POST",
        "/api/copilot/pi/projects/{project_id}/data-package-review/prepare",
        "post_pi_copilot_data_package_review_prepare",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/data-workbench-snapshot",
        "get_pi_copilot_data_workbench_snapshot",
    ),
    (
        "POST",
        "/api/copilot/pi/projects/{project_id}/data-workbench-snapshot/prepare",
        "post_pi_copilot_data_workbench_snapshot_prepare",
    ),
    (
        "GET",
        "/api/copilot/pi/projects/{project_id}/runs/{run_id}/documents/{document_name}",
        "get_pi_copilot_research_document",
    ),
    ("GET", "/api/copilot/pi/sessions", "get_pi_copilot_sessions"),
    ("GET", "/api/copilot/pi/sessions/{session_id}", "get_pi_copilot_session"),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/message",
        "post_pi_copilot_message",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/cohort-eligibility-selection",
        "post_pi_copilot_cohort_eligibility_selection",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/plan-decision-selection",
        "post_pi_copilot_plan_decision_selection",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/regenerate",
        "post_pi_copilot_regenerate",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/data-source-authorization",
        "post_pi_copilot_data_source_authorization",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/rebind",
        "post_pi_copilot_rebind",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/presentation",
        "post_pi_copilot_presentation_pin",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/child-jobs/{job_id}/archive",
        "post_pi_copilot_child_job_archive",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/host-actions",
        "post_pi_copilot_host_action",
    ),
    (
        "POST",
        "/api/copilot/pi/sessions/{session_id}/abort",
        "post_pi_copilot_abort",
    ),
]

EXPECTED_PAGE_GUIDE_ROUTES = [
    ("POST", "/api/page-guide/sessions", "post_page_guide_session"),
    ("POST", "/api/page-guide/message", "post_page_guide_message"),
    ("POST", "/api/page-guide/action", "post_page_guide_action"),
    ("POST", "/api/page-guide/sessions/list", "post_page_guide_sessions_list"),
]

EXPECTED_IDEAS_ROUTES = [
    ("POST", "/api/ideas/mine", "post_ideas_mine"),
    ("POST", "/api/ideas/resolve-source", "post_ideas_resolve_source"),
    ("POST", "/api/ideas/discover", "post_ideas_discover"),
    ("POST", "/api/ideas/ingest-pdf", "post_ideas_ingest_pdf"),
    ("POST", "/api/ideas/literature-folder", "post_ideas_literature_folder"),
    ("POST", "/api/ideas/prior-art", "post_ideas_prior_art"),
    ("POST", "/api/ideas/plan", "post_ideas_plan"),
    (
        "POST",
        "/api/ideas/bounded-feasibility",
        "post_ideas_bounded_feasibility",
    ),
    ("POST", "/api/ideas/handoff", "post_ideas_handoff"),
    (
        "POST",
        "/api/ideas/create-agent-project",
        "post_ideas_create_agent_project",
    ),
    ("POST", "/api/ideas/agent-projects", "post_ideas_agent_projects"),
    ("POST", "/api/ideas/history", "post_ideas_history"),
    ("POST", "/api/ideas/run", "post_ideas_run"),
]

EXPECTED_LOCAL_DATA_ROUTES = [
    ("GET", "/api/fs/list", "fs_list"),
    ("POST", "/api/fs/mkdir", "fs_mkdir"),
    ("POST", "/api/data/scan", "data_scan"),
    ("POST", "/api/workspace/summary", "workspace_summary"),
]

EXPECTED_WORKSPACE_ROUTES = [
    (
        "POST",
        "/api/workspaces/crossdb-summary",
        "workspaces_crossdb_summary",
    ),
    ("GET", "/api/workspaces/registry", "workspaces_registry"),
    ("POST", "/api/workspaces/registry", "post_workspaces_registry"),
    ("POST", "/api/workspaces/register", "post_workspaces_register"),
    ("POST", "/api/workspaces/rename", "post_workspaces_rename"),
    ("POST", "/api/workspaces/remove", "post_workspaces_remove"),
    ("POST", "/api/workspaces/download", "post_workspaces_download"),
]

EXPECTED_STUDY_CONTEXT_ROUTES = [
    ("GET", "/api/study-contexts/active", "get_active_study_context"),
    ("GET", "/api/study-contexts", "get_study_contexts"),
    ("POST", "/api/study-contexts", "post_study_context"),
    ("POST", "/api/study-contexts/handoff", "post_study_context_handoff"),
    ("GET", "/api/study-contexts/{context_id}", "get_study_context"),
]

EXPECTED_REVIEW_ROUTES = [
    (
        "POST",
        "/api/patient-review/drilldown",
        "patient_review_drilldown",
    ),
    ("POST", "/api/patient-review/entities", "patient_review_entities"),
    ("POST", "/api/patient-review/entity", "patient_review_entity"),
    (
        "POST",
        "/api/patient-review/table-preview",
        "patient_review_table_preview",
    ),
    ("POST", "/api/patient-review/feature", "patient_review_feature"),
    ("POST", "/api/patient-review/sources", "patient_review_sources"),
    ("POST", "/api/cohort-review/summary", "cohort_review_summary"),
    ("POST", "/api/crossdb-review/summary", "crossdb_review_summary"),
    (
        "POST",
        "/api/crossdb-review/raw-distribution",
        "crossdb_raw_distribution",
    ),
    (
        "POST",
        "/api/crossdb-review/raw-root-scan",
        "crossdb_raw_root_scan",
    ),
    (
        "POST",
        "/api/crossdb-review/demo-distribution",
        "crossdb_demo_distribution",
    ),
]

EXPECTED_EXTRACTION_ROUTES = [
    (
        "POST",
        "/api/extraction/filter-options",
        "extraction_filter_options",
    ),
    (
        "POST",
        "/api/extraction/filter-preview",
        "extraction_filter_preview",
    ),
]

EXPECTED_DEMO_SOURCE_CATALOG_ROUTES = [
    ("GET", "/api/demo-sources", "get_demo_sources"),
]

EXPECTED_DEMO_SOURCE_SUBMISSION_ROUTES = [
    (
        "POST",
        "/api/jobs/demo-source-prepare",
        "jobs_demo_source_prepare",
    ),
]

EXPECTED_JOB_SUBMISSION_ROUTES = [
    ("POST", "/api/jobs/convert", "jobs_convert"),
    ("POST", "/api/jobs/extract", "jobs_extract"),
    (
        "POST",
        "/api/jobs/crossdb-summary",
        "jobs_crossdb_summary",
    ),
    (
        "POST",
        "/api/jobs/crossdb-raw-distribution",
        "jobs_crossdb_raw_distribution",
    ),
]

EXPECTED_JOB_LIFECYCLE_ROUTES = [
    ("GET", "/api/jobs/{job_id}", "jobs_get"),
    ("POST", "/api/jobs/{job_id}/cancel", "jobs_cancel"),
    ("POST", "/api/jobs/{job_id}/open-output", "jobs_open_output"),
    ("GET", "/api/jobs/{job_id}/events", "jobs_events"),
]

EXPECTED_AGENT_CONTROL_ROUTES = [
    ("POST", "/api/jobs/agent-run", "jobs_agent_run"),
    ("POST", "/api/jobs/agent-run-review", "jobs_agent_run_review"),
    (
        "GET",
        "/api/agent-runs/codex-auth/status",
        "get_agent_run_codex_auth_status",
    ),
    (
        "POST",
        "/api/agent-runs/codex-auth/login",
        "post_agent_run_codex_auth_login",
    ),
    (
        "POST",
        "/api/agent-runs/codex-auth/cancel",
        "post_agent_run_codex_auth_cancel",
    ),
    (
        "POST",
        "/api/agent-runs/codex-auth/logout",
        "post_agent_run_codex_auth_logout",
    ),
    (
        "GET",
        "/api/agent-runs/provider-status",
        "get_agent_run_provider_status",
    ),
    (
        "POST",
        "/api/agent-runs/provider-config",
        "post_agent_run_provider_config",
    ),
    ("POST", "/api/agent-runs/review", "post_agent_run_review"),
    (
        "POST",
        "/api/agent-runs/science-workbench",
        "post_agent_run_science_workbench",
    ),
    ("POST", "/api/agent-runs/signoff", "post_agent_run_signoff"),
    ("POST", "/api/agent-runs/history", "post_agent_run_history"),
]

EXPECTED_AGENT_ARTIFACT_ROUTES = [
    ("POST", "/api/agent-runs/artifact", "post_agent_run_artifact"),
    (
        "POST",
        "/api/agent-runs/download-artifact",
        "post_agent_run_download_artifact",
    ),
    (
        "POST",
        "/api/agent-runs/download-bundle",
        "post_agent_run_download_bundle",
    ),
]


def _router_routes(router) -> list[APIRoute]:
    return [route for route in router.routes if isinstance(route, APIRoute)]


def _router_registration_indices(router) -> list[int]:
    """Support both eager and lazy FastAPI ``include_router`` versions."""
    endpoints = {route.endpoint for route in _router_routes(router)}
    indices = [
        index
        for index, route in enumerate(app.routes)
        if getattr(route, "original_router", None) is router
        or getattr(route, "endpoint", None) in endpoints
    ]
    if not indices:
        raise AssertionError("router is not registered on the FastAPI app")
    return indices


def _router_registration_index(router) -> int:
    return _router_registration_indices(router)[0]


def _assert_router_contract(router, expected: list[tuple]) -> None:
    routes = _router_routes(router)
    endpoints = {route.endpoint for route in routes}
    actual = [
        (method, route.path, route.name)
        for route in routes
        for method in sorted(route.methods or set())
    ]
    eager_app_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.endpoint in endpoints
    ]
    eager_app_actual = [
        (method, route.path, route.name)
        for route in eager_app_routes
        for method in sorted(route.methods or set())
    ]
    registration_indices = _router_registration_indices(router)

    assert actual == expected
    if eager_app_routes:
        assert eager_app_actual == expected
    else:
        assert len(registration_indices) == 1
        assert (
            getattr(app.routes[registration_indices[0]], "original_router", None)
            is router
        )
    assert registration_indices == list(
        range(
            registration_indices[0],
            registration_indices[0] + len(registration_indices),
        )
    )


def test_system_route_method_path_snapshot() -> None:
    routes = _router_routes(system_router)
    actual = [
        (method, route.path)
        for route in routes
        for method in sorted(route.methods or set())
    ]

    assert actual == EXPECTED_SYSTEM_ROUTES


def test_system_route_operation_names_and_favicon_schema_contract() -> None:
    routes = _router_routes(system_router)

    assert [route.name for route in routes] == EXPECTED_SYSTEM_OPERATION_NAMES
    favicon_route = next(route for route in routes if route.path == "/favicon.ico")
    assert favicon_route.include_in_schema is False


def test_guided_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(guided_router, EXPECTED_GUIDED_ROUTES)


def test_copilot_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(copilot_router, EXPECTED_COPILOT_ROUTES)


def test_pi_copilot_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(pi_copilot_router, EXPECTED_PI_COPILOT_ROUTES)


def test_page_guide_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(
        page_guide_router,
        EXPECTED_PAGE_GUIDE_ROUTES,
    )


def test_ideas_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(ideas_router, EXPECTED_IDEAS_ROUTES)


def test_local_data_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(local_data_router, EXPECTED_LOCAL_DATA_ROUTES)


def test_workspace_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(workspaces_router, EXPECTED_WORKSPACE_ROUTES)


def test_study_context_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(study_contexts_router, EXPECTED_STUDY_CONTEXT_ROUTES)


def test_review_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(reviews_router, EXPECTED_REVIEW_ROUTES)


def test_extraction_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(extraction_router, EXPECTED_EXTRACTION_ROUTES)


def test_demo_source_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(
        demo_source_catalog_router, EXPECTED_DEMO_SOURCE_CATALOG_ROUTES
    )
    _assert_router_contract(
        demo_source_submission_router, EXPECTED_DEMO_SOURCE_SUBMISSION_ROUTES
    )


def test_job_submission_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(job_submission_router, EXPECTED_JOB_SUBMISSION_ROUTES)


def test_job_lifecycle_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(job_lifecycle_router, EXPECTED_JOB_LIFECYCLE_ROUTES)


def test_agent_control_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(agent_control_router, EXPECTED_AGENT_CONTROL_ROUTES)


def test_agent_artifact_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(agent_artifact_router, EXPECTED_AGENT_ARTIFACT_ROUTES)


def test_route_owner_boundaries() -> None:
    package_root = Path(__file__).parents[2] / "src" / "easyicu" / "webserver"
    app_source = (package_root / "app.py").read_text(encoding="utf-8")
    guided_source = (package_root / "routes" / "guided.py").read_text(encoding="utf-8")
    copilot_source = (package_root / "routes" / "copilot.py").read_text(
        encoding="utf-8"
    )
    page_guide_source = (package_root / "routes" / "page_guide.py").read_text(
        encoding="utf-8"
    )
    ideas_source = (package_root / "routes" / "ideas.py").read_text(encoding="utf-8")
    local_data_source = (package_root / "routes" / "local_data.py").read_text(
        encoding="utf-8"
    )
    workspaces_source = (package_root / "routes" / "workspaces.py").read_text(
        encoding="utf-8"
    )
    reviews_source = (package_root / "routes" / "reviews.py").read_text(
        encoding="utf-8"
    )
    extraction_source = (package_root / "routes" / "extraction.py").read_text(
        encoding="utf-8"
    )
    demo_source = (package_root / "routes" / "demo_sources.py").read_text(
        encoding="utf-8"
    )
    jobs_source = (package_root / "routes" / "jobs.py").read_text(encoding="utf-8")
    agent_source = (package_root / "routes" / "agent.py").read_text(encoding="utf-8")
    study_context_source = (package_root / "routes" / "study_contexts.py").read_text(
        encoding="utf-8"
    )

    assert "/api/guided/" not in app_source
    assert "/api/copilot/" not in app_source
    assert "/api/page-guide/" not in app_source
    assert "/api/ideas/" not in app_source
    assert "/api/fs/" not in app_source
    assert "/api/data/" not in app_source
    assert "/api/workspace/" not in app_source
    assert "/api/workspaces/" not in app_source
    assert "/api/study-contexts" not in app_source
    assert "/api/patient-review/" not in app_source
    assert "/api/cohort-review/" not in app_source
    assert "/api/crossdb-review/" not in app_source
    assert "/api/extraction/" not in app_source
    assert "/api/demo-sources" not in app_source
    assert '"/api/jobs/demo-source-prepare"' not in app_source
    assert '"/api/jobs/convert"' not in app_source
    assert '"/api/jobs/extract"' not in app_source
    assert '"/api/jobs/crossdb-raw-distribution"' not in app_source
    assert '"/api/jobs/{job_id}' not in app_source
    assert '"/api/jobs/agent-run"' not in app_source
    assert "/api/agent-runs/" not in app_source
    assert "/api/guided/" in guided_source
    assert "/api/copilot/" not in guided_source
    assert "/api/page-guide/" not in guided_source
    assert "/api/copilot/" in copilot_source
    assert "/api/guided/" not in copilot_source
    assert "/api/page-guide/" not in copilot_source
    assert "/api/page-guide/" in page_guide_source
    assert "/api/guided/" not in page_guide_source
    assert "/api/copilot/" not in page_guide_source
    assert "/api/ideas/" in ideas_source
    assert "/api/guided/" not in ideas_source
    assert "/api/copilot/" not in ideas_source
    assert "/api/page-guide/" not in ideas_source
    assert "/api/fs/" in local_data_source
    assert "/api/data/" in local_data_source
    assert "/api/workspace/" in local_data_source
    assert "/api/workspaces/" not in local_data_source
    assert "/api/workspaces/" in workspaces_source
    assert "/api/workspace/" not in workspaces_source
    assert "/api/study-contexts" in study_context_source
    assert "/api/jobs/" not in study_context_source
    assert "/api/jobs/" not in local_data_source
    assert "/api/jobs/" not in workspaces_source
    assert "-review/" not in local_data_source
    assert "-review/" not in workspaces_source
    assert "/api/patient-review/" in reviews_source
    assert "/api/cohort-review/" in reviews_source
    assert "/api/crossdb-review/" in reviews_source
    assert "/api/extraction/" not in reviews_source
    assert "/api/workspaces/" not in reviews_source
    assert "/api/jobs/" not in reviews_source
    assert "/api/extraction/" in extraction_source
    assert "-review/" not in extraction_source
    assert "/api/workspaces/" not in extraction_source
    assert "/api/jobs/" not in extraction_source
    assert "/api/demo-sources" in demo_source
    assert '"/api/jobs/demo-source-prepare"' in demo_source
    assert '"/api/jobs/convert"' not in demo_source
    assert '"/api/jobs/{job_id}' not in demo_source
    assert '"/api/jobs/convert"' in jobs_source
    assert '"/api/jobs/extract"' in jobs_source
    assert '"/api/jobs/crossdb-summary"' in jobs_source
    assert '"/api/jobs/crossdb-raw-distribution"' in jobs_source
    assert '"/api/jobs/{job_id}' in jobs_source
    assert '"/api/jobs/agent-run"' not in jobs_source
    assert "/api/agent-runs/" not in jobs_source
    assert "job_store.MANAGER" in jobs_source
    assert '"/api/jobs/agent-run"' in agent_source
    assert "/api/agent-runs/" in agent_source
    assert '"/api/jobs/convert"' not in agent_source
    assert '"/api/jobs/extract"' not in agent_source
    assert '"/api/jobs/{job_id}' not in agent_source
    assert "/api/guided/" not in agent_source
    assert "/api/copilot/" not in agent_source
    assert "/api/ideas/" not in agent_source
    assert "_pubmed_connector_gate" not in app_source
    assert "_pubmed_connector_gate" in ideas_source
    assert "easyicu.webserver.app" not in guided_source
    assert "easyicu.webserver.app" not in copilot_source
    assert "easyicu.webserver.app" not in page_guide_source
    assert "easyicu.webserver.app" not in ideas_source
    assert "easyicu.webserver.app" not in local_data_source
    assert "easyicu.webserver.app" not in workspaces_source
    assert "easyicu.webserver.app" not in reviews_source
    assert "easyicu.webserver.app" not in extraction_source
    assert "easyicu.webserver.app" not in demo_source
    assert "easyicu.webserver.app" not in jobs_source
    assert "easyicu.webserver.app" not in agent_source
    assert "easyicu.webserver.app" not in study_context_source


def test_patient_review_eligibility_owner_boundary() -> None:
    package_root = Path(__file__).parents[2] / "src" / "easyicu" / "webserver"
    patient_package = package_root / "patient_drilldown"
    facade_source = (patient_package / "__init__.py").read_text(encoding="utf-8")
    eligibility_source = (patient_package / "eligibility.py").read_text(
        encoding="utf-8"
    )
    owner_names = (
        "_eligibility_flow_payload",
        "_first_int",
        "_demographic_flow_label",
        "_demographic_flow_note",
        "_target_clinical_flow_preset",
        "_target_clinical_flow_label",
        "_target_clinical_flow_note",
        "_int_or_none",
    )

    assert not (package_root / "patient_drilldown.py").exists()
    for name in owner_names:
        assert getattr(patient_drilldown, name) is getattr(patient_eligibility, name)
        assert f"def {name}(" not in facade_source
        assert f"def {name}(" in eligibility_source
    assert "easyicu.webserver.patient_drilldown" not in eligibility_source


def test_root_static_mount_stays_last() -> None:
    static_mount = app.routes[-1]
    assert (
        _router_registration_index(system_router)
        < _router_registration_index(local_data_router)
        < _router_registration_index(reviews_router)
        < _router_registration_index(extraction_router)
        < _router_registration_index(workspaces_router)
        < _router_registration_index(study_contexts_router)
        < _router_registration_index(demo_source_catalog_router)
        < _router_registration_index(job_submission_router)
        < _router_registration_index(demo_source_submission_router)
        < _router_registration_index(agent_control_router)
        < _router_registration_index(guided_router)
        < _router_registration_index(copilot_router)
        < _router_registration_index(pi_copilot_router)
        < _router_registration_index(page_guide_router)
        < _router_registration_index(ideas_router)
        < _router_registration_index(agent_artifact_router)
        < _router_registration_index(job_lifecycle_router)
        < len(app.routes) - 1
    )
    assert isinstance(static_mount, Mount)
    assert static_mount.name == "static"
    assert static_mount.path == ""
