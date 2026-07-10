"""Registration contracts for native FastAPI route modules."""

from __future__ import annotations

from pathlib import Path

from fastapi.routing import APIRoute
from starlette.routing import Mount

from easyicu.webserver.app import app
from easyicu.webserver.routes.copilot import router as copilot_router
from easyicu.webserver.routes.guided import router as guided_router
from easyicu.webserver.routes.page_guide import router as page_guide_router
from easyicu.webserver.routes.system import router as system_router


EXPECTED_SYSTEM_ROUTES = [
    ("GET", "/api/health"),
    ("GET", "/favicon.ico"),
    ("GET", "/api/catalog"),
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
    ("POST", "/api/copilot/sessions", "post_copilot_session"),
    ("POST", "/api/copilot/message", "post_copilot_message"),
    ("POST", "/api/copilot/action", "post_copilot_action"),
    ("POST", "/api/copilot/sessions/list", "post_copilot_sessions_list"),
]

EXPECTED_PAGE_GUIDE_ROUTES = [
    ("POST", "/api/page-guide/sessions", "post_page_guide_session"),
    ("POST", "/api/page-guide/message", "post_page_guide_message"),
    ("POST", "/api/page-guide/action", "post_page_guide_action"),
    ("POST", "/api/page-guide/sessions/list", "post_page_guide_sessions_list"),
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


def _assert_router_contract(router, prefix: str, expected: list[tuple]) -> None:
    routes = _router_routes(router)
    actual = [
        (method, route.path, route.name)
        for route in routes
        for method in sorted(route.methods or set())
    ]
    eager_app_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith(prefix)
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
    _assert_router_contract(guided_router, "/api/guided/", EXPECTED_GUIDED_ROUTES)


def test_copilot_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(copilot_router, "/api/copilot/", EXPECTED_COPILOT_ROUTES)


def test_page_guide_route_method_path_and_operation_name_snapshot() -> None:
    _assert_router_contract(
        page_guide_router,
        "/api/page-guide/",
        EXPECTED_PAGE_GUIDE_ROUTES,
    )


def test_guided_route_owner_boundary() -> None:
    package_root = Path(__file__).parents[1] / "src" / "easyicu" / "webserver"
    app_source = (package_root / "app.py").read_text(encoding="utf-8")
    guided_source = (package_root / "routes" / "guided.py").read_text(encoding="utf-8")
    copilot_source = (package_root / "routes" / "copilot.py").read_text(
        encoding="utf-8"
    )
    page_guide_source = (package_root / "routes" / "page_guide.py").read_text(
        encoding="utf-8"
    )

    assert "/api/guided/" not in app_source
    assert "/api/copilot/" not in app_source
    assert "/api/page-guide/" not in app_source
    assert "/api/guided/" in guided_source
    assert "/api/copilot/" not in guided_source
    assert "/api/page-guide/" not in guided_source
    assert "/api/copilot/" in copilot_source
    assert "/api/guided/" not in copilot_source
    assert "/api/page-guide/" not in copilot_source
    assert "/api/page-guide/" in page_guide_source
    assert "/api/guided/" not in page_guide_source
    assert "/api/copilot/" not in page_guide_source
    assert "easyicu.webserver.app" not in guided_source
    assert "easyicu.webserver.app" not in copilot_source
    assert "easyicu.webserver.app" not in page_guide_source


def test_root_static_mount_stays_last() -> None:
    static_mount = app.routes[-1]
    fs_list_index = next(
        index
        for index, route in enumerate(app.routes)
        if getattr(route, "path", None) == "/api/fs/list"
    )
    agent_history_index = next(
        index
        for index, route in enumerate(app.routes)
        if getattr(route, "path", None) == "/api/agent-runs/history"
    )
    ideas_index = next(
        index
        for index, route in enumerate(app.routes)
        if getattr(route, "path", None) == "/api/ideas/mine"
    )

    assert (
        _router_registration_index(system_router)
        < fs_list_index
        < agent_history_index
        < _router_registration_index(guided_router)
        < _router_registration_index(copilot_router)
        < _router_registration_index(page_guide_router)
        < ideas_index
        < len(app.routes) - 1
    )
    assert isinstance(static_mount, Mount)
    assert static_mount.name == "static"
    assert static_mount.path == ""
