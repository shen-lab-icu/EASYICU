"""Registration contracts for native FastAPI route modules."""

from __future__ import annotations

from fastapi.routing import APIRoute
from starlette.routing import Mount

from easyicu.webserver.app import app
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


def _system_routes() -> list[APIRoute]:
    return [
        route
        for route in system_router.routes
        if isinstance(route, APIRoute)
    ]


def _system_router_registration_index() -> int:
    """Support both eager and lazy FastAPI ``include_router`` versions."""
    endpoints = {route.endpoint for route in _system_routes()}
    for index, route in enumerate(app.routes):
        if getattr(route, "original_router", None) is system_router:
            return index
        if getattr(route, "endpoint", None) in endpoints:
            return index
    raise AssertionError("system router is not registered on the FastAPI app")


def test_system_route_method_path_snapshot() -> None:
    routes = _system_routes()
    actual = [
        (method, route.path)
        for route in routes
        for method in sorted(route.methods or set())
    ]

    assert actual == EXPECTED_SYSTEM_ROUTES


def test_system_route_operation_names_and_favicon_schema_contract() -> None:
    routes = _system_routes()

    assert [route.name for route in routes] == EXPECTED_SYSTEM_OPERATION_NAMES
    favicon_route = next(route for route in routes if route.path == "/favicon.ico")
    assert favicon_route.include_in_schema is False


def test_root_static_mount_stays_last() -> None:
    static_mount = app.routes[-1]
    fs_list_index = next(
        index
        for index, route in enumerate(app.routes)
        if getattr(route, "path", None) == "/api/fs/list"
    )

    assert _system_router_registration_index() < fs_list_index < len(app.routes) - 1
    assert isinstance(static_mount, Mount)
    assert static_mount.name == "static"
    assert static_mount.path == ""
