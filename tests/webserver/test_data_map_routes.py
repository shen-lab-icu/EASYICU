"""Direct HTTP adapter checks; no ASGI transport or running server."""

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException

from easyicu.webserver import sources
from easyicu.webserver.routes.data_map import get_data_map, router


def test_read_route_passes_filters_and_has_no_write_method(monkeypatch):
    monkeypatch.setattr(sources, "_read_raw", lambda: {})
    result = get_data_map(database="eicu", concept="hr", family="survival", concept_limit=1, source_limit=1)
    assert len(result.databases) == 1
    row, = result.databases
    assert row.database == "eicu"
    assert [c.concept for c in row.concepts] == ["hr"]
    assert [s.family for s in row.studies] == ["survival"]
    assert result.concept_limit == result.source_limit == 1
    route, = router.routes
    assert route.path == "/api/data-map"
    assert route.methods == {"GET"}


def test_route_refuses_an_unknown_database_with_a_machine_code():
    with pytest.raises(HTTPException) as error:
        get_data_map(database="synthetic_unknown")
    assert error.value.status_code == 400
    assert error.value.detail == {"error": "data_map_unknown_database"}


def test_openapi_bounds_selectors_and_pages_without_exposing_export_coordinates():
    app = FastAPI()
    app.include_router(router)
    operation = app.openapi()["paths"]["/api/data-map"]["get"]
    parameters = {item["name"]: item["schema"] for item in operation["parameters"]}
    assert set(parameters) == {"database", "concept", "family", "concept_offset", "concept_limit", "source_offset", "source_limit"}
    for key in ("concept_limit", "source_limit"):
        assert parameters[key]["minimum"] == 1
        assert parameters[key]["maximum"] == 100
    for key in ("concept_offset", "source_offset"):
        assert parameters[key]["minimum"] == 0
        assert parameters[key]["maximum"] == 10000
    for key in ("database", "concept", "family"):
        assert parameters[key]["anyOf"][0]["maxLength"] == 160


def test_app_includes_the_data_map_router_without_importing_runtime_startup():
    path = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/app.py"
    tree = ast.parse(path.read_text())
    names = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
             and node.module == "easyicu.webserver.routes.data_map" for alias in node.names if alias.name == "router"}
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "include_router"
               and any(isinstance(arg, ast.Name) and arg.id in names for arg in node.value.args)
               for node in tree.body)
