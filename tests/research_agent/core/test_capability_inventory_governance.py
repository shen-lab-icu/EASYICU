from datetime import date
from pathlib import Path

from tools.audit_capability_inventory import (
    ALLOWED_STATUSES,
    _current_graph,
    audit_capability_inventory,
    parse_inventory,
    zero_inbound_leaf_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_every_zero_inbound_module_has_an_explicit_disposition() -> None:
    assert audit_capability_inventory(REPO_ROOT, today=date(2026, 8, 13)) == ()


def test_current_graph_uses_running_interpreter_without_repo_venv(tmp_path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "research_agent_module_graph.py").write_text(
        'import json\nprint(json.dumps({"modules": {"demo": "demo.py"}, "edges": []}))\n',
        encoding="utf-8",
    )

    assert _current_graph(tmp_path) == {
        "modules": {"demo": "demo.py"},
        "edges": [],
    }


def test_inventory_parser_keeps_status_and_review_decisions() -> None:
    rows = parse_inventory(
        REPO_ROOT / "docs" / "research_agent_capability_inventory.md"
    )
    by_module = {row.module: row for row in rows}

    assert by_module["methods/rmst.py"].status == "experimental"
    assert by_module["reporting/result_card.py"].status == "production_reachable"
    assert by_module["graph.py"].review == "2.0"
    assert {row.status for row in rows} <= ALLOWED_STATUSES


def test_every_production_capability_binds_a_real_reachability_test() -> None:
    rows = parse_inventory(
        REPO_ROOT / "docs" / "research_agent_capability_inventory.md"
    )

    production = [row for row in rows if row.status == "production_reachable"]
    assert production
    for row in production:
        assert "→" in row.activation
        reference = row.tests.strip().strip("`")
        relative_path, test_name = reference.split("::", 1)
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert f"def {test_name}(" in source
        assert "call:" in row.proof


def test_production_status_fails_closed_without_route_and_exact_test(tmp_path) -> None:
    package = tmp_path / "src" / "easyicu" / "research_agent"
    package.mkdir(parents=True)
    (package / "reachable.py").write_text("VALUE = 1\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "research_agent_capability_inventory.md").write_text(
        "| module | LOC | status | owner | activation precondition | tests | route proof | review |\n"
        "| --- | ---: | --- | --- | --- | --- | --- | --- |\n"
        "| `reachable.py` | 1 | `production_reachable` | owner | direct call | 3 | - | 2099-01-01 |\n",
        encoding="utf-8",
    )

    findings = audit_capability_inventory(
        tmp_path,
        today=date(2026, 8, 14),
        graph={"modules": {}, "edges": []},
    )

    assert any("lacks a public-API-to-executor route" in item for item in findings)
    assert any("lacks a valid reachability integration test" in item for item in findings)


def test_production_status_rejects_a_missing_test_function(tmp_path) -> None:
    package = tmp_path / "src" / "easyicu" / "research_agent"
    package.mkdir(parents=True)
    (package / "reachable.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_reachable.py").write_text(
        "def test_something_else():\n    pass\n", encoding="utf-8"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "research_agent_capability_inventory.md").write_text(
        "| module | LOC | status | owner | activation precondition | tests | route proof | review |\n"
        "| --- | ---: | --- | --- | --- | --- | --- | --- |\n"
        "| `reachable.py` | 1 | `production_reachable` | owner | API → executor | `tests/test_reachable.py::test_missing` | `call:public_api` | 2099-01-01 |\n",
        encoding="utf-8",
    )

    findings = audit_capability_inventory(
        tmp_path,
        today=date(2026, 8, 14),
        graph={"modules": {}, "edges": []},
    )

    assert any("points to missing test function" in item for item in findings)


def test_production_status_rejects_a_test_that_does_not_traverse_route(
    tmp_path,
) -> None:
    package = tmp_path / "src" / "easyicu" / "research_agent"
    package.mkdir(parents=True)
    (package / "reachable.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_reachable.py").write_text(
        "def test_route():\n    unrelated()\n    assert trace['other']\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "research_agent_capability_inventory.md").write_text(
        "| module | LOC | status | owner | activation precondition | tests | route proof | review |\n"
        "| --- | ---: | --- | --- | --- | --- | --- | --- |\n"
        "| `reachable.py` | 1 | `production_reachable` | owner | API → executor | `tests/test_reachable.py::test_route` | `call:public_api;trace:executor` | 2099-01-01 |\n",
        encoding="utf-8",
    )

    findings = audit_capability_inventory(
        tmp_path,
        today=date(2026, 8, 14),
        graph={"modules": {}, "edges": []},
    )

    assert any("declared public call is absent" in item for item in findings)


def test_zero_inbound_projection_excludes_package_initializers() -> None:
    graph = {
        "modules": {
            "easyicu.research_agent": "__init__.py",
            "easyicu.research_agent.pkg": "pkg/__init__.py",
            "easyicu.research_agent.pkg.orphan": "pkg/orphan.py",
            "easyicu.research_agent.pkg.used": "pkg/used.py",
            "easyicu.research_agent.caller": "caller.py",
        },
        "edges": [["easyicu.research_agent.caller", "easyicu.research_agent.pkg.used"]],
    }

    assert zero_inbound_leaf_paths(graph) == ("caller.py", "pkg/orphan.py")
