from datetime import date
from pathlib import Path

from tools.audit_capability_inventory import (
    audit_capability_inventory,
    parse_inventory,
    zero_inbound_leaf_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_zero_inbound_module_has_an_explicit_disposition() -> None:
    assert audit_capability_inventory(REPO_ROOT, today=date(2026, 8, 13)) == ()


def test_inventory_parser_keeps_status_and_review_decisions() -> None:
    rows = parse_inventory(
        REPO_ROOT / "docs" / "research_agent_capability_inventory.md"
    )
    by_module = {row.module: row for row in rows}

    assert by_module["methods/rmst.py"].status == "awaiting-wiring"
    assert by_module["reporting/result_card.py"].status == "external-consumer"
    assert by_module["graph.py"].review == "2.0"


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
