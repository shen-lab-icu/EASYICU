"""Invalid declarations are actionable schema failures, never browser pending."""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.tools import execute_tool
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
from easyicu.webserver.pi_copilot.workspace_runtime_qa import (
    RuntimeQaError, RuntimeQaStore, SCHEMA, contract_diagnostics,
    contract_from_html, contract_preflight_receipt, validate_contract,
)


def page(contract):
    return '<!doctype html><p data-rate>25.0%</p><script type="application/json" data-easyicu-runtime-qa>' + json.dumps(contract) + '</script>'


BAD = {"schema": SCHEMA, "checks": [
    {"id": "rate", "kind": "text_number", "selector": "[data-rate]", "expected": "DO_NOT_ECHO_SOURCE"},
    {"id": "filter", "kind": "select_cases", "selector": "select", "cases": [
        {"value": "subset", "checks": [{"kind": "text_number", "selector": "[data-count]", "expected": 7}]},
    ]},
]}


def test_all_relevant_schema_paths_survive_html_parsing_without_raw_values():
    diagnostic = contract_diagnostics(page(BAD))
    assert diagnostic["status"] == "invalid"
    assert diagnostic["polling_allowed"] is False
    assert diagnostic["next_action"] == "repair_contract_schema_then_check_once"
    assert [issue["path"] for issue in diagnostic["issues"]] == [
        "/checks/0/expected", "/checks/1/cases/0/checks/0/id",
    ]
    assert "DO_NOT_ECHO_SOURCE" not in json.dumps(diagnostic)
    with pytest.raises(RuntimeQaError) as error:
        contract_from_html(page(BAD))
    assert error.value.details["issues"] == diagnostic["issues"]


@pytest.mark.parametrize("expected", ["25.0%", "25", True, None, float("inf")])
def test_formatted_numeric_expectations_are_rejected_not_coerced(expected):
    contract = {"schema": SCHEMA, "checks": [{"id": "rate", "kind": "text_number", "selector": "[data-rate]", "expected": expected}]}
    assert contract_diagnostics(page(contract))["status"] == "invalid"


def test_valid_numeric_expectation_and_unique_child_id_pass_only_schema():
    contract = deepcopy(BAD)
    contract["checks"][0]["expected"] = 25.0
    contract["checks"][1]["cases"][0]["checks"][0]["id"] = "subset-count"
    diagnostic = contract_diagnostics(page(contract))
    assert diagnostic == {"status": "valid", "schema": SCHEMA, "scope": "static_contract_schema_only",
                          "declared_check_count": 3, "issues": []}
    assert contract_preflight_receipt(page(contract), "a" * 64) is None


def test_absent_contract_null_contract_and_bad_json_are_distinct():
    absent = contract_preflight_receipt("<p>No declarations</p>", "a" * 64)
    assert absent["status"] == "partial" and absent["code"] == "runtime_qa_contract_missing"
    assert absent["polling_allowed"] is False
    null = contract_preflight_receipt(page(None), "a" * 64)
    assert null["status"] == "failed" and null["code"] == "runtime_qa_contract_invalid"
    bad_json = '<script type="application/json" data-easyicu-runtime-qa>{broken}</script>'
    assert contract_diagnostics(bad_json)["issues"][0]["code"] == "runtime_qa_contract_json_invalid"
    observed = {"status": "partial", "warnings": ["suspicious_percent_bar"], "runtime_errors": 0,
                "csv_export_ready": True, "scientific_verified": False, "browser_download_verified": False}
    existing = contract_preflight_receipt("<p>No declarations</p>", "a" * 64, observed=observed)
    assert existing["warnings"] == observed["warnings"] and existing["csv_export_ready"] is True
    assert existing["status"] == "partial" and existing["browser_observed"] is True


def test_diagnostics_stay_bounded_for_many_bad_checks():
    contract = {"schema": SCHEMA, "checks": [{"kind": "text_number", "expected": "bad"}] * 32}
    diagnostic = contract_diagnostics(page(contract))
    assert len(diagnostic["issues"]) == 16
    assert len(json.dumps(diagnostic).encode()) < 10_000


def test_check_tool_returns_structured_blocked_receipt_and_readback_does_not_poll(tmp_path):
    workspace = ProjectWorkspace(tmp_path)
    saved = workspace.write_file("project", "view.html", page(BAD))
    with pytest.raises(PiCopilotError) as error:
        workspace.check_file("project", "view.html")
    assert error.value.code == "pi_workspace_runtime_contract_invalid"
    assert error.value.details["runtime_qa_contract"]["issues"][1]["path"].endswith("/id")
    context = ToolExecutionContext(
        data_handling="zero_retention_declared", workspace=workspace,
        session=PiSessionRecord(session_id="session", project_id="project", agent_mode="workspace"),
        runtime_qa_store=RuntimeQaStore(),
    )
    checked = execute_tool("easyicu_check_project_file", {"file": "view.html"}, context)
    assert checked["status"] == "blocked" and checked["details"]["valid"] is False
    assert "resource" not in checked["details"] and "checked_sha256" not in checked["details"]
    assert checked["details"]["runtime_qa_contract"]["polling_allowed"] is False
    readback = execute_tool("easyicu_get_project_runtime_qa", {"file": "view.html", "checked_sha256": saved["sha256"]}, context)
    qa = readback["details"]["runtime_qa"]
    assert qa["status"] == "failed" and qa["code"] == "runtime_qa_contract_invalid"
    assert qa["polling_allowed"] is False
    assert qa["scientific_verified"] is False and qa["browser_download_verified"] is False


def test_canonical_skill_example_is_an_executable_schema_example():
    skill = Path(__file__).parents[3] / "src/easyicu/webserver/pi_copilot/node_app/src/skills/web-prototype/SKILL.md"
    example = re.search(r"```json\n(.*?)\n```", skill.read_text(), re.S)
    assert example is not None
    normalized = validate_contract(json.loads(example.group(1)))
    assert {row["kind"] for row in normalized["checks"]} == {"text_number", "ratio", "bar_geometry", "select_cases", "download_control"}


def test_valid_contract_waiting_for_browser_has_explicit_no_poll_action():
    store = RuntimeQaStore()
    contract = {"schema": SCHEMA, "checks": [{"id": "count", "kind": "text_number", "selector": "[data-count]", "expected": 12}]}
    store.begin("project", "view.html", "a" * 64, contract)
    qa = store.read("project", "view.html", "a" * 64)
    assert qa["status"] == "partial" and qa["code"] == "browser_runtime_pending"
    assert qa["polling_allowed"] is False
    assert qa["next_action"] == "open_preview_once_then_report_browser_pending"
