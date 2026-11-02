"""Regression tests for the offline resource/context measurement gate."""

from __future__ import annotations

import importlib.util
import json
import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools/research_agent_resource_baseline.py"
BASELINE_PATH = REPO_ROOT / "tools/arch_baselines/research_agent_resource_context.json"
CURRENT_PATH = REPO_ROOT / "tools/arch_baselines/research_agent_resource_context.current.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "research_agent_resource_baseline", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_resource_measurement_covers_exact_canonical9_order() -> None:
    tool = _load_tool()
    measured = tool.measure()

    assert measured["task_order"] == [
        "E1",
        "E2",
        "E3",
        "M1",
        "M2",
        "M3",
        "H1",
        "H2",
        "H3",
    ]
    assert measured["provider_calls"] == 0
    assert measured["patient_data_reads"] == 0
    assert measured["summary"]["task_count"] == 9


def test_resource_measurement_binds_live_planner_schema_and_prompt_sources() -> None:
    tool = _load_tool()

    assert {
        "src/easyicu/research_agent/agents/planner.py",
        "src/easyicu/research_agent/agents/plan_payload.py",
        "src/easyicu/research_agent/providers/prompts/v1/system.txt",
        "src/easyicu/research_agent/schema.py",
        "src/easyicu/research_agent/planning/scientific_action_catalog.py",
        "src/easyicu/research_agent/planning/analysis_method_suite.py",
        "src/easyicu/research_agent/planning/method_adapter_catalog.py",
        "src/easyicu/research_agent/concept_catalog.py",
        "src/easyicu/concept/loader.py",
        "src/easyicu/resources.py",
        "src/easyicu/data/sic-case-duration-overlay.v1.json",
    } <= set(tool.SOURCE_FILES)


def test_offline_resource_measurement_is_byte_deterministic() -> None:
    tool = _load_tool()

    first = json.dumps(tool.measure(), ensure_ascii=False, sort_keys=True)
    second = json.dumps(tool.measure(), ensure_ascii=False, sort_keys=True)

    assert first == second


def test_each_selected_resource_is_digest_bound_and_selection_is_llm_free() -> None:
    tool = _load_tool()
    measured = tool.measure()

    for task in measured["tasks"]:
        assert task["resource_selection_provider_calls"] == 0
        assert task["planning_contract_bytes"] > 0
        assert len(task["planning_contract_sha256"]) == 64
        for resource in task["selected_know_how"]:
            assert len(resource["file_sha256"]) == 64
        assert task["planner_with_resources"]["total_bytes"] <= 80_000
        coder = task["coder_resources"]
        assert coder["provider_calls"] == 0
        assert coder["prompt_bytes"] <= coder["prompt_limit_bytes"] == 8_000
        assert len(coder["selection_receipt_sha256"]) == 3
        assert all(len(digest) == 64 for digest in coder["selection_receipt_sha256"])
        assert all(len(resource["sha256"]) == 64 for resource in coder["selected"])


def test_checked_in_resource_context_baseline_has_no_drift() -> None:
    tool = _load_tool()
    assert tool.verify_current_snapshot(CURRENT_PATH) == 0


def test_resource_snapshot_binds_current_sources_and_historical_ceilings(tmp_path) -> None:
    tool = _load_tool()
    historical = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    measured = tool.measure()
    assert tool.historical_ceiling_errors(historical, measured) == []
    current = tmp_path / "current.json"
    tool._write(current, measured, reason="current fixture")
    assert tool.verify_current_snapshot(current, measured=measured) == 0

    # A refreshed snapshot cannot approve a larger historical budget.
    increased = copy.deepcopy(measured)
    increased["tasks"][0]["planner_with_resources"]["total_bytes"] = (
        historical["tasks"][0]["planner_with_resources"]["total_bytes"] + 1
    )
    tool._write(current, increased, reason="simulated regression")
    assert tool.verify_current_snapshot(current, measured=increased) == 1

    missing_source = copy.deepcopy(measured)
    del missing_source["source_sha256"]["src/easyicu/research_agent/schema.py"]
    tool._write(current, missing_source, reason="simulated missing source")
    assert tool.verify_current_snapshot(current, measured=missing_source) == 1

    # Nor can an unchanged snapshot hide an altered source or prompt segment.
    for mutate in ("source", "prompt"):
        changed = copy.deepcopy(measured)
        if mutate == "source":
            changed["source_sha256"]["src/easyicu/research_agent/schema.py"] = "0" * 64
        else:
            changed["tasks"][0]["planner_with_resources"]["segments"][
                "planner_contract_and_typed_context"
            ]["sha256"] = "0" * 64
        tool._write(current, measured, reason="current fixture")
        assert tool.verify_current_snapshot(current, measured=changed) == 1

    damaged_history = tmp_path / "history.json"
    damaged_history.write_text(BASELINE_PATH.read_text(encoding="utf-8") + "\n")
    assert tool.verify_current_snapshot(
        current, historical_path=damaged_history, measured=measured,
    ) == 1


def test_historical_ceiling_does_not_accept_removed_resource_or_changed_limit() -> None:
    tool = _load_tool()
    historical = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    unchanged = copy.deepcopy(historical)
    unchanged["tool_version"] = tool.TOOL_VERSION
    assert tool.historical_ceiling_errors(historical, unchanged) == []
    for mutation in ("resource", "limit", "segment"):
        changed = copy.deepcopy(unchanged)
        if mutation == "resource":
            changed["tasks"][0]["selected_know_how"].clear()
        elif mutation == "limit":
            changed["tasks"][0]["planner_with_resources"]["limit_bytes"] += 1
        else:
            changed["tasks"][0]["planner_without_resources"]["segments"][
                "planner_contract_and_typed_context"
            ]["required"] = False
        assert tool.historical_ceiling_errors(historical, changed)


def test_resource_baseline_emit_requires_a_reason(tmp_path, monkeypatch) -> None:
    tool = _load_tool()
    out = tmp_path / "baseline.json"
    monkeypatch.setattr(tool, "measure", lambda: {"summary": {"task_count": 9}})

    assert tool.main(["--emit", str(out)]) == 2
    assert not out.exists()


def test_resource_baseline_history_is_append_only(tmp_path, monkeypatch) -> None:
    tool = _load_tool()
    out = tmp_path / "baseline.json"
    first = {
        "summary": {"max_planner_with_resources_bytes": 100},
        "source_sha256": {"planner.py": "a"},
    }
    second = {
        "summary": {"max_planner_with_resources_bytes": 120},
        "source_sha256": {"planner.py": "b"},
    }
    monkeypatch.setattr(tool, "measure", lambda: first)
    assert tool.main(["--emit", str(out), "--reason", "initial fixture"]) == 0
    monkeypatch.setattr(tool, "measure", lambda: second)
    assert tool.main(["--emit", str(out), "--reason", "typed contract growth"]) == 0

    recorded = json.loads(out.read_text(encoding="utf-8"))
    assert [item["reason"] for item in recorded["baseline_history"]] == [
        "initial fixture",
        "typed contract growth",
    ]
    assert recorded["baseline_change_summary"]["summary_changes"] == {
        "max_planner_with_resources_bytes": {"was": 100, "now": 120}
    }
    assert recorded["baseline_change_summary"]["source_digest_changes"] == [
        "planner.py"
    ]
    assert tool.diff(recorded, second) == 0
