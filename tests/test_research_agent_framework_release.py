from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "research_agent_framework_release.py"
    spec = importlib.util.spec_from_file_location("framework_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_gate_is_offline_and_covers_all_new_boundaries() -> None:
    module = _module()
    flattened = " ".join(
        token for _name, command in module.RELEASE_COMMANDS for token in command
    )
    assert "run_research_agent_bench" not in flattened
    assert "http" not in flattened
    assert "test_resource_scheduler.py" in flattened
    assert "test_permissioned_memory_store.py" in flattened
    assert "test_capability_requests.py" in flattened
    assert "test_graph_poc.py" in flattened
    assert "test_char_golden_run_bundle.py" in flattened


def test_release_gate_stops_at_first_failure(monkeypatch) -> None:
    module = _module()

    class _Completed:
        returncode = 1
        stdout = "failed"
        stderr = "error"

    calls = []

    def fake_run(command):
        calls.append(command)
        return _Completed()

    monkeypatch.setattr(module, "_run_command", fake_run)
    report = module.run_release_gate()
    assert report["status"] == "failed"
    assert len(calls) == 1
    assert report["provider_calls"] == 0
    assert report["patient_data_reads"] == 0
