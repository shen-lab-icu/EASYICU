from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "tools" / "research_agent_framework_release.py"
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
    assert "test_workflow.py" in flattened
    assert "test_graph_poc.py" not in flattened
    assert "test_char_golden_run_bundle.py" in flattened
    assert "test_permissioned_quarantine_mirror_failure" in flattened


def test_release_gate_references_only_existing_test_files() -> None:
    module = _module()
    missing = []
    for _name, command in module.RELEASE_COMMANDS:
        for token in command:
            if token.startswith("tests/") and ".py" in token:
                relative_path = token.split("::", maxsplit=1)[0]
                if not (module.REPO_ROOT / relative_path).is_file():
                    missing.append(relative_path)

    assert missing == []


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
    assert report["execution_policy"]["runtime_monitoring"] == "not_instrumented"


def test_release_report_binds_exact_clean_git_commit(monkeypatch) -> None:
    module = _module()

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(module, "_run_command", lambda _command: _Completed())
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda: {
            "commit": "a" * 40,
            "dirty": False,
            "status_porcelain_sha256": "0" * 64,
            "status_porcelain": [],
        },
    )
    report = module.run_release_gate()
    assert report["status"] == "passed"
    assert report["git"]["commit"] == "a" * 40
    assert report["git"]["dirty"] is False
    assert "provider_calls" not in report
    assert "patient_data_reads" not in report


def test_release_gate_rejects_dirty_tree_even_when_commands_pass(monkeypatch) -> None:
    module = _module()

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(module, "_run_command", lambda _command: _Completed())
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda: {
            "commit": "b" * 40,
            "dirty": True,
            "status_porcelain_sha256": "1" * 64,
            "status_porcelain": [" M src/example.py"],
        },
    )
    assert module.run_release_gate()["status"] == "failed"
