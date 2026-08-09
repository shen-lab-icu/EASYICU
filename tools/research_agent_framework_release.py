#!/usr/bin/env python3
"""Run the no-provider Research Agent framework release gate.

The gate exercises resource selection, bounded context, permissioned memory,
capability approval, the explicit workflow/HITL contract, semantic golden,
typed inputs, architecture and dependency direction.  It never calls an LLM
endpoint or reads ICU data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "easyicu.research_agent_framework_release/2"
TOOL_VERSION = "1.3.0"

RELEASE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "resource_context",
        (
            "tools/research_agent_resource_baseline.py",
            "--diff",
            "tools/arch_baselines/research_agent_resource_context.json",
        ),
    ),
    (
        "architecture",
        (
            "tools/arch_measure.py",
            "--diff",
            "tools/arch_baselines/execution_phase.json",
        ),
    ),
    (
        "module_graph",
        (
            "tools/research_agent_module_graph.py",
            "--diff",
            "tools/arch_baselines/research_agent_module_graph.json",
        ),
    ),
    (
        "framework_tests",
        (
            "-m",
            "pytest",
            "tests/research_agent/test_resource_scheduler.py",
            "tests/research_agent/test_bounded_context.py",
            "tests/research_agent/test_permissioned_memory_store.py",
            "tests/research_agent/test_capability_requests.py",
            "tests/research_agent/test_coder_resource_wiring.py",
            "tests/research_agent/test_reviewed_memory_wiring.py",
            "tests/research_agent/test_capability_workflow_wiring.py",
            "tests/research_agent/test_workflow.py",
            "tests/research_agent/test_char_golden_run_bundle.py",
            "tests/research_agent/test_typed_input_consumption_receipt.py",
            "tests/research_agent/test_typed_input_sdk.py",
            "tests/research_agent/test_experience_bank.py::test_permissioned_quarantine_mirror_failure_is_nonfatal_to_completed_run",
            "-q",
        ),
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_command(args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, *args]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT / "src"), existing_pythonpath) if part
    )
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_porcelain": status.splitlines(),
    }


def run_release_gate() -> dict[str, Any]:
    git_state = _git_state()
    results = []
    for name, args in RELEASE_COMMANDS:
        completed = _run_command(args)
        results.append(
            {
                "name": name,
                "command": [sys.executable, *args],
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    completed.stderr.encode("utf-8")
                ).hexdigest(),
                "stdout_tail": completed.stdout[-2_000:],
                "stderr_tail": completed.stderr[-2_000:],
            }
        )
        if completed.returncode != 0:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": _sha256(Path(__file__)),
        "git": git_state,
        "execution_policy": {
            "provider_access": "prohibited_by_static_release_command_allowlist",
            "patient_data_access": "prohibited_by_static_release_command_allowlist",
            "runtime_monitoring": "not_instrumented",
            "environment_marker": "EASYICU_OFFLINE_RELEASE_GATE=1",
        },
        "status": (
            "passed"
            if not git_state["dirty"]
            and len(results) == len(RELEASE_COMMANDS)
            and all(item["returncode"] == 0 for item in results)
            else "failed"
        ),
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.list:
        print(json.dumps(RELEASE_COMMANDS, indent=2))
        return 0
    os.environ["EASYICU_OFFLINE_RELEASE_GATE"] = "1"
    report = run_release_gate()
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
