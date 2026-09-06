"""Prove the named safety/science regressions are actually collectible in CI."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/research_agent/critical_regressions.txt"


def test_named_critical_regressions_collect_every_declared_test_function() -> None:
    paths = MANIFEST.read_text().splitlines()
    assert paths and len(paths) == len(set(paths))
    assert "tests/research_agent/core/test_run_heartbeat.py" in paths
    assert "tests/research_agent/authority/test_scientific_claim_intervals.py" in paths
    assert "tests/research_agent/reporting/test_findings_prose_admission.py" in paths
    expected = []
    for relative in paths:
        path = ROOT / relative
        assert path.is_file() and path.suffix == ".py"
        functions = [
            node.name
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        assert functions, f"Critical test file has no test functions: {relative}"
        expected.extend(f"{relative}::{name}" for name in functions)
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-q",
            "-m",
            "",
            "--collect-only",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    nodeids = {
        line.split("[", 1)[0] for line in collected.stdout.splitlines() if "::" in line
    }
    assert set(expected) <= nodeids
    assert "deselected" not in collected.stdout


def test_ci_runs_the_manifest_unfiltered_and_preserves_candidate_receipts() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/research_agent_ci.yml").read_text()
    )
    steps = workflow["jobs"]["test"]["steps"]
    critical = next(
        step for step in steps if "critical_regressions.txt" in step.get("run", "")
    )
    run = critical["run"]
    assert "-k " not in run
    assert "--collect-only" in run and "--junitxml=" in run
    assert run.count('pytest "${critical_tests[@]}" -q -m ""') == 2
    assert 'test "${#critical_tests[@]}" -gt 0' in run
    assert any("pip freeze --all" in step.get("run", "") for step in steps)
    assert any("git rev-parse HEAD" in step.get("run", "") for step in steps)
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "always()" in upload["if"]
    assert "github.sha" in upload["with"]["name"]
    assert upload["with"]["path"] == "ci-receipts/"
