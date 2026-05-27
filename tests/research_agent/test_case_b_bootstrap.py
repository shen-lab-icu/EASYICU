"""Case B pilot-bootstrap tests.

These tests intentionally read the active concept dictionary but never modify
it. Missing concept IDs are early feedback for the user's concept-dictionary
audit, not something this case bootstrap should patch around.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "benchmark" / "cases" / "case_b_sofa2_sepsis"
PATTERNS_PATH = CASE_DIR / "cohort_patterns.json"
PILOT_SCRIPT = REPO_ROOT / "tools" / "run_pilot_phase1.sh"


@pytest.fixture
def dirty_marker() -> Path:
    marker = REPO_ROOT / "__case_b_dirty_marker_for_test__"
    marker.write_text("dirty marker for pilot bootstrap test\n", encoding="utf-8")
    try:
        yield marker
    finally:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass


def _pattern_names() -> list[str]:
    payload = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    return sorted(payload["patterns"])


def _concept_ids_in_patterns() -> set[str]:
    payload = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    concept_ids: set[str] = set()
    for raw in payload["patterns"].values():
        definition = raw.get("definition", raw)
        for section in ("inclusion", "exclusion"):
            for pred in definition.get(section, []) or []:
                concept_ids.add(str(pred["concept_id"]))
    return concept_ids


def test_case_b_patterns_load_and_validate() -> None:
    from easyicu.research_agent.cohort_schema import PatternRegistry

    registry = PatternRegistry()
    registry.register_from_file(PATTERNS_PATH)

    for name in _pattern_names():
        expanded = registry.expand(name)
        assert expanded.derived_from_named == name
        assert expanded.inclusion


def test_case_b_patterns_concept_ids_exist() -> None:
    from easyicu.research_agent.cohort_schema import known_concept_ids

    missing = sorted(_concept_ids_in_patterns() - known_concept_ids())

    assert missing == []


def test_register_case_b_patterns_is_idempotent() -> None:
    from benchmark.cases.case_b_sofa2_sepsis import (
        register_case_b_patterns,
        register_patterns,
    )
    from easyicu.research_agent.cohort_schema import PatternRegistry

    registry = PatternRegistry()
    register_patterns(registry)
    register_patterns(registry)

    assert registry.expand("sepsis3_at_admission").derived_from_named == (
        "sepsis3_at_admission"
    )
    assert register_case_b_patterns is register_patterns


def test_pilot_runner_dry_run_with_mock_backend(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    out_root = tmp_path / "pilot_dry_run"

    result = subprocess.run(
        [
            "bash",
            str(PILOT_SCRIPT),
            "--dry-run",
            "--backend",
            "mock",
            "--allow-dirty",
            "--out-root",
            str(out_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest_path = out_root / "pilot_phase1_dry_run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry_run"
    assert payload["case"] == "case_b_sofa2_sepsis"
    assert payload["backend"] == "mock"
    assert "pilot_real_llm.py" in payload["command_preview"]
    assert "--bench-kind rule" in payload["command_preview"]
    assert "sofa2_mortality" in payload["command_preview"]
    assert payload["concept_dict_fingerprint"]["concept_dict_sha"]
    assert payload["concept_dict_fingerprint"]["sofa2_dict_sha"]


def test_run_pilot_aborts_on_dirty_tree_without_flag(
    tmp_path: Path,
    dirty_marker: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    assert dirty_marker.exists()
    result = subprocess.run(
        [
            "bash",
            str(PILOT_SCRIPT),
            "--dry-run",
            "--backend",
            "mock",
            "--out-root",
            str(tmp_path / "pilot_dirty"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3
    assert "dirty worktree" in result.stderr
