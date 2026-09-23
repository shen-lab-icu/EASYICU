"""The Skill Hub package for the fixed-landmark workflow ships the real scripts."""

from __future__ import annotations

import ast
from pathlib import Path

from easyicu.research_agent.method_skills import (
    DOCUMENTATION_ONLY_BOUNDARY,
    REFERENCE_SCRIPT_PACKAGES,
    REFERENCE_SCRIPTS_BOUNDARY,
    method_skill_package,
)

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "easyicu"
    / "research_agent"
    / "skill_packages"
    / "landmark_categorical_association"
)


def test_fixed_landmark_package_ships_executable_reference_scripts() -> None:
    package = method_skill_package("fixed-landmark-association-study", enabled=True)
    files = {row["path"]: row for row in package["files"]}

    assert package["execution_boundary"] == REFERENCE_SCRIPTS_BOUNDARY
    assert REFERENCE_SCRIPT_PACKAGES["fixed-landmark-association-study"] == (
        "landmark_categorical_association"
    )
    for script in (
        "scripts/spec.py",
        "scripts/load_cohort.py",
        "scripts/run_analysis.py",
        "scripts/generate_all_plots.py",
        "scripts/export_all.py",
        "scripts/run_all.py",
        "scripts/example_data.py",
    ):
        assert script in files
        ast.parse(files[script]["content"])
        on_disk = PACKAGE_ROOT / (script if script != "scripts/spec.py" else "spec.py")
        assert files[script]["content"] == on_disk.read_text(encoding="utf-8")
    for reference in (
        "references/reference_scripts.md",
        "references/methods.md",
        "references/caveat_flags.md",
        "references/reporting_checklist.md",
        "references/comparators_kdigo_mortality.md",
    ):
        assert reference in files
    assert files["references/reference_scripts.md"]["content"] == (
        PACKAGE_ROOT / "SKILL.md"
    ).read_text(encoding="utf-8")
    skill = files["SKILL.md"]["content"]
    assert "### Reference scripts (executable)" in skill
    assert "✓ Analysis completed successfully!" in skill
    assert "=== Export Complete ===" in skill
    assert "key_metrics.csv" in skill
    assert "## Workflow" in skill  # generated sections are retained
    assert all(len(row["sha256"]) == 64 for row in files.values())


def test_other_workflows_remain_documentation_only() -> None:
    package = method_skill_package("survival-time-to-event", enabled=True)
    assert package["execution_boundary"] == DOCUMENTATION_ONLY_BOUNDARY
    assert all(not row["path"].startswith("scripts/") for row in package["files"])
