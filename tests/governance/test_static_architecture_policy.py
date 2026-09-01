"""Regression checks for the first static architecture boundary set."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime
    import tomli as tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_research_agent_files_do_not_inherit_f401_f841_ignores() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]

    for pattern in (
        "src/easyicu/research_agent/*.py",
        "src/easyicu/research_agent/**/*.py",
    ):
        assert "F401" not in ignores[pattern]
        assert "F841" not in ignores[pattern]


def test_import_linter_contains_each_reviewed_boundary() -> None:
    policy = (ROOT / ".importlinter").read_text(encoding="utf-8")

    assert "Reporting must not import the orchestration pipeline" in policy
    assert "Authority must not import agent implementations" in policy
    assert "Core data modules must not import the webserver" in policy
    assert "Production package must not import benchmark fixtures" in policy
