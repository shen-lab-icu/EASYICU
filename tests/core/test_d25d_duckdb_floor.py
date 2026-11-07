"""The declared DuckDB range only admits engines the strict CSV read is tested on (delta25d).

B-D25-CSV-VERSION: pyproject allowed duckdb >=0.9.0,<2, but the strict
MIMIC-III CSV read binds strict_mode, store_rejects and rejects_scan. 0.9.2
accepts none of them and 1.1.3 lacks strict_mode, so a supported install
would fail at parameter binding. The floor is the lowest engine the strict
CSV regression ran on (1.5.5 locally, 1.5.6 on the server). Every
install-time constraint in the repository must agree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import duckdb
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
TESTED_FLOOR = Version("1.5.5")


def _pyproject_duckdb() -> Requirement:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    (spec,) = [r for r in data["project"]["dependencies"] if Requirement(r).name == "duckdb"]
    return Requirement(spec)


def test_the_declared_floor_is_the_tested_strict_csv_engine():
    requirement = _pyproject_duckdb()
    assert not requirement.specifier.contains("1.1.3")  # no strict_mode
    assert not requirement.specifier.contains("0.9.2")  # no strict_mode/store_rejects/rejects_scan
    assert requirement.specifier.contains(str(TESTED_FLOOR))
    assert not requirement.specifier.contains("2.0.0")


def test_every_install_constraint_agrees_with_pyproject():
    requirement = _pyproject_duckdb()
    ci = (ROOT / ".github/workflows/research_agent_ci.yml").read_text()
    pins = re.findall(r'"(duckdb[^"]*)"', ci)
    assert pins and all(Requirement(p).specifier == requirement.specifier for p in pins)
    for lock in sorted((ROOT / "desktop").glob("requirements-*.lock")):
        (pin,) = re.findall(r"^duckdb==([0-9.]+)", lock.read_text(), flags=re.M)
        assert requirement.specifier.contains(pin), lock.name


def test_the_installed_engine_is_in_range():
    assert _pyproject_duckdb().specifier.contains(duckdb.__version__)
