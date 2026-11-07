"""Privacy tests must run in CI, not skip themselves for a missing optional import.

``test_patient_report_export_privacy`` began with ``pytest.importorskip("plotly")``
while CI installed only the dev and webapp extras, so the privacy pin was skipped
on every shard and nothing reported it. These checks fail when a required test
file depends, through ``importorskip``, on a package that the CI test job does
not install, and (on GitHub Actions) when such a package is not importable.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.required_in_ci import (
    CI_INSTALLED_EXTRAS,
    IMPORT_TO_DISTRIBUTION,
    REPO_ROOT,
    required_test_files,
)
from tests.support.required_in_ci_guard import RequiredSkipGuard

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime
    import tomli as tomllib

CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _normal(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _ci_installed_distributions() -> set[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})
    for extra in CI_INSTALLED_EXTRAS:
        requirements.extend(extras[extra])
    names = set()
    for requirement in requirements:
        match = _NAME_RE.match(requirement)
        if match:
            names.add(_normal(match.group(1)))
    return names


def _importorskip_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            modules.append(node.args[0].value)
    return modules


def _distribution(module: str) -> str:
    top = module.split(".")[0]
    return _normal(IMPORT_TO_DISTRIBUTION.get(top, top))


def test_the_ci_test_job_installs_the_extras_this_guard_assumes():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    expected = 'pip install -e ".[' + ",".join(CI_INSTALLED_EXTRAS) + ']"'
    assert expected in workflow


def test_required_test_files_exist():
    missing = [str(path) for path in required_test_files() if not path.is_file()]
    assert not missing


def test_required_tests_only_skip_on_packages_ci_installs():
    installed = _ci_installed_distributions()
    gaps = {}
    for path in required_test_files():
        absent = sorted(
            module
            for module in _importorskip_modules(path)
            if not module.startswith("easyicu") and _distribution(module) not in installed
        )
        if absent:
            gaps[str(path.relative_to(REPO_ROOT))] = absent
    assert not gaps, f"required tests would skip in CI for packages CI does not install: {gaps}"


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true",
    reason="the install contract holds on GitHub Actions; local CI-equivalent runs install nothing",
)
def test_required_test_imports_are_importable_in_ci():
    unimportable = {}
    for path in required_test_files():
        for module in _importorskip_modules(path):
            try:
                importlib.import_module(module)
            except ImportError as error:
                unimportable[module] = f"{type(error).__name__}: {error}"
    assert not unimportable, f"required tests would skip in CI: {unimportable}"


def _report(nodeid: str, *, skipped: bool = True, wasxfail: bool = False) -> SimpleNamespace:
    report = SimpleNamespace(nodeid=nodeid, skipped=skipped)
    if wasxfail:
        report.wasxfail = "expected"
    return report


_PRIVACY = "tests/core/test_patient_report_export_privacy.py"


def _guard() -> RequiredSkipGuard:
    return RequiredSkipGuard(REPO_ROOT, required_test_files())


def test_a_skipped_required_file_fails_the_session_on_github_actions_only():
    for nodeid in (_PRIVACY, f"{_PRIVACY}::test_export_hides_identifier_and_selects_only_requested_patient"):
        guard = _guard()
        guard.record(_report(nodeid))

        on_actions = SimpleNamespace(exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED)
        assert guard.finish(on_actions, {"GITHUB_ACTIONS": "true"}) == [_PRIVACY]
        assert on_actions.exitstatus == pytest.ExitCode.TESTS_FAILED

        local = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
        assert guard.finish(local, {"CI": "true"}) == []
        assert local.exitstatus == pytest.ExitCode.OK


def test_other_skips_and_expected_failures_do_not_trip_the_guard():
    guard = _guard()
    guard.record(_report("tests/core/test_cohort_visualization_layout.py::test_x"))
    guard.record(_report(f"{_PRIVACY}::test_x", wasxfail=True))
    guard.record(_report(f"{_PRIVACY}::test_x", skipped=False))
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    assert guard.finish(session, {"GITHUB_ACTIONS": "true"}) == []
    assert session.exitstatus == pytest.ExitCode.OK


def test_a_required_file_left_uncollected_counts_only_inside_the_session_scope():
    privacy = (REPO_ROOT / _PRIVACY).resolve()

    def omitted(path: Path) -> bool:
        return path == privacy

    inside = _guard()
    inside.record_uncollected([REPO_ROOT / "tests"], omitted)
    assert inside.violations({"GITHUB_ACTIONS": "true"}) == [_PRIVACY]

    outside = _guard()
    outside.record_uncollected([REPO_ROOT / "tests" / "governance"], omitted)
    assert outside.violations({"GITHUB_ACTIONS": "true"}) == []


def test_the_shared_conftest_forwards_reports_to_the_guard():
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    for call in (
        "_required_in_ci_guard.configure(config, omitted=_optional_omission)",
        "_required_in_ci_guard.record(report)",
        "_required_in_ci_guard.finish(session)",
    ):
        assert call in conftest
    for hook in ("def pytest_collectreport(", "def pytest_runtest_logreport(", "def pytest_sessionfinish("):
        assert conftest.count(hook) == 1


def test_ci_workflows_print_skip_reasons():
    for name in ("ci.yml", "research_agent_ci.yml", "pi_workspace_security_ci.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for line in workflow.splitlines():
            command = line.strip().removeprefix("run:").strip()
            if re.match(r"(python -m )?pytest\b", command) and "--collect-only" not in command:
                assert "-rs" in command, f"{name}: {command}"
