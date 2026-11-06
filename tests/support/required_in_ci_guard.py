"""Fail a GitHub Actions pytest session when a required test file is skipped.

The static governance check keeps required files from depending on packages CI
does not install. This guard covers the remaining cases on GitHub Actions: a
required file (see ``required_in_ci``) that reports a skip, or that the session
would leave uncollected because tests/conftest.py omits its optional extra,
turns the session red. Outside GitHub Actions it only records.

tests/conftest.py forwards collection and run reports here. Under xdist the
controller receives every skipped collection report and every test report;
the uncollected-owner check runs on the controller at configure time, for the
required files inside the session's own collection paths.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

from tests.support.required_in_ci import required_test_files

Omission = Callable[[Path], object]


class RequiredSkipGuard:
    def __init__(self, root: Path, required: Iterable[Path]) -> None:
        self.root = Path(root).resolve()
        self.required = {Path(path).resolve() for path in required}
        self.skipped: set[str] = set()

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def record(self, report) -> None:
        if not getattr(report, "skipped", False) or hasattr(report, "wasxfail"):
            return
        file_part = str(report.nodeid).split("::", 1)[0]
        if not file_part:
            return
        path = (self.root / file_part).resolve()
        if path in self.required:
            self.skipped.add(self._relative(path))

    def record_uncollected(self, scope: Iterable[Path], omitted: Omission) -> None:
        """Count required files in ``scope`` that conftest would not collect."""

        roots = [Path(path).resolve() for path in scope]
        for path in sorted(self.required):
            in_scope = any(path == root or root in path.parents for root in roots)
            if in_scope and omitted(path):
                self.skipped.add(self._relative(path))

    def violations(self, environ: Mapping[str, str]) -> list[str]:
        if environ.get("GITHUB_ACTIONS") != "true":
            return []
        return sorted(self.skipped)

    def finish(self, session, environ: Mapping[str, str]) -> list[str]:
        found = self.violations(environ)
        if found and session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return found


def session_scope(config) -> list[Path]:
    """The paths this session collects from: its arguments, else ``testpaths``."""

    args = [str(arg).split("::", 1)[0] for arg in config.args if not str(arg).startswith("-")]
    if not args:
        args = list(config.getini("testpaths")) or ["."]
    scope = []
    for arg in args:
        path = Path(arg)
        scope.append(path if path.is_absolute() else Path(config.invocation_params.dir) / path)
    return scope


_ACTIVE: RequiredSkipGuard | None = None


def configure(config, omitted: Omission) -> None:
    global _ACTIVE
    if hasattr(config, "workerinput"):  # xdist worker: the controller decides
        _ACTIVE = None
        return
    _ACTIVE = RequiredSkipGuard(config.rootpath, required_test_files())
    _ACTIVE.record_uncollected(session_scope(config), omitted)


def record(report) -> None:
    if _ACTIVE is not None:
        _ACTIVE.record(report)


def finish(session) -> None:
    if _ACTIVE is None:
        return
    found = _ACTIVE.finish(session, os.environ)
    if found:
        message = "required-in-CI test files were skipped or not collected on GitHub Actions: " + ", ".join(found)
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(message, red=True, bold=True)
        else:
            print(message)
