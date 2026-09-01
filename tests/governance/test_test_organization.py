"""Test-suite ownership conventions."""

from __future__ import annotations

import re
from pathlib import Path


DATE_NAMED_TEST = re.compile(r"(?:^|_)20\d{6}(?:_|\.py$)")


def test_regression_files_use_functional_owner_names() -> None:
    """Review dates belong in docstrings, not in test module ownership."""

    tests_root = Path(__file__).resolve().parent
    offenders = [
        path.relative_to(tests_root).as_posix()
        for path in tests_root.rglob("test_*.py")
        if DATE_NAMED_TEST.search(path.name)
    ]

    assert offenders == []
