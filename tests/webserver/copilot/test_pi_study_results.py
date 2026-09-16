"""Executable contracts for the Copilot current-results shelf and reader."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_study_results_identity_and_reader_layout() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            node,
            str(root / "tests/js/guided_pi_study_results.test.js"),
            str(root / "src/easyicu/webserver/static/js"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
