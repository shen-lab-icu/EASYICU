"""Executable contracts for the Copilot current-results shelf and reader."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_study_results_identity_and_reader_layout() -> None:
    # frontend_contracts_ci.yml runs `node --version` on ubuntu-latest before
    # invoking the JS contracts, so Node is guaranteed in CI. A missing Node
    # here is a broken environment, not a skippable option.
    node = shutil.which("node")
    if not node:
        pytest.fail("Node is required for the study-results contract (see frontend_contracts_ci.yml)")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            node,
            str(root / "tests/js/guided_pi_study_results.test.js"),
            str(root / "src/easyicu/webserver/static/js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "guided_pi_study_results contract failed:\n"
            f"returncode={result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    assert "passed." in result.stdout, (
        "study-results harness must emit its 'passed.' receipt markers:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
