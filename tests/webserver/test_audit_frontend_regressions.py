"""Execute adversarial text and delayed-response probes at the affected JS owners."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_audit_frontend_regressions():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = (
        Path(__file__).resolve().parents[1] / "js/audit_regressions_20260915.test.js"
    )
    result = subprocess.run(
        [node, "--test", str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
