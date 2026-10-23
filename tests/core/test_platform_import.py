"""No ASGI transport: import the full Web application on the Windows branch."""

import os
from pathlib import Path
import subprocess
import sys


def test_windows_app_import_and_evidence_roundtrip(tmp_path):
    script = Path(__file__).resolve().parents[1] / "support" / "windows_platform.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "EASYICU_HOME": str(tmp_path), "EASYICU_HOSTED": "0"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "windows import, evidence and locks: ok" in result.stdout
