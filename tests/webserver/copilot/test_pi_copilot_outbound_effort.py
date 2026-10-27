"""Run the real bridge's injected-fetch effort contract when Pi is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_outbound_effort_through_both_openai_transports():
    root = Path(__file__).resolve().parents[3]
    app = root / "src/easyicu/webserver/pi_copilot/node_app"
    dependency = app / "node_modules/@earendil-works/pi-coding-agent/dist/index.js"
    node = shutil.which("node")
    if node is None or not dependency.is_file():
        pytest.skip("Pinned Pi dependencies and Node are required for outbound effort coverage")
    completed = subprocess.run(
        [node, str(root / "tests/js/pi_outbound_effort.test.js")],
        cwd=root, capture_output=True, text=True, timeout=100,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
