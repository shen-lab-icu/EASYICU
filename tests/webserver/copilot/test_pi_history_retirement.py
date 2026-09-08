"""Copilot retirement keeps governed history accessible without the Monitor shell."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_copilot_history_behavior_contract():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = Path(__file__).parents[2] / "js" / "copilot_history.test.js"
    subprocess.run([node, str(script)], check=True, capture_output=True, text=True)


def test_monitor_is_retired_but_shared_renderers_and_history_remain():
    static = Path(__file__).parents[3] / "src/easyicu/webserver/static"
    index = (static / "index.html").read_text()
    shell = (static / "js/app.js").read_text()
    assert 'src="js/screens-agent.js' not in index
    assert 'src="js/screens-agent-render.js' in index
    assert 'src="js/screens-guided-pi-history.js' in index
    assert "if (r === 'agent')" in shell
    assert "window.__euHistoryRequested = true" in shell
    assert "id: 'agent', label:" not in shell
    assert 'data-nav="agent"' not in shell
    assert 'data-gpi-history' in (static / "js/screens-guided-projects.js").read_text()
