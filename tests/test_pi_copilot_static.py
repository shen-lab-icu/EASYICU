"""Frontend ownership and wiring regressions for Guided Pi Copilot."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_pi_shell_assets_are_explicitly_wired_before_guided_owner() -> None:
    index = _read("index.html")
    assert "css/guided-pi.css?v=20260808-pi-setup1" in index
    assert "js/screens-guided-pi.js?v=20260808-pi-setup1" in index
    assert index.index("css/guided.css") < index.index("css/guided-pi.css")
    assert index.index("js/screens-guided-pi.js") < index.index("js/screens-guided.js")


def test_pi_owner_mounts_without_moving_scientific_workflow_logic() -> None:
    guided = _read("js/screens-guided.js")
    pi_owner = _read("js/screens-guided-pi.js")
    api = _read("js/api.js")
    assert 'id="gdPiShell"' in guided
    assert 'id="gdLegacyShell"' in guided
    assert "window.EU_GUIDED_PI.mount" in guided
    assert "window.EU_GUIDED_PI = { mount, unmount, setShell }" in pi_owner
    assert "new EventSource('/api/jobs/'" in pi_owner
    assert "external_llm_opt_in: true" in pi_owner
    assert "data-gpi-provider-form" in pi_owner
    assert 'type="password"' in pi_owner
    assert "savePiCopilotProviderConfig" in pi_owner
    assert "provider_connection_unverified" in pi_owner
    assert "localStorage.setItem('easyicu_pi_api" not in pi_owner
    assert "keyInput.value = ''" in pi_owner
    assert "data-gpi-grant=\"configure\"" in pi_owner
    assert "data-gpi-grant=\"run\"" in pi_owner
    assert "data-gpi-grant=\"cancel\"" in pi_owner
    for method in (
        "loadPiCopilotStatus",
        "savePiCopilotProviderConfig",
        "createPiCopilotSession",
        "loadPiCopilotSessions",
        "loadPiCopilotSession",
        "sendPiCopilotMessage",
        "rebindPiCopilotSession",
        "abortPiCopilotSession",
    ):
        assert method in api
    assert "fetch(" not in pi_owner


def test_pi_css_is_route_owned_and_does_not_pollute_catch_all_files() -> None:
    owner = _read("css/guided-pi.css")
    assert ".gpi-panel" in owner
    assert ".gd-conv.pi-active" in owner
    assert "!important" not in owner
    assert ":has(" not in owner
    for foreign in (".patient-", ".cohort-", ".crossdb-", ".settings-", ".idea-"):
        assert foreign not in owner
    for relative in ("css/app.css", "css/redesign.css", "css/guided.css", "css/tweaks.css"):
        assert ".gpi-" not in _read(relative)


def test_pi_css_has_balanced_comments_and_braces() -> None:
    owner = _read("css/guided-pi.css")
    assert owner.count("/*") == owner.count("*/")
    without_comments = re.sub(r"/\*.*?\*/", "", owner, flags=re.S)
    assert without_comments.count("{") == without_comments.count("}")


def test_pi_frontend_javascript_parses() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    subprocess.run(
        [node, "--check", str(STATIC / "js" / "screens-guided-pi.js")],
        check=True,
        capture_output=True,
        text=True,
    )
