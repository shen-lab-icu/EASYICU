"""The Pi setup panel must not show raw gateway readiness codes.

The panel used to render the failing keys verbatim:

    "The Pi runtime also needs attention before chat can open:
     dependency_installed, node_available"

The codes are named for the *satisfied* state, so listing them under "needs
attention" told the user the opposite of what was wrong, and nothing on the
page said how to fix any of it.
"""

from __future__ import annotations

from pathlib import Path
import re


STATIC = (
    Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"
)
PI_COPILOT = (
    Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "pi_copilot"
)


def _asset(*parts: str) -> str:
    return STATIC.joinpath(*parts).read_text(encoding="utf-8")


def test_setup_panel_delegates_blocker_copy_to_its_owner() -> None:
    panel = _asset("js", "screens-guided-pi.js")

    assert "window.EU_PI_BLOCKERS.describe(blockers, runtime)" in panel
    # The old rendering joined the raw codes straight into the sentence.
    assert "runtimeMissing.join(', ')" not in panel


def test_blocker_owner_explains_every_runtime_code_the_gateway_can_report() -> None:
    """The owner's catalog must cover service.py's `runtime_blockers` set."""

    service = (PI_COPILOT / "service.py").read_text(encoding="utf-8")
    block = re.search(
        r"runtime_blockers = \{(.*?)\}", service, re.S
    )
    assert block, "service.py should still define a runtime_blockers set"
    reported = set(re.findall(r'"([a-z_]+)"', block.group(1)))
    assert reported, "runtime_blockers set should not be empty"

    owner = _asset("js", "screens-guided-pi-blockers.js")
    described = set(re.findall(r"^    ([a-z_]+): \{$", owner, re.M))

    assert reported <= described, (
        "gateway codes with no human explanation: " f"{sorted(reported - described)}"
    )


def test_blocker_owner_pairs_each_code_with_a_remediation() -> None:
    owner = _asset("js", "screens-guided-pi-blockers.js")

    # Every catalog entry carries both a title and a fix.
    assert owner.count("title: ()") + owner.count("title: (runtime)") >= 7
    assert owner.count("fix: ()") + owner.count("fix: (runtime)") >= 7
    # The install command is the remediation for the runtime-file blockers.
    assert "easyicu-copilot-install" in owner
    # Unknown codes are passed through rather than silently dropped.
    assert "unknown.push" in owner


def test_setup_panel_keeps_the_raw_code_available_but_demoted() -> None:
    """Support conversations still need the exact code — just not as headline."""

    # The setup panel moved into its own owner when the Codex account
    # provider landed; the raw code must survive that move.
    panel = _asset("js", "screens-guided-pi-provider.js")
    css = _asset("css", "guided-pi.css")

    assert 'class="gpi-blocker-code mono"' in panel
    assert "Diagnostic code reported by the Pi runtime" in panel
    assert ".gpi-blocker-code{" in css
    assert ".gpi-config-note.gpi-blockers{display:block}" in css


def test_min_node_version_stays_in_sync_with_the_gateway() -> None:
    gateway = (PI_COPILOT / "gateway.py").read_text(encoding="utf-8")
    match = re.search(r"MIN_NODE_VERSION = \((\d+), (\d+), (\d+)\)", gateway)
    assert match, "gateway.py should still declare MIN_NODE_VERSION"
    expected = ".".join(match.groups())

    owner = _asset("js", "screens-guided-pi-blockers.js")
    assert f"const MIN_NODE = '{expected}'" in owner, (
        f"blocker copy quotes a Node version that gateway.py no longer requires "
        f"(expected {expected})"
    )
