"""Focused UX contracts for state continuity and honest Agent affordances."""

import shutil
import subprocess
from pathlib import Path

import pytest


STATIC_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
    / "js"
)


def _js(name: str) -> str:
    return (STATIC_JS / name).read_text(encoding="utf-8")


def _node_binary() -> str | None:
    direct = shutil.which("node")
    if direct:
        return direct
    candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
    return str(candidates[-1]) if candidates else None


def test_guided_global_rerender_preserves_conversation_and_composer_state() -> None:
    guided = _js("screens-guided.js")

    assert "guidedInitialRender = initializeGuidedState();" in guided
    assert "if (guidedMounted) {" in guided
    assert "if (initialRender) go('frontdoor');" in guided
    assert "captureGuidedComposerDraft();" in guided
    assert 'value="${attr(guidedComposerDraft)}"' in guided
    assert "input.addEventListener('input'" in guided

    render_body = guided.split("S.guided = {", 1)[1].split("afterRender(root)", 1)[0]
    assert "reset();" not in render_body


def test_language_switch_flushes_guided_slots_before_global_rerender() -> None:
    guided = _js("screens-guided.js")
    i18n = _js("i18n.js")

    assert "window.__euGuidedBeforeLanguageRerender = function ()" in guided
    assert "flushGuidedSlotSave('language_change')" in guided
    assert "clearTimeout(guidedSlotSaveTimer);" in guided
    assert i18n.index("window.__euGuidedBeforeLanguageRerender()") < i18n.index(
        "if (window.__euRender) window.__euRender();"
    )


def test_agent_provider_affordance_names_the_scaffold_and_has_no_fake_exports() -> None:
    agent = _js("screens-agent.js")

    assert "Generate provider scaffold" in agent
    assert "it does not run a complete research analysis" in agent
    # The dock-opening affordance moved to the shared topbar 'Page guide'
    # button; the agent screen must not ship its own duplicate opener.
    assert "data-cpopen" not in agent
    assert "Run full with provider" not in agent
    assert "A full agent analysis is a separate step" not in agent
    assert "t('Export ledger'" not in agent
    assert "t('Export notes'" not in agent


def test_guided_blocked_gate_never_uses_the_success_findings_path() -> None:
    guided = _js("screens-guided.js")

    complete = guided.split("function completeLivePipeline", 1)[1].split(
        "function failLivePipeline", 1
    )[0]
    blocked = complete.split("if (gateState.blocked)", 1)[1].split(
        "setVal({ analysis: () => analysisLine()", 1
    )[0]
    assert "outputsReady = !gateState.blocked" in complete
    assert 'class="pill bad"' in blocked
    assert "Verification blocked" in blocked
    assert "review_blocked" in blocked
    assert "@reviewBlocked" in blocked
    assert "@openAgent" in blocked
    assert "return;" in blocked
    assert "go('toFindings')" not in blocked

    assert "Agent preflight blocked" in guided
    assert "guidedAgent.error || gateBlocked ? 'bad'" in guided
    assert "guidedGateState(guidedAgent.result).blocked" in guided
    assert "setVal({ analysis: 'verification blocked', draft: 'locked · review_blocked' })" in guided
    assert "pending ? 'pending' : 'failed'" in guided
    assert "pending ? 'pending' : 'bad'" in guided


def test_guided_gate_contract_fails_closed_for_missing_or_unknown_results() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [
            node,
            str(Path(__file__).parents[1] / "tests" / "js" / "guided_gate_state.test.js"),
            str(STATIC_JS / "screens-guided.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert '"ok":true' in result.stdout
