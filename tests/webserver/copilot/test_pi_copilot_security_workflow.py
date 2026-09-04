"""CI coverage contract for the Copilot workspace security owner."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC = REPO_ROOT / "src" / "easyicu" / "webserver" / "static"
NODE_APP = STATIC.parent / "pi_copilot" / "node_app"

def test_workspace_security_workflow_covers_sidecar_and_browser_helper_dependencies() -> (
    None
):
    workflow = (
        STATIC.parents[3] / ".github" / "workflows" / "pi_workspace_security_ci.yml"
    ).read_text(encoding="utf-8")
    assert '"tools/qa_native_fastapi_patient_drilldown.py"' in workflow
    assert "tests/webserver/copilot/test_pi_copilot_*.py" in workflow
    assert '"src/easyicu/webserver/agent_runs.py"' in workflow
    assert '"src/easyicu/webserver/routes/agent.py"' in workflow
    assert '"src/easyicu/webserver/static/js/screens-agent-render.js"' in workflow
    assert '"tests/js/*.test.js"' in workflow
    assert "python tools/run_js_contracts.py" in workflow
    js_runner = (STATIC.parents[3] / "tools" / "run_js_contracts.py").read_text(
        encoding="utf-8"
    )
    assert '"agent_render_security.test.js"' in js_runner
    assert '"screens-agent-render.js"' in js_runner
    assert "src/easyicu/webserver/static/js/screens-agent-render.js" in workflow
    for sidecar in ("main.mjs", "event-projection.mjs", "shell-budget.mjs"):
        assert (
            f"node --check src/easyicu/webserver/pi_copilot/node_app/src/{sidecar}"
            in workflow
        )
