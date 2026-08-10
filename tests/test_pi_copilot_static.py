"""Frontend ownership and wiring regressions for Guided Pi Copilot."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = (
    Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"
)
NODE_APP = STATIC.parent / "pi_copilot" / "node_app"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_pi_shell_assets_are_explicitly_wired_before_guided_owner() -> None:
    index = _read("index.html")
    assert "css/guided-pi.css?v=20260810-research-workflow1" in index
    assert "css/guided-pi-preview.css?v=20260809-scientific-trust1" in index
    assert "js/screens-guided-pi-preview.js?v=20260809-scientific-trust1" in index
    assert "js/screens-guided-pi.js?v=20260810-live-pipeline1" in index
    assert "js/api.js?v=20260810-research-workflow1" in index
    assert index.index("css/guided.css") < index.index("css/guided-pi.css")
    assert index.index("js/screens-guided-pi-preview.js") < index.index(
        "js/screens-guided-pi.js"
    )
    assert index.index("js/screens-guided-pi.js") < index.index("js/screens-guided.js")


def test_pi_owner_mounts_without_moving_scientific_workflow_logic() -> None:
    guided = _read("js/screens-guided.js")
    projects_owner = _read("js/screens-guided-projects.js")
    pi_owner = _read("js/screens-guided-pi.js")
    api = _read("js/api.js")
    assert 'id="gdPiShell"' in guided
    assert 'id="gdLegacyShell"' in guided
    assert "window.EU_GUIDED_PI.mount" in guided
    assert (
        "window.EU_GUIDED_PI = { mount, unmount, setShell, bindProject, isActive }"
        in pi_owner
    )
    assert "new EventSource('/api/jobs/'" in pi_owner
    assert "external_llm_opt_in: true" in pi_owner
    assert pi_owner.count("project_id: projectId()") >= 4
    assert "loadPiCopilotSessions(30, expectedProjectId)" in pi_owner
    assert "easyicu_pi_copilot_session:' + encodeURIComponent(projectId())" in pi_owner
    assert "project_dir" not in pi_owner
    assert "window.EU_GUIDED_PI.bindProject" in guided
    assert "if (usePiSession) bindProjectToPi(result, row);" in guided
    assert "else restoreGuidedProjectThread(result, row, kind);" in guided
    assert (
        "if (piProjectShellActive()) bindProjectToPi(result, selectedGuidedDraft);"
        in guided
    )
    assert "Conversation memory" not in guided
    assert "对话记忆" not in guided
    assert "Pi keeps the conversation" in projects_owner
    assert "data-gpi-provider-form" in pi_owner
    assert "CLIProxyAPI / Local proxy" in pi_owner
    assert "gpt-5.6-luna" in pi_owner
    assert "gpt5.6 luna" not in pi_owner
    assert "anthropic-messages" in pi_owner
    assert "google-generative-ai" in pi_owner
    assert "static_preview_no_backend" in pi_owner
    assert "http://127.0.0.1:8765/#guided" in pi_owner
    assert "gpi-model-options" in pi_owner
    assert 'type="password"' in pi_owner
    assert "savePiCopilotProviderConfig" in pi_owner
    assert "provider_connection_unverified" in pi_owner
    assert "localStorage.setItem('easyicu_pi_api" not in pi_owner
    assert "keyInput.value = ''" in pi_owner
    assert 'data-gpi-grant="configure"' in pi_owner
    assert 'data-gpi-grant="idea"' in pi_owner
    assert 'data-gpi-grant="extract"' in pi_owner
    assert 'data-gpi-grant="run"' in pi_owner
    assert 'data-gpi-grant="provider_run"' in pi_owner
    assert 'data-gpi-grant="cancel"' in pi_owner
    assert 'data-gpi-grant="workspace_write"' in pi_owner
    assert 'data-gpi-resource-file' in pi_owner
    assert 'data-gpi-resource-run' in pi_owner
    assert 'data-gpi-resource-artifact' in pi_owner
    assert 'data-gpi-mode-switch="workspace"' in pi_owner
    assert "agentMode: 'research'" in pi_owner
    assert "pendingAuthorityRebind" in pi_owner
    assert "easyicu_run_submitted" in pi_owner
    assert "easyicu_full_run_submitted" in pi_owner
    assert "easyicu_extraction_submitted" in pi_owner
    assert "event.job_id" in pi_owner
    assert "watchChildJob" in pi_owner
    assert "childSource" in pi_owner
    assert "handleChildJobEvent" in pi_owner
    assert "if (state.session && sessionIsStale()) await rebind();" in pi_owner
    assert "['tool', 'pipeline', 'retry', 'compaction']" in pi_owner
    assert "Live progress connection stopped" in pi_owner
    assert "private chain-of-thought" in pi_owner
    assert "loadPiCopilotProjectWorkflow" in pi_owner
    assert "gpi-workflow" in pi_owner
    assert "Research workflow" in pi_owner
    assert "Used ${toolSteps.length} EasyICU tools" in pi_owner
    assert "gpi-activity-live" in pi_owner
    assert "completedToolLabel" in pi_owner
    assert "initializePiCopilotProject" in pi_owner
    assert "history-activity-" in pi_owner
    assert "closeHistoryActivity" in pi_owner
    assert "row.role === 'activity'" in pi_owner
    assert "gpi-avatar" not in pi_owner
    assert "private chain-of-thought" in pi_owner
    assert "assistantTextHtml" in pi_owner
    assert "row.role === 'assistant' ? assistantTextHtml(row.text) : esc(row.text)" in pi_owner
    assert "event.type === 'run_start'" in pi_owner
    assert "event.type === 'tool_progress'" in pi_owner
    assert "event.type === 'run_end'" in pi_owner
    assert "Workspace file contents may be sent to your configured Pi model service." in pi_owner
    assert "Do not place PHI, patient rows, credentials, or private clinical data" in pi_owner
    for method in (
        "loadPiCopilotStatus",
        "savePiCopilotProviderConfig",
        "createPiCopilotSession",
        "initializePiCopilotProject",
        "loadPiCopilotProjectWorkflow",
        "loadPiCopilotSessions",
        "loadPiCopilotSession",
        "sendPiCopilotMessage",
        "rebindPiCopilotSession",
        "abortPiCopilotSession",
        "loadPiCopilotWorkspaceFile",
        "piCopilotWorkspacePreviewUrl",
        "loadPiCopilotResearchArtifact",
    ):
        assert method in api
    assert "fetch(" not in pi_owner


def test_pi_css_is_route_owned_and_does_not_pollute_catch_all_files() -> None:
    owner = _read("css/guided-pi.css")
    preview_owner = _read("css/guided-pi-preview.css")
    assert ".gpi-panel" in owner
    assert ".gpi-activity" in owner
    assert ".gpi-activity-live" in owner
    assert ".gpi-activity-step-copy>span" in owner
    assert "pi-gui's MIT-licensed timeline-item/timeline.css" in owner
    assert ".gpi-message{max-width:768px" in owner
    assert ".gpi-activity,.gpi-activity-live,.gpi-activity-running{max-width:768px" in owner
    assert ".gpi-preview-aside" in preview_owner
    assert ".gpi-preview-frame" in preview_owner
    assert ".gpi-preview-code" in preview_owner
    assert ".gpi-preview-provenance" in preview_owner
    assert ".gpi-resource-list" in owner
    assert "research-artifact preview" in preview_owner
    assert ".gpi-tool" not in owner
    assert "gpi-avatar" not in owner
    assert ".gd-conv.pi-active" in owner
    assert "!important" not in owner
    assert ":has(" not in owner
    for foreign in (".patient-", ".cohort-", ".crossdb-", ".settings-", ".idea-"):
        assert foreign not in owner
        assert foreign not in preview_owner
    for relative in (
        "css/app.css",
        "css/redesign.css",
        "css/guided.css",
        "css/tweaks.css",
    ):
        assert ".gpi-" not in _read(relative)


def test_pi_chat_uses_a_scrolling_transcript_and_bottom_composer() -> None:
    owner = _read("css/guided-pi.css")
    assert ".gpi-panel{height:100%;min-height:0;display:flex" in owner
    assert "flex-direction:column;overflow:hidden" in owner
    assert ".gpi-log{flex:1 1 auto;min-height:0;overflow:auto" in owner
    assert ".gpi-compose{flex:0 0 auto" in owner
    assert "grid-template-rows:auto auto minmax(0,1fr) auto auto" not in owner
    assert (
        ".gpi-text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px" in owner
    )
    assert "font-size:15px;line-height:1.5" in owner


def test_pi_gui_adaptation_is_attributed_and_packaged() -> None:
    notice = _read("THIRD_PARTY_NOTICES.md")
    pyproject = (STATIC.parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    assert "pi-gui" in notice
    assert "Copyright (c) 2026 Matthew Lam" in notice
    assert "eb9a7380705dffad36db3efa771ee825aafbef6f" in notice
    assert '"static/THIRD_PARTY_NOTICES.md"' in pyproject


def test_pi_css_has_balanced_comments_and_braces() -> None:
    for relative in ("css/guided-pi.css", "css/guided-pi-preview.css"):
        owner = _read(relative)
        assert owner.count("/*") == owner.count("*/")
        without_comments = re.sub(r"/\*.*?\*/", "", owner, flags=re.S)
        assert without_comments.count("{") == without_comments.count("}")


def test_pi_frontend_javascript_parses() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    for relative in ("js/screens-guided-pi.js", "js/screens-guided-pi-preview.js"):
        subprocess.run(
            [node, "--check", str(STATIC / relative)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_research_artifact_renderer_rejects_attribute_xss_and_non_png_data_urls() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    subprocess.run(
        [
            node,
            str(Path(__file__).resolve().parent / "js" / "agent_render_security.test.js"),
            str(STATIC / "js" / "screens-agent-render.js"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_workspace_preview_hard_codes_unvalidated_authority_and_iframe_sandbox() -> None:
    preview = _read("js/screens-guided-pi-preview.js")
    assert "authority_class: 'workspace_artifact'" in preview
    assert "scientific_evidence: false" in preview
    assert "validation_status: 'unvalidated'" in preview
    assert "claim_ceiling: 'unsupported'" in preview
    assert "Workspace artifact · Unvalidated" in preview
    assert "scientific evidence" in preview
    assert 'sandbox="allow-scripts"' in preview
    assert 'referrerpolicy="no-referrer"' in preview
    assert "EasyICU run artifact · Analysis-only" in preview
    assert "EasyICU run artifact · Reportable" not in preview
    assert "Human sign-off required" in preview
    assert "state.governance" in preview


def test_workspace_sidecar_requires_digest_for_edit_and_teaches_safe_egress() -> None:
    sidecar = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")
    skill = (
        NODE_APP / "src" / "skills" / "web-prototype" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert sidecar.count("expected_sha256") >= 1
    assert "Create a new bounded artifact. Existing files must be changed" in sidecar
    assert "To change an existing file, read it first" in skill
    assert "expected_sha256" in skill
    assert "may be sent to the\nconfigured Pi model service" in skill
    assert "PHI" in skill


def test_workspace_security_workflow_covers_sidecar_and_browser_helper_dependencies() -> None:
    workflow = (
        STATIC.parents[3] / ".github" / "workflows" / "pi_workspace_security_ci.yml"
    ).read_text(encoding="utf-8")
    assert '"tools/qa_native_fastapi_patient_drilldown.py"' in workflow
    assert "tests/test_pi_copilot_install.py" in workflow
    assert '"src/easyicu/webserver/agent_runs.py"' in workflow
    assert '"src/easyicu/webserver/static/js/screens-agent-render.js"' in workflow
    assert '"tests/js/agent_render_security.test.js"' in workflow
    assert "node tests/js/agent_render_security.test.js" in workflow
    assert "src/easyicu/webserver/static/js/screens-agent-render.js" in workflow
    for sidecar in ("main.mjs", "event-projection.mjs", "shell-budget.mjs"):
        assert f"node --check src/easyicu/webserver/pi_copilot/node_app/src/{sidecar}" in workflow
