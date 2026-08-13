"""Frontend ownership and wiring regressions for Guided Pi Copilot."""

from __future__ import annotations

import hashlib
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
    assert "css/guided-pi.css?v=20260812-natural-chat-artifacts1" in index
    assert "css/guided-pi-demo.css?v=20260811-product-demo5" in index
    assert "css/guided-pi-preview.css?v=20260811-research-docs1" in index
    assert "css/guided-pi-workbench-preview.css?v=20260813-workbench1" in index
    assert "css/guided-pi-literature.css?v=20260812-literature3" in index
    assert "js/screens-guided-pi-literature.js?v=20260812-literature3" in index
    assert "js/screens-guided-pi-markdown.js?v=20260811-message-links1" in index
    assert "js/screens-guided-pi-demo.js?v=20260811-science-readiness1" in index
    assert "js/screens-guided-pi-workbench-preview.js?v=20260813-workbench1" in index
    assert "js/screens-guided-pi-preview.js?v=20260812-data-package1" in index
    assert "js/screens-guided-pi-replay.js?v=20260813-review-replay2" in index
    assert "js/screens-guided-pi.js?v=20260813-review-replay2" in index
    assert (
        "js/screens-guided-project-continuity.js?v=20260813-project-continuity1"
        in index
    )
    assert "js/api.js?v=20260812-extension-manager1" in index
    assert index.index("css/guided.css") < index.index("css/guided-pi.css")
    assert index.index("js/screens-guided-pi-literature.js") < index.index(
        "js/screens-guided-pi-markdown.js"
    )
    assert index.index("js/screens-guided-pi-markdown.js") < index.index(
        "js/screens-guided-pi-demo.js"
    )
    assert index.index("js/screens-guided-pi-demo.js") < index.index(
        "js/screens-guided-pi-workbench-preview.js"
    )
    assert index.index("js/screens-guided-pi-workbench-preview.js") < index.index(
        "js/screens-guided-pi-preview.js"
    )
    assert index.index("js/screens-guided-pi-preview.js") < index.index(
        "js/screens-guided-pi.js"
    )
    assert index.index("js/screens-guided-pi.js") < index.index("js/screens-guided.js")


def test_guided_project_refresh_continuity_has_a_small_dedicated_owner() -> None:
    index = _read("index.html")
    continuity = _read("js/screens-guided-project-continuity.js")
    guided = _read("js/screens-guided.js")

    assert index.index("js/screens-guided-project-continuity.js") < index.index(
        "js/screens-guided.js"
    )
    assert "easyicu_guided_active_project:v1" in continuity
    assert "window.EU_GUIDED_PROJECT_CONTINUITY" in continuity
    assert "project_dir" not in continuity
    assert "EU_GUIDED_PROJECT_CONTINUITY.remember" in guided
    assert "continuity.remembered()" in guided
    assert "openGuidedProjectMemory(rememberedRow, null, 'draft')" in guided


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
    assert "syncProjectWorkflowAside" in pi_owner
    assert "completed_required_stages" in pi_owner
    assert "operator_plan_approval_required" in pi_owner
    assert (
        "I approve this exact evidence-bound plan without changing the study configuration"
        in pi_owner
    )
    assert "本轮不新增可选的科学设定" in pi_owner
    assert "preserve every open scientific finding as a limitation" in pi_owner
    assert "reviewResources" in pi_owner
    assert "打开分析计划" in pi_owner
    assert "打开文献绑定" in pi_owner
    assert "gpi-confirmation-resources" in pi_owner
    assert "stage.status === 'review_required'" in pi_owner
    assert "data-gpi-project-workflow-aside" in pi_owner
    assert "pi_model_provider_unavailable" in pi_owner
    assert "pi_shell_token_budget_exhausted" in pi_owner
    assert "Research Agent 规划任务已提交" in pi_owner
    assert "EasyICU 完整科研分析已提交" not in pi_owner
    assert "同一研究项目中新建后续对话" in pi_owner
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
    assert "ACCESS_MODE_GRANTS" in pi_owner
    assert "data-gpi-access-mode" in pi_owner
    assert "Ask first" in pi_owner
    assert "Auto-approve" in pi_owner
    assert "Full access" in pi_owner
    assert "data-gpi-grant" not in pi_owner
    assert "data-gpi-resource-file" in pi_owner
    assert "data-gpi-resource-run" in pi_owner
    assert "data-gpi-resource-artifact" in pi_owner
    assert 'data-gpi-mode-switch="workspace"' in pi_owner
    assert "agentMode: 'research'" in pi_owner
    assert "pendingAuthorityRebind" in pi_owner
    assert "event.host_rebind_after_turn === true" in pi_owner
    assert "easyicu_run_submitted" in pi_owner
    assert "easyicu_full_run_submitted" in pi_owner
    assert "easyicu_extraction_submitted" in pi_owner
    assert "event.job_id" in pi_owner
    assert "watchChildJob" in pi_owner
    assert "childSource" in pi_owner
    assert "handleChildJobEvent" in pi_owner
    assert "if (state.session && sessionIsStale()) await rebind();" in pi_owner
    assert "function reconcileSettledSession()" in pi_owner
    assert "state.session.streaming !== false" in pi_owner
    assert pi_owner.count("reconcileSettledSession();") == 2
    assert (
        "['submitted', 'agent', 'turn', 'assistant', 'tool', 'pipeline', 'retry', 'compaction']"
        in pi_owner
    )
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
    assert (
        "row.role === 'assistant' ? assistantTextHtml(row.text) : esc(row.text)"
        in pi_owner
    )
    assert "event.type === 'run_start'" in pi_owner
    assert "event.type === 'tool_progress'" in pi_owner
    assert "event.type === 'run_end'" in pi_owner
    assert "workspace file contents may be sent to this configured service" in pi_owner
    assert (
        "Do not place PHI, patient rows, credentials, or private clinical data"
        in pi_owner
    )
    assert "data-gpi-confirm-action" in pi_owner
    assert "data-gpi-demo" in pi_owner
    assert "data-gpi-demo-exit" in pi_owner
    assert "查看完整科研流程演示" in pi_owner
    assert "state.demoMode ? demoPanel()" in pi_owner
    assert "extraction_ready" in pi_owner
    assert "plan_ready" in pi_owner
    assert "provider_plan_ready" in pi_owner
    assert "生成 Agent 计划" in pi_owner
    assert "plan_configuration_superseded" in pi_owner
    assert "重新生成计划" in pi_owner
    assert "operator_plan_approval_required" in pi_owner
    assert "hydrateProjectedJob" in pi_owner
    assert "visibleSteps.length} steps" in pi_owner
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
        "pinPiCopilotPresentation",
        "archivePiCopilotChildJob",
        "abortPiCopilotSession",
        "loadPiCopilotWorkspaceFile",
        "piCopilotWorkspacePreviewUrl",
        "loadPiCopilotResearchArtifact",
        "loadPiCopilotDataPackageReview",
    ):
        assert method in api
    assert "fetch(" not in pi_owner


def test_existing_project_study_setup_stays_in_bound_pi_conversation() -> None:
    owner = _read("js/screens-guided-pi.js")

    assert "data-gpi-study-setup" in owner
    assert "function studySetupReviewPrompt(workflow)" in owner
    assert "function openStudySetupInConversation()" in owner
    assert "setShell('pi')" in owner
    assert "workflow && workflow.study_setup_receipt" in owner
    assert "workflow && workflow.missing_setup_fields" in owner
    assert "Preserve study_context_id and revision" in owner
    assert "const prompt = studySetupReviewPrompt(state.workflow)" in owner
    assert "sendText(prompt, ['configure'])" in owner
    assert "event.target.closest('[data-gpi-study-setup]')" in owner
    assert "openStudySetupInConversation();" in owner
    assert ".then(() => { if (projectId() === next.id) render(); })" in owner
    assert "legacy 0/8 aside" in owner
    assert "data-gpi-project-workflow-loading" in owner
    assert "Loading authoritative configuration…" in owner
    assert "if (!workflow)" in owner
    session_panel = owner[
        owner.index("function sessionPanel()") : owner.index("function demoPanel()")
    ]
    assert "data-gpi-legacy" not in session_panel


def test_scientific_review_continues_as_one_question_in_chat() -> None:
    owner = _read("js/screens-guided-pi.js")

    assert "Answer next scientific question" in owner
    assert "回答下一个科学问题" in owner
    assert "review.authorization_questions" in owner
    assert "questions[0].question" in owner
    assert "请一次只问我一个尚未解决的科学设定问题" in owner


def test_agent_handoff_receipt_is_forwarded_to_project_initialization() -> None:
    owner = _read("js/screens-guided-pi.js")
    guided = _read("js/screens-guided.js")

    assert "binding_receipt: state.project.binding_receipt || undefined" in owner
    assert "binding_receipt: project.binding_receipt || null" in owner
    assert (
        "study_context_id: guidedBinding.binding_receipt && guidedBinding.binding_receipt.study_context_id"
        in guided
    )
    assert (
        "study_context_revision: guidedBinding.binding_receipt && guidedBinding.binding_receipt.study_context_revision"
        in guided
    )


def test_agent_handoff_project_remains_visible_without_a_guided_folder() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [
            node,
            str(
                Path(__file__).resolve().parent
                / "js"
                / "guided_project_handoff.test.js"
            ),
            str(STATIC / "js" / "screens-guided-projects.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout == '{"bound":true,"unbound":true,"empty":true}'


def test_pi_css_is_route_owned_and_does_not_pollute_catch_all_files() -> None:
    owner = _read("css/guided-pi.css")
    preview_owner = _read("css/guided-pi-preview.css")
    literature_owner = _read("css/guided-pi-literature.css")
    demo_owner = _read("css/guided-pi-demo.css")
    assert ".gpi-panel" in owner
    assert ".gpi-activity" in owner
    assert ".gpi-activity-live" in owner
    assert ".gpi-activity-step-copy>span" in owner
    assert ".gpi-access-menu" in owner
    assert ".gpi-confirmation" in owner
    assert ".gpi-grants" not in owner
    assert "grid-template-columns:repeat(8,minmax(0,1fr))" in owner
    assert "pi-gui's MIT-licensed timeline-item/timeline.css" in owner
    assert ".gpi-message{max-width:768px" in owner
    assert (
        ".gpi-activity,.gpi-activity-live,.gpi-activity-running{max-width:768px"
        in owner
    )
    assert ".gpi-preview-aside" in preview_owner
    assert ".gpi-preview-frame" in preview_owner
    assert ".gpi-preview-code" in preview_owner
    assert ".gpi-preview-provenance" in preview_owner
    assert ".gpi-resource-list" in owner
    assert ".gpi-lit-card" in literature_owner
    assert ".gpi-lit-step" in literature_owner
    assert ".gpi-demo-note" in demo_owner
    assert ".gpi-demo-footer" in demo_owner
    assert ".gpi-demo-artifact" in demo_owner
    assert "research-artifact preview" in preview_owner
    assert ".gpi-tool" not in owner
    assert "gpi-avatar" not in owner
    assert ".gd-conv.pi-active" in owner
    assert "!important" not in owner
    assert ":has(" not in owner
    for foreign in (".patient-", ".cohort-", ".crossdb-", ".settings-", ".idea-"):
        assert foreign not in owner
        assert foreign not in preview_owner
        assert foreign not in literature_owner
        assert foreign not in demo_owner
    for relative in (
        "css/app.css",
        "css/redesign.css",
        "css/guided.css",
        "css/tweaks.css",
    ):
        assert ".gpi-" not in _read(relative)


def test_scientific_plan_review_has_a_readable_multidimensional_preview() -> None:
    renderer = _read("js/screens-agent-render.js")

    assert "scientific_plan_review.json" in renderer
    assert "Top-journal plan scorecard" in renderer
    assert "Required changes before analysis" in renderer
    assert "What each article actually contributes to the plan" in renderer
    assert "literature_design_bindings" in renderer
    assert "Rendered figures assessed" in renderer
    assert "score_interpretation" in renderer
    pi_owner = _read("js/screens-guided-pi.js")
    assert "plan_review_summary" in pi_owner
    assert "gpi-confirmation-scorecard" in pi_owner
    assert "authorization_questions" in pi_owner
    assert "remediation_buckets" in pi_owner
    assert "Agent can repair in a fresh plan" in pi_owner


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
    for relative in (
        "css/guided-pi.css",
        "css/guided-pi-preview.css",
        "css/guided-pi-workbench-preview.css",
        "css/guided-pi-literature.css",
        "css/guided-pi-demo.css",
    ):
        owner = _read(relative)
        assert owner.count("/*") == owner.count("*/")
        without_comments = re.sub(r"/\*.*?\*/", "", owner, flags=re.S)
        assert without_comments.count("{") == without_comments.count("}")


def test_pi_frontend_javascript_parses() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    for relative in (
        "js/screens-guided-pi.js",
        "js/screens-guided-pi-preview.js",
        "js/screens-guided-pi-workbench-preview.js",
        "js/screens-guided-pi-literature.js",
        "js/screens-guided-pi-markdown.js",
        "js/screens-guided-pi-demo.js",
        "js/screens-guided-pi-replay.js",
    ):
        subprocess.run(
            [node, "--check", str(STATIC / relative)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_pi_project_reopens_latest_session_and_replays_safe_lifecycle() -> None:
    owner = _read("js/screens-guided-pi.js")
    replay = _read("js/screens-guided-pi-replay.js")
    assert "state.session.active_message_job_id" in owner
    assert "watchJob(activeMessageJob)" in owner
    assert "await openSession(state.sessions[0].session_id)" in owner
    assert "session.last_turn_events" in replay
    assert "next_cursor" in replay
    assert "saved-activity-" in owner
    assert "state.session.archived_child_jobs" in owner
    assert "archiveChildJob(jobId)" in owner
    assert "childJobPresentation" in replay
    assert "Analysis plan ready for review" in replay
    assert "activity.displayTitle" in owner
    assert "row.durationKnown === false" in owner
    assert "data-gpi-presentation-pin" in owner
    assert "pinPiCopilotPresentation" in owner
    assert "private chain-of-thought" in owner


def test_data_package_opens_in_a_route_owned_read_only_workbench() -> None:
    preview = _read("js/screens-guided-pi-preview.js")
    workbench = _read("js/screens-guided-pi-workbench-preview.js")
    css = _read("css/guided-pi-workbench-preview.css")
    assert 'data-gpi-preview-mode="workbench"' in preview
    assert "window.EU_GUIDED_PI_WORKBENCH_PREVIEW" in workbench
    assert "data-gpi-wb-query" in workbench
    assert "data-gpi-wb-status" in workbench
    assert "typed proposal" in workbench
    assert "effect estimates" in workbench
    assert ".gpi-wb" in css
    assert ".patient-" not in css
    assert ".cohort-" not in css
    assert ".crossdb-" not in css


def test_complete_research_demo_is_natural_truthful_and_clickable() -> None:
    demo = _read("js/screens-guided-pi-demo.js")
    pi_owner = _read("js/screens-guided-pi.js")
    preview = _read("js/screens-guided-pi-preview.js")
    assert "帮我从 MIMIC-IV 里找一个关于早期脓毒症和院内死亡" in demo
    assert "把第 2 个作为工程验证案例继续，使用 MIMIC-IV。" in demo
    assert "可以，继续提取数据。" in demo
    assert "只继续工程验证，所有科学与投稿论断保持阻断。" in demo
    assert "帮我解读结果，并整理成论文初稿。" in demo
    assert "继续 Web E1 工程 UAT" not in demo
    assert "run_type='full'" not in demo
    assert "11/11 · 绑定证据并核验数值" in demo
    assert "53 / 140 (37.9%)" in demo
    assert "8 / 53 (15.1%)" in demo
    assert "7 / 87 (8.0%)" in demo
    assert "1.50 (95% CI 0.49–4.60)" in demo
    assert "run_20260811T030843_4d45a8" in demo
    assert "engineering_canary_demo_only" in demo
    assert "reportable: false" in demo
    assert "不是正式论文证据" in demo
    assert "https://pubmed.ncbi.nlm.nih.gov/26903338/" in demo
    assert "kind: 'demo_artifact'" in demo
    assert "title: item ? item.title : name" in demo
    assert "resource.kind === 'demo_artifact'" in pi_owner
    assert "resource.title || resourceLabel(resource)" in pi_owner
    assert "value.kind === 'demo_artifact'" in preview
    assert "Product demo · Real engineering-canary aggregate" in preview
    assert "Historical run plus independent current audit" in preview
    assert "state.payload && state.payload.source_authority" in preview
    assert "safe.kind !== 'demo_artifact'" in preview
    assert "literature_evidence.json" in demo
    assert "scientific_readiness.json" in demo
    assert "result_tables.json" in demo
    assert "figure_gallery.json" in demo
    assert "打开历史与当前文献审计" in demo
    assert "search_conducted: false" in demo
    assert "curated_seed_count: 9" in demo
    assert "检出'), value: '14'" not in demo
    assert "去重'), value: '5'" not in demo
    assert "检索在制定计划前完成" not in demo
    assert "SOFA-2 versus SOFA-1 for mortality prediction" in demo
    assert "https://pubmed.ncbi.nlm.nih.gov/41877184/" in demo
    assert "SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN" in demo
    assert "PAPER_AUTHORITY_NOT_GRANTED" in demo
    assert demo.count("step_id: '") == 11
    assert "citation_keys:" in demo
    assert "projection_note:" in demo
    assert "该历史 canary 早于当前运行" in demo
    assert "e1-publication-figure.png" in demo
    assert "required_stage_count: 8" in demo
    assert "实验性 SOFA-2 敏感性指标" in demo
    assert "不是标准 Sepsis-3" in demo
    assert "Sepsis-3 indicator present" not in demo
    assert "tr('Sepsis-3 prevalence', 'Sepsis-3 比例')" not in demo
    assert "Table 1 · Baseline characteristics by Sepsis-3 indicator" not in demo
    assert "operator_plan_approved" in pi_owner
    assert "validated_analysis_complete" in pi_owner
    assert "validated_analysis_ready" in pi_owner
    assert "interpretation_complete" in pi_owner
    assert "evidence_bound_interpretation_ready" in pi_owner
    assert "manuscript_draft_ready_for_review" in pi_owner
    assert "human_review_required" in pi_owner
    assert "prior_art_authority_not_established" in pi_owner
    assert "source_population_scope_open" in pi_owner
    assert "publication_analysis_incomplete" in pi_owner
    assert "paper_authority_not_granted" in pi_owner


def test_guided_copilot_does_not_expose_benchmark_specific_navigation() -> None:
    guided_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC / "js").glob("screens-guided*.js"))
    )

    assert "Canonical9" not in guided_sources
    assert "打开 Canonical9 九题独立对话" not in guided_sources


def test_pi_messages_project_governed_tool_artifacts_beside_the_answer() -> None:
    owner = _read("js/screens-guided-pi.js")
    css = _read("css/guided-pi.css")

    assert "turnResources" in owner
    assert "currentTurnResources" in owner
    assert "gpi-message-resources" in owner
    assert "Referenced run artifacts" in owner
    assert "result_tables.json" in owner
    assert "figure_gallery.json" in owner
    assert ".gpi-message-resources" in css


def test_complete_research_demo_reuses_the_unchanged_agent_figure() -> None:
    figure = STATIC / "assets" / "demo" / "e1-publication-figure.png"
    assert figure.is_file()
    assert figure.stat().st_size == 93_214
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == (
        "34a46b54558a6f08cc02434a6958558ecb8077abd59db78713ef8f9dd4172e4b"
    )


def test_complete_research_demo_renderer_escapes_untrusted_values() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-demo.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({source!r});
      console.log(window.EU_GUIDED_PI_DEMO.renderArtifact({{
        title: '<img src=x onerror=alert(1)>',
        summary: '<script>alert(2)</script>',
        metrics: [{{ label: '<b>label</b>', value: 'javascript:alert(3)' }}],
        sections: [{{ heading: '<svg onload=alert(4)>', items: ['<iframe>'] }}],
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "<img" not in completed.stdout
    assert "<script>" not in completed.stdout
    assert "<svg" not in completed.stdout
    assert "<iframe" not in completed.stdout
    assert "&lt;img" in completed.stdout
    assert "&lt;script" in completed.stdout


def test_research_artifact_renderer_rejects_attribute_xss_and_non_png_data_urls() -> (
    None
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    subprocess.run(
        [
            node,
            str(
                Path(__file__).resolve().parent / "js" / "agent_render_security.test.js"
            ),
            str(STATIC / "js" / "screens-agent-render.js"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_literature_renderer_escapes_metadata_and_rejects_unsafe_links() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({source!r});
      const html = window.EU_GUIDED_PI_LITERATURE.renderArtifact({{
        search: {{ search_conducted: true }},
        citations: [{{ key: 'safe_key', title: '<img src=x onerror=alert(1)>',
          source_url: 'javascript:alert(1)', relevance: '<script>alert(2)</script>' }},
          {{ key: 'linked_key', title: 'Linked design paper',
             source_url: 'https://pubmed.ncbi.nlm.nih.gov/12345/' }}],
        plan_step_count: 1,
        mapped_step_count: 1,
        step_citation_map: [{{ step_id: 'primary', intent: 'Primary analysis',
          citation_keys: ['linked_key'] }}],
      }});
      console.log(html);
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "<img" not in completed.stdout
    assert "<script>" not in completed.stdout
    assert 'href="javascript:' not in completed.stdout
    assert "&lt;img" in completed.stdout
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/12345/"' in completed.stdout
    assert 'rel="noopener noreferrer"' in completed.stdout


def test_assistant_message_renderer_makes_https_citations_clickable_and_safe() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    literature = _read("js/screens-guided-pi-literature.js")
    markdown = _read("js/screens-guided-pi-markdown.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({literature!r});
      eval({markdown!r});
      console.log(window.EU_GUIDED_PI_MARKDOWN.render(
        '[PMID: 26903338](https://pubmed.ncbi.nlm.nih.gov/26903338/)\\n' +
        '[unsafe](javascript:alert(1)) **strong** *journal* <script>alert(2)</script>'
      ));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/26903338/"' in completed.stdout
    assert 'target="_blank"' in completed.stdout
    assert 'rel="noopener noreferrer"' in completed.stdout
    assert 'href="javascript:' not in completed.stdout
    assert "<script>" not in completed.stdout
    assert "&lt;script&gt;" in completed.stdout
    assert "<strong>strong</strong>" in completed.stdout
    assert "<em>journal</em>" in completed.stdout


def test_literature_preview_distinguishes_auxiliary_steps_from_scientific_gaps() -> (
    None
):
    literature_owner = _read("js/screens-guided-pi-literature.js")

    assert "个科学决策已绑定" in literature_owner
    assert "不计作文献决策缺口" in literature_owner
    assert "该科学决策没有绑定文献，需要审阅" in literature_owner


def test_workspace_preview_hard_codes_unvalidated_authority_and_iframe_sandbox() -> (
    None
):
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
    skill = (NODE_APP / "src" / "skills" / "web-prototype" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert sidecar.count("expected_sha256") >= 1
    assert "Create a new bounded artifact. Existing files must be changed" in sidecar
    assert "To change an existing file, read it first" in skill
    assert "expected_sha256" in skill
    assert "may be sent to the\nconfigured Pi model service" in skill
    assert "PHI" in skill
    assert "llm_provider:" not in sidecar


def test_workspace_security_workflow_covers_sidecar_and_browser_helper_dependencies() -> (
    None
):
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
        assert (
            f"node --check src/easyicu/webserver/pi_copilot/node_app/src/{sidecar}"
            in workflow
        )
