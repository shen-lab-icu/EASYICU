from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from easyicu import concept_catalog as cc
from easyicu.webserver import dataio
from easyicu.webserver.app import app
from easyicu.webserver.routes.request_parsing import body_bool as _body_bool

STATIC_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"
)


def _static_js(name: str) -> str:
    return (STATIC_DIR / "js" / name).read_text(encoding="utf-8")


def _static_css(name: str) -> str:
    return (STATIC_DIR / "css" / name).read_text(encoding="utf-8")


def _static_html(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_native_favicon_request_is_quiet() -> None:
    response = TestClient(app).get("/favicon.ico")

    assert response.status_code == 204


def test_copilot_demo_serves_the_agent_produced_publication_figure() -> None:
    response = TestClient(app).get("/assets/demo/e1-publication-figure.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) == 93_214


def test_native_fs_mkdir_creates_local_export_folder(tmp_path: Path) -> None:
    target = tmp_path / "exports" / "new parent"
    client = TestClient(app)

    response = client.post("/api/fs/mkdir", json={"path": str(target)})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["created"] is True
    assert Path(body["path"]).is_dir()

    duplicate = client.post("/api/fs/mkdir", json={"path": str(target)})
    assert duplicate.status_code == 200
    assert duplicate.json()["created"] is False


def test_native_static_route_registry_contains_fallback_only_routes() -> None:
    screen_ids: set[str] = set()
    for path in (STATIC_DIR / "js").glob("screens-*.js"):
        screen_ids.update(
            re.findall(r"\bS\.([a-zA-Z0-9_]+)\s*=", path.read_text(encoding="utf-8"))
        )

    assert {
        "entry",
        "extraction",
        "patient",
        "cohort",
        "crossdb",
        "agent",
        "ideas",
        "settings",
        "dictionary",
        "states",
        "tutorial",
        "guided",
    } <= screen_ids


def test_native_hash_router_has_help_alias_and_unknown_hash_fails_safe() -> None:
    app_js = _static_js("app.js")

    assert "const FALLBACK_ROUTE = 'guided';" in app_js
    assert "if (raw === 'entry')" in app_js
    assert "replaceHash('guided')" in app_js
    assert "if (r === 'help') return 'tutorial';" in app_js
    assert "history.replaceState(null, '', next)" in app_js
    assert "resolveRoute(rawRouteFromHash(), { rewrite: true })" in app_js
    assert "resolved.fallback" in app_js


def test_native_shell_language_icon_is_stateful() -> None:
    app_js = _static_js("app.js")
    api_js = _static_js("api.js")
    i18n_js = _static_js("i18n.js")
    dock_js = _static_js("copilot-dock.js")
    settings_js = _static_js("screens-settings.js")
    index_html = _static_html("index.html")

    assert "data-lang-toggle" in app_js
    assert "window.EU_LANG === 'zh' ? 'en' : 'zh'" in app_js
    assert "localStorage.getItem('easyicu_lang')" in api_js
    assert "localStorage.getItem('easyicu_home_data')" in api_js
    assert "syncPatch.language = browserLang" in api_js
    assert "syncPatch.data_mode = browserMode" in api_js
    assert "window.applySettingsState(window.EU_SETTINGS" in api_js
    assert "window.applySettingsState = function" in i18n_js
    assert "const previousLang = window.EU_LANG" in i18n_js
    assert "opts.notifyLanguage" in i18n_js
    assert "document.body.setAttribute('data-density', density)" in i18n_js
    assert "document.body.setAttribute('data-reduce-motion'" in i18n_js
    assert "window.EU_SETTINGS.language = l" in i18n_js
    assert "window.EU_API.saveSetting('language', l)" in i18n_js
    assert "window.dispatchEvent(new CustomEvent('easyicu:languagechange'" in i18n_js
    assert "window.EUPageGuide = { open, close, toggle, refreshLanguage }" in dock_js
    assert "window.setLang(val);" in settings_js
    assert "window.EU_LANG = val;" not in settings_js
    assert "window.EU_API.saveSetting('data_mode', m)" in i18n_js
    assert "js/i18n.js?v=20260728-demo-mode1" in index_html
    assert "js/api.js?v=20260828-plan-review1" in index_html


def test_floating_copilot_launcher_is_removed_but_shell_hooks_survive() -> None:
    """The launcher is gone; the shell entry points that used it are not.

    The floating #cpFab sat on top of the composer's own send control on the
    guided route, and the route it opened is the page the user is already on.
    app.js still opens the one #guided conversation from [data-cpopen] and
    Cmd/Ctrl+K through EUPageGuide, so the hook must outlive the button.
    """

    dock_css = _static_css("dock.css")
    dock_js = _static_js("copilot-dock.js")
    app_js = _static_js("app.js")

    assert "#cpFab" not in dock_css
    assert "cpFab" not in dock_js
    assert "document.body.appendChild(fab)" not in dock_js
    assert "window.EUPageGuide = { open, close, toggle, refreshLanguage }" in dock_js
    assert "window.EUCopilot = window.EUPageGuide" in dock_js
    assert "window.EUPageGuide || window.EUCopilot" in app_js
    assert "data-cpopen" in app_js


def test_native_assistant_labels_expose_one_primary_copilot_conversation() -> (
    None
):
    app_js = _static_js("app.js")
    dock_js = _static_js("copilot-dock.js")
    extraction_js = _static_js("screens-extraction.js")
    agent_js = _static_js("screens-agent.js")
    help_js = _static_js("screens-help.js")
    index_html = _static_html("index.html")

    assert "EasyICU Copilot" in app_js
    assert "打开唯一的 EasyICU 研究助手对话" in app_js
    assert "window.EUPageGuide || window.EUCopilot" in app_js
    assert "t('EasyICU Copilot', 'EasyICU 研究助手')" in app_js
    # All shell affordances open the one #guided Pi conversation. Project
    # Monitor must not add an agent-specific conversation opener.
    assert "Agent guide" not in agent_js
    assert "data-cpopen" not in agent_js
    assert "function open()" in dock_js
    assert "location.hash = '#guided'" in dock_js
    assert "The historical page-guide dock intentionally is not constructed" in dock_js
    assert "createPageGuideSession" not in dock_js
    assert "sendPageGuideMessage" not in dock_js
    assert "document.body.appendChild(dock)" not in dock_js
    assert "Start Guided Copilot" in extraction_js
    assert "Cancel accepted. Stopping the current database query" in extraction_js
    assert "Stopping extraction" in extraction_js
    assert "当前数据库读取可能会先完成" not in extraction_js
    assert "Continue in Guided Copilot" in agent_js
    assert "Open EasyICU Copilot" in help_js

    assert "Quick help" not in app_js
    assert "Quick help" not in dock_js
    assert "Quick help" not in help_js
    assert 'title="Ask the Copilot"' not in app_js
    assert "${t('Copilot','助手')}" not in app_js
    assert "Open Guided Copilot" not in dock_js
    assert '<div class="cp-name">Copilot</div>' not in dock_js
    assert "Let Copilot drive" not in extraction_js
    assert "Continue in Copilot" not in agent_js
    assert "Open Copilot" not in help_js

    assert "css/dock.css?v=20260827-no-fab1" in index_html
    assert "js/app.js?v=20260826-copilot-home1" in index_html
    assert "js/copilot-dock.js?v=20260827-no-fab1" in index_html
    assert "js/screens-extraction.js?v=20260825-source-binding1" in index_html
    assert "js/screens-agent.js?v=20260823-run-history-authority1" in index_html
    assert "js/screens-help.js?v=20260817-copilot-boundary1" in index_html


def test_project_monitor_run_history_has_a_dedicated_projection_owner() -> None:
    index_html = _static_html("index.html")
    monitor_js = _static_js("screens-agent.js")
    history_js = _static_js("screens-agent-run-history.js")

    owner_asset = "js/screens-agent-run-history.js?v=20260823-run-history-owner1"
    monitor_asset = "js/screens-agent.js?v=20260823-run-history-authority1"
    assert owner_asset in index_html
    assert index_html.index(owner_asset) < index_html.index(monitor_asset)
    assert "window.EU_AGENT_RUN_HISTORY_VIEW" in history_js
    assert "const RUN_HISTORY_VIEW = window.EU_AGENT_RUN_HISTORY_VIEW" in monitor_js
    assert "function historyRunForStudy" not in monitor_js


def test_agent_science_workbench_has_dedicated_owner_files_and_wiring() -> None:
    index_html = _static_html("index.html")
    api_js = _static_js("api.js")
    agent_js = _static_js("screens-agent.js")
    science_js = _static_js("screens-agent-science.js")
    science_css = _static_css("agent-science.css")
    science_detail_css = _static_css("agent-science-detail.css")
    screens_css = _static_css("screens.css")
    redesign_css = _static_css("redesign.css")

    assert "css/agent-science.css?v=20260707-evidence-merge" in index_html
    assert "css/agent-science-detail.css?v=20260702-science-workbench-v7" in index_html
    assert "js/screens-agent-science.js?v=20260707-design" in index_html
    assert "/api/agent-runs/science-workbench" in api_js
    assert "loadAgentScienceWorkbench" in api_js
    assert "/api/capabilities" in api_js
    assert "hydrateCapabilities" in api_js
    assert "checkCapabilityTool" in api_js
    assert "searchZotero" in api_js
    assert "loadCapabilityAuditEvents" in api_js
    assert "window.EU_AGENT_SCIENCE.render" in agent_js
    assert "window.EU_AGENT_SCIENCE.wire" in agent_js

    assert "window.EU_AGENT_SCIENCE" in science_js
    assert "artifact_history" in science_js
    assert "reviewer_gate" in science_js
    assert "run_summary" in science_js
    assert "workflow_scope" in science_js
    assert "fig5_checklist" in science_js
    assert "feature_alignment" in science_js
    assert "discovery_pipeline" in science_js
    assert "data-ag-sci-module" in science_js
    assert "navMetric" in science_js
    assert 'role="tab" aria-selected="' in science_js
    assert 'aria-controls="ag-sci-module-panel"' in science_js
    assert 'role="tabpanel"' in science_js
    assert "scienceModuleBody" in science_js
    assert "bi('Section status', '分区状态')" in science_js
    assert "Evidence readiness checklist" in science_js
    assert "bi('Evidence coverage', '证据覆盖')" in science_js
    assert "bi('Discovery pipeline', '发现流程')" in science_js
    assert "bi('Research tool stack', '研究工具栈')" in science_js
    assert "bi('Skills', '技能')" in science_js
    assert "bi('Connectors', '连接器')" in science_js
    assert "bi('MCP tools', 'MCP 工具')" in science_js
    assert "bi('Prompt contracts', '提示词契约')" in science_js
    assert "bi('Tool audit', '工具审计')" in science_js
    assert "bi('Compute', '计算环境')" in science_js
    assert "window.EU_SETTINGS" in science_js
    assert "window.EU_CAPABILITIES" in science_js
    assert "data && data.capability_policy" in science_js
    assert "policySetting(data, 'science_skills_enabled'" in science_js
    assert "Skills disabled in Settings" in science_js
    assert "connector_pubmed_enabled" in science_js
    assert "connector_zotero_enabled" in science_js
    assert "mcp_tools_enabled" in science_js
    assert "prompt_contracts_enabled" in science_js
    assert "tool_audit_enabled" in science_js
    assert "remote_compute_enabled" in science_js
    assert "Current capability switches from Settings" in science_js
    assert "data-ag-sci-open-settings" in science_js
    assert "核对 Claude Science 可借鉴模式是否落到 EasyICU 真实功能" not in science_js
    assert "Function alignment" not in science_js
    assert "工作台对齐清单" not in science_js
    assert "data-ag-sci-focus-art" in science_js
    assert "data-ag-sci-open-ideas" in science_js
    assert "reusable_protocols" in science_js
    assert "native_renderers" in science_js
    assert ".ag-sci-module-nav" in science_css
    assert ".ag-sci-module-card" in science_css
    assert ".ag-sci-capstack" in science_css
    assert ".ag-sci-capgrid" in science_css
    assert ".ag-sci-capcard" in science_css
    assert "grid-template-columns:repeat(3,minmax(0,1fr));" in science_css
    assert "grid-template-columns:repeat(2,minmax(0,1fr));" in science_css
    assert "grid-template-columns:repeat(5,minmax(112px,1fr));" in science_css
    assert "min-height:44px;" in science_css
    assert "min-height:104px;" not in science_css
    assert "grid-template-columns:repeat(4,minmax(0,1fr));" in science_css
    assert "min-height:112px;" in science_css
    assert "-webkit-line-clamp:2;" in science_css
    assert ".ag-wrap.list-collapsed .ag-sci-summary" in science_css
    assert "grid-template-columns:minmax(0,1.25fr) minmax(300px,.75fr);" in science_css
    assert ".ag-wrap.list-collapsed .ag-sci-kpi-grid" in science_css
    assert ".ag-sci-summary" in science_css
    assert ".ag-sci-checklist" in science_css
    assert ".ag-sci-align" in science_css
    assert ".ag-sci-discovery" in science_css
    assert ".ag-sci-layout" in science_detail_css
    assert ".ag-sci-tabpanel" in science_detail_css
    assert ".ag-sci-card-grid" in science_detail_css
    assert ".ag-sci-renderer" in science_detail_css
    assert ".ag-sci-layout" not in science_css
    assert ".ag-sci-card-grid" not in science_css
    assert ".ag-sci-layout" not in screens_css
    assert ".ag-sci-summary" not in screens_css
    assert ".ag-sci-align" not in screens_css
    assert ".ag-sci-discovery" not in screens_css
    assert ".ag-sci-capstack" not in screens_css
    assert ".ag-sci-capgrid" not in screens_css
    assert ".ag-sci-module-nav" not in screens_css
    assert ".ag-sci-layout" not in redesign_css
    assert ".ag-sci-summary" not in redesign_css
    assert ".ag-sci-align" not in redesign_css
    assert ".ag-sci-discovery" not in redesign_css
    assert ".ag-sci-capstack" not in redesign_css
    assert ".ag-sci-capgrid" not in redesign_css
    assert ".ag-sci-module-nav" not in redesign_css


def test_native_tutorial_screen_uses_active_language_without_mixed_copy() -> None:
    app_js = _static_js("app.js")
    help_js = _static_js("screens-help.js")
    index_html = _static_html("index.html")

    assert "const CRUMB_LABELS = {" in app_js
    assert "'Get Started': ['Get Started', '快速上手']" in app_js
    assert "const actionHtmlOf = (scr) =>" in app_js
    assert "typeof scr.actionHtml === 'function' ? scr.actionHtml()" in app_js
    assert "crumbLabel(scr.crumbs[scr.crumbs.length - 1])" in app_js

    assert "actionHtml() {" in help_js
    assert "t('Start demo', '开始演示')" in help_js
    assert "t('Get started', '快速上手')" in help_js
    assert (
        "t('A quiet, reviewable path from data to draft', '从数据到草稿，一条安静、可审阅的路径')"
        in help_js
    )
    assert "t('The four stages', '四个阶段')" in help_js
    assert "t('Common questions', '常见问题')" in help_js

    assert "Get started · 快速上手" not in help_js
    assert "New here? Take the 2-minute demo tour</div>" not in help_js
    assert ">No tokens, no setup, no patient data. The demo generates" not in help_js
    assert "How a study moves through EasyICU</h2>" not in help_js

    assert "js/app.js?v=20260826-copilot-home1" in index_html
    assert "js/screens-help.js?v=20260817-copilot-boundary1" in index_html


def test_native_guided_and_single_copilot_entry_are_bilingual() -> None:
    guided_js = _static_js("screens-guided.js")
    dock_js = _static_js("copilot-dock.js")
    index_html = _static_html("index.html")

    assert "function bi(en, zh)" in guided_js
    assert "function htmlOf(value)" in guided_js
    assert "htmlOf(t.html)" in guided_js
    assert "你好，我是<strong>研究引导</strong>" in guided_js
    assert "脚本化演示流程" in guided_js
    # The launcher that carried these labels is removed; the shell entry in
    # app.js is now the only labelled opener.
    assert "打开唯一的 EasyICU 研究助手对话" in _static_js("app.js")
    assert "Page guide" not in dock_js
    assert (
        "js/screens-guided-projects.js?v=20260825-remove-project1" in index_html
    )
    assert (
        "js/screens-guided-idea-provider.js?v=20260627-ideas-feasibility-plan"
        in index_html
    )
    assert "js/screens-guided.js?v=20260827-aside-owner1" in index_html
    assert "js/copilot-dock.js?v=20260827-no-fab1" in index_html


def test_native_page_guide_backend_is_retired_from_the_shell_entry() -> None:
    api_js = _static_js("api.js")
    dock_js = _static_js("copilot-dock.js")
    index_html = _static_html("index.html")

    assert "/api/page-guide/sessions" in api_js
    assert "/api/page-guide/message" in api_js
    assert "/api/page-guide/action" in api_js
    assert "/api/page-guide/sessions/list" in api_js
    assert "createPageGuideSession" in api_js
    assert "sendPageGuideMessage" in api_js
    assert "runPageGuideAction" in api_js
    assert "loadPageGuideSessions" in api_js

    # Old copilot routes remain a compatibility wrapper, but the floating dock
    # must no longer consume them.
    assert "/api/copilot/sessions" in api_js
    assert "/api/copilot/message" in api_js
    assert "/api/copilot/action" in api_js
    assert "/api/copilot/sessions/list" in api_js
    assert "createCopilotSession" in api_js
    assert "sendCopilotMessage" in api_js
    assert "runCopilotAction" in api_js
    assert "loadCopilotSessions" in api_js

    assert "location.hash = '#guided'" in dock_js
    assert "document.querySelector('[data-gpi-input]')" in dock_js
    assert "createPageGuideSession" not in dock_js
    assert "sendPageGuideMessage" not in dock_js
    assert "runPageGuideAction" not in dock_js
    assert "createCopilotSession" not in dock_js
    assert "sendCopilotMessage" not in dock_js
    assert "runCopilotAction" not in dock_js
    assert "page-guide dock intentionally is not constructed" in dock_js
    assert "js/api.js?v=20260828-plan-review1" in index_html
    assert "js/copilot-dock.js?v=20260827-no-fab1" in index_html


def test_native_guided_copilot_runs_extraction_inline_and_answers_catalog_questions() -> (
    None
):
    guided_js = _static_js("screens-guided.js")
    # The Idea Mining sub-flow has its own owner; these assertions check
    # ownership, not just presence — a copy left behind in the shell is the
    # failure mode the split exists to prevent.
    idea_js = _static_js("screens-guided-idea.js")
    projects_js = _static_js("screens-guided-projects.js")
    provider_js = _static_js("screens-guided-idea-provider.js")
    api_js = _static_js("api.js")
    guided_css = _static_css("guided.css")
    index_html = _static_html("index.html")

    assert "function startGuidedExtractionFlow" in guided_js
    assert "function renderGuidedExtractionCard" in guided_js
    assert "function scanGuidedExtractionPath" in guided_js
    assert "function runGuidedExtractionJob" in guided_js
    assert "function registerGuidedModuleExport" in guided_js
    assert "GUIDED_EXTRACT_MODULES" in guided_js
    assert "GUIDED_EXTRACT_WINDOW_HOURS = 24 * 30" in guided_js
    assert "data-gx-path" in guided_js
    assert "data-gx-analyze" in guided_js
    assert "data-gx-run" in guided_js
    assert 'data-gx-module-set="all"' in guided_js
    assert 'data-gx-module-set="none"' in guided_js
    assert "format: 'parquet'" in guided_js
    assert 'data-gx-format="${fmt}"' in guided_js
    assert "window.EU_API.startExtractionJob" in guided_js
    assert (
        "new EventSource('/api/jobs/' + encodeURIComponent(r.job_id) + '/events')"
        in guided_js
    )
    assert "window.EU_API.registerWorkspaceSource(out" in guided_js
    assert "window.EU_API.scanPath(path, null)" in guided_js
    assert "source !== 'module'" in guided_js
    assert "No path is prefilled because every user machine is different" in guided_js
    assert "goal === 'data_extraction'" in guided_js
    assert "isGuidedExtractionIntent(v)" in guided_js
    assert "function startGuidedReviewFlow" in guided_js
    assert "function renderGuidedReviewCard" in guided_js
    assert "function loadGuidedReviewData" in guided_js
    assert "window.EU_API.loadPatientReviewDrilldown" in guided_js
    assert "window.EU_API.loadCohortReviewSummary" in guided_js
    assert "KM / log-rank" in guided_js
    assert "Number at risk" in guided_js
    assert "goal === 'review_data'" in guided_js
    assert "isGuidedReviewIntent(v)" in guided_js
    assert "function startGuidedAgentFlow" in guided_js
    assert "function renderGuidedAgentCard" in guided_js
    assert "function runGuidedAgentPreflight" in guided_js
    assert "window.EU_API.startAgentRun" in guided_js
    assert "run_type: 'preflight'" in guided_js
    assert "provider: 'mock'" in guided_js
    assert "llm_provider: runToken.provider" in guided_js
    assert "external_llm_opt_in: false" in guided_js
    assert "goal === 'run_agent'" in guided_js
    assert "isGuidedAgentIntent(v)" in guided_js
    assert "window.EU_GUIDED_IDEA_PROVIDER.requestStatus" in idea_js
    assert "loadAgentProviderStatus" in provider_js
    assert "window.EU_API.saveAgentProviderConfig" in _static_js("api.js")
    assert "saveAgentProviderConfig" in provider_js
    assert "goal === 'idea_mining'" in guided_js
    assert "data-gi-mine" in guided_js
    assert "data-gi-pdf-file" in guided_js
    assert "data-gi-lit-browse" in guided_js
    assert "data-gi-lit-scan" in guided_js
    assert "data-gi-provider-refresh" in provider_js
    assert "data-gi-provider-config-toggle" in provider_js
    assert "data-gi-provider-key" in provider_js
    assert "data-gi-provider-base" in provider_js
    assert "data-gi-provider-model" in provider_js
    assert "data-gi-provider-save" in provider_js
    assert "data-gi-enable-ai" in provider_js
    assert "data-gi-api-continue" in provider_js
    assert "data-gi-api-back" in provider_js
    assert "function requestGuidedIdeaProviderStatus" in idea_js
    assert "API readiness setup" in provider_js
    assert "API setup gate" not in guided_js
    assert "API setup gate" not in provider_js
    assert "window.EU_GUIDED_IDEA_PROVIDER = {" in provider_js
    assert "function renderCapabilityPanel(" in provider_js
    assert "function renderSetupPrompt(" in provider_js
    assert "function renderMiniStatus(" in provider_js
    assert "data-gi-handoff" in guided_js
    assert "data-gi-project" in guided_js

    assert "function findLocalConceptQuery" in guided_js
    assert "function answerConceptQuestion" in guided_js
    assert "sofa-2" in guided_js
    assert "window.EU_CATALOG" in guided_js
    assert "Open Data Dictionary" in guided_js
    assert "This answer is local and code-backed" in guided_js

    assert "function startExtractionJob" in api_js
    assert "postJSON('/api/jobs/extract'" in api_js
    assert "window.EU_API.startExtractionJob = startExtractionJob" in api_js

    assert ".gd-x-card" in guided_css
    assert ".gdx-pathrow" in guided_css
    assert ".gdx-modgrid" in guided_css
    assert ".gdx-presets" in guided_css
    assert ".gd-review-card" in guided_css
    assert ".gdr-panel" in guided_css
    assert ".gdr-risk" in guided_css
    assert ".gd-agent-card" in guided_css
    assert ".gda-question" in guided_css
    assert ".gd-idea-card" in guided_css
    assert ".gdi-flow" in guided_css
    assert ".gdi-flow-step.active" in guided_css
    assert ".gdi-field" in guided_css
    assert ".gdi-source-mode-note" in guided_css
    assert ".gdi-ledger-grid" in guided_css
    assert ".gdi-feature-row" in guided_css
    assert ".gdi-plan" in guided_css
    assert ".gd-concept-answer" in guided_css
    assert "let guidedPipelineOpen = false;" in guided_js
    assert "function renderStudyPipelineSummary" in guided_js
    assert "function renderStudyItemList" in guided_js
    assert "data-gd-pipeline-toggle" in guided_js
    assert "data-gd-pipeline-list" in guided_js
    assert "id=\"gdPipelineList\" ${guidedPipelineOpen ? '' : 'hidden'}" in guided_js
    assert ".gd-pipeline-summary" in guided_css
    assert ".gd-pipeline-toggle" in guided_css
    assert ".gd-pipeline-list[hidden]" in guided_css

    guided_plan_js = _static_js("screens-guided-idea-plan.js")
    guided_plan_css = _static_css("guided-idea-plan.css")
    redesign_css = _static_css("redesign.css")

    assert "css/guided.css?v=20260827-type-scale1" in index_html
    assert "css/guided-projects.css?v=20260827-type-scale1" in index_html
    assert "css/guided-idea-plan.css?v=20260827-type-scale1" in index_html
    assert "js/api.js?v=20260828-plan-review1" in index_html
    assert (
        "js/screens-guided-projects.js?v=20260825-remove-project1" in index_html
    )
    provider_pos = index_html.find("screens-guided-idea-provider.js")
    projects_pos = index_html.find("screens-guided-projects.js")
    idea_plan_pos = index_html.find("screens-guided-idea-plan.js")
    guided_pos = index_html.find("screens-guided.js?")
    assert (
        projects_pos != -1
        and provider_pos != -1
        and idea_plan_pos != -1
        and guided_pos != -1
    )
    assert projects_pos < guided_pos
    assert provider_pos < guided_pos
    assert provider_pos < idea_plan_pos < guided_pos

    # Progressive extraction + study-design stepper owner file (screens-guided-extract.js)
    extract_js = _static_js("screens-guided-extract.js")
    assert "js/screens-guided-extract.js?v=20260707-copilot" in index_html
    extract_pos = index_html.find("screens-guided-extract.js")
    assert extract_pos != -1 and extract_pos < guided_pos
    assert "window.EU_GUIDED_EXTRACT = {" in extract_js
    # study-design vocabulary is owned by the stepper module, not the shell
    assert "Primary outcome / endpoint" in extract_js
    assert "Observation window" in extract_js
    assert "Comparison" in extract_js
    assert "Export destination" in extract_js
    assert "resolveOutcome" in extract_js and "windowHours" in extract_js
    # main file owns state + wiring and delegates rendering to the sibling
    assert "window.EU_GUIDED_EXTRACT.render" in guided_js
    assert "function goGuidedExtractStep" in guided_js
    assert "function commitGuidedDesign" in guided_js
    assert "function resetGuidedDesignState" in guided_js
    assert "guidedDesignWindowHours()" in guided_js
    assert "out_dir:" in guided_js  # export destination reaches the extraction job
    assert "study_design" in guided_js  # study design persisted to project memory
    # step + design controls are wired
    for marker in (
        "data-gx-step-next",
        "data-gx-goto-step",
        "data-gx-outcome",
        "data-gx-window",
        "data-gx-comparator",
        "data-gx-exportdir",
    ):
        assert marker in guided_js, marker
    # stepper CSS lives in the guided owner file, not a catch-all
    for cls in (
        ".gdx-steps",
        ".gdx-step",
        ".gdx-recap",
        ".gdx-summary",
        ".gdi-next-cue",
    ):
        assert cls in guided_css, cls
    redesign_css_body = _static_css("redesign.css")
    assert ".gdx-steps" not in redesign_css_body
    # free-text disambiguation + honest real-mode fallback (no demo-shortcut coaching)
    assert "function reflectGuidedFrontdoor" in guided_js
    assert "run the whole demo" in guided_js  # still present, but demo-gated

    assert "window.EU_GUIDED_PROJECTS = {" in projects_js
    assert "window.EU_GUIDED_IDEA_PLAN = {" in guided_plan_js
    assert "window.EU_GUIDED_IDEA_PLAN.render" in idea_js
    assert "guidedProjectContext()" in guided_js
    assert "function runGuidedIdeaPlan" in idea_js
    assert "window.EU_API.planIdea" in idea_js
    assert "data-gi-plan" in guided_js
    assert "data-gi-replan" in guided_js
    assert "Create a study plan before Agent handoff" in guided_plan_js
    assert "Reference method patterns" in guided_plan_js
    assert "ICU constraints" in guided_plan_js
    assert "Plan / replan notes" in guided_plan_js
    assert "Applied replan note" in guided_plan_js
    assert (
        "Generate and review the study plan before freezing an Agent handoff"
        in idea_js
    )
    # Renamed on the move: inside its own owner the guided/idea prefix is
    # noise. A saved mid-run session still re-fetches that run on restore.
    assert "function restoreArtifacts(runId)" in idea_js
    assert "IDEA.restoreSlot(slots.idea);" in guided_js
    assert "idea: IDEA.slotSnapshot()," in guided_js
    assert "dataContextConfirmed" in idea_js
    assert "function confirmGuidedIdeaDataContext" in idea_js
    assert (
        "This only turns a source clue into a candidate research question" in idea_js
    )
    assert "requires explicit data-context confirmation" in idea_js
    assert "Manual idea mode" in idea_js
    assert "Article URL mode" in idea_js
    assert "PDF file mode" in idea_js
    assert "Literature folder mode" in idea_js
    assert "Frontier topic mode" in idea_js
    assert "source-${attr(tab)}" in idea_js
    assert ".gdi-plan-details" in guided_plan_css
    assert ".gdi-feature-row.one" in guided_plan_css
    assert ".gdi-plan-details" not in redesign_css
    assert "js/screens-guided.js?v=20260827-aside-owner1" in index_html

    assert "function startGuidedIdeaFlow" in idea_js
    assert "function renderGuidedIdeaApiSetupCard" in idea_js
    assert "function showGuidedIdeaSourceForm" in idea_js
    assert "function showGuidedIdeaApiSetup" in idea_js
    assert "function renderGuidedIdeaCard" in idea_js
    assert "function runGuidedIdeaMine" in idea_js
    assert "function runGuidedIdeaPriorArt" in idea_js
    assert "function runGuidedIdeaHandoff" in idea_js
    assert "function runGuidedIdeaCreateProject" in idea_js
    assert "window.EU_API.mineIdeas" in idea_js
    assert "window.EU_API.resolveIdeaSource" in idea_js
    assert "window.EU_API.ingestIdeaPdf" in idea_js
    assert "window.EU_API.scanIdeaLiteratureFolder" in idea_js
    assert "window.EU_API.checkIdeaPriorArt" in idea_js
    assert "window.EU_API.handoffIdea" in idea_js
    assert "window.EU_API.createIdeaAgentProject" in idea_js
    assert "function saveGuidedIdeaProviderConfig" in idea_js
    assert "host.thread().push({ guidedIdeaApiSetup: true })" in idea_js
    for moved in ("function runGuidedIdeaMine", "function renderGuidedIdeaCard"):
        assert moved not in guided_js, f"{moved} is still in the shell too"
    # The shell reaches the sub-flow only through the published owner.
    assert "const IDEA = window.EU_GUIDED_IDEA;" in guided_js
    assert "IDEA.runGuidedIdeaMine(" in guided_js
    assert "IDEA.isGuidedIdeaIntent(v)" in guided_js

def test_native_agent_outputs_fail_closed_to_real_artifacts() -> None:
    agent_js = _static_js("screens-agent.js")
    render_js = _static_js("screens-agent-render.js")
    agent_css = _static_css("agent.css")
    agent_layout_css = _static_css("agent-layout.css")
    agent_header_css = _static_css("agent-header.css")
    agent_review_css = _static_css("agent-review.css")
    agent_cap_css = _static_css("agent-capabilities.css")
    screens_css = _static_css("screens.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert "js/screens-agent.js?v=20260823-run-history-authority1" in index_html
    assert "css/agent.css?v=20260817-project-monitor-states2" in index_html
    assert "css/agent-layout.css?v=20260817-project-monitor-states2" in index_html
    assert "css/agent-header.css?v=20260702-agent-compact-header" in index_html
    assert "css/agent-review.css?v=20260702-agent-review-compact" in index_html
    assert "css/agent-capabilities.css?v=20260627-agent-capabilities" in index_html
    assert "function artifactsForLive(live)" in agent_js
    assert "function reviewableRunForStudy()" in agent_js
    assert "let agListMode = 'auto';" in agent_js
    assert (
        "const AG_FOCUS_TABS = new Set(['science', 'runs', 'outputs', 'notes', 'draft']);"
        in agent_js
    )
    assert "function agentListCollapsed()" in agent_js
    assert "const detail = document.querySelector('#agHost .ag-detail');" in agent_js
    assert "document.querySelector('#agHost .ag-body');" not in agent_js
    assert "data-ag-toggle-list" in agent_js
    assert 'aria-controls="agStudyList"' in agent_js
    assert "ag-wrap ${listCollapsed ? 'list-collapsed' : 'list-open'}" in agent_js
    assert "data-ag-list-state=\"${listCollapsed ? 'collapsed' : 'open'}\"" in agent_js
    assert "const compactHeader = agTab !== 'overview';" in agent_js
    assert "ag-dhead ${compactHeader ? 'compact' : ''}" in agent_js
    assert "function outputCountForStudy()" in agent_js
    assert "ag-review-layout" in agent_js
    assert "ag-review-claims" in agent_js
    assert "Show all artifacts" in agent_js
    assert "visibleArtifacts = artifacts.slice(0, 6)" in agent_js
    assert "['outputs', t('Outputs', '产出'), outputCountForStudy()]" in agent_js
    assert "No real output artifacts yet" in agent_js
    assert "placeholders are not shown in Real mode" in agent_js
    assert (
        "It will not show demo Table 1, missingness, ROC, or calibration placeholders"
        in agent_js
    )
    assert 'data-ag-artifact-view="${esc(name)}"' in agent_js
    assert "artifactTitle(name)" in agent_js
    assert "artifactSummary(name)" in agent_js
    assert "artifactCategory(name)" in agent_js
    assert "Primary review outputs" in agent_js
    # Artifact renderers live in the screens-agent-render.js owner file.
    assert "function artifactStructuredView(name, payload)" in render_js
    assert "Readable artifact summary" in render_js
    assert "可读产物摘要" in render_js
    assert "row.step_id || row.id || row.step" in render_js
    assert "row.intent || row.title || row.name" in render_js
    assert "Array.isArray(row.expected_outputs)" in render_js
    assert (
        "Raw JSON is kept for audit, but the default view is table-based." in render_js
    )
    # The raw-JSON <details> lives in artifactViewer, which stays in main.
    assert "View raw JSON" in agent_js
    assert "查看原始 JSON" in agent_js
    assert "function featuredFigurePreview(live)" in agent_js
    assert "Result figures" in agent_js
    assert "function evidenceLinkPanel(live, s)" in agent_js
    assert "function crossDataPanel(live, s)" in agent_js
    assert "function capabilityHighlights(live, s)" in agent_js
    assert "Evidence Link" in agent_js
    assert "证据链接" in agent_js
    assert "Cross-data scope" in agent_js
    assert "跨数据范围" in agent_js
    assert "data-ag-artifact-jump" in agent_js
    assert "Completed analysis outputs" in agent_js
    assert "Download bundle" in agent_js
    assert "['outputs', t('Outputs', '产出'), 6]" not in agent_js
    assert "Seeded demo artifacts." not in agent_js
    assert "Illustrative outputs for layout" not in agent_js
    assert "Project Monitor route-owned layout fixes" in agent_css
    assert ".ag-pipe .pline" in agent_layout_css
    assert ".ag-pipe .pstep" in agent_layout_css
    assert ".ag-pipe .pt" in agent_layout_css
    assert ".ag-pipe .pd" in agent_layout_css
    assert ".ag-wrap .chip" in agent_css
    assert "Project Monitor focus layout" in agent_layout_css
    assert ".ag-wrap.list-collapsed" in agent_layout_css
    assert ".ag-wrap.list-collapsed .ag-list" in agent_layout_css
    assert ".ag-wrap [data-ag-toggle-list]" in agent_layout_css
    assert ".ag-wrap.list-collapsed" not in agent_css
    assert ".ag-wrap.list-collapsed" not in screens_css
    assert ".ag-wrap.list-collapsed" not in redesign_css
    assert ".ag-dhead.compact" in agent_header_css
    assert ".ag-dhead.compact + .ag-tabs" in agent_header_css
    assert ".ag-dhead.compact" not in agent_css
    assert ".ag-dhead.compact" not in redesign_css
    assert ".ag-review-layout" in agent_review_css
    assert ".ag-review-side" in agent_review_css
    assert ".ag-review-claims" in agent_review_css
    assert ".ag-review-layout" not in agent_css
    assert ".ag-review-claims" not in agent_css
    assert ".ag-review-layout" not in redesign_css
    assert ".ag-review-claims" not in redesign_css
    assert "Project Monitor output review cards" in agent_css
    assert ".ag-wrap .ag-output-brief" in agent_css
    assert ".ag-wrap .ag-featured-results" in agent_css
    assert ".ag-wrap .outcard.on" in agent_css
    assert ".ag-artifact-readable" in agent_css
    assert ".ag-artifact-table" in agent_css
    assert ".ag-raw-json" in agent_css
    assert ".ag-raw-json:not([open]) pre" in agent_css
    assert "Project Monitor capability highlights" in agent_cap_css
    assert ".ag-wrap .ag-cap-grid" in agent_cap_css
    assert ".ag-wrap .ag-link-chain" in agent_cap_css


def test_project_monitor_renders_one_load_state_and_fits_the_pipeline() -> None:
    """Loading, index failure, successful-empty, and ready are exclusive.

    Blank states omit the project header, pipeline, and tabs instead of
    presenting an invented active Plan while the local index is unavailable.
    """
    agent_js = _static_js("screens-agent.js")
    agent_css = _static_css("agent.css")
    layout_css = _static_css("agent-layout.css")

    assert "function monitorViewState(studies)" in agent_js
    assert "if (studies.length) return 'ready';" in agent_js
    assert "if (agIdeaProjects.error) return 'error';" in agent_js
    assert "if (monitorState !== 'ready') return" in agent_js
    assert 'data-ag-monitor-state="${monitorState}"' in agent_js
    assert "const count = monitorState === 'loading' || monitorState === 'error' ? '—' : studies.length;" in agent_js
    assert "Checking project index" in agent_js
    for state in ("Loading projects", "Could not load local projects", "No projects to monitor yet"):
        assert state in agent_js
    assert "No project selected for monitoring" not in agent_js
    assert "Local research projects unavailable" not in agent_js
    assert "ideas-empty-list" not in agent_js
    assert "ag-empty-steps" not in agent_js

    for selector in (".ag-wrap-blank", ".ag-detail-blank", ".ag-monitor-state", ".ag-list-state"):
        assert selector in layout_css
        assert selector not in agent_css
    assert "@media (min-width:721px)" in layout_css
    assert ".ag-pipe{ overflow:hidden; }" in layout_css
    assert ".ag-pipe .pstep{ flex:1 1 0; min-width:0; }" in layout_css
    assert ".ag-pipe .pd{ display:none; }" in layout_css


def test_project_monitor_loads_persisted_run_history_before_claiming_zero() -> None:
    agent_js = _static_js("screens-agent.js")
    history_js = _static_js("screens-agent-run-history.js")

    assert "function rows(history, study)" in history_js
    assert "function count(history, study, realMode)" in history_js
    assert "return live || historyRunForStudy(selected)" in agent_js
    assert "persisted && persisted.project_dir" in history_js
    assert "requestRunHistory();\n      if (window.__euRender)" in agent_js
    assert "const noRun = monitorRunCount(s) === 0;" in agent_js
    assert "Checking run history" in agent_js
    assert "Run history unavailable" in agent_js


def test_project_monitor_excludes_copilot_setup_and_run_initiation() -> None:
    agent_js = _static_js("screens-agent.js")
    render_js = _static_js("screens-agent-render.js")
    agent_css = _static_css("agent.css")
    app_js = _static_js("app.js")
    guided_js = _static_js("screens-guided-pi.js")
    provider_js = _static_js("screens-guided-pi-provider.js")
    screens_css = _static_css("screens.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    forbidden_js = (
        "BLOCK_LIBRARY",
        "BLOCK_FAMILIES",
        "NATURE_PACK",
        "AG_BLOCKS_VERSION",
        "Planning Blocks",
        "data-ag-block",
        "data-ag-new",
        "data-ag-runbtn",
        "data-ag-promote",
        "window.EU_API.startAgentRun",
        "function startRun",
        "function startRealRun",
        "function startDemoRun",
    )
    for marker in forbidden_js:
        assert marker not in agent_js
        assert marker not in render_js

    for marker in (".ag-block", ".ag-wf", ".ag-lib"):
        assert marker not in agent_css
        assert marker not in screens_css
        assert marker not in redesign_css

    assert "Requirements, model setup, and run initiation stay in Guided Copilot" in agent_js
    assert "Requirements and execution live in Guided Copilot" in agent_js
    assert "Continue in Guided Copilot" in agent_js
    assert "data-ag-guided" in agent_js
    assert "loadAgentRunHistory" in agent_js
    assert "loadAgentRunReview" in agent_js
    assert "signoffAgentRun" in agent_js
    assert "Project Monitor" in app_js
    assert "runs · outputs · evidence · review" in app_js
    assert "sendPiCopilotMessage" in guided_js
    assert "easyicu_run_submitted" in guided_js
    assert "data-gpi-provider-form" in provider_js
    assert "css/agent.css?v=20260817-project-monitor-states2" in index_html
    assert "js/screens-agent.js?v=20260823-run-history-authority1" in index_html


def test_native_agent_render_layer_is_split_into_owner_file() -> None:
    """The fixture data and pure artifact
    renderers are owned by screens-agent-render.js, not inlined in the
    screens-agent.js monolith (owner-file carve-out, 2026-07-03). The main
    file rebinds them from window.AGENT_RENDER so call sites stay unchanged."""
    agent_js = _static_js("screens-agent.js")
    render_js = _static_js("screens-agent-render.js")
    index_html = _static_html("index.html")

    # Fixture data + pure renderers are defined in the render file. Copilot
    # configuration catalogs do not belong in this project-monitor owner.
    assert "const DEMO_STUDIES = [" in render_js
    assert "const BLOCK_LIBRARY = [" not in render_js
    assert "function artifactStructuredView(name, payload)" in render_js
    assert "function runStatusLabel(status)" in render_js
    assert "function thumb(kind)" in render_js
    assert "window.AGENT_RENDER = {" in render_js

    # they are NOT re-defined in the main file (no duplicate definitions)
    assert "const DEMO_STUDIES = [" not in agent_js
    assert "const BLOCK_LIBRARY = [" not in agent_js
    assert "function artifactStructuredView(name, payload)" not in agent_js
    assert "function runStatusLabel(status)" not in agent_js

    # main file rebinds the exports so call sites stay unchanged
    assert "} = R;" in agent_js
    assert "window.AGENT_RENDER" in agent_js
    assert "artifactStructuredView" in agent_js  # still called

    # render file loads BEFORE the main file in index.html
    render_pos = index_html.find("screens-agent-render.js")
    main_pos = index_html.find("screens-agent.js?")
    assert render_pos != -1 and main_pos != -1
    assert (
        render_pos < main_pos
    ), "screens-agent-render.js must load before screens-agent.js"
    assert "js/screens-agent-render.js?v=20260828-plan-reader5" in index_html
    assert "css/agent-plan.css?v=20260828-plan-reader1" in index_html


def test_candidate_plan_styles_have_one_explicit_owner() -> None:
    plan_css = _static_css("agent-plan.css")
    review_css = _static_css("agent-scientific-review.css")
    agent_css = _static_css("agent.css")

    for selector in (
        ".ag-plan-reader",
        ".ag-plan-hero",
        ".ag-plan-design-grid",
        ".ag-plan-steps",
        ".ag-plan-section.is-gap",
    ):
        assert selector in plan_css
        assert selector not in review_css
        assert selector not in agent_css
    for foreign_route in ("crossdb", "patient", "cohort", "settings"):
        assert foreign_route not in plan_css.lower()
    assert "!important" not in plan_css


def test_copilot_owns_provider_selection_and_agent_projects_do_not() -> None:
    agent_js = _static_js("screens-agent.js")
    guided_js = _static_js("screens-guided-pi.js")
    provider_js = _static_js("screens-guided-pi-provider.js")
    index_html = _static_html("index.html")

    assert "window.EU_GUIDED_PI_PROVIDER =" in provider_js
    assert "ChatGPT / Codex account" in provider_js
    assert "DeepSeek API" in provider_js
    assert "OpenRouter API" in provider_js
    assert "data-gpi-codex-login" in provider_js
    assert "data-gpi-codex-device" in provider_js
    assert "data-gpi-codex-model" in provider_js
    assert "ONE MODEL CONNECTION" in provider_js
    assert "same selected provider and model powers" in provider_js
    assert "Conversation model API" not in provider_js
    assert "Analysis model" not in provider_js
    assert "research_provider: state.researchProvider" in guided_js
    assert "startPiCopilotCodexLogin" in guided_js
    assert "AGENT_PROVIDER_PANEL" not in agent_js
    assert "data-ag-codex-login" not in agent_js
    assert "data-ag-external-run" not in agent_js
    assert "screens-agent-provider.js" not in index_html

    provider_pos = index_html.find("screens-guided-pi-provider.js?")
    main_pos = index_html.find("screens-guided-pi.js?")
    assert provider_pos != -1 and main_pos != -1
    assert provider_pos < main_pos
    assert (
        "js/screens-guided-pi-provider.js?v=20260825-api-consent1"
        in index_html
    )

    for foreign_marker in ("data-ag-", "data-cd-", "data-pt-", "data-idea-"):
        assert foreign_marker not in provider_js


def test_native_agent_overview_renders_object_idea_plan_steps() -> None:
    agent_js = _static_js("screens-agent.js")

    assert "function seedPlanStepDisplay(row)" in agent_js
    assert "typeof row === 'object'" in agent_js
    assert "row.title || row.action" in agent_js
    assert "const step = seedPlanStepDisplay(x);" in agent_js
    assert "return [step.title, step.detail, 'ready'];" in agent_js
    assert '<div class="pi-t">${esc(ti)}</div>' in agent_js
    assert "seedPlan.map(x => [x," not in agent_js


def test_native_agent_historical_evaluation_import_uses_normal_project_surface() -> None:
    agent_js = _static_js("screens-agent.js")
    agent_css = _static_css("agent.css")
    agent_cap_css = _static_css("agent-capabilities.css")
    agent_question_css = _static_css("agent-question.css")
    app_js = _static_js("app.js")
    ideas_js = _static_js("screens-ideas.js")
    guided_js = _static_js("screens-guided.js")
    index_html = _static_html("index.html")
    screens_css = _static_css("screens.css")
    redesign_css = _static_css("redesign.css")
    ideas_css = _static_css("ideas.css")
    guided_css = _static_css("guided.css")
    agent_runs_py = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "easyicu"
        / "webserver"
        / "agent_runs.py"
    ).read_text(encoding="utf-8")

    assert "canonical9_import" in agent_js
    assert "function importedRunForStudy(s)" in agent_js
    assert "function reviewableRunForStudy()" in agent_js
    # runStatusLabel / readableArtifactText moved to the render owner file.
    assert "function runStatusLabel(status)" in _static_js("screens-agent-render.js")
    assert "function readableArtifactText(value)" in _static_js(
        "screens-agent-render.js"
    )
    assert "function evidenceLinkPanel(live, s)" in agent_js
    assert "function crossDataPanel(live, s)" in agent_js
    assert "function capabilityHighlights(live, s)" in agent_js
    assert "function questionParts(text)" in agent_js
    assert "function questionTags(s, raw)" in agent_js
    assert "function renderStructuredQuestion(s)" in agent_js
    assert "function focusAgentBody()" in agent_js
    assert "function importedResultSummary(s)" in agent_js
    assert "function featuredFigurePreview(live)" in agent_js
    assert "function benchmarkPanel(s)" not in agent_js
    assert "data-ag-open-seed-run" in agent_js
    assert (
        "artifactStructuredView(artifact.name || agArtifact.name || '', data.payload || {})"
        in agent_js
    )
    assert "benchmark_scorecard.json" in agent_js
    # workflow_graph.json only appears in the moved artifact label maps.
    assert "workflow_graph.json" in _static_js("screens-agent-render.js")
    assert "figure_gallery.json" in agent_js
    assert "source_run_manifest.json" in agent_js
    assert "Completed analysis" in agent_js
    assert "Research idea" in agent_js
    assert "Read-only review · manuscript not unlocked" in agent_js
    assert "Study brief" in agent_js
    assert "汇报摘要" in agent_js
    # "verification passed" is a runStatusLabel string, now in the render file.
    assert "verification passed" in _static_js("screens-agent-render.js")
    assert "readableArtifactText(row.text || '')" in agent_js
    assert "Claim-to-artifact trace is explicit" in agent_js
    assert "Open Cross-DB workspace" in agent_js
    assert "Core question" in agent_js
    assert "Analysis requirements" in agent_js
    assert "Data context" in agent_js
    assert "scrollIntoView({ block: 'start', behavior: 'auto' })" in agent_js
    assert "function studyListContext(studies)" in agent_js
    assert "Demo mode includes example projects for exploration" in agent_js
    assert "function agentTermStrip(s)" in agent_js
    assert "automated checks passed; human review is still required" in agent_js
    assert "想法种子" not in agent_js
    assert "九问运行" not in agent_js
    assert "Idea seed" not in agent_js
    assert "Canonical run" not in agent_js
    assert "s.id === 'aki' ? 'kdigo' : 'lactate'" not in agent_js
    assert "Figure 2 question package" not in agent_js
    assert "Figure 2 问题包" not in agent_js
    assert "clinical benchmark task" not in agent_js
    assert "临床 benchmark 问题" not in agent_js
    assert "Current canonical9 package" not in agent_js
    assert "当前 canonical9 包" not in agent_js
    assert ".ag-list-context" in agent_css
    assert ".ag-term-strip" in agent_css
    assert ".ag-bench-card" not in agent_css
    assert ".ag-bench-metrics" not in agent_css
    assert ".ag-score-grid" not in agent_css
    assert ".ag-figure-gallery" in agent_css
    assert ".ag-present-brief" in agent_css
    assert ".ag-wrap .ag-output-brief" in agent_css
    assert ".ag-wrap .ag-featured-results" in agent_css
    assert ".ag-wrap .ag-cap-grid" in agent_cap_css
    assert ".ag-wrap .ag-cap-card.evidence" in agent_cap_css
    assert ".ag-wrap .ag-cap-card.cross" in agent_cap_css
    assert "Project Monitor structured question brief" in agent_question_css
    assert ".ag-wrap .ag-question-brief" in agent_question_css
    assert ".ag-wrap .ag-q-section + .ag-q-section" in agent_question_css
    assert ".ag-wrap .ag-req-list" in agent_question_css
    assert "css/agent-question.css?v=20260629-ux-readability" in index_html
    assert "css/agent.css?v=20260817-project-monitor-states2" in index_html
    assert "js/screens-agent.js?v=20260823-run-history-authority1" in index_html

    for name in (
        "benchmark_scorecard.json",
        "workflow_graph.json",
        "figure_gallery.json",
        "source_run_manifest.json",
    ):
        assert name in agent_runs_py

    assert "canonical9_import" not in app_js
    assert "canonical9_import" not in ideas_js
    assert "canonical9_import" not in guided_js
    for css in (screens_css, redesign_css, ideas_css, guided_css):
        assert ".ag-bench-card" not in css
        assert ".ag-bench-metrics" not in css
        assert ".ag-score-grid" not in css
        assert ".ag-figure-gallery" not in css
        assert ".ag-cap-grid" not in css
        assert ".ag-link-chain" not in css
        assert ".ag-question-brief" not in css
        assert ".ag-req-list" not in css


def test_native_route_qa_allows_only_explicit_truncation_and_scroll_regions() -> None:
    route_qa = Path("tools/qa_native_fastapi_routes.py").read_text(encoding="utf-8")
    app_css = _static_css("app.css")

    assert "insideHorizontalScrollRegion" in route_qa
    assert (
        ".table-scroll, .risk-table-wrap, .dict-table"
        in route_qa
    )
    assert "intentionallyEllipsized" in route_qa
    assert "textOverflow === 'ellipsis'" in route_qa
    assert "intentionalVerticalViewportClip" in route_qa
    assert "isExplicitHorizontalScroller" in route_qa
    assert "欢迎使用 EasyICU" in route_qa
    assert ".table-scroll{\n    overflow-x: auto;" in app_css


def test_native_settings_controls_are_backend_wired() -> None:
    api_js = _static_js("api.js")
    settings_js = _static_js("screens-settings.js")
    extensions_js = _static_js("screens-settings-extensions.js")
    settings_css = _static_css("settings.css")
    i18n_js = _static_js("i18n.js")
    tweaks_css = _static_css("tweaks.css")
    index_html = _static_html("index.html")
    settings_py = (STATIC_DIR.parent / "settings.py").read_text(encoding="utf-8")

    assert "/api/settings/reset" in api_js
    assert "resetSettings" in settings_js
    assert "data-setting-path" in settings_js
    assert "pathCtl('export_dir'" in settings_js
    assert "pathCtl('working_dir'" not in settings_js
    assert "openSettingsFolderPicker" in settings_js
    assert "window.EU_API.listDir" in settings_js
    assert "data-settings-diagnostics" in settings_js
    assert "data-settings-jump" in settings_js
    assert "scrollIntoView({ behavior: 'smooth', block: 'start' })" in settings_js
    assert 'href="#set-' not in settings_js
    assert ".set-nav-btn" in _static_css("pages.css")
    assert "easyicu_settings_diagnostics.json" in settings_js
    assert "lockedCtl" in settings_js
    # The Research Agent section used to hold four inert placards asserting run
    # behaviour the page could not observe. It now points at the surface that
    # actually owns those decisions.
    assert "data-settings-open=\"agent\"" in settings_js
    assert "strict enforced" not in settings_js
    assert "There is no telemetry collector" not in settings_js
    assert "恢复默认设置" in settings_js
    assert "工作区 · 设置" in settings_js
    assert "本地路径" in settings_js
    assert "本地优先保障" in settings_js
    assert "研究代理" in settings_js
    assert "选择默认导出文件夹" in settings_js
    assert "设置已恢复为后端默认值。" in settings_js
    assert "dual('Capabilities', '能力')" in settings_js
    assert "Research controls" in settings_js
    assert "settingsCapabilityTab" in settings_js
    assert "data-settings-cap-tab" in settings_js
    assert "data-settings-open" in settings_js
    assert "window.EU_CAPABILITIES" in settings_js
    assert "cap('zotero_connector')" in settings_js
    assert "cap('mcp_tools')" in settings_js
    assert "cap('prompt_contracts')" in settings_js
    assert "cap('tool_audit')" in settings_js
    assert "cap('remote_compute')" in settings_js
    assert "Zotero auto-connect" in settings_js
    assert "Zotero local API" not in settings_js
    assert "Backend policy" in settings_js
    assert "Backend contract rules" in settings_js
    assert "Compute adapter" in settings_js
    assert "science_skills_enabled" in settings_js
    assert "connector_pubmed_enabled" in settings_js
    assert "connector_zotero_enabled" in settings_js
    assert "mcp_tools_enabled" in settings_js
    assert "prompt_contracts_enabled" in settings_js
    assert "tool_audit_enabled" in settings_js
    assert "remote_compute_enabled" in settings_js
    assert "PubMed connector" in settings_js
    assert "dual('MCP tools', 'MCP 工具')" in settings_js
    assert "dual('Prompt contracts', '提示词契约')" in settings_js
    assert "Tool audit ledger" in settings_js
    assert "功能对齐" not in settings_js
    assert '"connector_pubmed_enabled": True' in settings_py
    assert '"connector_zotero_enabled": False' in settings_py
    assert '"mcp_tools_enabled": False' in settings_py
    assert '"prompt_contracts_enabled": True' in settings_py
    assert '"tool_audit_enabled": True' in settings_py
    assert '"remote_compute_enabled": False' in settings_py
    assert "crumbs: ['Home', 'Settings']" not in settings_js
    assert (
        "actionHtml: `<button class=\"btn\" data-settings-reset>${icon('refresh', 13)} Reset to defaults</button>`"
        not in settings_js
    )
    assert '<h1 style="margin-top:6px;">Settings</h1>' not in settings_js
    assert (
        '<div class="rail-head"><span class="t">Settings</span></div>'
        not in settings_js
    )
    assert 'data-setting="telemetry_enabled"' not in settings_js
    assert 'data-setting-input="token_budget"' not in settings_js
    assert 'data-setting="module_folder_mode"' not in settings_js
    assert "if (key === 'data_mode' && val && window.setDataMode)" in settings_js
    assert (
        "window.applySettingsState(window.EU_SETTINGS, { syncStorage: true })" in api_js
    )
    assert "window.EU_API.hydrateCapabilities = hydrateCapabilities" in api_js
    assert "await hydrateCapabilities();" in api_js
    assert "/api/capabilities/zotero/test" in api_js
    assert "/api/capabilities/zotero/source" in api_js
    assert "/api/capabilities/zotero/import" in api_js
    assert "testZoteroConnection" in api_js
    assert "zoteroSource" in api_js
    assert "importZoteroSource" in api_js
    assert "/api/extensions/skills/install" in api_js
    assert "/api/extensions/mcp/install" in api_js
    assert "/api/extensions/mcp/test" in api_js
    assert "window.EU_API.loadExtensions" in api_js
    assert "window.EU_SETTINGS_EXTENSIONS" in extensions_js
    assert "data-ext-install-skill" in extensions_js
    assert "data-ext-test-mcp" in extensions_js
    assert "allowed_tools" in extensions_js
    assert "Frozen into each new Copilot session and Agent run" in extensions_js
    assert "capabilities: C0()" in settings_js
    assert "data-settings-zotero-test" in settings_js
    assert "data-settings-audit-refresh" in settings_js
    assert "settingsZoteroTest" in settings_js
    assert "settingsAuditEvents" in settings_js
    assert "Recent audit events" in settings_js
    assert "连接测试" in settings_js
    assert 'body[data-reduce-motion="true"]' in tweaks_css
    assert "css/tweaks.css?v=20260625-stage96" in index_html
    assert "css/settings.css?v=20260812-extension-manager1" in index_html
    assert "js/screens-settings-extensions.js?v=20260812-extension-manager1" in index_html
    assert "js/screens-settings.js?v=20260817-copilot-boundary1" in index_html
    assert ".settings-cap-panel" in settings_css
    assert ".settings-cap-tabs" in settings_css
    assert ".settings-cap-tile" in settings_css
    assert ".settings-cap-control" in settings_css
    assert ".settings-zotero-test" in settings_css
    assert ".settings-audit-log" in settings_css
    assert ".settings-audit-list" in settings_css
    assert ".settings-ext-manager" in settings_css
    assert ".settings-ext-card" in settings_css
    assert ".settings-cap-panel" not in _static_css("pages.css")
    assert ".settings-cap-panel" not in _static_css("screens.css")
    assert ".settings-cap-panel" not in _static_css("redesign.css")
    assert ".settings-zotero-test" not in _static_css("screens.css")
    assert ".settings-audit-log" not in _static_css("redesign.css")
    assert ".settings-ext-manager" not in _static_css("screens.css")
    assert ".settings-ext-manager" not in _static_css("redesign.css")
    assert "window.EU_API.saveSetting('data_mode', m)" in i18n_js
    assert "All controls are demo-interactive" not in settings_js


def test_native_extraction_omits_unbound_registered_source_metadata() -> None:
    api_js = _static_js("api.js")
    extraction_js = _static_js("screens-extraction.js")

    assert "/api/fs/mkdir" in api_js
    assert "function createDir(path)" in api_js
    assert "window.EU_API.createDir = createDir" in api_js
    assert "/api/extraction/filter-options" in api_js
    assert "/api/extraction/filter-preview" in api_js
    assert "loadExtractionFilterOptions" not in extraction_js
    assert "Registered source metadata" not in extraction_js
    assert "sourceMetadataBody" not in extraction_js
    assert "previewExtractionFilters" not in extraction_js
    assert "Minimum module coverage" not in extraction_js
    assert "Quality status" not in extraction_js
    assert "data-ex-filter-coverage" not in extraction_js
    assert "data-ex-filter-quality" not in extraction_js
    assert "Use matched modules" not in extraction_js


def test_native_extraction_folder_connect_defaults_to_auto_detection() -> None:
    extraction_js = _static_js("screens-extraction.js")
    extraction_css = _static_css("extraction.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert "css/extraction.css?v=20260630-gate-first-ia" in index_html
    assert "data-ex-analyze" in extraction_js
    assert "Choose folder and identify" in extraction_js
    assert "function copilotPrefillSummary()" in extraction_js
    assert "Loaded from Copilot" in extraction_js
    assert "Select the ICU data folder" in extraction_js
    assert "function pathNeedsFolderChoice(path)" in extraction_js
    assert "if (pathNeedsFolderChoice(exPath))" in extraction_js
    assert "const chooseDataFolder = () =>" in extraction_js
    assert "startScan(null);" in extraction_js
    assert "Let EasyICU identify the folder" not in extraction_js
    assert "Analyze folder" not in extraction_js
    assert "data-ex-manual" in extraction_js
    assert "Advanced: choose manually" in extraction_js
    assert "Use this only if automatic detection is wrong" in extraction_js
    assert "Then tell us what kind of folder it is" not in extraction_js
    assert ".ex-connect-card" in extraction_css
    assert ".ex-connect-primary" in extraction_css
    assert ".ex-connect-actions" in extraction_css
    assert ".ex-connect-card" not in redesign_css
    assert ".ex-connect-primary" not in redesign_css


def test_native_extraction_custom_modules_default_empty_with_bulk_actions() -> None:
    extraction_js = _static_js("screens-extraction.js")
    embedded_js = _static_js("screens-extraction-embedded.js")
    sepsis_js = _static_js("screens-extraction-sepsis.js")
    extraction_css = _static_css("extraction.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert (
        "let exAdvCohort = true, exAdvExport = false, exShowAllMods = true, exIncludeDefinitions = true;"
        in extraction_js
    )
    assert "let exCustomOpen = false;" in extraction_js
    assert (
        "if (window.__euExtractFocusICD) { exAdvCohort = true; exCustomOpen = true; exCohortPreset = 'icd'; }"
        in extraction_js
    )
    module_block = extraction_js.split("const MODS = [", 1)[1].split("];", 1)[0]
    assert module_block.count(", false, true]") == 6
    assert module_block.count(", false, false]") == 13
    assert ", true, true]" not in module_block
    assert ", true, false]" not in module_block
    assert (
        "const saved = Array.isArray(exSelectedConcepts[key]) ? exSelectedConcepts[key] : (m[3] ? ids : []);"
        in extraction_js
    )
    assert (
        "function setAllModules(on) { MODS.forEach(m => { m[3] = !!on; }); exSelectedConcepts = {};"
        in extraction_js
    )
    assert "function extractionSetupSummary()" in extraction_js
    assert "setupSummary: extractionSetupSummary" in extraction_js
    assert "const summary = owner.setupSummary();" in embedded_js
    assert "custom.querySelector('[data-ex-run=\"custom\"]')" in embedded_js
    assert 'data-ex-run="custom"' in embedded_js
    assert 'data-ex-run="recommended"' not in embedded_js
    assert "function escapeHtml(value)" in embedded_js
    assert "escHtml(" not in embedded_js
    assert "data-ex-selectall" in extraction_js
    assert "Select all" in extraction_js
    assert "data-ex-clearmods" in extraction_js
    assert "Clear all" in extraction_js
    assert "Core 6" in extraction_js
    assert "data-ex-mod-details" in extraction_js
    assert "data-ex-concept" in extraction_js
    assert "data-ex-concepts-all" in extraction_js
    assert "data-ex-concepts-clear" in extraction_js
    assert "selectedConceptPayload" in extraction_js
    assert "payload.concepts = conceptSelection" in extraction_js
    assert "include_feature_definitions: exIncludeDefinitions" in extraction_js
    assert "Feature definition manifest" in extraction_js
    assert "What will be written?" in extraction_js
    assert "easyicu.api.load_concepts" in extraction_js
    assert "not_declared_in_current_catalog" in extraction_js
    assert 'data-ex-switch="definitions"' in extraction_js
    assert "feature_definitions.json" in extraction_js
    assert ".ex-definition-option" in extraction_css
    assert ".def-example-grid" in extraction_css
    assert "function sepsisDefinitionPanel()" in extraction_js
    assert "sepsis_definition: sepsisDefinitionContract()" in extraction_js
    assert "window.EUExtractionSepsis.bind" in extraction_js
    assert (
        "js/screens-extraction-sepsis.js?v=20260629-sepsis-anchor-readonly"
        in index_html
    )
    assert index_html.find("screens-extraction-sepsis.js") < index_html.find(
        "screens-extraction.js"
    )
    assert "window.EUExtractionSepsis = {" in sepsis_js
    assert "detailsOpen: false" in sepsis_js
    assert "state.detailsOpen = !!details.open" in sepsis_js
    assert "function refreshPanel(root, ctx)" in sepsis_js
    assert "refreshPanel(root, ctx);" in sepsis_js
    assert "ctx.repaintStable" not in sepsis_js
    assert "ctx.repaint()" not in sepsis_js
    assert "metadata_current_runtime_defaults" in sepsis_js
    assert "Sepsis-3 definition locked" in sepsis_js
    assert "not part of the normal setup flow" in sepsis_js
    assert "Advanced audit details" in sepsis_js
    assert "Audit anchors" in sepsis_js
    assert "sepsis-def-static" in sepsis_js
    assert "Advanced callback kwargs are intentionally not exposed here" in sepsis_js
    assert "sepsis-def-audit-strip" in sepsis_js
    assert "sepsis-def-details" in sepsis_js
    assert "const IMPLEMENTATION_PROFILE = 'selected_module_defaults'" in sepsis_js
    assert "const SCORE_FAMILY = 'module-specific SOFA source'" in sepsis_js
    assert "['icd_abx', 'ICD + ABX fallback'" not in sepsis_js
    assert "eICU-only fallback: infection ICD + ABX" in sepsis_js
    assert "仅 eICU 兜底：感染 ICD + 抗菌药" in sepsis_js
    assert "Auto by database" not in sepsis_js
    assert "按数据库自动" not in sepsis_js
    assert "optionSeg(ctx, 'si_mode'" not in sepsis_js
    assert 'data-ex-sepsis="si_mode"' not in sepsis_js
    assert "Score source" not in sepsis_js
    assert "评分来源" not in sepsis_js
    assert "data-ex-sepsis-profile" not in sepsis_js
    assert "root.querySelectorAll('[data-ex-sepsis-profile]')" not in sepsis_js
    assert 'data-ex-sepsis="${key}"' in sepsis_js
    assert "root.querySelectorAll('[data-ex-sepsis]')" in sepsis_js
    assert "definition_locked: true" in sepsis_js
    assert "abx_count_win_hours: LOCKED.abxCountWinHours" in sepsis_js
    assert "abx_min_count: LOCKED.abxMinCount" in sepsis_js
    assert "positive_cultures_required: LOCKED.positiveCultures" in sepsis_js
    assert "si_window: state.siWindow" in sepsis_js
    assert "delta_function: LOCKED.deltaFunction" in sepsis_js
    assert "threshold: LOCKED.threshold" in sepsis_js
    assert "optionSeg(ctx, 'threshold'" not in sepsis_js
    assert "Δ ≥ 3" not in sepsis_js
    assert "ABX only" not in sepsis_js
    assert "Sample only" not in sepsis_js
    assert "ABX or sample" not in sepsis_js
    assert 'data-ex-sepsis="abx_min_count"' not in sepsis_js
    assert 'data-ex-sepsis="delta_function"' not in sepsis_js
    assert "antibiotic_to_sample_hours: exSepsisAbxToSampleHours" not in extraction_js
    assert 'data-ex-run="recommended"' in extraction_js
    assert "function coreModuleKeys()" in extraction_js
    assert "Select at least one module before extracting." in extraction_js
    assert ".cohort-preset-grid" in extraction_css
    assert ".range-ctl" in extraction_css
    assert ".ex-export-destination" in extraction_css
    assert ".ex-export-browse" in extraction_css
    assert ".ex-export-create" in extraction_css
    assert ".modgrid" in extraction_css
    assert ".mod-concepts" in extraction_css
    assert ".modcard.open .modcard-head" in extraction_css
    assert ".modcard.open .mod-detail-btn" in extraction_css
    assert ".mod-concepts::before" in extraction_css
    assert ".concept-toggle" in extraction_css
    assert (
        ".concept-toggle.on{\n  border-color:color-mix(in srgb,var(--accent-border) 58%,var(--hair));\n  background:var(--surface);"
        in extraction_css
    )
    assert ".sepsis-def-panel" in extraction_css
    assert ".sepsis-def-audit-strip" in extraction_css
    assert ".sepsis-def-details-lite" in extraction_css
    assert ".sepsis-def-details summary" in extraction_css
    assert ".sepsis-def-detail-title" in extraction_css
    assert ".sepsis-def-grid" in extraction_css
    assert ".sepsis-def-grid.compact" in extraction_css
    assert ".sepsis-def-control" in extraction_css
    assert ".sepsis-def-seg button.active" in extraction_css
    assert ".sepsis-def-chip.current" in extraction_css
    assert ".ex2-summary{ align-self:stretch; min-width:0; }" in extraction_css
    assert ".sumcard{\n  position:sticky; top:74px; z-index:3;" in extraction_css
    assert "max-height:calc(100vh - 92px); overflow:auto;" in extraction_css
    assert (
        ".sumcard{ position:static; max-height:none; overflow:visible; }"
        in extraction_css
    )
    assert ".cohort-preset-grid" not in redesign_css
    assert ".range-ctl" not in redesign_css
    assert ".ex-export-destination" not in redesign_css
    assert ".sepsis-def-panel" not in redesign_css
    assert ".concept-toggle" not in redesign_css
    assert ".modcard.open .modcard-head" not in redesign_css
    assert ".mod-concepts::before" not in redesign_css
    assert ".ex2-summary" not in redesign_css


def test_native_extraction_prefers_parquet_export_by_default() -> None:
    extraction_js = _static_js("screens-extraction.js")

    assert "let exFormat = 'parquet';" in extraction_js
    assert (
        'data-val="parquet">Parquet</button><button class="${exFormat === \'csv\''
        in extraction_js
    )
    assert "let exExportDir = null;" in extraction_js
    assert "data-ex-export-browse" in extraction_js
    assert "data-ex-export-create" in extraction_js
    assert "Choose or create export destination" in extraction_js
    assert "选择或创建导出目录" in extraction_js
    assert "Create folder" in extraction_js
    assert "No export destination selected" in extraction_js
    assert "尚未选择导出目录" in extraction_js
    assert "Choose an export destination before extracting." in extraction_js
    assert "请先选择导出目录再开始抽取。" in extraction_js
    assert "const exportReady = !!currentExportDir();" in extraction_js
    assert (
        "extractDisabled = !selMods().length || !support.ok || !exportReady"
        in extraction_js
    )
    assert "window.EU_API.saveSetting('export_dir', exExportDir)" in extraction_js
    assert "payload.out_dir = outDir" in extraction_js
    assert "if (outDir) payload.out_dir = outDir" not in extraction_js
    assert "rememberExportPath(exportResult.out_dir)" in extraction_js
    assert "window.EU_PATIENT_SOURCES = null;" in extraction_js
    assert "~/easyicu/exports/${dataMode()}" not in extraction_js
    assert "Default export root from Settings" not in extraction_js
    assert "timestamped folder inside the selected destination" in extraction_js
    assert "human-readable extraction README" in extraction_js


def test_native_ui_does_not_prefill_author_machine_paths() -> None:
    static_payload = "\n".join(
        [
            _static_js("screens-extraction.js"),
            _static_js("screens-viz.js"),
            _static_js("screens-guided.js"),
            _static_js("screens-agent.js"),
            _static_js("screens-settings.js"),
        ]
    )

    for forbidden in [
        "/Users/haibo",
        "/Volumes/外置硬盘",
        "/Volumes/data",
        "~/easyicu/exports",
        "~/easyicu/workspace",
    ]:
        assert forbidden not in static_payload
    assert 'value="~/easyicu/projects' not in static_payload


def test_native_extraction_module_counts_match_backend_catalog() -> None:
    extraction_js = _static_js("screens-extraction.js")

    module_block = extraction_js.split("const MODS = [", 1)[1].split("];", 1)[0]
    key_block = extraction_js.split("const EX_KEYS = {", 1)[1].split("};", 1)[0]
    entries = re.findall(
        r"\['([^']+)',\s*'[^']+',\s*(\d+),\s*(true|false),\s*(true|false)\]",
        module_block,
    )
    keys = dict(re.findall(r"'([^']+)':\s*'([^']+)'", key_block))

    assert len(entries) == len(cc.CONCEPT_GROUPS_INTERNAL) == 19
    assert "function moduleConceptCount(m)" in extraction_js
    assert "window.EU_CATALOG && window.EU_CATALOG.groupConcepts" in extraction_js

    fallback_total = 0
    for name, count_text, selected, _is_core in entries:
        group_key = keys[name]
        expected = len(cc.CONCEPT_GROUPS_INTERNAL[group_key])
        count = int(count_text)
        fallback_total += count
        assert selected == "false", f"{name} should require an explicit user selection"
        assert count == expected, f"{name} fallback count should match {group_key}"

    assert fallback_total == len(cc.CONCEPT_DICTIONARY)
    assert fallback_total != 219


def test_native_idea_mining_is_first_class_route_and_backend_wired() -> None:
    app_js = _static_js("app.js")
    api_js = _static_js("api.js")
    icons_js = _static_js("icons.js")
    ideas_js = _static_js("screens-ideas.js")
    ideas_zotero_js = _static_js("screens-ideas-zotero.js")
    agent_js = _static_js("screens-agent.js")
    redesign_css = _static_css("redesign.css")
    ideas_css = _static_css("ideas.css")
    ideas_review_css = _static_css("ideas-review.css")
    ideas_connectors_css = _static_css("ideas-connectors.css")
    shell_css = _static_css("shell.css")
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'data-nav="ideas"' in app_js
    # One destination, one name: the sidebar carries the same label as the crumb
    # and page title (Idea Mining / 想法挖掘), not a third alias.
    assert "Idea Mining" in app_js
    assert "Find a Study Idea" not in app_js
    assert "Discovery & Plan" in app_js
    assert "ideas-entry" in app_js
    assert "paper, PDF, or topic → feasible plan" in app_js
    assert "Project Monitor" in app_js
    assert "runs · outputs · evidence · review" in app_js
    assert "Data & Review" in app_js
    assert "Data Workspace" in app_js
    # The misleading "N / 4" progress counter on the Data Workspace group was
    # removed: patient/cohort/crossdb are parallel review lenses, not sequential
    # steps, so a running counter framed them as ordered progress.
    assert "workspaceIndex + 1} / ${CLASSIC.length}" not in app_js
    assert 'class="wsg-prog"' not in app_js
    assert "function render(opts = {})" in app_js
    assert "const resetScroll = !!opts.resetScroll;" in app_js
    assert "window.__euRender = function (opts) { render(opts || {}); };" in app_js
    assert "render({ resetScroll: true });" in app_js
    assert "render();" in app_js
    assert "stale: true" not in app_js
    assert "jp-stale" not in app_js
    assert "t('stale', '过期')" not in app_js
    assert "window.EU_STALE && c.stale" not in app_js
    assert "Math.min(cur, goal)" not in app_js
    assert "window.__euRender = render;" not in app_js
    assert "Classic Workspace" not in app_js
    assert "Copilot and Classic share one study" not in app_js
    assert "Already have data? Start with Extract Data." in app_js
    assert "wsi-sub" in app_js
    assert "css/ideas.css?v=20260803-owner-migration" in index_html
    assert "css/shell.css?v=20260812-route-a11y1" in index_html
    assert "js/icons.js?v=20260825-message-actions1" in index_html
    assert "js/app.js?v=20260826-copilot-home1" in index_html
    assert "css/ideas-review.css?v=20260702-idea-review-handoff" in index_html
    assert "css/ideas-connectors.css?v=20260702-zotero-simple" in index_html
    assert "js/screens-ideas-zotero.js?v=20260702-zotero-origin" in index_html
    assert "js/screens-ideas.js?v=20260817-copilot-boundary1" in index_html
    assert "discoverIdeas" in api_js
    assert "/api/ideas/discover" in ideas_js
    assert "Discover papers" in ideas_js
    assert "data-idea-use-discovery" in ideas_js
    assert "/api/ideas/mine" in api_js
    assert "/api/ideas/resolve-source" in api_js
    assert "/api/ideas/ingest-pdf" in api_js
    assert "/api/ideas/literature-folder" in api_js
    assert "/api/ideas/prior-art" in api_js
    assert "/api/ideas/plan" in api_js
    assert "planIdea" in api_js
    assert "/api/ideas/bounded-feasibility" in api_js
    assert "checkIdeaSampleFeasibility" in api_js
    assert "/api/ideas/handoff" in api_js
    assert "/api/ideas/create-agent-project" in api_js
    assert "/api/ideas/agent-projects" in api_js
    assert "/api/ideas/history" in api_js
    assert "/api/ideas/run" in api_js
    assert "S.ideas =" in ideas_js
    assert "pre_experiment" in ideas_js
    assert "Plan / replan before Agent" in ideas_js
    assert "Generate study plan" in ideas_js
    assert "Feasibility assessment on active export" in ideas_js
    assert "Run bounded sample check" in ideas_js
    assert "data-idea-sample-feasibility" in ideas_js
    assert "Literature inspiration" in ideas_js
    assert "ideas-plan-step" in ideas_js
    assert "normalizePlanStep" in ideas_js
    assert "Check literature" in ideas_js
    assert "Reference method patterns" in ideas_js
    assert "ICU constraints" in ideas_js
    assert (
        "Generate and review the study plan before freezing an Agent handoff"
        in ideas_js
    )
    assert "data-idea-plan" in ideas_js
    assert "data-idea-replan" in ideas_js
    assert "Freeze handoff for Agent" in ideas_js
    assert "selectedRecordKey" in ideas_js
    assert "data-idea-record-key" in ideas_js
    assert "if (!rows.some(r => String(r.id) === String(current.id)))" in ideas_js
    assert "Create project seed" in ideas_js
    assert "Resolve source" in ideas_js
    assert "sourceResolvedNote" in ideas_js
    assert "display_status" in ideas_js
    assert "Source ready" in ideas_js
    assert "来源已就绪" in ideas_js
    assert "loadIdeaRun" in ideas_js
    assert "data-idea-record" in ideas_js
    assert "postLocalJSON('/api/ideas/run'" in ideas_js
    assert "applyRunPayload(data, recordKey)" in ideas_js
    assert "upsertHistoryRun(data)" in ideas_js
    assert "activeStep = 'source';" in ideas_js
    assert "data-idea-step" in ideas_js
    assert 'button type="button" class="${cls}" data-idea-step' in ideas_js
    assert "'ideas-summary-item'" in ideas_js
    assert "function stepNav" not in ideas_js
    assert "Create idea ledger" in ideas_js
    assert "ideas-advanced" in ideas_js
    assert "Network and provider opt-in" in ideas_js
    assert "idea-workbench" in ideas_js
    assert "ideas-work-grid" in ideas_js
    assert "Local idea runs" in ideas_js
    assert "stored on this machine · not Agent analysis runs" in ideas_js
    assert "Idea Mining decides what is worth running" in ideas_js
    assert "ideas-source-card" in ideas_js
    assert "idea-ledger-card" in ideas_js
    assert "idea-ledger-grid" in ideas_js
    assert "ideas-pre-summary" in ideas_js
    assert "ideas-feature-row" in ideas_js
    assert "ideas-compact-details" in ideas_js
    assert "ideas-prior-card" in ideas_js
    assert "ideas-prior-gate" in ideas_js
    assert "ideas-query-details" in ideas_js
    assert "ideas-plan-edits" in ideas_js
    assert "ideas-handoff-receipt" in ideas_js
    assert "Source opt-in required" in ideas_js
    assert "pubmedConnectorEnabled" in ideas_js
    assert "sourceNetworkOptIn" in ideas_js
    assert "window.EU_CAPABILITIES" in ideas_js
    assert "connector_pubmed_enabled" in ideas_js
    assert "PubMed connector is off" in ideas_js
    assert "data-idea-open-settings" in ideas_js
    assert "Project seed ready" in ideas_js
    assert "可作为结果报告" in ideas_js
    assert "querySelectorAll('#ideaNetworkOptIn')" in ideas_js
    assert "search: '<circle" in icons_js
    assert "rail-block" in ideas_js
    assert "setup-row" in ideas_js
    assert "rail-title" not in ideas_js
    assert "rail-kv" not in ideas_js
    assert "rail-note" not in ideas_js
    assert "grid4 mt-12" not in ideas_js
    assert "css/ideas.css" in index_html
    assert ".idea-workbench" in ideas_css
    assert ".ideas-source-form" in ideas_css
    assert ".ideas-source-gate" in ideas_css
    assert ".ideas-secondary-fields" in ideas_css
    assert ".ideas-list-context" in ideas_css
    assert ".ideas-url-stack" in ideas_css
    assert ".ideas-source-picker" in ideas_css
    assert ".ideas-folder-source" in ideas_css
    assert ".ideas-step-nav" not in ideas_css
    assert ".ideas-advanced" in ideas_css
    assert ".ideas-source-card" in ideas_css
    assert ".idea-ledger-card" in ideas_css
    assert ".idea-ledger-grid" in ideas_css
    assert ".ideas-pre-summary" in ideas_css
    assert ".ideas-feature-row" in ideas_css
    assert ".ideas-query-details" in ideas_css
    assert ".ideas-prior-card" in ideas_css
    assert ".ideas-plan-edits" in ideas_css
    assert ".ideas-prior-gate" in ideas_review_css
    assert ".ideas-handoff-receipt" in ideas_review_css
    assert ".ideas-handoff-actions" in ideas_review_css
    assert "height:auto;" in ideas_css
    assert "min-height:92px;" in ideas_css
    assert "grid-template-columns:repeat(2,minmax(0,1fr));" in ideas_css
    assert "@media (min-width:1680px)" not in ideas_css
    assert ".idea-workbench .ideas-step-panel .statgrid" in ideas_css
    assert ".idea-workbench .ideas-step-panel table" in ideas_css
    assert ".idea-workbench" not in redesign_css
    assert ".ideas-primary-grid" not in redesign_css
    assert ".ideas-source-gate" not in redesign_css
    assert ".ideas-secondary-fields" not in redesign_css
    assert ".ideas-list-context" not in redesign_css
    assert ".ideas-step-nav" not in redesign_css
    assert ".ideas-advanced" not in redesign_css
    assert ".ideas-entry" not in redesign_css
    assert ".ideas-prior-gate" not in redesign_css
    assert ".ideas-handoff-receipt" not in redesign_css
    assert ".ideas-handoff-receipt" not in ideas_css
    assert ".wsi-sub" not in redesign_css
    assert ".nav-sec" in shell_css
    assert ".ideas-entry" in shell_css
    assert ".wsi-sub" in shell_css
    assert "function validatePayload(payload)" in ideas_js
    assert "function sourceSpecificForm()" in ideas_js
    assert "function sourceModeGuide()" in ideas_js
    assert "function optionalMetadataBlock(body, title)" in ideas_js
    assert "function ideaListContext(rows)" in ideas_js
    assert "ideas-source-form manual" in ideas_js
    assert "ideas-source-form url" in ideas_js
    assert "ideas-source-form pdf" in ideas_js
    assert "ideas-source-form literature_folder" in ideas_js
    assert "ideas-source-form frontier" in ideas_js
    assert "Optional article metadata" in ideas_js
    assert "Optional PDF metadata" in ideas_js
    assert "These are metadata-only idea ledgers from this machine" in ideas_js
    assert "Manual idea" in ideas_js
    assert "Article URL" in ideas_js
    assert "PDF file" in ideas_js
    assert "Literature folder" in ideas_js
    assert "Frontier topic" in ideas_js
    assert "Article URL mode resolves bounded metadata" in ideas_js
    assert "Choose a local PDF or paste a bounded PDF excerpt before mining" in ideas_js
    assert "Literature folder mode scans local PDFs" in ideas_js
    assert "data-idea-pdf-pick" in ideas_js
    assert "data-idea-lit-scan" in ideas_js
    assert "Zotero library" in ideas_js
    assert "ideas-source-form zotero" in ideas_js
    assert "window.EU_IDEA_ZOTERO.create" in ideas_js
    assert "data-idea-zotero-search" in ideas_zotero_js
    assert "data-idea-use-zotero" in ideas_zotero_js
    assert "data-idea-zotero-import" in ideas_zotero_js
    assert "ideaZoteroPaste" in ideas_zotero_js
    assert "Use pasted source" in ideas_zotero_js
    assert "使用粘贴文献" in ideas_zotero_js
    assert "Literature source ready" in ideas_zotero_js
    assert "文献来源已就绪" in ideas_zotero_js
    assert "No Zotero setup required" in ideas_zotero_js
    assert "不需要配置 Zotero" in ideas_zotero_js
    assert "window.EU_API.zoteroSource" in ideas_zotero_js
    assert "window.EU_API.importZoteroSource" in ideas_zotero_js
    assert (
        "setSourceType(value) { srcType = value; draft.source_type = value; }"
        in ideas_js
    )
    assert "citation_key" in ideas_js
    assert "zotero_key" in ideas_js
    assert "source_origin" in ideas_js
    assert "source_origin_label" in ideas_js
    assert "pasted_zotero_source_ready" not in ideas_js
    assert "pasted_zotero_source_ready" not in ideas_zotero_js
    assert "zotero_source_ready" not in ideas_js
    assert "zotero_source_ready" not in ideas_zotero_js
    assert ".ideas-zotero-source" in ideas_connectors_css
    assert ".ideas-zotero-results" in ideas_connectors_css
    assert ".ideas-zotero-row" in ideas_connectors_css
    assert ".ideas-zotero-paste" in ideas_connectors_css
    assert ".ideas-zotero-source" not in ideas_css
    assert ".ideas-zotero-source" not in ideas_review_css
    assert ".ideas-zotero-source" not in redesign_css
    assert ".ideas-zotero-paste" not in ideas_css
    assert ".ideas-zotero-paste" not in redesign_css
    assert "loadIdeaAgentProjects" in agent_js
    assert "seedStudy(row)" in agent_js
    # DEMO_STUDIES data moved to screens-agent-render.js; the consumer stays.
    assert "const DEMO_STUDIES" in _static_js("screens-agent-render.js")
    assert "const base = realMode() ? [] : DEMO_STUDIES" in agent_js
    assert "No projects to monitor yet" in agent_js
    # Real mode must not fabricate studies or collect setup on the monitor.
    assert "Start a study in Guided Copilot" in agent_js
    assert "Open Guided Copilot" in agent_js
    assert "data-ag-new" not in agent_js
    assert "data-ag-runbtn" not in agent_js
    assert "data-ag-mode" not in agent_js
    assert "Idea exploration" not in agent_js
    assert "Open Idea Mining" not in agent_js


def test_native_extraction_cohort_controls_are_continuous_and_icd_is_empty() -> None:
    extraction_js = _static_js("screens-extraction.js")
    icd_js = _static_js("screens-icd.js")

    assert 'data-ex-range="age_min"' in extraction_js
    assert 'data-ex-range="age_max"' in extraction_js
    assert "rangeCtl('los_min'" in extraction_js
    assert "rangeCtl('window'" in extraction_js
    assert 'type="range"' in extraction_js
    assert "data-ex-cohort-preset" in extraction_js
    assert "All ICU stays" in extraction_js
    assert "AKI / renal dysfunction" in extraction_js
    assert "Mechanical ventilation" in extraction_js
    assert "Vasopressor exposure" in extraction_js
    assert "Diagnosis / ICD cohort" in extraction_js
    assert "Sepsis-3 positive only" not in extraction_js
    assert "icd_enabled: exCohortPreset === 'icd'" in extraction_js
    assert "window.EUIcd.contract" in extraction_js
    assert "icd_include" in _static_js("screens-icd.js")
    assert "let icdInclude = '';" in icd_js
    assert "let icdInclude = 'A41, R65'" not in icd_js
    assert "ICD_TOTAL" not in icd_js
    assert "tokenFraction" not in icd_js
    assert "Top matching ICD codes" not in icd_js
    assert "MIMIC-IV · MIMIC-III · eICU" not in icd_js
    assert "No estimated patient count or synthetic code frequency is shown" in icd_js
    assert "window.EUIcd.block(icdSourceContext())" in extraction_js


def test_native_extraction_exposes_real_cohort_gate_and_recommended_contract() -> None:
    extraction_js = _static_js("screens-extraction.js")

    assert "const DEFAULT_OBSERVATION_WINDOW_HOURS = 24 * 30;" in extraction_js
    assert "let exWindowHours = DEFAULT_OBSERVATION_WINDOW_HOURS;" in extraction_js
    assert "observation_window_hours: DEFAULT_OBSERVATION_WINDOW_HOURS" in extraction_js
    assert "MAX_OBSERVATION_WINDOW_HOURS" in extraction_js
    assert "full available · 30d cap" in extraction_js
    assert "first 24 hours" not in extraction_js
    assert "first 24h" not in extraction_js
    assert "let exMaxPatients = 0;" in extraction_js
    assert "let exCohortPreset = 'all_icu';" in extraction_js
    assert "let exAgeMin = 0;" in extraction_js
    assert "let exExcludeReadmissions = false;" in extraction_js
    assert "preset: 'all_icu'" in extraction_js
    assert "age_min: 0" in extraction_js
    assert "exclude_readmissions: false" in extraction_js
    assert (
        "const REAL_EXPORT_COHORT_PRESETS = new Set(['all_icu', 'adult_first', "
        "'adult_all', 'sepsis3', 'aki', 'ventilation', 'vasopressor', 'respiratory', 'icd']);"
    ) in extraction_js
    assert "planned, not export-ready" in extraction_js
    assert "concept prefilter, slower" in extraction_js
    assert "Clinical cohort prefilter" in extraction_js
    assert "Real export is blocked for this cohort" in extraction_js
    assert (
        "Add at least one ICD include or exclude token before exporting"
        in extraction_js
    )
    assert (
        "const support = runMode === 'recommended' ? { ok: true, reason: 'recommended' } : cohortExportSupport();"
        in extraction_js
    )
    assert (
        "cohort: runMode === 'recommended' ? recommendedCohortContract() : cohortContract()"
        in extraction_js
    )
    assert "const progressText = p.message" in extraction_js
    assert "data-ex-cancel" in extraction_js
    assert "'/api/jobs/' + exportJobId + '/cancel'" in extraction_js
    assert (
        "Cancel accepted. Stopping the current database query"
        in extraction_js
    )


def test_native_extraction_manifest_records_sepsis_definition_metadata() -> None:
    cohort = dataio._normalize_export_cohort(
        {
            "preset": "adult_first",
            "sepsis_definition": {
                "runtime_profile": "ui-test",
                "implementation_profile": "sofa1_sensitivity",
                "score_family": "SOFA-2 + SOFA-1",
                "suspected_infection": {
                    "mode": "abx",
                    "antibiotic_to_sample_hours": 48,
                    "sample_to_antibiotic_hours": 24,
                    "abx_count_win_hours": 12,
                    "abx_min_count": 2,
                    "positive_cultures_required": True,
                },
                "sofa_increase": {
                    "si_event": "last",
                    "window_before_si_hours": 72,
                    "window_after_si_hours": 12,
                    "delta_function": "first_observed",
                    "threshold": 3,
                    "keep_components": True,
                },
            },
        }
    )

    sepsis_definition = cohort["sepsis_definition"]
    assert sepsis_definition["record_scope"] == "metadata_current_runtime_defaults"
    assert sepsis_definition["runtime_profile"] == "ui-test"
    assert sepsis_definition["implementation_profile"] == "sofa1_sensitivity"
    assert sepsis_definition["score_family"] == "SOFA-1"
    assert sepsis_definition["definition_locked"] is True
    assert sepsis_definition["suspected_infection"]["mode"] == "auto"
    assert sepsis_definition["suspected_infection"]["abx_win_hours"] == 24
    assert sepsis_definition["suspected_infection"]["samp_win_hours"] == 72
    assert sepsis_definition["suspected_infection"]["abx_count_win_hours"] == 24
    assert sepsis_definition["suspected_infection"]["abx_min_count"] == 1
    assert (
        sepsis_definition["suspected_infection"]["positive_cultures_required"] is False
    )
    assert sepsis_definition["sofa_increase"]["si_window"] == "first"
    assert sepsis_definition["sofa_increase"]["window_before_si_hours"] == 48
    assert sepsis_definition["sofa_increase"]["window_after_si_hours"] == 24
    assert sepsis_definition["sofa_increase"]["delta_function"] == "delta_cummin"
    assert sepsis_definition["sofa_increase"]["threshold"] == 2
    assert sepsis_definition["sofa_increase"]["keep_components"] is False
    runtime_kwargs = dataio._sepsis_runtime_kwargs(sepsis_definition)
    assert runtime_kwargs == {
        "si_mode": "auto",
        "abx_win": "24h",
        "samp_win": "72h",
        "abx_count_win": "24h",
        "abx_min_count": 1,
        "positive_cultures": False,
        "si_window": "first",
        "delta_fun": "delta_cummin",
        "sofa_thresh": 2,
        "si_lwr": "48h",
        "si_upr": "24h",
        "keep_components": False,
    }
    assert "implementation_profile" not in sepsis_definition["review_options"]
    assert "score_family" not in sepsis_definition["review_options"]
    assert "si_mode" not in sepsis_definition["review_options"]
    assert sepsis_definition["review_options"]["si_window"] == ["first", "any"]
    assert "threshold" not in sepsis_definition["review_options"]
    assert "abx_min_count" not in sepsis_definition["review_options"]
    assert "fixed_threshold" not in sepsis_definition["review_options"]

    readme = dataio._render_export_readme(
        {
            "generated": "2026-06-26T00:00:00",
            "database": "miiv",
            "data_path": "/tmp/icu",
            "format": "parquet",
            "max_patients": 500,
            "cohort_contract": cohort,
            "cohort_report": {"cohort_size": 10},
        },
        files=[
            {
                "file": "sepsis3_sofa2.parquet",
                "module": "sepsis3_sofa2",
                "concepts": 1,
                "rows": 10,
            }
        ],
        definition_files=[
            {
                "file": "feature_definitions.json",
                "records": 1,
            },
            {
                "file": "feature_definitions.csv",
                "records": 1,
            },
        ],
    )

    assert "Sepsis runtime profile: `ui-test`" in readme
    assert "Sepsis implementation profile: `sofa1_sensitivity`" in readme
    assert "Sepsis score family: `SOFA-1`" in readme
    assert "ABX->sample `24h`, sample->ABX `72h`" in readme
    assert "ABX count `≥1/24h`, positive cultures `False`" in readme
    assert (
        "SI event `first`, window `-48h/+24h`, "
        "delta `delta_cummin`, threshold `2`, keep components `False`"
    ) in readme
    assert "Sepsis runtime kwargs: `{'si_mode': 'auto'" in readme
    assert "Definition note scope: `metadata_current_runtime_defaults`" in readme
    assert "`feature_definitions.json` and `feature_definitions.csv`" in readme
    assert "`concept_id=age`, `module=demographics`, `unit=years`" in readme
    assert "`raw_metadata_status=not_declared_in_current_catalog`" in readme
    dataio_py = Path(dataio.__file__).read_text(encoding="utf-8")
    callbacks_py = (
        Path(dataio.__file__).resolve().parents[1] / "concept" / "callbacks.py"
    ).read_text(encoding="utf-8")
    assert "sepsis_load_kwargs = _sepsis_runtime_kwargs" in dataio_py
    assert "module_kwargs.update(sepsis_load_kwargs)" in dataio_py
    assert "abx_count_win=abx_count_win" in callbacks_py
    assert "sofa_thresh=_callback_int" in callbacks_py


def test_native_extraction_feature_definition_manifest_records_callback_provenance(
    tmp_path: Path,
) -> None:
    import easyicu.api as easyicu_api

    payload = dataio._feature_definition_payload(
        database="miiv",
        data_path="/Volumes/example/miiv",
        export_path=tmp_path,
        concept_plan={
            "demographics": ["age"],
            "sepsis3_sofa2": ["sep3_sofa2"],
        },
        files=[
            {
                "file": "demographics.parquet",
                "module": "demographics",
                "concept_ids": ["age"],
                "rows": 10,
            },
            {
                "file": "sepsis3_sofa2.parquet",
                "module": "sepsis3_sofa2",
                "concept_ids": ["sep3_sofa2"],
                "rows": 4,
            },
        ],
        api_module=easyicu_api,
    )
    definition_files = dataio._write_feature_definition_files(tmp_path, payload)

    assert [item["file"] for item in definition_files] == [
        "feature_definitions.json",
        "feature_definitions.csv",
    ]
    exported = json.loads(
        (tmp_path / "feature_definitions.json").read_text(encoding="utf-8")
    )
    assert exported["schema_version"] == "easyicu_feature_definitions_v1"
    assert exported["record_count"] == 2
    assert exported["local_path_policy"] == (
        "absolute_paths_omitted_from_shareable_feature_definitions"
    )
    serialized = json.dumps(exported)
    assert "/Volumes/example/miiv" not in serialized
    assert str(tmp_path) not in serialized

    age = next(row for row in exported["records"] if row["concept_id"] == "age")
    assert age["unit"] == "years"
    assert age["source"]["export_files"] == ["demographics.parquet"]
    assert age["source"]["raw_metadata_status"] == "not_declared_in_current_catalog"
    assert age["source"]["data_source_ref"]["hint"] == "miiv"
    assert age["source"]["data_source_ref"]["absolute_path_omitted"] is True
    assert age["source"]["export_ref"]["absolute_path_omitted"] is True
    assert age["callback"]["import_path"] == "easyicu.api.load_concepts"
    assert age["callback"]["source_module_file"] == "src/easyicu/api/concepts.py"
    assert age["callback"]["source_file_ref"]["absolute_path_omitted"] is True
    assert age["callback"]["project_ref"]["hint"] == "EASYICU"

    sep3 = next(row for row in exported["records"] if row["concept_id"] == "sep3_sofa2")
    assert sep3["module"] == "sepsis3_sofa2"
    assert sep3["source"]["export_files"] == ["sepsis3_sofa2.parquet"]
    csv_text = (tmp_path / "feature_definitions.csv").read_text(encoding="utf-8")
    assert "callback_project_ref" in csv_text
    assert "callback_project_path" not in csv_text
    assert "/Volumes/example/miiv" not in csv_text
    assert str(tmp_path) not in csv_text
    assert "not_declared_in_current_catalog" in csv_text


def test_feature_definition_project_reference_is_checkout_name_independent(
    tmp_path: Path,
) -> None:
    project_ref = dataio._shareable_project_reference(
        tmp_path / "arbitrary-worktree-name"
    )

    assert project_ref["hint"] == "EASYICU"
    assert project_ref["absolute_path_omitted"] is True


def test_native_extraction_include_feature_definitions_bool_parsing() -> None:
    assert (
        _body_bool(
            {"include_feature_definitions": "false"},
            "include_feature_definitions",
            True,
        )
        is False
    )
    assert (
        _body_bool(
            {"include_feature_definitions": "true"},
            "include_feature_definitions",
            False,
        )
        is True
    )
    assert _body_bool({}, "include_feature_definitions", True) is True


def test_native_crossdb_uses_progressive_setup_and_one_chart_results() -> None:
    api_js = _static_js("api.js")
    viz_js = _static_js("screens-viz.js")
    setup_js = _static_js("screens-viz-crossdb-setup.js")
    charts_js = _static_js("screens-viz-crossdb-charts.js")
    results_js = _static_js("screens-viz-crossdb-results.js")
    continuity_js = _static_js("screens-viz-crossdb-job-continuity.js")
    progress_js = _static_js("screens-viz-crossdb-progress.js")
    screens_css = _static_css("screens.css")
    crossdb_css = _static_css("crossdb.css")
    unrelated_route_css = {
        name: _static_css(name)
        for name in (
            "agent.css",
            "cohort.css",
            "extraction.css",
            "guided.css",
            "ideas.css",
            "patient.css",
            "settings.css",
        )
    }
    index_html = _static_html("index.html")

    assert "startCrossdbRawDistributionJob" in api_js
    assert "/api/jobs/crossdb-raw-distribution" in api_js
    assert "startCrossdbReviewSummaryJob" in api_js
    assert "/api/jobs/crossdb-summary" in api_js
    assert "scanCrossdbRawRoot" in api_js
    assert "/api/crossdb-review/raw-root-scan" in api_js
    assert "loadDemoCrossdb" in viz_js
    assert "loadCrossdbDemoDistribution" in viz_js
    assert "legacy_simulated_multidb_feature_frames" in viz_js
    assert "feature_scope: 'all_catalog'" in viz_js
    assert "window.EU_CROSSDB_RAW.buildRequest" in viz_js
    assert "max_patients: sampleProfile.maxPatients" in viz_js
    assert "sample_size: sampleProfile.sampleSize" in viz_js

    # Setup is progressive: choose prepared exports or raw folders, then reveal
    # only the controls owned by the selected path.
    assert "sourceMethod: 'registered'" in setup_js
    assert "data-crossdb-source-method" in setup_js
    assert "data-crossdb-source-path" in setup_js
    assert 'class="crossdb-advanced mt-14"' in setup_js
    assert "Advanced settings (optional)" in setup_js
    assert "Start complete comparison" in setup_js
    assert "data-crossdb-root-scan" in setup_js
    assert "data-crossdb-select-detected" in setup_js
    assert "missingSelectedKeys.length === 0" in setup_js
    assert "data-crossdb-sample-mode" in setup_js
    assert "maxPatients: 200" in setup_js
    assert "sampleSize: 600" in setup_js
    assert "data-crossdb-run-raw" in setup_js
    assert "data-crossdb-run-demo" in setup_js

    # Results are a separate owner: four sections, complete catalog filtering,
    # one selected feature chart, and no repeated mini-plot grid.
    assert "window.EU_CROSSDB_RESULTS = {" in results_js
    assert "data-crossdb-result-tab" in results_js
    assert "data-crossdb-result-panel=\"overview\"" in results_js
    assert "data-crossdb-result-panel=\"coverage\"" in results_js
    assert "data-crossdb-result-panel=\"distributions\"" in results_js
    assert "data-crossdb-result-panel=\"quality\"" in results_js
    assert "data-crossdb-scope=\"all\"" in results_js
    assert "data-crossdb-feature-query" in results_js
    assert "class=\"xdb-main-chart\"" in charts_js
    assert "window.EU_CROSSDB_CHARTS = {" in charts_js
    assert "const chartOwner = window.EU_CROSSDB_CHARTS" in results_js
    assert "xdb-density-features" not in results_js
    assert "crossRealFeatureDensityByModule" not in viz_js
    assert "crossFeatureDensityPanel" not in viz_js
    assert "const crossResults = window.EU_CROSSDB_RESULTS" in viz_js
    assert "crossResults.bind(root" in viz_js

    # Existing raw-job privacy/cancellation fences stay intact.
    assert "new window.EventSource('/api/jobs/' + encodeURIComponent(meta.job_id) + '/events')" in continuity_js
    assert "data-crossdb-cancel" in progress_js
    assert "api.cancelJob(state.jobId, 'user_requested')" in progress_js
    assert "progress.requestCancel" in setup_js
    assert "crossdb-run-strip" in setup_js
    assert ".crossdb-run-strip" in crossdb_css
    assert "padding-right: 168px" in crossdb_css
    assert "scroll-margin-bottom: 84px" in crossdb_css

    assert "js/screens-viz-crossdb-setup.js?v=20260728-one-click-raw2" in index_html
    assert "js/screens-viz-crossdb-charts.js?v=20260728-shared-echarts1" in index_html
    assert "js/screens-viz-crossdb-results.js?v=20260817-copilot-boundary1" in index_html
    assert "js/screens-viz.js?v=20260823-native-preview1" in index_html
    assert "css/crossdb.css?v=20260728-one-click-raw1" in index_html
    for selector in (
        ".crossdb-method-grid",
        ".xdb-result-tabs",
        ".xdb-feature-workspace",
        ".xdb-main-chart",
    ):
        assert selector in crossdb_css
        assert selector not in screens_css
        for name, css in unrelated_route_css.items():
            assert selector not in css, f"Cross-DB CSS leaked into {name}"

def test_native_cohort_snapshot_renders_real_clinical_profile() -> None:
    """The real cohort snapshot must show interpretable clinical dimensions.

    Age/SOFA/LOS distributions remain available, but the old age x LOS proxy
    heatmap is not allowed to stand in for a cohort phenotype.
    """
    viz_js = _static_js("screens-viz.js")
    cohort_css = _static_css("cohort.css")
    index_html = _static_html("index.html")

    # Owner-file: distribution renderers live in screens-viz.js (viz route owner).
    assert "function cohortDistBars(" in viz_js
    assert "function cohortCompositionBars(" in viz_js
    # Real-data snapshot now binds backend bins for every population distribution.
    assert "cohortDistBars(s.age && s.age.bins)" in viz_js
    assert "cohortDistBars(s.sofa2 && s.sofa2.bins)" in viz_js
    assert "cohortDistBars(s.los_icu_days && s.los_icu_days.bins)" in viz_js
    assert "ICU LOS distribution" in viz_js
    assert "ICU 住院时长分布" in viz_js
    assert "Cohort composition" in viz_js
    assert "队列构成" in viz_js
    # Admission-type categorical distribution borrowed from the legacy dashboard.
    assert "cohortDistBars(s.admission.bins)" in viz_js
    assert "Admission type" in viz_js
    assert "入院类型" in viz_js
    # Clinical phenotype replaces the old age x LOS proxy heatmap.
    assert "function cohortClinicalProfile(" in viz_js
    assert "cohortClinicalProfile(s.clinical_profile)" in viz_js
    assert "Clinical phenotype" in viz_js
    assert "临床画像" in viz_js
    assert "function cohortComplexityHeatmap(" not in viz_js
    assert "Age × ICU LOS complexity" not in viz_js
    assert ".cprof-grid" in cohort_css
    assert ".cxh" not in cohort_css
    # Cache-bust bumped so the restored charts ship to existing clients.
    assert "js/screens-viz.js?v=20260823-native-preview1" in index_html


def test_native_cohort_groups_render_comparison_bar_chart() -> None:
    """The descriptive-split group view must visualise the per-metric profile as
    grouped bars (legacy cohort_group_page), not only the numeric table."""
    viz_js = _static_js("screens-viz.js")
    cohort_css = _static_css("cohort.css")

    assert "function cohortGroupComparisonChart(" in viz_js
    assert "cohortGroupComparisonChart(profileRows, profileColumns)" in viz_js
    # Reuses the bounded descriptive profile payload; no new inferential stats.
    assert "cohortProfileValue(row, v)" in viz_js
    assert ".cgc-bar" in cohort_css
    assert ".cgc-fill" in cohort_css
    # The exact-value profile table stays alongside the chart (additive).
    assert "Aggregate-only group characteristics" in viz_js


def test_native_patient_time_series_uses_module_grouped_single_feature_charts() -> None:
    """Single-patient Time Series should show module-grouped per-feature charts,
    not a same-module multi-signal overlay."""
    viz_js = _static_js("screens-viz.js")
    patient_charts_js = _static_js("screens-viz-patient-charts.js")
    patient_series_js = _static_js("screens-viz-patient-series.js")
    patient_css = _static_css("patient.css")
    patient_series_css = _static_css("patient-series.css")
    index_html = _static_html("index.html")

    assert "patientVitalTimeline" not in viz_js
    assert "patientVitalTimeline(readyLanes" not in viz_js
    assert "function renderModulePanels(" in patient_series_js
    assert "data-patient-series-module" in patient_series_js
    assert "Time series and feature catalog by module" in patient_series_js
    assert "按模块分组的时间序列与特征目录" in patient_series_js
    assert "window.EU_PATIENT_CHARTS" in patient_charts_js
    assert "window.echarts.init" in patient_charts_js
    assert "renderer: 'svg'" in patient_charts_js
    assert "renderMode: 'richText'" in patient_charts_js
    assert "smooth: false" in patient_charts_js
    assert "window.echarts.init" not in patient_series_js
    assert "window.echarts.init" not in viz_js
    assert ".pt-echart" in patient_series_css
    for unrelated_css in ("crossdb.css", "cohort.css", "redesign.css", "screens.css"):
        assert ".pt-echart" not in _static_css(unrelated_css)
    assert (
        index_html.index("vendor/echarts/echarts.common.min.js?v=6.1.0")
        < index_html.index("js/screens-viz-echarts.js?v=20260728-shared-echarts1")
        < index_html.index("js/screens-viz-patient-charts.js?v=20260728-shared-echarts1")
        < index_html.index("js/screens-viz-patient-series.js?v=20260727-patient-demo2")
    )
    assert ".pt-module-card" in patient_css
    assert ".pt-matrix-details .table-scroll" in patient_css
    for unrelated_css in ("crossdb.css", "cohort.css", "patient-series.css"):
        assert ".pt-matrix-details .table-scroll" not in _static_css(unrelated_css)
    assert "pt-matrix-details" in viz_js
    assert "Exact value audit matrices" in viz_js
    assert "精确值审计矩阵" in viz_js
    assert "Data-table companion audit" in viz_js


def test_native_patient_feature_catalog_has_a_dedicated_owner() -> None:
    """Catalog-to-trajectory merging belongs to one Patient Review owner."""
    owner_js = _static_js("screens-viz-patient-features.js")
    viz_js = _static_js("screens-viz.js")
    series_js = _static_js("screens-viz-patient-series.js")
    index_html = _static_html("index.html")

    assert "window.EU_PATIENT_FEATURES" in owner_js
    assert "function signalKey(" in owner_js
    assert "function catalogLanes(" in owner_js
    assert "demoCatalog.demoCatalogModules()" in owner_js
    assert "demoCatalog.catalogFeatureMeta(feature)" in owner_js
    assert "function signalAvailability(signal)" in owner_js
    assert "return numericCount >= 2 ? 'numeric_trajectory' : 'observed_categorical'" in owner_js
    assert "trajectory: availability === 'numeric_trajectory'" in owner_js
    assert "numeric_trajectory_count: numericTrajectoryCount" in owner_js
    assert "signals.length ? 'observed' : 'metadata_only'" in owner_js
    assert "const uncatalogued = []" in owner_js
    assert "catalogLanes.concat(uncatalogued)" in owner_js

    assert "signalKey: ptSignalKey" in viz_js
    assert "catalogLanes: patientCatalogLanes" in viz_js
    assert "function ptSignalKey(" not in viz_js
    assert "function patientCatalogLanes(" not in viz_js
    assert "window.EU_PATIENT_FEATURES" not in series_js

    feature_owner = index_html.index(
        "js/screens-viz-patient-features.js?v=20260727-patient-demo2"
    )
    assert index_html.index("js/screens-viz-demo.js?") < feature_owner
    assert feature_owner < index_html.index("js/screens-viz-patient-series.js?")
    assert feature_owner < index_html.index("js/screens-viz.js?")


def test_native_crossdb_availability_matrix_is_a_heatmap() -> None:
    """The Cross-DB module availability matrix must colour cells by coverage,
    restoring the legacy availability heatmap instead of plain text cells."""
    results_js = _static_js("screens-viz-crossdb-results.js")
    crossdb_css = _static_css("crossdb.css")

    assert "function availabilityCell(" in results_js
    assert "(row.values || []).map(value => availabilityCell(value, config))" in results_js
    assert ".xdb-avail-cell" in crossdb_css
    assert "h.t('Present', '存在')" in results_js


def test_native_dictionary_and_states_reference_controls_are_stateful() -> None:
    dict_js = _static_js("screens-dict.js")
    states_js = _static_js("screens-states.js")

    assert "dictSearchInput" in dict_js
    assert "data-dict-clear" in dict_js
    assert "data-dict-cat" in dict_js
    assert "matchedRows()" in dict_js
    assert "data-ctx" in states_js
    assert "data-mode" in states_js
    assert "data-state" in states_js
    assert (
        "actionHtml: `<span class=\"pill\">${icon('eye', 13)} Reference</span>`"
        in states_js
    )
    assert 'aria-disabled="true" tabindex="-1"' in states_js


def test_native_dictionary_distinguishes_mapping_audit_from_export_coverage() -> None:
    catalog = TestClient(app).get("/api/catalog").json()
    dict_js = _static_js("screens-dict.js")
    api_js = _static_js("api.js")
    data_catalog_js = _static_js("data-catalog.js")
    deepdive_css = _static_css("deepdive.css")
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert catalog["conceptCoverage"]["hr"]["kind"] == "audited"
    assert catalog["conceptCoverage"]["hr"]["databases"] == 6
    assert catalog["conceptCoverage"]["sofa2_resp"]["kind"] == "derived"
    assert (
        catalog["conceptCoverage"]["sofa2_resp"]["basis"] == "score_or_rule_component"
    )
    assert (
        catalog["coverageSummary"]["audited"]
        + catalog["coverageSummary"]["derived"]
        + catalog["coverageSummary"]["notAudited"]
        == catalog["totalConcepts"]
    )
    assert catalog["activeExportCoverage"]["payload_scope"] == "aggregate_only_no_rows"
    assert catalog["activeExportCoverage"]["status"] in {
        "ready",
        "no_active_source",
        "invalid_active_source",
        "unavailable",
    }

    assert "conceptCoverage = real.conceptCoverage" in api_js
    assert "coverageSummary = real.coverageSummary" in api_js
    assert "activeExportCoverage = real.activeExportCoverage" in api_js
    assert "conceptCoverage" in data_catalog_js
    assert "coverageSummary" in data_catalog_js
    assert "activeExportCoverage" in data_catalog_js
    assert "Object.prototype.hasOwnProperty.call" in dict_js
    assert "C().cov[k] || 0" not in dict_js
    assert "coverage unknown" not in dict_js
    assert "Database coverage" in dict_js
    assert "数据库覆盖" in dict_js
    assert "Dictionary database coverage" in dict_js
    assert "This column counts how many supported ICU databases" in dict_js
    assert "Active export coverage" not in dict_js
    assert "当前导出覆盖" not in dict_js
    assert "not extracted" not in dict_js
    assert "未提取" not in dict_js
    assert "activeExportCoverage" not in dict_js
    assert "Current-export coverage is computed after extraction" not in dict_js
    assert "This column uses the active registered export" not in dict_js
    assert ".dict-catalog-note" in deepdive_css
    assert ".cov-badge.derived" in deepdive_css
    assert ".cov-badge.unaudited" in deepdive_css
    assert "data-catalog.js?v=20260727-patient-demo2" in index_html
    assert "api.js?v=20260828-plan-review1" in index_html
    assert "screens-dict.js?v=20260712-ux-fixes" in index_html
    assert "deepdive.css?v=20260625-stage85" in index_html


def test_native_ui_uses_verification_terms_instead_of_gate_literal_translations() -> (
    None
):
    ui_text = "\n".join(
        _static_js(name)
        for name in [
            "app.js",
            "copilot-dock.js",
            "screens-agent.js",
            "screens-extraction.js",
            "screens-guided.js",
            "screens-guided-projects.js",
            "screens-help.js",
            "screens-ideas.js",
            "screens-settings.js",
            "screens-states.js",
            "screens-viz.js",
        ]
    )

    assert "抽取并核验数据" in ui_text
    assert "证据核验" in ui_text
    assert "review-ready draft" in ui_text
    for old_term in [
        "抽取并" + "门" + "控数据",
        "审阅" + "门" + "控",
        "显式配置" + "门" + "控",
        "门" + "控",
        "门" + "禁",
        "受" + "闸",
        "证据" + "闸",
        "\\u8bc1\\u636e\\u95f8",
        "可行性" + "闸",
        "Extract & " + "gate the data",
        "gated " + "draft",
        "gated-" + "draft",
        "gated " + "manuscript",
        "evidence-" + "gated " + "draft",
        "Evidence " + "gate",
        "evidence " + "gate",
        "review " + "gate",
        "setup " + "gate",
    ]:
        assert old_term not in ui_text


def test_native_guided_local_rail_shows_only_real_local_context() -> None:
    guided_js = _static_js("screens-guided.js")
    projects_js = _static_js("screens-guided-projects.js")
    guided_project_surface = guided_js + projects_js
    guided_css = _static_css("guided.css")
    projects_css = _static_css("guided-projects.css")
    api_js = _static_js("api.js")
    index_html = _static_html("index.html")

    assert "loadAgentRunReview(row.project_dir)" in guided_js
    assert "/api/guided/drafts" in api_js
    assert "/api/guided/drafts/list" in api_js
    assert "/api/guided/drafts/remove" in api_js
    assert "loadGuidedDrafts({ limit: 20 })" in guided_js
    assert "createGuidedDraft(payload)" in guided_js
    assert "function blankGuidedDraftPayload(label)" in guided_js
    assert "data_mode: 'unbound'" in guided_js
    assert "source: null" in guided_js
    assert "bindGuidedDraftMemory(selectedGuidedDraft, true)" in guided_js
    assert "No data selected" in projects_js
    assert "未选择数据" in projects_js
    assert "removeGuidedDraft" in api_js
    assert "data-remove-localdraft" in guided_js
    assert "removeLocalGuidedDraft(row)" in guided_js
    assert "delete_project_folder: false" in guided_js
    assert "trash_project_folder: trashProjectFolder" in guided_js
    assert "trash_confirmation: trashProjectFolder ? row.id : null" in guided_js
    assert "data-remove-project-folder" in projects_js
    assert "Also move the local project folder to the system trash" in projects_js
    assert "By default, the project folder and all files on disk are preserved" in projects_js
    assert "data-confirm-remove-draft" in projects_js
    assert "The project folder on disk was left untouched" in guided_js
    assert "/api/guided/session" in api_js
    assert "/api/guided/project/open" in api_js
    assert "/api/guided/message" in api_js
    assert "/api/guided/action" in api_js
    assert "/api/guided/sessions/list" in api_js
    assert "window.EU_API.listDir = listDir" in api_js
    assert "createGuidedSession" in guided_js
    assert "openGuidedProject" in guided_js
    assert "openGuidedProjectMemory(row, localDraftEl, 'draft')" in guided_js
    assert "configuration_health.status === 'configuration_missing'" in projects_js
    assert "配置已失效" in projects_js
    assert ".gd-sess.configuration-missing" in projects_css
    assert "Memory is scoped to" in guided_js
    assert "Idea Mining and the Research Agent backend still own their artifacts" in guided_js
    assert "Pick a goal to start" in guided_js
    assert "Project memory bound" in guided_js
    assert "pendingGuidedGoal" in guided_js
    assert "requireGuidedProjectMemory(goal, label)" in guided_js
    assert "if (!force) return Promise.resolve(null);" in guided_js
    assert "saveGuidedSlotsNow(reason)" in guided_js
    assert "saveGuidedSlots" in api_js
    assert "scheduleGuidedSlotSave" in guided_js
    assert "restoreGuidedSlotsFromSession" in guided_js
    assert (
        "Required Copilot configuration is only persisted inside a local project folder"
        not in guided_js
    )
    assert "sendGuidedMessage" in guided_js
    assert "runGuidedAction" in guided_js
    assert "Choose a goal" in guided_js
    assert "data-guided-goal" in guided_js
    assert "data-guided-handoff" in guided_js
    assert (
        "If no folder is bound yet, I set up a starter folder in one click" in guided_js
    )
    assert "Find a Study Idea" in guided_js
    assert "Prepare Data" in guided_js
    assert "Run a Research Project" in guided_js
    assert "Research projects" in projects_js
    assert "Local research workspace" in projects_js
    assert "renderProjectRail" in projects_js
    assert "guidedProjectRenderer('renderProjectRail')" in guided_js
    assert "button class=\"gd-sess draft" not in guided_js
    assert "row.project_dir ? esc(compactPath(row.project_dir))" not in guided_project_surface
    assert "Conversation memory" not in guided_js
    assert "data-localdraft" in guided_js
    assert "Agent run artifacts" not in guided_js
    assert "Read-only results" not in guided_js
    assert "Run a confirmed Agent project" not in guided_js
    assert "data-localrun" not in guided_js
    assert "data-refreshruns" not in guided_js
    assert "loadAgentRunHistory({ limit: 20 })" not in guided_js
    assert "existing Agent run folder" in guided_js
    assert ".gd-project-summary" in projects_css
    assert "~/easyicu/projects" in guided_js
    assert "/Users/haibo" not in guided_js
    assert "Seeded example · not a local project" not in guided_js
    assert "Seeded examples" not in guided_js
    assert "data-sess" not in guided_js
    assert "That is a seeded example" not in guided_js
    assert "New / open research folder" in projects_js
    assert "gdFolderControls" in projects_js
    assert "gdFolderControls" not in guided_js
    assert "gdFolderDialogHost" in guided_js
    assert "data-folder-menu-toggle" in projects_js
    assert "data-folder-choice" in projects_js
    assert "guidedFolderMenuOpen && !e.target.closest('.gd-folder-picker')" in guided_js
    assert "guidedDraftRemoval && !guidedDraftRemoval.busy" in guided_js
    assert "data-folder-dialog" in projects_js
    assert "New blank study folder" in projects_js
    assert (
        "Choose a parent folder, then create a metadata-only Guided project subfolder"
        in projects_js
    )
    assert (
        "Create a metadata-only local folder under the EasyICU projects root"
        not in guided_project_surface
    )
    assert "Use existing folder" in projects_js
    assert (
        "Required setup stays here instead of jumping to Classic Workspace"
        in projects_js
    )
    assert "guidedKnownProjectRows" in guided_js
    assert "ctx.guidedKnownProjectRows()" in projects_js
    assert "Recent local projects" in projects_js
    assert "Optional shortcut. The list stays collapsed until you ask" in projects_js
    assert "Shown only after you ask" in projects_js
    assert "Detected local project folders" not in guided_project_surface
    assert "Scanning local project folders" not in guided_project_surface
    assert "guidedKnownProjectsOpen" in guided_js
    assert "guidedKnownProjectsOpen" in projects_js
    assert "data-toggle-known-projects" in projects_js
    assert "data-known-project" in projects_js
    assert "data-refreshfolderchoices" in projects_js
    assert "Path paste remains an advanced fallback" in projects_js
    assert "data-browseprojectfolder" in projects_js
    assert "loadGuidedFolderBrowser" in guided_js
    assert "data-guided-folder-browser" in projects_js
    assert "data-folder-browser-entry" in projects_js
    assert "data-folder-browser-shortcut" in projects_js
    assert "data-folder-browser-use" in projects_js
    assert "Use Browse to choose a folder" in projects_js
    assert "Choose a local study folder" in projects_js
    assert "Open project or extracted data folder" in projects_js
    assert "Local folder path" in projects_js
    assert "local EasyICU project or export folder" in projects_js
    assert "data-openprojectfolder" in projects_js
    assert "data-reviewexportfolder" in projects_js
    assert "data-existing-project-dir" in projects_js
    assert "data-project-open-status" in projects_js
    assert "setGuidedProjectOpenStatus" in guided_js
    assert "openExistingGuidedProject(pathEl && pathEl.value, box)" in guided_js
    assert "registerExistingExportForReview(pathEl && pathEl.value, box)" in guided_js
    assert "window.EU_API.registerWorkspaceSource(raw" in guided_js
    assert "Review extracted data" in guided_project_surface
    assert "project memory is optional for this read-only review" in guided_js
    assert "Opening folder memory and restoring this project context" in guided_js
    assert (
        "This folder is neither an openable Guided project nor a valid EasyICU export"
        in guided_js
    )
    assert "openExistingGuidedProject" in guided_js
    assert "Creating a <strong>metadata-only local study folder</strong>" in guided_js
    assert "Create new local study folder" in projects_js
    assert "Choose the parent folder first" in projects_js
    assert "data-draft-parent-dir" in projects_js
    assert "data-browsedraftparent" in projects_js
    assert "Create inside folder" in projects_js
    assert "Will create" in projects_js
    assert "guided-${slug || 'study'}-..." in projects_js
    assert "payload.parent_dir = parent" in guided_js
    assert "captureGuidedDraftDialogState(box)" in guided_js
    assert "normalize('NFKC')" in guided_js
    assert "\\p{L}\\p{N}" in guided_js
    assert "[^a-z0-9._-]" not in guided_js
    assert "function startFreshGuidedProjectThread(title, path)" in guided_js
    assert "A new Guided conversation has started for this project" in guided_js
    assert "pendingGuidedGoal = null;" in guided_js
    assert "startFreshGuidedProjectThread(title, path)" in guided_js
    assert "if (!continuePendingGuidedGoal())" not in guided_js
    assert "data-createdraft" in projects_js
    assert "data-draft-title" in projects_js
    assert "folder_slug" in guided_js
    assert "createLocalGuidedDraft('New study draft')" not in guided_js
    assert "Studies · local folders" not in guided_project_surface
    assert "Creates a new local project folder" not in guided_project_surface
    assert "Created a new project folder" not in guided_project_surface
    assert 'class="gd-top"' not in guided_js
    assert 'class="gd-home-link"' not in guided_js
    assert '<button class="gd-rail-brand" type="button" data-open="entry"' in projects_js
    assert 'data-open="entry"' in projects_js
    assert "Back to EasyICU home" in projects_js
    assert "${t('Exit', '退出')}" not in guided_js
    assert 'class="gd-rail-utils"' in projects_js
    assert 'data-open="settings"' in projects_js
    assert "data-lang-toggle" in projects_js
    assert "Switch language" in projects_js
    assert "${t('Data workspace', '数据工作台')}" in projects_js
    assert ".gd-empty-local" in projects_css
    assert ".gd-sessline" in projects_css
    assert ".gd-sess-action" in projects_css
    assert ".gd-sess.active" in projects_css
    assert ".gd-home-link" not in guided_css
    assert ".gd-top" not in guided_css
    assert ".gd-rail-brand" in projects_css
    assert ".gd-rail-utils" in projects_css
    assert ".gd-utilbtn.lang" in projects_css
    assert ".gd-data-workspace" in projects_css
    assert ".gd-draft-setup" in guided_css
    assert ".gd-folder-picker" in projects_css
    assert ".gd-folder-menu" in projects_css
    assert ".gd-remove-option" in projects_css
    assert ".gd-remove-option" not in guided_css
    assert "right:0;left:auto;width:min(270px" in projects_css
    assert ".gd-folder-dialog" in guided_css
    assert ".gd-folder-tabs" in guided_css
    assert ".gds-known" in guided_css
    assert ".gds-known.collapsed" in guided_css
    assert ".gds-known-row" in guided_css
    assert ".gds-known-empty" in guided_css
    assert ".gds-browser" in guided_css
    assert ".gds-browser-row" in guided_css
    assert ".gds-browser-actions" in guided_css
    assert ".gds-choice" in guided_css
    assert ".gds-status" in guided_css
    assert ".gds-status.loading" in guided_css
    assert ".gds-status.ok" in guided_css
    assert ".gds-status.error" in guided_css
    assert ".gd-frontdoor" in guided_css
    assert ".gdf-memory" in guided_css
    assert ".gdf-card" in guided_css
    assert ".gd-handoff-ready" in guided_css
    for selector in (
        ".gd-main.threecol",
        ".gd-rail",
        ".gd-project-summary",
        ".gd-sessline",
        ".gd-rail-utils",
    ):
        assert selector in projects_css
        assert selector not in guided_css
    for foreign in (".gpi-", ".gdx-", ".gd-pipeline-", ".patient-", ".cohort-"):
        assert foreign not in projects_css
    assert "!important" not in projects_css
    assert ":has(" not in projects_css
    assert "api.js?v=20260828-plan-review1" in index_html
    assert "screens-guided-projects.js?v=20260825-remove-project1" in index_html
    assert (
        "screens-guided-idea-provider.js?v=20260627-ideas-feasibility-plan"
        in index_html
    )
    assert "screens-guided.js?v=20260827-aside-owner1" in index_html
    assert "guided.css?v=20260827-type-scale1" in index_html
    assert "guided-projects.css?v=20260827-type-scale1" in index_html
    assert "gd-name\">${t('EasyICU Copilot', 'EasyICU 研究助手')}</span>" in projects_js
    assert "${t('New / open research folder', '新建/打开研究目录')}" in projects_js
    assert "Guided Copilot · local first · nothing leaves your machine" in guided_js
    assert "[t('Review Data', '审阅已有数据'), '@guidedGoal:review_data']" in guided_js


def test_native_agent_run_controls_are_reconnectable_and_cancelable() -> None:
    agent_js = _static_js("screens-agent.js")
    provider_js = _static_js("screens-guided-pi-provider.js")
    api_js = _static_js("api.js")

    assert "DeepSeek API" in provider_js
    assert "Anthropic / Claude API" in provider_js
    assert "OpenAI-compatible gateway" in provider_js
    assert "ChatGPT / Codex account" in provider_js
    assert "loadJobSnapshot" in api_js
    assert "cancelJob" in api_js
    assert "getJSON('/api/jobs/' + encodeURIComponent(jobId || ''))" in api_js
    assert (
        "postJSON('/api/jobs/' + encodeURIComponent(jobId || '') + '/cancel'" in api_js
    )
    assert "easyicu.agent.activeJob.v1" in agent_js
    assert "rememberAgentJob" in agent_js
    assert "maybeRestoreAgentJob" in agent_js
    assert "restoreAgentJobFromSnapshot" in agent_js
    assert "data-ag-cancel-job" in agent_js
    assert "data-ag-reconnect" in agent_js
    assert "Resume stream" in agent_js
    assert "Restart from active export" not in agent_js
    assert "continue safely from Guided Copilot" in agent_js
    assert "Continue in Guided Copilot" in agent_js
    assert "seedGateBlocksRun" in agent_js
    assert "host.clientWidth < 1040" in agent_js
    assert "data-ag-" not in provider_js
    assert "project_seed_dir" in agent_js
    assert "Project readiness checks are not complete" in agent_js
    # The remedy sentences moved out of a regex-over-English table in
    # screens-agent.js into gate-remedy.js, keyed on the backend reason code.
    # Assert the property at its new owner rather than dropping it.
    remedy_js = _static_js("gate-remedy.js")
    assert "prior_art_not_reviewed" in remedy_js
    assert "prior-art review" in remedy_js
    assert "/prior-art/i" not in agent_js
    assert "Continue in Guided Copilot so it can refresh" in agent_js


def test_native_patient_source_radios_are_real_controls() -> None:
    viz_js = _static_js("screens-viz.js")
    patient_navigation_js = _static_js("screens-viz-patient-navigation.js")
    patient_tables_js = _static_js("screens-viz-patient-tables.js")
    patient_series_js = _static_js("screens-viz-patient-series.js")
    patient_overview_js = _static_js("screens-viz-patient-overview.js")
    demo_drilldown_js = _static_js("screens-viz-demo-drilldown.js")
    demo_sources_js = _static_js("screens-viz-patient-demo-sources.js")
    api_js = _static_js("api.js")
    i18n_js = _static_js("i18n.js")
    pages_css = _static_css("pages.css")
    patient_css = _static_css("patient.css")
    patient_navigation_css = _static_css("patient-navigation.css")
    patient_tables_css = _static_css("patient-tables.css")
    patient_series_css = _static_css("patient-series.css")
    official_demo_sources_css = _static_css("official-demo-sources.css")
    index_html = _static_html("index.html")

    assert 'data-datamode="real"' in viz_js
    assert 'data-datamode="demo"' in viz_js
    assert "Previously exported data" in viz_js
    assert "Demo data" in viz_js
    assert "loadPatientReviewSources" in api_js
    assert "/api/patient-review/sources" in api_js
    assert "loadPatientReviewEntities" in api_js
    assert "/api/patient-review/entities" in api_js
    assert "loadPatientReviewEntity" in api_js
    assert "/api/patient-review/entity" in api_js
    assert "loadPatientReviewTablePreview" in api_js
    assert "/api/patient-review/table-preview" in api_js
    assert "loadPatientSources" in viz_js
    assert "Ready to load local export" in viz_js
    assert "No registered export is active" in viz_js
    assert "Reading bounded Patient Review from local export" in viz_js
    assert "const wsMatchesActive = ws" in viz_js
    assert "const patientSource = active === 'patient' ? patientActiveSourceMeta() : null" in viz_js
    assert "route: 'patient'" in viz_js
    assert "route: 'cohort'" in viz_js
    assert "route: 'crossdb'" in viz_js
    assert "skeletonWorkspace(window.EU_DATA)" in viz_js
    assert "full cohort', '完整队列'" in viz_js
    assert "bounded browser review', '浏览器有界审阅'" in viz_js
    assert "data-patient-export" in viz_js
    assert "bounded_patient_review_drilldown" in viz_js
    assert "data-pt-table-module" in viz_js
    assert "data-pt-page-prev" in viz_js
    assert "data-pt-page-next" in viz_js
    assert "data-pt-page-size" in viz_js
    assert "table_page" in patient_tables_js
    assert "table_page_size" in patient_tables_js
    assert "loadPatientReviewTablePreview" in patient_tables_js
    assert "loadPatientReviewTablePreview" not in viz_js
    assert "loadPatientReviewEntity" in patient_navigation_js
    assert "loadPatientReviewEntity" not in viz_js
    assert "Pseudonymous entity" in viz_js
    assert "伪匿名实体" in viz_js
    assert "display_column_labels" in viz_js
    assert "label_i18n" in viz_js
    assert "patientModuleLabel" in viz_js
    assert "patientColumnLabel(c, activePreview)" in viz_js
    assert "Module table overview" in viz_js
    assert "模块表格概览" in viz_js
    assert "Direct clinical identifiers stay on disk." in viz_js
    assert "data-patient-table-preview" in viz_js
    assert "data-patient-feature-matrix" in viz_js
    assert "patientFeatureMatrix" in viz_js
    assert "Exact value audit matrices" in viz_js
    assert "精确值审计矩阵" in viz_js
    assert "Rows are time windows; columns are selected features." in viz_js
    assert "行是时间窗口；列是已选特征。" in viz_js
    # Patient Overview is not a second time-series page: it renders a one-entity
    # clinical profile, category-level latest/ever signals, and module
    # availability. Curves stay in the Time Series tab.
    assert "data-patient-category-review" in patient_overview_js
    assert "patientOverviewWorkbench" in viz_js
    assert "EU_PATIENT_OVERVIEW.renderOverview" in viz_js
    assert "patientCategoryReview" not in viz_js
    assert "function patientConceptChart(" not in viz_js
    assert "function patientCategoryCard(" not in viz_js
    assert "Patient category dashboard" not in viz_js
    assert "患者分类看板" not in viz_js
    assert "patientOverviewAtlas" not in viz_js
    assert "data-patient-overview-workbench" in patient_overview_js
    assert "Clinical case overview" in patient_overview_js
    assert "病例画像工作台" in patient_overview_js
    assert "Category signal summary" in patient_overview_js
    assert "分类信号摘要" in patient_overview_js
    assert "This is not another trajectory view" in patient_overview_js
    assert "这里不是另一组曲线" in patient_overview_js
    assert "data-patient-category=" in patient_overview_js
    assert "pcs-thr-line" not in viz_js
    assert "pcs-thr-line" not in patient_overview_js
    assert "data-patient-overview-module-ledger" in patient_overview_js
    assert "data-patient-overview-module-card" in patient_overview_js
    assert "Module map" in patient_overview_js
    assert "模块图谱" in patient_overview_js
    assert "Export module availability" in patient_overview_js
    assert "导出模块可用性" in patient_overview_js
    assert "data-patient-overview-missingness" in patient_overview_js
    assert "Missingness and coverage" in patient_overview_js
    assert "缺失率与覆盖率" in patient_overview_js
    assert "Event / exposure prevalence" in patient_overview_js
    assert "事件 / 暴露发生率" in patient_overview_js
    assert "not missingness" in patient_overview_js
    assert "不是缺失率" in patient_overview_js
    assert "renderQualityAudit" in patient_overview_js
    assert "renderQualityAudit" in viz_js
    assert "value == null || value === '' || typeof value === 'boolean'" in patient_overview_js
    assert "coverage across all selected entities" not in patient_overview_js
    assert "entity coverage was not computed" in patient_overview_js
    assert 'data-patient-module-coverage="${hasCoverage ? \'computed\' : \'not-computed\'}"' in viz_js
    assert "q.coverage_pct == null ? 0" not in viz_js
    assert "metricKind === 'event_rate'" in viz_js
    assert "metricKind === 'exposure_rate'" in viz_js
    assert "Selected entity trend tiles" not in viz_js
    assert "Table preview" in viz_js
    assert "表格预览" in viz_js
    assert "table_previews" in viz_js
    assert "pseudonymous entity tokens" in viz_js
    assert "data-patient-eligibility-flow" in viz_js
    assert "patientEligibilityFlow" in viz_js
    assert "Eligibility flow (ICU stays)" in viz_js
    assert "入组筛选流程（ICU 住院）" in viz_js
    assert "Sepsis-3 cohort" in demo_drilldown_js
    assert "Sepsis-3 脓毒症队列" in demo_drilldown_js
    assert "suspected infection + SOFA signal" in demo_drilldown_js
    assert "疑似感染 + SOFA 信号" in demo_drilldown_js
    assert "Review window available" not in viz_js
    assert "可用审阅时间窗" not in viz_js
    assert "cohort_attrition_metadata_only" in demo_drilldown_js
    assert "patient-flow-diagram" in viz_js
    assert "patient-flow-node" in viz_js
    assert "patient-flow-side-link" in viz_js
    assert ".patient-table-scroll" in patient_tables_css
    assert ".patient-preview-table" in patient_tables_css
    assert ".patient-table-pager" in patient_tables_css
    assert ".patient-table-scroll" not in pages_css
    assert ".patient-preview-table" not in pages_css
    assert ".patient-table-pager" not in pages_css
    assert ".pt-entity-nav" in patient_navigation_css
    assert ".pt-entity-nav" not in patient_css
    assert ".pt-entity-nav" not in pages_css
    assert ".patient-flow-card" in patient_css
    assert ".patient-flow-diagram" in patient_css
    assert ".patient-flow-node.has-next::after" in patient_css
    assert ".patient-flow-side-link::after" in patient_css
    assert ".patient-flow-excluded" in patient_css
    assert ".pt-overview-workbench" in patient_css
    assert ".pt-category-section" in patient_css
    assert ".pt-missingness-workbench" in patient_css
    assert ".pt-missingness-row" in patient_css
    assert ".pt-presence-panel" in patient_css
    assert ".pt-module-ledger-card" in patient_css
    assert ".patient-flow-card" not in pages_css
    assert ".patient-flow-card" not in _static_css("cohort.css")
    assert "css/pages.css?v=20260710-patient-owner-split" in index_html
    assert "css/patient-navigation.css?v=20260710-bounded-pages" in index_html
    assert "css/patient-tables.css?v=20260710-lazy-pages" in index_html
    assert "css/patient.css?v=20260710-owner-split" in index_html
    assert "css/patient-series.css?v=20260727-patient-demo2" in index_html
    assert "js/screens-viz-demo.js?v=20260727-patient-demo2" in index_html
    assert "js/screens-viz-demo-drilldown.js?v=20260727-owner-split" in index_html
    assert "js/screens-viz-patient-features.js?v=20260727-patient-demo2" in index_html
    assert "js/screens-viz-patient-charts.js?v=20260728-shared-echarts1" in index_html
    assert "js/screens-viz-patient-series.js?v=20260727-patient-demo2" in index_html
    assert (
        "js/screens-viz-patient-demo-sources.js?v=20260728-shared-source1"
        in index_html
    )
    assert "css/official-demo-sources.css?v=20260728-shared-source1" in index_html
    assert (
        "js/screens-viz-patient-navigation.js?v=20260710-bounded-pages"
        in index_html
    )
    assert "js/screens-viz-patient-tables.js?v=20260710-lazy-pages" in index_html
    assert "js/screens-viz-patient-overview.js?v=20260710-review-scope" in index_html
    assert index_html.index("js/screens-viz-patient-navigation.js?") < index_html.index(
        "js/screens-viz.js?"
    )
    assert index_html.index("js/screens-viz-patient-tables.js?") < index_html.index(
        "js/screens-viz.js?"
    )
    assert "js/screens-viz.js?v=20260823-native-preview1" in index_html
    assert "bounded browser review', '浏览器有界审阅" in viz_js
    assert "function buildPatientDrilldown" in demo_drilldown_js
    assert "function demoTablePreviewRowContext" in demo_drilldown_js
    assert "const timepointsPerEntity = 12" in demo_drilldown_js
    assert "const previewLimit = timeIndexed ? 24 : 8" in demo_drilldown_js
    assert "entityRef: `demo_ent_${entityIndex + 1}`" in demo_drilldown_js
    assert "charttime: demoCharttimeAt(timeIndex)" in demo_drilldown_js
    assert "row_cap: previewRows.length" in demo_drilldown_js
    assert "function buildPatientDrilldown" not in viz_js
    assert "function demoTablePreviewRowContext" not in viz_js
    assert "Seeded observations" in viz_js
    # Patient time-series stays per-feature, but is grouped by the backend module
    # lanes instead of a fixed vitals-only shortlist.
    assert "function patientVitalSmallMultiples(" in viz_js
    assert "EU_PATIENT_SERIES.renderModulePanels" in viz_js
    assert "EU_PATIENT_SERIES.renderTimeSeriesWorkspace" in viz_js
    assert "data-patient-series-mode" in viz_js
    assert "function renderModulePanels(" in patient_series_js
    assert "function renderTimeSeriesWorkspace(" in patient_series_js
    assert "function numericSamples(sig)" in patient_series_js
    assert "times: samples.times" in patient_series_js
    assert "data-patient-series-module" in patient_series_js
    assert "Clinical trajectory review" in patient_series_js
    assert "临床轨迹审阅" in patient_series_js
    assert "Cross-patient comparison" in patient_series_js
    assert "跨患者对比" in patient_series_js
    assert "Module overview" in patient_series_js
    assert "模块总览" in patient_series_js
    assert "Trajectory gallery" in patient_series_js
    assert "轨迹画廊" in patient_series_js
    assert "Time series and feature catalog by module" in patient_series_js
    assert "按模块分组的时间序列与特征目录" in patient_series_js
    assert "Clinical reference guide" in patient_series_js
    assert "临床参考线" in patient_series_js
    assert "Low threshold" not in patient_series_js
    assert "High threshold" not in patient_series_js
    assert "window.EU_OFFICIAL_DEMO_SOURCES = owner" in demo_sources_js
    assert "window.EU_PATIENT_DEMO_SOURCES = owner" in demo_sources_js
    assert "loadOfficialDemoSources" in api_js
    assert "startOfficialDemoSourcePrepare" in api_js
    assert "data-official-demo-sources" in demo_sources_js
    assert "data-demo-source-prepare" in demo_sources_js
    assert "const isActive = isPrepared && status.active" in demo_sources_js
    assert "data-demo-source-open-after-prepare" in demo_sources_js
    assert "config.openPrepared(sourceId)" in demo_sources_js
    assert "download_rate_bps" in demo_sources_js
    assert "eta_seconds" in demo_sources_js
    assert "official-demo-progress" in demo_sources_js
    assert ".official-demo-progress{" in official_demo_sources_css
    assert ".official-demo-progress" not in patient_series_css
    assert "source.status.active" in viz_js
    assert "demoSourceOwner.rememberOpened(sourceId)" in viz_js
    assert "function activeMetadata(registrySources, activePath)" in demo_sources_js
    assert "data-patient-official-demo" in viz_js
    assert "data-gen" in demo_sources_js
    assert "window.EU_OFFICIAL_DEMO_SOURCES = owner" not in viz_js
    assert ".official-demo-sources" in official_demo_sources_css
    for unrelated_css in ("patient.css", "patient-series.css", "cohort.css", "crossdb.css"):
        assert ".official-demo-sources" not in _static_css(unrelated_css)
    assert ".pt-series-workbench" in patient_series_css
    assert ".pt-series-modebar" in patient_series_css
    assert ".pt-single-grid" in patient_series_css
    assert ".pt-compare-bar" in patient_series_css
    assert ".patient-flow-card" not in patient_series_css
    assert ".cross-density" not in patient_series_css
    assert ".cohort-survival" not in patient_series_css
    assert ".pt-module-card" in patient_css
    assert ".pt-matrix-details" in patient_css
    assert ".pt-matrix-details" not in pages_css
    assert ".pt-matrix-details" not in _static_css("cohort.css")
    assert ".pt-module-card" not in pages_css
    assert ".pt-module-card" not in _static_css("cohort.css")
    assert (
        "payload_scope: 'clinically_constrained_synthetic_demo_no_real_patient_rows'"
        in demo_drilldown_js
    )
    assert "Clinically constrained synthetic fallback ready" in viz_js
    assert "synthetic entities" in viz_js
    assert "10 stays · 19 modules · 0 errors" not in viz_js
    assert "Fast demo profile" not in viz_js
    assert (
        "fmtInt(m.review_features != null ? m.review_features : m.feature_count)"
        in viz_js
    )
    assert "fmtInt(m.observed_features != null ? m.observed_features" not in viz_js
    assert "window.__euVizResetForDataMode" in viz_js
    assert "window.__euVizResetForDataMode()" in i18n_js


def test_native_source_registry_add_gives_feedback_instead_of_silent_noop() -> None:
    viz_js = _static_js("screens-viz.js")

    assert "data-src-mode=\"${multi ? 'multi' : 'single'}\"" in viz_js
    assert "data-src-add-feedback" in viz_js
    assert (
        'data-src-add-feedback hidden aria-hidden="true" role="status" style="display:none;"'
        in viz_js
    )
    assert "data-src-browse" in viz_js
    assert "Choose EasyICU export folder" in viz_js
    assert "选择 EasyICU 导出文件夹" in viz_js
    assert "openSourceFolderPicker" in viz_js
    assert "registrySources().slice().sort((a, b)" in viz_js
    assert "if (aOn !== bOn) return aOn ? -1 : 1;" in viz_js
    assert "window.EU_API.listDir(path)" in viz_js
    assert "function setSourceAddFeedback" in viz_js
    assert "const clean = message == null ? '' : String(message).trim();" in viz_js
    assert "box.setAttribute('aria-hidden', 'true');" in viz_js
    assert "box.style.display = 'none';" in viz_js
    assert "box.removeAttribute('aria-hidden');" in viz_js
    assert "box.style.display = '';" in viz_js
    assert "function registerSourceFromInput" in viz_js
    assert (
        "Use Browse to choose a local EasyICU export folder, or paste its path before pressing Add."
        in viz_js
    )
    assert "请点击“浏览”选择本地 EasyICU 导出文件夹，或粘贴路径后再点击添加。" in viz_js
    assert "Folder selected. Registering and switching to this export..." in viz_js
    assert "已选择文件夹，正在注册并切换到这个导出..." in viz_js
    assert "Local workspace API is not ready. Refresh the page and try again." in viz_js
    assert "Checking and adding this local export..." in viz_js
    assert "input.setAttribute('aria-invalid', 'true');" in viz_js
    assert "e.key !== 'Enter'" in viz_js
    assert (
        "registerSourceFromInput(container, screenId, container.querySelector('[data-src-add]'))"
        in viz_js
    )
    assert (
        "const multi = container && container.dataset && container.dataset.srcMode === 'multi';"
        in viz_js
    )
    assert (
        "if (!path || !(window.EU_API && window.EU_API.registerWorkspaceSource)) return;"
        not in viz_js
    )


def test_native_cohort_real_page_is_backend_backed_and_bilingual() -> None:
    viz_js = _static_js("screens-viz.js")
    api_js = _static_js("api.js")

    assert "loadCohortReviewSummary" in api_js
    assert "window.EU_API.loadCohortReviewSummary(body)" in viz_js
    assert "window.EU_COHORT_REVIEW = payload;" in viz_js
    assert "cohortWorkspaceFromReview(payload)" in viz_js
    assert "function cohortLoaded()" in viz_js
    assert "window.EU_DATA !== 'real' || !!(review && review.summary)" in viz_js
    assert "function reloadStaleRealCohortIfNeeded" in viz_js
    assert "function cohortText" in viz_js
    assert "function cohortReason" in viz_js
    assert "'Backend evidence checks': '后端证据检查'" in viz_js
    assert "'Draft review': '草稿核验'" in viz_js
    assert "'Local export cohort review ready': '本地导出队列审阅已就绪'" in viz_js
    assert "'Fail-closed': '保守拦截'" in viz_js
    assert "'Blocked cohort functions': '已拦截的队列功能'" in viz_js
    assert "function cohortSurvivalSourceHint" in viz_js
    assert "data-survival-source-hint" in viz_js
    assert "Current export is already loaded" in viz_js
    assert "当前导出已加载" in viz_js
    assert "data-survival-current-export" in viz_js
    assert "data-survival-source-picker" not in viz_js
    assert "No re-import is required" in viz_js
    assert "不需要重新导入" in viz_js
    assert "'Hospital mortality': '院内死亡'" in viz_js
    assert "Object.prototype.hasOwnProperty.call(map, raw)" in viz_js
    assert "'Not manuscript-ready by itself': '不能单独用于稿件结论'" in viz_js
    assert "choosing a source immediately recomputes the backend summary" not in viz_js
    assert "function cohortRealModuleSummary" in viz_js
    assert "data-cohort-real-modules" in viz_js
    assert "Open coverage audit" in viz_js
    assert "function cohortRealFeaturePicker" in viz_js
    assert "data-cohort-feature-picker" in viz_js
    assert "data-cohort-feature-toggle" in viz_js
    assert "data-cohort-feature-module" in viz_js
    assert "selected_features" in viz_js
    assert "Full export feature catalog" in viz_js
    assert "全量导出特征目录" in viz_js
    assert "Restore default features" in viz_js
    assert "恢复默认特征" in viz_js
    assert "cohortSelectedFeatures" in viz_js
    assert (
        "loadRealCohort(ok => { cohortView = ok ? 'loaded' : 'idle'; repaintScreen('cohort'); });"
        in viz_js
    )
    assert (
        "manifest parsed · denominators previewed · aggregate payload returned"
        in viz_js
    )
    assert "聚合载荷已就绪；打开项目监控做证据绑定草稿核验。" in viz_js
    assert "Draft gate" not in viz_js
    assert "Evidence checks" not in viz_js
    assert "locked · needs reviewer sign-off" not in viz_js


def test_native_cohort_comparison_radios_are_stateful_controls() -> None:
    viz_js = _static_js("screens-viz.js")
    cohort_charts_js = _static_js("screens-viz-cohort-charts.js")
    cohort_css = _static_css("cohort.css")
    cohort_charts_css = _static_css("cohort-charts.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert "css/cohort.css?v=20260707-design" in index_html
    assert "js/screens-viz.js?v=20260823-native-preview1" in index_html
    assert "let cohortView = 'idle';" in viz_js
    assert "let cohortFeatureScope = 'recommended';" in viz_js
    assert 'data-cohort-config-required="true"' in viz_js
    assert "Choose one cohort data source" in viz_js
    assert "data-cohort-demo-fallback" in viz_js
    assert "{ scope: 'cohort', fallbackAttribute: 'data-cohort-demo-fallback' }" in viz_js
    assert "demoSourceOwner.rememberOpened(sourceId)" in viz_js
    assert "cohortView = ok ? 'loaded' : 'idle';" in viz_js
    assert "function cohortMissingExportMessage" in viz_js
    assert (
        "Choose or add a local EasyICU export before loading Cohort Statistics."
        in viz_js
    )
    assert "请先选择或添加本地 EasyICU 导出，再加载队列统计。" in viz_js
    assert "const body = { source_path: active };" in viz_js
    assert "if (!registryActivePath()) {" in viz_js
    assert "No active export selected" in viz_js
    assert "cohortView = 'idle';" in viz_js
    assert "window.EU_COHORT_REVIEW = null;" in viz_js
    assert "data-viz-reset" in viz_js
    assert "let cohortCompare = 'outcome';" in viz_js
    assert "let cohortSurvivalOutcome = 'mort_28d';" in viz_js
    assert "let cohortSurvivalGroup = 'sepsis';" in viz_js
    assert 'data-cohort-comp="${key}"' in viz_js
    assert "cohortCompare = b.dataset.cohortComp || 'outcome';" in viz_js
    assert "data-cohort-surv-outcome" not in viz_js
    assert "function cohortSurvivalOutcomeCards" in viz_js
    assert "event_summary" in viz_js
    assert "Outcome overview" in viz_js
    assert "data-cohort-surv-group" in viz_js
    assert "Kaplan-Meier curves and log-rank" in viz_js
    assert "Number at risk" in viz_js
    assert "Math.log10(n)" in viz_js
    assert "p = ${esc(pValueLabel)}" in viz_js
    assert "cohortSurvivalWindowNote" in viz_js
    assert "dedicated flag + follow-up" in viz_js
    assert "derived from hospital death + LOS" not in viz_js
    assert "hospital_mortality_time_window" not in viz_js
    assert ".surv-outcome-card" in cohort_css
    assert (
        "ICU mortality is unavailable because this export does not include ICU-specific event and time columns."
        in viz_js
    )
    assert "p <0.001" not in viz_js
    assert "cohortSurvivalDemoBody" in viz_js
    assert "data-demo-survival-simulated" in viz_js
    assert "Demo simulated KM preview" in viz_js
    assert (
        "This Kaplan-Meier curve is a fixed simulated preview for the demo workspace"
        in viz_js
    )
    assert "Demo mode does not fabricate survival curves" not in viz_js
    assert "function cohortDemoCatalogScope" in viz_js
    assert "function cohortDemoFeaturePicker" in viz_js
    assert "data-cohort-catalog-scope" in viz_js
    assert "data-cohort-feature-scope" in viz_js
    assert "Load all modules" in viz_js
    assert "加载全部模块" in viz_js
    assert "Use recommended modules" in viz_js
    assert "恢复推荐模块" in viz_js
    assert "The simulated preview can take a little longer" in viz_js
    assert "演示预览可能稍慢一点" in viz_js
    assert (
        "Features to load')}: ${fmtInt(scope.selectedFeatureCount)} / ${fmtInt(scope.totalFeatureCount)}"
        in viz_js
    )
    assert "Features to load')}: 9" not in viz_js
    assert "window.EU_STALE = true;" in viz_js
    assert "function cohortProfileValue" in viz_js
    assert "function cohortSurvivalBody" in viz_js
    assert "function cohortSurvivalChart" in viz_js
    assert "function cohortUnavailablePanel" in viz_js
    assert "metadata_row_count_only" in viz_js
    assert "Large export coverage optimized" in viz_js
    assert "大导出覆盖率已优化" in viz_js
    assert "They are loaded modules, not missing modules." in viz_js
    assert "它们是已加载模块，不是缺失模块。" in viz_js
    assert (
        "Current export is loaded, but the cohort is above the interactive KM preview limit"
        in viz_js
    )
    assert "同一个导出上继续运行本地审计分析任务" in viz_js
    assert "function cohortDemoCoverageReview" in viz_js
    assert "function cohortDemoSofaReview" in viz_js
    assert "function cohortDemoPanelNote" in viz_js
    assert "data-demo-cohort-panel" in viz_js
    assert "Demo module coverage and quality" in viz_js
    assert "演示模块覆盖率与质量" in viz_js
    assert "function cohortCoverageMetricLabel" in viz_js
    assert "function cohortQualityStatusLabel" in viz_js
    assert "Event/exposure rows show cohort incidence or exposure prevalence" in viz_js
    assert "事件/暴露行显示队列发生率或暴露率" in viz_js
    assert "cohortCoverageMetricValue(row)" in viz_js
    assert "esc(row.quality_status || 'unknown')" not in viz_js
    assert "Demo SOFA-2 aggregate preview" in viz_js
    assert "演示 SOFA-2 聚合预览" in viz_js
    assert (
        "review ? cohortCoverageBody(review) : (demoLoaded ? cohortCoverageBody(cohortDemoCoverageReview(), { demo: true })"
        in viz_js
    )
    assert (
        "review ? cohortSofaBody(review) : (demoLoaded ? cohortSofaBody(cohortDemoSofaReview(), { demo: true })"
        in viz_js
    )
    assert "The old seeded audit panel has been removed." in viz_js
    assert "window.EUAudit" not in viz_js
    assert "window.EUSofa" not in viz_js
    assert "screens-audit.js" not in index_html
    assert not (STATIC_DIR / "js" / "screens-audit.js").exists()
    assert "Aggregate-only group characteristics" in viz_js
    assert "profileRows.map" in viz_js
    assert "active.profile" in viz_js
    assert "SOFA-1 to SOFA-2 movement" in viz_js
    assert "Worst-ICU severity transition matrix" in viz_js
    assert "function cohortSofaHeatmap" in viz_js
    assert "window.EU_COHORT_CHARTS = {" in cohort_charts_js
    assert "type: 'heatmap'" in cohort_charts_js
    assert "step: 'end'" in cohort_charts_js
    assert "cohortSofaMatrixMode" in viz_js
    assert "data-cohort-sofa-matrix-mode" in viz_js
    assert "SOFA_MATRIX_GRANULARITIES" in viz_js
    assert "cohortSofaMatrixGranularity = 'medium'" in viz_js
    assert "data-cohort-sofa-granularity" in viz_js
    assert "exact_score_matrix" in viz_js
    assert "cohortCharts.heatmapSlot" in viz_js
    assert "Rows are SOFA-1 score bands; columns are SOFA-2 score bands." in viz_js
    assert "Rows are SOFA-1 severity bands; columns are SOFA-2 bands." in viz_js
    assert "reclass.status === 'ready'" in viz_js
    assert "Demo threshold uses SOFA ≥ 6" in viz_js
    assert "Age Groups' overview" not in viz_js
    assert ".surv-toolbar" in cohort_css
    assert ".cohort-echart" in cohort_charts_css
    assert ".risk-table" in cohort_css
    assert ".cohort-heat-legend" in cohort_charts_css
    assert ".sofa-matrix-toggle" in cohort_css
    assert ".sofa-matrix-controls" in cohort_css
    assert ".surv-toolbar" not in redesign_css
    assert ".cohort-echart" not in redesign_css
    assert ".cohort-echart" not in cohort_css
    assert ".sofa-matrix-controls" not in redesign_css
    for key in ["outcome", "age", "sex", "los", "sepsis", "custom"]:
        assert f"{key}:" in viz_js


def test_visual_routes_share_source_choice_with_single_and_multi_source_contracts() -> None:
    """Patient/Cohort select one source; Cross-DB selects an official pair."""
    viz_js = _static_js("screens-viz.js")
    demo_sources_js = _static_js("screens-viz-patient-demo-sources.js")
    crossdb_setup_js = _static_js("screens-viz-crossdb-setup.js")
    crossdb_source_js = _static_js("screens-viz-crossdb-source.js")
    shared_css = _static_css("official-demo-sources.css")
    crossdb_css = _static_css("crossdb.css")
    index_html = _static_html("index.html")

    assert "function sourceModeSelector(realMode)" in viz_js
    assert "Previously exported data" in viz_js
    assert "Demo data" in viz_js
    assert "{ scope: 'cohort', fallbackAttribute: 'data-cohort-demo-fallback' }" in viz_js
    assert "window.EU_OFFICIAL_DEMO_SOURCES = owner" in demo_sources_js
    assert "function registeredSources(registryRows)" in demo_sources_js
    assert "function rememberPair(registryRows)" in demo_sources_js
    assert "kind: 'official_demo_pair'" in demo_sources_js
    assert "sourceModeHtml(realMode)" not in crossdb_setup_js
    assert "sourceMethod: 'registered'" in crossdb_setup_js
    assert "data-crossdb-demo-source-choice" in crossdb_source_js
    assert "compactOfficialPair" in crossdb_source_js
    assert "Start consistency check" in crossdb_source_js
    assert "data-crossdb-synthetic-fallback" in crossdb_source_js
    assert "UI rehearsal only" in crossdb_source_js
    assert "officialPaths()" in viz_js
    assert "runOfficial()" in viz_js
    assert "owner.rememberPair(registrySources())" in viz_js
    assert ".official-demo-sources" in shared_css
    assert ".crossdb-offline-fallback" in crossdb_css
    assert "css/official-demo-sources.css?v=20260728-shared-source1" in index_html
    assert "js/screens-viz-crossdb-setup.js?v=20260728-one-click-raw2" in index_html
    assert "js/screens-viz-crossdb-source.js?v=20260812-crossdb-jobs" in index_html


def test_native_webapp_foreground_interrupt_returns_shell_status(monkeypatch) -> None:
    from easyicu.webserver import __main__ as webmain

    def fake_run(cmd, env):  # noqa: ANN001
        raise KeyboardInterrupt

    monkeypatch.setattr(webmain.subprocess, "run", fake_run)

    assert webmain.run_app(port=9876) == 130


def test_native_home_landing_styles_are_owned_by_home_css() -> None:
    """Entry/Home landing page owns its CSS in home.css, not the shared
    screens.css workflow-primitives file (split 2026-06-26)."""
    home_css = _static_css("home.css")
    screens_css = _static_css("screens.css")
    index_html = _static_html("index.html")

    # home.css owns the landing-page selectors
    assert ".home-wrap" in home_css
    assert ".entry-shell" in home_css
    assert ".way-card" in home_css
    assert ".mode-card" in home_css
    assert ".col-entry" in home_css
    assert "css/home.css?v=20260707-design" in index_html

    # screens.css is no longer a landing-page catch-all
    assert ".home-wrap" not in screens_css
    assert ".entry-shell" not in screens_css
    assert ".way-card" not in screens_css
    assert ".mode-card" not in screens_css

    # screens.css still owns the shared cross-screen workflow primitives
    assert ".preflight" in screens_css
    assert ".pipeline" in screens_css
    assert ".planlist" in screens_css
    assert ".ledger-row" in screens_css

    # landing-page styles must not leak back into other transitional buckets
    assert ".home-wrap" not in _static_css("redesign.css")
    assert ".way-card" not in _static_css("app.css")


def test_native_viz_demo_layer_is_split_into_owner_file() -> None:
    """Demo data generation and Patient drilldown assembly have separate owners."""
    viz_js = _static_js("screens-viz.js")
    demo_js = _static_js("screens-viz-demo.js")
    demo_drilldown_js = _static_js("screens-viz-demo-drilldown.js")
    index_html = _static_html("index.html")

    # demo generators + catalog accessors are DEFINED in the demo file
    assert "function demoCatalogModules(" in demo_js
    assert "function demoRowsForModule(" in demo_js
    assert "function demoCategorySection(" in demo_js
    assert "function catalogModuleLabel(" in demo_js
    assert "function catalogFeatureMeta(" in demo_js
    assert "window.VIZ_DEMO = {" in demo_js
    assert "const DEMO_CHART_HOURS = [-1, 0.5, 2, 4, 7, 11" in demo_js
    assert "DEMO_ENTITY_COUNT, DEMO_DURATION_HOURS, DEMO_CHART_HOURS" in demo_js

    # they are NOT re-defined in the main file (no duplicate definitions)
    assert "function demoCatalogModules(" not in viz_js
    assert "function catalogModuleLabel(" not in viz_js
    assert "const DEMO_ENTITY_COUNT" not in viz_js

    # Patient drilldown assembly is defined only in its dependency-neutral owner.
    assert "function buildPatientDrilldown(" in demo_drilldown_js
    assert "function demoTablePreviewRowContext(" in demo_drilldown_js
    assert "window.VIZ_DEMO_DRILLDOWN = { buildPatientDrilldown };" in demo_drilldown_js
    assert "} = window.VIZ_DEMO;" in demo_drilldown_js
    assert "function buildPatientDrilldown(" not in viz_js
    assert "function buildDemoPatientDrilldown(" not in viz_js
    assert "function demoTablePreviewRowContext(" not in viz_js

    # main file rebinds the two owner contracts so existing call sites stay unchanged
    assert "} = window.VIZ_DEMO;" in viz_js
    assert (
        "const { buildPatientDrilldown: buildDemoPatientDrilldown } = "
        "window.VIZ_DEMO_DRILLDOWN;"
    ) in viz_js
    assert "demoCatalogModules" in viz_js  # still called

    # Generator -> drilldown owner -> main shell load order is explicit.
    demo_pos = index_html.find("screens-viz-demo.js")
    drilldown_pos = index_html.find("screens-viz-demo-drilldown.js")
    main_pos = index_html.find("screens-viz.js?")
    assert demo_pos != -1 and drilldown_pos != -1 and main_pos != -1
    assert demo_pos < drilldown_pos < main_pos


def test_shell_copilot_entry_opens_the_pi_route_without_a_second_bridge_chat() -> None:
    """The floating entry is a route/focus control, not another conversation."""
    dock_js = _static_js("copilot-dock.js")
    assert "location.hash = '#guided'" in dock_js
    assert "document.querySelector('[data-gpi-input]')" in dock_js
    assert "__cpBridge" not in dock_js
    assert "createPageGuideSession" not in dock_js


def test_guided_handoff_banner_surfaces_full_study_design() -> None:
    """Regression: the Copilot->module handoff banner must surface the collected
    study design (outcome / window / comparator / export destination), not just the
    question, so a handed-off study is visible on the target page instead of dropped."""
    app_js = _static_js("app.js")
    i18n_js = _static_js("i18n.js")
    assert "p.outcome_hint" in app_js
    assert "p.time_window_hint" in app_js
    assert "p.comparator_hint" in app_js
    assert "p.export_destination_hint" in app_js
    # labelled, and long export paths are shortened
    assert "shortPath" in app_js


def test_home_data_toggle_routes_through_setdatamode() -> None:
    """Regression: the home Demo/Real toggle must go through the canonical
    setDataMode (workspace invalidation + confirm-on-switch guard), not a bare
    EU_DATA write that leaves stale workspaces bound to the wrong source."""
    ext_js = _static_js("screens-extraction.js")
    assert "function setHomeData(m) {" in ext_js
    assert "if (window.setDataMode) { window.setDataMode(m); return; }" in ext_js


def test_guided_terminal_path_opens_project_monitor_without_moving_setup_there() -> None:
    """After the guided mock preflight, the review action may open Project
    Monitor, but provider/model setup remains in Guided Copilot."""
    guided_js = _static_js("screens-guided.js")
    assert "function guidedAgentHandoffPrefill()" in guided_js
    assert "function openGuidedAgentHandoff()" in guided_js
    assert "data-ga-open-agent" in guided_js
    assert "Open Project Monitor" in guided_js
    assert "Provider and model selection stay in Guided Copilot" in guided_js
    assert "local, no-cost preflight" in guided_js


def test_analysis_handoffs_route_to_copilot_and_monitor_links_use_one_name() -> None:
    """Starting/configuring analysis belongs to Guided Copilot; #agent is only
    the consistently named Project Monitor destination for existing runs."""
    viz_js = _static_js("screens-viz.js")
    crossdb_js = _static_js("screens-viz-crossdb-results.js")
    ideas_js = _static_js("screens-ideas.js")
    help_js = _static_js("screens-help.js")
    ext_js = _static_js("screens-extraction.js")
    settings_js = _static_js("screens-settings.js")
    app_js = _static_js("app.js")
    guided_js = _static_js("screens-guided.js")

    for owner in (viz_js, crossdb_js, ext_js):
        assert 'data-study-target="agent"' not in owner
    for owner in (viz_js, crossdb_js):
        assert 'data-study-target="guided"' in owner
    assert "data-ex-sync-guided" in ext_js
    assert "syncExtractionToCopilot" in ext_js
    assert "Continue in Guided Copilot" in viz_js
    assert "Continue in Guided Copilot" in ext_js
    assert "Plan in Guided Copilot" in crossdb_js

    for owner in (ideas_js, help_js, settings_js, guided_js):
        assert "Open Project Monitor" in owner
        assert "Open Agent Projects" not in owner
    assert "t('Project Monitor', '项目监控')" in app_js


def test_extraction_outputs_are_local_open_controls_and_sync_is_visible() -> None:
    index_html = _static_html("index.html")
    api_js = _static_js("api.js")
    extraction_js = _static_js("screens-extraction.js")
    embedded_js = _static_js("screens-extraction-embedded.js")
    guided_js = _static_js("screens-guided-pi.js")
    output_css = _static_css("extraction-output.css")

    assert "css/extraction-output.css?v=20260824-local-open1" in index_html
    assert "js/screens-extraction-embedded.js?v=20260825-source-binding1" in index_html
    assert "js/screens-guided-pi-starters.js?v=20260827-independent-starters1" in index_html
    assert "js/screens-guided-pi-header.js?v=20260827-type-scale1" in index_html
    assert "js/screens-guided-pi.js?v=20260828-plan-resubmit1" in index_html
    assert "/api/jobs/' + encodeURIComponent(jobId || '') + '/open-output" in api_js
    assert "window.EU_API.openExtractionOutput = openExtractionOutput" in api_js
    assert "data-ex-open-output" in extraction_js
    assert "column_metadata" in extraction_js
    assert "syncToCopilot: syncExtractionToCopilot" in extraction_js
    assert "notifyExtractionHandoff" in embedded_js
    assert "role: 'workflow_receipt'" in _static_js(
        "screens-guided-pi-data-binding.js"
    )
    assert "row.receipt_kind === 'extraction_result'" in guided_js
    assert "Extraction setup saved" in guided_js
    assert "No extraction has been claimed yet." in guided_js
    assert "StudyContext" in guided_js
    assert "This is EasyICU state, not a model reply." in guided_js
    assert ".ex-output-path" in output_css
    assert ".ex-output-file" in output_css
    assert "!important" not in output_css
    for foreign_route in ("crossdb", "patient", "cohort", "settings", "agent"):
        assert foreign_route not in output_css.lower()


def test_agent_run_status_labels_cover_success_statuses() -> None:
    """[11] The two most common completed-run statuses must be humanized, and the
    Runs history tab must route status through runStatusLabel (no raw snake_case)."""
    render_js = _static_js("screens-agent-render.js")
    agent_js = _static_js("screens-agent.js")
    assert "publication_ready: t('publication-ready'" in render_js
    assert "manuscript_ready: t('manuscript-ready'" in render_js
    # Runs history tab humanizes the status token + the tampered tag (no raw snake_case)
    assert "runStatusLabel(status)" in agent_js
    assert "changed since sign-off" in agent_js


def test_crossdb_selected_density_plot_has_readable_axis_and_legend() -> None:
    """The single selected-feature chart exposes x-axis values, units, and a
    source legend so readers can identify where aggregate distributions diverge."""
    charts_js = _static_js("screens-viz-crossdb-charts.js")
    crossdb_css = _static_css("crossdb.css")
    assert "type: 'line'" in charts_js
    assert "Relative density" in charts_js
    assert "legend: chartCore.legend" in charts_js
    assert "smooth: false" in charts_js
    assert "data-crossdb-echart" in charts_js
    assert ".xdb-echart" in crossdb_css
    assert ".xdb-main-grid" in crossdb_css
    assert ".xdb-main-legend" in crossdb_css


def test_review_routes_share_echarts_theme_without_cross_route_owner_leaks() -> None:
    index_html = _static_html("index.html")
    shared_js = _static_js("screens-viz-echarts.js")
    patient_js = _static_js("screens-viz-patient-charts.js")
    cohort_js = _static_js("screens-viz-cohort-charts.js")
    crossdb_js = _static_js("screens-viz-crossdb-charts.js")
    cohort_css = _static_css("cohort.css")
    cohort_charts_css = _static_css("cohort-charts.css")
    patient_css = _static_css("patient-series.css")
    crossdb_css = _static_css("crossdb.css")

    vendor = index_html.index("vendor/echarts/echarts.common.min.js?v=6.1.0")
    shared = index_html.index("js/screens-viz-echarts.js?v=20260728-shared-echarts1")
    patient = index_html.index("js/screens-viz-patient-charts.js?v=20260728-shared-echarts1")
    cohort = index_html.index("js/screens-viz-cohort-charts.js?v=20260728-shared-echarts1")
    crossdb = index_html.index("js/screens-viz-crossdb-charts.js?v=20260728-shared-echarts1")
    assert vendor < shared < patient < cohort < crossdb

    assert "window.EU_ECHARTS = {" in shared_js
    assert "renderer: 'svg'" in shared_js
    assert "renderMode: 'richText'" in shared_js
    assert "new ResizeObserver(() => chart.resize())" in shared_js
    assert "owner: 'patient'" in patient_js
    assert "owner: 'cohort'" in cohort_js
    assert "owner: 'crossdb'" in crossdb_js
    assert "window.echarts.init" not in cohort_js
    assert "window.echarts.init" not in crossdb_js

    assert ".pt-echart" in patient_css
    assert ".pt-echart" not in cohort_charts_css
    assert ".pt-echart" not in crossdb_css
    assert ".cohort-echart" in cohort_charts_css
    assert ".cohort-echart" not in patient_css
    assert ".cohort-echart" not in crossdb_css
    assert ".xdb-echart" in crossdb_css
    assert ".xdb-echart" not in patient_css
    assert ".xdb-echart" not in cohort_charts_css
    for stale_patient_selector in (".pvt-", ".pcat", ".pcc-", ".pcs-"):
        assert stale_patient_selector not in cohort_css


def test_km_panel_surfaces_effect_size() -> None:
    """[6] The KM panel must surface an effect contrast (end-of-follow-up survival +
    absolute risk difference), not just a lone log-rank p-value."""
    viz_js = _static_js("screens-viz.js")
    cohort_css = _static_css("cohort.css")
    assert "function cohortSurvivalEffect(curve)" in viz_js
    assert "absolute risk difference" in viz_js
    assert ".surv-effect" in cohort_css


def test_review_breadcrumb_parent_matches_sidebar_group() -> None:
    """[25]/[33] The review-screen breadcrumb parent must be 'Data Workspace' (the
    sidebar group name), not the orphan 'Data Visualization'."""
    viz_js = _static_js("screens-viz.js")
    app_js = _static_js("app.js")
    assert "'Home', 'Data Workspace'" in viz_js
    assert "'Data Visualization'" not in viz_js  # gone as a crumb parent
    assert "'Data Workspace': 'patient'" not in app_js
    assert "CRUMB_NAV" not in app_js  # the group label is not a fake patient link
    assert 'else node = `<span class="mid">${label}</span>`;' in app_js


def test_seeded_demo_pipeline_is_named_guided_copilot_and_labeled() -> None:
    """[23] The seeded demo pipeline must identify as 'Guided Copilot' (one name)
    and flag itself as a scripted demo, not the divergent 'Research Copilot'."""
    guided_js = _static_js("screens-guided.js")
    assert "Research Copilot" not in guided_js
    assert "研究 Copilot" not in guided_js
    assert "scripted demo walkthrough" in guided_js
    assert "你好，我是<strong>研究引导</strong>" in guided_js


def test_guided_frontdoor_offers_one_click_starter_folder() -> None:
    """First-run usability: a new user must be able to start a chosen goal without
    the folder-path dialog. The frontdoor offers a one-click '@folderquick' starter
    that creates a metadata-only folder and resumes the pending goal."""
    guided_js = _static_js("screens-guided.js")
    # one-click token + handler + helper all present
    assert "@folderquick" in guided_js
    assert "function quickCreateGuidedStarterFolder(" in guided_js
    # the require-folder gate leads with the one-click option (not only the dialog)
    assert "Create a starter folder & continue" in guided_js
    # quick-create resumes the goal the user already picked
    assert "continueGoal" in guided_js
    # the frontdoor banner is reassuring, not a hard prerequisite wall
    assert "Start by binding a local study folder" not in guided_js
    assert "Pick a goal to start" in guided_js


def test_science_workbench_framed_as_part_of_study() -> None:
    """[12] The Science Workbench must read as an advanced view OF the current study
    (tied to Runs/Outputs), not a bolted-on second app with disconnected vocabulary."""
    science_js = _static_js("screens-agent-science.js")
    assert "Not a separate tool" in science_js
    assert "the same run" in science_js.lower()
    # implementation-heritage jargon must not leak to clinical users
    assert "molecular biology renderers" not in science_js


def test_science_tab_is_merged_into_agent_flow_as_evidence() -> None:
    """[12] full IA merge: the standalone 'Science Workbench' identity is dissolved
    into the Project Monitor flow. The tab is named 'Evidence', the panel no longer
    announces a separate app, the Claude-Science reference card is removed, and the
    Evidence view is cross-linked bidirectionally with Outputs."""
    agent_js = _static_js("screens-agent.js")
    science_js = _static_js("screens-agent-science.js")
    science_css = _static_css("agent-science.css")

    # Tab renamed Science -> Evidence (both idea + full tab arrays); no user-facing
    # "Science" / "科学工作台" identity strings survive in either owner file.
    assert "t('Evidence', '证据')" in agent_js
    assert "t('Science', '科学工作台')" not in agent_js
    assert "Science Workbench" not in agent_js and "Science Workbench" not in science_js
    assert "科学工作台" not in agent_js and "科学工作台" not in science_js

    # The panel is framed as Evidence & provenance, not a self-announcing app card.
    assert "Evidence & provenance" in science_js
    assert "ag-evidence-panel" in science_js

    # The Claude-Science mimicry reference card is gone.
    assert "referenceCard" not in science_js
    assert "Claude Science" not in science_js
    assert "visual_reference" not in science_js

    # Bidirectional cross-links: Outputs -> Evidence and Evidence -> Outputs.
    assert 'data-ag-tab="science"' in agent_js  # Outputs header link
    assert 'data-ag-tab="outputs"' in science_js  # Evidence "Back to Outputs"

    # Inner section nav is de-emphasised to a subordinate sub-control (still a
    # tablist for a11y), not a second app-level tab bar.
    assert "ag-sci-sections-label" in science_js
    assert ".ag-evidence-panel .ag-sci-sections-label" in science_css


def test_seeded_autopilot_pipeline_is_demo_contained() -> None:
    """The seeded 'Product B' autopilot/welcome pipeline stays unreachable in Real
    mode: autopilot self-forces demo, the dock bridge routes Real -> real frontdoor,
    and frontdoor free-text returns before the seeded parseIntent/autopilot router."""
    guided_js = _static_js("screens-guided.js")
    # autopilot self-forces the demo data mode
    assert "autop = true; branch = branch || 'predict'; dataMode = 'demo';" in guided_js
    # the dock bridge gates Real mode to the real frontdoor, never seeded welcome
    assert "if (dataMode === 'real') {" in guided_js
    # frontdoor free-text is fully handled (returns) before parseIntent is reached
    frontdoor_guard = guided_js.find("if (currentId === 'frontdoor') {")
    parse_call = guided_js.find("const fn = parseIntent(v);")
    assert frontdoor_guard != -1 and parse_call != -1
    assert frontdoor_guard < parse_call


def test_evidence_module_has_no_hardcoded_dual_language_literals() -> None:
    """Design-consistency lock: the Evidence module must localize via bi(en, zh)
    like every other screen — never hardcoded 'English / 中文' dual literals that
    ignore the language toggle and double the visual text density."""
    import re

    science_js = _static_js("screens-agent-science.js")
    # single-quoted literals whose EN half precedes ' / ' and whose second half
    # contains CJK — the pattern the 2026-07 cleanup removed (46+ instances)
    dual_quoted = re.findall(r"'[A-Za-z][^'一-鿿]* / [^']*[一-鿿][^']*'", science_js)
    assert dual_quoted == [], f"hardcoded dual literals reintroduced: {dual_quoted[:5]}"
    # raw dual text embedded in HTML template markup
    dual_html = re.findall(r">[A-Z][A-Za-z ,.-]+ / [一-鿿][^<]*<", science_js)
    assert dual_html == [], f"hardcoded dual HTML text reintroduced: {dual_html[:5]}"
    # backend payload labels also arrive as duals — the load-time localizer must
    # stay wired so they render in ONE language like the rest of the app
    assert "function biLabel(" in science_js
    assert "function localizeDualLabels(" in science_js
    assert "state.data = localizeDualLabels(data);" in science_js


def test_destination_names_consistent_across_sidebar_crumb_and_page() -> None:
    """Product-logic lock: one destination has ONE name. Sidebar labels, crumb
    labels, and page heads must not drift (患者审阅 vs 患者明细 vs 加载审阅工作区
    made the same screen read as three different places)."""
    app_js = _static_js("app.js")
    viz_js = _static_js("screens-viz.js")
    help_js = _static_js("screens-help.js")
    dock_js = _static_js("copilot-dock.js")
    series_js = _static_js("screens-viz-patient-series.js")

    # canonical zh names in the shell crumb table
    assert "'Patient Review': ['Patient Review', '患者审阅']" in app_js
    assert (
        "'Cross-database comparison': ['Cross-database comparison', '跨库对比']"
        in app_js
    )
    # the retired aliases must not resurface anywhere user-facing
    for src in (viz_js, help_js, dock_js, series_js):
        assert "患者明细" not in src
        assert "跨库基准" not in src
    # the patient idle page head identifies itself as the destination, not as a
    # generic "Quick visualization" tool
    assert "${t('Patient Review', '患者审阅')}" in viz_js
    assert "'Quick visualization', '快速可视化'" not in viz_js
    # cohort keeps one constant page title; the load-state moves to the lead
    assert (
        "<h1 style=\"margin-top:0;\">${t('Cohort Statistics', '队列统计')}</h1>"
        in viz_js
    )


def test_cohort_and_crossdb_consume_guided_handoff() -> None:
    """A study configured in Guided Copilot must not silently vanish when the
    conversation lands the user on Cohort Statistics or Cross-database comparison."""
    viz_js = _static_js("screens-viz.js")
    setup_js = _static_js("screens-viz-crossdb-setup.js")
    assert "window.EU_GUIDED_HANDOFF.take('cohort')" in viz_js
    assert "window.EU_GUIDED_HANDOFF.noteHtml('cohort')" in viz_js
    assert "window.EU_GUIDED_HANDOFF.take('crossdb')" in setup_js
    assert "window.EU_GUIDED_HANDOFF.noteHtml('crossdb')" in setup_js


def test_topbar_actions_only_appear_once_workspace_is_loaded() -> None:
    """Before any data is loaded the page body owns the single primary action;
    a context-free topbar "Render"/"Run" button is noise that confused users."""
    viz_js = _static_js("screens-viz.js")
    setup_js = _static_js("screens-viz-crossdb-setup.js")
    # the pre-load Render button is gone from the patient screen
    assert "${t('Render', '渲染')}" not in viz_js
    # each viz actionHtml falls through to an empty string when not loaded
    assert viz_js.count("Topbar actions only exist once") == 2
    assert "function actionHtml(config)" in setup_js
    assert "if (!loaded) return '';" in setup_js
    assert "${rawLoaded ? '' : `<button" in setup_js


def test_result_charts_carry_reading_captions() -> None:
    """Every result-bearing chart explains what it means and what to do next:
    KM curve, SOFA transition matrix, group comparison bars, Cross-DB density
    view, and the per-database record cards."""
    viz_js = _static_js("screens-viz.js")
    crossdb_results_js = _static_js("screens-viz-crossdb-results.js")
    assert viz_js.count('class="viz-cap"') >= 5
    assert "曲线每下降一格代表一次事件" in viz_js  # KM
    assert "对角线上的格子是 SOFA-1 与 SOFA-2 评分一致的患者" in viz_js  # SOFA matrix
    assert "仅为描述性对比，未做统计检验" in viz_js  # group bars
    assert "曲线重叠表示聚合测量分布较一致" in crossdb_results_js
    assert "不是结局结果" in crossdb_results_js
    app_css = _static_css("app.css")
    assert ".viz-cap{" in app_css


def test_demo_mode_is_unmistakable_and_single_source_of_truth() -> None:
    """Demo state must be loud (amber topbar segment + explicit tooltip) and the
    global toggle is the source of truth on Guided entry (sync-up, demo→real)."""
    app_js = _static_js("app.js")
    i18n_js = _static_js("i18n.js")
    app_css = _static_css("app.css")
    guided_js = _static_js("screens-guided.js")
    assert "demo-active" in app_js
    assert ".mode-seg.demo-active" in app_css
    assert "官方公开去标识 Demo 数据集" in app_js
    assert "明确标注的种子示例" in app_js
    assert "所有数字都是种子示例" not in app_js
    assert "window.EU_DATA_MODE_CONTEXT = null;" in i18n_js
    assert "window.setDataModeContext = function" in i18n_js
    assert "window.getDataMode = function" in i18n_js
    assert "window.setDataModeContext(null);" in i18n_js
    assert "const dataMode = displayedDataMode();" in app_js
    assert "t('Official demo', '官方演示')" in app_js
    assert (
        "if (window.EU_DATA === 'real' && dataMode !== 'real') dataMode = 'real';"
        in guided_js
    )
    # demo mode no longer ships fake decorative sliders on the patient setup card
    viz_js = _static_js("screens-viz.js")
    demo_sources_js = _static_js("screens-viz-patient-demo-sources.js")
    assert "数据时长（小时）" not in viz_js
    assert "选择官方去标识化 ICU 演示数据" in viz_js
    assert "官方公开 ICU 演示数据" in demo_sources_js
    assert "加载合成兜底" in demo_sources_js
    assert "processing_mode: 'real'" in demo_sources_js
    assert "const realMode = dataMode === 'real';" in viz_js
    assert "localStorage.setItem('easyicu_home_data', 'real')" not in viz_js[
        viz_js.index("openPrepared: sourceId =>") : viz_js.index(
            "root.querySelectorAll('.radio[data-datamode]')"
        )
    ]


def test_dead_end_screens_gained_onward_paths() -> None:
    """Cross-DB, the agent run-history empty state, the export picker wall and
    the Data Dictionary all offer an explicit next step instead of dead-ending."""
    viz_js = _static_js("screens-viz.js")
    crossdb_results_js = _static_js("screens-viz-crossdb-results.js")
    agent_js = _static_js("screens-agent.js")
    dict_js = _static_js("screens-dict.js")
    # crossdb loaded nextbar links back to cohort as well as forward to agent
    assert "返回队列统计" in crossdb_results_js
    # Project Monitor sends configuration and execution back to Guided Copilot.
    assert "请在研究引导中确认研究并发起受治理运行" in agent_js
    # example projects are chip-labelled per item in demo mode
    assert "${t('Example', '示例')} · " in agent_js
    # export picker folds older registrations instead of rendering a wall
    assert 'class="src-fold"' in viz_js
    assert "个较早注册的导出" in viz_js
    # dictionary points onward to extraction
    assert "到「数据抽取」勾选它们所属的模块" in dict_js
