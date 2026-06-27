from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from easyicu import concept_catalog as cc
from easyicu.webserver import dataio
from easyicu.webserver.app import app

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

    assert "const FALLBACK_ROUTE = 'entry';" in app_js
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
    assert (
        "window.addEventListener('easyicu:languagechange', refreshLanguage)" in dock_js
    )
    assert "window.EUPageGuide = { open, close, toggle, refreshLanguage }" in dock_js
    assert "setTimeout(refreshLanguage, 250)" in dock_js
    assert "window.setLang(val);" in settings_js
    assert "window.EU_LANG = val;" not in settings_js
    assert "window.EU_API.saveSetting('data_mode', m)" in i18n_js
    assert "js/i18n.js?v=20260626-language-refresh-dock" in index_html
    assert "js/api.js?v=20260627-idea-plan" in index_html


def test_native_mobile_page_guide_fab_does_not_cover_bottom_nav() -> None:
    dock_css = _static_css("dock.css")

    assert (
        "#cpFab{ right: 14px; bottom: calc(76px + env(safe-area-inset-bottom)); }"
        in dock_css
    )


def test_native_assistant_labels_disambiguate_page_guide_guided_copilot_and_agent_guide() -> (
    None
):
    app_js = _static_js("app.js")
    dock_js = _static_js("copilot-dock.js")
    extraction_js = _static_js("screens-extraction.js")
    agent_js = _static_js("screens-agent.js")
    help_js = _static_js("screens-help.js")
    index_html = _static_html("index.html")

    assert "Page guide" in app_js
    assert "页面指南" in app_js
    assert "window.EUPageGuide || window.EUCopilot" in app_js
    assert "Guided study" in app_js
    assert "Agent guide" in agent_js
    assert "Open EasyICU page guide" in dock_js
    assert "打开 EasyICU 页面指南" in dock_js
    assert "Page guide" in dock_js
    assert "页面指南" in dock_js
    assert "Open Guided Copilot" in dock_js
    assert "当前页面 · 安全快捷操作 · 仅本地" in dock_js
    assert "label: bi('Cohort Statistics', '队列统计')" in dock_js
    assert "Start Guided study" in extraction_js
    assert "Continue in Guided study" in agent_js
    assert "Open Page guide" in help_js

    assert "Quick help" not in app_js
    assert "Quick help" not in dock_js
    assert "Quick help" not in help_js
    assert 'title="Ask the Copilot"' not in app_js
    assert "${t('Copilot','助手')}" not in app_js
    assert "Open EasyICU Copilot" not in dock_js
    assert '<div class="cp-name">Copilot</div>' not in dock_js
    assert "Let Copilot drive" not in extraction_js
    assert "Continue in Copilot" not in agent_js
    assert "Open Copilot" not in help_js

    assert "css/dock.css?v=20260625-stage99" in index_html
    assert "js/app.js?v=20260626-nav-stale-cleanup" in index_html
    assert "js/copilot-dock.js?v=20260626-page-guide-refresh2" in index_html
    assert "js/screens-extraction.js?v=20260627-extraction-sepsis-runtime" in index_html
    assert "js/screens-agent.js?v=20260627-canonical9-import" in index_html
    assert "js/screens-help.js?v=20260626-tutorial-i18n" in index_html


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

    assert "js/app.js?v=20260626-nav-stale-cleanup" in index_html
    assert "js/screens-help.js?v=20260626-tutorial-i18n" in index_html


def test_native_guided_and_page_guide_messages_are_bilingual() -> None:
    guided_js = _static_js("screens-guided.js")
    dock_js = _static_js("copilot-dock.js")
    index_html = _static_html("index.html")

    assert "function bi(en, zh)" in guided_js
    assert "function htmlOf(value)" in guided_js
    assert "htmlOf(t.html)" in guided_js
    assert "你好，我是 EasyICU <strong>研究 Copilot</strong>" in guided_js
    assert "你可以随时停下" in guided_js
    assert "正在打开 Guided Copilot" in dock_js
    assert "页面指南会解释当前页面" in dock_js
    assert "页面指南只支持固定快捷操作" in dock_js
    assert "htmlOf(t.html)" in dock_js
    assert "htmlOf(label)" in dock_js
    assert (
        "js/screens-guided-projects.js?v=20260626-guided-projects-split" in index_html
    )
    assert (
        "js/screens-guided-idea-provider.js?v=20260626-guided-api-first" in index_html
    )
    assert "js/screens-guided.js?v=20260627-guided-idea-plan" in index_html
    assert "js/copilot-dock.js?v=20260626-page-guide-refresh2" in index_html


def test_native_page_guide_uses_backend_page_guide_contract() -> None:
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

    assert "function currentContext()" in dock_js
    assert "ensureSession()" in dock_js
    assert "sendBackendMessage" in dock_js
    assert "runBackendAction" in dock_js
    assert "data-cp-action" in dock_js
    assert "selected_source" in dock_js
    assert "createPageGuideSession" in dock_js
    assert "sendPageGuideMessage" in dock_js
    assert "runPageGuideAction" in dock_js
    assert "createCopilotSession" not in dock_js
    assert "sendCopilotMessage" not in dock_js
    assert "runCopilotAction" not in dock_js
    assert "Page guide backend unavailable, using local fallback" in dock_js
    assert "js/api.js?v=20260627-idea-plan" in index_html
    assert "js/copilot-dock.js?v=20260626-page-guide-refresh2" in index_html


def test_native_guided_copilot_runs_extraction_inline_and_answers_catalog_questions() -> (
    None
):
    guided_js = _static_js("screens-guided.js")
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
    assert "llm_provider: 'mock'" in guided_js
    assert "external_llm_opt_in: false" in guided_js
    assert "goal === 'run_agent'" in guided_js
    assert "isGuidedAgentIntent(v)" in guided_js
    assert "function startGuidedIdeaFlow" in guided_js
    assert "thread.push({ guidedIdeaApiSetup: true })" in guided_js
    assert "function renderGuidedIdeaApiSetupCard" in guided_js
    assert "function showGuidedIdeaSourceForm" in guided_js
    assert "function showGuidedIdeaApiSetup" in guided_js
    assert "function renderGuidedIdeaCard" in guided_js
    assert "function runGuidedIdeaMine" in guided_js
    assert "function runGuidedIdeaPriorArt" in guided_js
    assert "function runGuidedIdeaHandoff" in guided_js
    assert "function runGuidedIdeaCreateProject" in guided_js
    assert "window.EU_API.mineIdeas" in guided_js
    assert "window.EU_API.resolveIdeaSource" in guided_js
    assert "window.EU_API.ingestIdeaPdf" in guided_js
    assert "window.EU_API.scanIdeaLiteratureFolder" in guided_js
    assert "window.EU_GUIDED_IDEA_PROVIDER.requestStatus" in guided_js
    assert "loadAgentProviderStatus" in provider_js
    assert "window.EU_API.saveAgentProviderConfig" in _static_js("api.js")
    assert "saveAgentProviderConfig" in provider_js
    assert "window.EU_API.checkIdeaPriorArt" in guided_js
    assert "window.EU_API.handoffIdea" in guided_js
    assert "window.EU_API.createIdeaAgentProject" in guided_js
    assert "goal === 'idea_mining'" in guided_js
    assert "isGuidedIdeaIntent(v)" in guided_js
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
    assert "function saveGuidedIdeaProviderConfig" in guided_js
    assert "function requestGuidedIdeaProviderStatus" in guided_js
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

    guided_plan_js = _static_js("screens-guided-idea-plan.js")
    guided_plan_css = _static_css("guided-idea-plan.css")
    redesign_css = _static_css("redesign.css")

    assert "css/guided.css?v=20260627-guided-source-tabs" in index_html
    assert "css/guided-idea-plan.css?v=20260627-guided-idea-plan" in index_html
    assert "js/api.js?v=20260627-idea-plan" in index_html
    assert (
        "js/screens-guided-projects.js?v=20260626-guided-projects-split" in index_html
    )
    provider_pos = index_html.find("screens-guided-idea-provider.js")
    projects_pos = index_html.find("screens-guided-projects.js")
    idea_plan_pos = index_html.find("screens-guided-idea-plan.js")
    guided_pos = index_html.find("screens-guided.js?")
    assert projects_pos != -1 and provider_pos != -1 and idea_plan_pos != -1 and guided_pos != -1
    assert projects_pos < guided_pos
    assert provider_pos < guided_pos
    assert provider_pos < idea_plan_pos < guided_pos
    assert "window.EU_GUIDED_PROJECTS = {" in projects_js
    assert "window.EU_GUIDED_IDEA_PLAN = {" in guided_plan_js
    assert "window.EU_GUIDED_IDEA_PLAN.render" in guided_js
    assert "guidedProjectContext()" in guided_js
    assert "function runGuidedIdeaPlan" in guided_js
    assert "window.EU_API.planIdea" in guided_js
    assert "data-gi-plan" in guided_js
    assert "data-gi-replan" in guided_js
    assert "Create a study plan before Agent handoff" in guided_plan_js
    assert "Reference method patterns" in guided_plan_js
    assert "ICU constraints" in guided_plan_js
    assert "Plan / replan notes" in guided_plan_js
    assert "Applied replan note" in guided_plan_js
    assert "Generate and review the study plan before freezing an Agent handoff" in guided_js
    assert "restoreGuidedIdeaArtifacts" in guided_js
    assert "dataContextConfirmed" in guided_js
    assert "function confirmGuidedIdeaDataContext" in guided_js
    assert "This only turns a source clue into a candidate research question" in guided_js
    assert "requires explicit data-context confirmation" in guided_js
    assert "Manual idea mode" in guided_js
    assert "Article URL mode" in guided_js
    assert "PDF file mode" in guided_js
    assert "Literature folder mode" in guided_js
    assert "Frontier topic mode" in guided_js
    assert "source-${attr(tab)}" in guided_js
    assert ".gdi-plan-details" in guided_plan_css
    assert ".gdi-feature-row.one" in guided_plan_css
    assert ".gdi-plan-details" not in redesign_css
    assert "js/screens-guided.js?v=20260627-guided-idea-plan" in index_html


def test_native_agent_outputs_fail_closed_to_real_artifacts() -> None:
    agent_js = _static_js("screens-agent.js")
    agent_css = _static_css("agent.css")
    agent_cap_css = _static_css("agent-capabilities.css")
    index_html = _static_html("index.html")

    assert "js/screens-agent.js?v=20260627-canonical9-import" in index_html
    assert "css/agent.css?v=20260627-canonical9-import" in index_html
    assert "css/agent-capabilities.css?v=20260627-agent-capabilities" in index_html
    assert "function artifactsForLive(live)" in agent_js
    assert "function reviewableRunForStudy()" in agent_js
    assert "function outputCountForStudy()" in agent_js
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
    assert "Agent Projects route-owned layout fixes" in agent_css
    assert ".ag-pipe .pline" in agent_css
    assert ".ag-pipe .pstep" in agent_css
    assert ".ag-pipe .pt" in agent_css
    assert ".ag-pipe .pd" in agent_css
    assert ".ag-wrap .chip" in agent_css
    assert "Agent Projects output review cards" in agent_css
    assert ".ag-wrap .ag-output-brief" in agent_css
    assert ".ag-wrap .ag-featured-results" in agent_css
    assert ".ag-wrap .outcard.on" in agent_css
    assert "Agent Projects capability highlights" in agent_cap_css
    assert ".ag-wrap .ag-cap-grid" in agent_cap_css
    assert ".ag-wrap .ag-link-chain" in agent_cap_css


def test_native_agent_research_blocks_are_project_owned() -> None:
    agent_js = _static_js("screens-agent.js")
    agent_css = _static_css("agent.css")
    app_js = _static_js("app.js")
    screens_css = _static_css("screens.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert "const BLOCK_LIBRARY = [" in agent_js
    assert "const BLOCK_FAMILIES = [" in agent_js
    assert "AG_BLOCKS_VERSION" in agent_js
    assert "Workflow Blocks" in agent_js
    assert "Research Blocks" in agent_js
    assert "Nature writing block" in agent_js
    assert "Nature figure block" in agent_js
    assert "Outcome-blind feasibility" in agent_js
    assert "Evidence-bound analysis run" in agent_js
    assert "Data availability block" in agent_js
    assert "data-ag-block-add" in agent_js
    assert "data-ag-block-pack" in agent_js
    assert "workflowBlocks(s).length" in agent_js
    assert (
        "Blocks define required inputs, generated artifacts, and evidence contracts"
        in agent_js
    )
    assert (
        "They do not change global prompts or run anything until you explicitly start an Agent run"
        in agent_js
    )

    assert "Agent Projects Research Blocks module" in agent_css
    assert ".ag-block-grid" in agent_css
    assert ".ag-wf-row" in agent_css
    assert ".ag-lib-card" in agent_css
    assert ".ag-block-contract" in agent_css
    assert "css/agent.css?v=20260627-canonical9-import" in index_html
    assert "js/screens-agent.js?v=20260627-canonical9-import" in index_html

    assert "ag-block-grid" not in app_js
    assert "Research Blocks" not in app_js
    assert ".ag-block-grid" not in screens_css
    assert ".ag-wf-row" not in screens_css
    assert ".ag-lib-card" not in screens_css
    assert ".ag-block-grid" not in redesign_css
    assert ".ag-wf-row" not in redesign_css
    assert ".ag-lib-card" not in redesign_css


def test_native_agent_canonical9_import_is_project_owned() -> None:
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
    assert "function runStatusLabel(status)" in agent_js
    assert "function readableArtifactText(value)" in agent_js
    assert "function evidenceLinkPanel(live, s)" in agent_js
    assert "function crossDataPanel(live, s)" in agent_js
    assert "function capabilityHighlights(live, s)" in agent_js
    assert "function questionParts(text)" in agent_js
    assert "function questionTags(s, raw)" in agent_js
    assert "function renderStructuredQuestion(s)" in agent_js
    assert "function focusAgentBody()" in agent_js
    assert "function presentationSummary(s)" in agent_js
    assert "function featuredFigurePreview(live)" in agent_js
    assert "benchmarkPanel(s)" in agent_js
    assert "data-ag-open-seed-run" in agent_js
    assert "figureGallery(data.payload || {})" in agent_js
    assert "benchmark_scorecard.json" in agent_js
    assert "workflow_graph.json" in agent_js
    assert "figure_gallery.json" in agent_js
    assert "source_run_manifest.json" in agent_js
    assert "Completed analysis" in agent_js
    assert "Research idea" in agent_js
    assert "Read-only review · manuscript not unlocked" in agent_js
    assert "Study brief" in agent_js
    assert "汇报摘要" in agent_js
    assert "verification passed" in agent_js
    assert "readableArtifactText(row.text || '')" in agent_js
    assert "Claim-to-artifact trace is explicit" in agent_js
    assert "Open Cross-DB workspace" in agent_js
    assert "Core question" in agent_js
    assert "Analysis requirements" in agent_js
    assert "Data context" in agent_js
    assert "scrollIntoView({ block: 'start', behavior: 'auto' })" in agent_js
    assert "想法种子" not in agent_js
    assert "九问运行" not in agent_js
    assert "Idea seed" not in agent_js
    assert "Canonical run" not in agent_js
    assert "s.id === 'aki' ? 'kdigo' : 'lactate'" not in agent_js
    assert "Agent Projects Canonical9 import module" in agent_css
    assert ".ag-bench-card" in agent_css
    assert ".ag-bench-metrics" in agent_css
    assert ".ag-score-grid" in agent_css
    assert ".ag-figure-gallery" in agent_css
    assert ".ag-present-brief" in agent_css
    assert ".ag-wrap .ag-output-brief" in agent_css
    assert ".ag-wrap .ag-featured-results" in agent_css
    assert ".ag-wrap .ag-cap-grid" in agent_cap_css
    assert ".ag-wrap .ag-cap-card.evidence" in agent_cap_css
    assert ".ag-wrap .ag-cap-card.cross" in agent_cap_css
    assert "Agent Projects structured question brief" in agent_question_css
    assert ".ag-wrap .ag-question-brief" in agent_question_css
    assert ".ag-wrap .ag-q-section + .ag-q-section" in agent_question_css
    assert ".ag-wrap .ag-req-list" in agent_question_css
    assert "css/agent-question.css?v=20260627-agent-question" in index_html

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
        ".table-scroll, .risk-table-wrap, .dict-table, .xdb-density-detail-table"
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
    i18n_js = _static_js("i18n.js")
    tweaks_css = _static_css("tweaks.css")
    index_html = _static_html("index.html")

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
    assert "per run in Agent Projects" in settings_js
    assert "strict enforced" in settings_js
    assert "There is no telemetry collector" in settings_js
    assert "恢复默认设置" in settings_js
    assert "工作区 · 设置" in settings_js
    assert "本地路径" in settings_js
    assert "本地优先保障" in settings_js
    assert "研究代理" in settings_js
    assert "选择默认导出文件夹" in settings_js
    assert "设置已恢复为后端默认值。" in settings_js
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
    assert 'body[data-reduce-motion="true"]' in tweaks_css
    assert "css/tweaks.css?v=20260625-stage96" in index_html
    assert "js/screens-settings.js?v=20260626-settings-i18n" in index_html
    assert "window.EU_API.saveSetting('data_mode', m)" in i18n_js
    assert "All controls are demo-interactive" not in settings_js


def test_native_extraction_advanced_filters_are_backend_wired() -> None:
    api_js = _static_js("api.js")
    extraction_js = _static_js("screens-extraction.js")

    assert "/api/fs/mkdir" in api_js
    assert "function createDir(path)" in api_js
    assert "window.EU_API.createDir = createDir" in api_js
    assert "/api/extraction/filter-options" in api_js
    assert "/api/extraction/filter-preview" in api_js
    assert "loadExtractionFilterOptions" in extraction_js
    assert "previewExtractionFilters" in extraction_js
    assert "Real-source filter audit" in extraction_js
    assert "Unsupported filters stay blocked" in extraction_js


def test_native_extraction_folder_connect_defaults_to_auto_detection() -> None:
    extraction_js = _static_js("screens-extraction.js")
    extraction_css = _static_css("extraction.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert "css/extraction.css?v=20260627-extraction-sepsis-runtime" in index_html
    assert "data-ex-analyze" in extraction_js
    assert "Analyze folder" in extraction_js
    assert "Let EasyICU identify the folder" in extraction_js
    assert "data-ex-manual" in extraction_js
    assert "Advanced: choose manually" in extraction_js
    assert "Use this only if automatic detection is wrong" in extraction_js
    assert "Then tell us what kind of folder it is" not in extraction_js
    assert ".ex-connect-card" in extraction_css
    assert ".ex-connect-primary" in extraction_css
    assert ".ex-connect-actions" in extraction_css
    assert ".ex-connect-card" not in redesign_css
    assert ".ex-connect-primary" not in redesign_css


def test_native_extraction_custom_modules_default_to_all_with_bulk_actions() -> None:
    extraction_js = _static_js("screens-extraction.js")
    sepsis_js = _static_js("screens-extraction-sepsis.js")
    extraction_css = _static_css("extraction.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert (
        "let exAdvCohort = false, exAdvExport = false, exShowAllMods = true;"
        in extraction_js
    )
    assert "['SOFA-1 scores', 'SOFA-1 评分', 7, true, false]" in extraction_js
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
    assert "function sepsisDefinitionPanel()" in extraction_js
    assert "sepsis_definition: sepsisDefinitionContract()" in extraction_js
    assert "window.EUExtractionSepsis.bind" in extraction_js
    assert "js/screens-extraction-sepsis.js?v=20260627-extraction-sepsis-runtime" in index_html
    assert index_html.find("screens-extraction-sepsis.js") < index_html.find("screens-extraction.js")
    assert "window.EUExtractionSepsis = {" in sepsis_js
    assert "metadata_current_runtime_defaults" in sepsis_js
    assert "Sepsis-3 implementation profile" in sepsis_js
    assert "These controls mirror easyicu.scores.sepsis.susp_inf()" in sepsis_js
    assert "const PROFILES = [" in sepsis_js
    assert "['sofa2_primary', 'SOFA-2 primary', 'SOFA-2 主口径', 'SOFA-2']" in sepsis_js
    assert "['sofa1_sensitivity', 'SOFA-1 sensitivity', 'SOFA-1 敏感性', 'SOFA-1']" in sepsis_js
    assert "['dual_audit', 'SOFA-2 + SOFA-1 audit', 'SOFA-2 + SOFA-1 审计', 'SOFA-2 + SOFA-1']" in sepsis_js
    assert "data-ex-sepsis-profile" in sepsis_js
    assert "root.querySelectorAll('[data-ex-sepsis-profile]')" in sepsis_js
    assert 'data-ex-sepsis="${key}"' in sepsis_js
    assert "root.querySelectorAll('[data-ex-sepsis]')" in sepsis_js
    assert "threshold: 2," in sepsis_js
    assert "abx_count_win_hours: state.abxCountWinHours" in sepsis_js
    assert "abx_min_count: state.abxMinCount" in sepsis_js
    assert "positive_cultures_required: state.positiveCultures" in sepsis_js
    assert "si_window: state.siWindow" in sepsis_js
    assert "delta_function: state.deltaFunction" in sepsis_js
    assert "threshold: state.threshold" in sepsis_js
    assert "optionSeg(ctx, 'threshold'" in sepsis_js
    assert "Δ ≥ 3" in sepsis_js
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
    assert ".sepsis-def-grid" in extraction_css
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
        r"\['([^']+)',\s*'[^']+',\s*(\d+),\s*true,\s*(true|false)\]", module_block
    )
    keys = dict(re.findall(r"'([^']+)':\s*'([^']+)'", key_block))

    assert len(entries) == len(cc.CONCEPT_GROUPS_INTERNAL) == 19
    assert len(cc.CONCEPT_DICTIONARY) == 247
    assert "function moduleConceptCount(m)" in extraction_js
    assert "window.EU_CATALOG && window.EU_CATALOG.groupConcepts" in extraction_js

    fallback_total = 0
    for name, count_text, _is_core in entries:
        group_key = keys[name]
        expected = len(cc.CONCEPT_GROUPS_INTERNAL[group_key])
        count = int(count_text)
        fallback_total += count
        assert count == expected, f"{name} fallback count should match {group_key}"

    assert fallback_total == len(cc.CONCEPT_DICTIONARY)
    assert fallback_total != 219


def test_native_idea_mining_is_first_class_route_and_backend_wired() -> None:
    app_js = _static_js("app.js")
    api_js = _static_js("api.js")
    icons_js = _static_js("icons.js")
    ideas_js = _static_js("screens-ideas.js")
    agent_js = _static_js("screens-agent.js")
    redesign_css = _static_css("redesign.css")
    ideas_css = _static_css("ideas.css")
    shell_css = _static_css("shell.css")
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'data-nav="ideas"' in app_js
    assert "Find a Study Idea" in app_js
    assert "Discovery & Plan" in app_js
    assert "ideas-entry" in app_js
    assert "paper, PDF, or topic → feasible plan" in app_js
    assert "Run a Research Project" in app_js
    assert "Data & Review" in app_js
    assert "Data Workspace" in app_js
    assert "const workspaceIndex = CLASSIC.findIndex(c => c.id === route);" in app_js
    assert "workspaceIndex + 1} / ${CLASSIC.length}" in app_js
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
    assert "css/ideas.css?v=20260627-ideas-source-tabs" in index_html
    assert "css/shell.css?v=20260626-owner" in index_html
    assert "js/icons.js?v=20260625-stage84" in index_html
    assert "js/app.js?v=20260626-nav-stale-cleanup" in index_html
    assert "js/screens-ideas.js?v=20260627-ideas-source-tabs" in index_html
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
    assert "/api/ideas/handoff" in api_js
    assert "/api/ideas/create-agent-project" in api_js
    assert "/api/ideas/agent-projects" in api_js
    assert "/api/ideas/history" in api_js
    assert "/api/ideas/run" in api_js
    assert "S.ideas =" in ideas_js
    assert "pre_experiment" in ideas_js
    assert "Plan / replan before Agent" in ideas_js
    assert "Generate study plan" in ideas_js
    assert "Reference method patterns" in ideas_js
    assert "ICU constraints" in ideas_js
    assert "Generate and review the study plan before freezing an Agent handoff" in ideas_js
    assert "data-idea-plan" in ideas_js
    assert "data-idea-replan" in ideas_js
    assert "Freeze handoff for Agent" in ideas_js
    assert "selectedRecordKey" in ideas_js
    assert "data-idea-record-key" in ideas_js
    assert "if (!rows.some(r => String(r.id) === String(current.id)))" in ideas_js
    assert "Create Agent project" in ideas_js
    assert "Check prior art" in ideas_js
    assert "Resolve source" in ideas_js
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
    assert "Separated from Research Projects" in ideas_js
    assert "ideas-source-card" in ideas_js
    assert "idea-ledger-card" in ideas_js
    assert "idea-ledger-grid" in ideas_js
    assert "ideas-pre-summary" in ideas_js
    assert "ideas-feature-row" in ideas_js
    assert "ideas-compact-details" in ideas_js
    assert "ideas-prior-card" in ideas_js
    assert "ideas-query-details" in ideas_js
    assert "ideas-plan-edits" in ideas_js
    assert "search: '<circle" in icons_js
    assert "rail-block" in ideas_js
    assert "setup-row" in ideas_js
    assert "rail-title" not in ideas_js
    assert "rail-kv" not in ideas_js
    assert "rail-note" not in ideas_js
    assert "grid4 mt-12" not in ideas_js
    assert "css/ideas.css" in index_html
    assert ".idea-workbench" in ideas_css
    assert ".ideas-primary-grid" in ideas_css
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
    assert "height:auto;" in ideas_css
    assert "min-height:92px;" in ideas_css
    assert "grid-template-columns:repeat(2,minmax(0,1fr));" in ideas_css
    assert "@media (min-width:1680px)" not in ideas_css
    assert ".idea-workbench .ideas-step-panel .statgrid" in ideas_css
    assert ".idea-workbench .ideas-step-panel table" in ideas_css
    assert ".idea-workbench" not in redesign_css
    assert ".ideas-primary-grid" not in redesign_css
    assert ".ideas-step-nav" not in redesign_css
    assert ".ideas-advanced" not in redesign_css
    assert ".ideas-entry" not in redesign_css
    assert ".wsi-sub" not in redesign_css
    assert ".nav-sec" in shell_css
    assert ".ideas-entry" in shell_css
    assert ".wsi-sub" in shell_css
    assert "function validatePayload(payload)" in ideas_js
    assert "function sourceSpecificForm()" in ideas_js
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
    assert "loadIdeaAgentProjects" in agent_js
    assert "seedStudy(row)" in agent_js
    assert "const DEMO_STUDIES" in agent_js
    assert "const base = realMode() ? [] : DEMO_STUDIES" in agent_js
    assert "No local projects yet" in agent_js
    assert "Agent Projects no longer shows fabricated studies in Real mode" in agent_js
    assert "No active registered export is selected" in agent_js
    assert "data-ag-mode" not in agent_js
    assert "Idea exploration" not in agent_js
    assert "Open Idea Mining" in agent_js


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


def test_native_extraction_exposes_real_cohort_gate_and_recommended_contract() -> None:
    extraction_js = _static_js("screens-extraction.js")

    assert "const DEFAULT_OBSERVATION_WINDOW_HOURS = 24 * 30;" in extraction_js
    assert "let exWindowHours = DEFAULT_OBSERVATION_WINDOW_HOURS;" in extraction_js
    assert "observation_window_hours: DEFAULT_OBSERVATION_WINDOW_HOURS" in extraction_js
    assert "MAX_OBSERVATION_WINDOW_HOURS" in extraction_js
    assert "full available · 30d cap" in extraction_js
    assert "first 24 hours" not in extraction_js
    assert "first 24h" not in extraction_js
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
        "Cancel requested. The current database read may finish before the job stops."
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
    assert sepsis_definition["suspected_infection"]["mode"] == "auto"
    assert sepsis_definition["suspected_infection"]["abx_win_hours"] == 48
    assert sepsis_definition["suspected_infection"]["samp_win_hours"] == 24
    assert sepsis_definition["suspected_infection"]["abx_count_win_hours"] == 12
    assert sepsis_definition["suspected_infection"]["abx_min_count"] == 2
    assert (
        sepsis_definition["suspected_infection"]["positive_cultures_required"]
        is True
    )
    assert sepsis_definition["sofa_increase"]["si_window"] == "last"
    assert sepsis_definition["sofa_increase"]["window_before_si_hours"] == 72
    assert sepsis_definition["sofa_increase"]["window_after_si_hours"] == 12
    assert sepsis_definition["sofa_increase"]["delta_function"] == "delta_start"
    assert sepsis_definition["sofa_increase"]["threshold"] == 3
    assert sepsis_definition["sofa_increase"]["keep_components"] is True
    runtime_kwargs = dataio._sepsis_runtime_kwargs(sepsis_definition)
    assert runtime_kwargs == {
        "si_mode": "auto",
        "abx_win": "48h",
        "samp_win": "24h",
        "abx_count_win": "12h",
        "abx_min_count": 2,
        "positive_cultures": True,
        "si_window": "last",
        "delta_fun": "delta_start",
        "sofa_thresh": 3,
        "si_lwr": "72h",
        "si_upr": "12h",
        "keep_components": True,
    }
    assert sepsis_definition["review_options"]["implementation_profile"] == [
        "sofa2_primary",
        "sofa1_sensitivity",
        "dual_audit",
    ]
    assert sepsis_definition["review_options"]["threshold"] == [2, 3]
    assert sepsis_definition["review_options"]["abx_min_count"] == [1, 2, 3]
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
    )

    assert "Sepsis runtime profile: `ui-test`" in readme
    assert "Sepsis implementation profile: `sofa1_sensitivity`" in readme
    assert "Sepsis score family: `SOFA-1`" in readme
    assert "ABX->sample `48h`, sample->ABX `24h`" in readme
    assert "ABX count `≥2/12h`, positive cultures `True`" in readme
    assert (
        "SI event `last`, window `-72h/+12h`, "
        "delta `delta_start`, threshold `3`, keep components `True`"
    ) in readme
    assert "Sepsis runtime kwargs: `{'si_mode': 'auto'" in readme
    assert "Definition note scope: `metadata_current_runtime_defaults`" in readme
    dataio_py = Path(dataio.__file__).read_text(encoding="utf-8")
    callbacks_py = (
        Path(dataio.__file__).resolve().parents[1] / "concept" / "callbacks.py"
    ).read_text(encoding="utf-8")
    assert "sepsis_load_kwargs = _sepsis_runtime_kwargs" in dataio_py
    assert "module_kwargs.update(sepsis_load_kwargs)" in dataio_py
    assert "abx_count_win=abx_count_win" in callbacks_py
    assert "sofa_thresh=_callback_int" in callbacks_py


def test_native_crossdb_restores_distribution_visuals() -> None:
    api_js = _static_js("api.js")
    viz_js = _static_js("screens-viz.js")
    screens_css = _static_css("screens.css")
    crossdb_css = _static_css("crossdb.css")
    index_html = _static_html("index.html")

    assert "startCrossdbRawDistributionJob" in api_js
    assert "/api/jobs/crossdb-raw-distribution" in api_js
    assert "scanCrossdbRawRoot" in api_js
    assert "/api/crossdb-review/raw-root-scan" in api_js
    assert "Multi-database feature density grid" in viz_js
    assert "crossRealFeatureDensityByModule" in viz_js
    assert "crossFeatureDensityPanel" in viz_js
    assert "loadDemoCrossdb" in viz_js
    assert "loadCrossdbDemoDistribution" in viz_js
    assert "legacy_simulated_multidb_feature_frames" in viz_js
    assert "feature_scope: 'all_catalog'" in viz_js
    assert "max_features: 90" not in viz_js
    assert "records_per_feature: 96" in viz_js
    assert "demoCurvePoints" not in viz_js
    assert "crossFeatureCurve" in viz_js
    assert "one subplot per feature" in viz_js
    assert "['SICdb', true, 'sic']" in viz_js
    assert "all supported catalog concepts" in viz_js
    assert "全部受支持的标准概念" in viz_js
    assert "Feature distribution by shared concept" not in viz_js
    assert "Module coverage distribution by export" not in viz_js
    assert "xdb-density-panel" in viz_js
    assert "data-density-module-select" in viz_js
    assert "Module to display" in viz_js
    assert "showing every catalog module" in viz_js
    assert "data-density-module-filter" in viz_js
    assert "data-density-module" in viz_js
    assert "data-density-feature-key" in viz_js
    assert "xdb-density-detail" in viz_js
    assert "startCrossdbRawDistributionJob" in viz_js
    assert "new EventSource('/api/jobs/' + r.job_id + '/events')" in viz_js
    assert "data-crossdb-cancel" in viz_js
    assert "data-crossdb-root-browse" in viz_js
    assert "data-crossdb-root-scan" in viz_js
    assert "Check folders" in viz_js
    assert "检查文件夹" in viz_js
    assert "Detected database folders" in viz_js
    assert "Missing selected database folders" in viz_js
    assert "Unrecognized folders" in viz_js
    assert "crossRawScanReadyFor" in viz_js
    assert "function crossRawSelectionStatusFor(" in viz_js
    assert (
        "toggling a database changes selection,\n          // not whether sibling folders were recognized"
        in viz_js
    )
    assert "if (window.EU_DATA === 'real') {\n          invalidateCrossRawRootScan();" not in viz_js
    assert "Check the ICU data root first" in viz_js
    assert "Choose local ICU data root" in viz_js
    assert "选择本地 ICU 数据根目录" in viz_js
    assert (
        "Local folder picker API is not ready. Paste a raw ICU data root path instead."
        in viz_js
    )
    assert "'/api/jobs/' + jobId + '/cancel'" in viz_js
    assert "Raw Cross-DB density job cancellation requested." in viz_js
    assert (
        "Choose a local ICU data root before loading real Cross-DB densities." in viz_js
    )
    assert "加载真实跨库密度前，请先选择本地 ICU 数据根目录。" in viz_js
    assert "crossRawRootDraft" in viz_js
    assert "let crossRawSampleMode = 'quick';" in viz_js
    assert "function crossRawSampleProfiles()" in viz_js
    assert "Quick preview" in viz_js
    assert "快速预览" in viz_js
    assert "maxPatients: 200" in viz_js
    assert "sampleSize: 600" in viz_js
    assert "data-crossdb-sample-mode" in viz_js
    assert "Sampling budget before plotting" in viz_js
    assert "绘图前抽样预算" in viz_js
    assert "max_patients: sampleProfile.maxPatients" in viz_js
    assert "sample_size: sampleProfile.sampleSize" in viz_js
    assert "Queued local raw Cross-DB density job" in viz_js
    assert "本地原始跨库密度任务已排队。" in viz_js
    assert "跨库对比" in viz_js
    assert "'Not configured': '未配置'" in viz_js
    assert "原始 ICU 数据根目录" in viz_js
    assert "加载真实密度对比" in viz_js
    assert "正在从本地数据库加载真实特征密度" in viz_js
    assert "选择要对比的数据库" in viz_js
    assert "多数据库特征密度网格" in viz_js
    assert "兼容性核验" in viz_js
    assert "opts.rawRoot" in viz_js
    assert "easyicu_crossdb_data_root" not in viz_js
    assert "easyicu_raw_data_root" not in viz_js
    assert "window.EU_DATA === 'real' && !crossRawJobId" not in viz_js
    assert "loadRealWorkspace(done);" not in viz_js
    assert "crossdbRunBound" in viz_js
    assert "crossdbRootBound" in viz_js
    assert "--xdb-grid-cols" in viz_js
    assert "xdb-density-svg" in viz_js
    assert "xdb-density-line" in viz_js
    assert "js/screens-viz.js?v=20260627-survival-outcome-summary" in index_html
    assert "css/crossdb.css?v=20260626-viz-richness" in index_html
    assert ".xdb-dist-panel" in crossdb_css
    assert ".xdb-dist-row" in crossdb_css
    assert ".xdb-density-panel" in crossdb_css
    assert ".xdb-density-selectrow" in crossdb_css
    assert ".xdb-density-controls" in crossdb_css
    assert ".xdb-density-detail" in crossdb_css
    assert "grid-template-columns:repeat(var(--xdb-grid-cols, 3)" in crossdb_css
    assert ".xdb-density-feature" in crossdb_css
    assert ".xdb-density-feature.selected" in crossdb_css
    assert ".xdb-density-svg" in crossdb_css
    assert ".xdb-density-line" in crossdb_css
    assert ".xdb-dist-panel" not in screens_css
    assert ".xdb-dist-row" not in screens_css
    assert ".xdb-density-panel" not in screens_css
    assert ".xdb-density-selectrow" not in screens_css


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
    assert "js/screens-viz.js?v=20260627-survival-outcome-summary" in index_html


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


def test_native_patient_view_renders_multichannel_vital_timeline() -> None:
    """Patient Time-Series view must overlay a multi-channel vital timeline (not
    only the bounded matrix), restoring the legacy vital-timeline subplots."""
    viz_js = _static_js("screens-viz.js")
    cohort_css = _static_css("cohort.css")

    assert "function patientVitalTimeline(" in viz_js
    assert "patientVitalTimeline(readyLanes" in viz_js
    assert "Vital-sign timeline" in viz_js
    assert "生命体征时间线" in viz_js
    # Owner-file: timeline SVG styles live with the other viz chart styles.
    assert ".pvt-svg" in cohort_css
    assert ".pvt-lane" in cohort_css
    # Timeline is additive — the bounded matrix ledger must remain.
    assert "Time-window × feature matrices" in viz_js


def test_native_crossdb_availability_matrix_is_a_heatmap() -> None:
    """The Cross-DB module availability matrix must colour cells by coverage,
    restoring the legacy availability heatmap instead of plain text cells."""
    viz_js = _static_js("screens-viz.js")
    crossdb_css = _static_css("crossdb.css")

    assert "function crossAvailCell(" in viz_js
    assert "(row.values || []).map(v => crossAvailCell(v))" in viz_js
    assert ".xdb-avail-cell" in crossdb_css
    assert "'Present': '存在'" in viz_js


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
    assert "data-catalog.js?v=20260625-stage93" in index_html
    assert "api.js?v=20260627-idea-plan" in index_html
    assert "screens-dict.js?v=20260626-dictionary-db-coverage" in index_html
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
    api_js = _static_js("api.js")
    index_html = _static_html("index.html")

    assert "loadAgentRunReview(row.project_dir)" in guided_js
    assert "/api/guided/drafts" in api_js
    assert "/api/guided/drafts/list" in api_js
    assert "/api/guided/drafts/remove" in api_js
    assert "loadGuidedDrafts({ limit: 20 })" in guided_js
    assert "createGuidedDraft(payload)" in guided_js
    assert "removeGuidedDraft" in api_js
    assert "data-remove-localdraft" in guided_js
    assert "removeLocalGuidedDraft(row)" in guided_js
    assert "delete_project_folder: false" in guided_js
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
    assert "Memory is scoped to" in guided_js
    assert "Idea Mining and Agent Projects still own their own artifacts" in guided_js
    assert "Start by binding a local study folder" in guided_js
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
        "If no folder is bound yet, I will ask you to create or open one first"
        in guided_js
    )
    assert "Find a Study Idea" in guided_js
    assert "Prepare Data" in guided_js
    assert "Run a Research Project" in guided_js
    assert "Study folders" in guided_js
    assert "Conversation memory" in guided_js
    assert "data-localdraft" in guided_js
    assert "Agent run artifacts" not in guided_js
    assert "Read-only results" not in guided_js
    assert "Run a confirmed Agent project" not in guided_js
    assert "data-localrun" not in guided_js
    assert "data-refreshruns" not in guided_js
    assert "loadAgentRunHistory({ limit: 20 })" not in guided_js
    assert "existing Agent run folder" in guided_js
    assert ".gd-rail-note" in guided_css
    assert "~/easyicu/projects" in guided_js
    assert "/Users/haibo" not in guided_js
    assert "Seeded example · not a local project" not in guided_js
    assert "Seeded examples" not in guided_js
    assert "data-sess" not in guided_js
    assert "That is a seeded example" not in guided_js
    assert "New / open study folder" in projects_js
    assert "gdFolderControls" in guided_js
    assert "gdFolderDialogHost" in guided_js
    assert "data-folder-menu-toggle" in projects_js
    assert "data-folder-choice" in projects_js
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
    assert 'class="gd-home-link"' in guided_js
    assert 'data-open="entry"' in guided_js
    assert "Back to EasyICU home" in guided_js
    assert 'class="gd-rail-utils"' in guided_js
    assert 'data-open="settings"' in guided_js
    assert "data-lang-toggle" in guided_js
    assert "Switch language" in guided_js
    assert "${t('Data workspace', '数据工作台')}" in guided_js
    assert ".gd-empty-local" in guided_css
    assert ".gd-sessline" in guided_css
    assert ".gd-sess-action" in guided_css
    assert ".gd-sess.draft.active" in guided_css
    assert ".gd-sess.local.active" in guided_css
    assert ".gd-sess.example.active" in guided_css
    assert ".gd-home-link" in guided_css
    assert ".gd-rail-utils" in guided_css
    assert ".gd-utilbtn.lang" in guided_css
    assert ".gd-data-workspace" in guided_css
    assert ".gd-draft-setup" in guided_css
    assert ".gd-folder-picker" in guided_css
    assert ".gd-folder-menu" in guided_css
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
    assert "api.js?v=20260627-idea-plan" in index_html
    assert "screens-guided-projects.js?v=20260626-guided-projects-split" in index_html
    assert "screens-guided-idea-provider.js?v=20260626-guided-api-first" in index_html
    assert "screens-guided.js?v=20260627-guided-idea-plan" in index_html
    assert "guided.css?v=20260627-guided-source-tabs" in index_html
    assert '<span class="gd-name">Guided Copilot</span>' in guided_js
    assert "Guided Copilot · local first · nothing leaves your machine" in guided_js
    assert "[t('Review Data', '审阅已有数据'), '@guidedGoal:review_data']" in guided_js


def test_native_agent_run_controls_are_reconnectable_and_cancelable() -> None:
    agent_js = _static_js("screens-agent.js")
    api_js = _static_js("api.js")

    assert "DeepSeek-compatible" in agent_js
    assert "Custom / local OpenAI-compatible" in agent_js
    assert "Custom/local endpoints must be OpenAI-compatible" in agent_js
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
    assert "Restart from active export" in agent_js
    assert "safe continuation is to restart from the active export" in agent_js


def test_native_patient_source_radios_are_real_controls() -> None:
    viz_js = _static_js("screens-viz.js")
    api_js = _static_js("api.js")
    i18n_js = _static_js("i18n.js")
    pages_css = _static_css("pages.css")
    index_html = _static_html("index.html")

    assert 'data-datamode="real"' in viz_js
    assert 'data-datamode="demo"' in viz_js
    assert "Previously exported data" in viz_js
    assert "Demo data" in viz_js
    assert "loadPatientReviewSources" in api_js
    assert "/api/patient-review/sources" in api_js
    assert "loadPatientSources" in viz_js
    assert "Ready to load local export" in viz_js
    assert "No registered export is active" in viz_js
    assert "data-patient-export" in viz_js
    assert "bounded_patient_review_drilldown" in viz_js
    assert "data-pt-table-module" in viz_js
    assert "data-pt-page-prev" in viz_js
    assert "data-pt-page-next" in viz_js
    assert "data-pt-page-size" in viz_js
    assert "table_page" in viz_js
    assert "table_page_size" in viz_js
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
    assert "Time-window × feature matrices" in viz_js
    assert "时间窗口 × 特征矩阵" in viz_js
    assert "Rows are time windows; columns are selected features." in viz_js
    assert "行是时间窗口；列是已选特征。" in viz_js
    # Patient Overview restores the legacy clinical-category dashboard (category
    # cards with value/delta/threshold tone + per-concept trend subplots), so the
    # module "signal atlas" layout is intentionally replaced.
    assert "data-patient-category-review" in viz_js
    assert "patientCategoryReview" in viz_js
    assert "function patientConceptChart(" in viz_js
    assert "function patientCategoryCard(" in viz_js
    assert "Patient category dashboard" in viz_js
    assert "患者分类看板" in viz_js
    assert "patientOverviewAtlas" not in viz_js
    assert "data-patient-category=" in viz_js
    assert "pcs-thr-line" in viz_js
    assert "data-patient-overview-module-ledger" in viz_js
    assert "data-patient-overview-module-card" in viz_js
    assert "Module map" in viz_js
    assert "模块图谱" in viz_js
    assert "Export module overview" in viz_js
    assert "导出模块总览" in viz_js
    assert "Selected entity trend tiles" not in viz_js
    assert "Table preview" in viz_js
    assert "表格预览" in viz_js
    assert "table_previews" in viz_js
    assert "pseudonymous entity tokens" in viz_js
    assert ".patient-table-scroll" in pages_css
    assert ".patient-preview-table" in pages_css
    assert ".patient-table-pager" in pages_css
    assert "css/pages.css?v=20260626-patient-table-pagination" in index_html
    assert "js/screens-viz.js?v=20260627-survival-outcome-summary" in index_html
    assert "browser review', '浏览器审阅" in viz_js
    assert "function buildDemoPatientDrilldown" in viz_js
    assert "payload_scope: 'catalog_shaped_seeded_demo_no_real_patient_rows'" in viz_js
    assert "Catalog-shaped demo review workspace ready" in viz_js
    assert "48 seeded entities" in viz_js
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
    assert "聚合载荷已就绪；打开 Agent 做证据绑定草稿核验。" in viz_js
    assert "Draft gate" not in viz_js
    assert "Evidence checks" not in viz_js
    assert "locked · needs reviewer sign-off" not in viz_js


def test_native_cohort_comparison_radios_are_stateful_controls() -> None:
    viz_js = _static_js("screens-viz.js")
    cohort_css = _static_css("cohort.css")
    redesign_css = _static_css("redesign.css")
    index_html = _static_html("index.html")

    assert "css/cohort.css?v=20260627-survival-outcome-summary" in index_html
    assert "js/screens-viz.js?v=20260627-survival-outcome-summary" in index_html
    assert "let cohortView = 'idle';" in viz_js
    assert "let cohortFeatureScope = 'recommended';" in viz_js
    assert 'data-cohort-config-required="true"' in viz_js
    assert "Cohort Statistics no longer opens with preloaded seeded results" in viz_js
    assert "Run demo cohort review" in viz_js
    assert "data-cohort-use-real" in viz_js
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
    assert "derived from hospital death + LOS" in viz_js
    assert ".surv-outcome-card" in cohort_css
    assert "ICU mortality is unavailable because this export does not include ICU-specific event and time columns." in viz_js
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
    assert "cohortSofaMatrixMode" in viz_js
    assert "data-cohort-sofa-matrix-mode" in viz_js
    assert "SOFA_MATRIX_GRANULARITIES" in viz_js
    assert "cohortSofaMatrixGranularity = 'medium'" in viz_js
    assert "data-cohort-sofa-granularity" in viz_js
    assert "exact_score_matrix" in viz_js
    assert "--sofa-min-width" in viz_js
    assert (
        "Rows are SOFA-1 score bands; columns are SOFA-2 score bands."
        in viz_js
    )
    assert "Rows are SOFA-1 severity bands; columns are SOFA-2 bands." in viz_js
    assert "reclass.status === 'ready'" in viz_js
    assert "Demo threshold uses SOFA ≥ 6" in viz_js
    assert "Age Groups' overview" not in viz_js
    assert ".surv-toolbar" in cohort_css
    assert ".km-chart" in cohort_css
    assert ".risk-table" in cohort_css
    assert ".sofa-heatmap" in cohort_css
    assert ".sofa-heat-cell" in cohort_css
    assert ".sofa-matrix-toggle" in cohort_css
    assert ".sofa-matrix-controls" in cohort_css
    assert "--sofa-cell-min" in cohort_css
    assert "--sofa-min-width" in cohort_css
    assert ".surv-toolbar" not in redesign_css
    assert ".km-chart" not in redesign_css
    assert ".sofa-heatmap" not in redesign_css
    assert ".sofa-matrix-controls" not in redesign_css
    for key in ["outcome", "age", "sex", "los", "sepsis", "custom"]:
        assert f"{key}:" in viz_js


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
    assert "css/home.css?v=20260626-home-owner" in index_html

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
    """The demo/fixture data layer + catalog accessors are owned by
    screens-viz-demo.js, not inlined in the screens-viz.js monolith
    (first owner-file carve-out, 2026-06-26). Main file rebinds them."""
    viz_js = _static_js("screens-viz.js")
    demo_js = _static_js("screens-viz-demo.js")
    index_html = _static_html("index.html")

    # demo generators + catalog accessors are DEFINED in the demo file
    assert "function demoCatalogModules(" in demo_js
    assert "function demoRowsForModule(" in demo_js
    assert "function demoCategorySection(" in demo_js
    assert "function catalogModuleLabel(" in demo_js
    assert "function catalogFeatureMeta(" in demo_js
    assert "window.VIZ_DEMO = {" in demo_js

    # they are NOT re-defined in the main file (no duplicate definitions)
    assert "function demoCatalogModules(" not in viz_js
    assert "function catalogModuleLabel(" not in viz_js
    assert "const DEMO_ENTITY_COUNT" not in viz_js

    # main file rebinds the exports so call sites stay unchanged
    assert "} = window.VIZ_DEMO;" in viz_js
    assert "demoCatalogModules" in viz_js  # still called

    # demo file loads BEFORE the main file in index.html
    demo_pos = index_html.find("screens-viz-demo.js")
    main_pos = index_html.find("screens-viz.js?")
    assert demo_pos != -1 and main_pos != -1
    assert demo_pos < main_pos, "screens-viz-demo.js must load before screens-viz.js"
