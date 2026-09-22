"""Reference-UI alignment contracts for the Guided Copilot workspace.

Moved out of test_pi_copilot_static.py, which is past its large-module
baseline: the conversation trace, follow-ups, composer menus and dismissal,
effort level, project switcher, run record, project notes, skin, data-source
card and demo follow-through. Same fixtures, same assertions.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    NODE_APP as NODE_APP,
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


def test_workbench_tab_and_reader_stay_open_together_with_global_navigation() -> None:
    """A figure and the code that produced it read side by side; opening one
    never closes the other, and the global rail survives on laptop widths."""
    source = _read("js/screens-guided-pi-source-view.js")
    preview = _read("js/screens-guided-pi-preview.js")
    owner = _read("js/screens-guided-pi.js")
    source_css = _read("css/guided-pi-source-view.css")
    desktop_css = _read("css/guided-pi-desktop.css")
    # Mutual close is gone from both openers; only the session owner closes the workbench.
    assert "if (preview && preview.close) preview.close();" not in source
    assert "if (sourceView && sourceView.close) sourceView.close();" not in preview
    assert "if (sourceView && sourceView.close) sourceView.close();" in owner
    # Laptop widths keep a compact 59px global rail instead of hiding navigation.
    assert ".gd-main.gpi-workspace.gpi-source-open>.gd-rail{display:flex;grid-column:1;grid-row:1;box-sizing:border-box;width:59px" in source_css
    assert ".gd-main.gpi-workspace.gpi-source-open>.gd-rail>:not(.gpi-global-nav){display:none}" in source_css
    assert ".gd-main.gpi-source-open>.gd-rail,.gd-main.gpi-source-open>.gd-rail-restore{display:none}" not in source_css
    # Combined state: workbench center, reader right, shelf yields, conversation yields below 1600px.
    assert ".gd-main.gpi-workspace.gpi-preview-open.gpi-source-open>#gdContextAside>#gdStudyAside{display:none}" in desktop_css
    assert ".gd-main.gpi-workspace.gpi-preview-open.gpi-source-open>#gdContextAside>#gdSourceAside{display:flex;grid-row:1" in desktop_css
    assert ".gd-main.gpi-workspace.gpi-preview-open.gpi-source-open>#gdContextAside>#gdPreviewAside{display:flex;grid-row:1" in desktop_css
    assert ".gd-main.gpi-workspace.gpi-preview-open.gpi-source-open>.gd-conv{display:none}" in desktop_css
    assert ".gd-main.gpi-workspace.gpi-preview-open.gpi-source-open:not(.gpi-preview-focus)>.gd-conv{display:flex;grid-column:2;grid-row:1}" in desktop_css


def test_composer_row_and_compact_rail_heading_never_overlap() -> None:
    workspace_css = _read("css/guided-pi-workspace.css")
    desktop_css = _read("css/guided-pi-desktop.css")
    activity = _read("js/screens-guided-pi-activity.js")
    assert ".gpi-actions{gap:8px;flex-wrap:wrap}" in workspace_css
    assert ".gpi-action-leading .gpi-access-menu>summary{white-space:nowrap}" in workspace_css
    assert "margin-left:auto}" in workspace_css.split(".gpi-action-trailing{", 1)[1].split("\n", 1)[0]
    assert ".gpi-conversations-heading [data-gpi-rail-new] span{display:none}" in workspace_css
    assert ".gpi-conversations-heading-actions [data-gpi-rail-new] span{display:inline}" in desktop_css
    # Results keep the shelf's working space; the other sections size to content.
    assert '.gpi-aside-section[data-gpi-aside-section="results"]{flex:1 1 40%;min-height:220px}' in desktop_css
    assert '.gpi-aside-section[data-gpi-aside-section="notes"]{flex:0 1 auto;max-height:22%}' in desktop_css
    # Step count and total elapsed time are visible on the trace row.
    assert '<span class="gpi-activity-meta" aria-hidden="true">${esc(traceMeta)}</span>' in activity
    assert ".gpi-activity.complete>summary .gpi-activity-meta," in desktop_css


def test_entry_search_covers_the_whole_catalogue_and_collapsed_rail_keeps_navigation() -> None:
    starters = _read("js/screens-guided-pi-starters.js")
    guided = _read("js/screens-guided.js")
    projects = _read("js/screens-guided-projects.js")
    desktop_css = _read("css/guided-pi-desktop.css")
    settings_css = _read("css/settings.css")
    workspace = _read("js/screens-guided-pi-study-workspace.js")
    # Entry search re-renders matches from the full reviewed catalogue, capped.
    assert "function filter(host, value, tr)" in starters
    assert "methodCards(translate).filter(row => cardSearchText(row).includes(query)).slice(0, SEARCH_LIMIT)" in starters
    assert "row.hidden = Boolean(query && !String(row.dataset.gpiStarterSearchText" not in starters
    # Manual collapse compacts the rail to the global nav; Projects restores it.
    assert ".gd-main.gpi-workspace.gd-project-rail-collapsed>.gd-rail{display:flex;grid-column:1;grid-row:1;box-sizing:border-box;width:59px" in desktop_css
    assert ".gd-main.gpi-workspace.gd-project-rail-collapsed>.gd-rail-restore{display:none}" in desktop_css
    assert ".gd-main.threecol.gpi-workspace.gd-project-rail-collapsed{grid-template-columns:59px minmax(0,1fr) var(--gd-context-aside-width)}" in desktop_css
    # The nav's Projects button belongs to the project rail owner; the guided
    # screen only delegates to it.
    assert "const collapsedMain = document.querySelector('.gd-main.gd-project-rail-collapsed');" in projects
    assert "collapsedMain.classList.remove('gd-project-rail-collapsed');" in projects
    assert "    showProjects,\n" in projects
    assert "if (projectOwner && projectOwner.showProjects) projectOwner.showProjects();" in guided
    assert "collapsedMain" not in guided
    # Settings anchors can all reach the top; the reference chip drops its instruction.
    assert ".settings-page{padding-bottom:min(60vh,640px)}" in settings_css
    assert "在下方填写你的问题" not in workspace


def test_home_reuses_empty_drafts_and_project_switcher_stays_a_switcher() -> None:
    owner = _read("js/screens-guided-pi.js")
    events = _read("js/screens-guided-pi-events.js")
    workspace = _read("js/screens-guided-pi-study-workspace.js")
    guided = _read("js/screens-guided.js")
    desktop_css = _read("css/guided-pi-desktop.css")
    start_entry = owner.split("async function startEntry()", 1)[1].split("async function stopMessage()", 1)[0]
    # Home and 新会话 open an existing empty draft before creating another session.
    assert "const emptyDraft = state.sessions.find(row => STUDY_WORKSPACE.isEmptyConversation(row));" in start_entry
    assert "if (emptyDraft) { await openSession(emptyDraft.session_id); return; }" in start_entry
    assert "isEmptyConversation };" in workspace
    assert "if (event.target.closest('[data-gpi-new]')) {\n          clearSessionSelection();\n          render();\n          void startEntry();" in events
    # The nav Projects button toggles the in-rail switcher, whose list is capped.
    assert "    picker.open = !picker.open;\n" in _read("js/screens-guided-projects.js")
    assert ".gd-main.gpi-workspace .gd-project-picker[open]>.gd-project-picker-list{position:absolute;z-index:60;" in desktop_css
    assert "max-height:min(60vh,560px)" in desktop_css


def test_skill_detail_is_addressable_and_errors_read_as_sentences() -> None:
    app = _read("js/app.js")
    skills = _read("js/screens-skills.js")
    model_menu = _read("js/screens-guided-pi-model-menu.js")
    run_outcome = _read("js/screens-guided-pi-run-outcome.js")
    session_view = _read("js/screens-guided-pi-session-view.js")
    events = _read("js/screens-guided-pi-events.js")
    pi_css = _read("css/guided-pi.css")
    # #skills/<id> resolves to the skills screen; the owner syncs selection with the hash.
    assert "return slash > 0 && window.SCREENS && window.SCREENS[raw.slice(0, slash)] ? raw.slice(0, slash) : raw;" in app
    assert "function skillIdFromHash()" in skills
    assert "history.pushState(null, '', next)" in skills
    assert "window.addEventListener('popstate', syncSelectionFromHash);" in skills
    assert "afterRender(root) { bind(root); void hydrateIfStale(); syncSelectionFromHash(); }" in skills
    # Transport failures never surface as a bare HTTP reason phrase.
    assert "function friendlyError(error)" in model_menu
    assert "state.error = friendlyError(error);" in model_menu
    assert "无法打开本次运行的数据可视化" in run_outcome
    # The composer error banner can be dismissed.
    assert 'data-gpi-dismiss-error' in session_view
    assert "if (event.target.closest('[data-gpi-dismiss-error]')) { state.error = ''; render(); return; }" in events
    assert ".gpi-error-close{" in pi_css


def test_planning_progress_and_run_failures_read_as_progress_and_reasons() -> None:
    """A ten-minute plan run must not show one static label and a bare code.

    The child-job owner tallies what the progressive planner has validated,
    the activity card shows that tally as its live subtitle, the current
    to-do row shows the stage sentence instead of "planning · 1/4" (a
    validation attempt count), and a failed run states what stopped it.
    """
    childjob = _read("js/screens-guided-pi-childjob.js")
    activity = _read("js/screens-guided-pi-activity.js")
    aside = _read("js/screens-guided-pi-aside.js")
    error_text = _read("js/screens-guided-pi-error-text.js")
    confirmation = _read("js/screens-guided-pi-confirmation.js")
    shell = _read("js/screens-guided-pi.js")
    pi_css = _read("css/guided-pi.css")

    assert "function notePlanningProgress(activity, event)" in childjob
    assert "progress.validatedSteps += 1" in childjob
    assert "activity.runningNote = progressText" in childjob
    assert "function planningProgressText(progress)" in activity
    assert "String(row.runningNote || '').trim() ||" in activity
    assert "planningProgressText, reasoningHtml, render" in activity
    # To-do row: stage sentence + failure explanation, never the raw step/code.
    assert "host.progressLabel(progress)" in aside
    assert "host.runFailureText(failureCode" in aside
    assert "progress.total ? ` · ${Number(progress.current || 0)}/${Number(progress.total)}`" not in aside
    assert "progressLabel: event => ACTIVITY.pipelineEventLabel(event)" in shell
    # Failure codes become sentences with a next step.
    assert "function runFailureText(code" in error_text
    for code in (
        "research_pipeline_planning_identity_unavailable",
        "research_pipeline_progressive_compile_failed",
        "research_pipeline_required_concept_structurally_unavailable",
    ):
        assert f"{code}: tr(" in error_text
    assert "reason: failureReason" in confirmation
    assert 'class="gpi-confirmation-reason"' in confirmation
    assert ".gpi-confirmation-reason{" in pi_css


def test_streamed_reply_grows_in_place_instead_of_rebuilding_the_conversation() -> None:
    """Each token used to re-render the whole host; long transcripts stuttered.

    The streaming assistant row is marked, `text_delta` patches that row's
    text node and keeps the log pinned to the bottom, and a caret marks the
    growing reply. Every other event still goes through the full render.
    """
    shell = _read("js/screens-guided-pi.js")
    session_view = _read("js/screens-guided-pi-session-view.js")
    pi_css = _read("css/guided-pi.css")
    activity = _read("js/screens-guided-pi-activity.js")

    live_stream = _read("js/screens-guided-pi-live-stream.js")
    assert "function patchStreamingMessage(row)" in live_stream
    assert "if (patchStreamingMessage(streamingRow)) return;" in live_stream
    assert "const handlePiEvent = LIVE_STREAM.handlePiEvent;" in shell
    index = _read("index.html")
    assert index.index("screens-guided-pi-host-jobs.js") < index.index("screens-guided-pi-live-stream.js")
    assert index.index("screens-guided-pi-live-stream.js") < index.index("js/screens-guided-pi.js?v=")
    assert "text.innerHTML = assistantTextHtml(FOLLOW_UPS ? FOLLOW_UPS.split(visible).text : visible);" in live_stream
    assert "data-gpi-streaming-message" in session_view
    assert "row.complete === false && !messageActions.editorHtml" in session_view
    assert ".gpi-message.is-streaming .gpi-message-body>.gpi-text>:last-child::after{" in pi_css
    assert "@media (prefers-reduced-motion:reduce){.gpi-message.is-streaming" in pi_css
    # Planning facts are recovered from the job label when typed fields are absent.
    assert "function planningEventFacts(event)" in activity
    assert "message.includes('executable plan step')" in activity
    assert "running && stage === 'plan' && step.status === 'complete'" in activity


def test_refused_plan_configuration_becomes_a_decision_card_not_a_banner() -> None:
    """After a 10-minute plan run the host compiler may refuse the automatic
    execution configuration (all-stay analysis without patient grouping).

    That refusal is recorded on host state, the confirmation card shows the
    plan preview, the localized reason and the review materials, the cohort
    owner offers the server-issued admission rules next to it, and a passive
    page open shows an explicit apply-settings card instead of nothing.
    """
    shell = _read("js/screens-guided-pi.js")
    plan_actions = _read("js/screens-guided-pi-plan-actions.js")
    confirmation = _read("js/screens-guided-pi-confirmation.js")
    cohort = _read("js/screens-guided-pi-cohort-eligibility.js")
    session_view = _read("js/screens-guided-pi-session-view.js")
    error_text = _read("js/screens-guided-pi-error-text.js")

    assert "planConfigurationError: ''," in shell
    assert "setPlanConfigurationError: value => { state.planConfigurationError = String(value || ''); }," in shell
    assert shell.count("planConfigurationError: () => state.planConfigurationError,") == 2
    assert "host.setPlanConfigurationError(String(error && error.code || 'agent_plan_configuration_failed'));" in plan_actions
    assert "if (confirmation.code === 'agent_plan_configuration_required') {" in plan_actions
    assert "await compileAgentPlanConfiguration();" in plan_actions
    assert "if (code === 'agent_plan_configuration_required' && !configurationError) return {" in confirmation
    assert "if (code === 'agent_plan_configuration_required' && configurationError) return {" in confirmation
    assert "|| confirmation.showPlanPreview" in confirmation
    assert '<details class="gpi-plan-conversation-summary" open>' in confirmation
    assert "const configurationRefused = actionCode === 'agent_plan_configuration_required'" in cohort
    assert "function continuationCardsHtml()" in session_view
    assert "if (refused && eligibility && confirmation) return confirmation + eligibility;" in session_view
    assert "agent_plan_patient_grouping_unavailable: tr(" in error_text


def test_reference_skin_is_route_scoped_and_loads_after_the_layout_owners() -> None:
    """The conversation route wears the reference visual language as one
    presentational owner loaded last; layout owners keep their measurements.
    """
    index = _read("index.html")
    skin = _read("css/guided-pi-skin.css")
    cohort = _read("js/screens-guided-pi-cohort-eligibility.js")

    assert index.index("css/guided-pi-desktop.css") < index.index("css/guided-pi-skin.css")
    assert index.index("css/guided-pi-skin.css") < index.index("<script")
    # Palette and controls are scoped to the route, never to :root globals.
    assert ".gd-main.gpi-workspace," in skin
    # One accent token (the reference's lime was declined); every tint
    # derives from it so the accent can be retuned in one place.
    assert "--skin-primary:#a3dbd6;" in skin and "--skin-primary-rgb:163,219,214;" in skin
    assert "229,233,109" not in skin and "#e5e96d;" not in skin
    module_skin = _read("css/module-shell-skin.css")
    assert index.index("css/guided-pi-skin.css") < index.index("css/module-shell-skin.css")
    assert index.index("css/skills-hub.css") < index.index("css/module-shell-skin.css")
    assert index.index("css/extraction.css") < index.index("css/module-shell-skin.css")
    assert ".euh-shell,.eusk-shell,.eudata-shell{" in module_skin
    assert "--sk-accent:var(--skin-primary)" in module_skin
    assert ".eusk-group h2{" in module_skin and "border-bottom:2px solid var(--skin-primary)" in module_skin
    assert ".eudata-modulenav button.active::after{" in module_skin
    assert ".gpi-" not in module_skin
    assert ".gpi-activity li,.gpi-activity-running li,.gpi-workspace .gpi-activity li{" in skin
    assert ".gpi-action-trailing .btn.primary{width:36px" in skin
    assert ".gpi-next-actions button{" in skin
    assert ".gpi-message.user .gpi-message-body{max-width:70%" in skin
    # The global rail keeps the owner's 59px measurement; only its colours change.
    assert "width:60px" not in skin and "padding-left:60px" not in skin
    assert "@media(min-width:1181px){\n .gd-main.gpi-workspace .gpi-global-nav{" in skin
    # A refused all-stay configuration recommends the first-admission rule.
    assert "const groupingUnavailable = configurationRefused" in cohort
    assert "firstAdmissionIds.has(String(option.id || ''))" in cohort


def test_demo_source_preparation_reports_back_and_offers_one_confirmation() -> None:
    """The prepare job finishes outside the turn; the host must say so.

    The host-jobs owner watches the submitted job, announces the registered
    export with a single "use it for this conversation" action that binds it
    to the study and confirms the session source through the existing
    authorization contract, and re-derives that state on reload. It never
    sends model text.
    """
    host_jobs = _read("js/screens-guided-pi-host-jobs.js")
    shell = _read("js/screens-guided-pi.js")
    session_view = _read("js/screens-guided-pi-session-view.js")
    events = _read("js/screens-guided-pi-events.js")
    index = _read("index.html")
    pi_css = _read("css/guided-pi.css")

    assert "window.EasyICU.guidedPi.declare('hostJobs', { create });" in host_jobs
    assert "const DEMO_PREP_CODE = 'easyicu_demo_source_preparation_submitted';" in host_jobs
    assert "api().loadJobSnapshot(id)" in host_jobs
    assert "job.kind === DEMO_PREP_JOB_KIND" in host_jobs
    assert "DATA_CONSENT.requiresConfirmation(host.session())" in host_jobs
    assert "reason: 'demo-source-binding'" in host_jobs
    assert "host.authorizeDataSource('use_study_required_data')" in host_jobs
    assert "host.authorizeDataSource('begin_local_selection')" in host_jobs
    assert "receipt_kind: 'data_source_binding'" in host_jobs
    assert "sendText" not in host_jobs and "sendMessage" not in host_jobs
    assert 'data-gpi-host-notice-action="use"' in host_jobs
    assert 'data-gpi-host-notice-action="other"' in host_jobs
    # Shell wiring: live tool result, reload/refresh sync, project reset.
    assert "const HOST_JOBS = MODULES.require('hostJobs').create({" in shell
    assert "HOST_JOBS.noteToolResult(event);" in _read("js/screens-guided-pi-live-stream.js")
    assert shell.count("HOST_JOBS.sync();") == 2
    assert "HOST_JOBS.stopAll();" in shell
    assert "workflowConfirmationHtml, hostJobs: HOST_JOBS," in shell
    assert "if (row.role === 'host_notice') return HOST_JOBS" in session_view
    assert "event.target.closest('[data-gpi-host-notice-action]')" in events
    assert "HOST_JOBS.handleAction(" in events
    assert index.index("screens-guided-pi-data-binding.js") < index.index("screens-guided-pi-host-jobs.js")
    assert index.index("screens-guided-pi-host-jobs.js") < index.index("js/screens-guided-pi.js?v=")
    assert ".gpi-host-notice-actions{" in pi_css


def test_a_question_that_names_an_official_demo_is_offered_that_demo() -> None:
    """「在 eICU demo 数据里…」 should not cost four model turns of 选哪个库.

    The opening question already names the data. While the conversation
    still needs a source, the host-jobs owner matches the researcher's own
    first message against the official demo catalog (the title's product
    token next to a demo word; exactly one match, never a bare database
    name) and the data-source card offers that demo as its primary action.
    The click is the source decision: the owner submits the allowlisted
    prepare job and, once the export is registered, binds and confirms it
    through the same path as the ready notice. No model text is sent.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    host_jobs = _read("js/screens-guided-pi-host-jobs.js")
    consent = _read("js/screens-guided-pi-data-consent.js")
    view = _read("js/screens-guided-pi-session-view.js")
    events = _read("js/screens-guided-pi-events.js")
    assert "return Object.freeze({ handleAction, namedDemo, noteToolResult, renderNotice, stopAll, sync, useNamedDemo, watchDemoSourceJob });" in host_jobs
    assert "const caller = api().startOfficialDemoSourcePrepare;" in host_jobs
    assert "namedDemo: HOST_JOBS && typeof HOST_JOBS.namedDemo === 'function' ? HOST_JOBS.namedDemo() : null" in view
    assert "if (namedDemo && HOST_JOBS && typeof HOST_JOBS.useNamedDemo === 'function') { void HOST_JOBS.useNamedDemo(namedDemo.dataset.gpiNamedDemo); return; }" in events
    assert events.index("[data-gpi-named-demo]") < events.index("const dataSourceAction = DATA_CONSENT && DATA_CONSENT.actionFromEvent(event);")
    script = f"""
      global.window = global;
      window.EU_LANG = 'zh';
      eval({_ESCAPE_OWNER!r});
      eval({host_jobs!r});
      eval({consent!r});
      const rows = [
        {{ id: 'mimic_iv_demo_v2_2', title: 'MIMIC-IV Clinical Database Demo', version: '2.2', status: {{ export_ready: true }} }},
        {{ id: 'eicu_demo_v2_0_1', title: 'eICU Collaborative Research Database Demo', version: '2.0.1', status: {{ export_ready: false }} }},
      ];
      let messages = [];
      let pending = true;
      const prepared = [];
      const host = {{
        tr: (en, zh) => zh,
        api: () => ({{
          loadOfficialDemoSources: async () => ({{ sources: rows }}),
          startOfficialDemoSourcePrepare: async id => {{ prepared.push(id); return {{ job_id: 'job-1' }}; }},
          loadJobSnapshot: async () => ({{ status: 'running' }}),
        }}),
        dataConsent: {{ requiresConfirmation: () => pending }},
        session: () => ({{ session_id: 's1' }}), messages: () => messages, render() {{}},
        busy: () => false, workflowReceipts: () => [], setWorkflowReceipts() {{}}, setError() {{}}, errorText: e => String(e),
      }};
      const owner = window.EasyICU.guidedPi.require('hostJobs').create(host);
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const ask = text => {{ messages = [{{ role: 'user', text }}]; const offer = owner.namedDemo(); return offer ? offer.id : null; }};
      const beforeCatalog = ask('在 eICU demo 数据里，评估最高乳酸与 ICU 死亡率');
      setTimeout(async () => {{
        const out = {{ beforeCatalog }};
        out.matches = {{
          eicuDemo: ask('在 eICU demo 数据里，评估最高乳酸与 ICU 死亡率'),
          mimicDemo: ask('用 MIMIC-IV Demo 看看乳酸分布'),
          chineseWord: ask('用mimic iv 演示数据做一个描述'),
          bareDatabase: ask('在 eICU 数据里评估'),
          twoDemos: ask('mimic-iv demo 和 eicu demo 对比'),
        }};
        ask('在 eICU demo 数据里，评估最高乳酸与 ICU 死亡率');
        const offer = owner.namedDemo();
        const session = {{ data_source_authorization: {{ status: 'pending' }} }};
        const ctx = {{ tr: (en, zh) => zh, esc: value => String(value), icon: () => '', namedDemo: offer }};
        const card = consent.render(session, ctx);
        out.card = {{
          names: card.includes('你的问题指定了这份数据') && card.includes('eICU Collaborative Research Database Demo v2.0.1'),
          primary: card.includes('data-gpi-named-demo="eicu_demo_v2_0_1"') && card.includes('准备并用于本次会话'),
          other: card.includes('data-gpi-data-source-action="begin_local_selection"'),
          plainWithoutOffer: consent.render(session, {{ ...ctx, namedDemo: null }}).includes('接下来，请为这个问题选择数据源'),
        }};
        out.wrongId = await owner.useNamedDemo('mimic_iv_demo_v2_2');
        out.started = await owner.useNamedDemo('eicu_demo_v2_0_1');
        out.prepared = prepared;
        out.pendingOffer = owner.namedDemo().pending;
        out.secondClickIgnored = await owner.useNamedDemo('eicu_demo_v2_0_1');
        pending = false;
        out.confirmedHidesOffer = owner.namedDemo() === null;
        owner.stopAll();
        process.stdout.write(JSON.stringify(out));
      }}, 10);
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == {
        "beforeCatalog": None,
        "matches": {
            "eicuDemo": "eicu_demo_v2_0_1",
            "mimicDemo": "mimic_iv_demo_v2_2",
            "chineseWord": "mimic_iv_demo_v2_2",
            "bareDatabase": None,
            "twoDemos": None,
        },
        "card": {"names": True, "primary": True, "other": True, "plainWithoutOffer": True},
        "wrongId": False,
        "started": True,
        "prepared": ["eicu_demo_v2_0_1"],
        "pendingOffer": True,
        "secondClickIgnored": False,
        "confirmedHidesOffer": True,
    }


def test_source_binding_projection_targets_the_current_extraction_markup() -> None:
    """The reader's confirm control must hang off markup the owner still renders.

    The extraction page header is `.eudata-head` + `.eudata-workspace`; the
    older `.page-head` / `.express` blocks are gone, so a projection keyed on
    them silently rendered the module wizard instead of the confirm step.
    """
    extraction = _read("js/screens-extraction.js")
    embedded = _read("js/screens-extraction-embedded.js")
    render = extraction.split("    render() {", 1)[1].split("    afterRender(root) {", 1)[0]
    source_binding = embedded.split("function projectSourceBinding(root)", 1)[1].split(
        "function paint()", 1
    )[0]
    paint = embedded.split("function paint()", 1)[1].split("function confirmSourceBinding", 1)[0]
    snapshot = extraction.split("function sourceBindingSnapshot()", 1)[1].split(
        "function bindSourceToCopilot()", 1
    )[0]

    assert '<header class="eudata-head">' in render
    assert '<section class="eudata-workspace">' in render
    assert "root.querySelector('.eudata-head')" in source_binding
    assert "root.querySelector('.eudata-workspace')" in source_binding
    assert ".page-head" not in source_binding and ".express" not in source_binding
    assert "data-gpi-source-binding-confirm" in source_binding
    assert "data-gpi-extraction-real" in source_binding
    assert ".handoff" in source_binding
    # The desktop module shell must not nest a second navigation in the reader.
    assert "unwrapModuleShell(host)" in paint
    assert paint.index("unwrapModuleShell(host)") < paint.index("projectSourceBinding(host)")
    assert "shell.querySelector('.eudata-module-content')" in embedded
    # A scan placeholder label never becomes the persisted study source label.
    assert "unknown|未知" in snapshot


def test_workflow_to_dos_read_as_a_checklist_with_the_current_decision() -> None:
    """The to-do panel is a task checklist, not a gate-vocabulary summary.

    Every stage is a row with a status mark; the current row states what is
    needed now in the decision card's own words (or the running task) and
    carries the one action; the stage reason is only the fallback.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-aside.js")
    shell = _read("js/screens-guided-pi.js")
    assert "gd-pipeline-checklist" in owner
    assert "gd-pipeline-disclosure" not in owner
    assert "const currentText = activeTask || decision || reasonText(current);" in owner
    assert "pendingDecisionTitle: () => {" in shell
    assert "activeTaskTitle: () => {" in shell
    script = f"""
      global.window = global;
      const head = {{ innerHTML: '' }};
      const body = {{ innerHTML: '', querySelector: () => null }};
      global.document = {{ getElementById(id) {{
        return id === 'gdStudyAside' ? {{ querySelector: () => head }} : id === 'gdAsideBody' ? body : null;
      }} }};
      eval({owner!r});
      let decision = '候选研究计划已生成，执行前还需要你确认一项设置';
      let task = '';
      let activeJob = {{ present: false }};
      let latestRun = {{ present: false }};
      const panel = window.EasyICU.guidedPi.require('aside').create({{
        tr: (_en, zh) => zh, esc: value => String(value == null ? '' : value), iconHtml: name => '[' + name + ']',
        projectId: () => 'p1', displayProjectTitle: value => String(value || ''), demoMode: () => false, shell: () => 'pi',
        project: () => ({{}}), workflow: () => ({{
          current_stage: 'plan', completed_required_stages: 2, required_stage_count: 7, active_job: activeJob,
          stages: [
            {{ id: 'idea', status: 'optional', required_for_completion: false }},
            {{ id: 'question', status: 'complete' }},
            {{ id: 'setup', status: 'complete' }},
            {{ id: 'plan', status: 'review_required', reason_code: 'plan_scientific_changes_required' }},
            {{ id: 'extraction', status: 'blocked' }},
          ],
        }}),
        hasPendingReview: () => true, pendingDecisionTitle: () => decision, activeTaskTitle: () => task,
        latestRun: () => latestRun, progressLabel: event => '候选计划 2/4 已通过校验 (' + event.step + ')',
        runFailureText: code => code === 'research_pipeline_plan_invalid' ? '分析计划未通过校验，需要重新生成。' : '',
      }});
      panel.syncProjectWorkflowAside();
      const withDecision = body.innerHTML;
      decision = ''; task = '正在生成研究计划';
      activeJob = {{ present: true, status: 'running', progress: [{{ step: 'planning', current: 1, total: 4 }}] }};
      panel.syncProjectWorkflowAside();
      const running = body.innerHTML;
      decision = ''; task = ''; activeJob = {{ present: false }};
      latestRun = {{ present: true, run_id: 'run_20260921T173133_b0b3a6', status: 'failed', gate_status: 'blocked', gate_reason_code: 'research_pipeline_plan_invalid' }};
      panel.syncProjectWorkflowAside();
      const fallback = body.innerHTML;
      process.stdout.write(JSON.stringify({{
        rows: (withDecision.match(/<li class="study-item /g) || []).length,
        currentRow: /study-item active current" aria-current="step"><span class="si-dot">\[dot\]<\/span><div class="si-txt"><div class="si-t">分析计划/.test(withDecision),
        decisionText: withDecision.includes('候选研究计划已生成，执行前还需要你确认一项设置'),
        gateTextHidden: !withDecision.includes('科学计划审阅要求'),
        action: withDecision.includes('data-gpi-aside-pending'),
        laterStageCaption: /研究数据准备<span class="gd-pipeline-next">后续阶段<\/span>/.test(withDecision),
        doneMark: /study-item done"><span class="si-dot">\[check\]/.test(withDecision),
        optionalMark: /study-item optional"><span class="si-dot">\[dot\]/.test(withDecision),
        count: withDecision.includes('<strong>2/7</strong> 个必需阶段已完成') && head.innerHTML !== undefined,
        headerCount: body.innerHTML.includes('<small class="gpi-aside-count">2/7</small>'),
        runningSpinner: running.includes('gpi-running-spinner') && running.includes('正在生成研究计划'),
        runningStage: running.includes('<div class="si-s">候选计划 2/4 已通过校验 (planning)</div>') && !withDecision.includes('si-s'),
        fallbackReason: fallback.includes('科学计划审阅要求先形成新的研究/计划版本') && !fallback.includes('gpi-running-spinner'),
        failureCause: fallback.includes('<div class="si-s gpi-run-failure"><code>run_20260921T173133_b0b3a6</code> 分析计划未通过校验，需要重新生成。</div>') && !running.includes('gpi-run-failure'),
        noComputePanel: !fallback.includes('data-gpi-aside-section="compute"') && !fallback.includes('gpi-run-status'),
      }}));
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == {
        "rows": 5,
        "currentRow": True,
        "decisionText": True,
        "gateTextHidden": True,
        "action": True,
        "laterStageCaption": True,
        "doneMark": True,
        "optionalMark": True,
        "count": True,
        "headerCount": True,
        "runningSpinner": True,
        "runningStage": True,
        "fallbackReason": True,
        "failureCause": True,
        "noComputePanel": True,
    }


def test_results_section_lists_the_project_run_record() -> None:
    """成果 shows the project's run record below the current run's files.

    The workflow payload now carries every bound run (`runs`); the aside
    renders them newest first as facts — type, outcome, time, file count,
    the failure cause as a sentence, the authoritative run marked 当前 —
    open by default only when the shelf itself is empty, and it keeps its
    disclosure state across re-renders.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-aside.js")
    workflow = (NODE_APP.parent / "workflow.py").read_text(encoding="utf-8")
    assert "runs: Sequence[Mapping[str, Any]] = ()" in workflow
    assert "def project_run_history(" in workflow
    assert "function runHistoryHtml(workflow, open)" in owner
    assert "if (state.workflow) state.workflow.runs = Array.isArray(payload && payload.runs) ? payload.runs : [];" in _read("js/screens-guided-pi.js")
    assert "(results || emptyResults) + runHistoryHtml(workflow, runHistoryOpen(body, !results))" in owner
    script = f"""
      global.window = global;
      const head = {{ innerHTML: '' }};
      let previousOpen = null;
      const body = {{ innerHTML: '', querySelector: selector => selector === '[data-gpi-run-history]' && previousOpen !== null ? {{ open: previousOpen }} : null }};
      global.document = {{ getElementById(id) {{
        return id === 'gdStudyAside' ? {{ querySelector: () => head }} : id === 'gdAsideBody' ? body : null;
      }} }};
      window.EU_GUIDED_CONTRACTS = {{ fmtRunTime: value => 'T(' + String(value).slice(5, 16) + ')' }};
      eval({owner!r});
      const runs = [
        {{ run_id: 'run_b', run_type: 'full', run_status: 'failed', gate_status: 'blocked', gate_reason_code: 'research_pipeline_planner_provider_unavailable', artifact_count: 1, updated_at: '2026-09-21T17:40:00Z', authoritative: false }},
        {{ run_id: 'run_a', run_type: 'full', run_status: 'human_review_pending', gate_status: 'blocked', plan_available: true, artifact_count: 12, updated_at: '2026-09-21T17:31:00Z', authoritative: true }},
        {{ run_id: 'run_0', run_type: 'full', run_status: 'pass', gate_status: 'pass', artifact_count: 20, updated_at: '2026-09-20T09:00:00Z', authoritative: false }},
      ];
      let results = '';
      const panel = window.EasyICU.guidedPi.require('aside').create({{
        tr: (_en, zh) => zh, esc: value => String(value == null ? '' : value), iconHtml: name => '[' + name + ']',
        projectId: () => 'p1', displayProjectTitle: value => String(value || ''), demoMode: () => false, shell: () => 'pi',
        project: () => ({{}}), workflow: () => ({{ current_stage: 'plan', completed_required_stages: 2, required_stage_count: 7, stages: [{{ id: 'plan', status: 'review_required' }}], runs }}),
        hasPendingReview: () => false, pendingDecisionTitle: () => '', activeTaskTitle: () => '',
        resultsHtml: () => results, latestRun: () => ({{ present: false }}),
        runFailureText: code => code === 'research_pipeline_planner_provider_unavailable' ? '模型服务不可用，规划未能完成。' : '',
      }});
      panel.syncProjectWorkflowAside();
      const emptyShelf = body.innerHTML;
      results = '<section class="gpi-study-results">files</section>'; previousOpen = null;
      panel.syncProjectWorkflowAside();
      const withShelf = body.innerHTML;
      previousOpen = true;
      panel.syncProjectWorkflowAside();
      const kept = body.innerHTML;
      const rowsOf = html => html.split('<li class="gpi-run-row').slice(1).map(part => part.split('"')[0]);
      process.stdout.write(JSON.stringify({{
        openWhenEmpty: emptyShelf.includes('<details class="gpi-run-history" data-gpi-run-history open>'),
        closedWithShelf: withShelf.includes('<details class="gpi-run-history" data-gpi-run-history>') && withShelf.indexOf('gpi-study-results') < withShelf.indexOf('gpi-run-history'),
        keptOpen: kept.includes('data-gpi-run-history open>'),
        count: emptyShelf.includes('运行记录<small class="gpi-aside-count">3</small>'),
        rows: rowsOf(emptyShelf),
        failedRow: emptyShelf.includes('<strong>完整分析 · 失败</strong><small>T(09-21T17:40) · 1 个文件</small><small class="gpi-run-cause">模型服务不可用，规划未能完成。</small><code>run_b</code>'),
        currentRow: emptyShelf.includes('<strong>完整分析 · 待审阅<span class="gpi-run-current">当前</span></strong><small>T(09-21T17:31) · 12 个文件</small><code>run_a</code>'),
        passedRow: emptyShelf.includes('<strong>完整分析 · 已通过</strong>'),
        noHistoryWithoutRuns: (() => {{ runs.length = 0; panel.syncProjectWorkflowAside(); return !body.innerHTML.includes('gpi-run-history'); }})(),
      }}));
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == {
        "openWhenEmpty": True,
        "closedWithShelf": True,
        "keptOpen": True,
        "count": True,
        "rows": [" is-failed", " is-review is-current", " is-done"],
        "failedRow": True,
        "currentRow": True,
        "passedRow": True,
        "noHistoryWithoutRuns": True,
    }


def test_project_notes_are_project_memory_in_the_project_folder() -> None:
    """笔记 is stored as project_notes.md in the project's local folder.

    The panel reads the folder through the guided project-notes route, saves
    typed text after a short pause, reports where it lives, keeps the
    browser copy only as the fallback for a project without a folder, and
    offers a pre-existing browser-only note into the folder once.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-aside.js")
    api = _read("js/api.js")
    guided = (NODE_APP.parent.parent / "guided_sessions.py").read_text(encoding="utf-8")
    routes = (NODE_APP.parent.parent / "routes" / "guided.py").read_text(encoding="utf-8")
    assert 'def read_project_notes(project_id: str)' in guided and 'def write_project_notes(project_id: str, text: Any)' in guided
    assert '_NOTES_FILE = "project_notes.md"' in guided
    assert '@router.get("/api/guided/projects/{project_id}/notes")' in routes
    assert '@router.post("/api/guided/projects/{project_id}/notes")' in routes
    assert "'/api/guided/projects/' + encodeURIComponent(projectId) + '/notes'" in api
    assert "按项目保存在此浏览器" not in owner
    script = f"""
      global.window = global;
      const stored = {{ 'easyicu.pi.projectNotes.v1.p-local': '浏览器里的旧笔记' }};
      global.localStorage = {{ getItem: k => stored[k] ?? null, setItem: (k, v) => {{ stored[k] = String(v); }}, removeItem: k => {{ delete stored[k]; }} }};
      const head = {{ innerHTML: '' }};
      const body = {{ innerHTML: '', querySelector: () => null }};
      global.document = {{ getElementById(id) {{
        return id === 'gdStudyAside' ? {{ querySelector: () => head }} : id === 'gdAsideBody' ? body : null;
      }} }};
      window.EU_GUIDED_CONTRACTS = {{ fmtRunTime: value => 'T(' + String(value).slice(5, 16) + ')' }};
      eval({owner!r});
      const saves = [];
      let currentProject = 'p-server';
      const api = () => ({{
        loadGuidedProjectNotes: id => Promise.resolve(id === 'p-server'
          ? {{ ok: true, available: true, present: true, text: '文件夹里的笔记', updated_at: '2026-09-22T01:00:00Z' }}
          : id === 'p-local' ? {{ ok: true, available: true, present: false, text: '', updated_at: null }}
          : {{ ok: true, available: false, present: false, text: '', updated_at: null }}),
        saveGuidedProjectNotes: (id, text) => {{ saves.push([id, text]); return Promise.resolve({{ ok: true, available: true, present: true, text, updated_at: '2026-09-22T01:05:00Z' }}); }},
      }});
      const panel = window.EasyICU.guidedPi.require('aside').create({{
        tr: (_en, zh) => zh, esc: value => String(value == null ? '' : value), iconHtml: name => '[' + name + ']', api,
        projectId: () => currentProject, displayProjectTitle: value => String(value || ''), demoMode: () => false, shell: () => 'pi',
        project: () => ({{}}), workflow: () => ({{ current_stage: 'plan', completed_required_stages: 1, required_stage_count: 7, stages: [{{ id: 'plan', status: 'ready' }}] }}),
        hasPendingReview: () => false, pendingDecisionTitle: () => '', activeTaskTitle: () => '',
      }});
      panel.syncProjectWorkflowAside();
      const loading = body.innerHTML;
      setTimeout(() => {{
        panel.syncProjectWorkflowAside();
        const loaded = body.innerHTML;
        body.oninput({{ target: {{ matches: s => s === '[data-gpi-project-notes]', value: '文件夹里的笔记，补一句' }} }});
        setTimeout(() => {{
          panel.syncProjectWorkflowAside();
          const saved = body.innerHTML;
          currentProject = 'p-local';
          panel.syncProjectWorkflowAside();
          setTimeout(() => {{
            const migrated = saves.some(([id, text]) => id === 'p-local' && text === '浏览器里的旧笔记');
            currentProject = 'p-none';
            panel.syncProjectWorkflowAside();
            setTimeout(() => {{
              panel.syncProjectWorkflowAside();
              const fallback = body.innerHTML;
              process.stdout.write(JSON.stringify({{
                loading: loading.includes('正在读取项目文件夹') && loading.includes(' readonly>'),
                loaded: loaded.includes('>文件夹里的笔记</textarea>') && loaded.includes('已保存到项目文件夹 · T(09-22T01:00)'),
                saved: saves[0] && saves[0][0] === 'p-server' && saves[0][1] === '文件夹里的笔记，补一句' && saved.includes('>文件夹里的笔记，补一句</textarea>'),
                localCleared: !('easyicu.pi.projectNotes.v1.p-server' in stored),
                migrated,
                fallback: fallback.includes('此项目没有本地文件夹，笔记仅保存在此浏览器'),
              }}));
            }}, 20);
          }}, 1000);
        }}, 1000);
      }}, 20);
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(completed.stdout) == {
        "loading": True,
        "loaded": True,
        "saved": True,
        "localCleared": True,
        "migrated": True,
        "fallback": True,
    }


def test_reference_controls_without_an_easyicu_owner_are_not_rendered() -> None:
    """Borrowed layout only where an EasyICU owner stands behind it.

    The reference's right shelf has a Compute panel (machines, background
    workers, jobs, pipelines on cloud machines), an "Auto" switch, and reply
    ratings. EasyICU executes locally with remote compute disabled, has no
    feedback ledger, and its approval control is the access-level menu — so
    those three controls are not rendered as empty shells. The aside offers
    exactly the panels its owners fill, and the run facts the Compute panel
    used to carry (stage sentence, failure cause) sit on the current to-do row.
    """
    aside = _read("js/screens-guided-pi-aside.js")
    header = _read("js/screens-guided-pi-header.js")
    session_view = _read("js/screens-guided-pi-session-view.js")
    events = _read("js/screens-guided-pi-events.js")
    actions = _read("js/screens-guided-pi-message-actions.js")
    css = "".join(_read(f"css/{name}") for name in (
        "guided-pi-workspace.css", "guided-pi-desktop.css", "guided-pi-skin.css", "workspace-canvas.css"))

    assert "const panelKeys = ['progress', 'results', 'notes'];" in aside
    assert "section('compute'" not in aside
    assert "function runFacts(workflow)" in aside
    assert "['compute', tr('Compute', '计算')]" not in header
    for marker in ("data-gpi-auto-toggle", "autoMode", "Auto advance"):
        assert marker not in session_view and marker not in events
    for marker in ("data-gpi-like", "data-gpi-dislike", "thumbup", "thumbdown"):
        assert marker not in actions and marker not in events
    assert "${copy}${edit}${retry}</div>" in actions
    for selector in (".gpi-run-status", ".gpi-switch", ".is-liked", 'data-gpi-aside-section="compute"'):
        assert selector not in css
    assert ".gd-pipeline-checklist .si-s{" in _read("css/guided-pi-skin.css")


def test_rail_reads_as_one_project_switcher_over_its_conversations() -> None:
    """A research project is the study; the rows under it are its conversations.

    The reference panel is "PROJECT / name ▾" (a combobox whose list is a
    popover) over a task list. EasyICU's rail used to print the project name
    twice — a brand line and, below the new-project button, a picker summary
    that expanded inline into a second list styled like the conversation
    rows — and called those rows "tasks", the word the running research job
    already uses. Now the picker is the heading's only control, its list
    opens as a popover, and the second level is named for what it is.
    """
    projects = _read("js/screens-guided-projects.js")
    workspace = _read("js/screens-guided-pi-study-workspace.js")
    shell = _read("js/screens-guided-pi.js")
    header = _read("js/screens-guided-pi-header.js")
    outcome = _read("js/screens-guided-pi-run-outcome.js")
    desktop_css = _read("css/guided-pi-desktop.css")
    skin = _read("css/guided-pi-skin.css")
    workspace_css = _read("css/guided-pi-workspace.css")

    heading_at = projects.index('<div class="gd-rail-heading">')
    assert heading_at < projects.index('<div class="gd-rail-list" id="gdSessions"></div>') < projects.index('<button class="gd-rail-collapse"')
    assert "gd-rail-brand" not in projects
    assert "brandName" not in projects
    assert "<small>${t('Project', '项目')}</small><span title=\"${esc(pickerTitle)}\">" in projects
    # Popover, not an inline block: conversations never move when it opens.
    assert ".gd-main.gpi-workspace .gd-rail-top{position:relative}" in desktop_css
    assert ".gd-main.gpi-workspace .gd-project-picker{position:static}" in desktop_css
    assert ".gd-main.gpi-workspace .gd-project-picker[open]>.gd-project-picker-list{position:absolute;z-index:60;" in desktop_css
    assert ".gd-main.gpi-workspace .gd-project-picker[open]>.gd-project-picker-list{border:1px solid var(--skin-border);border-radius:8px;" in skin
    # The one-line definition of a project is shown, not hidden once projects exist.
    assert ".gd-main.gpi-workspace .gd-rail-list:has(.gd-sess) .gd-project-summary span{display:block;" in skin
    assert ".gd-rail-brand" not in desktop_css and ".gd-rail-brand" not in workspace_css and ".gd-rail-brand" not in skin
    # Vocabulary: conversations under a project; "任务" is the running research job.
    assert "<strong>${tr('Conversations', '对话')}</strong>" in workspace
    assert "<span>${tr('Conversation', '对话')}</span></button>" in workspace
    assert "tr('Empty conversations', '空对话')" in workspace
    assert "tr('Remove empty conversation', '删除空对话')" in workspace
    for stale in ("tr('Tasks', '任务')", "tr('Task', '任务')", "tr('Empty tasks', '空任务')", "tr('Search tasks', '搜索任务')"):
        assert stale not in workspace
    assert "tr('New conversation', '新对话')" in shell and "tr('New conversation', '新对话')" in header
    assert "tr('Rename this conversation', '重命名对话')" in shell
    assert "新研究任务" not in shell and "新会话" not in header
    assert "tr('Ask in a new conversation', '在新对话中追问')" in outcome
    assert "tr('EasyICU research task is running', 'EasyICU 科研任务正在运行')" in shell


def test_reply_turn_reads_as_intro_traces_with_reasoning_and_answer() -> None:
    """One reply turn is laid out the way the reference research UI lays it out.

    The model's opening sentence sits above the traces, the traces list
    interleaves the model's interim narration after the tool call it follows,
    the model's reasoning summary is a row with its headline and full text,
    the streaming reasoning row exposes a patch target, and the answer with
    its actions stays last. A running turn groups the same way.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    activity_owner = _read("js/screens-guided-pi-activity.js")
    transcript_owner = _read("js/screens-guided-pi-transcript.js")
    script = f"""
      global.window = {{
        EU_LANG: 'zh',
        EU_GUIDED_PI_REPLAY: {{ lifecycleTurns: session => session.replayTurns || [] }},
      }};
      eval({_ESCAPE_OWNER!r});
      eval({activity_owner!r});
      eval({transcript_owner!r});
      const activity = window.EU_GUIDED_PI_ACTIVITY.create({{
        tr: (_en, zh) => zh,
        esc: value => String(value == null ? '' : value).replace(/</g, '&lt;'),
        iconHtml: name => '<i class="ic">' + name + '</i>',
        resourceName: () => '', resourceKey: () => '', resourceButton: () => '',
        publicText: value => String(value).replace(/pi_[a-z_]+/g, 'EasyICU 内部状态'),
      }});
      const trace = {{
        id: 'trace', role: 'activity', status: 'complete', startedAt: 1000, endedAt: 9000,
        steps: [
          {{ id: 'thinking-1', kind: 'thinking', status: 'complete', startedAt: 1000, endedAt: 2200,
             text: '**Listing eICU data sources**\\n\\nNeed the catalog before pi_session_bind.' }},
          {{ id: 'tool-1', kind: 'tool', toolName: 'easyicu_list_data_sources', status: 'complete', startedAt: 2200, endedAt: 2900 }},
          {{ id: 'tool-2', kind: 'tool', toolName: 'easyicu_update_study_context', status: 'complete', startedAt: 3000, endedAt: 3400 }},
        ],
      }};
      const rows = [
        {{ id: 'q', role: 'user', text: '问题' }},
        trace,
        {{ id: 'a1', role: 'assistant', text: '我先看数据源目录。', afterSteps: 0, complete: true }},
        {{ id: 'a2', role: 'assistant', text: '目录里有 demo，保存配置。', afterSteps: 1, complete: true }},
        {{ id: 'a3', role: 'assistant', text: '已保存，可以生成计划。', afterSteps: 2, complete: true }},
      ];
      const renderRow = (row, options) => row.role === 'activity'
        ? activity.render(row, options)
        : '<article data-row="' + row.id + '">' + row.text + '</article>';
      const renderSegment = (row, kind) => '<div data-segment="' + kind + '" data-row="' + row.id + '">' + row.text + '</div>';
      const grouped = activity.renderTimeline(rows, renderRow, renderSegment);
      const legacy = activity.renderTimeline(rows, renderRow);
      const running = activity.renderTimeline([
        rows[0], {{ ...trace, status: 'running', steps: trace.steps.slice(0, 2).concat([
          {{ id: 'thinking-2', kind: 'thinking', status: 'running', startedAt: Date.now() - 800, text: '**Saving the study setup**' }},
        ]) }},
        rows[2], {{ id: 'a2s', role: 'assistant', text: '正在写', afterSteps: 1, complete: false }},
      ], renderRow, renderSegment);
      const order = (html, markers) => markers.map(marker => html.indexOf(marker));
      const ascending = values => values.every((value, index) => value >= 0 && (index === 0 || value > values[index - 1]));
      const owner = window.EU_GUIDED_PI_TRANSCRIPT.create({{
        activity, upsertActivityStep: (target, step) => {{
          const at = target.steps.findIndex(row => row.id === step.id);
          if (at >= 0) target.steps[at] = {{ ...target.steps[at], ...step }}; else target.steps.push(step);
        }},
        timeMs: value => Date.parse(value || '2026-09-21T09:00:00Z'),
        resourceKey: resource => JSON.stringify(resource || null),
        modelErrorText: code => code, workflowActionCode: () => '',
      }});
      const persisted = owner.transcriptMessages({{ transcript: [
        {{ role: 'user', timestamp: '2026-09-21T09:00:00.000Z', content: [{{ type: 'text', text: '问题' }}] }},
        {{ role: 'assistant', timestamp: '2026-09-21T09:00:00.200Z', content: [
          {{ type: 'thinking', text: '**Listing eICU data sources**' }},
          {{ type: 'text', text: '我先看数据源目录。' }},
          {{ type: 'tool_call', tool_call_id: 'c1', tool_name: 'easyicu_list_data_sources' }},
        ] }},
        {{ role: 'tool', timestamp: '2026-09-21T09:00:06.300Z', content: [
          {{ type: 'tool_result', tool_call_id: 'c1', tool_name: 'easyicu_list_data_sources', summary: '已列出。' }} ] }},
        {{ role: 'assistant', timestamp: '2026-09-21T09:00:06.400Z', content: [
          {{ type: 'thinking', text: '**Saving the setup**' }},
          {{ type: 'text', text: '已保存，可以生成计划。' }},
        ] }},
      ], replayTurns: [{{ status: 'done', events: [
        {{ type: 'run_start', at: '2026-09-21T09:00:00.100Z' }},
        {{ type: 'assistant_start', at: '2026-09-21T09:00:00.200Z' }},
        {{ type: 'tool_start', at: '2026-09-21T09:00:06.200Z', tool_call_id: 'c1', tool_name: 'easyicu_list_data_sources' }},
        {{ type: 'message_end', at: '2026-09-21T09:00:06.250Z', stop_reason: 'toolUse' }},
        {{ type: 'tool_end', at: '2026-09-21T09:00:06.300Z', tool_call_id: 'c1', tool_name: 'easyicu_list_data_sources', code: 'ok' }},
        {{ type: 'assistant_start', at: '2026-09-21T09:00:06.400Z' }},
        {{ type: 'message_end', at: '2026-09-21T09:00:12.400Z', stop_reason: 'stop' }},
        {{ type: 'run_end', at: '2026-09-21T09:00:12.500Z' }},
      ] }}] }});
      const persistedActivity = persisted.find(row => row.role === 'activity');
      process.stdout.write(JSON.stringify({{
        groupedOrder: ascending(order(grouped, ['data-segment="intro" data-row="a1"', '<details class="gpi-activity complete"', 'kind-thinking', 'aria-label="Listing eICU data sources"', 'data-gpi-thinking-stream="thinking-1"', 'EasyICU 内部状态', 'kind-tool', 'gpi-activity-narration', 'data-row="a2"', 'aria-label="已保存研究配置"', 'data-row="a3"'])),
        narrationAfterFirstTool: grouped.indexOf('gpi-activity-narration') > grouped.indexOf('检查 EasyICU 数据源目录')
          && grouped.indexOf('gpi-activity-narration') < grouped.indexOf('保存研究配置'),
        introNotAnArticleRow: !grouped.includes('<article data-row="a1">') && !grouped.includes('<article data-row="a2">'),
        answerIsFullRow: grouped.includes('<article data-row="a3">'),
        reasoningBodyRendered: grouped.includes('<strong>Listing eICU data sources</strong><span class="gpi-activity-reasoning" data-gpi-thinking-stream="thinking-1"><p>Need the catalog before EasyICU 内部状态.</p></span>'),
        reasoningRowDuration: grouped.includes('1.2 秒'),
        legacyKeepsFullRows: legacy.includes('<article data-row="a1">') && legacy.includes('<article data-row="a2">') && !legacy.includes('gpi-activity-narration'),
        runningOrder: ascending(order(running, ['data-segment="intro" data-row="a1"', 'gpi-activity-running', 'kind-thinking', 'gpi-activity-reasoning is-streaming', 'data-gpi-thinking-stream="thinking-2"', 'gpi-activity-note', 'data-row="a2s"'])),
        runningRows: (running.match(/<li class="/g) || []).length,
        persistedThinking: persistedActivity.steps.filter(step => step.kind === 'thinking').map(step => step.text),
        persistedAfterSteps: persisted.filter(row => row.role === 'assistant').map(row => row.afterSteps),
        persistedOrder: persistedActivity.steps.filter(step => ['thinking', 'tool', 'submitted', 'settled'].includes(step.kind)).map(step => step.kind).join(','),
        persistedThinkingDurations: persistedActivity.steps.filter(step => step.kind === 'thinking').map(step => step.endedAt - step.startedAt),
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "groupedOrder": True,
        "narrationAfterFirstTool": True,
        "introNotAnArticleRow": True,
        "answerIsFullRow": True,
        "reasoningBodyRendered": True,
        "reasoningRowDuration": True,
        "legacyKeepsFullRows": True,
        "runningOrder": True,
        "runningRows": 3,
        "persistedThinking": ["**Listing eICU data sources**", "**Saving the setup**"],
        "persistedAfterSteps": [0, 1],
        "persistedOrder": "submitted,thinking,tool,thinking,settled",
        "persistedThinkingDurations": [6000, 6000],
    }
    shell = _read("js/screens-guided-pi.js")
    live_stream = _read("js/screens-guided-pi-live-stream.js")
    assert "streamingRow.afterSteps = activity.steps.filter(item => item.kind === 'tool').length" in live_stream
    assert "event.type === 'thinking_start' || event.type === 'thinking_delta' || event.type === 'thinking_end'" in live_stream
    assert "if (patchStreamingThinking(step)) return;" in live_stream
    assert "}, renderSegment);" in shell
    assert "return ACTIVITY.render(row, options && options.trace) + RUN_FILES.render(row);" in shell
    skin = _read("css/guided-pi-skin.css")
    assert ".gpi-workspace .gpi-activity-body.is-clipped>ol{max-height:300px;overflow-y:auto" in skin
    assert ".gpi-activity-reasoning{" in skin
    assert ".gpi-activity-narration-text{" in skin


def test_model_follow_up_questions_are_lifted_from_the_reply_and_offered_as_suggestions() -> None:
    """The model ends an answer with 2-3 follow-up questions; the host shows
    them as clickable suggestions under the latest reply, not as reply text."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-follow-ups.js")
    view = _read("js/screens-guided-pi-session-view.js")
    events = _read("js/screens-guided-pi-events.js")
    live_stream = _read("js/screens-guided-pi-live-stream.js")
    prompt = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")
    index = _read("index.html")
    assert "Follow-up questions rule:" in prompt and "'**可以继续问：**'" in prompt
    assert "Trace narration rule:" in prompt
    assert "nothing may follow the choices except the follow-up block described next" in prompt
    assert "FOLLOW_UPS.split(publicAssistantText(row.text))" in view
    assert "${messageActions.actionsHtml}\n          ${followUpsHtml}" in view
    assert "const modelFollowUp = event.target.closest('[data-gpi-model-followup]');" in events
    assert "if (question) sendText(question, []);" in events
    assert "FOLLOW_UPS ? FOLLOW_UPS.split(visible).text : visible" in live_stream
    assert index.index("screens-guided-pi-follow-ups.js") < index.index("js/screens-guided-pi.js?v=")
    script = f"""
      global.window = global;
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const owner = window.EasyICU.guidedPi.require('followUps');
      const zh = owner.split('主要分析是 logistic 回归。\\n\\n**下一步：**请在计划审查控件中确认。\\n\\n**可以继续问：**\\n- 这个计划的样本量够吗？\\n- 乳酸缺失会怎么处理？\\n- 能否加入 SOFA 作为协变量？\\n- 第四个会被截断\\n');
      const en = owner.split('Answer.\\n\\nFollow-up questions:\\n- What is the sample size?\\n- How are missing lactate values handled?');
      const none = owner.split('Answer with no block.\\n\\n**下一步：**\\n- 选项一\\n- 选项二');
      const html = owner.render(zh.questions, {{ tr: (_en, zhText) => zhText, iconHtml: name => '<i>' + name + '</i>' }});
      process.stdout.write(JSON.stringify({{
        zhText: zh.text, zhQuestions: zh.questions, enQuestions: en.questions, enText: en.text,
        untouched: none.text.endsWith('- 选项二') && none.questions.length === 0,
        buttons: (html.match(/data-gpi-model-followup="/g) || []).length,
        escaped: html.includes('data-gpi-model-followup="能否加入 SOFA 作为协变量？"') && html.includes('<span aria-hidden="true">↳</span>'),
        heading: html.includes('继续追问') && html.includes('<i>help</i>'),
        empty: owner.render([], {{}}) === '',
      }}));
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == {
        "zhText": "主要分析是 logistic 回归。\n\n**下一步：**请在计划审查控件中确认。",
        "zhQuestions": ["这个计划的样本量够吗？", "乳酸缺失会怎么处理？", "能否加入 SOFA 作为协变量？"],
        "enQuestions": ["What is the sample size?", "How are missing lactate values handled?"],
        "enText": "Answer.",
        "untouched": True,
        "buttons": 3,
        "escaped": True,
        "heading": True,
        "empty": True,
    }


def test_effort_level_is_a_per_conversation_menu_on_the_composer() -> None:
    """The composer offers the reference's effort control, backed by the runtime.

    One owner renders 低 / 中 / 高 beside the model chip with the
    conversation's own level marked — the three values every OpenAI-style
    endpoint accepts as an explicit `reasoning_effort`; "off" (no parameter
    sent) and "minimal" are shown as the current state when the model
    clamped to them, not offered — the host reads conversations from before
    the menu at the default, so neither is a legacy state. New conversations
    start at the remembered choice (medium by default); picking another
    level calls the host route and writes the effective level back onto the
    session. The popover carries the three rows and nothing else: the
    tooltip on the chip already explains the current level.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-effort-menu.js")
    view = _read("js/screens-guided-pi-session-view.js")
    events = _read("js/screens-guided-pi-events.js")
    shell = _read("js/screens-guided-pi.js")
    api = _read("js/api.js")
    index = _read("index.html")
    routes = (NODE_APP.parent.parent / "routes" / "pi_copilot.py").read_text(encoding="utf-8")
    service = (NODE_APP.parent / "service.py").read_text(encoding="utf-8")
    bridge = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")

    assert "window.EasyICU.guidedPi.declare('effortMenu', {" in owner
    assert "const DEFAULT_LEVEL = 'medium';" in owner
    assert "const LEVELS = ['low', 'medium', 'high'];" in owner
    assert "${EFFORT_MENU ? EFFORT_MENU.render({ iconHtml, level: session.thinking_level, disabled: interactionLocked || stale }) : ''}${HEADER.renderModelControl(headerOptions)}" in view
    assert "if (EFFORT_MENU && EFFORT_MENU.handleClick(event, {" in events
    assert "thinking_level: EFFORT_MENU.preferred(), external_llm_opt_in: true," in shell
    assert "thinking_level: 'off'" not in shell
    assert "/thinking-level', body || {});" in api
    assert index.index("screens-guided-pi-effort-menu.js") < index.index("js/screens-guided-pi.js?v=")
    # Host: the level is requested (default medium), changeable, and reopened
    # with the session's own value; nothing forces "off" any more.
    assert 'thinking_level: Literal["off", "minimal", "low", "medium", "high"] = "medium"' in routes
    assert '@router.post("/api/copilot/pi/sessions/{session_id}/thinking-level")' in routes
    assert 'resolved_thinking = "off"' not in service
    assert '"thinking_level": resolve_thinking_level(record.thinking_level),' in service
    assert 'DEFAULT_THINKING_LEVEL = "medium"' in service
    assert 'const thinkingLevel = "off";' not in bridge
    assert "const thinkingLevel = thinkingLevelParam(params.thinking_level);" in bridge
    assert 'case "session.set_thinking_level": {' in bridge
    assert "record.session.setThinkingLevel(requested);" in bridge
    script = f"""
      global.window = global;
      window.EU_LANG = 'zh';
      const stored = {{}};
      global.localStorage = {{ getItem: key => stored[key] || null, setItem: (key, value) => {{ stored[key] = String(value); }} }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const owner = window.EasyICU.guidedPi.require('effortMenu');
      const icon = name => '<i>' + name + '</i>';
      const html = owner.render({{ iconHtml: icon, level: 'medium', disabled: false }});
      const clamped = owner.render({{ iconHtml: icon, level: 'minimal', disabled: true }});
      const clampedOff = owner.render({{ iconHtml: icon, level: 'off', disabled: false }});
      const calls = [];
      let session = {{ session_id: 's1', thinking_level: 'medium' }};
      let rendered = 0; let error = '';
      const callbacks = {{
        session: () => session, busy: () => false, projectId: () => 'p1',
        api: () => ({{ setPiCopilotThinkingLevel: (id, body) => {{ calls.push([id, body]); return Promise.resolve({{ thinking_level: 'high', session: {{ session_id: 's1', thinking_level: 'high', title: 'T' }} }}); }} }}),
        render: () => {{ rendered += 1; }}, setSession: value => {{ session = value; }}, setError: value => {{ error = value; }},
      }};
      const menu = {{ removed: false, removeAttribute() {{ this.removed = true; }} }};
      const target = level => ({{ closest: selector => selector === '[data-gpi-effort-level]' ? {{ dataset: {{ gpiEffortLevel: level }}, closest: s => s === '[data-gpi-effort-menu]' ? menu : null }} : null }});
      const same = owner.handleClick({{ target: target('medium'), preventDefault() {{}} }}, callbacks);
      const changed = owner.handleClick({{ target: target('high'), preventDefault() {{}} }}, callbacks);
      const ignored = owner.handleClick({{ target: {{ closest: () => null }}, preventDefault() {{}} }}, callbacks);
      setTimeout(() => process.stdout.write(JSON.stringify({{
        rows: (html.match(/data-gpi-effort-level="/g) || []).length,
        noOffChoice: !html.includes('data-gpi-effort-level="off"'),
        pressed: html.includes('data-gpi-effort-level="medium" aria-pressed="true"') && !html.includes('data-gpi-effort-level="high" aria-pressed="true"'),
        summary: html.includes('<span>努力 · 中</span>') && html.includes('<i>spark</i>'),
        noFootnote: !html.includes('<p>') && html.includes('data-popover-menu'),
        clampedRow: clamped.includes('data-gpi-effort-level="minimal" aria-pressed="true"') && clamped.includes('aria-disabled="true"') && (clamped.match(/data-gpi-effort-level="/g) || []).length === 4,
        offState: clampedOff.includes('<span>努力 · 未指定</span>') && clampedOff.includes('data-gpi-effort-level="off" aria-pressed="true"') && clampedOff.includes('不向模型指定推理档位') && !clampedOff.includes('菜单出现前'),
        same, changed, ignored, calls, sessionLevel: session.thinking_level, sessionTitle: session.title,
        remembered: stored['easyicu.pi.thinkingLevel.v1'], preferred: owner.preferred(), rendered, error, menuClosed: menu.removed,
      }})), 0);
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == {
        "rows": 3,
        "noOffChoice": True,
        "pressed": True,
        "summary": True,
        "noFootnote": True,
        "clampedRow": True,
        "offState": True,
        "same": True,
        "changed": True,
        "ignored": False,
        "calls": [["s1", {"project_id": "p1", "thinking_level": "high"}]],
        "sessionLevel": "high",
        "sessionTitle": "T",
        "remembered": "high",
        "preferred": "high",
        "rendered": 1,
        "error": "",
        "menuClosed": True,
    }


def test_floating_menus_share_one_dismissal_owner() -> None:
    """Every floating menu closes the way a menu is expected to.

    Native <details> only toggles from its own summary, so the composer's
    + / access / effort / model menus, the header's layout and overflow
    menus, a conversation's ••• menu, the project switcher, and the skills
    page's menus each stayed open until their button was clicked again, and
    two could be open at once. One shell owner (`popover-menus.js`) now
    gives every `<details data-popover-menu>` the missing behaviour —
    opening one closes the others, a press or focus move outside closes it,
    Escape closes it and returns focus to its summary — instead of each
    route owner reconstructing part of it. Inline disclosures are not menus
    and never carry the attribute.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/popover-menus.js")
    index = _read("index.html")
    events = _read("js/screens-guided-pi-events.js")
    header = _read("js/screens-guided-pi-header.js")
    idea_source = _read("js/screens-guided-pi-idea-source.js")
    workspace = _read("js/screens-guided-pi-study-workspace.js")
    effort = _read("js/screens-guided-pi-effort-menu.js")
    projects = _read("js/screens-guided-projects.js")
    skills = _read("js/screens-skills.js")
    aside = _read("js/screens-guided-pi-aside.js")

    assert "window.EU_POPOVER_MENUS = Object.freeze({" in owner
    for consumer in ("js/screens-guided-projects.js", "js/screens-skills.js", "js/screens-guided-pi.js"):
        assert index.index("js/popover-menus.js?v=") < index.index(consumer)
    # Every floating menu is marked; the project switcher only while it is a
    # switcher (with no project chosen, or while managing, the list is the surface).
    floating = {
        header: ('<details class="gpi-model-control" data-popover-menu>', '<details class="gpi-layout-control" data-popover-menu>', '<details class="gpi-head-overflow" data-popover-menu>'),
        idea_source: ('<details class="gpi-idea-source-menu" data-popover-menu>',),
        workspace: ('<details class="gpi-access-menu" data-popover-menu>', '<details class="gpi-conversation-menu" data-popover-menu>'),
        effort: ('<details class="gpi-effort-menu" data-gpi-effort-menu data-popover-menu>',),
        projects: ("const pickerDismissible = !!activeId && !projectManagementActive;", "${pickerDismissible ? ' data-popover-menu' : ''}>"),
        skills: ('<details class="eusk-create-menu" data-popover-menu>', '<details class="eusk-actions-menu" data-popover-menu>'),
    }
    for source, markers in floating.items():
        for marker in markers:
            assert marker in source, marker
    for disclosure in ('<details class="gpi-idea-source-group"', '<details class="gpi-run-history"'):
        source = idea_source if "idea-source-group" in disclosure else aside
        assert disclosure in source and "data-popover-menu" not in source.split(disclosure, 1)[1].split(">", 1)[0]
    # A menu whose open state lives in its owner registers with the same
    # owner: the rail's folder menu, whose re-render used to drop focus to the
    # body so its shell-scoped Escape branch never ran.
    guided = _read("js/screens-guided.js")
    assert "if (window.EU_POPOVER_MENUS) window.EU_POPOVER_MENUS.register({" in guided
    assert "isOpen: () => guidedFolderMenuOpen, contains: node => Boolean(node && node.closest && node.closest('.gd-folder-picker'))," in guided
    assert "if (guidedFolderMenuOpen && !e.target.closest('.gd-folder-picker')) {" not in guided
    assert "        if (guidedFolderMenuOpen) {\n          guidedFolderMenuOpen = false;\n          renderGuidedFolderControls();\n          e.preventDefault();" not in guided
    assert index.index("js/popover-menus.js?v=") < index.index("js/screens-guided.js?v=")
    # The route owner no longer reconstructs outside-click or Escape policy.
    assert "!menu.contains(event.target)" not in events
    assert "!sourceMenu.contains(event.target)" not in events
    assert ".gpi-head-overflow[open]');\n          if (menu) {" not in events
    # The two popovers with footnotes lost them: rows only.
    assert "Access levels never reveal credentials" not in workspace
    assert "Applies to this conversation from the next reply" not in effort

    script = f"""
      const listeners = {{}};
      const menu = name => {{
        const summary = {{ focused: 0, focus() {{ this.focused += 1; }} }};
        const inner = {{ name: name + '-inner' }};
        const m = {{ name, open: false, summary, inner,
          matches: selector => selector === 'details[data-popover-menu]',
          contains: node => node === m || node === summary || node === inner,
          querySelector: selector => selector === ':scope > summary' ? summary : null }};
        return m;
      }};
      const menus = [menu('access'), menu('effort'), menu('picker')];
      const outside = {{ name: 'composer' }};
      global.document = {{
        addEventListener(type, fn, capture) {{ listeners[type] = {{ fn, capture: !!capture }}; }},
        querySelectorAll: selector => selector === 'details[data-popover-menu][open]' ? menus.filter(m => m.open) : [],
      }};
      global.window = global;
      eval({owner!r});
      const fire = (type, target, extra) => listeners[type].fn(Object.assign({{ target, key: '', defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }}, extra || {{}}));
      const openState = () => menus.map(m => m.name + ':' + m.open).join(' ');
      const out = {{ captured: ['toggle', 'pointerdown', 'focusin'].map(t => listeners[t].capture), keydownBubbles: listeners.keydown.capture === false }};
      menus[0].open = true; fire('toggle', menus[0]);
      out.oneOpen = openState();
      menus[1].open = true; fire('toggle', menus[1]);
      out.secondClosesFirst = openState();
      fire('pointerdown', menus[1].inner);
      out.insidePressKeeps = openState();
      fire('pointerdown', outside);
      out.outsidePressCloses = openState();
      menus[2].open = true; fire('toggle', menus[2]);
      fire('focusin', outside);
      out.focusOutCloses = openState();
      menus[1].open = true; fire('toggle', menus[1]);
      const escape = {{ target: outside, key: 'Escape', defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
      listeners.keydown.fn(escape);
      out.escapeCloses = openState();
      out.escapeFocusesSummary = menus[1].summary.focused;
      out.escapeConsumed = escape.defaultPrevented;
      menus[0].open = true;
      const handled = {{ target: outside, key: 'Escape', defaultPrevented: true, preventDefault() {{}} }};
      listeners.keydown.fn(handled);
      out.handledEscapeLeavesMenu = menus[0].open;
      const other = {{ target: outside, key: 'Enter', defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
      listeners.keydown.fn(other);
      out.otherKeyIgnored = menus[0].open && !other.defaultPrevented;
      out.closeAll = window.EU_POPOVER_MENUS.closeAll() === 1 && !menus[0].open;
      // A state-driven menu: open state held by its owner, not a <details>.
      const folder = {{ open: true, closes: [], inner: {{ name: 'folder-item' }} }};
      const unregister = window.EU_POPOVER_MENUS.register({{
        isOpen: () => folder.open, contains: node => node === folder.inner,
        close: opts => {{ folder.open = false; folder.closes.push(Boolean(opts && opts.focus)); }},
      }});
      fire('pointerdown', folder.inner);
      out.stateInsideKeeps = folder.open;
      fire('pointerdown', outside);
      out.stateOutsideCloses = !folder.open;
      folder.open = true;
      const stateEscape = {{ target: outside, key: 'Escape', defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
      listeners.keydown.fn(stateEscape);
      out.stateEscape = [!folder.open, stateEscape.defaultPrevented, folder.closes];
      unregister(); folder.open = true;
      fire('pointerdown', outside);
      out.unregisteredIgnored = folder.open;
      process.stdout.write(JSON.stringify(out));
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == {
        "captured": [True, True, True],
        "keydownBubbles": True,
        "oneOpen": "access:true effort:false picker:false",
        "secondClosesFirst": "access:false effort:true picker:false",
        "insidePressKeeps": "access:false effort:true picker:false",
        "outsidePressCloses": "access:false effort:false picker:false",
        "focusOutCloses": "access:false effort:false picker:false",
        "escapeCloses": "access:false effort:false picker:false",
        "escapeFocusesSummary": 1,
        "escapeConsumed": True,
        "handledEscapeLeavesMenu": True,
        "otherKeyIgnored": True,
        "closeAll": True,
        "stateInsideKeeps": True,
        "stateOutsideCloses": True,
        "stateEscape": [True, True, [False, True]],
        "unregisteredIgnored": True,
    }


def test_composer_menus_open_inside_the_conversation_panel() -> None:
    """Every composer menu lands where it can be seen, at laptop widths too.

    The model menu kept the header-era rule `top: calc(100% + 7px)` after it
    moved to the composer, so it opened below the composer — off the bottom
    of the window and clipped by the panel (the user: 「点击模型切换，窗口在
    下面，我都看不到」). At 1280×760 the three-column layout leaves the
    conversation panel 520px wide and the effort menu, right-aligned to its
    own chip, ran past the panel's left edge. Composer menus now anchor to
    the compose card: they open above it, inset from its edges (leading
    menus left, trailing menus right), a long model list scrolls inside the
    popover, and the entry composer — high on an empty page — opens them
    below the card with a shorter list on a short window. The rail's folder
    menu fills its button instead of overhanging the rail, the context
    strip's data-source disclosure is a popover menu, and the run-record
    summary has a real hit area.
    """

    pi_css = _read("css/guided-pi-composer-menus.css")
    index = _read("index.html")
    skin = _read("css/guided-pi-skin.css")
    projects_css = _read("css/guided-projects.css")
    consent = _read("js/screens-guided-pi-data-consent.js")

    # Placement is its own small owner, loaded right after the route's base sheet.
    assert pi_css.startswith("/* Owner: Guided Copilot composer menus")
    assert index.index("css/guided-pi.css?v=") < index.index("css/guided-pi-composer-menus.css?v=") < index.index("css/guided-pi-idea-source.css?v=")
    assert ".gpi-compose-card .gpi-actions,.gpi-compose-card .gpi-idea-source-menu,.gpi-compose-card .gpi-access-menu,.gpi-compose-card .gpi-effort-menu,.gpi-compose-card .gpi-model-control{position:static}" in pi_css
    assert ".gpi-compose-card .gpi-idea-source-popover,.gpi-compose-card .gpi-access-popover,.gpi-compose-card .gpi-model-popover{position:absolute;z-index:72;top:auto;bottom:calc(100% + 8px)}" in pi_css
    assert ".gpi-compose-card .gpi-idea-source-popover,.gpi-compose-card .gpi-access-menu>.gpi-access-popover{left:10px;right:auto}" in pi_css
    assert ".gpi-compose-card .gpi-effort-menu>.gpi-effort-popover,.gpi-compose-card .gpi-model-popover{left:auto;right:10px}" in pi_css
    assert ".gpi-compose-card .gpi-effort-menu>.gpi-effort-popover{width:min(320px,calc(100% - 20px))}" in pi_css
    assert ".gpi-compose-card .gpi-model-popover{width:min(236px,calc(100% - 20px));max-height:min(420px,calc(100vh - 140px));overflow:auto}" in pi_css
    assert ".gpi-entry-compose .gpi-compose-card .gpi-idea-source-popover,.gpi-entry-compose .gpi-compose-card .gpi-access-popover,.gpi-entry-compose .gpi-compose-card .gpi-model-popover{top:calc(100% + 8px);bottom:auto}" in pi_css
    assert "@media(max-height:760px){.gpi-entry-compose .gpi-compose-card .gpi-model-rows{max-height:150px}}" in pi_css
    # The skin no longer pins a fixed + menu width that could exceed a narrow card.
    assert ".gpi-compose-card .gpi-idea-source-popover{width:min(240px,calc(100% - 20px));" in skin
    assert "min-height:26px;color:var(--skin-muted);font:500 12px/18px var(--skin-font);list-style:none;cursor:pointer}" in skin
    assert "position:absolute;z-index:70;top:calc(100% + 7px);left:0;right:0;width:auto;" in projects_css
    assert '<details class="gpi-data-consent" data-popover-menu aria-label=' in consent
    # Phone: the trailing group takes its own line; the skin had overridden the
    # workspace phone cap on the model chip with max-width:none.
    assert "@media(max-width:520px){\n .gpi-compose-card .gpi-action-trailing{flex:1 1 100%;justify-content:flex-end}" in skin
    assert " .gpi-compose-card .gpi-model-binding{max-width:min(150px,40vw)}" in skin
    # Below 921px the rail scrolls, so the folder menu opens in place (the
    # workspace owner that makes the rail scroll carries the rule).
    workspace_css = _read("css/guided-pi-workspace.css")
    narrow = workspace_css[workspace_css.index("@media(max-width:920px){\n .gd-main.gpi-workspace .gd-rail{max-height:55vh;overflow:auto}"):]
    assert " .gd-main.gpi-workspace .gd-folder-menu{position:static;margin-top:7px;box-shadow:none}" in narrow[: narrow.index("\n}")]


def test_reply_markdown_renders_github_tables() -> None:
    """A reply's result summary is often a table; it must not show raw pipes."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-markdown.js")
    script = f"""
      global.window = global;
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const md = window.EasyICU.guidedPi.require('markdown');
      const html = md.render('有以下产物：\\n\\n| 产物 | 用途 |\\n|---|:---:|\\n| `agent_plan.json` | 计划，含 a \\\\| b |\\n| <b>x</b> | 短行 |\\n\\n之后的段落。');
      const plain = md.render('| 不是表格 |\\n普通行');
      process.stdout.write(JSON.stringify({{ html, plain }}));
    """
    completed = subprocess.run([node, "--eval", script], check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["html"] == (
        '<p>有以下产物：</p><table class="gpi-md-table"><thead><tr><th>产物</th><th>用途</th></tr></thead>'
        '<tbody><tr><td><code>agent_plan.json</code></td><td>计划，含 a | b</td></tr>'
        '<tr><td>&lt;b&gt;x&lt;/b&gt;</td><td>短行</td></tr></tbody></table><p>之后的段落。</p>'
    )
    assert "<table" not in payload["plain"]
    skin = _read("css/guided-pi-skin.css")
    assert ".gpi-text table{width:100%;border-collapse:collapse;border:1px solid var(--skin-border)" in skin


def test_copilot_plus_menu_keeps_article_url_entry_in_the_conversation() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-idea-source.js")
    script = f"""
      global.window = {{EU_API: {{}}}};
      eval({source!r});
      const owner = window.EU_GUIDED_PI_IDEA_SOURCE;
      const tr = (_en, zh) => zh;
      const html = owner.controls({{
        tr, esc: value => String(value), icon: name => `<i>${{name}}</i>`, disabled: false,
        extensions: {{ mcpEnabled: true, pubmedEnabled: true, zoteroEnabled: false,
          mcpServers: [{{ name: 'pubmed-mcp', tools: 2 }}], skills: [{{ name: 'sepsis-notes' }}] }},
      }});
      const bare = owner.controls({{ tr, esc: value => String(value), icon: name => `<i>${{name}}</i>`, disabled: false }});
      const masterOff = owner.controls({{ tr, esc: value => String(value), icon: name => `<i>${{name}}</i>`, disabled: false,
        extensions: {{ mcpEnabled: false, pubmedEnabled: true, zoteroEnabled: false, mcpServers: [], skills: [] }} }});
      const menu = {{removed: false, removeAttribute: () => {{ menu.removed = true; }}}};
      const input = {{
        value: '', focused: false, placeholder: '', selection: null,
        focus() {{ this.focused = true; }},
        setSelectionRange(start, end) {{ this.selection = [start, end]; }},
      }};
      const target = {{
        closest: selector => selector === '[data-gpi-idea-url-focus]'
          ? target : (selector === '.gpi-idea-source-menu' ? menu : null),
      }};
      const handled = owner.handleClick({{target}}, {{
        host: () => ({{querySelector: () => input}}), tr, render: () => {{}},
      }});
      console.log(JSON.stringify({{html, bare, masterOff, handled, menuRemoved: menu.removed, input}}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)

    assert 'aria-label="添加到问题"' in payload["html"]
    assert 'aria-haspopup="menu"' in payload["html"]
    assert 'role="menu"' in payload["html"]
    assert "上传 PDF" in payload["html"]
    assert "粘贴文章链接" in payload["html"]
    assert 'data-gpi-composer-picker="materials"' in payload["html"]
    assert 'data-gpi-composer-picker="skills"' in payload["html"]
    # The two drawers also open from the keyboard, as the reference's
    # "Type @" / "Type /" rows say; the hint is a kbd chip, not a chevron.
    assert '<span class="gpi-menu-hint">输入 <kbd>@</kbd></span>' in payload["html"]
    assert '<span class="gpi-menu-hint">输入 <kbd>/</kbd></span>' in payload["html"]
    # The two drawer rows carry kbd hints, not chevrons (the chevron now marks
    # only the nested Connectors group).
    assert payload["html"].split('data-gpi-composer-picker="skills"', 1)[1].split('</button>', 1)[0].count('›') == 0
    assert payload["html"].split('data-gpi-composer-picker="materials"', 1)[1].split('</button>', 1)[0].count('›') == 0
    events = _read("js/screens-guided-pi-events.js")
    assert "function composerPickerTrigger(value)" in events
    # The reference's "Connectors ▸ Manage / Not enabled" row, filled with what
    # EasyICU really freezes into a conversation: MCP servers with their tool
    # allowlists, user Skills, and the literature connector switches. Each row
    # opens the matching Settings capability tab; no row is rendered without data.
    assert 'data-gpi-extensions-group' in payload["html"]
    assert '连接器与工具' in payload["html"]
    assert 'data-gpi-manage-extensions="mcp"' in payload["html"] and 'pubmed-mcp · 2 个白名单工具' in payload["html"]
    assert 'data-gpi-manage-extensions="skills"' in payload["html"] and 'sepsis-notes' in payload["html"]
    assert 'data-gpi-manage-extensions="connectors"' in payload["html"]
    assert '已启用 · Idea Mining' in payload["html"] and '未启用 · Idea Mining' in payload["html"]
    assert '在新建对话时固化' in payload["html"]
    assert 'data-gpi-extensions-group' not in payload["bare"]
    assert '总开关已关' in payload["masterOff"] and '未安装' in payload["masterOff"]
    view = _read("js/screens-guided-pi-session-view.js")
    assert "function composerExtensions(session)" in view
    assert "extensions: composerExtensions(session)" in view
    assert "mcpEnabled: settings.mcp_tools_enabled === true," in view
    assert "const manageExtensions = event.target.closest('[data-gpi-manage-extensions]');" in events
    assert "window.sessionStorage.setItem('easyicu.settings.openCapabilityTab', manageExtensions.dataset.gpiManageExtensions || 'overview');" in events
    settings = _read("js/screens-settings.js")
    assert "if (requestedTab && capabilityTabs().some(([id]) => id === requestedTab)) {" in settings
    assert "return /(^|\\s)[@/]$/.test(text) ? text.slice(-1) : '';" in events
    assert "if (trigger === '@') STUDY_WORKSPACE.openMaterials(projectId(), state.session.session_id);" in events
    assert "else STUDY_WORKSPACE.openSkills(projectId(), state.session);" in events
    assert "event.target.value = event.target.value.slice(0, -1);" in events
    assert "!event.isComposing" in events
    assert payload["handled"] is True
    assert payload["menuRemoved"] is True
    assert payload["input"]["focused"] is True
    assert "粘贴文章链接" in payload["input"]["placeholder"]
