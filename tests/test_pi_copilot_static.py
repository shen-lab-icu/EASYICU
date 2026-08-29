"""Frontend ownership and wiring regressions for Guided Pi Copilot."""

from __future__ import annotations

import hashlib
import json
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


def test_node_prompt_obeys_owner_order_before_internal_resolution() -> None:
    prompt = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")

    assert "Workflow-order authority rule" in prompt
    assert "missing_setup_fields is ordered by the EasyICU owner" in prompt
    assert "Never offer a generic continue/继续对话 action" in prompt

# The screen modules destructure `esc` from window.EU_HTML at the top of their
# IIFE, so these Node harnesses have to install the shared escaping owner into
# the stub window before evaluating a module — the same order index.html uses.
_ESCAPE_OWNER = _read("js/html-escape.js")


def test_pi_shell_assets_are_explicitly_wired_before_guided_owner() -> None:
    index = _read("index.html")
    assert "css/guided-pi.css?v=20260829-post-plan-data1" in index
    assert "css/guided-pi-demo.css?v=20260815-reviewer-demo2" in index
    assert "css/guided-pi-preview.css?v=20260827-type-scale1" in index
    assert "css/guided-pi-workbench-preview.css?v=20260813-workbench1" in index
    assert "css/guided-pi-literature.css?v=20260828-literature-reader2" in index
    assert "js/screens-guided-pi-literature.js?v=20260828-literature-search1" in index
    assert "js/screens-guided-pi-markdown.js?v=20260827-readable-reply1" in index
    assert "js/screens-guided-pi-next-actions.js?v=20260829-compact-choices1" in index
    assert "js/screens-guided-pi-message-actions.js?v=20260828-plan-resubmit1" in index
    assert "js/screens-guided-pi-regeneration.js?v=20260828-regeneration-branch2" in index
    assert "js/screens-guided-pi-starters.js?v=20260827-independent-starters1" in index
    assert "js/screens-guided-pi-header.js?v=20260827-type-scale1" in index
    assert "js/screens-guided-pi-demo.js?v=20260815-real-render2" in index
    assert "js/screens-guided-pi-workbench-preview.js?v=20260829-post-plan-data2" in index
    assert (
        "js/screens-guided-pi-evidence-preview.js?v=20260825-evidence-preview1" in index
    )
    assert "js/screens-guided-pi-preview.js?v=20260829-data-scope1" in index
    assert "js/screens-guided-pi-replay.js?v=20260828-edit-plan1" in index
    assert "js/screens-guided-pi-activity.js?v=20260828-failure-history2" in index
    assert (
        "js/screens-guided-pi-provider.js?v=20260825-api-consent1"
        in index
    )
    assert "js/screens-guided-pi-project.js?v=20260827-conversation-first1" in index
    assert "js/screens-guided-pi-data-consent.js?v=20260829-data-scope1" in index
    assert "js/screens-guided-pi-data-binding.js?v=20260829-data-scope1" in index
    assert "js/screens-guided-pi-confirmation.js?v=20260829-post-plan-data1" in index
    assert "js/screens-guided-pi-childjob.js?v=20260828-plan-review1" in index
    assert "js/screens-guided-pi.js?v=20260829-post-plan-data1" in index
    assert (
        "js/screens-guided-project-continuity.js?v=20260813-project-continuity1"
        in index
    )
    assert "js/api.js?v=20260829-post-plan-data1" in index
    assert index.index("css/guided.css") < index.index("css/guided-pi.css")
    assert index.index("js/screens-guided-pi-literature.js") < index.index(
        "js/screens-guided-pi-markdown.js"
    )
    assert index.index("js/screens-guided-pi-markdown.js") < index.index(
        "js/screens-guided-pi-next-actions.js"
    )
    assert index.index("js/screens-guided-pi-next-actions.js") < index.index(
        "js/screens-guided-pi-message-actions.js"
    )
    assert index.index("js/screens-guided-pi-message-actions.js") < index.index(
        "js/screens-guided-pi-regeneration.js"
    )
    assert index.index("js/screens-guided-pi-message-actions.js") < index.index(
        "js/screens-guided-pi-starters.js"
    )
    assert index.index("js/screens-guided-pi-starters.js") < index.index(
        "js/screens-guided-pi-header.js"
    )
    assert index.index("js/screens-guided-pi-header.js") < index.index(
        "js/screens-guided-pi-demo.js"
    )
    assert index.index("js/screens-guided-pi-demo.js") < index.index(
        "js/screens-guided-pi-workbench-preview.js"
    )
    assert index.index("js/screens-guided-pi-workbench-preview.js") < index.index(
        "js/screens-guided-pi-evidence-preview.js"
    )
    assert index.index("js/screens-guided-pi-evidence-preview.js") < index.index(
        "js/screens-guided-pi-preview.js"
    )
    assert index.index("js/screens-guided-pi-preview.js") < index.index(
        "js/screens-guided-pi-activity.js"
    )
    assert index.index("js/screens-guided-pi-activity.js") < index.index(
        "js/screens-guided-pi-provider.js"
    )
    assert index.index("js/screens-guided-pi-provider.js") < index.index(
        "js/screens-guided-pi-project.js"
    )
    assert index.index("js/screens-guided-pi-project.js") < index.index(
        "js/screens-guided-pi-data-consent.js"
    )
    assert index.index("js/screens-guided-pi-data-consent.js") < index.index(
        "js/screens-guided-pi.js"
    )
    assert index.index("js/screens-guided-pi.js") < index.index("js/screens-guided.js")


def test_guided_pi_project_switch_clears_project_scoped_extraction_receipts() -> None:
    owner = _read("js/screens-guided-pi.js")
    bind_project = owner.split("function bindProject(project)", 1)[1].split(
        "function isActive()", 1
    )[0]

    assert "state.workflowReceipts = [];" in bind_project


def test_settled_turn_adopts_persisted_entry_ids_for_the_optimistic_row() -> None:
    """A sent message keeps no server entry id until the timeline is rebuilt.

    Regression: an ordinary turn settles with preserveTimeline=true so its live
    activity rows survive, which left the optimistically appended user row with
    an empty entryId and silently disabled explicit retry/edit actions until the
    user reloaded by hand.
    """

    owner = _read("js/screens-guided-pi.js")

    # The settle path must backfill instead of rebuilding the preserved timeline.
    assert "await refreshSession(!replacedBranch);" in owner
    assert "if (!replacedBranch) adoptPersistedEntryIds();" in owner

    backfill = owner.split("function adoptPersistedEntryIds()", 1)[1].split(
        "async function loadProjectSessions", 1
    )[0]
    # Only rows missing an id are touched, and only on an exact text match that
    # never moves backwards, so a replay can never target a different turn.
    assert "if (row.role !== 'user' || String(row.entryId || '').trim()) return;" in backfill
    assert "if (String(persisted[index].text || '').trim() !== text) continue;" in backfill
    assert "cursor = index + 1;" in backfill

def test_refresh_hides_intermediate_provider_project_and_activation_panels() -> None:
    shell = _read("js/screens-guided-pi.js")
    guided = _read("js/screens-guided.js")
    startup = _read("js/screens-guided-startup.js")
    startup_css = _read("css/guided-startup.css")
    index = _read("index.html")

    assert (
        "const restoring = state.loading || state.projectLoading || "
        "state.projectDiscoveryLoading"
    ) in shell
    assert "restoring\n      ? restoringPanel()" in shell
    assert "正在恢复当前研究" in shell
    assert "state.projectLoading = !!next" in shell
    assert "state.projectLoading = false" in shell
    assert "function setProjectDiscoveryLoading(active)" in shell
    assert "return window.EU_API.loadGuidedDrafts" in guided
    assert "await openGuidedProjectMemory" in guided
    assert "const draftsReady = loadGuidedDrafts()" in guided
    assert "Promise.allSettled([Promise.resolve(draftsReady), piReady]).finally" in guided
    assert guided.index("piOwner.mount(root.querySelector('#gdPiShell'))") < guided.index(
        "const draftsReady = loadGuidedDrafts()"
    )
    assert "piOwner.setProjectDiscoveryLoading(true)" in guided
    assert "piOwner.setProjectDiscoveryLoading(false)" in guided
    assert "正在恢复当前研究" in guided
    assert "return bindProjectToPi(result, row);" in guided
    assert "state.startupPromise = Promise.resolve(loadStatus())" in shell
    assert "return state.startupPromise || Promise.resolve();" in shell
    assert "window.EU_GUIDED_STARTUP" in startup
    assert "data-guided-startup-shield" in startup
    assert ".gd-main.gd-startup-active" in startup_css
    assert ".gd-startup-shield" in startup_css
    assert "css/guided-startup.css?v=20260827-atomic-restore1" in index
    assert "js/screens-guided-startup.js?v=20260827-atomic-restore1" in index
    assert index.index("js/screens-guided-startup.js") < index.index("js/screens-guided.js")
    assert "gd-startup" not in _read("css/guided-projects.css")


def test_new_research_conversation_keeps_chat_open_until_data_is_needed() -> None:
    owner = _read("js/screens-guided-pi-data-consent.js")
    shell = _read("js/screens-guided-pi.js")
    data_binding_owner = _read("js/screens-guided-pi-data-binding.js")
    header = _read("js/screens-guided-pi-header.js")
    api = _read("js/api.js")

    assert "selection_in_progress" in owner
    assert "Return to local folder selection" in owner
    assert "Local data selection is open" in owner
    assert "No data selected" not in owner
    assert "Keep chatting; choose data" not in owner
    assert "Paths remain in the EasyICU host UI" in owner
    assert "data-gpi-data-source-action" in owner
    assert "data-gpi-data-demo" not in owner
    assert "data-gpi-data-planning" not in owner
    assert '<section class="gpi-data-consent"' in owner
    assert "<summary>" not in owner
    assert "authorizePiCopilotDataSource" in data_binding_owner
    assert "confirm_selected_source" in data_binding_owner
    assert "data-source-authorization" in api
    assert "window.EU_API.authorizePiCopilotDataSource" in api

    session_panel = shell[
        shell.index("function sessionPanel()") : shell.index("function demoPanel()")
    ]
    send_text = shell[
        shell.index("async function sendText(text, grantsOverride, turnIntent, visibleUserMessage = true)") : shell.index(
            "async function sendMessage()"
        )
    ]
    assert '<div class="gpi-compose">' in session_panel
    assert '<div class="gpi-compose-card${activeChild ? \' is-running\' : \'\'}">' in session_panel
    assert "dataConsentRequired ? '' : `<div class=\"gpi-compose\">" not in session_panel
    assert "DATA_CONSENT.requiresConfirmation" not in send_text
    assert "Reviewer demo" in header
    assert "Full demo" not in header


def test_formal_plan_buttons_launch_the_governed_job_without_model_prompt_roundtrip() -> None:
    owner = _read("js/screens-guided-pi.js")

    assert "async function startCurrentFormalPlanGeneration(reasonCode)" in owner
    assert "api().startAgentRun" in owner
    assert "engine: 'research_agent_pipeline'" in owner
    assert "literature_search_authorized: true" in owner
    approval = owner.split("async function submitCurrentPlanReview", 1)[1].split(
        "async function startCurrentFormalPlanGeneration", 1
    )[0]
    assert "api().submitAgentRunReview" in approval
    assert "sendText(" not in approval


def test_new_research_session_starters_minimize_interaction_cost() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-starters.js")
    shell = _read("js/screens-guided-pi.js")
    index = _read("index.html")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value) }} }};
      eval({owner!r});
      const tr = (en, zh) => zh;
      const html = window.EU_GUIDED_PI_STARTERS.render({{ tr, disabled: false }});
      function eventFor(kind, value) {{
        const attribute = kind === 'compose' ? 'gpiStarterCompose' : 'gpiStarterSend';
        const marker = kind === 'compose' ? 'starter-compose' : 'starter-send';
        const node = {{ dataset: {{}}, closest: selector => selector.includes(marker) ? node : null }};
        node.dataset[attribute] = value;
        return {{ target: node }};
      }}
      const compose = window.EU_GUIDED_PI_STARTERS.actionFromEvent(eventFor('compose', '我想研究：'));
      const send = window.EU_GUIDED_PI_STARTERS.actionFromEvent(eventFor('send', '帮我寻找研究方向'));
      console.log(JSON.stringify({{ html, compose, send }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert "你想从哪里开始？" in completed.stdout
    assert "队列构建、特征标准化、研究计划、统计分析、图表和研究报告" in completed.stdout
    assert "开始一个研究问题" in completed.stdout
    assert 'class=\\"primary\\"' in completed.stdout
    assert "从文献寻找研究方向" in completed.stdout
    assert "提取并分析 ICU 数据" in completed.stdout
    assert "评估一个研究想法" in completed.stdout
    assert "从现有数据发现机会" not in completed.stdout
    assert "只有研究需要读取数据时，再让我确认数据源" in completed.stdout
    assert '"kind":"compose"' in completed.stdout
    assert '"kind":"send"' in completed.stdout
    assert "sendText(starterAction.text, []);" in shell
    assert "state.draft = starterAction.text;" in shell
    assert "你想从哪里开始？" not in shell
    assert "从文献寻找研究方向" not in shell
    assert index.index("screens-guided-pi-starters.js") < index.index("screens-guided-pi.js")


def test_guided_header_and_progress_keep_secondary_controls_available() -> None:
    header = _read("js/screens-guided-pi-header.js")
    shell = _read("js/screens-guided-pi.js")
    aside_owner = _read("js/screens-guided-pi-aside.js")

    assert "window.EU_GUIDED_PI_HEADER = { render };" in header
    assert "gpi-head-new" in header and "data-gpi-new" in header
    assert "gpi-head-overflow-menu" in header
    for selector in (
        "data-gpi-study-setup",
        "data-gpi-config",
        "data-gpi-presentation-pin",
        "data-gpi-demo",
    ):
        assert selector in header
    assert "HEADER.render({" in shell
    assert "function dismissHeaderOverflow(event)" in shell
    assert 'data-gpi-input rows="2"' in shell
    assert "gpi-log-start" in shell
    assert "gd-pipeline-next" in aside_owner
    assert "gd-pipeline-disclosure" in aside_owner


def test_pending_data_source_status_is_hidden_until_selection_starts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-data-consent.js")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value || '') }} }};
      eval({owner!r});
      const ctx = {{
        tr: (en) => en,
        esc: (value) => String(value),
        icon: () => '',
      }};
      const pending = window.EU_GUIDED_PI_DATA_CONSENT.render({{
        data_source_authorization: {{status: 'pending', reason: 'local_data_selection_required'}},
      }}, ctx);
      const selecting = window.EU_GUIDED_PI_DATA_CONSENT.render({{
        data_source_authorization: {{status: 'selection_in_progress', reason: 'local_data_selection_required'}},
      }}, ctx);
      const reusable = window.EU_GUIDED_PI_DATA_CONSENT.render({{
        data_source_authorization: {{
          status: 'pending', reason: 'project_source_confirmation_required',
          source: {{label: 'MIMIC-IV', reference_release: '3.1'}},
        }},
      }}, ctx);
      if (pending !== '') throw new Error('pending data state must not occupy the conversation');
      if (!selecting.includes('<section class="gpi-data-consent"')) throw new Error('active selection must stay visible');
      if (!reusable.includes('data-gpi-data-source-action="reuse_project_source"')) throw new Error('bound project source must be confirmable');
      if (!reusable.includes('data-gpi-data-source-action="use_study_required_data"')) throw new Error('study-required preparation must be offered');
      if (!reusable.includes('data-gpi-data-source-action="begin_full_data_selection"')) throw new Error('full extraction must be offered');
      if (!reusable.includes('Prepare only study-required data (recommended)')) throw new Error('recommended scope must be explicit');
      if (!reusable.includes('MIMIC-IV v3.1')) throw new Error('bound source identity must be path free and visible');
      console.log('ok');
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "ok"


def test_local_source_picker_activates_the_sessions_bound_study_context() -> None:
    owner = _read("js/screens-guided-pi.js")
    data_binding_owner = _read("js/screens-guided-pi-data-binding.js")
    starters = _read("js/screens-guided-pi-starters.js")
    authorization = data_binding_owner.split(
        "async function authorizeDataSource(action, options)", 1
    )[1].split("function notifyExtractionHandoff", 1)[0]

    assert "payload.resource && payload.resource.study_context_id" in authorization
    assert "host.session().binding.study_context_id" in authorization
    assert "await store.hydrate({ force: true })" in authorization
    assert "await store.activate(contextId)" in authorization
    assert authorization.index("await store.hydrate({ force: true })") < authorization.index(
        "await store.activate(contextId)"
    )
    assert authorization.index("await store.activate(contextId)") < authorization.index(
        "window.EU_GUIDED_PI_PREVIEW.open"
    )
    assert "emptyResearchHtml" in owner
    assert "需要读取或分析数据时" in starters


def test_data_binding_owner_executes_local_picker_and_preserves_its_host_contract() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-data-binding.js")
    script = f"""
      global.window = {{
        EU_STUDY_CONTEXT: {{
          hydrate: async () => calls.push('hydrate'),
          active: () => null,
          activate: async id => calls.push('activate:' + id),
        }},
        EU_GUIDED_PI_PREVIEW: {{open: resource => calls.push('open:' + resource.kind)}},
      }};
      global.requestAnimationFrame = callback => callback();
      eval({owner!r});
      const calls = [];
      let session = {{session_id: 'session-1', binding: {{study_context_id: 'context-1'}}}};
      let error = '';
      let receipts = [];
      const api = {{
        authorizePiCopilotDataSource: async (_sessionId, request) => {{
          calls.push('authorize:' + request.action + ':' + request.database);
          return {{
            session: {{...session, data_source_authorization: {{status: 'selection_in_progress'}}}},
            resource: {{kind: 'native_workspace', study_context_id: 'context-1'}},
          }};
        }},
      }};
      const binding = window.EU_GUIDED_PI_DATA_BINDING.create({{
        api: () => api,
        render: () => calls.push('render'),
        projectId: () => 'project-1',
        loadWorkflow: async () => calls.push('workflow'),
        dataConsent: {{selectionInProgress: () => false}},
        errorText: value => 'mapped:' + String(value && value.message || value),
        rememberSession: id => calls.push('remember:' + id),
        continueAfterDataSourceConfirmation: async () => true,
        root: () => null,
        busy: () => false,
        session: () => session,
        setSession: value => {{ session = value; }},
        setError: value => {{ error = value; }},
        workflowReceipts: () => receipts,
        setWorkflowReceipts: value => {{ receipts = value; }},
      }});
      await binding.authorizeDataSource('begin_full_data_selection', {{database: 'miiv'}});
      binding.notifyExtractionHandoff({{id: 'receipt-1', database: 'miiv'}});
      api.authorizePiCopilotDataSource = async () => {{ throw new Error('picker failed'); }};
      await binding.authorizeDataSource('begin_local_selection', {{database: 'miiv'}});
      console.log(JSON.stringify({{calls, error, receipts, status: session.data_source_authorization.status}}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert "authorize:begin_full_data_selection:miiv" in payload["calls"]
    assert "remember:session-1" in payload["calls"]
    assert "hydrate" in payload["calls"]
    assert "activate:context-1" in payload["calls"]
    assert "open:native_workspace" in payload["calls"]
    assert payload["status"] == "selection_in_progress"
    assert payload["error"] == "mapped:picker failed"
    assert payload["receipts"][0]["id"] == "receipt-1"


def test_new_conversation_binds_only_a_source_before_model_guided_setup() -> None:
    extraction = _read("js/screens-extraction.js")
    embedded = _read("js/screens-extraction-embedded.js")
    preview = _read("js/screens-guided-pi-preview.js")
    shell = _read("js/screens-guided-pi.js")
    data_binding_owner = _read("js/screens-guided-pi-data-binding.js")
    guided = _read("js/screens-guided.js")

    source_binding = embedded.split("function projectSourceBinding(root)", 1)[1].split(
        "function paint()", 1
    )[0]
    source_persist = extraction.split("function bindSourceToCopilot()", 1)[1].split(
        "function syncExtractionToCopilot()", 1
    )[0]

    assert "entry_mode" in preview and "source_binding" in preview
    assert "entry_mode === 'source_binding'" in embedded
    assert "beginSourceBinding" in extraction
    assert "No research settings have been chosen yet" in source_binding
    assert "Confirm data source and continue" in source_binding
    assert "research question" in source_binding
    assert "Current extraction setup" not in source_binding
    assert "Start extraction" not in source_binding
    assert "Save setup to Copilot" not in source_binding
    assert "data_source: snapshot.data_source" in source_persist
    assert "cohort" not in source_persist
    assert "modules" not in source_persist
    assert "export_format" not in source_persist
    assert "confirmDataSourceBinding" in shell
    assert "action: 'confirm_selected_source'" in data_binding_owner
    assert "EU_GUIDED_PI_PREVIEW.close()" in shell
    assert "easyicu:guided-projects-refresh" in data_binding_owner
    assert "easyicu:guided-projects-refresh" in guided
    assert "loadGuidedDrafts(true)" in guided


def test_source_binding_shell_refresh_preserves_the_validated_folder() -> None:
    embedded = _read("js/screens-extraction-embedded.js")
    extraction = _read("js/screens-extraction.js")

    mount = embedded.split("mount(nextHost, nextOptions)", 1)[1].split(
        "unmount(nextHost)", 1
    )[0]
    use_data = extraction.split(
        "const useDataBtn = root.querySelector('[data-ex-usedata]')", 1
    )[1].split("const startConvBtn", 1)[0]

    assert "sourceBindingCoordinate" in embedded
    assert "coordinate !== sourceBindingCoordinate" in mount
    assert mount.index("coordinate !== sourceBindingCoordinate") < mount.index(
        "owner.beginSourceBinding()"
    )
    assert "Promise.resolve(registered).then" in use_data
    assert use_data.index("rememberExportPath(exPath)") < use_data.index(
        "exReal = 'ready'"
    )


def test_extraction_result_refreshes_study_context_before_cas_handoff() -> None:
    embedded = _read("js/screens-extraction-embedded.js")
    study_context = _read("js/study-context.js")
    sync = embedded.split("function syncToCopilot(event)", 1)[1].split(
        "function refreshJob(event)", 1
    )[0]

    assert "function refreshActiveFromServer()" in study_context
    assert "api.loadStudyContext(contextId)" in study_context
    assert "text(saved.id) !== contextId" in study_context
    assert "setDirty(contextId, false)" in study_context
    assert "refreshActiveFromServer," in study_context
    assert "store.refreshActiveFromServer()" in sync
    assert sync.index("store.refreshActiveFromServer()") < sync.index(
        "owner.syncToCopilot()"
    )


def test_new_session_selection_cannot_be_overwritten_by_a_stale_restore() -> None:
    owner = _read("js/screens-guided-pi.js")
    create = owner.split("async function createSession()", 1)[1].split(
        "async function openSession", 1
    )[0]
    open_session = owner.split("async function openSession", 1)[1].split(
        "function assistantRow", 1
    )[0]
    restore = owner.split("async function loadProjectSessions(", 1)[1].split(
        "async function loadWorkflow", 1
    )[0]

    assert "const selectionRevision = ++state.sessionSelectionRevision" in create
    assert "selectionRevision !== state.sessionSelectionRevision" in create
    assert "expectedSelectionRevision !== state.sessionSelectionRevision" in open_session
    assert "const selectionRevision = state.sessionSelectionRevision" in restore
    assert "await openSession(preferred, selectionRevision, refreshWorkflow)" in restore
    assert "state.sessionSelectionRevision += 1" in owner


def test_extraction_handoff_preserves_the_registered_source_identity() -> None:
    owner = _read("js/screens-extraction.js")
    snapshot = owner.split("window.EU_EXTRACTION_CONTEXT =", 1)[1].split(
        "/* Minimal closure adapter", 1
    )[0]

    assert "const sourcePath = resultPath || (active && active.path) || '';" in snapshot
    assert "source_id: String(active && active.id || '')" in snapshot
    assert "const sourceLabel = (active && active.label)" in snapshot


def test_conversational_data_workbench_assets_are_route_owned() -> None:
    index = _read("index.html")
    preview = _read("js/screens-guided-pi-data-preview.js")
    resources = _read("js/screens-guided-pi-resources.js")
    css = _read("css/guided-pi-data-preview.css")

    assert index.index("css/guided-pi-data-preview.css") < index.index(
        "js/screens-viz-embedded.js"
    )
    assert index.index("js/screens-viz-context.js") < index.index("js/screens-viz.js")
    assert index.index("js/screens-viz.js") < index.index("js/screens-viz-embedded.js")
    assert index.index("js/screens-viz-embedded.js") < index.index(
        "js/screens-guided-pi-data-preview.js"
    )
    assert index.index("js/screens-guided-pi-data-preview.js") < index.index(
        "js/screens-guided-pi-preview.js"
    )
    assert index.index("js/screens-guided-pi-resources.js") < index.index(
        "js/screens-guided-pi.js"
    )
    assert "window.EU_GUIDED_PI_DATA_PREVIEW" in preview
    assert "window.EU_VIZ_EMBEDDED_WORKBENCH" in _read("js/screens-viz-embedded.js")
    assert "data_workbench_snapshot" in resources
    assert ".gpi-viz-embed" in css
    for path in sorted((STATIC / "css").glob("*.css")):
        if path.name != "guided-pi-data-preview.css":
            assert ".gpi-viz-embed" not in path.read_text(encoding="utf-8")
    for marker in (".patient-", ".cohort-", ".crossdb-", ".extract-"):
        assert marker not in css


def test_user_facing_copilot_copy_hides_the_pi_runtime_brand() -> None:
    label_owner = STATIC / "js" / "product-labels.js"
    scripts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC / "js").glob("*.js"))
        if path != label_owner
    )
    product_copy = re.sub(r"/\*.*?\*/", "", scripts, flags=re.S)

    assert not re.search(r"\bPi\b|PI AGENTSESSION|PI COPILOT", product_copy)
    assert "'Pi Copilot'" in label_owner.read_text(encoding="utf-8")
    assert "EasyICU Copilot" in product_copy
    assert "EasyICU 研究助手" in product_copy


def test_data_source_copy_uses_easyicu_availability_and_hides_registry_labels() -> None:
    prompt_owner = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")
    activity_owner = _read("js/screens-guided-pi-activity.js")

    assert "recommended_source is a recommendation only" in prompt_owner
    assert "an already available EasyICU data export" in prompt_owner
    assert "aggregate.stays and module_count" in prompt_owner
    assert "choose and register another local data directory" in prompt_owner
    assert "Never describe the already available export as a file the user still needs to locate or download" in prompt_owner
    assert "Use each returned display_label exactly" in prompt_owner
    assert "present every returned row as its own clickable choice" in prompt_owner
    assert "sole exception to the ordinary 2-to-4 next-step choice limit" in prompt_owner
    assert "When reference_release is null" in prompt_owner
    assert "MIMIC-IV v3.1" in prompt_owner
    assert "MIMIC-III v1.4" in prompt_owner
    assert "never expose registry terminology or internal run labels" in prompt_owner
    assert "Offer the official demo choice only" in prompt_owner
    assert "Official-demo listing fast path" in prompt_owner
    assert "exactly once without a database filter" in prompt_owner
    assert "never offer a local or full-database workflow" in prompt_owner
    assert "call easyicu_prepare_demo_source directly" in prompt_owner
    assert "never pass an official demo catalog id to bind_source_id" in prompt_owner
    assert "then ask one source-mode question" not in prompt_owner
    assert "Confirmed conversation source rule:" in prompt_owner
    assert "Prepared registered-export reuse rule:" in prompt_owner
    assert "A database name repeated inside the research question is not a source-change request" in prompt_owner
    assert "List supported and registered data sources" not in prompt_owner
    assert "Check EasyICU data availability" in prompt_owner
    assert "检查 EasyICU 数据源目录" in activity_owner
    assert "检查 EasyICU 可用数据" not in activity_owner
    assert "已检查 EasyICU 数据源目录" in activity_owner
    assert "已确认 EasyICU 可用数据" not in activity_owner
    assert "列出已登记数据源" not in activity_owner
    assert "已列出已登记数据源" not in activity_owner


def test_model_guidance_keeps_locked_clinical_implementation_off_the_user() -> None:
    prompt_owner = (NODE_APP / "src" / "main.mjs").read_text(encoding="utf-8")
    phenotype_rule = prompt_owner.split("Clinical phenotype rule:", 1)[1].split(
        '",', 1
    )[0]

    assert "one owner-locked canonical clinical definition" in phenotype_rule
    assert "do not ask the user to choose its internal windows" in phenotype_rule
    assert "genuinely unresolved, clinically non-equivalent variant" in phenotype_rule
    assert "hide internal identifiers" in phenotype_rule
    assert "User decision burden rule:" in prompt_owner
    assert "EasyICU owns implementation details" in prompt_owner
    assert "Question phrasing rule:" in prompt_owner
    assert "Ordinary outcome semantics rule:" in prompt_owner
    assert "Semantic consistency rule:" in prompt_owner
    assert "never ask the user to approve a wording-only synchronization" in prompt_owner
    assert "ICU mortality means death during that ICU stay" in prompt_owner
    assert "arbitrary 24-hour, 48-hour, or 72-hour mortality variants" in prompt_owner
    assert "Next-step choice quality rule:" in prompt_owner
    assert "instead of adding a duplicate recommendation option" in prompt_owner
    assert "without Markdown markers" in prompt_owner
    assert "Simple-decision fast path:" in prompt_owner
    assert "A repeated selection of the already-bound source is not a new setup decision" in prompt_owner
    assert "return to the same formal-plan generation confirmation" in prompt_owner
    assert "do not produce a candidate brief or individual setup question" in prompt_owner
    assert "Final-slot convergence rule:" in prompt_owner
    assert "Do not call easyicu_start_extraction while workflow" in prompt_owner
    assert "must not be expanded into an execution-readiness search" in prompt_owner
    assert "Save an explicit human decision first" in prompt_owner
    assert "is not by itself authorization to start catalog resolution" in prompt_owner
    assert "never open with a list of internal missing fields" in prompt_owner
    assert "Persistence wording rule:" in prompt_owner
    assert "unless a successful EasyICU mutation receipt occurred" in prompt_owner
    assert "say only that the user selected or stated the value" in prompt_owner
    assert "Plan-intent rule:" in prompt_owner
    assert "do not generate a Research Brief, shadow plan" in prompt_owner
    assert "the host will show exactly two user actions" in prompt_owner
    assert "propose the unresolved scientific design" in prompt_owner
    assert "Formal-plan display authority rule:" in prompt_owner
    assert "Only the digest-bound agent_plan.json" in prompt_owner
    assert "Never claim literature support when no literature receipt exists" in prompt_owner
    assert "Save it only after the user directly confirms that outcome" in prompt_owner
    assert "Analysis-unit approval rule:" in prompt_owner
    assert "belong together in the formal plan review" in prompt_owner


def test_activation_initializes_first_use_projects_and_surfaces_failures() -> None:
    owner = _read("js/screens-guided-pi.js")
    project_owner = _read("js/screens-guided-pi-project.js")
    panel = owner.split("function activatePanel()", 1)[1].split(
        "function projectRequiredPanel()", 1
    )[0]
    create = owner.split("async function createSession()", 1)[1].split(
        "async function openSession", 1
    )[0]
    load_status = owner.split("async function loadStatus()", 1)[1].split(
        "function stopCodexPoll", 1
    )[0]

    assert 'class="gpi-error" role="alert"' in panel
    assert (
        "state.projectInitialization && state.projectInitialization.required" in create
    )
    assert "confirm_initialization: true" in create
    assert create.index("confirm_initialization: true") < create.index(
        "createPiCopilotSession"
    )
    assert (
        "if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;"
        in create
    )
    assert "try { await prepareProject(); }" in load_status
    assert "catch (error) { state.error = errorText(error); }" in load_status
    assert "pi_project_study_context_missing" in project_owner
    assert "当前项目保存的研究配置已不存在" in owner
    assert "关联的研究配置已经失效" in panel
    assert "EasyICU 不会静默创建或绑定另一份配置" in panel
    assert "data-newstudy" in panel
    assert "data-refreshdrafts" in panel
    assert "state.projectIssue === 'pi_project_study_context_missing'" in panel
    assert "state.projectIssue = error.code" in project_owner
    assert "confirm_initialization: false" in project_owner
    assert "confirm_initialization: false" not in owner
    assert "const workflowReady = loadWorkflow().then(render)" in project_owner
    assert "await loadProjectSessions(false)" in project_owner
    assert "await workflowReady" in project_owner
    assert "if (refreshWorkflow !== false) await loadWorkflow()" in owner
    assert "state.projectPrepareId === expectedProjectId" in owner
    assert "state.projectPreparePromise = Promise.resolve(pending)" in owner
    render = owner.split("function render()", 1)[1].split(
        "async function loadStatus()", 1
    )[0]
    panel_selection = render.split("state.host.innerHTML", 1)[1]
    assert panel_selection.index(
        "state.projectIssue === 'pi_project_study_context_missing'"
    ) < (panel_selection.index("state.showSetup || !connectionReady()"))


def test_get_requests_preserve_typed_backend_error_codes() -> None:
    api_owner = _read("js/api.js")
    get_json = api_owner.split("async function getJSON(path)", 1)[1].split(
        "async function postJSON(path, body)", 1
    )[0]

    assert "payload && payload.detail" in get_json
    assert "throw apiError(path, res, d)" in get_json


def test_transient_codex_catalog_failure_preserves_last_verified_models() -> None:
    pi_owner = _read("js/screens-guided-pi.js")
    load_models = pi_owner.split("async function loadCodexModels", 1)[1].split(
        "async function loadCodexResearchStatus", 1
    )[0]
    error_branch = load_models.split("} catch (error) {", 1)[1]

    assert "state.codexModels = []" not in error_branch
    assert "if (renderAfter) state.error = errorText(error)" in error_branch
    assert "state.codexLogin = null; state.codexModels = []" in pi_owner


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
    aside_owner = _read("js/screens-guided-pi-aside.js")
    childjob_owner = _read("js/screens-guided-pi-childjob.js")
    transcript_owner = _read("js/screens-guided-pi-transcript.js")
    activity_owner = _read("js/screens-guided-pi-activity.js")
    provider_owner = _read("js/screens-guided-pi-provider.js")
    header_owner = _read("js/screens-guided-pi-header.js")
    resource_owner = _read("js/screens-guided-pi-resources.js")
    api = _read("js/api.js")
    assert 'id="gdPiShell"' in guided
    assert 'id="gdLegacyShell"' in guided
    assert "piOwner.mount" in guided
    assert "window.EU_GUIDED_PI = {" in pi_owner
    for public_method in (
        "mount",
        "unmount",
        "setShell",
        "bindProject",
        "isActive",
        "rebind",
        "notifyExtractionHandoff",
        "confirmDataSourceBinding",
    ):
        assert public_method in pi_owner
    assert "new EventSource('/api/jobs/'" in pi_owner
    assert "syncProjectWorkflowAside" in pi_owner
    assert "completed_required_stages" in pi_owner
    assert "operator_plan_approval_required" in aside_owner
    assert "plan_execution_upgrade_required" in aside_owner
    confirmation_owner = _read("js/screens-guided-pi-confirmation.js")
    assert (
        "I approve this exact evidence-bound plan without changing the study configuration"
        in confirmation_owner
    )
    assert "本轮不新增可选的科学设定" in confirmation_owner
    assert "preserve every open scientific finding as a limitation" in confirmation_owner
    assert "reviewResources" in confirmation_owner
    assert "计划审阅材料" in confirmation_owner
    assert "预览正式研究计划" in confirmation_owner
    assert "查看该计划的文献依据" in confirmation_owner
    assert "已复用之前准备好的完整数据包" in confirmation_owner
    assert "批准后不会重新提取数据" in confirmation_owner
    assert "先预览分析数据" in confirmation_owner
    assert "批准计划并开始分析" in confirmation_owner
    assert "失败关闭运行的只读产物" in confirmation_owner
    assert "预览上一版候选计划" in confirmation_owner
    assert "查看上一版文献快照" in confirmation_owner
    assert "预览未验证结果表" in confirmation_owner
    assert "预览未验证图件" in confirmation_owner
    assert "预览证据绑定文章" in confirmation_owner
    assert "manuscript_provenance.json" in confirmation_owner
    assert "仍属未验证状态，不能签署或发表" in confirmation_owner
    assert "gpi-confirmation-resources" in confirmation_owner
    assert "检索来源、筛选理由和每个计划步骤的精确引用绑定" in confirmation_owner
    assert "重新生成新计划" in confirmation_owner
    assert "submitAgentRunReview" in pi_owner
    assert "easyicu_review_submitted" in pi_owner
    assert "submitAgentRunReview" in api
    assert "'/api/jobs/agent-run-review'" in api
    assert "stage.status === 'review_required'" in aside_owner
    assert "data-gpi-project-workflow-aside" in aside_owner
    assert "pi_model_provider_unavailable" in pi_owner
    assert "pi_shell_token_budget_exhausted" in pi_owner
    assert "Research Agent 规划任务已提交" in childjob_owner
    assert "EasyICU 完整科研分析已提交" not in pi_owner
    assert "同一研究项目中新建后续对话" in pi_owner
    assert "external_llm_opt_in: true" in pi_owner
    assert pi_owner.count("project_id: projectId()") >= 4
    assert "loadPiCopilotSessions(100, expectedProjectId)" in pi_owner
    assert "easyicu_pi_copilot_session:' + encodeURIComponent(projectId())" in pi_owner
    assert "project_dir" not in pi_owner
    assert "window.EU_GUIDED_PI.bindProject" in guided
    assert "if (usePiSession) return bindProjectToPi(result, row);" in guided
    assert "restoreGuidedProjectThread(result, row, kind);" in guided
    assert (
        "if (piProjectShellActive()) bindProjectToPi(result, selectedGuidedDraft);"
        in guided
    )
    assert "Conversation memory" not in guided
    assert "对话记忆" not in guided
    assert (
        "Study setup, runs, evidence, and conversation history stay here."
        in projects_owner
    )
    assert "function renderShellRail(ctx)" in projects_owner
    assert 'class="gd-rail"' in projects_owner
    assert 'class="gd-rail"' not in guided
    assert "guidedProjectRenderer('renderShellRail')" in guided
    assert "data-gpi-provider-form" in provider_owner
    assert '<form class="gpi-provider-section"' not in pi_owner
    assert "window.EU_GUIDED_PI_PROVIDER" in provider_owner
    assert "CLIProxyAPI / Local proxy" in provider_owner
    assert "gpt-5.6-luna" in provider_owner
    assert "gpt5.6 luna" not in provider_owner
    assert "anthropic-messages" in provider_owner
    assert "google-generative-ai" in provider_owner
    assert "data-ag-" not in provider_owner
    assert "static_preview_no_backend" in pi_owner
    assert "http://127.0.0.1:8765/#guided" in pi_owner
    assert "gpi-model-options" in provider_owner
    assert 'type="password"' in provider_owner
    assert 'name="enable_ai"' not in provider_owner
    assert 'type="checkbox"' not in provider_owner
    assert "验证并保存连接" in provider_owner
    assert "科研运行仍需另行确认" in provider_owner
    assert "savePiCopilotProviderConfig" in pi_owner
    assert "enable_ai: true" in pi_owner
    assert "data.get('enable_ai')" not in pi_owner
    assert "provider_connection_unverified" in pi_owner
    assert "localStorage.setItem('easyicu_pi_api" not in pi_owner
    assert "keyInput.value = ''" in pi_owner
    assert "ACCESS_MODE_GRANTS" in pi_owner
    assert "data-gpi-access-mode" in pi_owner
    assert "Ask first" in pi_owner
    assert "Auto-approve" in pi_owner
    assert "Full access" in pi_owner
    assert "data-gpi-grant" not in pi_owner
    assert "data-gpi-resource-file" in resource_owner
    assert "data-gpi-resource-run" in resource_owner
    assert "data-gpi-resource-artifact" in resource_owner
    assert "RESOURCE_OWNER.fromButton(resource)" in pi_owner
    assert 'data-gpi-mode-switch="workspace"' in header_owner
    assert "const HEADER = window.EU_GUIDED_PI_HEADER;" in pi_owner
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
    assert "if (host.session() && sessionIsStale()) await rebind();" in childjob_owner
    assert "const archiveChildJob = host.archiveChildJob;" in childjob_owner
    assert "archiveChildJob: (...args) => archiveChildJob(...args)," in pi_owner
    assert "wrapupTimedOut" in pi_owner
    assert "operator_plan_approval_required" in pi_owner
    assert "planner_checkpoint_resume_available" in pi_owner
    assert "wrapupRecovered" in transcript_owner
    assert "workflowActionCode" in transcript_owner
    assert "function reconcileDurableWrapupActivity()" in pi_owner
    assert "latest.status = 'complete'" in pi_owner
    assert "function reconcileSettledSession()" in pi_owner
    assert "state.session.streaming !== false" in pi_owner
    assert pi_owner.count("reconcileSettledSession();") == 2
    assert "const VISIBLE_KINDS = new Set([" in activity_owner
    for kind in (
        "'submitted'",
        "'agent'",
        "'turn'",
        "'assistant'",
        "'tool'",
        "'pipeline'",
        "'retry'",
        "'compaction'",
    ):
        assert kind in activity_owner
    assert "Live progress connection stopped" in childjob_owner
    assert "private chain-of-thought" in activity_owner
    assert "loadPiCopilotProjectWorkflow" in pi_owner
    assert "gpi-workflow" in pi_owner
    assert "Research workflow" in pi_owner
    assert "Used ${toolSteps.length} EasyICU tools" in activity_owner
    assert "gpi-activity-live" in activity_owner
    assert "completedToolLabel" in activity_owner
    assert "initializePiCopilotProject" in pi_owner
    assert "history-activity-" in transcript_owner
    assert "closeHistoryActivity" in transcript_owner
    assert "row.role === 'activity'" in pi_owner
    assert "gpi-avatar" not in pi_owner
    assert "private chain-of-thought" in activity_owner
    assert "assistantTextHtml" in pi_owner
    assert (
        "row.role === 'assistant' ? assistantTextHtml(visibleText) : esc(visibleText)"
        in pi_owner
    )
    assert "event.type === 'run_start'" in pi_owner
    assert "event.type === 'tool_progress'" in pi_owner
    assert "event.type === 'run_end'" in pi_owner
    assert "workspace file contents may be sent to this service" in provider_owner
    assert "PHI-safe summaries" in provider_owner
    assert "patient rows, credentials, or arbitrary host files" in pi_owner
    assert "data-gpi-confirm-action" in pi_owner
    assert "data-gpi-confirm-preview-data" in pi_owner
    assert "data-gpi-confirm-reject" in pi_owner
    assert "本次只提交“拒绝”审核决定" in confirmation_owner
    assert "data-gpi-demo" in pi_owner
    assert "data-gpi-demo-exit" in pi_owner
    assert "查看完整科研流程演示" in pi_owner
    assert "开始研究对话" in pi_owner
    assert "研究进度会自动保存" in pi_owner
    assert "开始前，先创建空白研究配置" not in pi_owner
    assert "gpi-mode-intro" not in pi_owner
    assert "data-gpi-mode-choice" not in pi_owner
    assert "gpi-secondary-actions" in pi_owner
    assert "我同意——启用 EasyICU 研究助手" not in pi_owner
    assert "align-self:flex-start" in _read("css/guided-pi.css")
    assert "state.demoMode ? demoPanel()" in pi_owner
    assert "extraction_ready" in confirmation_owner
    assert "plan_ready" in aside_owner
    assert "grants: ['run']" in confirmation_owner
    assert "运行本地预检" in confirmation_owner
    assert "provider_ready_to_generate_plan" in pi_owner
    assert "开始生成研究计划" in confirmation_owner
    assert "这一阶段不强制要求已准备的数据包" in confirmation_owner
    assert "只依据数据库能力目录出计划且不读取患者行" in confirmation_owner
    assert "我想先补充研究要求" in confirmation_owner
    assert "grants: ['provider_run', 'literature']" in confirmation_owner
    assert "plan_configuration_superseded" in aside_owner
    assert "重新生成计划" in confirmation_owner
    assert "sendText(message, governedNextChoiceGrants(null, message))" in pi_owner
    assert "operator_plan_approval_required" in aside_owner
    assert "hydrateProjectedJob" in pi_owner
    assert "visibleSteps.length} steps" in activity_owner
    for method in (
        "loadPiCopilotStatus",
        "savePiCopilotProviderConfig",
        "createPiCopilotSession",
        "initializePiCopilotProject",
        "loadPiCopilotProjectWorkflow",
        "loadPiCopilotSessions",
        "loadPiCopilotSession",
        "sendPiCopilotMessage",
        "regeneratePiCopilotMessage",
        "rebindPiCopilotSession",
        "pinPiCopilotPresentation",
        "archivePiCopilotChildJob",
        "abortPiCopilotSession",
        "loadPiCopilotWorkspaceFile",
        "piCopilotWorkspacePreviewUrl",
        "loadPiCopilotResearchArtifact",
        "loadPiCopilotDataPackageReview",
        "preparePiCopilotDataPackageReview",
    ):
        assert method in api
    assert "fetch(" not in pi_owner


def test_existing_project_study_setup_stays_in_bound_pi_conversation() -> None:
    owner = _read("js/screens-guided-pi.js")
    aside_owner = _read("js/screens-guided-pi-aside.js")

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
    assert "state.projectLoading = !!next" in owner
    assert ".finally(() =>" in owner
    assert "legacy 0/8 aside" in owner
    assert "data-gpi-project-workflow-loading" in aside_owner
    assert "Loading authoritative configuration…" in aside_owner
    assert "if (!workflow)" in aside_owner
    session_panel = owner[
        owner.index("function sessionPanel()") : owner.index("function demoPanel()")
    ]
    assert "data-gpi-legacy" not in session_panel


def test_scientific_review_continues_as_one_question_in_chat() -> None:
    owner = _read("js/screens-guided-pi.js")
    confirmation = _read("js/screens-guided-pi-confirmation.js")

    # The question catalogue and the card that surfaces it belong to the
    # confirmation owner...
    assert "Answer decision 1" in confirmation
    assert "localizedAuthorizationQuestion" in confirmation
    assert "OUTCOME_DEFINITION_UNRESOLVED" in confirmation
    assert "这项研究应使用哪个当前数据可支持的临床结局及时间范围？" in confirmation
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in confirmation
    assert "ADJUSTMENT_SET_NOT_USER_CONFIRMED" in confirmation
    assert "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED" in confirmation
    assert "回答第 1 项" in confirmation
    assert "review.authorization_questions" in confirmation
    # ...while composing and sending the one open question stays in the shell,
    # which is the only place a turn is actually sent.
    assert "localizedAuthorizationQuestion(questions[0])" in owner
    assert "请一次只问我一个尚未解决的科学设定问题" in owner


def test_pi_composer_enter_sends_without_breaking_shift_enter_or_ime() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    owner = STATIC / "js" / "composer-keyboard.js"
    script = r"""
global.window = global;
require(process.argv[1]);
const shouldSend = global.EU_COMPOSER_KEYBOARD.enterShouldSend;
const cases = {
  enter: shouldSend({key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13}),
  shiftEnter: shouldSend({key: 'Enter', shiftKey: true, isComposing: false, keyCode: 13}),
  composing: shouldSend({key: 'Enter', shiftKey: false, isComposing: true, keyCode: 13}),
  legacyIme: shouldSend({key: 'Enter', shiftKey: false, isComposing: false, keyCode: 229}),
  otherKey: shouldSend({key: 'a', shiftKey: false, isComposing: false, keyCode: 65}),
};
process.stdout.write(JSON.stringify(cases));
"""
    result = subprocess.run(
        [node, "-e", script, str(owner)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "enter": True,
        "shiftEnter": False,
        "composing": False,
        "legacyIme": False,
        "otherKey": False,
    }
    index = _read("index.html")
    pi_owner = _read("js/screens-guided-pi.js")
    resource_owner = _read("js/screens-guided-pi-resources.js")
    fallback_owner = _read("js/screens-guided.js")
    assert index.index("js/composer-keyboard.js") < index.index("js/screens-guided-pi.js")
    assert "EU_COMPOSER_KEYBOARD.enterShouldSend(event)" in pi_owner
    assert "EU_COMPOSER_KEYBOARD.enterShouldSend(e)" in fallback_owner


def test_assistant_markdown_renders_headings_and_lists_as_blocks() -> None:
    """Older turns leaked literal "###" and "- " to the screen.

    Only the newest reply is converted into host next-step buttons; every
    earlier turn kept its raw markers because the renderer handled inline
    formatting only.
    """

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner = Path(__file__).resolve().parents[1] / (
        "src/easyicu/webserver/static/js/screens-guided-pi-markdown.js"
    )
    script = r"""
global.window = { EU_HTML: { esc: (value) => String(value)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;') } };
require(process.argv[1]);
const render = window.EU_GUIDED_PI_MARKDOWN.render;
process.stdout.write(JSON.stringify({
  blocks: render(process.argv[2]),
  link: render('see [x](https://e.org/a) now'),
  unsafeLink: render('[x](javascript:alert(1))'),
  html: render('<img src=x onerror=alert(1)>'),
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(owner), "已提交。\n\n### 下一步\n- A\n- B"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    rendered = json.loads(result.stdout)
    assert rendered["blocks"] == (
        "<p>已提交。</p>"
        '<p class="gpi-md-heading">下一步</p>'
        '<ul class="gpi-md-list"><li>A</li><li>B</li></ul>'
    )
    assert "###" not in rendered["blocks"]
    assert '<a href="https://e.org/a"' in rendered["link"]
    # The renderer still refuses everything it refused before: a rejected URL
    # stays inert escaped text rather than becoming an anchor, and raw HTML is
    # never turned into DOM.
    assert "<a " not in rendered["unsafeLink"]
    assert "&lt;img" in rendered["html"] and "<img" not in rendered["html"]


def test_assistant_replies_drop_pre_wrap_but_user_turns_keep_it() -> None:
    """Block-rendered replies bring their own spacing; escaped user text does not."""

    css = _read("css/guided-pi.css")

    assert ".gpi-text{white-space:pre-wrap" in css
    assert ".gpi-message.assistant .gpi-text{white-space:normal}" in css
    assert ".gpi-text .gpi-md-heading{" in css
    assert ".gpi-text .gpi-md-list{" in css


def test_finished_activity_blocks_and_failures_collapse() -> None:
    """Historical failures must not unfold every diagnostic row on refresh."""

    owner = _read("js/screens-guided-pi-activity.js")

    terminal = owner[owner.index("const title = failed"):owner.index("function focusLatest(")]
    assert " open>" not in terminal
    # focusLatest must no longer re-open the newest finished turn either.
    focus = owner[owner.index("function focusLatest("):]
    focus = focus[: focus.index("\n    }")]
    assert "row.expanded = false" in focus
    assert "expanded = true" not in focus


def test_running_child_job_is_visually_busy_and_locks_new_messages() -> None:
    """A background plan job must not look like a finished transcript entry."""

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    activity_path = STATIC / "js" / "screens-guided-pi-activity.js"
    script = r"""
global.window = { EU_LANG: 'zh' };
require(process.argv[1]);
const activity = window.EU_GUIDED_PI_ACTIVITY.create({
  tr: (en, zh) => zh || en,
  esc: value => String(value),
  iconHtml: () => '',
  resourceName: () => '',
  resourceKey: () => '',
  resourceButton: () => '',
});
process.stdout.write(activity.render({
  role: 'activity', status: 'running', startedAt: Date.now() - 1500,
  runningTitle: '正在生成正式研究计划', steps: [],
}));
"""
    completed = subprocess.run(
        [node, "--eval", script, str(activity_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'aria-busy="true"' in completed.stdout
    assert "正在进行" in completed.stdout
    assert "正在生成正式研究计划" in completed.stdout
    assert "任务仍在进行" in completed.stdout
    assert "gpi-running-spinner" in completed.stdout

    shell = _read("js/screens-guided-pi.js")
    childjob = _read("js/screens-guided-pi-childjob.js")
    css = _read("css/guided-pi.css")
    assert "const activeChild = timeline.slice().reverse().find" in shell
    assert "const interactionLocked = state.busy || Boolean(activeChild)" in shell
    assert "busy: interactionLocked" in shell
    assert "gpi-compose-card${activeChild ? ' is-running' : ''}" in shell
    assert "任务完成或需要你确认后，才可继续发送消息" in shell
    assert "state.busy || state.childJobId || sessionIsStale()" in shell
    assert "runningTitle: runningJobTitle(code)" in childjob
    assert "正在生成正式研究计划" in childjob
    assert 'data-gpi-cancel-child-job="${esc(activeChild.childJobId)}"' in shell
    assert "停止生成" in shell
    assert "CHILDJOB.cancelChildJob(jobId)" in shell
    assert "api().cancelJob(requestedId, 'user_requested_from_copilot')" in childjob
    assert ".gpi-activity-running{" in css
    assert ".gpi-compose-card.is-running" in css
    assert ".gpi-compose-running>[data-gpi-cancel-child-job]" in css
    assert "@keyframes gpi-running-spin" in css


def test_running_child_job_stop_calls_real_cancel_api_once() -> None:
    """The visible control must cancel the server job, not fake a local state."""

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner_path = STATIC / "js" / "screens-guided-pi-childjob.js"
    script = r"""
global.window = {};
require(process.argv[1]);
const messages = [];
const calls = [];
let childJobId = 'job-running';
const host = {
  tr: (en, zh) => zh || en,
  activity: { pipelineEventLabel: event => event.step || event.type },
  upsertActivityStep: (activity, step) => activity.steps.push(step),
  render: () => calls.push('render'),
  api: () => ({
    cancelJob: async (jobId, reason) => {
      calls.push('cancel:' + jobId + ':' + reason);
      return { status: 'running', cancel_request_accepted: true };
    },
  }),
  loadWorkflow: async () => {}, sessionIsStale: () => false,
  rebind: async () => {}, refreshSession: async () => {}, archiveChildJob: async () => {},
  messages: () => messages, session: () => ({ session_id: 'pi-1' }),
  childJobId: () => childJobId, setChildJobId: value => { childJobId = value; },
  childSource: () => null, setChildSource: () => {},
};
const owner = window.EU_GUIDED_PI_CHILDJOB.create(host);
Promise.all([
  owner.cancelChildJob('job-running'),
  owner.cancelChildJob('job-running'),
]).then(results => process.stdout.write(JSON.stringify({
  results,
  calls,
  cancelRequested: messages[0] && messages[0].cancelRequested,
})));
"""
    result = subprocess.run(
        [node, "-e", script, str(owner_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["results"] == [True, False]
    assert payload["cancelRequested"] is True
    assert payload["calls"].count(
        "cancel:job-running:user_requested_from_copilot"
    ) == 1


def test_pending_review_activity_does_not_unfold_its_build_log() -> None:
    """The review card carries the decision; the log is 22 lines of noise."""

    owner = _read("js/screens-guided-pi-replay.js")

    presentation = owner[owner.index("function childJobPresentation("):]
    presentation = presentation[: presentation.index("terminalLabel")]
    assert "expanded: false," in presentation
    assert "expanded: reviewPending," not in presentation
    # The pending state is still announced, just in the collapsed summary.
    assert "Analysis plan ready for review" in owner


def test_new_reviewable_plan_hides_only_superseded_plan_attempts() -> None:
    """The main conversation shows the authoritative plan, not every retry."""

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    replay_owner = STATIC / "js" / "screens-guided-pi-replay.js"
    child_owner = STATIC / "js" / "screens-guided-pi-childjob.js"
    script = r"""
global.window = {};
require(process.argv[1]);
require(process.argv[2]);
const messages = [];
const host = {
  tr: (en, zh) => zh || en,
  activity: { pipelineEventLabel: event => event.step || event.type },
  upsertActivityStep: (activity, step) => {
    const index = activity.steps.findIndex(row => row.id === step.id);
    if (index >= 0) activity.steps[index] = step; else activity.steps.push(step);
  },
  render: () => {}, api: {}, loadWorkflow: async () => {},
  sessionIsStale: () => false, rebind: async () => {},
  refreshSession: async () => {}, archiveChildJob: async () => {},
  messages: () => messages, session: () => ({ session_id: 'pi-1' }),
  childJobId: () => '', setChildJobId: () => {},
  childSource: () => null, setChildSource: () => {},
};
const child = window.EU_GUIDED_PI_CHILDJOB.create(host);
const job = (job_id, created_at_epoch, extra = {}) => ({
  present: true, job_id, kind: 'agent-run', status: 'failed',
  created_at_epoch, artifact_refs: [], progress: [], ...extra,
});
child.hydrateProjectedJob(job('failed-old', 1));
child.hydrateProjectedJob(job('review-old', 2, {
  status: 'done', human_review_pending: true,
  gate_reason_code: 'human_plan_review_required', artifact_refs: [{artifact: 'agent_plan.json'}],
}));
child.hydrateProjectedJob(job('failed-current', 3));
const beforeReplacement = messages.map(row => row.childJobId);
child.hydrateProjectedJob(job('analysis-history', 3.5, {
  status: 'done', artifact_refs: [{artifact: 'result_tables.json'}],
}));
child.hydrateProjectedJob(job('review-current', 4, {
  status: 'done', human_review_pending: true,
  gate_reason_code: 'human_plan_review_required', artifact_refs: [{artifact: 'agent_plan.json'}],
}));
process.stdout.write(JSON.stringify({
  beforeReplacement,
  afterReplacement: messages.map(row => row.childJobId),
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(replay_owner), str(child_owner)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "beforeReplacement": ["review-old", "failed-current"],
        "afterReplacement": ["analysis-history", "review-current"],
    }


def test_workflow_confirmation_owner_is_split_out_and_read_only() -> None:
    """screens-guided-pi.js was 620 lines past its ratchet.

    The confirmation catalogue is the first seam taken out of it: it decides
    which confirmation a workflow state requires and how that card renders,
    and it does so without mutating host state or sending anything.
    """

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner_path = Path(__file__).resolve().parents[1] / (
        "src/easyicu/webserver/static/js/screens-guided-pi-confirmation.js"
    )
    script = r"""
global.window = {};
require(process.argv[1]);
const host = {
  tr: (en, zh) => zh || en, esc: (v) => String(v), iconHtml: () => '',
  resourceButton: () => '<button/>', sessionIsStale: () => false,
  workflow: () => ({ next_action_code: 'provider_ready_to_generate_plan' }),
  session: () => ({ binding: { run_id: 'r1' } }), busy: () => false,
};
const made = window.EU_GUIDED_PI_CONFIRMATION.create(host);
const ready = made.workflowConfirmation();
const busy = window.EU_GUIDED_PI_CONFIRMATION.create(
  Object.assign({}, host, { busy: () => true })).workflowConfirmationHtml();
const unknown = window.EU_GUIDED_PI_CONFIRMATION.create(
  Object.assign({}, host, { workflow: () => ({ next_action_code: 'other' }) })
).workflowConfirmation();
process.stdout.write(JSON.stringify({
  code: ready.code,
  grants: ready.grants,
  message: ready.message,
  rendersCard: made.workflowConfirmationHtml().indexOf('gpi-confirmation') >= 0,
  silentWhileBusy: busy === '',
  nullForUnknownState: unknown === null,
  exposesLocalizer: typeof made.localizedAuthorizationQuestion === 'function',
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(owner_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "code": "provider_ready_to_generate_plan",
        "grants": ["provider_run", "literature"],
        "message": "开始生成正式研究计划。",
        "rendersCard": True,
        "silentWhileBusy": True,
        "nullForUnknownState": True,
        "exposesLocalizer": True,
    }

    shell = _read("js/screens-guided-pi.js")
    owner = _read("js/screens-guided-pi-confirmation.js")
    # The shell keeps the mount and the send; the owner keeps the catalogue.
    assert "window.EU_GUIDED_PI_CONFIRMATION.create({" in shell
    assert "function workflowConfirmation()" not in shell
    assert "function workflowConfirmation()" in owner
    assert "请基于已确认的研究问题和 EasyICU 数据库能力目录" not in owner
    assert "confirm_formal_plan_generation" in shell
    # Read-only by contract: no host mutation, no transport, no grant spend.
    for forbidden in ("state.", "sendText", "api()", "render()"):
        assert forbidden not in owner, f"{forbidden} does not belong in this owner"
    index = _read("index.html")
    assert index.index("screens-guided-pi-confirmation.js") < index.index(
        "js/screens-guided-pi.js?"
    )


def test_failed_plan_review_links_to_last_reviewable_run_not_failed_binding() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner_path = Path(__file__).resolve().parents[1] / (
        "src/easyicu/webserver/static/js/screens-guided-pi-confirmation.js"
    )
    script = r"""
global.window = {};
require(process.argv[1]);
const host = {
  tr: (en, zh) => zh || en, esc: (v) => String(v), iconHtml: () => '',
  resourceButton: () => '<button/>', sessionIsStale: () => false,
  workflow: () => ({ next_action_code: 'failed_pipeline_requires_fresh_plan' }),
  session: () => ({
    binding: { run_id: 'run_failed' },
    archived_child_jobs: [
      {kind: 'agent-run', status: 'done', run_id: 'run_reviewable', artifact_refs: [
        {artifact: 'agent_plan.json', run_id: 'run_reviewable'},
        {artifact: 'literature_evidence.json', run_id: 'run_reviewable'},
      ]},
      {kind: 'agent-run', status: 'failed', error_code: 'research_pipeline_plan_contract_exhausted'},
    ],
  }),
  busy: () => false,
};
const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create(host).workflowConfirmation();
process.stdout.write(JSON.stringify({
  title: confirmation.title,
  grants: confirmation.grants,
  runIds: confirmation.reviewResources.map(row => row.run_id),
  artifacts: confirmation.reviewResources.map(row => row.artifact),
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(owner_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["title"] == "修订版计划未通过科学合同"
    assert payload["grants"] == ["provider_run", "literature"]
    assert payload["runIds"] == ["run_reviewable"] * 6
    assert payload["artifacts"][:2] == ["agent_plan.json", "literature_evidence.json"]


def test_legacy_shell_does_not_overwrite_the_copilot_authority_panel() -> None:
    """Two workflow models shared one panel, and the wrong one kept winning.

    The Copilot owner writes the bound 7-stage workflow into #gdAsideBody; the
    legacy Guided shell re-rendered its own 8-step STUDY model into the same
    node from twelve call sites. A panel headed "项目权威状态" therefore read
    "0/8 · next: data source" while the bound workflow was 3/7 with extraction
    already complete.
    """

    guided = _read("js/screens-guided.js")
    pi_owner = _read("js/screens-guided-pi.js")
    aside_owner = _read("js/screens-guided-pi-aside.js")

    aside_writer = guided[guided.index("function renderAside()"):]
    aside_writer = aside_writer[: aside_writer.index("\n  }")]
    assert "if (!host || piProjectShellActive()) return;" in aside_writer
    # The Copilot owner remains the writer while its shell is mounted.
    assert "function syncProjectWorkflowAside()" in aside_owner
    assert "document.getElementById('gdAsideBody')" in aside_owner
    assert "host.shell() !== 'pi'" in aside_owner


def test_pipeline_progress_is_composed_not_echoed() -> None:
    """The runner writes English prose; the UI must not print it verbatim.

    A live run showed 20 rows, every one English inside a Chinese UI and most
    carrying an internal step id ("Step 10/13 started: assemble_visual_displays.").
    The same event already carries the structured fields that line was built
    from, so compose it in the UI language instead.
    """

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner = Path(__file__).resolve().parents[1] / (
        "src/easyicu/webserver/static/js/screens-guided-pi-activity.js"
    )
    script = r"""
global.window = {};
require(process.argv[1]);
const mk = (tr) => window.EU_GUIDED_PI_ACTIVITY.create({
  tr, esc: (v) => String(v), iconHtml: () => '',
  resourceName: () => '', resourceKey: () => '', resourceButton: () => '',
});
const zh = mk((en, z) => z || en);
const en = mk((e) => e);
const noisy = { type: 'progress', step: 'assemble_visual_displays', current: 10,
  total: 13, label: 'Step 10/13 started: assemble_visual_displays.' };
process.stdout.write(JSON.stringify({
  zh: zh.pipelineEventLabel(noisy),
  en: en.pipelineEventLabel(noisy),
  numberedPrefixStripped: zh.pipelineEventLabel(
    { type: 'progress', step: '13_data_quality_figure', current: 13, total: 13 }),
  planningRetryTwo: zh.pipelineEventLabel(
    { type: 'progress', step: 'planning', current: 2, total: 3 }),
  planningRetryOne: zh.pipelineEventLabel(
    { type: 'progress', step: 'planning', current: 1, total: 3 }),
  countOnly: zh.pipelineEventLabel({ type: 'progress', current: 4, total: 9 }),
  started: zh.pipelineEventLabel({ type: 'start' }),
  fallback: zh.pipelineEventLabel({ type: 'progress' }),
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(owner)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    rendered = json.loads(result.stdout)
    assert rendered["zh"] == "第 10/13 步 · assemble visual displays"
    assert rendered["en"] == "Step 10/13 · assemble visual displays"
    # The runner's own sentence never reaches the screen.
    assert "started:" not in rendered["zh"]
    assert "_" not in rendered["zh"]
    assert rendered["numberedPrefixStripped"] == "第 13/13 步 · data quality figure"
    assert rendered["planningRetryTwo"] == "正在生成研究计划的组成部分"
    assert rendered["planningRetryOne"] == "正在生成研究计划的组成部分"
    assert "1/3" not in rendered["planningRetryOne"]
    assert "2/3" not in rendered["planningRetryTwo"]
    assert rendered["countOnly"] == "第 4/9 步"
    assert rendered["started"] == "EasyICU 科研流程已启动"
    assert rendered["fallback"] == "EasyICU 科研流程已更新"

    shell = _read("js/screens-guided-pi.js")
    childjob_owner = _read("js/screens-guided-pi-childjob.js")
    assert "ACTIVITY.pipelineEventLabel(event)" in childjob_owner
    assert "if (event.label) return String(event.label);" not in shell
    # One row per step, updated in place, instead of one row per event.
    assert "id: 'pipeline-' + step," in childjob_owner
    assert "String(event.seq == null ? step : event.seq)" not in shell


def test_child_job_terminal_event_archives_and_refreshes_without_reload() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner_path = STATIC / "js" / "screens-guided-pi-childjob.js"
    script = r"""
global.window = {};
require(process.argv[1]);
const messages = [];
const calls = [];
let childJobId = 'job-1';
let childSource = { close: () => calls.push('close') };
const host = {
  tr: (en, zh) => zh || en,
  activity: { pipelineEventLabel: event => event.label || event.step || event.type },
  upsertActivityStep: (activity, step) => activity.steps.push(step),
  render: () => calls.push('render'),
  api: {},
  loadWorkflow: async () => { calls.push('workflow'); },
  sessionIsStale: () => false,
  rebind: async () => { calls.push('rebind'); },
  refreshSession: async () => { calls.push('session'); },
  archiveChildJob: async jobId => { calls.push('archive:' + jobId); },
  messages: () => messages,
  session: () => ({ session_id: 'pi-1' }),
  childJobId: () => childJobId,
  setChildJobId: value => { childJobId = value; },
  childSource: () => childSource,
  setChildSource: value => { childSource = value; },
};
const owner = window.EU_GUIDED_PI_CHILDJOB.create(host);
owner.handleChildJobEvent('job-1', 'easyicu_full_run_submitted', {
  type: 'end', status: 'done', error: null,
  result: { run_id: 'run-1', human_review_pending: true, gate: { status: 'blocked' } },
});
setTimeout(() => process.stdout.write(JSON.stringify({
  calls,
  status: messages[0] && messages[0].status,
  terminal: messages[0] && messages[0].steps.some(step => step.id === 'pipeline-terminal'),
})), 20);
"""
    result = subprocess.run(
        [node, "-e", script, str(owner_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["terminal"] is True
    assert "archive:job-1" in payload["calls"]
    assert "session" in payload["calls"]
    assert "workflow" in payload["calls"]
    assert "render" in payload["calls"]


def test_blocked_plan_job_is_not_presented_as_completed() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    replay_owner = STATIC / "js" / "screens-guided-pi-replay.js"
    child_owner = STATIC / "js" / "screens-guided-pi-childjob.js"
    script = r"""
global.window = {};
require(process.argv[1]);
require(process.argv[2]);
const messages = [];
let childJobId = 'job-blocked';
let childSource = { close: () => {} };
const host = {
  tr: (en, zh) => zh || en,
  activity: { pipelineEventLabel: event => event.step || event.type },
  upsertActivityStep: (activity, step) => {
    const index = activity.steps.findIndex(row => row.id === step.id);
    if (index >= 0) activity.steps[index] = step; else activity.steps.push(step);
  },
  render: () => {}, api: {}, loadWorkflow: async () => {},
  sessionIsStale: () => false, rebind: async () => {},
  refreshSession: async () => {}, archiveChildJob: async () => {},
  messages: () => messages, session: () => ({ session_id: 'pi-1' }),
  childJobId: () => childJobId, setChildJobId: value => { childJobId = value; },
  childSource: () => childSource, setChildSource: value => { childSource = value; },
};
window.EU_GUIDED_PI_CHILDJOB.create(host).handleChildJobEvent(
  'job-blocked', 'easyicu_full_run_submitted', {
    type: 'end', status: 'done',
    result: {
      run_id: 'run-blocked', human_review_pending: false,
      gate: { status: 'blocked', reason: 'data_foundation_blocked', reportable: false },
    },
  });
const terminal = messages[0].steps.find(step => step.id === 'pipeline-terminal');
process.stdout.write(JSON.stringify({
  status: messages[0].status,
  title: messages[0].displayTitle,
  label: terminal.label,
  terminalStatus: terminal.status,
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(replay_owner), str(child_owner)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "status": "blocked",
        "title": "研究计划未生成",
        "label": "研究计划未生成：数据准备未通过",
        "terminalStatus": "error",
    }


def test_planner_checkpoint_is_presented_once_as_resumable_not_failed() -> None:
    """A bounded Planner turn is a saved checkpoint, not a scientific failure."""

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    replay_owner = STATIC / "js" / "screens-guided-pi-replay.js"
    child_owner = STATIC / "js" / "screens-guided-pi-childjob.js"
    script = r"""
global.window = {};
require(process.argv[1]);
require(process.argv[2]);
const messages = [];
const host = {
  tr: (en, zh) => zh || en,
  activity: { pipelineEventLabel: event => event.step || event.type },
  upsertActivityStep: (activity, step) => {
    const index = activity.steps.findIndex(row => row.id === step.id);
    if (index >= 0) activity.steps[index] = step; else activity.steps.push(step);
  },
  render: () => {}, api: {}, loadWorkflow: async () => {},
  sessionIsStale: () => false, rebind: async () => {},
  refreshSession: async () => {}, archiveChildJob: async () => {},
  messages: () => messages, session: () => ({ session_id: 'pi-1' }),
  childJobId: () => '', setChildJobId: () => {},
  childSource: () => null, setChildSource: () => {},
};
const child = window.EU_GUIDED_PI_CHILDJOB.create(host);
child.hydrateProjectedJob({
  present: true, job_id: 'job-checkpoint', kind: 'agent', status: 'failed',
  error_code: 'research_pipeline_planner_efficiency_budget_exhausted',
  progress: [{ type: 'progress', step: 'planning', label: 'planning' }],
});
child.hydrateProjectedJob({
  present: true, job_id: 'job-checkpoint', kind: 'agent', status: 'failed',
  error_code: 'research_pipeline_planner_efficiency_budget_exhausted',
  progress: [{ type: 'progress', step: 'planning', label: 'planning' }],
});
const terminals = messages[0].steps.filter(step => step.step === 'terminal');
process.stdout.write(JSON.stringify({
  status: messages[0].status,
  expanded: messages[0].expanded,
  title: messages[0].displayTitle,
  label: terminals[0].label,
  terminalCount: terminals.length,
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(replay_owner), str(child_owner)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "status": "blocked",
        "expanded": False,
        "title": "规划器已保存验证检查点",
        "label": "已保存验证检查点；可继续完成研究计划",
        "terminalCount": 1,
    }


def test_planner_compile_failure_is_not_presented_as_a_continue_checkpoint() -> None:
    """A compiler failure is a failure, never a normal user pause."""

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    replay_owner = STATIC / "js" / "screens-guided-pi-replay.js"
    script = r"""
global.window = {};
require(process.argv[1]);
const presentation = window.EU_GUIDED_PI_REPLAY.childJobPresentation({
  status: 'failed',
  error_code: 'research_pipeline_progressive_compile_failed',
}, (en, zh) => zh || en);
process.stdout.write(JSON.stringify(presentation));
"""
    result = subprocess.run(
        [node, "-e", script, str(replay_owner)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    presentation = json.loads(result.stdout)
    assert presentation["blocked"] is False
    assert presentation["title"] == ""
    assert presentation["terminalLabel"] == ""


def test_next_step_card_offers_no_generic_continue_button() -> None:
    """"继续对话" only moved the cursor into the composer beneath it.

    The shared prompt already forbids offering a generic continue action; this
    makes the UI stop rendering one when the model supplied no choices.
    """

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const N = window.EU_GUIDED_PI_NEXT_ACTIONS;
      const opts = {{ language: 'zh', workflowActionCode: 'analysis_ready',
        dataSourceAuthorization: {{ status: 'confirmed', source: {{ label: 'X' }} }} }};
      const render = (raw) => N.render(N.project(raw), opts) || '';
      const noChoice = render(process.argv[1]);
      const asking = render(process.argv[2]);
      const choices = render(process.argv[3]);
      console.log(JSON.stringify({{
        noChoiceHasContinue: noChoice.indexOf('继续对话') >= 0,
        noChoiceHasEmptyActions: noChoice.indexOf('gpi-next-actions') >= 0,
        noChoiceKeepsNote: noChoice.indexOf('gpi-next-step') >= 0,
        askingKeepsAnswerButton: asking.indexOf('回答这个问题') >= 0,
        choicesStillRender: choices.indexOf('使用 ICU 死亡作为结局') >= 0,
      }}));
    """
    completed = subprocess.run(
        [
            node,
            "--eval",
            script,
            "EasyICU 将继续执行规划。\n**下一步：**\n继续等待计划完成。",
            "要用哪个结局？",
            # Deliberately not a formal-plan choice: that path belongs to
            # the premature-plan guard, which this test is not about.
            "好的。\n**下一步：**\n- 使用 ICU 死亡作为结局\n- 使用住院死亡",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "noChoiceHasContinue": False,
        # No affordance at all rather than an empty control row.
        "noChoiceHasEmptyActions": False,
        # The note itself still renders; it says what happens next.
        "noChoiceKeepsNote": True,
        # A direct question keeps its button: that label does carry meaning.
        "askingKeepsAnswerButton": True,
        "choicesStillRender": True,
    }
    assert "'继续对话'" not in owner


def test_every_assistant_turn_projects_its_next_step_block() -> None:
    """One screen carried seven "下一步" headings and one live card.

    Only the newest turn was projected, so older turns rendered their own block
    as raw markdown: four bullet lists that read as offers but could not be
    clicked, plus three that were really progress narration wearing the label
    of a next step.
    """

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const N = window.EU_GUIDED_PI_NEXT_ACTIONS;
      const offer = N.project(process.argv[1]);
      const narration = N.project(process.argv[2]);
      console.log(JSON.stringify({{
        offerBody: N.bodyText(offer),
        offerPast: N.renderPast(offer, 'zh'),
        narrationBody: N.bodyText(narration),
        narrationPast: N.renderPast(narration, 'zh'),
      }}));
    """
    completed = subprocess.run(
        [
            node,
            "--eval",
            script,
            "好的。\n**下一步：**\n- 使用 ICU 死亡\n- 使用住院死亡",
            "已提交。\n**下一步：**\nEasyICU 将继续执行规划，并在审核闸门处暂停。",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(completed.stdout)

    # An earlier turn's offer becomes history, not a dead lookalike of the card.
    assert out["offerBody"] == "好的。"
    assert "当时提供的选项" in out["offerPast"]
    assert "is-past" in out["offerPast"]
    assert "<button" not in out["offerPast"]
    # Narration is not a next step at all: fold it back into ordinary prose.
    assert out["narrationBody"] == (
        "已提交。\n\nEasyICU 将继续执行规划，并在审核闸门处暂停。"
    )
    assert out["narrationPast"] == ""

    shell = _read("js/screens-guided-pi.js")
    projection = shell[shell.index("const nextStep = row.role === 'assistant'"):]
    projection = projection[: projection.index("const messageActions")]
    assert "const interactive = Boolean(options && options.interactive);" in projection
    assert "nextOwner.renderPast(nextStep, window.EU_LANG)" in projection
    assert "nextOwner.bodyText(nextStep)" in projection


def test_type_scale_lifts_the_small_end_of_the_copilot_ui() -> None:
    """The action card was the smallest text on screen at 11.5px.

    CJK glyphs need more size than Latin for the same legibility, and this is a
    card the reader has to act on. These lock the floor rather than every size.
    """

    pi_css = _read("css/guided-pi.css")
    study_css = _read("css/guided.css")

    assert ".gpi-next-step>p{margin:3px 0 0;color:var(--ink-3);font-size:13px" in pi_css
    assert "font-size:11.5px" not in pi_css.split(".gpi-next-step")[1][:400]
    # The panel that states the bound workflow.
    assert ".gd-aside-head .at{ font-size: 16.5px" in study_css
    # The transcript itself.
    assert ".gpi-text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:16.5px" in pi_css


def test_workflow_stage_list_is_open_by_default() -> None:
    """The panel measured 984 of 1181 px empty with the stage list collapsed."""

    owner = _read("js/screens-guided-pi-aside.js")

    assert '<details class="gd-pipeline-disclosure" open><summary>' in owner


def test_conversation_header_does_not_print_the_same_name_twice() -> None:
    owner = _read("js/screens-guided-pi-header.js")

    assert (
        "if (!project || (session && session.indexOf(project) >= 0)) "
        "return 'EASYICU COPILOT';"
    ) in owner
    assert "EASYICU COPILOT · ${esc(options.projectTitle)}" not in owner
    # The header squeezes the title to ellipsis on a 1280 laptop; keep the
    # full name reachable rather than unrecoverable.
    assert '<div class="gpi-title" title="${esc(options.sessionTitle)}">' in owner


def test_workflow_strip_leaves_the_stage_count_to_the_status_panel() -> None:
    """Three renderings of 3/7 on one screen; the pips already show position."""

    owner = _read("js/screens-guided-pi.js")
    aside_owner = _read("js/screens-guided-pi-aside.js")

    strip = owner[owner.index('<div class="gpi-workflow-meta">'):]
    strip = strip[: strip.index("</div>")]
    assert 'class="shell-sr-only"' in strip
    # The authoritative panel still states it in words.
    assert "required stages complete" in aside_owner


def test_agent_handoff_receipt_is_forwarded_to_project_initialization() -> None:
    owner = _read("js/screens-guided-pi.js")
    project_owner = _read("js/screens-guided-pi-project.js")
    guided = _read("js/screens-guided.js")

    assert "binding_receipt: bindingReceipt || undefined" in project_owner
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
            str(STATIC / "js" / "product-labels.js"),
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
    assert ".gpi-activity-kicker" in owner
    assert ".gpi-activity-step-copy>span" in owner
    assert ".gpi-access-menu" in owner
    assert ".gpi-confirmation" in owner
    assert ".gpi-confirmation-resources>summary" in owner
    assert ".gpi-confirmation-resources>div" in owner
    assert ".gpi-grants" not in owner
    assert ".gpi-workflow ol{display:flex" in owner
    assert "pi-gui's MIT-licensed timeline-item/timeline.css" in owner
    assert ".gpi-research-start{max-width:900px" in owner
    assert ".gpi-starter-actions strong{font-size:15.5px" in owner
    assert ".gpi-message{max-width:860px" in owner
    assert (
        ".gpi-activity,.gpi-activity-live,.gpi-activity-running{max-width:860px"
        in owner
    )
    assert ".gpi-preview-aside" in preview_owner
    assert ".gpi-preview-frame" in preview_owner
    assert ".gpi-preview-code" in preview_owner
    assert ".gpi-preview-provenance" in preview_owner
    assert ".gpi-preview-recent" in preview_owner
    assert ".gpi-resource-list" in owner
    assert ".gpi-lit-card" in literature_owner
    assert ".gpi-lit-step" in literature_owner
    assert ".gpi-lit-history" in literature_owner
    assert ".gpi-lit-coverage" in literature_owner
    assert ".gpi-lit-reporting" in literature_owner
    assert ".gpi-demo-note" in demo_owner
    assert ".gpi-demo-footer" in demo_owner
    assert ".gpi-demo-artifact" not in demo_owner
    assert ".gpi-demo-reviewer" not in demo_owner
    assert "research-artifact preview" in preview_owner
    assert ".gpi-tool" not in owner
    assert "gpi-avatar" not in owner
    assert ".gd-conv.pi-active" in owner
    assert ".gd-main.threecol.gpi-setup-focus" in owner
    assert ".gd-main.threecol.gpi-setup-focus>.gd-aside{display:none}" in owner
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


def test_guided_shell_readability_changes_stay_with_their_css_owners() -> None:
    pi_css = _read("css/guided-pi.css")
    starters = _read("js/screens-guided-pi-starters.js")
    projects_css = _read("css/guided-projects.css")
    study_css = _read("css/guided.css")
    catch_all = _read("css/redesign.css") + _read("css/tweaks.css")

    assert ".gpi-research-start h2" in pi_css and "font-size:28px" in pi_css
    assert ".gpi-compose-card" in pi_css and "width:min(900px,100%)" in pi_css
    assert ".gpi-compose textarea" in pi_css and "width:100%" in pi_css
    assert ".gpi-starter-actions button.primary" in pi_css
    assert "→" not in starters
    assert ".gpi-starter-actions b{" not in pi_css
    assert ".gpi-head-overflow-menu" in pi_css
    assert ".gd-main.threecol.gpi-empty-session-focus>.gd-aside{display:none}" in pi_css
    assert ".gpi-panel.gpi-empty-session .gpi-workflow{display:none}" in pi_css
    assert ".gpi-panel.gpi-empty-session .gpi-compose" not in pi_css
    assert ".gpi-panel.gpi-empty-session .gpi-log-start{flex:0" not in pi_css
    assert ".gd-sess .ss-t" in projects_css and "font-size:15.5px" in projects_css
    assert "grid-template-columns:292px minmax(0,1fr) 292px" in projects_css
    assert ".gd-aside-head .at{ font-size: 16.5px" in study_css
    assert ".study-item .si-t{ font-size: 14.75px" in study_css
    assert ".gd-pipeline-disclosure" in study_css
    for selector in (".gpi-research-start", ".gd-sess .ss-t", ".study-item .si-t"):
        assert selector not in catch_all


def test_model_connection_setup_owns_the_temporary_focus_layout() -> None:
    owner = _read("js/screens-guided-pi.js")
    provider = _read("js/screens-guided-pi-provider.js")
    projects_css = _read("css/guided-projects.css")

    assert "main.classList.toggle('gpi-setup-focus', setupFocused)" in owner
    assert "state.showSetup || !connectionReady()" in owner
    assert "Finish connection setup" in provider
    assert "完成连接设置" in provider
    assert "gpi-setup-focus" not in projects_css


def test_empty_research_session_prioritizes_the_conversation_entry() -> None:
    owner = _read("js/screens-guided-pi.js")
    projects_css = _read("css/guided-projects.css")

    assert "const emptyResearch = !workspace && !messages" in owner
    assert "gpi-empty-session'" in owner
    assert "main.classList.toggle('gpi-empty-session-focus', emptySessionFocused)" in owner
    assert "state.messages.length === 0 && state.workflowReceipts.length === 0" in owner
    assert "gpi-empty-session-focus" not in projects_css


def test_scientific_plan_review_has_a_readable_multidimensional_preview() -> None:
    renderer = _read("js/screens-agent-render.js")
    review_css = _read("css/agent-scientific-review.css")
    agent_css = _read("css/agent.css")
    index = _read("index.html")

    assert "scientific_plan_review.json" in renderer
    assert "function scientificPlanReviewView(payload)" in renderer
    assert "if (n === 'scientific_plan_review.json') return scientificPlanReviewView(p);" in renderer
    assert "Do this now" in renderer
    assert "No action needed now" in renderer
    assert "Methods references" in renderer
    assert "Raw scores, finding codes, and digest-bound details remain available in the JSON audit view." in renderer
    assert "Top-journal plan scorecard" not in renderer
    assert "literature_design_bindings" in renderer
    assert ".ag-science-review-hero" in review_css
    assert ".ag-science-current-question" in review_css
    assert ".ag-science-review-details" in review_css
    assert ".ag-science-lanes" in review_css
    assert ".ag-science-review" not in agent_css
    assert "css/agent-scientific-review.css?v=20260828-science-review3" in index
    assert "css/agent-plan.css?v=20260829-plan-flow1" in index
    confirmation_owner = _read("js/screens-guided-pi-confirmation.js")
    assert "plan_review_summary" in confirmation_owner
    assert "gpi-confirmation-review-status" in confirmation_owner
    assert "authorization_questions" in confirmation_owner
    assert "remediation_buckets" in confirmation_owner
    assert "Generate revised candidate plan" in confirmation_owner
    assert "View the plan and references" in confirmation_owner
    assert "gpi-confirmation-scorecard" not in confirmation_owner
    assert "Scientific review ${review.score" not in confirmation_owner


def test_scientific_plan_review_defaults_to_actions_not_scores() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")
    payload = {
        "score": 68,
        "approval_allowed": False,
        "findings": [
            {
                "code": "OUTCOME_DEFINITION_UNRESOLVED",
                "remediation_route": "study_authority_change",
                "requires_user_authorization": True,
            },
            {
                "code": "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
                "remediation_route": "study_authority_change",
                "requires_user_authorization": True,
            },
            {
                "code": "FIGURE_ROLE_COVERAGE_INCOMPLETE",
                "remediation_route": "agent_plan_revision",
            },
            {
                "code": "NOVELTY_NOT_ESTABLISHED",
                "remediation_route": "external_evidence",
            },
        ],
        "facts": {
            "literature_design_bindings": {
                "steps": [
                    {
                        "citations": [
                            {
                                "citation_key": "strobe",
                                "title": "STROBE statement",
                                "year": "2007",
                                "application": "Report the observational design clearly.",
                            }
                        ]
                    }
                ]
            }
        },
    }
    script = f"""
global.window = {{
  EU_HTML: {{
    esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    escAttr: value => String(value ?? ''),
  }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView(
  'scientific_plan_review.json',
  {json.dumps(payload)},
));
"""
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    html = result.stdout

    assert "EasyICU 需要修订这份候选计划" in html
    assert "现在只做这一步" not in html
    assert "在左侧对话中点击「回答第 1 项」" not in html
    assert "现在不需你处理" in html
    assert "可选查看" in html
    assert "Planner 将补全主要结局定义" in html
    assert "Planner 将提出敏感性分析" in html
    assert "补齐数据质量与分布图" in html
    assert "核对研究创新性" in html
    assert "原始评分、问题代码和摘要绑定细节仍保留在 JSON 审计视图中" in html
    assert "68 / 100" not in html
    assert "OUTCOME_DEFINITION_UNRESOLVED" not in html
    assert html.count("<details class=\"ag-science-review-details\"") == 2


def test_candidate_agent_plan_defaults_to_researcher_reading_order() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")
    payload = {
        "analysis_type": "data_quality_audit",
        "research_question": "ICU 患者的乳酸水平与院内死亡是否相关？",
        "display_labels": {"lact": "乳酸", "death": "院内死亡", "age": "年龄"},
        "endpoint": None,
        "robustness_specs": [],
        "rationale": "当前候选版需要补齐可执行结局与敏感性分析。",
        "design_selection": {
            "candidates": [
                {
                    "disposition": "selected",
                    "analysis_type": "data_quality_audit",
                    "decision_reason": "先核对数据可用性。",
                    "estimand": "描述测量覆盖。",
                    "time_zero": "ICU 入科。",
                    "observation_window": "待计划明确。",
                    "primary_method": "描述性审计。",
                    "supports": "支持数据可行性判断。",
                    "cannot_prove": "不能回答乳酸与死亡的关联。",
                    "required_variables": ["stay_id", "lact", "death", "age"],
                    "literature_citation_keys": ["strobe_2007"],
                }
            ]
        },
        "steps": [
            {
                "step_id": "cohort_accounting",
                "intent": "明确分析分母。",
                "expected_outputs": ["table:cohort_flow"],
            },
            {
                "step_id": "04_publication_figure_fallback",
                "method": "visualization",
                "intent": "Render a publication-ready overview " + ("x" * 300),
                "expected_outputs": ["figure:overview"],
            },
            {
                "step_id": "cohort_flow_figure",
                "method": "visualization",
                "intent": "Render the exact cohort-accounting table using its registered contract.",
                "expected_outputs": ["figure:cohort_flow"],
            },
        ],
    }
    script = f"""
global.window = {{
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
    html = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout

    for text in (
        "候选研究计划",
        "设计选择",
        "这套设计能回答",
        "这套设计不能证明",
        "候选计划涉及的变量",
        "分析路径",
        "EasyICU 需要修订",
        "主要结局尚缺可执行定义",
        "敏感性分析方案尚未形成",
        "乳酸",
        "院内死亡",
        "队列流程",
        "研究概览图",
        "根据已登记的队列流程表绘制纳入与排除流程图",
        "已绑定 1 篇计划依据",
    ):
        assert text in html
    assert "data_quality_audit" not in html
    assert "cohort_accounting" not in html
    assert "04_publication_figure_fallback" not in html
    assert "Render a publication-ready overview" not in html
    assert "Render the exact cohort-accounting table" not in html
    assert "可读产物摘要" not in html
    assert "工作流步骤" not in html
    assert "strobe 2007" not in html


def test_plan_step_notes_never_assert_another_study_subject() -> None:
    """A step note must describe the method, not the study. Keying canned
    prose off step_id made every plan that reused a step id claim the same
    exposure and outcome, so a ventilation study rendered as a lactate study.
    The plan's own wording is authoritative; when it is not in the reader's
    language the note falls back to a variable-free method description and
    the original wording is kept verbatim next to it."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")

    payload = {
        "analysis_type": "association_study",
        "research_question": "机械通气时长与 ICU 转出去向的关系",
        "design_selection": {"candidates": [{"disposition": "selected"}]},
        "steps": [
            {"step_id": "measurement_audit", "method": "missing_data", "intent": "Audit ventilation-duration coverage.", "expected_outputs": ["table:m"]},
            {"step_id": "primary_adjusted_association", "method": "logistic", "intent": "Estimate the association.", "expected_outputs": ["table:p"]},
            {"step_id": "robustness_replay", "method": "robustness_sensitivity", "intent": "Replay.", "expected_outputs": ["table:r"]},
            {"step_id": "cohort_accounting", "method": "cohort_definition_and_attrition", "intent": "", "expected_outputs": ["table:cohort_flow"]},
        ],
    }
    script = f"""
global.window = {{
  EU_LANG: 'zh',
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
    html = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout

    # no variable, outcome, score, or database from any other study appears
    for foreign in ("乳酸", "院内死亡", "lactate", "SOFA", "sepsis", "MIMIC", "eICU"):
        assert foreign not in html, foreign
    # method-family notes stand in, and they name no variable
    assert "检查本计划所需变量的测量覆盖与缺失情况" in html
    assert "按预先设定的调整变量估计校正后关联" in html
    assert "按预先规定的敏感性设定复核主要结论是否稳健" in html
    # a step with no stated intent still gets a true method description
    assert "统计纳入与排除的记录数，确定最终分析分母" in html
    # the plan's own wording is preserved, not discarded
    assert "计划原文" in html
    for stated in ("Audit ventilation-duration coverage.", "Estimate the association.", "Replay."):
        assert stated in html


def test_plan_reader_prefers_the_owner_compiled_phase() -> None:
    """The run phase is study semantics, so the reader reads the value the
    planning layer compiled (`planned_phase`, stamped by the Copilot artifact
    projection) instead of rebuilding it from method names. The local
    heuristic survives only for payloads that carry no compiled phase -- demo
    fixtures, hand-built previews, an artifact served by an older host."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")

    def stages(steps: list[dict]) -> dict[str, list[str]]:
        payload = {
            "research_question": "Q",
            "design_selection": {"candidates": [{"disposition": "selected"}]},
            "steps": steps,
        }
        script = f"""
global.window = {{
  EU_LANG: 'zh',
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
        html = subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout
        found: dict[str, list[str]] = {}
        for block in html.split('<li class="ag-plan-flow-stage is-')[1:]:
            found[block[: block.index('"')]] = re.findall(r"<b>(\d+)</b>", block)
        return found

    # every compiled phase maps to a stage
    compiled = stages([
        {"step_id": "a", "method": "x", "planned_phase": "cohort"},
        {"step_id": "b", "method": "x", "planned_phase": "data_check"},
        {"step_id": "c", "method": "x", "planned_phase": "analysis"},
        {"step_id": "d", "method": "x", "planned_phase": "robustness"},
        {"step_id": "e", "method": "x", "planned_phase": "reporting"},
        {"step_id": "f", "method": "x", "planned_phase": "support"},
    ])
    assert compiled == {
        "population": ["1"],
        "quality": ["2"],
        "primary": ["3"],
        "robustness": ["4"],
        "figure": ["5"],
        "support": ["6"],
    }

    # the compiled phase wins over both the method text and the role, because
    # the layer that compiled it saw more than either
    overridden = stages([
        {
            "step_id": "g",
            "method": "table_one_baseline_audit",
            "planned_analysis_role": "auxiliary",
            "planned_phase": "analysis",
        }
    ])
    assert overridden == {"primary": ["1"]}

    # no compiled phase: the local fallback still refuses to promote an
    # auxiliary step into a result
    legacy = stages([
        {
            "step_id": "h",
            "method": "descriptive_counts_of_exposure_outcome",
            "planned_analysis_role": "auxiliary",
        }
    ])
    assert legacy == {"support": ["1"]}

    # an unknown phase value is ignored rather than trusted blindly
    unknown = stages([
        {"step_id": "i", "method": "cohort_definition_and_attrition", "planned_phase": "nonsense"}
    ])
    assert unknown == {"population": ["1"]}


def test_plan_stage_placement_never_contradicts_the_planner_role() -> None:
    """Root-cause guard. Every contradiction found in this reader came from the
    same habit: the Web layer re-deriving study semantics from free text that
    the Planner had already compiled into a typed field. `planned_analysis_role`
    is Planner-owned and the pipeline gates on it (schema.py
    PlannedAnalysisRole), so the reader may never place a step somewhere the
    role denies. The role answers "which step carries the claim", not "which
    phase is this" - cohort building, data checks, and figures are all
    `auxiliary` - so the method heuristic resolves only that remaining
    question, and can never promote an auxiliary step into a result."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")

    def stages(steps: list[dict]) -> dict[str, list[str]]:
        payload = {
            "research_question": "Q",
            "design_selection": {"candidates": [{"disposition": "selected"}]},
            "steps": steps,
        }
        script = f"""
global.window = {{
  EU_LANG: 'zh',
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
        html = subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout
        found: dict[str, list[str]] = {}
        for block in html.split('<li class="ag-plan-flow-stage is-')[1:]:
            key = block[: block.index('"')]
            found[key] = re.findall(r"<b>(\d+)</b>", block)
        return found

    # the method text looks like a result, but the plan calls it auxiliary:
    # it must not be shown as the study's main analysis
    demoted = stages([
        {"step_id": "x", "method": "descriptive_counts_of_exposure_outcome", "planned_analysis_role": "auxiliary", "expected_outputs": ["table:a"]},
        {"step_id": "y", "method": "adjusted_association_model", "planned_analysis_role": "primary", "expected_outputs": ["table:b"]},
    ])
    assert demoted["support"] == ["1"]
    assert demoted["primary"] == ["2"]

    # the method text looks like a data check, but the plan calls it primary
    promoted = stages([
        {"step_id": "q", "method": "table_one_primary_result", "planned_analysis_role": "primary", "expected_outputs": ["table:c"]},
    ])
    assert promoted["primary"] == ["1"]
    assert "quality" not in promoted

    # a sensitivity step whose wording never says "sensitivity" still lands
    # in the robustness stage
    sensitivity = stages([
        {"step_id": "z", "method": "alternative_exposure_coding", "planned_analysis_role": "sensitivity", "expected_outputs": ["table:d"]},
    ])
    assert sensitivity["robustness"] == ["1"]

    # secondary results are results, not checks
    secondary = stages([
        {"step_id": "s", "method": "absolute_risk_context", "planned_analysis_role": "secondary", "expected_outputs": ["table:e"]},
    ])
    assert secondary["primary"] == ["1"]

    # plans that state no role (older runs, fixtures) still fold by method
    legacy = stages([
        {"step_id": "c", "method": "cohort_definition_and_attrition", "expected_outputs": ["table:cohort_flow"]},
        {"step_id": "p", "method": "descriptive_counts", "expected_outputs": ["table:e"]},
    ])
    assert legacy["population"] == ["1"]
    assert legacy["primary"] == ["2"]


def test_plan_stage_folding_holds_across_study_designs() -> None:
    """Stage assignment is method-driven, so designs the flow map was not
    built against still fold correctly: a survival study reaches the main
    analysis, and a descriptive study's counts step is its result, not a
    data check."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")

    def render(steps: list[dict]) -> str:
        payload = {
            "research_question": "Q",
            "design_selection": {"candidates": [{"disposition": "selected"}]},
            "steps": steps,
        }
        script = f"""
global.window = {{
  EU_LANG: 'zh',
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
        return subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout

    survival = render([
        {"step_id": "primary_survival", "method": "cox_proportional_hazards", "expected_outputs": ["table:hazard_ratios"]},
        {"step_id": "ph_sensitivity", "method": "proportional_hazards_sensitivity", "expected_outputs": ["table:ph"]},
        {"step_id": "km_figure", "method": "visualization", "expected_outputs": ["figure:km_plot"]},
    ])
    assert "时间结局分析" in survival
    assert "is-primary" in survival and "is-robustness" in survival and "is-figure" in survival

    prediction = render([
        {"step_id": "model_development", "method": "prediction_discrimination_calibration", "expected_outputs": ["table:auroc"]},
    ])
    assert "预测模型" in prediction
    assert "is-primary" in prediction

    # a counts-only study's distribution step is the result, not a data check
    descriptive = render([
        {"step_id": "prevalence", "method": "descriptive_counts", "expected_outputs": ["table:prevalence"]},
    ])
    assert "ag-plan-flow-stage is-primary" in descriptive
    assert "ag-plan-flow-stage is-quality" not in descriptive
    assert "描述性分布" in descriptive

    # ...but an audit that happens to be described as descriptive stays a check
    audit = render([
        {"step_id": "descriptive_quality_summary", "method": "measurement_audit", "expected_outputs": ["table:q"]},
    ])
    assert "ag-plan-flow-stage is-quality" in audit

    # an unmapped output key still tells the reader what kind of thing it is
    assert "图件 · km plot" in survival
    assert "结果表 · auroc" in prediction


def test_candidate_agent_plan_leads_with_a_stage_flow_map() -> None:
    """A generated plan is a flat list of typed steps (often 8-12). The reader
    must see the shape of the run before any prose, so the analysis-path
    section leads with a stage flow map and folds the per-step wording into a
    collapsed detail list. Stage assignment is method-driven and case-neutral."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")
    plan_css = _read("css/agent-plan.css")

    payload = {
        "analysis_type": "association_study",
        "research_question": "研究问题",
        "endpoint": {"name": "death", "kind": "binary"},
        "robustness_specs": [{"id": "complete_case"}],
        "design_selection": {"candidates": [{"disposition": "selected"}]},
        "steps": [
            {
                "step_id": "cohort_accounting",
                "method": "cohort_definition_and_attrition",
                "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
            },
            {"step_id": "baseline_context", "method": "table_one", "expected_outputs": ["table:baseline_table"]},
            {
                "step_id": "measurement_quality",
                "method": "missing_data",
                "expected_outputs": ["table:measurement_process_audit", "table:measurement_missingness"],
            },
            {
                "step_id": "primary_landmark_association",
                "method": "signed_landmark_restricted_cubic_spline",
                "expected_outputs": [
                    "table:landmark_rcs_curve",
                    "table:landmark_rcs_contrasts",
                    "table:landmark_linear_sensitivity",
                    "table:landmark_adjusted_absolute_risk",
                    "table:landmark_population_flow",
                    "log:landmark_scientific_runtime_receipt",
                ],
            },
            {
                "step_id": "sensitivity_planner_proposed_exposure_rcs",
                "method": "restricted_cubic_spline_sensitivity",
                "expected_outputs": ["table:sensitivity_planner_proposed_exposure_rcs"],
            },
            {"step_id": "absolute_risk_context", "method": "absolute_risk_context", "expected_outputs": ["table:absolute_risk_context"]},
            {"step_id": "robustness_suite", "method": "robustness_sensitivity", "expected_outputs": ["table:robustness_summary"]},
            {"step_id": "robustness_suite_figure", "method": "visualization", "expected_outputs": ["figure:robustness_plot"]},
            {"step_id": "09_cohort_accounting_figure", "method": "visualization", "expected_outputs": ["figure:cohort_flow"]},
            {"step_id": "10_data_quality_figure", "method": "visualization", "expected_outputs": ["figure:data_quality"]},
        ],
    }
    script = f"""
global.window = {{
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
    html = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout

    # the flow map replaces the flat list as the default reading surface
    assert '<ol class="ag-plan-flow">' in html
    assert "共 10 个步骤 · 5 个阶段" in html
    for stage in (
        "ag-plan-flow-stage is-population",
        "ag-plan-flow-stage is-quality",
        "ag-plan-flow-stage is-primary",
        "ag-plan-flow-stage is-robustness",
        "ag-plan-flow-stage is-figure",
    ):
        assert stage in html
    assert html.count('class="ag-plan-flow-stage') == 5
    # figure steps are stage 5 even though their ids mention the earlier stages
    assert html.index("is-robustness") < html.index("is-figure")
    # every step keeps a localized short title, none leaks an internal step id
    for title in (
        "队列与分母账本",
        "基线特征表",
        "测量覆盖与缺失审计",
        "主关联分析（landmark 起点）",
        "敏感性分析 · 暴露形式设定",
        "绝对风险背景",
        "稳健性复核",
        "稳健性分析图",
        "队列流程图",
        "数据质量图",
    ):
        assert title in html
    for internal in ("step_id", "robustness_suite_figure", "signed_landmark", "10_data_quality_figure"):
        assert internal not in html
    # per-step prose is preserved but folded away
    assert '<details class="ag-plan-step-detail">' in html
    assert "逐步说明 · 共 10 步" in html
    # long output rosters are capped instead of printed as one chain
    assert '<span class="is-more">+2</span>' in html
    # the whole plan is summarized before any prose
    assert '<div class="ag-plan-glance">' in html
    assert "个计划步骤" in html and "个阶段" in html
    # the flow styles have one explicit owner
    for selector in (".ag-plan-flow", ".ag-plan-flow-stage", ".ag-plan-glance", ".ag-plan-step-detail"):
        assert selector in plan_css


def test_long_readable_artifacts_lead_with_a_section_index() -> None:
    """A readable artifact with many sections cannot be understood without
    scrolling it end to end, so the generic view leads with a section index."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")
    agent_css = _read("css/agent.css")
    plan_css = _read("css/agent-plan.css")

    payload = {
        "schema_version": "easyicu.web-pipeline-result-tables/1",
        "table_count": 3,
        "tables": [
            {"label": f"Result table {index}", "headers": ["a"], "rows": [["1"]]}
            for index in range(1, 4)
        ],
    }
    script = f"""
global.window = {{
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('result_tables.json', {json.dumps(payload)}));
"""
    html = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout

    assert '<nav class="ag-artifact-contents"' in html
    assert "本产物共 4 个区块" in html
    assert html.index("ag-artifact-contents") < html.index("Result table 1")
    assert ".ag-artifact-contents" in agent_css
    assert ".ag-artifact-contents" not in plan_css


def test_candidate_plan_glosses_provider_prose_without_replacing_it() -> None:
    """Design-level fields carry the plan's own answers. The retired copy was
    keyed on the field NAME, so it asserted an ICU-stay time zero and a
    logistic primary method for every association plan - false for the
    landmark design this very payload describes. A gloss is now allowed only
    where it is derivable from `analysis_type` or from an unambiguous method
    token in the plan's own wording, and the wording is always kept beside
    it."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the executable renderer contract")
    renderer = _read("js/screens-agent-render.js")
    payload = {
        "analysis_type": "association_study",
        "research_question": "乳酸与院内死亡是否相关？",
        "endpoint": {"name": "death", "kind": "binary"},
        "robustness_specs": [{"id": "complete_case"}],
        "design_selection": {
            "candidates": [
                {
                    "disposition": "selected",
                    "analysis_type": "association_study",
                    "decision_reason": "Selected because it directly addresses the association question.",
                    "estimand": "The adjusted association between lactate and death.",
                    # a landmark start: the retired canned copy claimed an
                    # ICU-stay time zero for every association plan
                    "time_zero": "A prespecified 24-hour ICU landmark for each stay.",
                    "observation_window": "From the landmark through in-hospital discharge or death.",
                    "primary_method": "Covariate-adjusted logistic association model.",
                    "supports": "The strength and uncertainty of the association.",
                    "cannot_prove": "It cannot prove causality.",
                    "reviewable_plan": [
                        "纳入符合条件的 ICU 住院，以每次 ICU stay 为分析单位。",
                        "推荐先采用 ICU 入科后 24 小时内首次乳酸作为主要暴露，并把峰值乳酸作为敏感性分析。",
                        "主要结局定义为本次住院期间发生院内死亡。",
                        "采用多变量 Logistic 回归，调整预先指定的基线混杂因素。",
                        "先报告乳酸和协变量缺失率，再按预设规则进行完整病例与插补分析。",
                        "分析前检查样本量、死亡事件数、乳酸覆盖率和缺失情况。",
                    ],
                    "required_variables": [
                        "lact",
                        "death",
                        "age",
                        "sex",
                        "adm",
                        "hr",
                        "map",
                        "ph",
                        "crea",
                        "bili",
                    ],
                }
            ]
        },
        "steps": [
            {
                "step_id": "measurement_audit",
                "intent": "Audit measurement completeness.",
                "expected_outputs": ["table:measurement_audit"],
            },
            {
                "step_id": "primary_adjusted_model",
                "intent": "Estimate the adjusted association.",
                "expected_outputs": ["table:adjusted_association_estimates"],
            },
        ],
    }
    script = f"""
global.window = {{
  EU_LANG: 'zh',
  EU_HTML: {{ esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'), escAttr: value => String(value ?? '') }},
  t: (en, zh) => zh,
  icon: () => '',
}};
eval({json.dumps(renderer)});
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('agent_plan.json', {json.dumps(payload)}));
"""
    html = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout

    for text in (
        # rationale, time zero, and observation window have no derivable gloss,
        # so the plan's own wording stands alone rather than being invented
        "Selected because it directly addresses the association question.",
        "A prespecified 24-hour ICU landmark",
        # the method gloss summarizes the plan's own wording, and says so
        "计划写明的方法：多变量 Logistic 回归",
        "计划原文",
        "乳酸",
        "院内死亡",
        "入院类型",
        "心率",
        "平均动脉压",
        "血液酸碱度（pH）",
        "肌酐",
        "胆红素",
        # the step note describes the METHOD and names no variable; the
        # plan's own English wording is preserved verbatim beside it
        "检查本计划所需变量的测量覆盖与缺失情况",
        "计划原文",
        "Audit measurement completeness.",
        "变量可用性与缺失情况",
        "报告效应大小与不确定性",
        "校正后关联估计",
        "Planner 推荐方案（待审阅）",
        "先给方案，再由你修改或批准",
        "暴露定义、时间窗与汇总方式",
        "推荐先采用 ICU 入科后 24 小时内首次乳酸作为主要暴露",
        "敏感性分析与数据可行性检查",
        "以下内容由 EasyICU 先行推荐，尚未视为研究者确认",
    ):
        assert text in html
    assert "Planner 尚未给出完整、可审阅的推荐方案" not in html
    # nothing asserts a time zero, observation window, or model family that
    # the plan did not itself state
    for invented in (
        "以每次 ICU 住院记录及该次住院中已记录的主要暴露测量作为研究起点",
        "从研究起点观察至本次住院结局被记录",
        "使用预先明确变量编码的多变量 Logistic 回归",
        "并同时报告校正后关联、绝对风险和稳健性分析",
    ):
        assert invented not in html
    for raw in ("lact</span>", "death</span>", "measurement audit", "strobe 2007"):
        assert raw not in html


def test_pi_chat_uses_a_scrolling_transcript_and_bottom_composer() -> None:
    owner = _read("css/guided-pi.css")
    assert ".gpi-panel{height:100%;min-height:0;display:flex" in owner
    assert "flex-direction:column;overflow:hidden" in owner
    assert ".gpi-log{flex:1 1 auto;min-height:0;overflow:auto" in owner
    assert ".gpi-compose{flex:0 0 auto" in owner
    assert ".gpi-compose-card{" in owner
    assert "border-radius:20px" in owner
    assert "grid-template-rows:auto auto minmax(0,1fr) auto auto" not in owner
    assert (
        ".gpi-text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:16.5px" in owner
    )
    assert "font-size:16.5px;line-height:1.55" in owner


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
        "js/screens-guided-pi-activity.js",
        "js/screens-guided-pi-provider.js",
        "js/screens-guided-pi.js",
        "js/screens-guided-pi-evidence-preview.js",
        "js/screens-guided-pi-preview.js",
        "js/screens-guided-pi-workbench-preview.js",
        "js/screens-guided-pi-literature.js",
        "js/screens-guided-pi-markdown.js",
        "js/screens-guided-pi-next-actions.js",
        "js/screens-guided-pi-message-actions.js",
        "js/screens-guided-pi-demo.js",
        "js/screens-guided-pi-replay.js",
    ):
        subprocess.run(
            [node, "--check", str(STATIC / relative)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_pi_activity_owner_renders_safe_expanded_lifecycle_details() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-activity.js")
    script = f"""
      global.window = {{EU_LANG: 'en'}};
      eval({source!r});
      const activity = window.EU_GUIDED_PI_ACTIVITY.create({{
        tr: (en, _zh) => en,
        esc: value => String(value == null ? '' : value),
        iconHtml: () => '',
        resourceName: resource => resource && resource.name || '',
        resourceKey: resource => resource && resource.name || '',
        resourceButton: (resource, label) => '<button>' + (label || resource.name) + '</button>',
      }});
      const html = activity.render({{
        status: 'complete', expanded: true, startedAt: 1000, endedAt: 3500,
        steps: [
          {{kind: 'assistant', phase: 1, status: 'complete', publicChars: 8, startedAt: 1000, endedAt: 1500}},
          {{
            kind: 'tool', toolName: 'easyicu_read_project_file', status: 'complete',
            startedAt: 1500, endedAt: 2500, resource: {{name: 'plan.json'}},
            text: 'Loaded the governed project file.',
            arguments: 'secret-token', reasoning: 'secret-thought',
          }},
          {{kind: 'retry', status: 'complete', label: 'Plan contract retry', startedAt: 2500, endedAt: 2500}},
        ],
      }});
      const live = {{status: 'running', startedAt: Date.now() - 1200, steps: [
        {{kind: 'assistant', phase: 2, status: 'running', startedAt: Date.now() - 1200}},
      ]}};
      activity.appendPublicDelta(live, 'Visible public answer chunk');
      const failedHtml = activity.render({{
        status: 'error', expanded: false, startedAt: 1000, endedAt: 2000,
        steps: [{{kind: 'tool', toolName: 'easyicu_inspect_context', status: 'error'}}],
      }});
      const currentFailedHtml = activity.render({{
        status: 'error', expanded: true, startedAt: 1000, endedAt: 2000,
        steps: [{{kind: 'tool', toolName: 'easyicu_inspect_context', status: 'error'}}],
      }});
      const liveHtml = activity.render(live);
      const turns = {{status: 'complete', startedAt: 1000, endedAt: 3000, steps: []}};
      activity.startTurn(turns, 1000); activity.finishTurn(turns, 1800);
      activity.startTurn(turns, 1900); activity.finishTurn(turns, 3000);
      const turnHtml = activity.render(turns);
      window.EU_LANG = 'zh';
      const zhActivity = window.EU_GUIDED_PI_ACTIVITY.create({{
        tr: (_en, zh) => zh,
        esc: value => String(value == null ? '' : value),
        iconHtml: () => '',
        resourceName: resource => resource && resource.name || '',
        resourceKey: resource => resource && resource.name || '',
        resourceButton: (resource, label) => '<button>' + (label || resource.name) + '</button>',
      }});
      const zhHtml = zhActivity.render({{
        status: 'complete', expanded: true, startedAt: 1000, endedAt: 2000,
        steps: [
          {{kind: 'tool', toolName: 'easyicu_inspect_context', status: 'complete', text: 'Loaded StudyContext revision 5.'}},
          {{kind: 'tool', toolName: 'easyicu_inspect_workflow', status: 'complete', text: '已读取科研流程。'}},
        ],
      }});
      process.stdout.write(JSON.stringify({{
        activity: html.includes('Activity'),
        finishedTurnCollapses: !html.includes(' open>'),
        historicalFailureCollapses: !failedHtml.includes(' open>'),
        currentFailureCollapses: !currentFailedHtml.includes(' open>'),
        modelPhase: html.includes('Public response phase 1 finished'),
        readTool: html.includes('Read project file · plan.json'),
        retryLabel: html.includes('Plan contract retry') && !html.includes('undefined'),
        stepDuration: html.includes('0.5s') && html.includes('1.0s') && html.includes('<0.1s'),
        totalDuration: html.includes('total 2.5s'),
        publicStream: liveHtml.includes('Streaming public response phase 2')
          && liveHtml.includes('Visible public answer chunk')
          && liveHtml.includes('data-gpi-live-elapsed'),
        noZeroMs: !html.includes('0 ms') && !liveHtml.includes('0 ms'),
        hiddenOverlappingTurns: !turnHtml.includes('Model turn 1 finished') && !turnHtml.includes('Model turn 2 finished'),
        englishReceiptVisibleInEnglish: html.includes('Loaded the governed project file.'),
        mismatchedReceiptHiddenInChinese: !zhHtml.includes('Loaded StudyContext revision 5.')
          && zhHtml.includes('已读取科研流程。'),
        privacyNotice: html.includes('private chain-of-thought is never displayed'),
        leaked: html.includes('secret-token') || html.includes('secret-thought'),
      }}));
    """
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "activity": True,
        # A finished turn collapses; its summary already names the tools and
        # the duration. Failed details remain available by clicking the summary.
        "finishedTurnCollapses": True,
        "historicalFailureCollapses": True,
        "currentFailureCollapses": True,
        "modelPhase": True,
        "readTool": True,
        "retryLabel": True,
        "stepDuration": True,
        "totalDuration": True,
        "publicStream": True,
        "noZeroMs": True,
        "hiddenOverlappingTurns": True,
        "englishReceiptVisibleInEnglish": True,
        "mismatchedReceiptHiddenInChinese": True,
        "privacyNotice": True,
        "leaked": False,
    }


def test_pi_project_reopens_latest_session_and_replays_safe_lifecycle() -> None:
    owner = _read("js/screens-guided-pi.js")
    childjob_owner = _read("js/screens-guided-pi-childjob.js")
    transcript_owner = _read("js/screens-guided-pi-transcript.js")
    activity_owner = _read("js/screens-guided-pi-activity.js")
    replay = _read("js/screens-guided-pi-replay.js")
    assert "state.session.active_message_job_id" in owner
    assert "watchJob(activeMessageJob)" in owner
    assert "preferredSessionId(state.sessions, remembered, '', uiLanguage())" in owner
    assert "preferredSessionId(state.sessions, '', next, uiLanguage())" in owner
    assert "loadPiCopilotSessions(100, projectId(), next)" in owner
    assert "replayOwner.preferredSessionId(matching, '', next, uiLanguage())" in owner
    assert "await openSession(existingSessionId)" in owner
    assert "session.last_turn_events" in replay
    assert "next_cursor" in replay
    assert "saved-activity-" in transcript_owner
    assert "const replayStarted = timeMs((replay[0] && replay[0].at)" in transcript_owner
    assert "const replayEnded = timeMs((replay[replay.length - 1]" in transcript_owner
    assert "Math.min(Number(replayActivity.startedAt), replayStarted)" in transcript_owner
    assert "state.session.archived_child_jobs" in owner
    assert "archiveChildJob(jobId)" in owner
    assert "childJobPresentation" in replay
    assert "Analysis plan ready for review" in replay
    assert "activity.displayTitle" in childjob_owner
    assert "row.durationKnown === false" in activity_owner
    assert "data-gpi-presentation-pin" in owner
    assert "pinPiCopilotPresentation" in owner
    assert "private chain-of-thought" in activity_owner


def test_child_job_handoff_reply_is_not_a_second_completed_message() -> None:
    """A governed background job has one visible lifecycle, not two."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    transcript_owner = _read("js/screens-guided-pi-transcript.js")
    script = f"""
      global.window = {{
        EU_GUIDED_PI_REPLAY: {{ lifecycleTurns: session => session.replayTurns || [] }},
      }};
      eval({_ESCAPE_OWNER!r});
      eval({transcript_owner!r});
      const upsert = (activity, step) => {{
        const at = activity.steps.findIndex(row => row.id === step.id);
        if (at >= 0) activity.steps[at] = {{ ...activity.steps[at], ...step }};
        else activity.steps.push(step);
      }};
      const activity = {{
        focusLatest: rows => rows,
        startTurn: () => {{}}, finishTurn: () => {{}},
      }};
      const owner = window.EU_GUIDED_PI_TRANSCRIPT.create({{
        activity, upsertActivityStep: upsert,
        timeMs: value => Date.parse(value || '2026-08-28T09:00:00Z'),
        resourceKey: resource => JSON.stringify(resource || null),
        modelErrorText: code => code,
        workflowActionCode: () => 'analysis_running',
      }});
      const submittedTranscript = [
        {{ role: 'user', timestamp: '2026-08-28T09:00:00Z',
          content: [{{ type: 'text', text: '开始生成正式研究计划。' }}] }},
        {{ role: 'assistant', timestamp: '2026-08-28T09:00:00Z',
          content: [{{ type: 'tool_call', tool_call_id: 'call-plan', tool_name: 'easyicu_run' }}] }},
        {{ role: 'tool', timestamp: '2026-08-28T09:00:01Z',
          content: [{{ type: 'tool_result', tool_call_id: 'call-plan', tool_name: 'easyicu_run',
            code: 'easyicu_full_run_submitted', job_id: 'child-plan-1' }}] }},
        {{ role: 'assistant', timestamp: '2026-08-28T09:00:01Z',
          content: [{{ type: 'text', text: '已提交正式研究计划生成任务。\\n\\n**下一步：**继续等待计划完成。' }}] }},
      ];
      const submitted = owner.transcriptMessages({{
        transcript: submittedTranscript,
        replayTurns: [{{ status: 'done', events: [] }}, {{ status: 'done', events: [
          {{ type: 'tool_end', at: '2026-08-28T09:00:01Z',
             code: 'easyicu_full_run_submitted', job_id: 'child-plan-1' }},
        ] }}],
      }});
      const ordinary = owner.transcriptMessages({{
        transcript: [
          {{ role: 'user', timestamp: '2026-08-28T09:00:00Z',
            content: [{{ type: 'text', text: '请解释当前计划。' }}] }},
          {{ role: 'assistant', timestamp: '2026-08-28T09:00:01Z',
            content: [{{ type: 'text', text: '这是普通回答。' }}] }},
        ],
        replayTurns: [{{ status: 'done', events: [
          {{ type: 'tool_end', at: '2026-08-28T09:00:01Z',
             code: 'easyicu_plan_projected' }},
        ] }}],
      }});
      process.stdout.write(JSON.stringify({{
        submittedAssistant: submitted.filter(row => row.role === 'assistant').length,
        ordinaryAssistant: ordinary.filter(row => row.role === 'assistant').length,
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert json.loads(completed.stdout) == {
        "submittedAssistant": 0,
        "ordinaryAssistant": 1,
    }
    shell = _read("js/screens-guided-pi.js")
    assert "row.childJobHandoff = Boolean(state.childJobId)" in shell
    assert "if (row.childJobHandoff) return ''" in shell
    assert (
        "timeline.slice().reverse().find(row => ['assistant', 'activity'].includes(row.role))"
        in shell
    )


def test_pi_project_restore_does_not_let_an_empty_session_hide_history() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-replay.js")
    script = f"""
      global.window = {{}};
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      const choose = window.EU_GUIDED_PI_REPLAY.preferredSessionId;
      const sessions = [
        {{ session_id: 'empty-new', agent_mode: 'workspace', language: 'en', message_count: 0, last_message_job_id: null }},
        {{ session_id: 'workspace-history', agent_mode: 'workspace', language: 'en', message_count: 0, last_message_job_id: 'job-1' }},
        {{ session_id: 'research-history', agent_mode: 'research', language: 'en', message_count: 3, last_message_job_id: 'job-2' }},
        {{ session_id: 'research-history-zh', agent_mode: 'research', language: 'zh', message_count: 4, last_message_job_id: 'job-3' }},
      ];
      console.log(choose(sessions, 'empty-new'));
      console.log(choose(sessions, 'workspace-history'));
      console.log(choose(sessions, ''));
      console.log(choose([sessions[0]], 'empty-new'));
      console.log(choose(sessions, '', 'workspace'));
      console.log(choose(sessions, '', 'research'));
      console.log(choose(sessions, 'empty-new', 'workspace'));
      console.log(choose(sessions, 'research-history-zh', 'research', 'en'));
      console.log(choose(sessions, '', 'research', 'zh'));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "workspace-history",
        "workspace-history",
        "workspace-history",
        "empty-new",
        "workspace-history",
        "research-history",
        "workspace-history",
        "research-history",
        "research-history-zh",
    ]


def test_pi_conversation_language_is_bound_to_the_active_ui_locale() -> None:
    owner = _read("js/screens-guided-pi.js")
    replay = _read("js/screens-guided-pi-replay.js")
    assert "function sessionMatchesUiLanguage" in owner
    assert "state.sessions.filter(sessionMatchesUiLanguage)" in owner
    assert "easyicu_pi_copilot_session:' + encodeURIComponent(projectId()) + ':' + uiLanguage()" in owner
    assert "window.addEventListener('easyicu:languagechange', handleLanguageChange)" in owner
    assert "if (!sessionMatchesUiLanguage(payload && payload.session))" in owner
    assert "if (!sessionMatchesUiLanguage(state.session))" in owner
    assert "requestedLanguage" in replay
    assert "(!language || sessionLanguage === language)" in replay


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
    assert "分析数据已准备" in workbench
    assert "候选计划精确绑定队列" in workbench
    assert "数据字段" in workbench
    assert "全部分析记录均有值" in workbench
    assert "部分分析记录缺少数值" in workbench
    assert ".gpi-wb" in css
    assert ".patient-" not in css
    assert ".cohort-" not in css
    assert ".crossdb-" not in css


def test_complete_research_demo_is_natural_truthful_and_clickable() -> None:
    demo = _read("js/screens-guided-pi-demo.js")
    pi_owner = _read("js/screens-guided-pi.js")
    aside_owner = _read("js/screens-guided-pi-aside.js")
    resource_owner = _read("js/screens-guided-pi-resources.js")
    preview = _read("js/screens-guided-pi-preview.js")
    assert "展示真实规划生命周期，在计划审阅处暂停" in demo
    assert "计划草案 1/5 未满足科学合同；正在重试" in demo
    assert "计划合同已通过；分析已暂停，等待人工审阅" in demo
    assert "我批准这份精确审阅计划" in demo
    assert "这是否意味着 Demo 失败" in demo
    assert "审稿包已完整生成" in demo
    assert "完成 6/6 · 数据质量图" in demo
    assert "正式稿件被确定性权限闸门拒绝" in demo
    assert "33,997 / 94,458 (35.991658%)" in demo
    assert "4,986 / 60,461 (8.246638%)" in demo
    assert "4,480 / 33,997 (13.177633%)" in demo
    assert "run_20260815T061842_5049c6" in demo
    assert "bounded_reviewer_projection_from_registered_run" in demo
    assert "reportable: false" in demo
    assert "publication_authorized: false" in demo
    assert "临床稿件是另一项交付物" in demo
    assert "https://pubmed.ncbi.nlm.nih.gov/17938396/" in demo
    assert "kind: 'demo_artifact'" in demo
    assert "title: item ? item.title : name" in demo
    assert "resource.kind === 'demo_artifact'" in resource_owner
    assert "resource.title || label(resource)" in resource_owner
    assert "value.kind === 'demo_artifact'" in preview
    assert "Reviewer demonstration complete · Engineering evidence" in preview
    assert "Bounded reviewer projection · Standard Web renderer" in preview
    assert "safe.kind !== 'demo_artifact'" in preview
    assert "reviewer_protocol.json" in demo
    assert "analysis_plan.json" in demo
    assert "descriptive_results.json" in demo
    assert "applicability_audit.json" in demo
    assert "execution_receipt.json" in demo
    assert "authority_verdict.json" in demo
    for standard_artifact in (
        "run_context.json",
        "cohort_summary.json",
        "quality_gate.json",
        "agent_plan.json",
        "literature_evidence.json",
        "scientific_plan_review.json",
        "scientific_readiness.json",
        "manuscript_draft.json",
        "figure_gallery.json",
        "result_tables.json",
        "source_run_manifest.json",
        "evidence_ledger.json",
    ):
        assert standard_artifact in demo
    assert "打开审稿人报告" in demo
    assert "人工种子，而不是已完成的新颖性检索" in demo
    assert demo.count("step_id: '") == 6
    assert "citation_keys:" in demo
    assert "projection_note:" in demo
    assert "required_stage_count: 8" in demo
    assert "completed_required_stages: 8" in demo
    assert "reviewer_demo_complete" in demo
    assert "['manuscript', 'complete', 'reviewer_dossier_complete']" in demo
    assert "10.021% 是死亡事件比例" in demo
    assert "94,458 / 94,458" in demo
    assert "0 / 9,466" in demo
    assert "primaryDocument" in demo
    assert "window.EU_GUIDED_PI_PREVIEW.open(primary" in pi_owner
    assert "Reviewer workflow" in pi_owner
    assert "gpi-demo-reviewer" not in pi_owner
    assert "reviewer_dossier_complete" in aside_owner
    assert "审稿 HTML 与 PDF 报告已完整生成" in aside_owner
    assert "Reviewer demonstration" in aside_owner
    assert "operator_plan_approved" in aside_owner
    assert "validated_analysis_complete" in aside_owner
    assert "validated_analysis_ready" in aside_owner
    assert "approved_plan_setup_receipt" in aside_owner
    assert "interpretation_complete" in aside_owner
    assert "evidence_bound_interpretation_ready" in aside_owner
    assert "manuscript_draft_ready_for_review" in aside_owner
    assert "human_review_required" in aside_owner


def test_reviewer_demo_contract_completes_all_stages_without_upgrading_authority() -> (
    None
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-demo.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      const demo = window.EU_GUIDED_PI_DEMO;
      const workflow = demo.workflow();
      const verdict = demo.artifact('authority_verdict.json');
      const results = demo.artifact('descriptive_results.json');
      console.log(JSON.stringify({{
        completed: workflow.completed_required_stages,
        required: workflow.required_stage_count,
        statuses: workflow.stages.map(stage => stage.status),
        verdict: verdict.status,
        reportable: verdict.reportable,
        publicationAuthorized: verdict.publication_authorized,
        result: results.metrics[2].value,
        primary: demo.primaryDocument().artifact,
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "completed": 8,
        "required": 8,
        "statuses": ["complete"] * 8,
        "verdict": "reviewer_demo_complete",
        "reportable": False,
        "publicationAuthorized": False,
        "result": "4,986 / 60,461 (8.246638%)",
        "primary": "system-validation-report.html",
    }


def test_reviewer_demo_lifecycle_exposes_only_resolvable_standard_artifacts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-demo.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      const demo = window.EU_GUIDED_PI_DEMO;
      const messages = demo.messages();
      const activities = messages.filter(message => message.role === 'activity');
      const resources = [];
      messages.forEach(message => {{
        (message.resources || []).forEach(resource => resources.push(resource));
        (message.steps || []).forEach(step => {{
          if (step.resource) resources.push(step.resource);
          (step.resources || []).forEach(resource => resources.push(resource));
        }});
      }});
      const standard = [
        'run_context.json', 'cohort_summary.json', 'quality_gate.json',
        'agent_plan.json', 'literature_evidence.json',
        'scientific_plan_review.json', 'scientific_readiness.json',
        'manuscript_draft.json', 'figure_gallery.json', 'result_tables.json',
        'source_run_manifest.json', 'evidence_ledger.json',
      ];
      const artifacts = resources
        .filter(resource => resource.kind === 'demo_artifact')
        .map(resource => resource.artifact);
      console.log(JSON.stringify({{
        activityCount: activities.length,
        stepCount: activities.reduce((count, activity) => count + activity.steps.length, 0),
        hasRetry: activities.some(activity => activity.steps.some(step => step.kind === 'retry')),
        hasPause: activities.some(activity => activity.steps.some(step => step.code === 'blocked')),
        hasStrictStop: activities.some(activity => activity.steps.some(step => step.code === 'withheld_as_designed')),
        missingStandard: standard.filter(name => !artifacts.includes(name)),
        unresolved: artifacts.filter(name => !demo.hasArtifact(name)),
        rendererShapes: {{
          planSteps: demo.artifact('agent_plan.json').steps.length,
          literatureCards: demo.artifact('literature_evidence.json').citations.length,
          reviewDimensions: Object.keys(demo.artifact('scientific_plan_review.json').dimension_scores).length,
          readinessDomains: demo.artifact('scientific_readiness.json').domains.length,
          gateChecks: demo.artifact('quality_gate.json').gate.checks.length,
          figures: demo.artifact('figure_gallery.json').figures.length,
          resultTables: demo.artifact('result_tables.json').tables.length,
          lockedClaims: demo.artifact('manuscript_draft.json').claims.length,
        }},
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "activityCount": 4,
        "stepCount": 42,
        "hasRetry": True,
        "hasPause": True,
        "hasStrictStop": True,
        "missingStandard": [],
        "unresolved": [],
        "rendererShapes": {
            "planSteps": 6,
            "literatureCards": 9,
            "reviewDimensions": 8,
            "readinessDomains": 5,
            "gateChecks": 7,
            "figures": 3,
            "resultTables": 1,
            "lockedClaims": 1,
        },
    }


def test_guided_copilot_does_not_expose_benchmark_specific_navigation() -> None:
    guided_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC / "js").glob("screens-guided*.js"))
    )

    assert "Canonical9" not in guided_sources
    assert "打开 Canonical9 九题独立对话" not in guided_sources


def test_pi_messages_project_governed_tool_artifacts_beside_the_answer() -> None:
    owner = _read("js/screens-guided-pi.js")
    transcript_owner = _read("js/screens-guided-pi-transcript.js")
    # Ranking and de-duplicating one message's artifacts is resource-identity
    # work: it uses the same key that de-duplicates them, so it lives with the
    # resource owner rather than inline in the screen shell.
    resource_owner = _read("js/screens-guided-pi-resources.js")
    css = _read("css/guided-pi.css")

    assert "turnResources" in transcript_owner
    assert "currentTurnResources" in owner
    assert "gpi-message-resources" in owner
    assert "Referenced run artifacts" in owner
    assert "RESOURCE_OWNER.forMessage(row, 8)" in owner
    assert "result_tables.json" in resource_owner
    assert "figure_gallery.json" in resource_owner
    assert "result_tables.json" not in owner
    assert ".gpi-message-resources" in css


def test_workspace_resource_button_preserves_checked_preview_digest() -> None:
    owner = _read("js/screens-guided-pi-resources.js")

    assert (
        "resource.snapshot_sha256 || resource.review_sha256 || resource.checked_sha256 || resource.sha256"
        in owner
    )
    assert "checked_sha256: element.dataset.gpiResourceDigest" in owner
    assert 'data-gpi-resource-entry-mode="${esc(resource.entry_mode || \'\')}"' in owner
    assert "entry_mode: element.dataset.gpiResourceEntryMode" in owner


def test_complete_research_demo_reuses_the_unchanged_agent_figure() -> None:
    figure = STATIC / "assets" / "demo" / "e1-publication-figure.png"
    assert figure.is_file()
    assert figure.stat().st_size == 93_214
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == (
        "34a46b54558a6f08cc02434a6958558ecb8077abd59db78713ef8f9dd4172e4b"
    )


def test_reviewer_demo_reuses_the_web_renderer_and_hydrates_registered_figures() -> (
    None
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-demo.js")
    preview = _read("js/screens-guided-pi-preview.js")
    assert "demo.renderArtifact" not in preview
    assert "renderer.artifactStructuredView(state.resource.artifact" in preview
    assert "EU_GUIDED_PI_LITERATURE" in preview
    assert "Bounded reviewer projection · Standard Web renderer" in preview
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      global.fetch = async () => ({{
        ok: true,
        text: async () => [
          '<img src="data:image/png;base64,AAAA">',
          '<img src="data:image/png;base64,BBBB">',
          '<img src="data:image/png;base64,CCCC">',
        ].join(''),
      }});
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      window.EU_GUIDED_PI_DEMO.previewArtifact('figure_gallery.json').then(item => {{
        console.log(JSON.stringify({{
          count: item.figures.length,
          images: item.figures.map(figure => figure.data_url),
          schema: item.schema_version,
        }}));
      }});
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "count": 3,
        "images": [
            "data:image/png;base64,AAAA",
            "data:image/png;base64,BBBB",
            "data:image/png;base64,CCCC",
        ],
        "schema": "easyicu.web-pipeline-figure-gallery/1",
    }


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


def test_evidence_bound_manuscript_reader_stays_in_its_preview_owner() -> None:
    renderer = _read("js/screens-agent-render.js")
    preview = _read("js/screens-guided-pi-preview.js")
    evidence_preview = _read("js/screens-guided-pi-evidence-preview.js")
    styles = _read("css/guided-pi-preview.css")

    assert "manuscriptProvenanceView" in renderer
    assert "manuscript_provenance.json" in renderer
    assert "data-gpi-claim" in renderer
    assert "source_json_pointer" in renderer
    assert "related_artifacts" in renderer
    assert "data-gpi-claim-panel" in preview
    assert "data-gpi-claim-close" in preview
    assert "loadPiCopilotResearchEvidence" in preview
    assert "data-gpi-evidence-tab" in preview
    assert "patient_level_rows_withheld" in evidence_preview
    assert "Code is displayed, never executed" in evidence_preview
    assert ".gpi-bound-number" in styles
    assert ".gpi-claim-drawer" in styles
    assert ".gpi-evidence-code" in styles
    for unrelated in ("css/app.css", "css/tweaks.css"):
        assert ".gpi-bound-number" not in _read(unrelated)


def test_evidence_preview_renderer_escapes_code_json_and_table_content() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    subprocess.run(
        [
            node,
            str(
                STATIC.parents[3] / "tests" / "js" / "evidence_preview_security.test.js"
            ),
            str(STATIC / "js" / "screens-guided-pi-evidence-preview.js"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_system_validation_document_has_a_distinct_guided_preview_owner() -> None:
    preview = _read("js/screens-guided-pi-preview.js")
    guided = _read("js/screens-guided-pi.js")
    renderer = _read("js/screens-agent-render.js")

    assert "system_validation_document" in preview
    assert "system_validation_report\\.(html|pdf)" in preview
    assert "Reviewer demonstration complete · Engineering evidence" in preview
    # The preferred-artifact order that puts this report first moved to the
    # resource owner with the rest of message-resource ranking.
    assert "system_validation_report.html" in _read(
        "js/screens-guided-pi-resources.js"
    )
    assert "system_validation_report.json" in renderer
    assert "kind: 'demo_document'" in preview
    assert "/assets/demo/${state.resource.artifact}" in preview
    assert "system-validation-report.html" in _read("js/screens-guided-pi-demo.js")
    report_html = (
        STATIC / "assets" / "demo" / "system-validation-report.html"
    ).read_text(encoding="utf-8")
    report_pdf = (
        STATIC / "assets" / "demo" / "system-validation-report.pdf"
    ).read_bytes()
    assert "NOT A CLINICAL MANUSCRIPT" in report_html
    assert "REVIEWER DEMONSTRATION COMPLETE" in report_html
    assert "WITHHELD AS DESIGNED" in report_html
    assert report_html.count("data:image/png;base64") == 3
    assert "/Users/" not in report_html
    assert "/Volumes/" not in report_html
    assert report_pdf.startswith(b"%PDF-")
    assert b"/Marked true" in report_pdf
    assert b"/StructTreeRoot" in report_pdf
    for unrelated in (
        "js/app.js",
        "js/tweaks.js",
        "js/screens-guided-pi-replay.js",
    ):
        assert "system_validation_document" not in _read(unrelated)


def test_literature_renderer_escapes_metadata_and_rejects_unsafe_links() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({_ESCAPE_OWNER!r});
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


def test_literature_reader_separates_direct_evidence_from_system_references() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      const html = window.EU_GUIDED_PI_LITERATURE.renderArtifact({{
        direct_comparator_count: 0,
        direct_comparator_keys: [],
        search: {{
          search_conducted: true,
          note: 'Retrieval ran; PRISMA counts describe the records returned.',
          prisma: {{identified: 8, screened: 8, included: 0}},
          queries: {{pubmed: ['lactate AND in-hospital mortality AND ICU']}},
        }},
        evidence_boundary: 'internal EvidenceStore boundary',
        citations: [
          {{ key: 'strobe_2007', title: 'STROBE statement', year: '2007',
             source_url: 'https://pubmed.ncbi.nlm.nih.gov/17938396/' }},
          {{ key: 'singer_sepsis3_2016', title: 'Sepsis-3 consensus', year: '2016' }},
          {{ key: 'retrieved_lar', title: 'Lactate-to-albumin ratio study', year: '2023',
             screening: {{disposition: 'exclude', population_match: true,
               exposure_match: false, outcome_match: true,
               design_excerpt_available: true, publication_type_eligible: true}} }},
        ],
        step_citation_map: [
          {{ step_id: 'descriptive_quality_summary', planned_analysis_role: 'primary',
             intent: '说明研究对象、变量定义和描述性结果。',
             citation_bindings: [{{ key: 'strobe_2007', title: 'STROBE statement',
               year: '2007', application: '规范报告研究对象与变量定义。',
               design_elements: ['reporting'],
               source_url: 'https://pubmed.ncbi.nlm.nih.gov/17938396/' }}] }},
          {{ step_id: '04_publication_figure_fallback', planned_analysis_role: 'auxiliary',
             intent: 'internal fallback', citation_bindings: [] }},
        ],
      }});
      console.log(html);
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    html = completed.stdout
    assert "没有找到能直接支持这个问题的研究" in html
    assert "共检索到 8 篇候选、完成 8 篇筛选" in html
    assert "本次检索暂无文章通过筛选" in html
    assert "0 项科学设计决定" in html
    assert "科学设计依据" in html
    assert "尚未显示" in html
    assert "另有 1 篇报告规范" in html
    assert "仅有报告规范不能决定研究因素时间窗" in html
    assert "这些文献只规范如何透明报告，不能替代科学设计依据" in html
    assert "1 个决定" not in html
    assert "系统参考库里的其他资料" in html
    assert "没有被当作当前问题的直接依据" in html
    assert "PRISMA" not in html
    assert "EvidenceStore" not in html
    assert "descriptive_quality_summary" not in html
    assert "04_publication_figure_fallback" not in html
    assert "auxiliary" not in html


def test_literature_reader_labels_old_evidence_when_plan_revision_failed() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      const html = window.EU_GUIDED_PI_LITERATURE.renderArtifact({{
        run_id: 'run_old',
        direct_comparator_count: 0,
        search: {{ search_conducted: true, prisma: {{identified: 8, screened: 8}} }},
        citations: [],
        step_citation_map: [],
      }}, {{
        runId: 'run_old',
        currentRunId: 'run_old',
        nextActionCode: 'failed_pipeline_requires_fresh_plan',
        failedJob: {{
          kind: 'agent-run', status: 'failed',
          progress: [{{step: 'planning', current: 4, total: 4}}],
        }},
      }});
      console.log(html);
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    html = completed.stdout
    assert "上一版计划快照 · 修订版未生成成功" in html
    assert "系统尝试了 4 版草案" in html
    assert "只是历史记录，不是当前结论" in html
    assert "research_pipeline_plan_contract_exhausted" not in html


def test_literature_reader_translates_plan_bindings_for_chinese_readers() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({ _ESCAPE_OWNER!r });
      eval({source!r});
      const html = window.EU_GUIDED_PI_LITERATURE.renderArtifact({{
        direct_comparator_count: 1,
        direct_comparator_keys: ['lactate_icu'],
        search: {{ search_conducted: true, prisma: {{identified: 2, screened: 2}} }},
        citations: [{{ key: 'lactate_icu', title: 'ICU lactate and mortality', year: '2023',
          screening: {{population_match: true, exposure_match: true, outcome_match: true,
            design_excerpt_available: true, publication_type_eligible: true}} }}],
        step_citation_map: [{{ intent: 'nonlinear lactate functional form', citation_bindings: [{{
          key: 'durrleman_splines_1989', title: 'Flexible regression models with cubic splines',
          application: 'Use restricted cubic splines for nonlinear exposure modeling',
          design_elements: ['modeling'],
        }}] }}],
      }});
      console.log(html);
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    html = completed.stdout
    assert "检验乳酸与死亡是否为非线性关系" in html
    assert "用于检验乳酸作为连续变量时是否存在弯曲或阈值关系" in html
    assert "nonlinear lactate functional form" not in html
    assert "Use restricted cubic splines" not in html


def test_assistant_message_renderer_makes_https_citations_clickable_and_safe() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    literature = _read("js/screens-guided-pi-literature.js")
    markdown = _read("js/screens-guided-pi-markdown.js")
    script = f"""
      global.window = {{ EU_LANG: 'en' }};
      eval({_ESCAPE_OWNER!r});
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


def test_copilot_next_step_owner_projects_clickable_choices_and_safe_fallback() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const choices = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '已确认 MIMIC-IV v3.1。\\n**下一步：**\\n请选择研究单位。\\n' +
        '- 所有符合条件的 ICU stays\\n- 每位患者首次 ICU stay'
      );
      const fallback = window.EU_GUIDED_PI_NEXT_ACTIONS.project('成人是否定义为 **年龄 ≥18 岁？**');
      const generic = window.EU_GUIDED_PI_NEXT_ACTIONS.project('研究配置已经保存。');
      const inline = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '定义已保存。\\n**下一步：请选择主要结局：**\\n- ICU 内死亡\\n- 住院死亡'
      );
      const markdownHeading = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        'Demo 仅为样本。\\n### 下一步：\\n请选择一个选项：\\n' +
        '- 选择 MIMIC-IV Demo\\n- 选择 eICU Demo\\n- 继续无数据规划'
      );
      const localSource = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '数据源待确认。\\n**下一步：**\\n' +
        '- 使用推荐的 MIMIC-IV v3.1 数据导出\\n' +
        '- 选择并注册本机上的其他 MIMIC-IV v3.1 数据目录'
      );
      const databases = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择数据库。\\n**下一步：**\\n' +
        '- 使用 MIMIC-IV v3.1\\n- 使用 eICU v2.0\\n' +
        '- 使用 AmsterdamUMCdb\\n- 使用 HiRID v1.1.1\\n' +
        '- 使用 MIMIC-III v1.4\\n- 使用 SICdb v1.0.6'
      );
      const compactDatabases = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择一个确切的数据库版本。\\n**下一步：**\\n' +
        '-选择 MIMIC-IV v3.1\\n-选择 eICU v2.0\\n' +
        '-选择 AmsterdamUMCdb\\n-选择 HiRID v1.1.1\\n' +
        '-选择 MIMIC-III v1.4\\n-选择 SICdb v1.0.6'
      );
      const genericSix = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择一项。\\n**下一步：**\\n' +
        '- 选项一\\n- 选项二\\n- 选项三\\n- 选项四\\n- 选项五\\n- 选项六'
      );
      console.log(JSON.stringify({{choices, fallback, generic, inline, markdownHeading, databases, compactDatabases, genericSix}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(choices, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(
        {{body: '请选择研究单位。', prompt: '', choices: ['首次 ICU stay']}},
        {{language: 'zh'}}
      ));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(markdownHeading, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(localSource, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(databases, {{language: 'zh'}}));
      console.log('COMPACT=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(compactDatabases, {{language: 'zh'}}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render({{
        body: '待确认的研究设定',
        prompt: '请选择一项：',
        choices: [
          '我确认采用成人 ICU 人群、ICU stay 分析单位、Sepsis-3 主要暴露、ICU 住院期间死亡结局和入科后 24 小时外层特征窗，并授权 EasyICU 准备数据。',
          '我想修改或增加一项关键研究要求。',
        ],
      }}, {{language: 'zh'}}));
      console.log('RESOLVED=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(localSource, {{
        language: 'zh',
        dataSourceAuthorization: {{
          status: 'confirmed',
          source: {{label: 'MIMIC-IV', reference_release: '3.1'}},
        }},
      }}));
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(generic, {{language: 'zh'}}));
      console.log('SUPPRESSED=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(
        generic,
        {{language: 'zh', suppressFallback: true}}
      ));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert '"body":"已确认 MIMIC-IV v3.1。"' in completed.stdout
    assert '"choices":["所有符合条件的 ICU stays","每位患者首次 ICU stay"]' in completed.stdout
    assert '"prompt":"请选择主要结局："' in completed.stdout
    assert '"prompt":"请选择主要结局：**"' not in completed.stdout
    assert '"explicit":false' in completed.stdout
    assert '"asking":false' in completed.stdout
    assert 'data-gpi-next-choice="所有符合条件的 ICU stays"' in completed.stdout
    assert '"choices":["选择 MIMIC-IV Demo","选择 eICU Demo","继续无数据规划"]' in completed.stdout
    assert 'data-gpi-next-choice="确认并授权本轮准备并注册 MIMIC-IV Demo。"' in completed.stdout
    assert "下载并准备 MIMIC-IV Demo" in completed.stdout
    assert 'data-gpi-next-choice="继续无数据规划"' in completed.stdout
    assert 'data-gpi-next-local-database="miiv"' in completed.stdout
    assert '"choices":["使用 MIMIC-IV v3.1","使用 eICU v2.0","使用 AmsterdamUMCdb","使用 HiRID v1.1.1","使用 MIMIC-III v1.4","使用 SICdb v1.0.6"]' in completed.stdout
    assert 'data-gpi-next-choice="使用 MIMIC-III v1.4"' in completed.stdout
    assert 'data-gpi-next-choice="使用 SICdb v1.0.6"' in completed.stdout
    assert '"compactDatabases":{"body":"请选择一个确切的数据库版本。","prompt":"","choices":["选择 MIMIC-IV v3.1","选择 eICU v2.0","选择 AmsterdamUMCdb","选择 HiRID v1.1.1","选择 MIMIC-III v1.4","选择 SICdb v1.0.6"]' in completed.stdout
    assert 'COMPACT=<section class="gpi-next-step"' in completed.stdout
    assert 'data-gpi-next-choice="选择 MIMIC-IV v3.1"' in completed.stdout
    assert '"choices":["选项一","选项二","选项三","选项四"]' in completed.stdout
    assert '"选项五"' not in completed.stdout
    assert 'data-gpi-next-resolved-source' in completed.stdout
    assert 'data-gpi-data-source-continue' in completed.stdout
    assert '已接入 MIMIC-IV v3.1' in completed.stdout
    assert '尚未开始数据提取或分析' in completed.stdout
    assert '查看数据准备确认' in completed.stdout
    assert '不会在这里生成研究计划' in completed.stdout
    assert '>确认并准备数据<' in completed.stdout
    assert 'data-gpi-next-choice="我确认采用成人 ICU 人群、ICU stay 分析单位、Sepsis-3 主要暴露、ICU 住院期间死亡结局和入科后 24 小时外层特征窗，并授权 EasyICU 准备数据。"' in completed.stdout
    assert "EasyICU 会直接执行对应操作或继续对话" in completed.stdout
    assert "其他，我自己输入" in completed.stdout
    assert 'class="gpi-next-custom"' in completed.stdout
    assert 'data-gpi-next-custom-form' in completed.stdout
    assert 'data-gpi-next-custom-input' in completed.stdout
    assert 'placeholder="输入其他选择或补充说明"' in completed.stdout
    assert '<button type="submit"' in completed.stdout
    assert "继续对话" in completed.stdout
    assert "SUPPRESSED=" in completed.stdout
    assert "SUPPRESSED=<" not in completed.stdout
    assert "<script>" not in completed.stdout


def test_planner_confirmation_owns_the_only_next_actions() -> None:
    screen = _read("js/screens-guided-pi.js")
    owner = _read("js/screens-guided-pi-next-actions.js")
    assert "provider_ready_to_generate_plan" in screen
    assert "workflowActionCode:" in screen
    assert "suppressFallback:" in screen
    assert "options.suppressFallback && !step.explicit" in owner
    assert "data-gpi-premature-plan-guard" in owner


def test_model_plan_choice_is_replaced_until_workflow_owner_is_ready() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const projected = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '研究问题已明确。\\n**下一步：**\\n' +
        '- 生成正式研究计划，由 EasyICU 说明依据\\n' +
        '- 先补充研究要求'
      );
      const authorization = {{status: 'confirmed', source: {{label: 'MIMIC-IV'}}}};
      console.log('BLOCKED=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{
        language: 'zh', dataSourceAuthorization: authorization,
        workflowActionCode: 'study_setup_incomplete',
      }}));
      console.log('READY=' + window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{
        language: 'zh', dataSourceAuthorization: authorization,
        workflowActionCode: 'provider_ready_to_generate_plan',
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    blocked, ready = completed.stdout.split("READY=", 1)
    assert "data-gpi-premature-plan-guard" in blocked
    assert "查看数据准备确认" in blocked
    assert "生成正式研究计划" not in blocked
    assert "data-gpi-premature-plan-guard" not in ready
    assert "生成正式研究计划" in ready


def test_model_plan_choice_can_only_receive_provider_grant_from_plan_workflow() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const grants = window.EU_GUIDED_PI_NEXT_ACTIONS.governedPlanGrants;
      console.log(JSON.stringify(grants(
        '授权基于当前 E2 StudyContext 生成新的正式研究计划，并在分析前暂停审核',
        'plan_configuration_superseded'
      )));
      console.log(JSON.stringify(grants(
        '按科学审阅要求生成新正式研究计划并在分析前暂停审核',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '在 EasyICU 主机确认一次性 provider_run 授权已注入当前回合后，重新发送修订请求',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '暂不生成修订版正式研究计划',
        'plan_scientific_changes_required'
      )));
      console.log(JSON.stringify(grants(
        '授权基于当前 E2 StudyContext 生成新的正式研究计划，并在分析前暂停审核',
        'study_setup_incomplete'
      )));
      console.log(JSON.stringify(grants(
        '在主机授权卡中启用一次性 provider_run grant',
        'plan_configuration_superseded'
      )));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert completed.stdout.splitlines() == [
        '["provider_run"]',
        '["provider_run"]',
        '["provider_run"]',
        "[]",
        "[]",
        "[]",
    ]


def test_scientific_plan_revision_is_submitted_directly_with_reviewed_run() -> None:
    guided = _read("js/screens-guided-pi.js")

    assert "'plan_scientific_changes_required'," in guided
    assert "plan_revision_source_run_id: reviewedRunId" in guided
    assert "reasonCode === 'plan_scientific_changes_required' && reviewedRunId" in guided


def test_demo_next_step_is_one_click_and_supports_an_existing_local_copy() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const projected = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '官方 Demo 尚未准备。\\n**下一步：**\\n' +
        '- 授权下载并准备官方 MIMIC-IV Demo\\n' +
        '- 暂不下载，继续完善研究设计'
      );
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{ language: 'zh' }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert 'data-gpi-next-grants="extract"' in completed.stdout
    assert "确认并授权本轮准备并注册" in completed.stdout
    assert "使用已经下载好的 MIMIC-IV Demo" in completed.stdout
    assert 'data-gpi-next-local-database="miiv"' in completed.stdout
    assert "暂不下载，继续完善研究设计" in completed.stdout


def test_recommended_prepared_export_choice_receives_configuration_grant() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const projected = window.EU_GUIDED_PI_NEXT_ACTIONS.project(
        '请选择数据来源。\\n**下一步：**\\n' +
        '- 使用推荐的 EasyICU 本地导出（MIMIC-IV v3.1，94,458 个 ICU stay，19 个模块）\\n' +
        '- 选择并注册其他本地 MIMIC-IV v3.1 数据目录'
      );
      console.log(window.EU_GUIDED_PI_NEXT_ACTIONS.render(projected, {{ language: 'zh' }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert 'data-gpi-next-grants="configure"' in completed.stdout
    assert 'data-gpi-next-local-database="miiv"' in completed.stdout


def test_preview_plan_confirmation_routes_to_data_preparation_not_same_plan_runner() -> None:
    shell = _read("js/screens-guided-pi.js")
    confirmation = _read("js/screens-guided-pi-confirmation.js")

    branch = shell.split(
        "if (confirmation.code === 'plan_execution_upgrade_required')", 1
    )[1].split("if ([", 1)[0]
    assert "sendText(confirmation.message, confirmation.grants)" in branch
    assert "startCurrentFormalPlanGeneration" not in branch
    assert "const projectedAllowlist = new Set(['extract', 'configure']);" in shell
    assert "['configure', 'extract']" in confirmation
    assert "候选研究计划已生成，请确认数据准备" in confirmation
    assert "并不代表数据包已经准备好" in confirmation
    assert "确认方案并准备数据" in confirmation


def test_copilot_public_projection_hides_internal_pi_codes_and_owner_paths() -> None:
    owner = _read("js/screens-guided-pi.js")
    activity = _read("js/screens-guided-pi-activity.js")

    assert "function publicAssistantText" in owner
    assert "pi_action_authorization_required" in owner
    assert "EasyICU 内部状态" in owner
    assert "本轮一次性数据准备授权" in owner
    assert "官方 Demo 准备流程" in owner
    assert "const meta = stepDuration(step);" in activity
    assert "step.code, step.owner" not in activity
    assert "easyicu_prepare_demo_source: tr('Download and prepare official demo data'" in activity


def test_copilot_message_owner_wires_only_latest_next_step_to_send_or_focus() -> None:
    owner = _read("js/screens-guided-pi.js")
    data_binding_owner = _read("js/screens-guided-pi-data-binding.js")
    css = _read("css/guided-pi.css")

    assert "window.EU_GUIDED_PI_NEXT_ACTIONS" in owner
    assert "row.complete !== false" in owner
    assert "row === latestAssistant && !interactionLocked && !stale" in owner
    assert "sendText(message, governedNextChoiceGrants(nextChoice, message))" in owner
    assert "function governedNextChoiceGrants(element, message)" in owner
    assert "nextOwner.governedPlanGrants(message, code)" in owner
    assert "event.target.closest('[data-gpi-next-focus]')" in owner
    assert "event.target.closest('[data-gpi-next-custom-form]')" in owner
    # The free-text box goes through the same governed grant decision as a
    # choice button. It used to send `[]`, which stripped the turn's grants and
    # made "其他，我自己输入" fail authorization where the button beside it
    # succeeded.
    assert "sendText(message, governedNextChoiceGrants(null, message))" in owner
    assert "sendText(message, [])" not in owner
    assert "dataSourceAuthorization: DATA_CONSENT && DATA_CONSENT.authorization(state.session)" in owner
    assert "event.target.closest('[data-gpi-data-source-continue]')" in owner
    assert "continueAfterDataSourceConfirmation()" in owner
    continuation = owner.split("async function continueAfterDataSourceConfirmation()", 1)[1].split(
        "async function sendMessage()", 1
    )[0]
    assert "await sendText(" in continuation
    assert "'advance_after_data_source_confirmation'" in continuation
    assert "false," in continuation
    assert "regenerateMessage(" not in continuation
    assert owner.index("${dataConsentHtml}") > owner.index("data-gpi-log")
    assert "if (await continueAfterDataSourceConfirmation()) return true;" in data_binding_owner
    next_actions = _read("js/screens-guided-pi-next-actions.js")
    assert "查看数据准备确认" in next_actions
    assert "只会先汇总数据准备所需的最少信息" in next_actions
    assert "不会在这里生成研究计划" in next_actions
    assert "不必等待全量数据提取" in next_actions
    assert "继续到下一个关键决策" not in next_actions
    assert ".gpi-next-step" in css
    assert ".gpi-next-actions" in css
    assert ".gpi-next-custom{flex:1 0 100%" in css
    assert ".gpi-next-custom-row input{min-width:0;flex:1" in css
    for non_owner in ("css/guided.css", "css/guided-projects.css", "css/guided-startup.css"):
        assert ".gpi-next-custom" not in _read(non_owner)


def test_successful_local_source_action_opens_native_workspace_immediately() -> None:
    owner = _read("js/screens-guided-pi.js")

    assert "event.code || '') === 'easyicu_local_source_workspace_ready'" in owner
    assert "resource && resource.kind === 'native_workspace'" in owner
    assert "window.EU_GUIDED_PI_PREVIEW.open(localWorkspace, projectId())" in owner
    assert "nextChoice.dataset.gpiNextLocalDatabase" in owner
    assert "authorizeDataSource('begin_local_selection', { database: localDatabase })" in owner


def test_copilot_message_action_owner_renders_copy_edit_and_latest_retry() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
      }});
      console.log(actions.render(
        {{id: 'u1', role: 'user', text: '原问题', complete: true}},
        {{allowEdit: true, canEdit: true}}
      ).actionsHtml);
      console.log(actions.render(
        {{id: 'u1', role: 'user', text: '原问题', complete: true}},
        {{editing: true, allowEdit: true, canEdit: true}}
      ).editorHtml);
      console.log(actions.render(
        {{id: 'a1', role: 'assistant', text: '回答', complete: true}},
        {{canRetry: true, retryUserEntryId: 'entry-u1'}}
      ).actionsHtml);
      console.log(actions.render(
        {{id: 'a0', role: 'assistant', text: '旧回答', complete: true}},
        {{canRetry: false, retryUserEntryId: 'entry-u0'}}
      ).actionsHtml);
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert "data-gpi-message-copy" in completed.stdout
    assert "data-gpi-message-edit" in completed.stdout
    assert "这条之后的回答会被替换；原分支仍保留在历史中可恢复。" in completed.stdout
    assert completed.stdout.count("data-gpi-message-retry") == 1


def test_editing_an_earlier_turn_rewinds_instead_of_appending() -> None:
    """Editing message N must replace what follows N, not append at the bottom.

    Regression: the edit submit called sendText, so an edited earlier turn was
    posted as a brand-new message while the original answers stayed below it.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const calls = [];
      const rows = [
        {{id: 'u1', role: 'user', text: '第一个问题', entryId: 'entry-u1', complete: true}},
        {{id: 'a1', role: 'assistant', text: '第一个回答', complete: true}},
        {{id: 'u2', role: 'user', text: '第二个问题', entryId: 'entry-u2', complete: true}},
        {{id: 'a2', role: 'assistant', text: '第二个回答', complete: true}},
      ];
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
        rows: () => rows,
        setEditing: () => {{}},
        sendText: value => calls.push(['sendText', value]),
        regenerate: (entryId, text, intent, target) =>
          calls.push(['regenerate', entryId, text, intent, target]),
      }});
      const form = {{
        dataset: {{ gpiMessageEditForm: 'u1' }},
        querySelector: () => ({{ value: '  改过的第一个问题  ' }}),
      }};
      actions.handleSubmit({{
        preventDefault: () => {{}},
        target: {{ closest: sel => (sel === '[data-gpi-message-edit-form]' ? form : null) }},
      }});
      console.log(JSON.stringify(calls));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    calls = json.loads(completed.stdout)
    # It rewinds to the edited turn with the new text, targeting that turn's
    # own answer -- it does not append a fresh message at the bottom.
    assert calls == [
        ["regenerate", "entry-u1", "改过的第一个问题", "user_edited_message", "a1"]
    ]


def test_editing_host_generated_plan_action_resubmits_without_appending() -> None:
    """A reloaded host plan action keeps direct routing despite a Pi entry id."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-message-actions.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const calls = [];
      const row = {{
        id: 'history-9', entryId: 'entry-plan-9', role: 'user',
        text: '重新生成研究计划', complete: true,
      }};
      const actions = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({{
        tr: (en, zh) => zh,
        iconHtml: name => `<i>${{name}}</i>`,
        rows: () => [row],
        setEditing: () => {{}},
        sendText: value => calls.push(['sendText', value]),
        regenerate: (...args) => calls.push(['regenerate', ...args]),
        resubmitHostGenerated: (target, text) => {{
          calls.push(['resubmitHostGenerated', target.id, text]);
          return true;
        }},
      }});
      const form = {{
        dataset: {{ gpiMessageEditForm: 'history-9' }},
        querySelector: () => ({{ value: '  重新生成研究计划  ' }}),
      }};
      actions.handleSubmit({{
        preventDefault: () => {{}},
        target: {{ closest: sel => (sel === '[data-gpi-message-edit-form]' ? form : null) }},
      }});
      console.log(JSON.stringify(calls));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert json.loads(completed.stdout) == [
        ["resubmitHostGenerated", "history-9", "重新生成研究计划"]
    ]


def test_copilot_message_actions_are_host_wired_without_history_rewrite() -> None:
    owner = _read("js/screens-guided-pi.js")
    css = _read("css/guided-pi.css")

    assert "window.EU_GUIDED_PI_MESSAGE_ACTIONS.create" in owner
    assert "state.editingMessageId === row.id" in owner
    assert "MESSAGE_ACTIONS.handleClick(event)" in owner
    assert "MESSAGE_ACTIONS.handleSubmit(event)" in owner
    message_owner = _read("js/screens-guided-pi-message-actions.js")
    assert "copyText(row.text)" in message_owner
    # Editing an earlier turn rewinds to it, so the replies after it are
    # replaced instead of a new message being appended at the bottom.  The Pi
    # branch keeps the original recoverable, so this is not a history rewrite.
    assert (
        "context.regenerate(entryId, text, 'user_edited_message', followingAssistantId(id))"
        in message_owner
    )
    assert "context.regenerate(user.entryId, user.text, '', articleId(retry))" in message_owner
    # A host-generated action is replaced in place; other edits with no
    # resolvable turn identifier must still not vanish.
    assert "context.resubmitHostGenerated(row, text)" in message_owner
    assert "context.sendText(text);" in message_owner
    assert "resubmitHostGenerated: resubmitHostGeneratedMessage" in owner
    assert "regeneratePiCopilotMessage" in owner
    assert "user_entry_id: entryId" in owner
    assert "advance_after_data_source_confirmation" in owner
    assert "dataSourceContinuationTarget" not in owner
    assert "turnGrants().filter(action => action === 'configure')" in owner
    assert "nextOwner.governedPlanGrants(text, workflowCode)" in owner
    assert "regenerationIntent === 'user_edited_message'" in owner
    assert "editedPlanGrants.includes('provider_run')" in owner
    assert "message: text, allowed_actions: replayGrants" in owner
    assert "turn_intent: replayIntent" in owner
    assert "regeneration_intent: regenerationIntent" in owner
    assert "interactive: row === latestAssistant && !interactionLocked && !stale" in owner
    assert ".gpi-message.user .gpi-message-actions{right:0;opacity:0}" in css
    assert ".gpi-message-editor" in css


def test_candidate_plan_can_be_explicitly_regenerated_before_data_preparation() -> None:
    """The review/data-prep gate must allow replacing, not executing, a plan."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-next-actions.js")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value || '') }} }};
      eval({json.dumps(source)});
      const grants = window.EU_GUIDED_PI_NEXT_ACTIONS.governedPlanGrants(
        '重新生成研究计划', 'plan_execution_upgrade_required'
      );
      console.log(JSON.stringify(grants));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == ["provider_run"]


def test_copilot_regeneration_projects_activity_and_answer_in_place() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    source = _read("js/screens-guided-pi-regeneration.js")
    script = f"""
      global.window = {{}};
      eval({json.dumps(source)});
      const owner = window.EU_GUIDED_PI_REGENERATION;
      const rows = [
        {{id: 'u1', role: 'user', text: 'question', entryId: 'entry-u1'}},
        {{id: 'activity-1', role: 'activity', status: 'complete'}},
        {{id: 'a1', role: 'assistant', text: 'old answer', complete: true}},
        {{id: 'u2', role: 'user', text: 'later question', entryId: 'entry-u2'}},
        {{id: 'activity-2', role: 'activity', status: 'complete'}},
        {{id: 'a2', role: 'assistant', text: 'later answer', complete: true}},
      ];
      const replacement = owner.create(rows, {{
        userEntryId: 'entry-u1', targetMessageId: 'a1', startedAt: 1234,
      }});
      console.log(JSON.stringify({{
        targetMessageId: replacement.targetMessageId,
        targetActivityId: replacement.targetActivityId,
        projectedRoles: owner.visibleRows(rows, replacement).map(row => owner.project(row, replacement).role),
        projectedTexts: owner.visibleRows(rows, replacement).map(row => owner.project(row, replacement).text || ''),
        projectedStatuses: owner.visibleRows(rows, replacement).map(row => owner.project(row, replacement).status || ''),
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "targetMessageId": "a1",
        "targetActivityId": "activity-1",
        "projectedRoles": ["user", "activity", "assistant"],
        "projectedTexts": ["question", "", ""],
        "projectedStatuses": ["", "running", ""],
    }
    owner = _read("js/screens-guided-pi.js")
    assert "REGENERATION.visibleRows(fullTimeline, state.regeneration)" in owner
    assert "REGENERATION.project(row, state.regeneration)" in owner
    assert "state.regeneration.message" in owner
    assert "state.regeneration.activity" in owner


def test_literature_preview_hides_execution_steps_and_explains_scientific_use() -> None:
    literature_owner = _read("js/screens-guided-pi-literature.js")

    assert "文献具体影响了计划的哪里" in literature_owner
    assert "为什么采用" in literature_owner
    assert "与研究问题直接相关的文章" in literature_owner
    assert "尚未执行检索。请重新生成计划" in literature_owner
    assert "目前 0 篇通过筛选" not in literature_owner
    assert "辅助执行或呈现步骤" not in literature_owner
    assert "planned_analysis_role" not in literature_owner.split(
        "function planMap(payload)", 1
    )[1].split("function articleGroup", 1)[0]


def test_literature_preview_renders_unsearched_bundle_without_zero_result_claim() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({owner!r});
      const html = window.EU_GUIDED_PI_LITERATURE.renderArtifact({{
        research_question: '乳酸与院内死亡',
        search: {{ search_conducted: false }},
        citations: [],
        direct_comparator_keys: [],
      }});
      console.log(html);
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )

    assert "尚未执行针对这个问题的文献检索" in completed.stdout
    assert "尚未执行检索。请重新生成计划" in completed.stdout
    assert "本次检索暂无文章通过筛选" not in completed.stdout


def test_literature_preview_receives_current_workflow_status_from_guided_owner() -> None:
    guided = _read("js/screens-guided-pi.js")
    preview = _read("js/screens-guided-pi-preview.js")
    confirmation = _read("js/screens-guided-pi-confirmation.js")

    assert "function previewWorkflowContext()" in guided
    assert "EU_GUIDED_PI_PREVIEW.setWorkflowContext(previewWorkflowContext())" in guided
    assert "RESOURCE_OWNER.fromButton(resource), projectId(), previewWorkflowContext()" in guided
    assert "state.workflow.active_job = (payload && payload.active_job) || { present: false }" in guided
    assert "setWorkflowContext" in preview
    assert "literature.renderArtifact(state.payload || {}, {" in preview
    assert "runId: state.resource.run_id" in preview
    assert "修订版计划未通过科学合同" in confirmation
    assert "患者分析尚未开始" in confirmation


def test_guided_analysis_outcome_stays_visible_after_refresh() -> None:
    guided = _read("js/screens-guided-pi.js")
    replay = _read("js/screens-guided-pi-replay.js")
    owner = _read("js/screens-guided-pi-run-outcome.js")
    index = _read("index.html")

    assert "state.latestRun = payload && payload.latest_run" in guided
    assert "RUN_OUTCOME.render(state.latestRun, state.workflow)" in guided
    assert "analysis_results_available" in replay
    assert "分析已完成；仍需完成投稿审阅" in replay
    assert "分析已完成，可以查看结果" in owner
    assert "result_tables.json" in owner
    assert "figure_gallery.json" in owner
    assert "manuscript_provenance.json" in owner
    assert "data-gpi-run-outcome-data" in owner
    assert "preparePiCopilotDataWorkbenchSnapshot" in owner
    assert "RUN_OUTCOME.openData(previewAnalysisData)" in guided
    assert "screens-guided-pi-run-outcome.js" in index
    assert "guided-pi-run-outcome.css" in index


def test_workspace_preview_hard_codes_unvalidated_authority_and_iframe_sandbox() -> (
    None
):
    preview = _read("js/screens-guided-pi-preview.js")
    assert "authority_class: 'workspace_artifact'" in preview
    assert "scientific_evidence: false" in preview
    assert "validation_status: 'unvalidated'" in preview
    assert "claim_ceiling: 'unsupported'" in preview
    assert "checked_sha256: checkedSha256" in preview
    assert "state.resource.checked_sha256" in preview
    assert "Workspace artifact · Unvalidated" in preview
    assert "scientific evidence" in preview
    assert 'sandbox="allow-scripts"' in preview
    assert 'referrerpolicy="no-referrer"' in preview
    assert "EasyICU run artifact · Analysis-only" in preview
    assert "EasyICU run artifact · Reportable" not in preview
    assert "Human sign-off required" in preview
    assert "state.governance" in preview


def test_workspace_preview_never_requests_an_empty_checked_digest() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-preview.js")
    digest = "c" * 64
    script = f"""
      const calls = [];
      global.window = {{
        EU_LANG: 'en',
        EU_API: {{
          piCopilotWorkspacePreviewUrl(projectId, file, checkedSha256) {{
            calls.push([projectId, file, checkedSha256]);
            return '/preview?checked_sha256=' + checkedSha256;
          }},
        }},
      }};
      global.document = {{ getElementById() {{ return null; }} }};
      eval({_ESCAPE_OWNER!r});
      eval({source!r});
      const host = {{
        hidden: false,
        innerHTML: '',
        addEventListener() {{}},
        replaceChildren() {{ this.innerHTML = ''; }},
      }};
      window.EU_GUIDED_PI_PREVIEW.mount(host);
      window.EU_GUIDED_PI_PREVIEW.open({{
        kind: 'file', file: 'prototype/index.html', media_type: 'text/html',
      }}, 'project-demo');
      console.log(String(host.innerHTML.includes('data-gpi-preview-mode="web"')));
      console.log(String(calls.length));
      window.EU_GUIDED_PI_PREVIEW.open({{
        kind: 'webpage', file: 'prototype/index.html', media_type: 'text/html',
      }}, 'project-demo');
      console.log(String(calls.length));
      console.log(String(host.innerHTML.includes('<iframe')));
      console.log(String(host.innerHTML.includes('checked file digest is missing')));
      window.EU_GUIDED_PI_PREVIEW.open({{
        kind: 'webpage', file: 'prototype/index.html', media_type: 'text/html',
        checked_sha256: '{digest}',
      }}, 'project-demo');
      console.log(String(calls.length));
      console.log(String(host.innerHTML.includes('{digest}')));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "false",
        "0",
        "0",
        "false",
        "true",
        "1",
        "true",
    ]


def test_preview_keeps_bounded_project_scoped_recent_resources() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    preview = _read("js/screens-guided-pi-preview.js")
    resources = _read("js/screens-guided-pi-resources.js")
    digest_a = "a" * 64
    digest_b = "b" * 64
    script = f"""
      global.window = {{ EU_LANG: 'en', EU_API: {{}} }};
      global.document = {{ getElementById() {{ return null; }} }};
      eval({_ESCAPE_OWNER!r});
      eval({resources!r});
      eval({preview!r});
      const host = {{
        hidden: false,
        innerHTML: '',
        addEventListener() {{}},
        replaceChildren() {{ this.innerHTML = ''; }},
      }};
      window.EU_GUIDED_PI_PREVIEW.mount(host);
      const first = {{
        kind: 'webpage', file: 'reports/cohort.html', label: 'Cohort review',
        media_type: 'text/html', checked_sha256: '{digest_a}',
      }};
      const second = {{
        kind: 'webpage', file: 'reports/timeline.html', label: 'Timeline review',
        media_type: 'text/html', checked_sha256: '{digest_b}',
      }};
      window.EU_GUIDED_PI_PREVIEW.open(first, 'project-a');
      console.log(String(host.innerHTML.includes('gpi-preview-recent')));
      window.EU_GUIDED_PI_PREVIEW.open(second, 'project-a');
      console.log(String(host.innerHTML.includes('Cohort review')));
      console.log(String(host.innerHTML.includes('Timeline review')));
      window.EU_GUIDED_PI_PREVIEW.close();
      window.EU_GUIDED_PI_PREVIEW.open(first, 'project-a');
      console.log(String(host.innerHTML.includes('Timeline review')));
      window.EU_GUIDED_PI_PREVIEW.open(first, 'project-b');
      console.log(String(host.innerHTML.includes('gpi-preview-recent')));
    """
    completed = subprocess.run(
        [node, "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["false", "true", "true", "true", "false"]


def test_continuing_a_conversation_does_not_close_the_open_preview() -> None:
    owner = _read("js/screens-guided-pi.js")
    send_text = owner.split("async function sendText", 1)[1].split(
        "async function sendMessage", 1
    )[0]

    assert "state.currentTurnResources = []" in send_text
    assert "render();" in send_text
    assert "EU_GUIDED_PI_PREVIEW.close" not in send_text
    assert "EU_GUIDED_PI_PREVIEW.clearProject" not in send_text


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
