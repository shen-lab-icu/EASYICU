/* Pi AgentSession client for Guided Copilot.
   Owner: Pi chat/session/tool UX only. Study cards and scientific execution
   remain in their existing EasyICU owners. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  const DATA_CONSENT = window.EU_GUIDED_PI_DATA_CONSENT;
  const STARTERS = window.EU_GUIDED_PI_STARTERS;
  const HEADER = window.EU_GUIDED_PI_HEADER;
  const REGENERATION = window.EU_GUIDED_PI_REGENERATION;

  const state = {
    host: null, conv: null, runtime: null, sessions: [], session: null,
    messages: [], loading: true, creating: false, busy: false, jobId: '',
    source: null, childSource: null, childJobId: '', error: '', shell: 'pi', draft: '', setupSaving: false,
    showSetup: false, availableModels: [], project: null,
    researchProvider: 'codex', researchModel: '', codexAuth: null,
    codexLogin: null, codexModels: [], codexBusy: false, codexPoll: null,
    projectInitialization: null, projectIssue: '', workflow: null, latestRun: null,
    projectLoading: false, projectDiscoveryLoading: false,
    agentMode: 'research', accessMode: 'assist', pendingAuthorityRebind: false,
    demoMode: false, demoScrollTopPending: false, currentTurnResources: [],
    workflowReceipts: [], editingMessageId: '', sessionSelectionRevision: 0,
    pendingLanguageReload: false, regenerating: false, regeneration: null,
    startupPromise: null, projectPreparePromise: null, projectPrepareId: '',
  };

  const ACCESS_MODE_GRANTS = Object.freeze({
    ask: Object.freeze([]),
    assist: Object.freeze(['idea', 'literature', 'configure', 'run', 'workspace_write', 'mcp_read']),
    full: Object.freeze(['idea', 'literature', 'configure', 'extract', 'run', 'provider_run', 'cancel', 'workspace_write', 'mcp_read']),
  });

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function publicAssistantText(value) {
    return String(value || '')
      .replace(/\b(?:one[-\s]time\s+)?extraction grant\b/gi, tr(
        'one-time data preparation authorization',
        '本轮一次性数据准备授权',
      ))
      .replace(/\beasyicu_prepare_demo_source\b/gi, tr(
        'the official demo preparation workflow',
        '官方 Demo 准备流程',
      ))
      .replace(/\bpi_action_authorization_required\b/gi, tr(
        'the required EasyICU data authorization is not active for this turn',
        '本轮数据准备授权尚未生效',
      ))
      .replace(/\bpi_[a-z0-9_]+\b/gi, tr('an EasyICU internal status', 'EasyICU 内部状态'))
      .replace(/\beasyicu\.webserver\.pi_copilot(?:\.[a-z0-9_.]+)?\b/gi, 'EasyICU Copilot');
  }
  function assistantTextHtml(value) {
    const renderer = window.EU_GUIDED_PI_MARKDOWN;
    return renderer && typeof renderer.render === 'function'
      ? renderer.render(value)
      : esc(value).replace(/\n/g, '<br>');
  }
  function api() { return window.EU_API || {}; }
  function isStaticPreview() { return window.location && window.location.protocol === 'file:'; }
  function runtimeReady() { return !!(state.runtime && state.runtime.status === 'ready'); }
  function shellReady() {
    return !!(state.runtime && (state.runtime.shell_ready === true || state.runtime.status === 'ready'));
  }
  function apiResearchReady() {
    const runtime = state.runtime || {};
    const config = runtime.configuration || {};
    return runtimeReady() && (config.api_transport || runtime.api_transport) === 'openai-completions';
  }
  function restoreConfiguredResearchProvider() {
    const runtime = state.runtime || {};
    const config = runtime.configuration || {};
    if (
      config.connection_verified === true
      && config.credential_present === true
      && (config.api_transport || runtime.api_transport) === 'openai-completions'
    ) {
      state.researchProvider = 'api';
    }
  }
  function connectionConfigured() {
    if (state.researchProvider !== 'codex') return apiResearchReady();
    return !!(
      state.runtime && state.runtime.status !== 'unavailable'
      && state.codexAuth && state.codexAuth.authentication_verified
      && state.researchModel
    );
  }
  function connectionReady() {
    return state.researchProvider === 'codex'
      ? connectionConfigured() && shellReady()
      : apiResearchReady();
  }
  function projectId() { return String((state.project && state.project.id) || '').trim(); }
  function previewWorkflowContext() {
    const archived = Array.isArray(state.session && state.session.archived_child_jobs)
      ? state.session.archived_child_jobs : [];
    const failedJob = archived.slice().reverse().find(job => (
      job && job.kind === 'agent-run' && job.status === 'failed'
    )) || null;
    return {
      nextActionCode: String((state.workflow && state.workflow.next_action_code) || ''),
      currentRunId: String((state.session && state.session.binding && state.session.binding.run_id) || ''),
      activeJob: (state.workflow && state.workflow.active_job) || null,
      failedJob,
    };
  }
  function uiLanguage() { return window.EU_LANG === 'zh' ? 'zh' : 'en'; }
  function sessionLanguage(session) { return session && session.language === 'zh' ? 'zh' : 'en'; }
  function sessionMatchesUiLanguage(session) { return sessionLanguage(session) === uiLanguage(); }
  function displaySessionTitle(value) { return window.EU_PRODUCT_LABELS.copilotTitle(value); }
  function displayProjectTitle(value, fallback) { return window.EU_PRODUCT_LABELS.projectTitle(value, fallback); }
  function agentMode() {
    return (state.session && state.session.agent_mode) || state.agentMode || 'research';
  }
  function accessModeLabel(mode) {
    if (mode === 'ask') return tr('Ask first', '请求访问');
    if (mode === 'full') return tr('Full access', '完全访问');
    return tr('Auto-approve', '自动审批');
  }
  function turnGrants() {
    const grants = ACCESS_MODE_GRANTS[state.accessMode] || ACCESS_MODE_GRANTS.assist;
    return grants.filter(action => action !== 'workspace_write' || agentMode() === 'workspace');
  }
  function setShell(shell) {
    state.shell = shell === 'pi' ? 'pi' : 'legacy';
    if (state.conv) state.conv.classList.toggle('pi-active', state.shell === 'pi');
    render();
  }
  function rememberSession(id) {
    const key = projectId()
      ? 'easyicu_pi_copilot_session:' + encodeURIComponent(projectId()) + ':' + uiLanguage()
      : '';
    if (!key) return;
    try {
      if (id) localStorage.setItem(key, id);
      else localStorage.removeItem(key);
    } catch (e) {}
  }
  function rememberedSession() {
    const key = projectId()
      ? 'easyicu_pi_copilot_session:' + encodeURIComponent(projectId()) + ':' + uiLanguage()
      : '';
    if (!key) return '';
    try { return localStorage.getItem(key) || ''; } catch (e) { return ''; }
  }
  function errorText(error) {
    if (!error) return '';
    if (error.code === 'pi_session_authority_stale') {
      return tr('The study binding changed after this conversation was saved. Rebind it before continuing.', '这段对话保存后研究绑定发生了变化，请先重新绑定再继续。');
    }
    if (error.code === 'pi_provider_auth_failed') {
      return tr('The model service rejected this API credential.', '模型服务拒绝了这个 API 凭据，请检查后重试。');
    }
    if (error.code === 'pi_provider_model_unavailable') {
      return tr('The selected model was not reported by this service.', '该服务没有返回所选模型，请从下方发现的模型中选择。');
    }
    if (error.code === 'pi_provider_connection_failed') {
      return tr('EasyICU could not reach the model service.', 'EasyICU 无法连接到模型服务，请检查地址和服务状态。');
    }
    if (error.code === 'pi_session_project_mismatch') {
      return tr('That Copilot conversation belongs to another research project.', '该研究助手对话属于另一个研究项目，不能在当前项目中打开。');
    }
    if (error.code === 'pi_project_study_context_missing') {
      return tr('This project’s saved study setup no longer exists. Recreate or rebind the project before starting Copilot.', '当前项目保存的研究配置已不存在。请重新创建或绑定项目后再启动研究助手。');
    }
    if (error.code === 'pi_project_initialization_required') {
      return tr('Confirm this project’s study setup before starting Copilot.', '请先确认当前项目的研究配置，再启动研究助手。');
    }
    if (error.code === 'codex_auth_login_required') {
      return tr('Sign in with your ChatGPT account before starting this conversation.', '请先登录你的 ChatGPT 账户，再开始这段对话。');
    }
    if (error.code === 'codex_auth_model_unavailable') {
      return tr('That model is no longer available for this Codex account. Refresh the account model list.', '该 Codex 账户已无法使用这个模型，请刷新账户模型列表。');
    }
    if (isStaticPreview() && String(error.message || '').includes('Failed to fetch')) {
      return tr('This is a static preview without the EasyICU backend. Start EasyICU and open http://127.0.0.1:8765/#guided.', '这是不带 EasyICU 后端的静态预览。请启动 EasyICU，再打开 http://127.0.0.1:8765/#guided。');
    }
    return String(error.message || error.code || error);
  }

  function providerPreset(config, runtime) {
    const transport = config.api_transport || runtime.api_transport || 'openai-completions';
    const base = String(config.base_url || '').toLowerCase();
    if (transport === 'anthropic-messages') return 'anthropic';
    if (transport === 'google-generative-ai') return 'google';
    if (base.includes('api.openai.com')) return 'openai';
    if (base.includes('openrouter.ai')) return 'openrouter';
    if (base.includes('api.deepseek.com')) return 'deepseek';
    if (base.includes('127.0.0.1:8317') || base.includes('localhost:8317')) return 'cliproxyapi';
    return 'custom-openai';
  }

  function option(value, selected, label) {
    return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`;
  }
  function sessionIsStale() {
    return !!(state.session && state.session.stale && state.session.stale.stale);
  }

  function iconHtml(name, size) {
    return typeof window.icon === 'function' ? window.icon(name, size || 16, 1.55) : '';
  }
  const RESOURCE_OWNER = window.EU_GUIDED_PI_RESOURCES.create({ esc });
  const resourceName = RESOURCE_OWNER.name;
  const resourceKey = RESOURCE_OWNER.key;
  const resourceLabel = RESOURCE_OWNER.label;
  const resourceButton = RESOURCE_OWNER.button;
  const RUN_OUTCOME = window.EU_GUIDED_PI_RUN_OUTCOME.create({
    tr, esc, iconHtml, resourceButton, api, projectId,
    canPreview: () => Boolean(state.session) && !state.busy && !state.childJobId && !sessionIsStale(),
    preview: () => window.EU_GUIDED_PI_PREVIEW,
    workflowContext: previewWorkflowContext,
    errorText,
    onError: value => { state.error = value; render(); },
  });
  const ACTIVITY = window.EU_GUIDED_PI_ACTIVITY.create({
    tr, esc, iconHtml, resourceName, resourceKey, resourceButton,
  });
  const CONFIRMATION = window.EU_GUIDED_PI_CONFIRMATION.create({
    tr, esc, iconHtml, resourceButton, sessionIsStale,
    workflow: () => state.workflow,
    session: () => state.session,
    busy: () => state.busy || Boolean(state.childJobId),
  });
  const workflowConfirmation = CONFIRMATION.workflowConfirmation;
  const workflowConfirmationHtml = CONFIRMATION.workflowConfirmationHtml;
  const localizedAuthorizationQuestion = CONFIRMATION.localizedAuthorizationQuestion;
  const COHORT_ELIGIBILITY = window.EU_GUIDED_PI_COHORT_ELIGIBILITY.create({
    tr, esc,
    session: () => state.session,
    busy: () => state.busy || Boolean(state.childJobId),
    sessionIsStale,
  });
  const MESSAGE_ACTIONS = window.EU_GUIDED_PI_MESSAGE_ACTIONS.create({
    tr, iconHtml,
    rows: () => state.messages.concat(state.workflowReceipts),
    canEdit: () => !state.busy && !state.childJobId && !sessionIsStale(),
    setEditing: id => { state.editingMessageId = id; },
    renderHost: render,
    sendText,
    regenerate: regenerateMessage,
    resubmitHostGenerated: resubmitHostGeneratedMessage,
    host: () => state.host,
  });
  const { timeMs } = ACTIVITY;
  const TRANSCRIPT = window.EU_GUIDED_PI_TRANSCRIPT.create({
    activity: ACTIVITY, upsertActivityStep, timeMs, resourceKey, modelErrorText,
    workflowActionCode: () => String((state.workflow && state.workflow.next_action_code) || ''),
  });
  const transcriptMessages = TRANSCRIPT.transcriptMessages;
  const CHILDJOB = window.EU_GUIDED_PI_CHILDJOB.create({
    tr, activity: ACTIVITY, upsertActivityStep, api,
    render: () => render(),
    loadWorkflow: (...args) => loadWorkflow(...args),
    sessionIsStale: () => sessionIsStale(),
    rebind: () => rebind(),
    refreshSession: (...args) => refreshSession(...args),
    archiveChildJob: (...args) => archiveChildJob(...args),
    messages: () => state.messages,
    session: () => state.session,
    childJobId: () => state.childJobId,
    setChildJobId: value => { state.childJobId = value; },
    childSource: () => state.childSource,
    setChildSource: value => { state.childSource = value; },
  });
  const ASIDE = window.EU_GUIDED_PI_ASIDE.create({
    tr, esc, iconHtml,
    projectId: () => projectId(),
    displayProjectTitle: (...args) => displayProjectTitle(...args),
    demoMode: () => state.demoMode,
    project: () => state.project,
    shell: () => state.shell,
    workflow: () => state.workflow,
  });
  const syncProjectWorkflowAside = ASIDE.syncProjectWorkflowAside;
  const DATA_BINDING = window.EU_GUIDED_PI_DATA_BINDING.create({
    api,
    render: () => render(),
    projectId: () => projectId(),
    loadWorkflow: (...args) => loadWorkflow(...args),
    dataConsent: DATA_CONSENT,
    errorText,
    rememberSession,
    continueAfterDataSourceConfirmation,
    root: () => state.host,
    busy: () => state.busy,
    session: () => state.session,
    setSession: value => { state.session = value; },
    setError: value => { state.error = value; },
    workflowReceipts: () => state.workflowReceipts,
    setWorkflowReceipts: value => { state.workflowReceipts = value; },
  });
  const authorizeDataSource = DATA_BINDING.authorizeDataSource;
  const notifyExtractionHandoff = DATA_BINDING.notifyExtractionHandoff;
  const confirmDataSourceBinding = DATA_BINDING.confirmDataSourceBinding;
  const closeChildSource = CHILDJOB.closeChildSource;
  const childActivity = CHILDJOB.childActivity;
  const handleChildJobEvent = CHILDJOB.handleChildJobEvent;
  const watchChildJob = CHILDJOB.watchChildJob;
  const hydrateProjectedJob = CHILDJOB.hydrateProjectedJob;

  function activeActivity() {
    if (state.regenerating && state.regeneration && state.regeneration.activity.status === 'running') {
      return state.regeneration.activity;
    }
    return state.messages.slice().reverse().find(row => row.role === 'activity' && !row.childJobId && row.status === 'running');
  }
  function ensureActivity(at) {
    let row = activeActivity();
    if (!row) {
      const startedAt = timeMs(at);
      row = { id: 'activity-' + startedAt, role: 'activity', status: 'running', startedAt, steps: [], expanded: true };
      state.messages.push(row);
    }
    return row;
  }
  function upsertActivityStep(activity, step) {
    if (!activity) return;
    const found = activity.steps.find(item => item.id === step.id);
    if (found) Object.assign(found, step);
    else activity.steps.push(step);
  }
  function finishActivity(status, at, terminalKind) {
    const activity = activeActivity();
    if (!activity) return;
    const endedAt = timeMs(at);
    activity.steps.forEach(step => {
      if (step.status === 'running') {
        step.status = status === 'complete' ? 'complete' : 'error';
        step.endedAt = endedAt;
      }
    });
    if (terminalKind) {
      upsertActivityStep(activity, {
        id: 'terminal', kind: terminalKind,
        status: status === 'complete' ? 'complete' : 'error', at: endedAt,
      });
    }
    activity.status = status;
    activity.endedAt = endedAt;
  }


  function statusBanner() {
    if (state.loading) {
      return `<div class="gpi-inline"><span class="gpi-dot waiting"></span>${tr('Checking EasyICU Copilot…', '正在检查 EasyICU 研究助手…')}</div>`;
    }
    if (!connectionReady()) {
      const blockers = (state.runtime && state.runtime.blockers) || [];
      const reason = blockers.includes('api_key_configured')
        ? tr('Connect and verify your model service before entering EasyICU Copilot.', '请先连接并验证模型服务，再进入 EasyICU 研究助手。')
        : blockers.includes('provider_connection_unverified')
          ? tr('Verify the saved model service before entering EasyICU Copilot.', '请先验证已保存的模型服务，再进入 EasyICU 研究助手。')
        : blockers.includes('easyicu_ai_opt_in_disabled')
          ? tr('Confirm external AI use before entering EasyICU Copilot.', '请先确认允许使用外部 AI，再进入 EasyICU 研究助手。')
          : tr('EasyICU Copilot is not ready on this machine. The local Guided workflow remains available.', '这台电脑上的 EasyICU 研究助手尚未就绪，仍可使用本地研究引导流程。');
      return `<div class="gpi-inline unavailable"><span class="gpi-dot"></span><span>${esc(reason)}</span><button class="gpi-link" type="button" data-gpi-setup>${tr('Set up', '开始配置')}</button></div>`;
    }
    if (state.shell === 'legacy') {
      return `<div class="gpi-inline ready"><span class="gpi-dot"></span><span>${tr('EasyICU Copilot is ready with EasyICU-only tools.', 'EasyICU 研究助手已就绪，仅开放 EasyICU 工具。')}</span><button class="gpi-link" type="button" data-gpi-open>${tr('Open Copilot', '打开研究助手')}</button></div>`;
    }
    return '';
  }

  function setupPanel() {
    const runtime = state.runtime || {};
    const config = runtime.configuration || {};
    const blockers = runtime.blockers || [];
    /* screens-guided-pi-blockers.js owns which codes are runtime problems,
       what each one means in plain language, and what fixes it. This file
       only lays the result out. */
    const runtimeMissing = window.EU_PI_BLOCKERS
      ? window.EU_PI_BLOCKERS.describe(blockers, runtime)
      : [];
    const owner = window.EU_GUIDED_PI_PROVIDER;
    if (!owner || typeof owner.renderSetup !== 'function') return '';
    return owner.renderSetup({
      state, runtime, config, blockers, runtimeMissing, tr, esc, option,
      providerPreset, runtimeReady: runtimeReady(), apiResearchReady: apiResearchReady(),
      connectionConfigured: connectionConfigured(), connectionReady: connectionReady(),
      staticPreview: isStaticPreview(),
    });
  }

  function providerBindingSummary() {
    const owner = window.EU_GUIDED_PI_PROVIDER;
    return owner && typeof owner.renderBindingSummary === 'function'
      ? owner.renderBindingSummary({ state, tr, esc, runtimeReady: runtimeReady(), apiResearchReady: apiResearchReady(), connectionReady: connectionReady() })
      : '';
  }

  function activatePanel() {
    const saved = state.sessions.filter(sessionMatchesUiLanguage).map(row => `
      <button class="gpi-session-row" type="button" data-gpi-session="${esc(row.session_id)}">
        <span><strong>${esc(displaySessionTitle(row.title))}</strong><small>${esc(row.updated_at || '')}</small></span>
        <span>${row.agent_mode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究')}</span>
      </button>`).join('');
    if (state.projectIssue === 'pi_project_study_context_missing') {
      return `
        <div class="gpi-activate gpi-project-recovery">
          <div class="gpi-kicker">${tr('EASYICU COPILOT · PROJECT RECOVERY', 'EASYICU COPILOT · 项目恢复')}</div>
          <h2>${tr('This old project can no longer be opened', '这个旧项目已无法继续打开')}</h2>
          <div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Research project', '研究项目')}: <strong>${esc(displayProjectTitle(state.project && state.project.title, projectId()))}</strong></div>
          ${providerBindingSummary()}
          <div class="gpi-recovery-card" role="alert">
            <span class="gpi-recovery-icon">${iconHtml('folder', 20)}</span>
            <div>
              <strong>${tr('The saved research setup is no longer available', '关联的研究配置已经失效')}</strong>
              <p>${tr('The project shortcut still exists, but its authoritative study setup was removed. EasyICU will not silently create or attach a different setup.', '项目快捷记录仍然存在，但它原来绑定的权威研究配置已经被移除。EasyICU 不会静默创建或绑定另一份配置。')}</p>
            </div>
          </div>
          <div class="gpi-recovery-actions">
            <button class="btn primary" type="button" data-newstudy>${tr('Create or open a project', '新建或打开项目')}</button>
            <button class="btn" type="button" data-refreshdrafts>${tr('Refresh project list', '刷新项目列表')}</button>
          </div>
          <div class="gpi-consent">${tr('You can also choose another existing project from the list on the left. Rebinding this old project remains an explicit recovery operation.', '也可以直接从左侧列表选择其他已有项目。若要恢复当前旧项目，仍需执行明确的重新绑定操作。')}</div>
          <button class="gpi-link" type="button" data-gpi-legacy>${tr('Use the local Guided workflow', '使用本地研究引导流程')}</button>
        </div>`;
    }
    return `
      <div class="gpi-activate">
        <div class="gpi-kicker">${tr('EASYICU COPILOT · RESEARCH WORKSPACE', 'EASYICU COPILOT · 科研工作区')}</div>
        <h2>${tr('Start a conversation in this project', '在当前项目中开始对话')}</h2>
        <div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Research project', '研究项目')}: <strong>${esc(displayProjectTitle(state.project && state.project.title, projectId()))}</strong></div>
        ${providerBindingSummary()}
        ${state.error ? `<div class="gpi-error" role="alert">${esc(state.error)}</div>` : ''}
        <button class="btn primary" type="button" data-gpi-create ${state.creating ? 'disabled' : ''}>
          ${state.creating ? tr('Starting…', '正在启动…') : tr('Start research conversation', '开始研究对话')}
        </button>
        <div class="gpi-consent">${iconHtml('shield', 13)}<span>${tr('Study progress is saved automatically. File access can be enabled later and remains limited to this project folder — never patient rows, credentials, or arbitrary host files.', '研究进度会自动保存；文件操作可稍后开启，且只能访问当前项目目录，不包括患者行级数据、凭据或其他本机文件。')}</span></div>
        ${saved ? `<div class="gpi-saved"><div class="gpi-section-title">${tr('Copilot conversations in this project', '当前项目中的研究助手对话')}</div>${saved}</div>` : ''}
        <div class="gpi-secondary-actions">
          <button class="gpi-link" type="button" data-gpi-demo>${tr('View workflow demo', '查看流程演示')}</button>
          <button class="gpi-link" type="button" data-gpi-legacy>${tr('Use the local Guided workflow', '使用本地研究引导流程')}</button>
        </div>
      </div>`;
  }

  function projectRequiredPanel() {
    return `
      <div class="gpi-activate">
        <div class="gpi-kicker">EASYICU PROJECT · COPILOT CONVERSATIONS</div>
        <h2>${tr('Select a research project first', '请先选择研究项目')}</h2>
        <p>${tr('Use the Research projects list on the left, or create a new project. EasyICU keeps study setup, runs, evidence, and conversation history in that project.', '请从左侧“研究项目”中选择一个项目，或新建项目。EasyICU 会在项目中保存研究配置、运行、证据和对话历史。')}</p>
        <button class="btn primary gpi-demo-launch" type="button" data-gpi-demo>${iconHtml('play', 16)} ${tr('View the complete research workflow demo', '查看完整科研流程演示')}</button>
        <button class="gpi-link" type="button" data-gpi-legacy>${tr('Use the local Guided workflow', '使用本地研究引导流程')}</button>
      </div>`;
  }

  function restoringPanel() {
    return `
      <div class="gpi-activate gpi-restoring" role="status" aria-live="polite">
        <div class="gpi-kicker">EASYICU COPILOT · ${tr('RESTORING PROJECT', '正在恢复项目')}</div>
        <h2>${tr('Restoring your current research', '正在恢复当前研究')}</h2>
        <p>${tr('EasyICU is loading the saved project, model connection, and conversation together.', 'EasyICU 正在一起读取已保存的项目、模型连接和对话。')}</p>
      </div>`;
  }

  function messageHtml(row, options) {
    if (row.childJobHandoff) return ''; if (row.role === 'activity') return ACTIVITY.render(row);
    if (row.role === 'workflow_receipt') {
      const rows = row.total_rows == null ? Number.NaN : Number(row.total_rows);
      const files = Number(row.data_file_count);
      const supports = Number(row.support_file_count);
      const isResult = row.receipt_kind === 'extraction_result';
      return `<article class="gpi-message assistant gpi-workflow-receipt" role="status">
        <div class="gpi-message-body">
          <div class="gpi-workflow-receipt-head">${iconHtml('check', 15)}<div><strong>${isResult ? tr('Extraction result synchronized', '抽取结果已同步') : tr('Extraction setup saved', '抽取配置已保存')}</strong><span>${tr('This is EasyICU state, not a model reply.', '这是 EasyICU 本地状态，不是模型回复。')}</span></div></div>
          <div class="gpi-workflow-receipt-grid">
            <span>${tr('Database', '数据库')}<b>${esc(row.database || row.source_label || '—')}</b></span>
            ${isResult ? `<span>${tr('Output', '产物')}<b>${Number.isFinite(files) ? files : 0} + ${Number.isFinite(supports) ? supports : 0} ${tr('files', '个文件')}</b></span>` : ''}
            ${isResult ? `<span>${tr('Rows', '行数')}<b>${Number.isFinite(rows) ? rows.toLocaleString() : '—'}</b></span>` : ''}
            <span>${tr('Cohort', '队列')}<b>${esc(row.cohort_summary || tr('Current confirmed cohort', '当前确认队列'))}</b></span>
            <span>${tr('Modules', '模块')}<b>${esc((row.modules || []).join(', ') || '—')}</b></span>
            <span>${tr('Format', '格式')}<b>${esc(String(row.export_format || '—').toUpperCase())}</b></span>
            <span>StudyContext<b>${esc(row.study_context_id || '—')} · rev ${Number(row.study_revision || 0)}</b></span>
          </div>
          ${row.output_dir ? `<div class="gpi-workflow-receipt-path"><span>${tr('Local output folder', '本机输出文件夹')}</span><code>${esc(row.output_dir)}</code></div>` : ''}
          <p>${isResult
            ? tr('Copilot now reads this completed local extraction result. The absolute local path is shown only in this host UI and is not inserted as model-authored text.', 'Copilot 现在可以读取这次已完成的本地抽取结果。本机绝对路径只显示在当前宿主界面，不会被伪装成模型生成的文字。')
            : tr('Copilot now reads this database, cohort, feature-module, time-window, and export-format setup from the saved StudyContext. No extraction has been claimed yet.', 'Copilot 现在会从已保存的 StudyContext 读取数据库、队列、特征模块、时间窗和导出格式；此时尚未声称已经完成抽取。')}</p>
        </div>
      </article>`;
    }
    const cls = row.role === 'user' ? 'user' : 'assistant';
    const messageResources = RESOURCE_OWNER.forMessage(row, 8);
    const publicRow = row.role === 'assistant' ? { ...row, text: publicAssistantText(row.text) } : row;
    const nextOwner = window.EU_GUIDED_PI_NEXT_ACTIONS;
    // Project every assistant turn, not only the newest one. Projecting only
    // the latest left older turns rendering their own "### 下一步" block as raw
    // markdown -- four bullet lists that read as offers but could not be
    // clicked, beside the one live card.
    const nextStep = row.role === 'assistant' && row.complete !== false
      && nextOwner && typeof nextOwner.project === 'function'
      ? nextOwner.project(publicRow.text) : null;
    const interactive = Boolean(options && options.interactive);
    const visibleText = nextStep ? nextOwner.bodyText(nextStep) : publicRow.text;
    const nextStepHtml = !nextStep
      ? ''
      : !interactive
        ? nextOwner.renderPast(nextStep, window.EU_LANG)
        : typeof nextOwner.render === 'function'
      ? nextOwner.render(nextStep, {
        language: window.EU_LANG,
        disabled: state.busy || sessionIsStale(),
        dataSourceAuthorization: DATA_CONSENT && DATA_CONSENT.authorization(state.session),
        workflowActionCode: String((state.workflow && state.workflow.next_action_code) || ''),
        suppressFallback: String((state.workflow && state.workflow.next_action_code) || '') === 'provider_ready_to_generate_plan',
      }) : '';
    const messageActions = MESSAGE_ACTIONS.render(publicRow, {
      editing: state.editingMessageId === row.id,
      allowEdit: Boolean(options && options.allowEdit),
      canEdit: Boolean(options && options.canEdit),
      canRetry: Boolean(options && options.canRetry),
      retryUserEntryId: String(options && options.retryUserEntryId || ''),
    });
    const contentHtml = messageActions.editorHtml
      || (visibleText ? `<div class="gpi-text${row.errorCode ? ' gpi-model-error' : ''}">${row.role === 'assistant' ? assistantTextHtml(visibleText) : esc(visibleText)}</div>` : `<div class="gpi-streaming"><i></i><i></i><i></i></div>`);
    return `<article class="gpi-message ${cls}${messageActions.actionsHtml ? ' has-actions' : ''}" data-gpi-message-id="${esc(row.id || '')}">
      <div class="gpi-message-body">
        ${contentHtml}
        ${messageResources.length ? `<div class="gpi-message-resources" aria-label="${tr('Referenced run artifacts', '本轮引用的运行产物')}"><span>${tr('Open evidence and artifacts', '打开证据和产物')}</span><div class="gpi-resource-list">${messageResources.map(resource => resourceButton(resource)).join('')}</div></div>` : ''}
        ${nextStepHtml}
        ${messageActions.actionsHtml}
      </div>
    </article>`;
  }

  function workflowHtml(workflowOverride) {
    const workflow = workflowOverride || state.workflow || {};
    const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
    if (!stages.length) return '';
    const reviewerDemo = workflow.kind === 'reviewer_validation_demo';
    const names = {
      question: reviewerDemo ? tr('Protocol', '审稿协议') : tr('Question', '问题'),
      idea: reviewerDemo ? tr('Validation scope', '验证范围') : tr('Ideas + literature', '选题与文献'),
      setup: reviewerDemo ? tr('Data contract', '数据合同') : tr('Study design', '研究设计'),
      extraction: reviewerDemo ? tr('Projection', '安全投影') : tr('Extract + review', '提取与审阅'),
      plan: tr('Plan + evidence', '计划与证据'), analysis: tr('Analyze + figures', '分析与图表'),
      interpretation: tr('Interpret', '结果解读'), manuscript: reviewerDemo ? tr('Dossier', '审稿报告') : tr('Paper', '论文'),
    };
    return `<nav class="gpi-workflow" aria-label="${tr('EasyICU research workflow', 'EasyICU 科研流程')}">
      <div class="gpi-workflow-meta"><strong>${reviewerDemo ? tr('Reviewer workflow', '审稿流程') : tr('Research workflow', '科研流程')}</strong><span class="shell-sr-only">${esc(workflow.completed_required_stages || 0)}/${esc(workflow.required_stage_count || 7)}</span></div>
      <ol>${stages.map(stage => `<li class="${esc(stage.status || 'blocked')}" title="${esc(stage.reason_code || '')}" aria-current="${stage.id === workflow.current_stage ? 'step' : 'false'}"><i></i><span>${esc(names[stage.id] || stage.label || stage.id)}</span></li>`).join('')}</ol>
    </nav>`;
  }

  function accessModeHtml() {
    const modes = [
      ['ask', tr('Ask before every tool action', '每次工具操作前都询问')],
      ['assist', tr('Auto-approve low-risk setup and inspection; ask before extraction and full analysis', '自动批准低风险配置与检查；提取和完整分析前仍询问')],
      ['full', tr('Allow all available tools; explicit scientific confirmation gates still apply', '允许所有可用工具；明确的科学确认门禁仍然有效')],
    ];
    return `<details class="gpi-access-menu">
      <summary>${iconHtml(state.accessMode === 'full' ? 'unlock' : 'shield', 15)}<span>${esc(accessModeLabel(state.accessMode))}</span><span class="gpi-access-chevron" aria-hidden="true">${iconHtml('chevron', 13)}</span></summary>
      <div class="gpi-access-popover" role="group" aria-label="${tr('Agent access level', 'Agent 访问级别')}">
        ${modes.map(([mode, description]) => `<button type="button" data-gpi-access-mode="${mode}" aria-pressed="${state.accessMode === mode}"><span><strong>${esc(accessModeLabel(mode))}</strong><small>${esc(description)}</small></span>${state.accessMode === mode ? iconHtml('check', 15) : ''}</button>`).join('')}
        <p>${tr('Access levels never reveal credentials, patient rows, or arbitrary host files.', '任何访问级别都不会开放凭据、患者行级数据或任意本机文件。')}</p>
      </div>
    </details>`;
  }

  function sessionPanel() {
    const session = state.session || {};
    const model = session.model || {};
    const research = session.research_provider || {};
    const connection = session.model_connection || null;
    const stale = sessionIsStale();
    const workspace = agentMode() === 'workspace';
    const dataConsentRequired = !workspace
      && DATA_CONSENT && DATA_CONSENT.requiresConfirmation(session);
    const fullTimeline = state.messages.concat(state.workflowReceipts);
    const timeline = state.regenerating && REGENERATION
      ? REGENERATION.visibleRows(fullTimeline, state.regeneration)
      : fullTimeline;
    const activeChild = timeline.slice().reverse().find(row => row.role === 'activity' && row.childJobId && row.status === 'running');
    const interactionLocked = state.busy || Boolean(activeChild);
    const latestAssistant = timeline.slice().reverse().find(row => ['assistant', 'activity'].includes(row.role));
    let precedingUserText = '';
    let precedingUserEntryId = '';
    const messages = timeline.map(row => {
      const displayRow = state.regenerating && REGENERATION
        ? REGENERATION.project(row, state.regeneration) : row;
      const html = messageHtml(displayRow, {
        interactive: row === latestAssistant && !interactionLocked && !stale,
        allowEdit: true,
        canEdit: !interactionLocked && !stale,
        canRetry: row.role === 'assistant' && !interactionLocked && !stale,
        retryText: row.role === 'assistant' ? precedingUserText : '',
        retryUserEntryId: row.role === 'assistant' ? precedingUserEntryId : '',
      });
      if (row.role === 'user') {
        precedingUserText = String(row.text || '');
        precedingUserEntryId = String(row.entryId || '');
      }
      return html;
    }).join('');
    const emptyResearch = !workspace && !messages;
    const dataConsentHtml = dataConsentRequired
      ? DATA_CONSENT.render(session, { tr, esc, icon: iconHtml })
      : '';
    const emptyResearchHtml = STARTERS && typeof STARTERS.render === 'function'
      ? STARTERS.render({ tr, disabled: interactionLocked || stale })
      : `<div class="gpi-empty"><strong>${tr('Start with the research question', '先描述研究问题')}</strong></div>`;
    return `
      <div class="gpi-panel${emptyResearch ? ' gpi-empty-session' : ''}">
        ${HEADER.render({
          tr, esc, icon: iconHtml,
          projectTitle: displayProjectTitle(state.project && state.project.title, projectId()),
          sessionTitle: displaySessionTitle(session.title),
          busy: interactionLocked,
          workspace,
          pinned: Boolean(session.pinned_for_presentation),
          connectionLabel: connection
            ? ([connection.provider, connection.model].filter(Boolean).join(' · ') || 'model')
            : ([model.id || (state.runtime && state.runtime.model), research.provider, research.model].filter(Boolean).join(' / ') || 'legacy model binding'),
          connectionTitle: connection
            ? tr('One model connection for conversation and analysis', '对话与分析共用的一套模型连接')
            : tr('Legacy conversation and analysis bindings', '旧会话的对话与分析绑定'),
        })}
        ${workflowHtml()}
        ${stale ? `<div class="gpi-stale"><strong>${tr('Authority changed', '权威状态已变化')}</strong><span>${tr('The EasyICU study binding, revision, or active run changed. Rebind before continuing.', 'EasyICU 研究绑定、版本或活动运行已变化，请先重新绑定。')}</span><button class="btn sm" type="button" data-gpi-rebind>${tr('Rebind current state', '重新绑定当前状态')}</button></div>` : ''}
        <div class="gpi-log${messages ? '' : ' gpi-log-start'}" data-gpi-log>
          ${messages || (workspace
              ? `<div class="gpi-empty"><strong>${tr('Build something in this project', '在当前项目中创建产物')}</strong><span>${tr('EasyICU Copilot can read, write, edit, check, and preview files in this project’s isolated workspace, while retaining EasyICU research tools.', 'EasyICU 研究助手可以在当前项目的隔离工作区中读取、写入、编辑、检查并预览文件，同时保留 EasyICU 研究工具。')}</span></div>`
              : emptyResearchHtml)}
          ${workspace ? '' : RUN_OUTCOME.render(state.latestRun, state.workflow)}
          ${dataConsentHtml}
          ${dataConsentRequired ? '' : workflowConfirmationHtml()}
        </div>
        ${state.error ? `<div class="gpi-error">${esc(state.error)}</div>` : ''}
        <div class="gpi-compose">
          <div class="gpi-compose-card${activeChild ? ' is-running' : ''}">
            ${activeChild ? `<div class="gpi-compose-running" role="status" aria-live="polite" aria-busy="true">
              <span class="gpi-running-spinner" aria-hidden="true"></span>
              <span><strong>${esc(activeChild.cancelRequested ? tr('Stopping the research task', '正在停止科研任务') : (activeChild.runningTitle || tr('EasyICU research task is running', 'EasyICU 科研任务正在运行')))}</strong><small>${activeChild.cancelRequested ? tr('The cancellation request was sent. Waiting for the current safe checkpoint.', '已发送停止请求，正在等待当前安全检查点结束。') : tr('New messages are paused until this task finishes or asks for confirmation.', '任务完成或需要你确认后，才可继续发送消息。')}</small></span>
              <time data-gpi-live-elapsed="${Number(activeChild.startedAt || Date.now())}">${esc(ACTIVITY.durationText ? ACTIVITY.durationText(activeChild.startedAt) : '')}</time>
              <button class="btn danger sm" type="button" data-gpi-cancel-child-job="${esc(activeChild.childJobId)}" ${activeChild.cancelRequested ? 'disabled' : ''}>${activeChild.cancelRequested ? tr('Stopping…', '正在停止…') : tr('Stop generation', '停止生成')}</button>
            </div>` : `<textarea data-gpi-input rows="2" maxlength="12000" placeholder="${workspace ? tr('Ask EasyICU Copilot to create or edit a project artifact — do not paste patient rows or identifiers.', '让 EasyICU 研究助手创建或编辑当前项目产物——请勿粘贴患者行级数据或标识符。') : tr('Ask EasyICU Copilot about this study — do not paste patient rows or identifiers.', '向 EasyICU 研究助手询问当前研究——请勿粘贴患者行级数据或标识符。')}" ${interactionLocked || stale ? 'disabled' : ''}>${esc(state.draft)}</textarea>
              <div class="gpi-actions">
                ${accessModeHtml()}
                ${state.busy ? `<button class="btn danger" type="button" data-gpi-stop>${tr('Stop', '停止')}</button>` : `<button class="btn primary" type="button" data-gpi-send ${stale ? 'disabled' : ''}>${tr('Send', '发送')}</button>`}
              </div>`}
          </div>
        </div>
    </div>`;
  }

  function demoPanel() {
    const demo = window.EU_GUIDED_PI_DEMO;
    if (!demo || typeof demo.messages !== 'function') {
      return `<div class="gpi-activate"><h2>${tr('Demo unavailable', '演示暂不可用')}</h2><button class="btn" type="button" data-gpi-demo-exit>${tr('Back', '返回')}</button></div>`;
    }
    const messages = demo.messages().map(row => messageHtml(row, { interactive: false })).join('');
    const workflow = demo.workflow();
    const reviewResources = typeof demo.reviewResources === 'function' ? demo.reviewResources() : [];
    return `<div class="gpi-panel gpi-demo-panel">
      <header class="gpi-head">
        <div><div class="gpi-kicker">${tr('EASYICU COPILOT · REVIEWER DEMONSTRATION', 'EASYICU COPILOT · 审稿人演示')}</div><div class="gpi-title">${tr('Complete governed workflow', '完整受治理科研流程')} <span class="gpi-live">${tr('complete', '已完成')}</span></div></div>
        <div class="gpi-head-meta"><span>${tr('Registered source run · 94,458 ICU stays', '登记 source run · 94,458 ICU stays')}</span><button class="gpi-link" type="button" data-gpi-demo-exit>${tr('Back to my project', '返回我的项目')}</button></div>
      </header>
      ${workflowHtml(workflow)}
      <div class="gpi-demo-note" role="note">${iconHtml('shield', 16)}<span><strong>${tr('Read-only reviewer walkthrough.', '只读审稿人演示。')}</strong> ${tr('The transcript and dossier are a bounded projection derived from one registered source run, not live artifact transport. Aggregate results use the explicitly requested experimental first-24-hour SOFA-2 phenotype and a descriptive-only claim ceiling; they are not a clinical manuscript.', '对话与报告是从同一个登记 source run 派生的有界投影，不是 live artifact transport。聚合结果使用用户明确要求的入 ICU 后 24 小时实验性 SOFA-2 表型，结论上限为仅描述；它们不是临床论文。')}</span></div>
      <div class="gpi-log" data-gpi-log>${messages}</div>
      <footer class="gpi-demo-footer"><span>${tr('The reviewer dossier opens automatically. Select any underlined receipt to inspect its bounded source view.', '审稿报告会自动打开；点击任意带下划线的回执可检查其有界来源视图。')}</span><button class="btn primary" type="button" data-gpi-demo-exit>${tr('Start my own research', '开始我自己的研究')}</button></footer>
    </div>`;
  }

  function openDemo() {
    const demo = window.EU_GUIDED_PI_DEMO;
    if (!demo || typeof demo.messages !== 'function') return;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) window.EU_GUIDED_PI_PREVIEW.close();
    state.demoMode = true;
    state.demoScrollTopPending = true;
    state.error = '';
    setShell('pi');
    const primary = typeof demo.primaryDocument === 'function' ? demo.primaryDocument() : null;
    if (primary && window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.open) {
      window.EU_GUIDED_PI_PREVIEW.open(primary, projectId());
    }
  }
  function closeDemo() {
    state.demoMode = false;
    state.demoScrollTopPending = false;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) window.EU_GUIDED_PI_PREVIEW.close();
    render();
  }


  function render() {
    if (!state.host) return;
    const restoring = state.loading || state.projectLoading || state.projectDiscoveryLoading;
    const setupFocused = !restoring && state.shell !== 'legacy'
      && !state.demoMode
      && (state.showSetup || !connectionReady() || state.projectIssue === 'pi_project_study_context_missing');
    const emptySessionFocused = !restoring && !setupFocused && state.shell !== 'legacy'
      && !state.demoMode && Boolean(state.session) && agentMode() !== 'workspace'
      && state.messages.length === 0 && state.workflowReceipts.length === 0;
    const main = state.host.closest('.gd-main');
    if (main) {
      main.classList.toggle('gpi-setup-focus', setupFocused);
      main.classList.toggle('gpi-empty-session-focus', emptySessionFocused);
    }
    state.host.hidden = false;
    state.host.innerHTML = restoring
      ? restoringPanel()
      : state.shell === 'legacy'
      ? statusBanner()
      : (state.demoMode ? demoPanel() : (state.projectIssue === 'pi_project_study_context_missing'
        ? activatePanel()
        : ((state.showSetup || !connectionReady()) ? setupPanel() : (!projectId() ? projectRequiredPanel() : (state.session ? sessionPanel() : activatePanel())))));
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.setWorkflowContext) {
      window.EU_GUIDED_PI_PREVIEW.setWorkflowContext(previewWorkflowContext());
    }
    syncProjectWorkflowAside();
    requestAnimationFrame(() => {
      const log = state.host && state.host.querySelector('[data-gpi-log]');
      if (log) {
        log.scrollTop = state.demoScrollTopPending ? 0 : log.scrollHeight;
        state.demoScrollTopPending = false;
      }
      ACTIVITY.syncLiveClock(state.host, state.busy || Boolean(state.childJobId));
    });
  }

  async function loadStatus() {
    state.loading = true; state.error = ''; render();
    if (isStaticPreview()) {
      state.runtime = { status: 'unavailable', blockers: ['static_preview_no_backend'] };
      state.showSetup = true; state.loading = false; state.projectLoading = false; render(); return;
    }
    try {
      const payload = await api().loadPiCopilotStatus();
      state.runtime = payload && payload.runtime;
      restoreConfiguredResearchProvider();
      await loadCodexResearchStatus(false);
      if (projectId()) {
        try { await prepareProject(); }
        catch (error) { state.error = errorText(error); }
      }
      if (!connectionReady()) {
        state.showSetup = true;
      }
    } catch (error) {
      state.runtime = { status: 'unavailable', blockers: ['status_request_failed'] };
      state.error = errorText(error);
    } finally {
      state.loading = false;
      state.projectLoading = false;
      render();
    }
  }

  function stopCodexPoll() {
    if (state.codexPoll) window.clearTimeout(state.codexPoll);
    state.codexPoll = null;
  }

  async function loadCodexModels(renderAfter) {
    if (!api().loadPiCopilotCodexModels) return;
    try {
      const payload = await api().loadPiCopilotCodexModels();
      state.codexModels = Array.isArray(payload && payload.models) ? payload.models : [];
      if (!state.researchModel || !state.codexModels.some(row => row.id === state.researchModel)) {
        const preferred = state.codexModels.find(row => row.is_default) || state.codexModels[0];
        state.researchModel = preferred ? String(preferred.id || '') : '';
      }
    } catch (error) {
      // Keep the last catalog that was successfully verified for this
      // browser-owned runtime. A transient account/catalog probe must not
      // erase the user's model choice; logout clears it explicitly.
      if (renderAfter) state.error = errorText(error);
    }
    if (renderAfter) render();
  }

  async function loadCodexResearchStatus(renderAfter) {
    if (!api().loadPiCopilotCodexStatus) return;
    try {
      const payload = await api().loadPiCopilotCodexStatus();
      state.codexAuth = payload && payload.auth ? payload.auth : null;
      if (state.codexAuth && state.codexAuth.authentication_verified) {
        stopCodexPoll();
        state.codexLogin = null;
        await loadCodexModels(false);
      } else if (state.codexAuth && state.codexAuth.account_session_status === 'codex_auth_login_pending') {
        stopCodexPoll();
        state.codexPoll = window.setTimeout(() => loadCodexResearchStatus(true), 1500);
      }
    } catch (error) {
      state.codexAuth = null;
      if (renderAfter) state.error = errorText(error);
    }
    if (renderAfter) render();
  }

  async function startCodexLogin(flow, popup) {
    if (state.codexBusy || !api().startPiCopilotCodexLogin) return;
    state.codexBusy = true; state.error = ''; render();
    try {
      if (api().saveSetting) await api().saveSetting('ai_enabled', true);
      const runtimePayload = await api().loadPiCopilotStatus();
      state.runtime = runtimePayload && runtimePayload.runtime;
      const payload = await api().startPiCopilotCodexLogin(flow || 'browser');
      state.codexAuth = payload && payload.auth ? payload.auth : state.codexAuth;
      state.codexLogin = payload && payload.login_started ? {
        auth_url: payload.auth_url || payload.verification_url || '',
        user_code: payload.user_code || '',
      } : null;
      const authUrl = String((payload && (payload.auth_url || payload.verification_url)) || '');
      if (authUrl) {
        if (popup && !popup.closed) popup.location.href = authUrl;
        else window.open(authUrl, '_blank', 'noopener,noreferrer');
      } else if (popup && !popup.closed) {
        popup.close();
      }
      await loadCodexResearchStatus(false);
    } catch (error) {
      if (popup && !popup.closed) popup.close();
      state.error = errorText(error);
    } finally {
      state.codexBusy = false; render();
    }
  }

  function openAuthorizationPopup() {
    const popup = window.open('about:blank', '_blank');
    if (popup) {
      try { popup.opener = null; } catch (error) {}
    }
    return popup;
  }

  async function cancelCodexLogin() {
    if (state.codexBusy || !api().cancelPiCopilotCodexLogin) return;
    state.codexBusy = true; state.error = ''; render();
    try {
      const payload = await api().cancelPiCopilotCodexLogin();
      stopCodexPoll();
      state.codexAuth = payload && payload.auth ? payload.auth : null;
      state.codexLogin = null;
    } catch (error) { state.error = errorText(error); }
    finally { state.codexBusy = false; render(); }
  }

  async function logoutCodex() {
    if (state.codexBusy || !api().logoutPiCopilotCodex) return;
    state.codexBusy = true; state.error = ''; render();
    try {
      const payload = await api().logoutPiCopilotCodex();
      stopCodexPoll();
      state.codexAuth = payload && payload.auth ? payload.auth : null;
      state.codexLogin = null; state.codexModels = []; state.researchModel = '';
    } catch (error) { state.error = errorText(error); }
    finally { state.codexBusy = false; render(); }
  }

  async function configureProvider(form) {
    if (state.setupSaving || !form) return;
    const data = new FormData(form);
    const apiKey = String(data.get('api_key') || '').trim();
    const keyInput = form.querySelector('[name="api_key"]');
    if (keyInput) keyInput.value = '';
    state.setupSaving = true; state.error = '';
    const submit = form.querySelector('[type="submit"]');
    if (submit) { submit.disabled = true; submit.textContent = tr('Verifying…', '正在验证…'); }
    try {
      const payload = await api().savePiCopilotProviderConfig({
        provider: String(data.get('provider') || '').trim(),
        api_key: apiKey,
        base_url: String(data.get('base_url') || '').trim(),
        model: String(data.get('model') || '').trim(),
        api_transport: String(data.get('api_transport') || 'openai-completions'),
        enable_ai: true,
      });
      state.runtime = payload && payload.runtime;
      if (!runtimeReady()) {
        state.error = tr('The model connection was saved, but the Copilot runtime is not ready yet.', '模型连接已保存，但研究助手运行环境尚未就绪。');
        return;
      }
      state.error = '';
    } catch (error) {
      state.availableModels = Array.isArray(error && error.details && error.details.available_models)
        ? error.details.available_models.map(String) : [];
      state.error = errorText(error);
    } finally {
      state.setupSaving = false; render();
    }
  }

  async function finishProviderSetup() {
    if (!connectionConfigured()) return;
    state.setupSaving = true; state.error = ''; render();
    try {
      if (!shellReady() && api().saveSetting) {
        await api().saveSetting('ai_enabled', true);
        const payload = await api().loadPiCopilotStatus();
        state.runtime = payload && payload.runtime;
      }
      if (!connectionReady()) {
        throw new Error(tr('The selected model connection is not ready yet.', '所选模型连接尚未就绪。'));
      }
      state.showSetup = false;
    } catch (error) {
      state.error = errorText(error);
    } finally {
      state.setupSaving = false; render();
    }
  }

  async function createSession() {
    if (state.creating || !projectId()) return;
    const expectedProjectId = projectId();
    const selectionRevision = ++state.sessionSelectionRevision;
    if (!connectionReady()) {
      state.showSetup = true;
      state.error = tr('Finish the one model connection before starting a conversation.', '请先完成这一套模型连接，再开始对话。');
      render(); return;
    }
    if (state.researchProvider === 'codex' && (!state.codexAuth || !state.codexAuth.authentication_verified || !state.researchModel)) {
      state.showSetup = true;
      state.error = tr('Connect your ChatGPT account and select an account model first.', '请先连接 ChatGPT 账户并选择账户模型。');
      render(); return;
    }
    if (state.researchProvider === 'api' && !apiResearchReady()) {
      state.showSetup = true;
      state.error = tr('Research Agent currently requires an OpenAI Chat Completions-compatible API connection.', 'Research Agent 当前需要 OpenAI Chat Completions 兼容 API 连接。');
      render(); return;
    }
    state.creating = true; state.error = ''; state.pendingAuthorityRebind = false; state.editingMessageId = ''; render();
    try {
      if (state.projectInitialization && state.projectInitialization.required) {
        const bindingReceipt = state.project && state.project.binding_receipt;
        const initialized = await api().initializePiCopilotProject({
          project_id: expectedProjectId,
          title: displayProjectTitle(state.project && state.project.title, expectedProjectId),
          confirm_initialization: true,
          binding_receipt: bindingReceipt || undefined,
        });
        if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
        state.projectInitialization = initialized || { status: 'ready' };
        if (bindingReceipt && initialized && initialized.binding_receipt) {
          state.project = { ...state.project, binding_receipt: null };
        }
        await loadWorkflow();
      }
      const payload = await api().createPiCopilotSession({
        project_id: expectedProjectId,
        title: `${displayProjectTitle(state.project && state.project.title, tr('Research project', '研究项目'))} · ${state.agentMode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究')}`,
        agent_mode: state.agentMode,
        language: window.EU_LANG === 'zh' ? 'zh' : 'en',
        thinking_level: 'off', external_llm_opt_in: true,
        research_provider: state.researchProvider,
        research_model: state.researchProvider === 'codex' ? state.researchModel : null,
      });
      if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
      state.session = payload.session; state.messages = transcriptMessages(state.session);
      state.agentMode = state.session.agent_mode || state.agentMode;
      hydrateProjectedJob(state.workflow && state.workflow.active_job);
      state.projectInitialization = null;
      rememberSession(state.session.session_id);
      state.sessions = [state.session].concat(state.sessions.filter(row => row.session_id !== state.session.session_id));
    } catch (error) { state.error = errorText(error); }
    finally { state.creating = false; render(); }
  }

  async function openSession(sessionId, selectionRevision, refreshWorkflow) {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    const expectedSelectionRevision = selectionRevision == null
      ? ++state.sessionSelectionRevision : selectionRevision;
    closeChildSource();
    state.error = '';
    state.editingMessageId = '';
    state.pendingAuthorityRebind = false;
    try {
      const payload = await api().loadPiCopilotSession(sessionId, expectedProjectId);
      if (expectedProjectId !== projectId() || expectedSelectionRevision !== state.sessionSelectionRevision) return;
      if (!sessionMatchesUiLanguage(payload && payload.session)) {
        rememberSession('');
        state.session = null;
        state.messages = [];
        state.error = tr(
          'This conversation uses another response language. Switch the interface language to open it.',
          '这段对话使用另一种回复语言，请切换界面语言后再打开。',
        );
        render();
        return;
      }
      const replayOwner = window.EU_GUIDED_PI_REPLAY;
      state.session = replayOwner && typeof replayOwner.hydrate === 'function'
        ? await replayOwner.hydrate(api(), payload.session, expectedProjectId)
        : payload.session;
      if (expectedProjectId !== projectId() || expectedSelectionRevision !== state.sessionSelectionRevision) return;
      state.messages = transcriptMessages(state.session);
      state.agentMode = state.session.agent_mode || 'research';
      (Array.isArray(state.session.archived_child_jobs) ? state.session.archived_child_jobs : []).forEach(hydrateProjectedJob);
      const activeMessageJob = String(state.session.active_message_job_id || '').trim();
      if (activeMessageJob) {
        state.busy = true;
        state.jobId = activeMessageJob;
        watchJob(activeMessageJob);
      }
      reconcileSettledSession();
      hydrateProjectedJob(state.workflow && state.workflow.active_job);
      rememberSession(sessionId); setShell('pi');
      if (refreshWorkflow !== false) await loadWorkflow();
    } catch (error) { rememberSession(''); state.error = errorText(error); render(); }
  }

  function assistantRow() {
    if (state.regenerating && state.regeneration) return state.regeneration.message;
    let row = state.messages[state.messages.length - 1];
    if (!row || row.role !== 'assistant' || row.complete) {
      row = {
        id: 'live-' + Date.now(), role: 'assistant', text: '', complete: false,
        resources: state.currentTurnResources.slice(0, 24),
      };
      state.messages.push(row);
    }
    return row;
  }
  function addAssistantResources(resources) {
    const existing = state.currentTurnResources;
    (Array.isArray(resources) ? resources : []).forEach(resource => {
      const key = resourceKey(resource);
      if (key && !existing.some(item => resourceKey(item) === key)) existing.push(resource);
    });
    state.currentTurnResources = existing.slice(0, 24);
    const row = state.regenerating && state.regeneration
      ? state.regeneration.message
      : state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
    if (row) row.resources = state.currentTurnResources.slice();
  }
  function completeLatestAssistant(stopReason) {
    const row = state.regenerating && state.regeneration
      ? state.regeneration.message
      : state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
    if (row) { row.complete = true; row.stopReason = stopReason || ''; row.childJobHandoff = Boolean(state.childJobId); }
  }
  function modelErrorText(code) {
    const value = String(code || '');
    if (value === 'pi_shell_token_budget_exhausted' || value === 'pi_shell_session_provider_call_budget_exhausted') {
      return tr(
        'This conversation reached its bounded safety budget. Start a new conversation in the same research project; the StudyContext, literature, data source, runs, and evidence remain bound to the project.',
        '本会话已达到安全预算。请在同一研究项目中新建后续对话；StudyContext、文献、数据源、运行和证据仍保留在项目中。'
      );
    }
    if (value === 'pi_model_context_limit') return tr('The model context limit was reached. Start a new conversation or shorten the request.', '模型上下文已达到上限，请新建会话或缩短请求。');
    if (value === 'pi_model_rate_limited') return tr('The model service is temporarily rate-limited. No EasyICU action was executed; retry shortly.', '模型服务暂时限流。本轮没有执行 EasyICU 操作，请稍后重试。');
    if (value === 'pi_model_provider_unavailable') return tr('The model service connection was interrupted. No EasyICU action was executed; retry after connectivity recovers.', '模型服务连接中断。本轮没有执行 EasyICU 操作，连接恢复后可直接重试。');
    return tr('The model service could not complete this turn. No EasyICU action should be assumed.', '模型服务未能完成本轮，不能据此认为任何 EasyICU 操作已经执行。');
  }
  async function switchMode(mode) {
    const next = mode === 'research' ? 'research' : 'workspace';
    if (state.busy || next === agentMode()) return;
    closeSource();
    closeChildSource();
    state.error = '';
    state.pendingAuthorityRebind = false;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) {
      window.EU_GUIDED_PI_PREVIEW.close();
    }
    const replayOwner = window.EU_GUIDED_PI_REPLAY;
    let existingSessionId = replayOwner && typeof replayOwner.preferredSessionId === 'function'
      ? replayOwner.preferredSessionId(state.sessions, '', next, uiLanguage())
      : String(state.sessions.find(row => (row.agent_mode || 'research') === next && sessionMatchesUiLanguage(row))?.session_id || '');
    if (!existingSessionId && projectId()) {
      try {
        const listed = await api().loadPiCopilotSessions(100, projectId(), next);
        const matching = Array.isArray(listed && listed.sessions) ? listed.sessions : [];
        if (matching.length) {
          const matchingIds = new Set(matching.map(row => row.session_id));
          state.sessions = matching.concat(state.sessions.filter(row => !matchingIds.has(row.session_id)));
          existingSessionId = replayOwner && typeof replayOwner.preferredSessionId === 'function'
            ? replayOwner.preferredSessionId(matching, '', next, uiLanguage())
            : String(matching.find(sessionMatchesUiLanguage)?.session_id || '');
        }
      } catch (error) {
        state.error = errorText(error);
        render();
        return;
      }
    }
    if (existingSessionId) {
      await openSession(existingSessionId);
      return;
    }
    state.agentMode = next;
    state.session = null;
    state.messages = [];
    state.editingMessageId = '';
    rememberSession('');
    await createSession();
  }
  function handlePiEvent(event) {
    if (!event || typeof event !== 'object') return;
    const at = timeMs(event.at);
    const activity = ensureActivity(event.at);
    if (event.type === 'run_start') {
      state.currentTurnResources = [];
      upsertActivityStep(activity, { id: 'agent', kind: 'agent', status: 'complete', at });
    } else if (event.type === 'turn_start') {
      ACTIVITY.startTurn(activity, at);
    } else if (event.type === 'assistant_start') {
      const phase = activity.steps.filter(item => item.kind === 'assistant').length + 1;
      upsertActivityStep(activity, { id: 'assistant-' + phase, kind: 'assistant', phase, status: 'running', at, startedAt: at });
    } else if (event.type === 'text_delta') {
      const delta = String(event.delta || '');
      assistantRow().text += delta; ACTIVITY.appendPublicDelta(activity, delta);
    } else if (event.type === 'message_end') {
      let row = state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
      if (event.error_code) {
        row = row || assistantRow();
        row.errorCode = String(event.error_code);
        if (!row.text) row.text = modelErrorText(row.errorCode);
      }
      completeLatestAssistant(event.stop_reason);
      const step = activity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
      if (step) { step.status = event.error_code ? 'error' : 'complete'; step.endedAt = at; step.stopReason = event.stop_reason || ''; }
    } else if (event.type === 'tool_start') {
      const assistant = activity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
      if (assistant) assistant.status = 'complete';
      upsertActivityStep(activity, {
        id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name,
        status: 'running', at, startedAt: at, resource: event.resource || null,
      });
    } else if (event.type === 'tool_progress') {
      upsertActivityStep(activity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: 'running', at });
    }
    else if (event.type === 'tool_end') {
      const toolResources = [event.resource].concat(Array.isArray(event.resources) ? event.resources : []).filter(Boolean);
      upsertActivityStep(activity, {
        id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name,
        status: event.is_error ? 'error' : 'complete', code: event.code || '',
        owner: event.owner || '', text: event.summary || '', at, endedAt: at,
        jobId: event.job_id || '',
        resource: event.resource || null,
        resources: Array.isArray(event.resources) ? event.resources : [],
      });
      addAssistantResources(toolResources);
      const localWorkspace = !event.is_error && String(event.code || '') === 'easyicu_local_source_workspace_ready'
        ? toolResources.find(resource => resource && resource.kind === 'native_workspace')
        : null;
      if (localWorkspace && window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.open) {
        window.EU_GUIDED_PI_PREVIEW.open(localWorkspace, projectId());
      }
      if (event.host_rebind_after_turn === true || ['study_context_updated', 'easyicu_extraction_submitted', 'easyicu_run_submitted', 'easyicu_full_run_submitted'].includes(String(event.code || ''))) {
        state.pendingAuthorityRebind = true;
      }
      if (/^(easyicu_(research_workflow_projected|idea_|active_export_reused|extraction_|run_|full_run_|result_|manuscript_))/.test(String(event.code || ''))) {
        loadWorkflow().then(render).catch(() => {});
      }
      if (event.job_id && ['easyicu_extraction_submitted', 'easyicu_run_submitted', 'easyicu_full_run_submitted'].includes(String(event.code || ''))) {
        watchChildJob(String(event.job_id), String(event.code || ''));
      }
    } else if (event.type === 'turn_end') {
      ACTIVITY.finishTurn(activity, at);
    } else if (event.type === 'retry') {
      upsertActivityStep(activity, { id: 'retry-' + event.attempt, kind: 'retry', status: 'running', attempt: event.attempt, maxAttempts: event.max_attempts, at, startedAt: at });
    } else if (event.type === 'compaction_start') {
      upsertActivityStep(activity, { id: 'compaction', kind: 'compaction', status: 'running', at, startedAt: at });
    } else if (event.type === 'compaction_end') {
      upsertActivityStep(activity, { id: 'compaction', kind: 'compaction', status: event.aborted ? 'error' : 'complete', at, endedAt: at });
    } else if (event.type === 'agent_cycle_end' && event.will_retry) {
      const retry = activity.steps.slice().reverse().find(item => item.kind === 'retry' && item.status === 'running');
      if (retry) { retry.status = 'complete'; retry.endedAt = at; }
    } else if (event.type === 'run_end') {
      finishActivity('complete', event.at, 'settled');
    }
    render();
  }
  function closeSource() { if (state.source) { state.source.close(); state.source = null; } }
  function reconcileSettledSession() {
    if (state.session && state.session.active_message_job_id) return;
    if (!state.session || state.session.streaming !== false || !state.busy) return;
    closeSource();
    state.busy = false;
    state.jobId = '';
    finishActivity('complete', null, 'settled');
  }
  async function refreshSession(preserveTimeline) {
    if (!state.session || !projectId()) return;
    try {
      const payload = await api().loadPiCopilotSession(state.session.session_id, projectId());
      const replayOwner = window.EU_GUIDED_PI_REPLAY;
      state.session = !preserveTimeline && replayOwner && typeof replayOwner.hydrate === 'function'
        ? await replayOwner.hydrate(api(), payload.session, projectId())
        : payload.session;
      if (!preserveTimeline) state.messages = transcriptMessages(state.session);
      (Array.isArray(state.session.archived_child_jobs) ? state.session.archived_child_jobs : []).forEach(hydrateProjectedJob);
      reconcileSettledSession();
    } catch (e) {}
  }

  function adoptPersistedEntryIds() {
    // A sent message is appended optimistically and carries no server entry id.
    // A settled ordinary turn deliberately preserves the timeline so its live
    // activity rows survive, so the optimistic row used to keep an empty id
    // until the user reloaded by hand -- which silently disabled regeneration
    // and made the data-source continuation fail with "missing turn identifier".
    // Copy the persisted ids onto the rows that lack one instead of rebuilding.
    if (!state.session) return;
    const persisted = transcriptMessages(state.session)
      .filter(row => row.role === 'user' && String(row.entryId || '').trim());
    if (!persisted.length) return;
    let cursor = 0;
    state.messages.forEach(row => {
      if (row.role !== 'user' || String(row.entryId || '').trim()) return;
      const text = String(row.text || '').trim();
      if (!text) return;
      // Match on text and never move backwards, so a drifted timeline can only
      // leave an id unresolved -- never point a replay at a different turn.
      for (let index = cursor; index < persisted.length; index += 1) {
        if (String(persisted[index].text || '').trim() !== text) continue;
        row.entryId = persisted[index].entryId;
        cursor = index + 1;
        return;
      }
    });
  }

  async function loadProjectSessions(refreshWorkflow) {
    const expectedProjectId = projectId();
    const selectionRevision = state.sessionSelectionRevision;
    if (!connectionReady() || !expectedProjectId) return;
    const listed = await api().loadPiCopilotSessions(100, expectedProjectId);
    if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
    state.sessions = (listed && listed.sessions) || [];
    const remembered = rememberedSession();
    const replayOwner = window.EU_GUIDED_PI_REPLAY;
    const preferred = replayOwner && typeof replayOwner.preferredSessionId === 'function'
      ? replayOwner.preferredSessionId(state.sessions, remembered, '', uiLanguage())
      : (remembered && state.sessions.some(row => row.session_id === remembered && sessionMatchesUiLanguage(row))
        ? remembered
        : String(state.sessions.find(sessionMatchesUiLanguage)?.session_id || ''));
    if (preferred) await openSession(preferred, selectionRevision, refreshWorkflow);
  }

  async function reloadSessionsForLanguage() {
    const expectedProjectId = projectId();
    const selectionRevision = ++state.sessionSelectionRevision;
    closeSource();
    closeChildSource();
    state.pendingLanguageReload = false;
    state.session = null;
    state.sessions = [];
    state.messages = [];
    state.currentTurnResources = [];
    state.editingMessageId = '';
    state.error = '';
    state.busy = false;
    state.jobId = '';
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) {
      window.EU_GUIDED_PI_PREVIEW.close();
    }
    state.projectLoading = Boolean(expectedProjectId && connectionReady());
    render();
    if (!state.projectLoading) return;
    try {
      await loadProjectSessions();
    } catch (error) {
      if (expectedProjectId === projectId() && selectionRevision === state.sessionSelectionRevision) {
        state.error = errorText(error);
      }
    } finally {
      if (expectedProjectId === projectId() && selectionRevision === state.sessionSelectionRevision) {
        state.projectLoading = false;
        render();
      }
    }
  }

  function handleLanguageChange() {
    if (!state.host) return;
    state.pendingLanguageReload = true;
    state.projectLoading = Boolean(projectId());
    render();
    if (!state.busy) reloadSessionsForLanguage();
  }

  function reconcileDurableWrapupActivity() {
    const code = String((state.workflow && state.workflow.next_action_code) || '');
    if (![
      'operator_plan_approval_required',
      'planner_checkpoint_resume_available',
    ].includes(code)) return;
    const latest = state.messages.slice().reverse().find(
      row => row && row.role === 'activity' && !row.childJobId
    );
    if (!latest || latest.status !== 'error') return;
    latest.status = 'complete';
    latest.expanded = false;
  }

  async function loadWorkflow() {
    const expectedProjectId = projectId();
    if (!expectedProjectId || !api().loadPiCopilotProjectWorkflow) return;
    try {
      const payload = await api().loadPiCopilotProjectWorkflow(expectedProjectId);
      if (expectedProjectId !== projectId()) return;
      state.workflow = payload && payload.workflow ? payload.workflow : null;
      state.latestRun = payload && payload.latest_run ? payload.latest_run : { present: false };
      if (state.workflow) state.workflow.active_job = (payload && payload.active_job) || { present: false };
      reconcileDurableWrapupActivity();
      hydrateProjectedJob(payload && payload.active_job);
      const activeJob = payload && payload.active_job;
      if (activeJob && activeJob.present && activeJob.status === 'running' && activeJob.job_id) {
        const kind = String(activeJob.kind || '');
        const code = /extract/i.test(kind)
          ? 'easyicu_extraction_submitted'
          : (/research|agent/i.test(kind) ? 'easyicu_full_run_submitted' : 'easyicu_run_submitted');
        watchChildJob(String(activeJob.job_id), code);
      }
    } catch (error) {
      if (expectedProjectId === projectId()) {
        state.workflow = null;
        state.latestRun = null;
      }
    }
  }

  async function prepareProject() {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    if (state.projectPreparePromise && state.projectPrepareId === expectedProjectId) {
      return state.projectPreparePromise;
    }
    const pending = window.EU_GUIDED_PI_PROJECT.prepare({
      state, api, projectId, connectionReady, loadWorkflow,
      loadProjectSessions, render,
    });
    state.projectPrepareId = expectedProjectId;
    state.projectPreparePromise = Promise.resolve(pending).finally(() => {
      if (state.projectPreparePromise !== pendingPromise) return;
      state.projectPreparePromise = null;
      state.projectPrepareId = '';
    });
    const pendingPromise = state.projectPreparePromise;
    return pendingPromise;
  }

  function bindProject(project) {
    const next = project && String(project.id || '').trim()
      ? {
          id: String(project.id).trim(),
          title: displayProjectTitle(project.title, project.id),
          binding_receipt: project.binding_receipt || null,
        }
      : null;
    const sameProject = projectId() === String((next && next.id) || '');
    const currentReceipt = state.project && state.project.binding_receipt;
    if (sameProject && JSON.stringify(currentReceipt || null) === JSON.stringify((next && next.binding_receipt) || null)) return;
    closeSource();
    closeChildSource();
    state.demoMode = false;
    state.demoScrollTopPending = false;
    state.project = next;
    state.session = null;
    state.sessions = [];
    state.messages = [];
    // Extraction receipts are project-scoped UI state. Keeping them while the
    // user switches to a blank project makes the new conversation look as if
    // it inherited the previous project's data configuration.
    state.workflowReceipts = [];
    state.editingMessageId = '';
    state.busy = false;
    state.jobId = '';
    state.error = '';
    state.projectIssue = '';
    state.projectInitialization = null;
    state.workflow = null;
    state.projectLoading = !!next;
    state.agentMode = 'research';
    state.pendingAuthorityRebind = false;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.clearProject) {
      window.EU_GUIDED_PI_PREVIEW.clearProject();
    }
    render();
    if (next) {
      // Project selection happens after the initial Copilot status request in the
      // common path. prepareProject() updates the authoritative workflow and
      // saved-session lists asynchronously, so render again when it settles;
      // otherwise the legacy 0/8 aside and empty-session panel remain visible
      // even though the server returned the bound StudyContext revision.
      if (!state.host || !state.runtime) return Promise.resolve();
      return prepareProject()
        .catch(error => { if (projectId() === next.id) state.error = errorText(error); })
        .finally(() => {
          if (projectId() !== next.id) return;
          state.projectLoading = false;
          render();
        });
    }
    state.projectLoading = false;
    return Promise.resolve();
  }

  function setProjectDiscoveryLoading(active) {
    state.projectDiscoveryLoading = active === true;
    render();
  }

  function isActive() { return state.shell === 'pi'; }
  function watchJob(jobId) {
    closeSource();
    state.source = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    state.source.onmessage = async event => {
      let row = null; try { row = JSON.parse(event.data); } catch (e) { return; }
      if (row.type === 'pi_event') handlePiEvent(row.event);
      if (row.type === 'end') {
        closeSource(); state.busy = false;
        const replacedBranch = state.regenerating;
        const wrapupTimedOut = row.status === 'failed'
          && /^pi_gateway_timeout(?:\s*:|$)/.test(String(row.error || ''));
        if (row.status === 'failed') {
          finishActivity('error', null, 'failed');
          state.error = /^pi_model_/.test(String(row.error || ''))
            ? modelErrorText(row.error)
            : String(row.error || tr('Copilot message failed.', '研究助手消息失败。'));
        } else if (row.status === 'cancelled') {
          finishActivity('cancelled', null, 'cancelled');
          state.error = tr('Copilot message stopped.', '研究助手消息已停止。');
        } else {
          finishActivity('complete', null, 'settled');
        }
        await refreshSession(!replacedBranch);
        if (!replacedBranch) adoptPersistedEntryIds();
        state.regenerating = false;
        state.regeneration = null;
        if (state.pendingLanguageReload) {
          await reloadSessionsForLanguage();
          return;
        }
        if (state.pendingAuthorityRebind && state.session && sessionIsStale()) {
          await rebind();
        }
        await loadWorkflow();
        // A conversation wrap-up can time out after its governed child job has already
        // produced a reviewable plan/checkpoint. The workflow projection is
        // authoritative for that durable outcome; do not put a raw transport
        // error underneath the successful review card.
        const durablePlanState = String((state.workflow && state.workflow.next_action_code) || '');
        if (wrapupTimedOut && [
          'operator_plan_approval_required',
          'planner_checkpoint_resume_available',
        ].includes(durablePlanState)) state.error = '';
        state.pendingAuthorityRebind = false;
        render();
      }
    };
    state.source.onerror = () => { if (!state.busy) closeSource(); };
  }
  async function sendText(text, grantsOverride, turnIntent, visibleUserMessage = true) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    if (!sessionMatchesUiLanguage(state.session)) {
      handleLanguageChange();
      return;
    }
    text = String(text || '').trim();
    if (!text) return;
    state.editingMessageId = '';
    const grants = Array.isArray(grantsOverride) ? grantsOverride : turnGrants();
    const submittedAt = Date.now();
    state.currentTurnResources = [];
    if (visibleUserMessage) state.messages.push({ id: 'user-' + submittedAt, role: 'user', text, complete: true });
    const activity = ensureActivity(new Date(submittedAt).toISOString());
    upsertActivityStep(activity, { id: 'submitted', kind: 'submitted', status: 'complete', at: submittedAt });
    if (visibleUserMessage) state.draft = '';
    state.busy = true; state.error = ''; render();
    try {
      const payload = await api().sendPiCopilotMessage(state.session.session_id, {
        project_id: projectId(), message: text, allowed_actions: grants,
        ...(turnIntent ? { turn_intent: turnIntent } : {}),
      });
      state.jobId = payload.job_id; watchJob(payload.job_id);
    } catch (error) {
      state.busy = false; finishActivity('error', null, 'failed');
      state.error = errorText(error); render();
    }
  }
  async function regenerateMessage(userEntryId, text, regenerationIntent, targetMessageId) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const entryId = String(userEntryId || '').trim();
    text = String(text || '').trim();
    if (!entryId || !text) return;
    state.editingMessageId = '';
    state.currentTurnResources = [];
    state.regeneration = REGENERATION && typeof REGENERATION.create === 'function'
      ? REGENERATION.create(state.messages, {
        userEntryId: entryId,
        targetMessageId: String(targetMessageId || ''),
        startedAt: Date.now(),
      }) : null;
    state.regenerating = true;
    state.busy = true; state.error = ''; render();
    try {
      // Assistant retry stays side-effect free. An explicit user edit is
      // different: the edited text replaces that branch, so an exact formal-
      // plan request may receive one fresh provider grant when the current
      // host workflow independently says plan generation/replanning is legal.
      // Never copy an old turn's provider grant from transcript history.
      const workflowCode = String((state.workflow && state.workflow.next_action_code) || '');
      const nextOwner = window.EU_GUIDED_PI_NEXT_ACTIONS;
      const editedPlanGrants = regenerationIntent === 'user_edited_message'
        && nextOwner && typeof nextOwner.governedPlanGrants === 'function'
        ? nextOwner.governedPlanGrants(text, workflowCode)
        : [];
      const replayGrants = Array.from(new Set([
        ...turnGrants().filter(action => action === 'configure'),
        ...editedPlanGrants,
      ]));
      const replayIntent = editedPlanGrants.includes('provider_run')
        && workflowCode === 'provider_ready_to_generate_plan'
        ? 'confirm_formal_plan_generation'
        : editedPlanGrants.includes('provider_run')
          && workflowCode === 'planner_checkpoint_resume_available'
          ? 'confirm_planner_checkpoint_resume'
          : '';
      const payload = await api().regeneratePiCopilotMessage(state.session.session_id, {
        project_id: projectId(), user_entry_id: entryId,
        message: text, allowed_actions: replayGrants,
        ...(replayIntent ? { turn_intent: replayIntent } : {}),
        ...(regenerationIntent ? { regeneration_intent: regenerationIntent } : {}),
      });
      state.jobId = payload.job_id;
      watchJob(payload.job_id);
    } catch (error) {
      state.busy = false;
      state.regenerating = false;
      state.regeneration = null;
      state.error = errorText(error);
      render();
    }
  }
  function resubmitHostGeneratedMessage(row, text) {
    const id = String((row && row.id) || '');
    const original = String((row && row.text) || '').trim();
    const message = String(text || '').trim();
    const planAction = /^(?:重新生成研究计划|生成正式研究计划|generate (?:a fresh|the formal) research plan)[。.!！]?$/i;
    // Host-generated messages are projected as plan-generation-* in memory,
    // but after reload the persisted transcript legitimately gives the same
    // message a history-* id and an entryId.  Its explicit plan-action text is
    // therefore part of the stable identity used for routing.
    if (!id.startsWith('plan-generation-') && !planAction.test(original)) return false;
    if (!planAction.test(message)) {
      return false;
    }
    const workflowCode = String((state.workflow && state.workflow.next_action_code) || '');
    const nextOwner = window.EU_GUIDED_PI_NEXT_ACTIONS;
    const grants = nextOwner && typeof nextOwner.governedPlanGrants === 'function'
      ? nextOwner.governedPlanGrants(message, workflowCode)
      : [];
    if (!grants.includes('provider_run')) return false;
    const at = state.messages.findIndex(item => String((item && item.id) || '') === id);
    if (at >= 0) state.messages.splice(at);
    void startCurrentFormalPlanGeneration(workflowCode);
    return true;
  }
  async function continueAfterDataSourceConfirmation() {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return false;
    await sendText(
      tr(
        'Continue this conversation after EasyICU confirmed the selected data source.',
        'EasyICU 已确认所选数据来源，请在当前对话中继续。',
      ),
      [],
      'advance_after_data_source_confirmation',
      false,
    );
    return true;
  }
  async function sendMessage() {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const input = state.host.querySelector('[data-gpi-input]');
    const text = String((input && input.value) || state.draft || '').trim();
    await sendText(text);
  }
  async function confirmCohortEligibility(selection) {
    if (!selection || !state.session || state.busy || state.childJobId || sessionIsStale()) return;
    state.busy = true;
    state.error = '';
    render();
    try {
      await api().confirmPiCopilotCohortEligibility(state.session.session_id, {
        project_id: projectId(),
        option_id: selection.option_id,
        expected_revision: selection.expected_revision,
        primary_cohort_contract_sha256: selection.primary_cohort_contract_sha256,
        selection_event_id: selection.selection_event_id,
      });
      await refreshSession(true);
      await loadWorkflow();
    } catch (error) {
      state.error = errorText(error);
    } finally {
      state.busy = false;
      render();
    }
  }
  async function confirmWorkflowAction() {
    const confirmation = workflowConfirmation();
    if (!confirmation) return;
    if (confirmation.code === 'operator_plan_approval_required') {
      await submitCurrentPlanReview('approved');
      return;
    }
    if (confirmation.code === 'plan_execution_upgrade_required') {
      await sendText(confirmation.message, confirmation.grants);
      return;
    }
    if ([
      'provider_ready_to_generate_plan',
      'plan_scientific_changes_required',
      'failed_pipeline_execution_retry_available',
      'failed_pipeline_requires_fresh_plan',
      'plan_configuration_superseded',
      'plan_review_not_resumable',
    ].includes(confirmation.code)) {
      await startCurrentFormalPlanGeneration(confirmation.code);
      return;
    }
    await sendText(
      confirmation.message,
      confirmation.grants,
      confirmation.code === 'provider_ready_to_generate_plan'
        ? 'confirm_formal_plan_generation'
        : confirmation.code === 'planner_checkpoint_resume_available'
          ? 'confirm_planner_checkpoint_resume'
          : '',
    );
  }
  async function previewApprovedPlanDataPackage(button) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    if (!api().preparePiCopilotDataPackageReview || !window.EU_GUIDED_PI_PREVIEW || !window.EU_GUIDED_PI_PREVIEW.open) {
      state.error = tr('The data preview is temporarily unavailable. Refresh this project and try again.', '数据预览暂时不可用，请刷新当前项目后重试。');
      render();
      return;
    }
    const original = button ? button.textContent : '';
    if (button) {
      button.disabled = true;
      button.textContent = tr('Preparing preview…', '正在准备预览…');
    }
    try {
      const payload = await api().preparePiCopilotDataPackageReview(projectId());
      const resource = payload && payload.resource;
      if (!resource) throw new Error(tr('EasyICU did not return a data preview.', 'EasyICU 未返回可预览的数据包。'));
      resource.label = tr('Analysis data preview', '分析数据预览');
      window.EU_GUIDED_PI_PREVIEW.open(resource, projectId(), previewWorkflowContext());
      state.error = '';
    } catch (error) {
      state.error = errorText(error);
      render();
    } finally {
      if (button && button.isConnected) {
        button.disabled = false;
        button.textContent = original;
      }
    }
  }
  async function rejectWorkflowAction() {
    const confirmation = workflowConfirmation();
    if (!confirmation || !confirmation.rejectMessage) return;
    if (confirmation.code === 'operator_plan_approval_required') {
      await submitCurrentPlanReview('rejected');
      return;
    }
    await sendText(confirmation.rejectMessage, confirmation.grants);
  }
  async function submitCurrentPlanReview(decision) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const runId = String((state.session.binding && state.session.binding.run_id) || '').trim();
    const studyContextId = String((state.session.binding && state.session.binding.study_context_id) || '').trim();
    if (!runId || !studyContextId || !api().submitAgentRunReview) {
      state.error = tr('The current plan review coordinates are unavailable. Refresh this project and try again.', '当前计划的审核坐标不可用，请刷新该项目后重试。');
      render();
      return;
    }
    const approved = decision === 'approved';
    state.messages.push({
      id: 'plan-review-' + Date.now(), role: 'user', complete: true,
      text: approved ? tr('Approve plan and start analysis', '批准计划并开始分析') : tr('Reject this plan', '拒绝当前计划'),
    });
    state.busy = true;
    state.error = '';
    render();
    try {
      const payload = await api().submitAgentRunReview({
        run_id: runId,
        study_context_id: studyContextId,
        decision,
        external_llm_opt_in: true,
      });
      state.busy = false;
      watchChildJob(String(payload.job_id || ''), 'easyicu_review_submitted');
    } catch (error) {
      state.busy = false;
      state.error = errorText(error);
      render();
    }
  }
  async function startCurrentFormalPlanGeneration(reasonCode) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const studyContextId = String((state.session.binding && state.session.binding.study_context_id) || '').trim();
    const research = state.session.research_provider || {};
    if (!studyContextId || !api().loadStudyContext || !api().startAgentRun) {
      state.error = tr('The current research bindings are unavailable. Refresh this project and try again.', '当前研究绑定不可用，请刷新项目后重试。');
      render();
      return;
    }
    const retryExecution = reasonCode === 'failed_pipeline_execution_retry_available';
    const fresh = reasonCode !== 'provider_ready_to_generate_plan' && !retryExecution;
    state.messages.push({
      id: 'plan-generation-' + Date.now(), role: 'user', complete: true,
      text: retryExecution
        ? tr('Retry analysis from the failed step', '从失败步骤重试分析')
        : fresh
          ? tr('Generate a fresh research plan', '重新生成研究计划')
          : tr('Generate the formal research plan', '生成正式研究计划'),
    });
    state.busy = true;
    state.error = '';
    render();
    try {
      const contextPayload = await api().loadStudyContext(studyContextId);
      const study = contextPayload && contextPayload.context ? contextPayload.context : null;
      const source = study && study.data_source ? study.data_source : {};
      const reviewedRunId = String((state.session.binding && state.session.binding.run_id) || '').trim();
      if (!study || !String(source.path || '').trim()) {
        throw new Error(tr('The prepared study data source is no longer bound.', '当前研究已不再绑定准备好的数据源。'));
      }
      const payload = await api().startAgentRun({
        path: String(source.path || '').trim(),
        study_context_id: studyContextId,
        question: String(study.question || '').trim(),
        run_type: 'full',
        engine: 'research_agent_pipeline',
        llm_provider: String(research.provider || '').trim(),
        credential_source: String(research.credential_source || '').trim(),
        external_llm_opt_in: true,
        literature_search_authorized: true,
        ...(retryExecution && reviewedRunId
          ? { execution_resume_source_run_id: reviewedRunId }
          : {}),
        ...(reasonCode === 'plan_scientific_changes_required' && reviewedRunId
          ? { plan_revision_source_run_id: reviewedRunId }
          : {}),
      });
      state.busy = false;
      watchChildJob(String(payload.job_id || ''), 'easyicu_full_run_submitted');
    } catch (error) {
      state.busy = false;
      state.error = errorText(error);
      render();
    }
  }
  function governedNextChoiceGrants(element, message) {
    const projected = String((element && element.dataset && element.dataset.gpiNextGrants) || '')
      .split(',').map(value => value.trim()).filter(Boolean);
    const projectedAllowlist = new Set(['extract', 'configure']);
    if (projected.length && projected.every(value => projectedAllowlist.has(value))) return projected;
    const code = String((state.workflow && state.workflow.next_action_code) || '');
    const nextOwner = window.EU_GUIDED_PI_NEXT_ACTIONS;
    const planGrants = nextOwner && typeof nextOwner.governedPlanGrants === 'function'
      ? nextOwner.governedPlanGrants(message, code)
      : [];
    return planGrants.length ? planGrants : null;
  }
  function editWorkflow() {
    const workflow = state.workflow || {};
    const code = String(workflow.next_action_code || '');
    if (code === 'provider_ready_to_generate_plan') {
      state.draft = tr(
        'Before generating the plan, I want to add this research requirement: ',
        '生成计划前，我想补充以下研究要求：',
      );
      render();
      requestAnimationFrame(() => {
        const input = state.host && state.host.querySelector('[data-gpi-input]');
        if (input) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }
      });
      return;
    }
    const review = workflow.plan_review_summary || {};
    const questions = Array.isArray(review.authorization_questions)
      ? review.authorization_questions.filter(item => item && (item.question || item.code))
      : [];
    const nextQuestion = questions.length ? localizedAuthorizationQuestion(questions[0]) : '';
    state.draft = nextQuestion || tr(
      'Please ask me the next unresolved scientific decision and save my answer in the typed study configuration.',
      '请一次只问我一个尚未解决的科学设定问题，并把我的回答保存到结构化研究配置。',
    );
    render();
    requestAnimationFrame(() => {
      const input = state.host && state.host.querySelector('[data-gpi-input]');
      if (input) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }
    });
  }
  function studySetupReviewPrompt(workflow) {
    const receipt = workflow && workflow.study_setup_receipt;
    const missing = workflow && workflow.missing_setup_fields;
    const missingText = Array.isArray(missing) && missing.length
      ? missing.join(', ')
      : 'none';
    const receiptText = JSON.stringify(receipt || {
      study_context_id: '',
      revision: 0,
      configured_fields: [],
      configuration: {},
    });
    return tr(
      `Review this existing project's study configuration in this conversation. Treat the following path-free Study Setup Receipt as the authoritative starting state: ${receiptText}. Preserve study_context_id and revision; do not create a new project or reset configured fields. Current missing fields: ${missingText}. Summarize the configured values first, then ask which field I want to edit.`,
      `请在当前对话中审阅这个已有项目的研究配置。以下不含本地路径的 Study Setup Receipt 是权威起始状态：${receiptText}。保留 study_context_id 和 revision；不要新建项目，也不要重置已配置字段。当前缺失字段：${missingText}。请先概括已有配置，再询问我要修改哪个字段。`,
    );
  }
  async function openStudySetupInConversation() {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    setShell('pi');
    state.showSetup = false;
    state.error = '';
    await loadWorkflow();
    const prompt = studySetupReviewPrompt(state.workflow);
    await sendText(prompt, ['configure']);
  }
  async function stopMessage() {
    if (!state.session || !state.busy) return;
    try {
      await api().abortPiCopilotSession(state.session.session_id, {
        project_id: projectId(), message_job_id: state.jobId || null,
      });
    }
    catch (error) { state.error = errorText(error); render(); }
  }
  async function stopChildJob(jobId) {
    try {
      await CHILDJOB.cancelChildJob(jobId);
    } catch (error) {
      state.error = errorText(error);
      render();
    }
  }
  async function rebind() {
    if (!state.session) return;
    try {
      const payload = await api().rebindPiCopilotSession(
        state.session.session_id,
        { project_id: projectId() },
      );
      state.session = payload.session; state.error = '';
      rememberSession(state.session && state.session.session_id);
      await loadWorkflow();
      render();
    } catch (error) { state.error = errorText(error); render(); }
  }


  async function archiveChildJob(jobId) {
    if (!state.session || !jobId || !api().archivePiCopilotChildJob) return null;
    return api().archivePiCopilotChildJob(
      state.session.session_id,
      jobId,
      { project_id: projectId() },
    );
  }

  async function togglePresentationPin() {
    if (!state.session || !api().pinPiCopilotPresentation) return;
    try {
      const pinned = !Boolean(state.session.pinned_for_presentation);
      const payload = await api().pinPiCopilotPresentation(
        state.session.session_id,
        { project_id: projectId(), pinned },
      );
      state.session.pinned_for_presentation = Boolean(
        payload && payload.session && payload.session.pinned_for_presentation,
      );
      state.error = '';
    } catch (error) {
      state.error = errorText(error);
    }
    render();
  }

  function dismissHeaderOverflow(event) {
    const menu = state.host && state.host.querySelector('.gpi-head-overflow[open]');
    if (!menu) return;
    const action = event.target && event.target.closest
      ? event.target.closest('.gpi-head-overflow-menu button') : null;
    if (!menu.contains(event.target) || action) menu.removeAttribute('open');
  }

  function wire() {
    if (!state.host) return;
    state.host.addEventListener('click', event => {
      const session = event.target.closest('[data-gpi-session]');
      if (session) { openSession(session.dataset.gpiSession); return; }
      if (event.target.closest('[data-gpi-demo-exit]')) { closeDemo(); return; }
      if (event.target.closest('[data-gpi-demo]')) { openDemo(); return; }
      const resource = event.target.closest('[data-gpi-resource-kind]');
      if (resource) {
        if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.open) {
          window.EU_GUIDED_PI_PREVIEW.open(
            RESOURCE_OWNER.fromButton(resource), projectId(), previewWorkflowContext(),
          );
        }
        return;
      }
      const modeSwitch = event.target.closest('[data-gpi-mode-switch]');
      if (modeSwitch) { switchMode(modeSwitch.dataset.gpiModeSwitch); return; }
      const accessMode = event.target.closest('[data-gpi-access-mode]');
      if (accessMode) { state.accessMode = accessMode.dataset.gpiAccessMode || 'assist'; render(); return; }
      const researchProvider = event.target.closest('[data-gpi-research-provider]');
      if (researchProvider) {
        state.researchProvider = researchProvider.dataset.gpiResearchProvider === 'codex' ? 'codex' : 'api';
        state.error = '';
        if (state.researchProvider === 'codex') loadCodexResearchStatus(true);
        else render();
        return;
      }
      if (event.target.closest('[data-gpi-codex-login]')) {
        const popup = openAuthorizationPopup();
        startCodexLogin('browser', popup); return;
      }
      if (event.target.closest('[data-gpi-codex-device]')) {
        const popup = openAuthorizationPopup();
        startCodexLogin('device_code', popup); return;
      }
      if (event.target.closest('[data-gpi-codex-cancel]')) { cancelCodexLogin(); return; }
      if (event.target.closest('[data-gpi-codex-logout]')) { logoutCodex(); return; }
      if (event.target.closest('[data-gpi-codex-models]')) { loadCodexModels(true); return; }
      if (event.target.closest('[data-gpi-provider-done]')) {
        if (state.researchProvider === 'codex' && (!state.codexAuth || !state.codexAuth.authentication_verified || !state.researchModel)) {
          state.error = tr('Connect your ChatGPT account and select an account model first.', '请先连接 ChatGPT 账户并选择账户模型。');
          render(); return;
        }
        if (state.researchProvider === 'api' && !apiResearchReady()) {
          state.error = tr('Research Agent currently requires an OpenAI Chat Completions-compatible API connection.', 'Research Agent 当前需要 OpenAI Chat Completions 兼容 API 连接。');
          render(); return;
        }
        finishProviderSetup(); return;
      }
      if (event.target.closest('[data-gpi-retry]')) { loadStatus(); return; }
      if (event.target.closest('[data-gpi-setup]')) { state.showSetup = true; setShell('pi'); return; }
      if (event.target.closest('[data-gpi-open]')) { setShell('pi'); return; }
      if (event.target.closest('[data-gpi-study-setup]')) { openStudySetupInConversation(); return; }
      if (event.target.closest('[data-gpi-legacy]')) { setShell('legacy'); return; }
      if (event.target.closest('[data-gpi-create]')) { createSession(); return; }
      const previewPlanData = event.target.closest('[data-gpi-confirm-preview-data]');
      if (previewPlanData) { previewApprovedPlanDataPackage(previewPlanData); return; }
      const previewAnalysisData = event.target.closest('[data-gpi-run-outcome-data]');
      if (previewAnalysisData) { RUN_OUTCOME.openData(previewAnalysisData); return; }
      if (event.target.closest('[data-gpi-confirm-action]')) { confirmWorkflowAction(); return; }
      if (event.target.closest('[data-gpi-confirm-reject]')) { rejectWorkflowAction(); return; }
      if (event.target.closest('[data-gpi-confirm-edit]')) { editWorkflow(); return; }
      const cohortSelection = COHORT_ELIGIBILITY.actionFromEvent(event);
      if (cohortSelection) { confirmCohortEligibility(cohortSelection); return; }
      const dataSourceAction = DATA_CONSENT && DATA_CONSENT.actionFromEvent(event);
      if (dataSourceAction) { authorizeDataSource(dataSourceAction); return; }
      if (event.target.closest('[data-gpi-data-demo]')) {
        sendText(tr(
          'I do not have local data yet. Show only the official EasyICU demo datasets and explain their limits. Do not download or use one until I choose it. Offer only each exact demo or continuing study planning without data; do not offer a local full-database workflow.',
          '我还没有本地数据。请只列出 EasyICU 官方 Demo 数据并说明局限；在我选择前不要下载或使用。下一步只提供每个准确 Demo 或继续无数据规划，不要提供本地完整数据库工作流。',
        ));
        return;
      }
      if (MESSAGE_ACTIONS.handleClick(event)) return;
      const starterAction = STARTERS && STARTERS.actionFromEvent(event);
      if (starterAction && starterAction.kind === 'send') {
        sendText(starterAction.text, []);
        return;
      }
      if (starterAction && starterAction.kind === 'compose') {
        state.draft = starterAction.text;
        render();
        window.requestAnimationFrame(() => {
          const input = state.host && state.host.querySelector('[data-gpi-input]');
          if (!input) return;
          input.focus();
          input.setSelectionRange(input.value.length, input.value.length);
          input.scrollIntoView({ block: 'nearest' });
        });
        return;
      }
      if (event.target.closest('[data-gpi-data-source-continue]')) {
        continueAfterDataSourceConfirmation();
        return;
      }
      const nextChoice = event.target.closest('[data-gpi-next-choice]');
      if (nextChoice) {
        const localDatabase = String(nextChoice.dataset.gpiNextLocalDatabase || '').trim();
        if (localDatabase) {
          authorizeDataSource('begin_local_selection', { database: localDatabase });
          return;
        }
        const message = nextChoice.dataset.gpiNextChoice;
        sendText(message, governedNextChoiceGrants(nextChoice, message));
        return;
      }
      if (event.target.closest('[data-gpi-next-focus]')) {
        const input = state.host.querySelector('[data-gpi-input]');
        if (input) { input.focus(); input.scrollIntoView({ block: 'nearest' }); }
        return;
      }
      if (event.target.closest('[data-gpi-send]')) { sendMessage(); return; }
      if (event.target.closest('[data-gpi-stop]')) { stopMessage(); return; }
      const childStop = event.target.closest('[data-gpi-cancel-child-job]');
      if (childStop) { stopChildJob(childStop.dataset.gpiCancelChildJob); return; }
      if (event.target.closest('[data-gpi-rebind]')) { rebind(); return; }
      if (event.target.closest('[data-gpi-presentation-pin]')) { togglePresentationPin(); return; }
      if (event.target.closest('[data-gpi-config]')) { state.showSetup = true; state.error = ''; render(); return; }
      if (event.target.closest('[data-gpi-cancel-setup]')) { state.showSetup = false; state.error = ''; render(); return; }
      if (event.target.closest('[data-gpi-new]')) { state.sessionSelectionRevision += 1; state.session = null; state.messages = []; state.editingMessageId = ''; rememberSession(''); render(); }
    });
    state.host.addEventListener('input', event => {
      if (event.target.matches('[data-gpi-input]')) state.draft = event.target.value;
    });
    state.host.addEventListener('change', event => {
      if (event.target.matches('[data-gpi-codex-model]')) {
        state.researchModel = String(event.target.value || '');
        state.error = ''; render(); return;
      }
      if (!event.target.matches('[data-gpi-provider-preset]')) return;
      const form = event.target.closest('[data-gpi-provider-form]');
      if (!form) return;
      const presets = {
        cliproxyapi: { provider: 'easyicu-local', base_url: 'http://127.0.0.1:8317/v1', api_transport: 'openai-completions', model: 'gpt-5.6-luna' },
        'custom-openai': { provider: 'custom-openai', base_url: 'https://example.com/v1', api_transport: 'openai-completions', model: '' },
        openai: { provider: 'openai', base_url: 'https://api.openai.com/v1', api_transport: 'openai-responses', model: 'gpt-5.6-luna' },
        openrouter: { provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', api_transport: 'openai-completions', model: '' },
        deepseek: { provider: 'deepseek', base_url: 'https://api.deepseek.com/v1', api_transport: 'openai-completions', model: 'deepseek-chat' },
        anthropic: { provider: 'anthropic', base_url: 'https://api.anthropic.com/v1', api_transport: 'anthropic-messages', model: 'claude-sonnet-4-6' },
        google: { provider: 'google', base_url: 'https://generativelanguage.googleapis.com/v1beta', api_transport: 'google-generative-ai', model: 'gemini-3.5-flash' },
      };
      const selected = presets[event.target.value];
      if (!selected) return;
      Object.keys(selected).forEach(name => {
        const field = form.elements.namedItem(name);
        if (field) field.value = selected[name];
      });
      state.availableModels = [];
      const modelList = form.querySelector('#gpi-model-options');
      if (modelList) modelList.replaceChildren();
    });
    state.host.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        const menu = state.host.querySelector('.gpi-head-overflow[open]');
        if (menu) {
          menu.removeAttribute('open');
          const summary = menu.querySelector('summary');
          if (summary) summary.focus();
          return;
        }
      }
      if (event.target.matches('[data-gpi-input]') && window.EU_COMPOSER_KEYBOARD.enterShouldSend(event)) {
        event.preventDefault(); sendMessage();
      }
    });
    state.host.addEventListener('submit', event => {
      if (MESSAGE_ACTIONS.handleSubmit(event)) return;
      const nextCustomForm = event.target.closest('[data-gpi-next-custom-form]');
      if (nextCustomForm) {
        event.preventDefault();
        const input = nextCustomForm.querySelector('[data-gpi-next-custom-input]');
        const message = String((input && input.value) || '').trim();
        if (message) sendText(message, governedNextChoiceGrants(null, message));
        return;
      }
      const form = event.target.closest('[data-gpi-provider-form]');
      if (!form) return;
      event.preventDefault(); configureProvider(form);
    });
  }
  function mount(host) {
    if (!host) return Promise.resolve();
    if (state.host === host) return state.startupPromise || Promise.resolve();
    closeSource(); closeChildSource(); state.host = host; state.conv = host.closest('.gd-conv'); state.shell = 'pi';
    if (state.conv) state.conv.classList.add('pi-active');
    wire(); document.addEventListener('click', dismissHeaderOverflow);
    state.startupPromise = Promise.resolve(loadStatus()).finally(() => {
      state.startupPromise = null;
    });
    return state.startupPromise;
  }
  function unmount() {
    document.removeEventListener('click', dismissHeaderOverflow);
    stopCodexPoll(); closeSource(); closeChildSource(); state.host = null; state.conv = null; state.busy = false; state.jobId = '';
  }
  window.addEventListener('easyicu:languagechange', handleLanguageChange);
  window.EU_GUIDED_PI = {
    mount,
    unmount,
    setShell,
    bindProject,
    isActive,
    rebind,
    notifyExtractionHandoff,
    confirmDataSourceBinding,
    setProjectDiscoveryLoading,
  };
})();
