/* Pi AgentSession client for Guided Copilot.
   Owner: Pi chat/session/tool UX only. Study cards and scientific execution
   remain in their existing EasyICU owners. */
(function () {
  'use strict';
  const MODULES = window.EasyICU.guidedPi;
  const { esc } = window.EU_HTML;
  const DATA_CONSENT = MODULES.require('dataConsent');
  const STARTERS = MODULES.require('starters');
  const IDEA_SOURCE = MODULES.require('ideaSource');
  const EFFORT_MENU = MODULES.require('effortMenu');
  const HEADER = MODULES.require('header');
  const REGENERATION = MODULES.require('regeneration');
  const STUDY_WORKSPACE = MODULES.require('studyWorkspace').create({ tr, esc, iconHtml });

  const state = {
    host: null, conv: null, runtime: null, sessions: [], session: null,
    messages: [], loading: true, creating: false, busy: false, jobId: '',
    planConfigurationError: '',
    source: null, childSource: null, childJobId: '', error: '', shell: 'pi', draft: '', setupSaving: false,
    showSetup: false, availableModels: [], project: null,
    researchProvider: 'codex', researchModel: '', codexAuth: null,
    codexLogin: null, codexModels: [], codexBusy: false, codexPoll: null,
    projectInitialization: null, projectIssue: '', workflow: null, latestRun: null, workflowError: '',
    projectLoading: false, projectDiscoveryLoading: false,
    workflowPromise: null, workflowPromiseId: '',
    agentMode: 'research', accessMode: 'assist', pendingAuthorityRebind: false,
    demoMode: false, demoScrollTopPending: false, currentTurnResources: [],
    workflowReceipts: [], editingMessageId: '', sessionSelectionRevision: 0,
    pendingLanguageReload: false, pendingEntryIntent: '', regenerating: false, regeneration: null,
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
      .replace(/\bpi_session_data_source_confirmation_required\b/gi, tr(
        'this conversation still needs the selected data source to be confirmed',
        '本会话仍需确认所选数据源',
      ))
      .replace(/\bpi_[a-z0-9_]+\b/gi, tr('an EasyICU internal status', 'EasyICU 内部状态'))
      .replace(/\beasyicu\.webserver\.pi_copilot(?:\.[a-z0-9_.]+)?\b/gi, 'EasyICU Copilot');
  }
  function enhanceWithInteractiveFilePills(html) {
    if (!html) return '';
    return html.replace(/<code>\s*([a-zA-Z0-9_\-\u4e00-\u9fa5]+\.(?:csv|tsv|parquet|png|jpg|jpeg|pdf|json|md|py|sh))\s*<\/code>/gi, (match, filename) => {
      const isImg = /\.(png|jpg|jpeg|svg|gif)$/i.test(filename);
      const iconSvg = isImg
        ? '<svg class="pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>'
        : '<svg class="pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
      return `<button type="button" class="gpi-file-pill" data-gpi-file="${esc(filename)}" title="${tr('Click to open in workbench', '点击在工作台预览')} ${esc(filename)}" aria-label="Open ${esc(filename)}">${iconSvg}<span class="pill-name">${esc(filename)}</span><svg class="pill-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg></button>`;
    });
  }

  function assistantTextHtml(value) {
    const renderer = MODULES.require('markdown');
    const raw = renderer && typeof renderer.render === 'function'
      ? renderer.render(value)
      : esc(value).replace(/\n/g, '<br>');
    return enhanceWithInteractiveFilePills(raw);
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
  async function recordHostAction(actionCode, actionKey, childJobId) {
    const sessionId = String((state.session && state.session.session_id) || '').trim();
    const key = String(actionKey || '').trim();
    if (!sessionId || !key || !api().recordPiCopilotHostAction) return null;
    try {
      return await api().recordPiCopilotHostAction(sessionId, {
        project_id: projectId(),
        action_code: String(actionCode || ''),
        action_key: key,
        ...(childJobId ? { child_job_id: String(childJobId) } : {}),
      });
    } catch (error) {
      // A child task is already durable before this optional replay receipt is
      // written. Its watcher reconciles the terminal session state, so never
      // make the user refresh or decide how to recover a telemetry failure.
      console.warn('EasyICU could not persist the host-action receipt.', error);
      return null;
    }
  }
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
  function displaySessionTitle(value, fallback) { return window.EU_PRODUCT_LABELS?.copilotTitle?.(value, fallback) ?? String(value ?? fallback ?? '').slice(0, 200); }
  function navigationSessionTitle(row) {
    if (row && row.automatic_title) {
      return tr('New conversation', '新对话');
    }
    return displaySessionTitle(row && row.title, tr('Research conversation', '研究对话'));
  }
  function sessionStatusLabel(row) {
    if (row && (row.active_message_job_id || row.last_turn_status === 'running')) return tr('Running', '运行中');
    if (row && ['failed', 'interrupted', 'cancelled'].includes(String(row.last_turn_status || ''))) return tr('Needs attention', '需处理');
    if (row && row.last_turn_status === 'done') return tr('Completed', '已完成');
    const count = Number(row && row.history_turn_count || 0);
    if (count > 0) return tr(`${count} turns`, `${count} 轮`);
    if (row && row.has_history) return tr('Has conversation', '已有对话');
    return tr('Not started', '未开始');
  }
  function compactSessionTime(row) {
    const raw = String(row && (row.last_activity_at || row.created_at) || '').trim();
    const timestamp = Date.parse(raw);
    if (!Number.isFinite(timestamp)) return '';
    const elapsed = Math.max(0, Date.now() - timestamp);
    const minute = 60000;
    const hour = 60 * minute;
    const day = 24 * hour;
    if (elapsed < minute) return tr('now', '刚刚');
    if (elapsed < hour) return tr(`${Math.floor(elapsed / minute)}m`, `${Math.floor(elapsed / minute)} 分钟前`);
    if (elapsed < day) return tr(`${Math.floor(elapsed / hour)}h`, `${Math.floor(elapsed / hour)} 小时前`);
    if (elapsed < 7 * day) return tr(`${Math.floor(elapsed / day)}d`, `${Math.floor(elapsed / day)} 天前`);
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat(window.EU_LANG === 'zh' ? 'zh-CN' : 'en', {
      month: 'numeric', day: 'numeric',
    }).format(date);
  }
  function sessionTraceLabel(row) {
    const sessionId = String(row && row.session_id || '').trim();
    const shortId = sessionId ? sessionId.slice(-8) : '';
    const updated = String(row && row.updated_at || '').trim();
    return [updated, shortId ? `${tr('Session', '会话')} ${shortId}` : ''].filter(Boolean).join(' · ');
  }
  // D-P2-1: defensive label projection — a bundle without product-labels.js
  // must still render bounded raw text instead of throwing.
  function displayProjectTitle(value, fallback) { return window.EU_PRODUCT_LABELS?.projectTitle?.(value, fallback) ?? String(value ?? fallback ?? '').slice(0, 200); }
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
    if (state.shell !== 'pi') closeSourceView();
    if (state.conv) state.conv.classList.toggle('pi-active', state.shell === 'pi');
    render();
  }
  function closeSourceView() {
    const sourceView = MODULES.optional('sourceView');
    if (sourceView && sourceView.close) sourceView.close();
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
  function sessionIsStale() {
    return !!(state.session && state.session.stale && state.session.stale.stale);
  }

  function iconHtml(name, size) {
    return typeof window.icon === 'function' ? window.icon(name, size || 16, 1.55) : '';
  }
  const ERROR_TEXT = MODULES.require('errorText').create({
    tr, staticPreview: isStaticPreview,
  });
  const { errorText, modelErrorText, providerPreset, option, runFailureText } = ERROR_TEXT;
  const RESOURCE_OWNER = MODULES.require('resources').create({ esc });
  const resourceName = RESOURCE_OWNER.name;
  const resourceKey = RESOURCE_OWNER.key;
  const resourceLabel = RESOURCE_OWNER.label;
  const resourceButton = RESOURCE_OWNER.button;
  function runFilesContext() {
    return { projectId: projectId(), title: state.project && state.project.title,
      sessionId: state.session && state.session.session_id,
      studyId: state.session && state.session.binding && state.session.binding.study_context_id,
      runId: state.session && state.session.binding && state.session.binding.run_id,
      busy: state.busy || Boolean(state.childJobId) };
  }
  const RUN_FILES = MODULES.require('runFiles').create({
    api, context: runFilesContext, changed: () => render(true), resourceButton,
  });
  const PROVIDER_CONTROL = MODULES.require('providerControl').create({
    state, api, tr, render, runtimeReady, shellReady,
    connectionConfigured, connectionReady, errorText,
  });
  const {
    stopCodexPoll, loadCodexModels, loadCodexResearchStatus,
    startCodexLogin, openAuthorizationPopup, cancelCodexLogin, logoutCodex,
    configureProvider, finishProviderSetup,
  } = PROVIDER_CONTROL;
  const RUN_OUTCOME = MODULES.require('runOutcome').create({
    tr, esc, iconHtml, resourceButton, api, projectId, host: () => state.host,
    canPreview: () => Boolean(state.session) && !state.busy && !state.childJobId && !sessionIsStale(),
    preview: () => MODULES.optional('preview'),
    workflowContext: previewWorkflowContext,
    errorText,
    recordHostAction,
    onError: value => { state.error = value; render(); },
  });
  const ACTIVITY = MODULES.require('activity').create({
    tr, esc, iconHtml, resourceName, resourceKey, resourceButton,
    publicText: publicAssistantText,
  });
  const CONFIRMATION = MODULES.require('confirmation').create({
    tr, esc, iconHtml, resourceButton, sessionIsStale, runFailureText,
    workflow: () => state.workflow,
    session: () => state.session,
    busy: () => state.busy || Boolean(state.childJobId),
    planConfigurationError: () => state.planConfigurationError,
    cohortEligibilityDecisionHtml: copies => COHORT_ELIGIBILITY.repeatedStayDecisionHtml(copies),
  });
  const workflowConfirmation = CONFIRMATION.workflowConfirmation;
  const workflowConfirmationHtml = CONFIRMATION.workflowConfirmationHtml;
  const COHORT_ELIGIBILITY = MODULES.require('cohortEligibility').create({
    tr, esc,
    session: () => state.session,
    workflow: () => state.workflow,
    busy: () => state.busy || Boolean(state.childJobId),
    sessionIsStale,
    planConfigurationError: () => state.planConfigurationError,
  });
  const { timeMs } = ACTIVITY;
  const TRANSCRIPT = MODULES.require('transcript').create({
    tr, activity: ACTIVITY, upsertActivityStep, timeMs, resourceKey, modelErrorText,
    activityHasCompletedAction,
    workflowActionCode: () => String((state.workflow && state.workflow.next_action_code) || ''),
  });
  const transcriptMessages = TRANSCRIPT.transcriptMessages;
  const CHILDJOB = MODULES.require('childJob').create({
    tr, activity: ACTIVITY, upsertActivityStep, api,
    render: () => render(),
    loadWorkflow: (...args) => loadWorkflow(...args),
    sessionIsStale: () => sessionIsStale(),
    rebind: () => rebind(),
    refreshSession: (...args) => refreshSession(...args),
    archiveChildJob: (...args) => archiveChildJob(...args),
    // The callback is evaluated only after PLAN_ACTIONS is initialized.  A
    // completed candidate or a plan with system-owned findings may therefore
    // continue once without inventing a user chat turn or another review gate.
    continueSystemOwnedPlanProgression: (...args) => (
      PLAN_ACTIONS && PLAN_ACTIONS.continueSystemOwnedPlanProgression(...args)
    ),
    messages: () => state.messages,
    session: () => state.session,
    childJobId: () => state.childJobId,
    setChildJobId: value => { state.childJobId = value; },
    childSource: () => state.childSource,
    setChildSource: value => { state.childSource = value; },
  });
  const ASIDE = MODULES.require('aside').create({
    tr, esc, iconHtml, api,
    projectId: () => projectId(),
    displayProjectTitle: (...args) => displayProjectTitle(...args),
    demoMode: () => state.demoMode,
    project: () => state.project,
    shell: () => state.shell,
    workflow: () => state.workflow,
    latestRun: () => state.latestRun,
    workflowError: () => state.workflowError,
    resultsHtml: (query, options) => RUN_OUTCOME.renderShelf(state.latestRun, state.workflow, query, options),
    reviewActionHtml: () => RUN_OUTCOME.renderReviewAction(state.latestRun, state.workflow),
    hasPendingReview: () => Boolean(state.host && state.host.querySelector('.gpi-confirmation')),
    // The to-do list names the decision the conversation is waiting for in
    // the decision card's own words, or the task EasyICU is running now.
    pendingDecisionTitle: () => {
      if (!state.session) return '';
      if (DATA_CONSENT.requiresConfirmation(state.session)) {
        return tr('Confirm the data source for this conversation', '确认本次会话使用的数据来源');
      }
      const code = String((state.workflow && state.workflow.next_action_code) || '');
      if (code === 'cohort_eligibility_confirmation_required') {
        return tr('Decide how repeat ICU stays are handled', '确认同一患者多次住 ICU 的处理方式');
      }
      const confirmation = workflowConfirmation();
      return confirmation && confirmation.title ? String(confirmation.title) : '';
    },
    activeTaskTitle: () => {
      const running = state.messages.slice().reverse()
        .find(row => row && row.role === 'activity' && row.childJobId && row.status === 'running');
      if (running) return String(running.runningTitle || tr('EasyICU research task is running', 'EasyICU 科研任务正在运行'));
      return state.busy ? tr('EasyICU is replying', 'EasyICU 正在回复') : '';
    },
    openResource: button => EVENTS.openResourceButton(button),
    revealPendingReview: () => EVENTS.revealPendingReview(),
    progressLabel: event => ACTIVITY.pipelineEventLabel(event),
    runFailureText,
  });
  const syncProjectWorkflowAside = ASIDE.syncProjectWorkflowAside;
  const DATA_BINDING = MODULES.require('dataBinding').create({
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
  const HOST_JOBS = MODULES.require('hostJobs').create({
    tr, api, iconHtml, errorText,
    dataConsent: DATA_CONSENT,
    render: () => render(),
    loadWorkflow: (...args) => loadWorkflow(...args),
    session: () => state.session,
    messages: () => state.messages,
    busy: () => state.busy || Boolean(state.childJobId),
    workflowReceipts: () => state.workflowReceipts,
    setWorkflowReceipts: value => { state.workflowReceipts = value; },
    setError: value => { state.error = value; },
    rebind: () => rebind(),
    authorizeDataSource: (...args) => authorizeDataSource(...args),
    confirmDataSourceBinding: (...args) => confirmDataSourceBinding(...args),
  });
  const closeChildSource = CHILDJOB.closeChildSource;
  const childActivity = CHILDJOB.childActivity;
  const handleChildJobEvent = CHILDJOB.handleChildJobEvent;
  const watchChildJob = CHILDJOB.watchChildJob;
  const hydrateProjectedJob = CHILDJOB.hydrateProjectedJob;
  const PLAN_ACTIONS = MODULES.require('planActions').create({
    tr, errorText, regeneration: REGENERATION,
    nextActions: MODULES.require('nextActions'),
    replay: MODULES.require('replay'),
    session: () => state.session,
    selectionRevision: () => state.sessionSelectionRevision,
    workflow: () => state.workflow,
    // The run the host's own projection treats as authoritative. A governed
    // action must name the run the offer was computed from, not the run id the
    // session last happened to bind.
    latestRun: () => state.latestRun,
    busy: () => state.busy || Boolean(state.childJobId),
    sessionIsStale,
    researchSourceReady: () => !DATA_CONSENT.requiresConfirmation(state.session),
    api, projectId, turnGrants, sendText, render, watchChildJob,
    recordHostAction,
    refreshSession: (...args) => refreshSession(...args),
    loadWorkflow: (...args) => loadWorkflow(...args),
    setBusy: value => { state.busy = Boolean(value); },
    setError: value => { state.error = String(value || ''); },
    // The host compiler may refuse the automatic configuration (for example
    // an all-stay analysis without source-owned patient grouping). The
    // confirmation and cohort owners read this to offer the researcher's
    // decision instead of leaving a bare error banner.
    setPlanConfigurationError: value => { state.planConfigurationError = String(value || ''); },
    setDraft: value => { state.draft = String(value || ''); },
    appendMessage: value => { state.messages.push(value); },
    truncateMessagesAt: id => {
      const at = state.messages.findIndex(item => String((item && item.id) || '') === id);
      if (at >= 0) state.messages.splice(at);
    },
  });
  const confirmWorkflowAction = () => PLAN_ACTIONS.confirmWorkflow(workflowConfirmation());
  const rejectWorkflowAction = () => PLAN_ACTIONS.rejectWorkflow(workflowConfirmation());
  const confirmPlanDecision = PLAN_ACTIONS.confirmDecision;
  const retryFailedExecution = PLAN_ACTIONS.retryFailedExecution;
  const startCurrentFormalPlanGeneration = PLAN_ACTIONS.startFormalPlanGeneration;
  const governedNextChoiceGrants = PLAN_ACTIONS.governedNextChoiceGrants;
  const MESSAGE_ACTIONS = MODULES.require('messageActions').create({
    tr, iconHtml,
    rows: () => state.messages.concat(state.workflowReceipts),
    canEdit: () => !state.busy && !state.childJobId && !sessionIsStale(),
    setEditing: id => { state.editingMessageId = id; },
    renderHost: render,
    sendText,
    regenerate: regenerateMessage,
    resubmitHostGenerated: PLAN_ACTIONS.resubmitHostGenerated,
    host: () => state.host,
  });
  const EVENTS = MODULES.require('events').create({
    state, RESOURCE_OWNER, RUN_FILES, MESSAGE_ACTIONS, STARTERS, IDEA_SOURCE, COHORT_ELIGIBILITY,
    DATA_CONSENT, RUN_OUTCOME, STUDY_WORKSPACE, ASIDE, render, projectId, previewWorkflowContext,
    openSession, closeDemo, openDemo, switchMode, loadCodexResearchStatus,
    openAuthorizationPopup, startCodexLogin, cancelCodexLogin, logoutCodex,
    loadCodexModels, tr, apiResearchReady, finishProviderSetup, loadStatus,
    setShell, openStudySetupInConversation, createSession, startEntry,
    previewApprovedPlanDataPackage, confirmWorkflowAction,
    retryFailedExecution,
    rejectWorkflowAction, editWorkflow, confirmCohortEligibility,
    confirmPlanDecision,
    authorizeDataSource, sendText, continueAfterDataSourceConfirmation,
    governedNextChoiceGrants, sendMessage, stopMessage, stopChildJob, rebind,
    togglePresentationPin, configureProvider, rememberSession, recordHostAction,
    HOST_JOBS, EFFORT_MENU, api,
  });
  const dismissHeaderOverflow = EVENTS.dismissHeaderOverflow;
  const wire = EVENTS.wire;

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
    const owner = MODULES.require('provider');
    if (!owner || typeof owner.renderSetup !== 'function') return '';
    return owner.renderSetup({
      state, runtime, config, blockers, runtimeMissing, tr, esc, option,
      providerPreset, runtimeReady: runtimeReady(), apiResearchReady: apiResearchReady(),
      connectionConfigured: connectionConfigured(), connectionReady: connectionReady(),
      staticPreview: isStaticPreview(),
    });
  }

  function providerBindingSummary() {
    const owner = MODULES.require('provider');
    return owner && typeof owner.renderBindingSummary === 'function'
      ? owner.renderBindingSummary({ state, tr, esc, runtimeReady: runtimeReady(), apiResearchReady: apiResearchReady(), connectionReady: connectionReady() })
      : '';
  }

  function activatePanel() {
    const saved = state.sessions.filter(sessionMatchesUiLanguage).map(row => `
      <button class="gpi-session-row" type="button" data-gpi-session="${esc(row.session_id)}">
        <span><strong>${esc(displaySessionTitle(row.title))}</strong><small>${esc(sessionTraceLabel(row))}</small></span>
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
      </div>`;
  }

  function restoringPanel() {
    return `
      <div class="gpi-panel gpi-restoring" aria-busy="true">
        <header class="gpi-head"><div class="gpi-head-title"><div class="gpi-kicker">EasyICU</div><div class="gpi-title">${esc(displayProjectTitle(state.project && state.project.title, projectId()))}</div></div></header>
        <div class="gpi-log" data-gpi-log><div class="gpi-restore-message" role="status" aria-live="polite"><span class="gpi-running-spinner" aria-hidden="true"></span>${tr('Loading this project’s conversations…', '正在读取当前项目的对话…')}</div></div>
        <div class="gpi-compose"><div class="gpi-compose-card"><textarea disabled aria-label="${tr('Message', '消息')}" placeholder="${tr('Continue when the conversation loads', '对话加载后可继续提问')}"></textarea></div></div>
      </div>`;
  }
  const SESSION_VIEW = MODULES.require('sessionView').create({
    state, modules: MODULES, dataConsent: DATA_CONSENT, starters: STARTERS,
    ideaSource: IDEA_SOURCE, header: HEADER, regeneration: REGENERATION,
    studyWorkspace: STUDY_WORKSPACE, activity: ACTIVITY, runFiles: RUN_FILES, resourceOwner: RESOURCE_OWNER,
    messageActions: MESSAGE_ACTIONS, transcript: TRANSCRIPT, runOutcome: RUN_OUTCOME, aside: ASIDE,
    cohortEligibility: COHORT_ELIGIBILITY, tr, esc, iconHtml, projectId, publicAssistantText,
    assistantTextHtml, sessionIsStale, agentMode, accessModeLabel,
    projectTitle: () => displayProjectTitle(state.project && state.project.title, projectId()), navigationSessionTitle,
    workflowConfirmationHtml, hostJobs: HOST_JOBS, followUps: MODULES.require('followUps'),
    effortMenu: EFFORT_MENU,
  });
  const { messageHtml, workflowHtml, sessionPanel, demoPanel } = SESSION_VIEW;
  function openDemo() {
    const demo = MODULES.optional('demo');
    if (!demo || typeof demo.messages !== 'function') return;
    const preview = MODULES.optional('preview');
    if (preview && preview.close) preview.close();
    state.demoMode = true;
    state.demoScrollTopPending = true;
    state.error = '';
    setShell('pi');
    const primary = typeof demo.primaryDocument === 'function' ? demo.primaryDocument() : null;
    if (primary && preview && preview.open) {
      preview.open(primary, projectId());
    }
  }
  function closeDemo() {
    state.demoMode = false;
    state.demoScrollTopPending = false;
    const preview = MODULES.optional('preview');
    if (preview && preview.close) preview.close();
    render();
  }
  function render(preserveScroll) {
    if (!state.host) return;
    STUDY_WORKSPACE.capture(state.host);
    const previousLog = preserveScroll && state.host.querySelector('[data-gpi-log]');
    const previousTop = previousLog ? previousLog.scrollTop : null;
    const previousConnection = state.host.querySelector('[data-gpi-connection-page]');
    const previousConnectionTop = previousConnection ? previousConnection.scrollTop : null;
    if (!state.demoMode) RUN_FILES.sync();
    const restoring = !state.session && (state.loading || state.projectLoading || state.projectDiscoveryLoading);
    const setupFocused = !restoring && state.shell !== 'legacy'
      && !state.demoMode
      && (state.showSetup || !connectionReady() || state.projectIssue === 'pi_project_study_context_missing');
    const emptySessionFocused = !restoring && !setupFocused && state.shell !== 'legacy'
      && !state.demoMode && Boolean(state.session) && agentMode() !== 'workspace'
      && state.messages.length === 0 && state.workflowReceipts.length === 0;
    const main = state.host.closest('.gd-main');
    if (main) {
      main.classList.toggle('gpi-workspace', state.shell !== 'legacy');
      main.classList.toggle('gpi-setup-focus', setupFocused);
      main.classList.toggle('gpi-empty-session-focus', emptySessionFocused);
    }
    const desktopShell = window.EU_DESKTOP_MODULE_SHELL;
    if (desktopShell && desktopShell.rememberContext && projectId()) {
      desktopShell.rememberContext({
        projectId: projectId(),
        projectTitle: displayProjectTitle(state.project && state.project.title, projectId()),
        sessionId: state.session && state.session.session_id,
        sessionTitle: state.session ? navigationSessionTitle(state.session) : tr('New conversation', '新对话'),
      });
    }
    state.host.hidden = false;
    state.host.innerHTML = restoring
      ? restoringPanel()
      : state.shell === 'legacy'
      ? statusBanner()
      : (state.demoMode ? demoPanel() : (state.projectIssue === 'pi_project_study_context_missing'
        ? activatePanel()
        : ((state.showSetup || !connectionReady()) ? setupPanel() : (!projectId() ? projectRequiredPanel() : (state.session ? sessionPanel() : activatePanel())))));
    const preview = MODULES.optional('preview');
    if (preview && preview.setWorkflowContext) {
      preview.setWorkflowContext(previewWorkflowContext());
    }
    if (preview && preview.setStudyResources) {
      preview.setStudyResources(RUN_OUTCOME.collection(state.latestRun, state.workflow), projectId(), EVENTS.openResource, {
        title: displayProjectTitle(state.project && state.project.title, projectId()),
        reference: EVENTS.referenceResource,
      });
    }
    syncProjectWorkflowAside();
    STUDY_WORKSPACE.syncNavigation({ projectId: projectId(), sessions: state.sessions.filter(sessionMatchesUiLanguage),
      selectedId: state.session && state.session.session_id, loading: restoring, disabled: state.busy || Boolean(state.childJobId) || state.projectLoading,
      visible: state.shell !== 'legacy', title: navigationSessionTitle, status: sessionStatusLabel,
      time: compactSessionTime, open: openSession, create: startEntry,
      resources: RUN_OUTCOME.collection(state.latestRun, state.workflow), openResource: EVENTS.openResource,
      materialsInteractive: !emptySessionFocused, rename: renameSession, remove: removeEmptySession });
    requestAnimationFrame(() => {
      STUDY_WORKSPACE.restoreMaterials(state.host);
      // Opening a deep-linked artifact while the project/session is still
      // restoring lets openSession() immediately close it and erase the URL.
      // Wait until the selected conversation is authoritative, then restore
      // the digest-pinned artifact/evidence view once the normal shell exists.
      if (!restoring && state.session && preview && preview.restoreFromLocation) {
        preview.restoreFromLocation(projectId(), previewWorkflowContext());
      }
      const log = state.host && state.host.querySelector('[data-gpi-log]');
      if (log) {
        log.scrollTop = previousTop !== null ? previousTop : state.demoScrollTopPending ? 0 : log.scrollHeight;
        state.demoScrollTopPending = false;
      }
      const connectionPage = state.host && state.host.querySelector('[data-gpi-connection-page]');
      if (connectionPage) connectionPage.scrollTop = previousConnectionTop === null ? 0 : previousConnectionTop;
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

  async function createSession() {
    if (state.creating || state.projectLoading || !projectId()) return;
    closeSourceView();
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
        thinking_level: EFFORT_MENU.preferred(), external_llm_opt_in: true,
        research_provider: state.researchProvider,
        research_model: state.researchProvider === 'codex' ? state.researchModel : null,
      });
      if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
      state.session = payload.session; state.messages = transcriptMessages(state.session);
      STUDY_WORKSPACE.applyHubIntent(expectedProjectId, state.session);
      state.agentMode = state.session.agent_mode || state.agentMode;
      hydrateProjectedJob(state.workflow && state.workflow.active_job);
      state.projectInitialization = null;
      rememberSession(state.session.session_id);
      const projectOwner = MODULES.require('project');
      if (projectOwner && projectOwner.syncLocation) {
        projectOwner.syncLocation(expectedProjectId, state.session.session_id);
      }
      state.sessions = [state.session].concat(state.sessions.filter(row => row.session_id !== state.session.session_id));
    } catch (error) { state.error = errorText(error); }
    finally { state.creating = false; render(); }
  }

  async function renameSession(row) {
    if (!row || state.busy || state.childJobId || !api().renamePiCopilotSession) return;
    const current = navigationSessionTitle(row);
    const requested = window.prompt(tr('Rename this conversation', '重命名对话'), current);
    if (requested === null) return;
    const title = String(requested || '').replace(/\s+/g, ' ').trim().slice(0, 100);
    if (!title || title === current) return;
    try {
      const payload = await api().renamePiCopilotSession(row.session_id, {
        project_id: projectId(), title,
      });
      const renamed = payload && payload.session;
      if (!renamed) return;
      state.sessions = state.sessions.map(item => item.session_id === renamed.session_id ? { ...item, ...renamed } : item);
      if (state.session && state.session.session_id === renamed.session_id) state.session = { ...state.session, ...renamed };
      state.error = '';
    } catch (error) {
      state.error = errorText(error);
    }
    render();
  }

  async function removeEmptySession(row) {
    if (!row || row.has_history || state.busy || state.childJobId || !api().deleteEmptyPiCopilotSession) return;
    if (!window.confirm(tr('Remove this empty conversation? No message or result will be deleted.', '删除这个空对话吗？此操作不会删除任何消息或研究成果。'))) return;
    const deletingSelected = Boolean(state.session && state.session.session_id === row.session_id);
    try {
      await api().deleteEmptyPiCopilotSession(row.session_id, {
        project_id: projectId(), confirm_empty: true,
      });
      state.sessions = state.sessions.filter(item => item.session_id !== row.session_id);
      state.error = '';
      if (!deletingSelected) { render(); return; }
      state.sessionSelectionRevision += 1;
      state.session = null;
      state.messages = [];
      state.workflowReceipts = [];
      state.draft = '';
      rememberSession('');
      const preview = MODULES.optional('preview');
      if (preview && preview.close) preview.close();
      const projectOwner = MODULES.require('project');
      if (projectOwner && projectOwner.syncLocation) projectOwner.syncLocation(projectId(), '');
      render();
      const next = state.sessions.find(sessionMatchesUiLanguage);
      if (next) await openSession(next.session_id);
    } catch (error) {
      state.error = errorText(error);
      render();
    }
  }

  async function openSession(sessionId, selectionRevision, refreshWorkflow) {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    closeSourceView();
    const preview = MODULES.optional('preview');
    const previewParams = typeof window !== 'undefined' && window.location
      ? new URLSearchParams(window.location.search || '') : null;
    const preservePreviewLocation = Boolean(previewParams
      && previewParams.get('pi_session') === String(sessionId || '')
      && ['artifact', 'evidence'].includes(previewParams.get('pi_view')));
    if (preview && preview.close) preview.close({ preserveLocation: preservePreviewLocation });
    const expectedSelectionRevision = selectionRevision == null
      ? ++state.sessionSelectionRevision : selectionRevision;
    closeChildSource();
    state.error = '';
    state.planConfigurationError = '';
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
      const replayOwner = MODULES.require('replay');
      const hydrated = replayOwner && typeof replayOwner.hydrate === 'function'
        ? await replayOwner.hydrate(api(), payload.session, expectedProjectId)
        : payload.session;
      if (expectedProjectId !== projectId() || expectedSelectionRevision !== state.sessionSelectionRevision) return;
      state.session = hydrated;
      if (state.sessions.some(row => row.session_id === state.session.session_id)) {
        state.sessions = state.sessions.map(row => row.session_id === state.session.session_id
          ? { ...row, ...state.session }
          : row);
      } else {
        state.sessions = [state.session].concat(state.sessions);
      }
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
      HOST_JOBS.sync();
      rememberSession(sessionId); setShell('pi');
      const projectOwner = MODULES.require('project');
      if (projectOwner && projectOwner.syncLocation) {
        projectOwner.syncLocation(expectedProjectId, sessionId);
      }
      // The transcript is ready now. Show it while the independent workflow
      // projection finishes, rather than keeping the conversation blocked.
      render();
      if (window.EU_GUIDED_STARTUP && window.EU_GUIDED_STARTUP.finish) {
        window.EU_GUIDED_STARTUP.finish(document);
      }
      if (refreshWorkflow !== false || !state.workflow) await loadWorkflow();
      await PLAN_ACTIONS.continueSystemOwnedPlanProgression({passive: true});
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
  function activityHasCompletedAction(activity) {
    return Boolean(activity && Array.isArray(activity.steps) && activity.steps.some(step => {
      const toolName = String((step && step.toolName) || '');
      return step && step.kind === 'tool' && step.status === 'complete'
        && toolName && !/^easyicu_(inspect|list)_/.test(toolName);
    }));
  }
  async function switchMode(mode) {
    const next = mode === 'research' ? 'research' : 'workspace';
    if (state.busy || next === agentMode()) return;
    closeSource();
    closeChildSource();
    closeSourceView();
    state.error = '';
    state.pendingAuthorityRebind = false;
    const preview = MODULES.optional('preview');
    if (preview && preview.close) {
      preview.close();
    }
    const replayOwner = MODULES.require('replay');
    let existingSessionId = replayOwner && typeof replayOwner.preferredSessionId === 'function'
      ? replayOwner.preferredSessionId(state.sessions, '', next, uiLanguage())
      : String(state.sessions.find(row => (row.agent_mode || 'research') === next && sessionMatchesUiLanguage(row))?.session_id || '');
    if (!existingSessionId && projectId()) {
      const expectedProjectId = projectId();
      const selectionRevision = state.sessionSelectionRevision;
      try {
        const listed = await api().loadPiCopilotSessions(100, projectId(), next);
        if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
        const matching = Array.isArray(listed && listed.sessions) ? listed.sessions : [];
        if (matching.length) {
          const matchingIds = new Set(matching.map(row => row.session_id));
          state.sessions = matching.concat(state.sessions.filter(row => !matchingIds.has(row.session_id)));
          existingSessionId = replayOwner && typeof replayOwner.preferredSessionId === 'function'
            ? replayOwner.preferredSessionId(matching, '', next, uiLanguage())
            : String(matching.find(sessionMatchesUiLanguage)?.session_id || '');
        }
      } catch (error) {
        if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
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
  const LIVE_STREAM = MODULES.require('liveStream').create({
    state, timeMs, ensureActivity, upsertActivityStep, finishActivity,
    assistantRow, addAssistantResources, completeLatestAssistant,
    activityHasCompletedAction, modelErrorText, activity: ACTIVITY,
    modules: MODULES, projectId: () => projectId(),
    loadWorkflow: (...args) => loadWorkflow(...args), render: () => render(),
    watchChildJob: (...args) => watchChildJob(...args), hostJobs: HOST_JOBS,
    assistantTextHtml, publicAssistantText, followUps: MODULES.require('followUps'),
  });
  const handlePiEvent = LIVE_STREAM.handlePiEvent;
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
    const expectedProjectId = projectId();
    const expectedSessionId = state.session.session_id;
    try {
      const payload = await api().loadPiCopilotSession(expectedSessionId, expectedProjectId);
      if (projectId() !== expectedProjectId || !state.session || state.session.session_id !== expectedSessionId) return;
      const replayOwner = MODULES.require('replay');
      const refreshed = !preserveTimeline && replayOwner && typeof replayOwner.hydrate === 'function'
        ? await replayOwner.hydrate(api(), payload.session, expectedProjectId)
        : payload.session;
      if (projectId() !== expectedProjectId || !state.session || state.session.session_id !== expectedSessionId) return;
      state.session = refreshed;
      state.sessions = [state.session].concat(
        state.sessions.filter(row => row.session_id !== state.session.session_id)
      );
      if (!preserveTimeline) state.messages = transcriptMessages(state.session);
      (Array.isArray(state.session.archived_child_jobs) ? state.session.archived_child_jobs : []).forEach(hydrateProjectedJob);
      reconcileSettledSession();
      HOST_JOBS.sync();
    } catch (error) {
      if (projectId() !== expectedProjectId || !state.session || state.session.session_id !== expectedSessionId) return;
      state.error = tr('Conversation refresh failed. Your saved records are unchanged: ', '对话刷新失败，已保存记录未改变：') + errorText(error);
    }
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
    const projectOwner = MODULES.require('project');
    const requested = projectOwner && typeof projectOwner.requestedSessionId === 'function'
      ? projectOwner.requestedSessionId(expectedProjectId)
      : '';
    const requestedRow = requested
      ? state.sessions.find(row => row.session_id === requested && sessionMatchesUiLanguage(row))
      : null;
    const remembered = rememberedSession();
    const replayOwner = MODULES.require('replay');
    const preferred = requestedRow
      ? requested
      : (replayOwner && typeof replayOwner.preferredSessionId === 'function'
      ? replayOwner.preferredSessionId(state.sessions, remembered, '', uiLanguage())
      : (remembered && state.sessions.some(row => row.session_id === remembered && sessionMatchesUiLanguage(row))
        ? remembered
        : String(state.sessions.find(sessionMatchesUiLanguage)?.session_id || '')));
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
    const preview = MODULES.optional('preview');
    if (preview && preview.close) {
      preview.close();
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

  function loadWorkflow() {
    const expectedProjectId = projectId();
    if (!expectedProjectId || !api().loadPiCopilotProjectWorkflow) return Promise.resolve();
    if (state.workflowPromise && state.workflowPromiseId === expectedProjectId) {
      return state.workflowPromise;
    }
    const pending = (async () => {
      try {
        const payload = await api().loadPiCopilotProjectWorkflow(expectedProjectId);
        if (expectedProjectId !== projectId()) return;
        state.workflowError = '';
        state.workflow = payload && payload.workflow ? payload.workflow : null;
        state.latestRun = payload && payload.latest_run ? payload.latest_run : { present: false };
        void RUN_OUTCOME.loadScientificReview(state.latestRun, state.workflow);
        if (state.workflow) state.workflow.active_job = (payload && payload.active_job) || { present: false };
        // The project's run record travels beside the snapshot; the aside
        // reads it from the same workflow object as the stages.
        if (state.workflow) state.workflow.runs = Array.isArray(payload && payload.runs) ? payload.runs : [];
        reconcileDurableWrapupActivity();
        hydrateProjectedJob(payload && payload.active_job);
        const activeJob = payload && payload.active_job;
        if (activeJob && activeJob.present && activeJob.status === 'running' && activeJob.job_id) {
          const kind = String(activeJob.kind || '');
          const code = activeJob.report_only === true
            ? 'easyicu_report_repair_submitted'
            : /extract/i.test(kind)
            ? 'easyicu_extraction_submitted'
            : (/research|agent/i.test(kind) ? 'easyicu_full_run_submitted' : 'easyicu_run_submitted');
          watchChildJob(String(activeJob.job_id), code);
        }
      } catch (error) {
        if (expectedProjectId === projectId()) {
          state.workflow = null;
          state.latestRun = null;
          state.workflowError = tr('Could not read current task status. Refresh to retry: ', '无法读取当前任务状态，请刷新重试：') + errorText(error);
        }
      }
    })();
    state.workflowPromiseId = expectedProjectId;
    const settled = pending.finally(() => {
      if (state.workflowPromiseId !== expectedProjectId || state.workflowPromise !== settled) return;
      state.workflowPromise = null;
      state.workflowPromiseId = '';
    });
    state.workflowPromise = settled;
    return settled;
  }

  async function prepareProject() {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    if (state.projectPreparePromise && state.projectPrepareId === expectedProjectId) {
      return state.projectPreparePromise;
    }
    const pending = MODULES.require('project').prepare({
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
    if (sameProject && JSON.stringify(currentReceipt || null) === JSON.stringify((next && next.binding_receipt) || null)) {
      if (next && state.project && state.project.title !== next.title) {
        state.project = { ...state.project, title: next.title };
        render();
      }
      return Promise.resolve();
    }
    closeSource();
    closeChildSource();
    closeSourceView();
    state.sessionSelectionRevision += 1;
    state.demoMode = false;
    state.demoScrollTopPending = false;
    state.project = next;
    STUDY_WORKSPACE.removeReference();
    STUDY_WORKSPACE.removeSkill();
    STUDY_WORKSPACE.removeMethod();
    STUDY_WORKSPACE.removeBuilder();
    const projectOwner = MODULES.require('project');
    if (projectOwner && projectOwner.syncLocation) {
      const requestedSession = next && projectOwner.requestedSessionId
        ? projectOwner.requestedSessionId(next.id)
        : '';
      projectOwner.syncLocation(next && next.id, requestedSession);
    }
    state.session = null;
    state.sessions = [];
    state.messages = [];
    state.draft = '';
    if (IDEA_SOURCE) IDEA_SOURCE.reset();
    state.pendingEntryIntent = '';
    state.currentTurnResources = [];
    state.childJobId = '';
    state.regenerating = false;
    state.regeneration = null;
    // Extraction receipts are project-scoped UI state. Keeping them while the
    // user switches to a blank project makes the new conversation look as if
    // it inherited the previous project's data configuration.
    state.workflowReceipts = [];
    state.planConfigurationError = '';
    HOST_JOBS.stopAll();
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
    const preview = MODULES.optional('preview');
    if (preview && preview.clearProject) {
      const previewParams = typeof window !== 'undefined' && window.location
        ? new URLSearchParams(window.location.search || '') : null;
      const preservePreviewLocation = Boolean(next && previewParams
        && previewParams.get('pi_project') === next.id
        && ['artifact', 'evidence'].includes(previewParams.get('pi_view')));
      preview.clearProject({ preserveLocation: preservePreviewLocation });
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
        const completedAction = activityHasCompletedAction(activeActivity());
        const wrapupTimedOut = row.status === 'failed'
          && /^pi_gateway_timeout(?:\s*:|$)/.test(String(row.error || ''));
        if (row.status === 'failed') {
          finishActivity('error', null, 'failed');
          state.error = /^pi_model_/.test(String(row.error || ''))
            ? modelErrorText(row.error, completedAction)
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
        const continued = await PLAN_ACTIONS.continueSystemOwnedPlanProgression();
        if (!continued) render();
      }
    };
    state.source.onerror = () => { if (!state.busy) closeSource(); };
  }
  async function sendText(text, grantsOverride, turnIntent, visibleUserMessage = true, messageOrigin = '') {
    if (!state.session || state.projectLoading || state.busy || state.childJobId || sessionIsStale()) return;
    if (!sessionMatchesUiLanguage(state.session)) {
      handleLanguageChange();
      return;
    }
    text = String(text || '').trim();
    if (!text) return;
    const expectedSessionId = state.session.session_id;
    const expectedProjectId = projectId();
    const selectionRevision = state.sessionSelectionRevision;
    const isCurrent = () => state.session && state.session.session_id === expectedSessionId
      && projectId() === expectedProjectId && state.sessionSelectionRevision === selectionRevision;
    let ideaSource = null;
    try {
      ideaSource = IDEA_SOURCE
        ? await IDEA_SOURCE.prepareForMessage(text, turnIntent || '')
        : null;
    } catch (error) {
      if (!isCurrent()) return;
      state.error = errorText(error); render(); return;
    }
    if (!isCurrent() || state.busy || state.childJobId) return;
    state.editingMessageId = '';
    const grants = Array.isArray(grantsOverride) ? grantsOverride : turnGrants();
    const submittedAt = Date.now();
    state.currentTurnResources = [];
    if (visibleUserMessage) state.messages.push({ id: 'user-' + submittedAt, role: 'user', text, complete: true });
    const activity = ensureActivity(new Date(submittedAt).toISOString());
    upsertActivityStep(activity, { id: 'submitted', kind: 'submitted', status: 'complete', at: submittedAt });
    if (visibleUserMessage) { state.draft = ''; STUDY_WORKSPACE.consume(expectedProjectId, expectedSessionId); STUDY_WORKSPACE.consumeSkill(expectedProjectId, state.session); }
    state.busy = true; state.error = ''; render();
    try {
      const payload = await api().sendPiCopilotMessage(state.session.session_id, {
        project_id: projectId(), message: text, allowed_actions: grants,
        ...(turnIntent ? { turn_intent: turnIntent } : {}),
        ...(ideaSource ? { idea_source: ideaSource } : {}),
        ...(messageOrigin ? { message_origin: messageOrigin } : {}),
      });
      if (!isCurrent()) return;
      if (ideaSource && IDEA_SOURCE) IDEA_SOURCE.consume();
      state.pendingEntryIntent = '';
      state.jobId = payload.job_id; watchJob(payload.job_id);
    } catch (error) {
      if (!isCurrent()) return;
      state.busy = false; finishActivity('error', null, 'failed');
      state.error = errorText(error); render();
    }
  }
  async function regenerateMessage(userEntryId, text, regenerationIntent, targetMessageId) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const entryId = String(userEntryId || '').trim();
    text = String(text || '').trim();
    if (!entryId || !text) return;
    const expectedSessionId = state.session.session_id;
    const expectedProjectId = projectId();
    const selectionRevision = state.sessionSelectionRevision;
    const isCurrent = () => state.session && state.session.session_id === expectedSessionId
      && projectId() === expectedProjectId && state.sessionSelectionRevision === selectionRevision;
    state.editingMessageId = '';
    state.currentTurnResources = [];
    state.regeneration = REGENERATION && typeof REGENERATION.create === 'function'
      ? REGENERATION.create(state.messages, {
        userEntryId: entryId,
        targetMessageId: String(targetMessageId || ''),
        startedAt: Date.now(),
      }) : null;
    if (regenerationIntent === 'user_edited_message') {
      const edited = state.messages.find(row => (
        row && row.role === 'user' && String(row.entryId || '') === entryId
      ));
      if (edited) edited.text = text;
    }
    state.regenerating = true;
    state.busy = true; state.error = ''; render();
    try {
      const authority = PLAN_ACTIONS.regenerationAuthority(text, regenerationIntent);
      const payload = await api().regeneratePiCopilotMessage(state.session.session_id, {
        project_id: projectId(), user_entry_id: entryId,
        message: text, allowed_actions: authority.grants,
        ...(authority.intent ? { turn_intent: authority.intent } : {}),
        ...(regenerationIntent ? { regeneration_intent: regenerationIntent } : {}),
      });
      if (!isCurrent()) return;
      state.jobId = payload.job_id;
      watchJob(payload.job_id);
    } catch (error) {
      if (!isCurrent()) return;
      state.busy = false;
      state.regenerating = false;
      state.regeneration = null;
      state.error = errorText(error);
      render();
    }
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
    if (!state.session || state.projectLoading || state.busy || state.childJobId || sessionIsStale()) return;
    const input = state.host.querySelector('[data-gpi-input]');
    const text = String((input && input.value) || state.draft || '').trim();
    const referencing = STUDY_WORKSPACE.hasReference(projectId(), state.session.session_id);
    if (!referencing && await PLAN_ACTIONS.continueUserRequestedSystemProgression(text)) return;
    const intent = referencing ? undefined : state.pendingEntryIntent
      || (IDEA_SOURCE && IDEA_SOURCE.suggestsIdeaMining(text) ? 'idea_mining_entry' : undefined);
    await sendText(STUDY_WORKSPACE.decorateMessage(
      STUDY_WORKSPACE.decorateSkillMessage(text, projectId(), state.session), projectId(), state.session.session_id,
    ), undefined, intent);
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
  async function previewApprovedPlanDataPackage(button) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const preview = MODULES.optional('preview');
    if (!api().preparePiCopilotDataPackageReview || !preview || !preview.open) {
      state.error = tr('The data preview is temporarily unavailable. Refresh this project and try again.', '数据预览暂时不可用，请刷新当前项目后重试。');
      render();
      return;
    }
    const expectedProjectId = projectId();
    const original = button ? button.textContent : '';
    if (button) {
      button.disabled = true;
      button.textContent = tr('Preparing preview…', '正在准备预览…');
    }
    try {
      const payload = await api().preparePiCopilotDataPackageReview(expectedProjectId);
      if (projectId() !== expectedProjectId) return;
      const resource = payload && payload.resource;
      if (!resource) throw new Error(tr('EasyICU did not return a data preview.', 'EasyICU 未返回可预览的数据包。'));
      resource.label = tr('Pre-analysis data readiness', '分析前数据准备检查');
      preview.open(resource, expectedProjectId, previewWorkflowContext());
      state.error = '';
    } catch (error) {
      if (projectId() !== expectedProjectId) return;
      state.error = errorText(error);
      render();
    } finally {
      if (button && button.isConnected) {
        button.disabled = false;
        button.textContent = original;
      }
    }
  }
  function editWorkflow() {
    const workflow = state.workflow || {};
    const code = String(workflow.next_action_code || '');
    if (code === 'failed_pipeline_execution_retry_available') {
      void startCurrentFormalPlanGeneration('failed_pipeline_requires_fresh_plan');
      return;
    }
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
    state.draft = CONFIRMATION.planChangeDraft();
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
  async function startEntry() {
    setShell('pi');
    const preview = MODULES.optional('preview');
    if (preview && preview.close) preview.close();
    if (state.session && state.messages.length === 0 && state.workflowReceipts.length === 0) {
      render();
      return;
    }
    state.sessionSelectionRevision += 1;
    state.session = null;
    state.messages = [];
    state.draft = '';
    state.editingMessageId = '';
    state.pendingEntryIntent = '';
    state.error = '';
    STUDY_WORKSPACE.removeReference();
    STUDY_WORKSPACE.removeSkill();
    STUDY_WORKSPACE.removeMethod();
    rememberSession('');
    const projectOwner = MODULES.require('project');
    if (projectOwner && projectOwner.syncLocation) projectOwner.syncLocation(projectId(), '');
    render();
    // Home reuses an empty draft this project already has instead of adding
    // another one to the rail; a session is only created when none is free.
    const emptyDraft = state.sessions.find(row => STUDY_WORKSPACE.isEmptyConversation(row));
    if (emptyDraft) { await openSession(emptyDraft.session_id); return; }
    await createSession();
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
    const expectedSessionId = state.session.session_id;
    const expectedProjectId = projectId();
    try {
      const payload = await api().rebindPiCopilotSession(
        expectedSessionId,
        { project_id: expectedProjectId },
      );
      if (projectId() !== expectedProjectId || !state.session || state.session.session_id !== expectedSessionId) return;
      state.session = payload.session; state.error = '';
      rememberSession(state.session && state.session.session_id);
      await loadWorkflow();
      const continued = await PLAN_ACTIONS.continueSystemOwnedPlanProgression({passive: true});
      if (!continued) render();
    } catch (error) {
      if (projectId() !== expectedProjectId || !state.session || state.session.session_id !== expectedSessionId) return;
      state.error = errorText(error); render();
    }
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

  function mount(host) {
    if (!host) return Promise.resolve();
    if (state.host === host) return state.startupPromise || Promise.resolve();
    closeSource(); closeChildSource(); state.host = host; state.conv = host.closest('.gd-conv'); state.shell = 'pi';
    // Restore the project/session first. That path resets the composer draft;
    // seeding before it completed made Open from Skill Hub silently disappear.
    if (state.conv) state.conv.classList.add('pi-active');
    wire(); document.addEventListener('click', dismissHeaderOverflow);
    const capabilitiesReady = api().loadCapabilities
      ? Promise.resolve(api().loadCapabilities()).catch(() => null)
      : Promise.resolve(null);
    state.startupPromise = Promise.resolve(loadStatus()).then(async () => {
      await capabilitiesReady;
      if (state.host) render(true);
      const hubIntent = STUDY_WORKSPACE.hasHubIntent(), previousSessionId = state.session && state.session.session_id;
      if (hubIntent && projectId() && state.session && !state.creating) await createSession();
      if (hubIntent && previousSessionId && state.session && state.session.session_id === previousSessionId) return;
      if (hubIntent && state.session && (!previousSessionId || state.session.session_id !== previousSessionId)) state.draft = '';
      try {
        const suggestion = window.sessionStorage.getItem('easyicu.skillHub.question');
        if (suggestion && state.session && (!previousSessionId || state.session.session_id !== previousSessionId)) {
          state.draft = suggestion;
          window.sessionStorage.removeItem('easyicu.skillHub.question');
          render();
        }
      } catch (_) {}
    }).finally(() => {
      state.startupPromise = null;
    });
    return state.startupPromise;
  }
  function unmount() {
    RUN_FILES.reset();
    document.removeEventListener('click', dismissHeaderOverflow);
    stopCodexPoll(); closeSource(); closeChildSource(); if (IDEA_SOURCE) IDEA_SOURCE.reset(); state.host = null; state.conv = null; state.busy = false; state.jobId = '';
  }
  window.addEventListener('easyicu:languagechange', handleLanguageChange);
  window.EasyICU.guidedPi.declare('shell', {
    mount,
    unmount,
    setShell,
    startEntry,
    bindProject,
    isActive,
    rebind,
    notifyExtractionHandoff,
    confirmDataSourceBinding,
    setProjectDiscoveryLoading,
    historyContext: () => ({ projectId: projectId(), title: state.project && state.project.title,
      studyId: (state.session && state.session.binding && state.session.binding.study_context_id)
        || (state.projectInitialization && state.projectInitialization.study_context_id) || '',
      runId: (state.session && state.session.binding && state.session.binding.run_id) || '',
      busy: state.busy || Boolean(state.childJobId) }),
  });
})();
