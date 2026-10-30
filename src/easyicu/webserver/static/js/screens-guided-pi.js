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
  const ACCESS = MODULES.require('access');
  const HEADER = MODULES.require('header');
  const REGENERATION = MODULES.require('regeneration');
  const STUDY_WORKSPACE = MODULES.require('studyWorkspace').create({ tr, esc, iconHtml }), TASKS = (MODULES.optional('taskTable') || { create: () => null }).create({ tr, esc, iconHtml, state: () => state, projectId, render, openSession, startEntry, title: navigationSessionTitle, sessions: () => state.sessions.filter(sessionMatchesUiLanguage), status: STUDY_WORKSPACE.sessionStatusLabel, railKey: STUDY_WORKSPACE.railKey });

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
    agentMode: 'research', accessMode: ACCESS.preferred(), pendingAuthorityRebind: false,
    demoMode: false, demoScrollTopPending: false, currentTurnResources: [],
    workflowReceipts: [], editingMessageId: '', sessionSelectionRevision: 0,
    pendingLanguageReload: false, pendingEntryIntent: '', regenerating: false, regeneration: null,
    openingQuestion: null, followUpDraft: null,
    startupPromise: null, projectPreparePromise: null, projectPrepareId: '',
  };
  const autoPreviewedWorkspaceVersions = new Set();

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  const { publicAssistantText } = MODULES.require('assistantText').create({ tr });
  function assistantTextHtml(value, resources, options = {}) {
    const renderer = MODULES.require('markdown');
    const drafts = MODULES.optional('researchDraft');
    let visible = value, preparing = false;
    if (options.streaming && drafts && typeof drafts.streaming === 'function') {
      ({ text: visible, preparing } = drafts.streaming(value));
    } else if (options.liftDraft !== false && drafts && typeof drafts.strip === 'function') visible = drafts.strip(value);
    // Only attached message resources can turn a filename into a controlled
    // preview action with the Host's file coordinates and checked digest.
    const html = renderer && typeof renderer.render === 'function'
      ? renderer.render(visible, { resources })
      : esc(visible).replace(/\n/g, '<br>');
    return html + (preparing ? `<p class="gpi-draft-preparing" role="status">${tr('Organizing the discussion draft…', '正在整理讨论草案…')}</p>` : '');
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
    const config = runtime.configuration || {}, hosted = MODULES.optional('hosted');
    // A hosted instance has only the administrator's API connection.
    if ((hosted && hosted.active()) || (
      config.connection_verified === true
      && config.credential_present === true
      && (config.api_transport || runtime.api_transport) === 'openai-completions'
    )) {
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
  function accessModeLabel(mode) { return ACCESS.label(mode); }
  function turnGrants() { return ACCESS.grants(state.accessMode, agentMode()); }
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
  const { errorText, modelErrorText, providerPreset, option, runFailureText, studyFamilyGapText, manuscriptNotPassedText } = ERROR_TEXT;
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
    api, context: runFilesContext, changed: () => render(true), resourceButton, manuscriptNotPassedText,
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
  // 遇到问题: a reference the administrator can look up (ids only).
  const HELP = MODULES.require('help').create({ tr, esc, context: () => ({ projectId: projectId(), sessionId: state.session && state.session.session_id, runId: state.latestRun && state.latestRun.run_id, jobId: state.childJobId || state.jobId }) });
  const CONNECT = MODULES.require('connect').create({ state, tr, esc, render, connectionReady, connectionConfigured, finishProviderSetup,
    body: () => MODULES.require('provider').renderConnectionBody(setupOptions()) });
  const RUN_OUTCOME = MODULES.require('runOutcome').create({
    tr, esc, iconHtml, resourceButton, api, projectId, host: () => state.host, session: () => state.session,
    canPreview: () => Boolean(state.session) && !state.busy && !state.childJobId && !sessionIsStale(),
    preview: () => MODULES.optional('preview'),
    workflowContext: previewWorkflowContext,
    errorText, manuscriptNotPassedText,
    recordHostAction,
    onError: value => { state.error = value; render(); },
  });
  const ACTIVITY = MODULES.require('activity').create({
    tr, esc, iconHtml, resourceName, resourceKey, resourceButton, runFailureText,
    mergeResources: RESOURCE_OWNER.merge,
    agentMode,
    publicText: publicAssistantText,
  });
  const CONFIRMATION = MODULES.require('confirmation').create({
    tr, esc, iconHtml, resourceButton, sessionIsStale, runFailureText, studyFamilyGapText,
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
    tr, activity: ACTIVITY, upsertActivityStep, timeMs, resourceKey, modelErrorText, runFailureText,
    mergeResources: RESOURCE_OWNER.merge,
    activityHasCompletedAction, activityHasSideEffectAttempt,
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
    taskScope: () => agentMode() === 'workspace' ? 'workspace'
      : MODULES.optional('taskScope')?.current(state).kind || '',
    projectId: () => projectId(),
    displayProjectTitle: (...args) => displayProjectTitle(...args),
    demoMode: () => state.demoMode,
    project: () => state.project,
    messages: () => state.messages,
    shell: () => state.shell,
    workflow: () => state.workflow,
    latestRun: () => state.latestRun,
    workflowError: () => state.workflowError,
    projectNotStarted: () => Boolean(state.projectInitialization && state.projectInitialization.required),
    resultsHtml: (query, options) => RUN_OUTCOME.renderShelf(state.latestRun, state.workflow, query, options),
    resourceButton,
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
    runFailureText, manuscriptNotPassedText,
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
    workflow: () => state.workflow,
    rebind: () => rebind(),
    researchQuestion: () => conversationResearchQuestion(),
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
    projectId: () => projectId(),
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
    setPlanStarting: value => { state.planStarting = value || null; }, planStarting: () => state.planStarting,
    setError: value => { state.error = String(value || ''); },
    // The host compiler may refuse the automatic configuration (for example
    // an all-stay analysis without source-owned patient grouping). The
    // confirmation and cohort owners read this to offer the researcher's
    // decision instead of leaving a bare error banner.
    setPlanConfigurationError: value => { state.planConfigurationError = String(value || ''); },
    setDraft: value => { state.draft = String(value || ''); state.followUpDraft = null; },
    appendMessage: value => { state.messages.push(value); },
    truncateMessagesAt: id => {
      const at = state.messages.findIndex(item => String((item && item.id) || '') === id);
      if (at >= 0) state.messages.splice(at);
    },
  });
  const confirmWorkflowAction = () => PLAN_ACTIONS.confirmWorkflow(workflowConfirmation());
  const authorizeCandidatePlan = {
    planContext: () => {
      const session = state.session || {};
      const workflow = state.workflow || {};
      const user = state.messages.slice().reverse().find(row => row && row.role === 'user');
      return {
        projectId: projectId(), sessionId: String(session.session_id || ''),
        studyId: String((session.binding || {}).study_context_id || ''),
        code: String(workflow.next_action_code || ''),
        decision: (workflow.host_decisions || {}).plan_transition || null,
        userMessage: String((user && user.text) || ''),
      };
    },
    loadStudy: id => api().loadStudyContext(id),
    confirmPlan: message => window.confirm(message),
    startPlan: (code, options) => PLAN_ACTIONS.startFormalPlanGeneration(code, options),
    planError: message => { state.error = String(message || ''); render(); },
  };
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
    continueReply: continueReplyExplanation,
    retryUnstarted: continueReplyExplanation,
    resubmitHostGenerated: PLAN_ACTIONS.resubmitHostGenerated,
    host: () => state.host,
  });
  const MESSAGE_QUEUE = MODULES.require('messageQueue').create({
    state, api, projectId, render, refreshSession, watchJob, errorText,
  });
  // Quick-answer cards (owner: screens-guided-pi-quick-answers.js).
  const quickAnswerOwner = MODULES.optional('quickAnswers');
  const QUICK_ANSWERS = quickAnswerOwner ? quickAnswerOwner.create({
    tr, esc, act: PLAN_ACTIONS.quickAnswerAction,
    disabled: () => Boolean(state.busy || state.childJobId || sessionIsStale()),
  }) : null;
  // Multi-database batch cards (owner: screens-guided-pi-multi-database.js).
  const multiDatabaseOwner = MODULES.optional('multiDatabase');
  const MULTI_DATABASE = multiDatabaseOwner ? multiDatabaseOwner.create({
    tr, esc, api, session: () => state.session, projectId, errorText, repaint: () => render(),
    disabled: () => Boolean(state.busy || sessionIsStale()),
  }) : null;
  const EVENTS = MODULES.require('events').create({
    state, RESOURCE_OWNER, RUN_FILES, MESSAGE_ACTIONS, STARTERS, IDEA_SOURCE, COHORT_ELIGIBILITY,
    DATA_CONSENT, RUN_OUTCOME, STUDY_WORKSPACE, ASIDE, render, projectId, previewWorkflowContext,
    openSession, closeDemo, openDemo, switchMode, loadCodexResearchStatus,
    openAuthorizationPopup, startCodexLogin, cancelCodexLogin, logoutCodex,
    loadCodexModels, tr, apiResearchReady, finishProviderSetup, loadStatus,
    setShell, openStudySetupInConversation, createSession, startEntry, loadWorkflow,
    previewApprovedPlanDataPackage, confirmWorkflowAction,
    retryFailedExecution,
    rejectWorkflowAction, editWorkflow, confirmCohortEligibility,
    confirmPlanDecision,
    authorizeDataSource, sendText, continueAfterDataSourceConfirmation,
    governedNextChoiceGrants, sendMessage, stopMessage, stopChildJob, rebind,
    togglePresentationPin, configureProvider, rememberSession, recordHostAction,
    HOST_JOBS, EFFORT_MENU, api, CONNECT, PROVIDER_CONTROL, HELP, MESSAGE_QUEUE, QUICK_ANSWERS, MULTI_DATABASE,
    ACCESS, turnGrants, authorizeCandidatePlan,
  });
  const dismissHeaderOverflow = EVENTS.dismissHeaderOverflow;
  const wire = EVENTS.wire;

  function activeActivity(includeTerminal) {
    const belongsToJob = row => !state.jobId || String(row.messageJobId || '') === String(state.jobId);
    if (state.regenerating && state.regeneration && belongsToJob(state.regeneration.activity)
        && (includeTerminal || state.regeneration.activity.status === 'running')) {
      return state.regeneration.activity;
    }
    return state.messages.slice().reverse().find(row => row.role === 'activity' && !row.childJobId
      && belongsToJob(row) && (includeTerminal || row.status === 'running'));
  }
  function ensureActivity(at, options) {
    let row = options && options.newAttempt ? null : activeActivity();
    if (!row) {
      const startedAt = timeMs(at);
      row = { id: 'activity-' + (state.jobId || startedAt), messageJobId: String(state.jobId || ''),
        role: 'activity', status: 'running', startedAt, steps: [], expanded: true };
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
    const activity = activeActivity(true);
    if (!activity) return;
    const endedAt = Math.max(Number(activity.startedAt || 0), Number(activity.endedAt || 0), timeMs(at));
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

  function setupOptions() {
    const runtime = state.runtime || {};
    const config = runtime.configuration || {};
    const blockers = runtime.blockers || [];
    return {
      state, runtime, config, blockers, tr, esc, option,
      providerPreset, runtimeReady: runtimeReady(), apiResearchReady: apiResearchReady(),
      connectionConfigured: connectionConfigured(), connectionReady: connectionReady(),
      staticPreview: isStaticPreview(),
    };
  }
  function setupPanel() {
    const owner = MODULES.require('provider');
    return owner && typeof owner.renderSetup === 'function' ? owner.renderSetup(setupOptions()) : '';
  }

  function providerBindingSummary() {
    const owner = MODULES.require('provider');
    return owner && typeof owner.renderBindingSummary === 'function'
      ? owner.renderBindingSummary({ state, tr, esc, runtimeReady: runtimeReady(), apiResearchReady: apiResearchReady(), connectionReady: connectionReady() })
      : '';
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
    effortMenu: EFFORT_MENU, workflowConfirmation, runFailureText, help: HELP,
    sessionMatchesUiLanguage, displaySessionTitle, sessionTraceLabel, providerBindingSummary,
    quickAnswers: QUICK_ANSWERS, multiDatabase: MULTI_DATABASE, dataBinding: DATA_BINDING,
  });
  const { messageHtml, workflowHtml, sessionPanel, demoPanel,
    activatePanel, projectRequiredPanel, restoringPanel } = SESSION_VIEW;
  function openDemo() {
    const demo = MODULES.optional('demo');
    if (!demo || typeof demo.messages !== 'function') return;
    const preview = MODULES.optional('preview');
    if (preview && preview.close) preview.close();
    state.demoMode = true;
    state.demoScrollTopPending = true;
    state.error = '';
    // The walkthrough replays from the question; it opens no file up front.
    const walkthrough = MODULES.optional('demoWalkthrough');
    if (walkthrough) walkthrough.start();
    setShell('pi');
    const primary = !walkthrough && typeof demo.primaryDocument === 'function' ? demo.primaryDocument() : null;
    if (primary && preview && preview.open) {
      preview.open(primary, projectId());
    }
  }
  function closeDemo() {
    const walkthrough = MODULES.optional('demoWalkthrough');
    if (walkthrough) walkthrough.stop();
    state.demoMode = false;
    state.demoScrollTopPending = false;
    const preview = MODULES.optional('preview');
    if (preview && preview.close) preview.close();
    render();
  }
  function render(preserveScroll) {
    if (!state.host) return;
    EVENTS.captureComposer(); STUDY_WORKSPACE.capture(state.host); PROVIDER_CONTROL.captureForm(state.host);
    const previousLog = preserveScroll && state.host.querySelector('[data-gpi-log]');
    const previousTop = previousLog ? previousLog.scrollTop : null;
    const previousConnection = state.host.querySelector('[data-gpi-connection-page]');
    const previousConnectionTop = previousConnection ? previousConnection.scrollTop : null;
    if (!state.demoMode) RUN_FILES.sync();
    const restoring = !state.session && (state.loading || state.projectLoading || state.projectDiscoveryLoading);
    const setupFocused = !restoring && state.shell !== 'legacy'
      && !state.demoMode
      && (CONNECT.pageRequired() || state.projectIssue === 'pi_project_study_context_missing');
    const emptySessionFocused = !restoring && !setupFocused && state.shell !== 'legacy'
      && !state.demoMode && Boolean(state.session) && agentMode() !== 'workspace'
      && !state.taskDraft
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
    const markup = restoring
      ? restoringPanel()
      : state.shell === 'legacy'
      ? statusBanner()
      : (state.demoMode ? demoPanel() : (state.projectIssue === 'pi_project_study_context_missing'
        ? activatePanel()
        : (CONNECT.pageRequired() ? setupPanel() : (!projectId() ? projectRequiredPanel() : (TASKS && TASKS.shown() ? TASKS.panel() : (state.session ? sessionPanel() : activatePanel())))))) + (setupFocused ? '' : CONNECT.dialogHtml());
    // A menu the researcher has open (layout, more, composer) survives the
    // repaint that every progress event of a running job triggers.
    const menus = window.EU_POPOVER_MENUS;
    if (menus && menus.keepOpen) menus.keepOpen(state.host, () => { state.host.innerHTML = markup; });
    else state.host.innerHTML = markup;
    EVENTS.restoreComposer(); PROVIDER_CONTROL.restoreForm(state.host);
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
    STUDY_WORKSPACE.syncNavigation({ projectId: projectId(), sessions: state.sessions.filter(sessionMatchesUiLanguage), otherLanguageSessions: state.sessions.filter(row => !sessionMatchesUiLanguage(row)).length, switchLanguage: () => window.setLang && window.setLang(uiLanguage() === 'zh' ? 'en' : 'zh'),
      selectedId: state.session && state.session.session_id, loading: restoring, disabled: state.busy || Boolean(state.childJobId) || state.projectLoading,
      visible: state.shell !== 'legacy', title: navigationSessionTitle, status: STUDY_WORKSPACE.sessionStatusLabel,
      time: STUDY_WORKSPACE.compactSessionTime, open: id => (TASKS ? TASKS.openSession(id) : openSession(id)), create: () => (TASKS ? TASKS.startEntry() : startEntry()), search: openSearch, openAll: TASKS ? TASKS.show : undefined,
      resources: RUN_OUTCOME.collection(state.latestRun, state.workflow), openResource: EVENTS.openResource,
      materialsInteractive: !emptySessionFocused, rename: renameSession, remove: removeSession,
      mark: id => { const attention = MODULES.optional('attention'); return attention ? attention.conversationMarkHtml(projectId(), id) : ''; } });
    const attention = MODULES.optional('attention');
    if (attention) void attention.seen();
    DATA_BINDING.resumeWaiting();
    requestAnimationFrame(() => {
      STUDY_WORKSPACE.restoreMaterials(state.host); CONNECT.focusDialog(state.host);
      const composer = state.focusComposerUntil > Date.now() && state.host && state.host.querySelector('[data-gpi-input]');
      if (composer && !composer.disabled && document.activeElement !== composer) {
        composer.focus();
        composer.setSelectionRange(composer.value.length, composer.value.length);
      }
      // Opening a deep-linked artifact while the project/session is still
      // restoring lets openSession() immediately close it and erase the URL.
      // Wait until the selected conversation is authoritative, then restore
      // the digest-pinned artifact/evidence view once the normal shell exists.
      if (!restoring && state.session && preview && preview.restoreFromLocation) {
        preview.restoreFromLocation(projectId(), previewWorkflowContext());
      }
      const log = state.host && state.host.querySelector('[data-gpi-log]');
      if (log) {
        // A conversation opens at its latest turn; the start page (examples,
        // no messages yet) opens at its top.
        log.scrollTop = previousTop !== null ? previousTop
          : state.demoScrollTopPending || log.classList.contains('gpi-log-start') ? 0 : log.scrollHeight;
        state.demoScrollTopPending = false;
      }
      const connectionPage = state.host && state.host.querySelector('[data-gpi-connection-page]');
      if (connectionPage) connectionPage.scrollTop = previousConnectionTop === null ? 0 : previousConnectionTop;
      ACTIVITY.syncLiveClock(state.host, state.busy || Boolean(state.childJobId));
      MESSAGE_QUEUE.sync();
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
      if (!connectionReady() && !state.connectionDeferred) state.showSetup = true;
    } catch (error) {
      state.runtime = { status: 'unavailable', blockers: ['status_request_failed'] };
      state.error = errorText(error);
    } finally {
      state.loading = false;
      state.projectLoading = false;
      render();
    }
  }

  const SESSION_NAVIGATION = MODULES.require('sessionNavigation').create({
    state, api, projectId, connectionReady, closeSourceView, CONNECT,
    displayProjectTitle, tr, EFFORT_MENU, loadWorkflow, transcriptMessages,
    STUDY_WORKSPACE, hydrateProjectedJob, rememberSession, errorText, render,
    navigationSessionTitle, sessionMatchesUiLanguage, rememberedSession,
    openSession, uiLanguage, closeSource, closeChildSource, setShell,
  });
  async function createSession() { return SESSION_NAVIGATION.createSession(); }
  async function renameSession(row) { return SESSION_NAVIGATION.renameSession(row); }
  async function removeSession(row) { return SESSION_NAVIGATION.removeSession(row); }

  // Search across projects; a hit in this project opens through openSession.
  function openSearch() {
    const search = MODULES.optional('globalSearch');
    if (!search) return;
    search.open({
      projectId,
      projectTitle: () => (state.project && state.project.title) || '',
      openSession: id => openSession(id),
      canOpen: () => !(state.busy || state.childJobId || state.projectLoading),
    });
  }

  async function openSession(sessionId, selectionRevision, refreshWorkflow) {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    if (state.followUpDraft) { state.draft = ''; state.followUpDraft = null; }
    closeSourceView();
    const preview = MODULES.optional('preview');
    const previewParams = typeof window !== 'undefined' && window.location
      ? new URLSearchParams(window.location.search || '') : null;
    const preservePreviewLocation = Boolean(previewParams
      && previewParams.get('pi_session') === String(sessionId || '')
      && ['artifact', 'evidence'].includes(previewParams.get('pi_view')));
    // Entering real-data mode remounts the shell and restores this same
    // conversation. Its open source picker still belongs to it; close the
    // preview only when the researcher actually selects another conversation.
    const sameSession = state.session && state.session.session_id === sessionId;
    if (!sameSession) state.taskDraft = '';
    if (!sameSession && preview && preview.close) preview.close({ preserveLocation: preservePreviewLocation });
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
      TRANSCRIPT.markSupersededPlanAttempts(state.messages, state.session);
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
      if (expectedProjectId !== projectId() || expectedSelectionRevision !== state.sessionSelectionRevision) return;
      await PLAN_ACTIONS.continueSystemOwnedPlanProgression({passive: true});
    } catch (error) {
      if (expectedProjectId !== projectId() || expectedSelectionRevision !== state.sessionSelectionRevision) return;
      rememberSession(''); state.error = errorText(error); render();
    }
  }

  function assistantRow() {
    if (state.regenerating && state.regeneration) return state.regeneration.message;
    let row = state.messages[state.messages.length - 1];
    if (!row || row.role !== 'assistant' || row.complete
        || String(row.messageJobId || '') !== String(state.jobId || '')) {
      row = {
        id: 'live-' + Date.now(), messageJobId: String(state.jobId || ''), role: 'assistant', text: '', complete: false,
        resources: state.currentTurnResources.slice(0, 24),
      };
      state.messages.push(row);
    }
    return row;
  }
  function addAssistantResources(resources) {
    state.currentTurnResources = RESOURCE_OWNER.merge(state.currentTurnResources, resources).slice(0, 24);
    const row = state.regenerating && state.regeneration
      ? state.regeneration.message
      : state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
    if (row) row.resources = state.currentTurnResources.slice();
    // Only newly arriving, Host-checked workspace HTML opens automatically.
    // A history refresh or a repeated tool snapshot cannot reopen a closed tab.
    if (agentMode() !== 'workspace') return;
    const incoming = new Set((Array.isArray(resources) ? resources : []).filter(Boolean).map(item => item.file));
    const checked = state.currentTurnResources.slice().reverse().find(item =>
      incoming.has(item.file) && ['file', 'webpage'].includes(item.kind)
      && /\.html?$/i.test(item.file || '') && /^[a-f0-9]{64}$/.test(item.checked_sha256 || ''));
    if (!checked) return;
    const key = JSON.stringify([projectId(), checked.file, checked.checked_sha256]);
    if (autoPreviewedWorkspaceVersions.has(key)) return;
    const preview = MODULES.optional('preview');
    if (!preview || typeof preview.open !== 'function') return;
    autoPreviewedWorkspaceVersions.add(key);
    if (autoPreviewedWorkspaceVersions.size > 128) autoPreviewedWorkspaceVersions.delete(autoPreviewedWorkspaceVersions.values().next().value);
    preview.open(checked, projectId(), previewWorkflowContext());
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
  function activityHasSideEffectAttempt(activity) {
    return Boolean(activity && Array.isArray(activity.steps) && activity.steps.some(step => (
      step && step.kind === 'tool' && /^(?:easyicu_(?:write|edit)_project_file|easyicu_(?:update_study_context|run|start_extraction|prepare_demo_source|repair_report|request_replan|cancel|accept_idea_handoff|mine_ideas))$/.test(String(step.toolName || ''))
    )));
  }
  async function switchMode(mode, options = {}) {
    const next = mode === 'research' ? 'research' : 'workspace';
    if (state.busy || (next === agentMode() && !options.newSession)) return;
    if (options.draft !== undefined) state.draft = String(options.draft);
    state.taskDraft = '';
    if (options.newSession) {
      state.pendingEntryIntent = '';
      state.followUpDraft = null;
      if (IDEA_SOURCE) IDEA_SOURCE.reset();
    }
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
    let existingSessionId = options.newSession ? '' : replayOwner && typeof replayOwner.preferredSessionId === 'function'
      ? replayOwner.preferredSessionId(state.sessions, '', next, uiLanguage())
      : String(state.sessions.find(row => (row.agent_mode || 'research') === next && sessionMatchesUiLanguage(row))?.session_id || '');
    if (!options.newSession && !existingSessionId && projectId()) {
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
    activityHasCompletedAction, activityHasSideEffectAttempt, modelErrorText, activity: ACTIVITY,
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
      const replyContinuation = refreshed.reply_continuation;
      if (preserveTimeline && replyContinuation && replyContinuation.source_error_entry_id) {
        const savedFailure = (refreshed.transcript || []).find(row => row.entry_id === replyContinuation.source_error_entry_id);
        const liveFailure = state.messages.slice().reverse().find(row => row.role === 'assistant');
        const sameJob = liveFailure && liveFailure.messageJobId
          && liveFailure.messageJobId === refreshed.last_message_job_id;
        if (savedFailure && liveFailure && (sameJob || liveFailure.errorCode === savedFailure.error_code)
            && !liveFailure.entryId) {
          liveFailure.entryId = replyContinuation.source_error_entry_id;
          liveFailure.hasCompletedAction = Boolean(liveFailure.hasCompletedAction
            || replyContinuation.mutation_state === 'completed');
        }
      }
      state.sessions = [state.session].concat(
        state.sessions.filter(row => row.session_id !== state.session.session_id)
      );
      if (!preserveTimeline) state.messages = transcriptMessages(state.session);
      (Array.isArray(state.session.archived_child_jobs) ? state.session.archived_child_jobs : []).forEach(hydrateProjectedJob);
      TRANSCRIPT.markSupersededPlanAttempts(state.messages, state.session);
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

  async function loadProjectSessions(refreshWorkflow) { return SESSION_NAVIGATION.loadProjectSessions(refreshWorkflow); }
  async function reloadSessionsForLanguage() { return SESSION_NAVIGATION.reloadSessionsForLanguage(); }
  function handleLanguageChange() { return SESSION_NAVIGATION.handleLanguageChange(); }

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
        // The project's run record and the host's decision coordinates travel
        // beside the snapshot; owners read them from the same workflow object.
        if (state.workflow) state.workflow.runs = Array.isArray(payload && payload.runs) ? payload.runs : [];
        if (state.workflow) Object.assign(state.workflow, { host_decisions: (payload && payload.host_decisions) || {}, starting: (payload && payload.starting) || null, target_trial_card: (payload && payload.target_trial_card) || null, raw_preparation: (payload && payload.raw_preparation) || null, raw_preparation_candidate_run_id: (payload && payload.raw_preparation_candidate_run_id) || null });
        reconcileDurableWrapupActivity();
        hydrateProjectedJob(payload && payload.active_job);
        const activeJob = payload && payload.active_job;
        if (activeJob && activeJob.present && activeJob.status === 'running' && activeJob.job_id) {
          const kind = String(activeJob.kind || '');
          const code = activeJob.report_only === true
            ? 'easyicu_report_repair_submitted'
            : kind === 'target-trial-compile' ? 'easyicu_target_trial_compile_submitted'
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
    state.taskDraft = '';
    state.followUpDraft = null;
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
    state.creating = false;
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
          openHostedConversation();
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
    state.jobId = jobId;
    const pendingActivity = state.regenerating && state.regeneration ? state.regeneration.activity
      : state.messages.slice().reverse().find(row => row.role === 'activity' && !row.childJobId
        && row.status === 'running' && (!row.messageJobId || String(row.messageJobId) === String(jobId)));
    if (pendingActivity) { pendingActivity.messageJobId = String(jobId); pendingActivity.steps = []; }
    if (state.regenerating && state.regeneration) state.regeneration.message.messageJobId = String(jobId);
    const expectedSessionId = state.session && state.session.session_id;
    const expectedProjectId = projectId();
    const selectionRevision = state.sessionSelectionRevision;
    const isCurrentAttempt = () => String(state.jobId) === String(jobId)
      && state.session && state.session.session_id === expectedSessionId
      && projectId() === expectedProjectId && state.sessionSelectionRevision === selectionRevision;
    state.source = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    const watchedSource = state.source;
    let lastSequence = -1;
    state.source.onmessage = async event => {
      if (state.source !== watchedSource || !isCurrentAttempt()) return;
      let row = null; try { row = JSON.parse(event.data); } catch (e) { return; }
      if (Number.isSafeInteger(row.seq)) {
        if (row.seq <= lastSequence) return;
        lastSequence = row.seq;
      }
      if (row.type === 'pi_event') handlePiEvent(row.event, jobId);
      if (row.type === 'end') {
        closeSource(); state.busy = false; state.continuingReplyEntryId = '';
        const replacedBranch = state.regenerating;
        const completedAction = activityHasCompletedAction(activeActivity(true));
        const wrapupTimedOut = row.status === 'failed'
          && /^pi_gateway_timeout(?:\s*:|$)/.test(String(row.error || ''));
        if (row.status === 'failed') {
          finishActivity('error', row.timestamp, 'failed');
          const failure = state.regenerating && state.regeneration ? state.regeneration.message
            : state.messages.slice().reverse().find(item => item.role === 'assistant'
              && String(item.messageJobId || '') === String(jobId) && (!item.complete || item.errorCode)) || assistantRow();
          failure.complete = true;
          const failureCode = String(row.error || '').split(/[\s:]/, 1)[0];
          failure.errorCode = /^(?:pi_|easyicu_|research_)[A-Za-z0-9_.-]{1,150}$/.test(failureCode)
            ? failureCode : 'pi_message_failed';
          failure.hasCompletedAction = completedAction;
          failure.hasSideEffectAttempt = activityHasSideEffectAttempt(activeActivity(true));
          if (!failure.text) failure.text = wrapupTimedOut
            ? tr('This attempt timed out; completed records are preserved.', '本次尝试等待超时，已完成记录已保留。')
            : modelErrorText(failure.errorCode, completedAction);
          state.error = wrapupTimedOut
            ? tr('This wait timed out. Completed records are preserved; check the recovery action or specific blocker below the failed reply.', '本次等待已超时。已完成的记录已保留；请查看该条失败回复下的恢复操作或具体阻断原因。')
            : /^pi_model_/.test(String(row.error || ''))
            ? modelErrorText(row.error, completedAction)
            : (row.error ? ERROR_TEXT.jobErrorText(row.error) : tr('Copilot message failed.', '研究助手消息失败。'));
        } else if (row.status === 'cancelled') {
          finishActivity('cancelled', row.timestamp, 'cancelled');
          state.error = tr('Copilot message stopped.', '研究助手消息已停止。');
        } else {
          finishActivity('complete', row.timestamp, 'settled');
        }
        if (replacedBranch && state.regeneration) {
          const finished = state.regeneration;
          state.messages = REGENERATION.visibleRows(state.messages, finished)
            .map(message => REGENERATION.project(message, finished));
          state.regenerating = false;
          state.regeneration = null;
        }
        await refreshSession(!replacedBranch);
        if (!isCurrentAttempt()) return;
        await MESSAGE_QUEUE.reconcile();
        if (!isCurrentAttempt()) return;
        if (state.busy) return;
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
        if (!isCurrentAttempt()) return;
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
    let checkingConnection = false;
    state.source.onerror = async () => {
      if (state.source !== watchedSource || !isCurrentAttempt()) return;
      if (!state.busy) { closeSource(); return; }
      if (checkingConnection) return;
      checkingConnection = true;
      try {
        await api().loadJobSnapshot(jobId);
      } catch (error) {
        if (state.source !== watchedSource || !isCurrentAttempt() || Number(error.status) !== 404) return;
        closeSource();
        state.busy = false;
        state.regenerating = false;
        state.regeneration = null;
        finishActivity('error', null, 'failed');
        state.error = tr('This task is no longer available. You can resend the message.', '当前任务已不可用，可以重新发送消息。');
        render();
      } finally {
        checkingConnection = false;
      }
    };
  }
  async function sendText(text, grantsOverride, turnIntent, visibleUserMessage = true, messageOrigin = 'host_prompt', openingQuestion = '') {
    if (!state.session || state.projectLoading || state.busy || state.childJobId || sessionIsStale()) return;
    if (!CONNECT.ask('send_message', () => sendText(text, grantsOverride, turnIntent, visibleUserMessage, messageOrigin, openingQuestion))) return;
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
    state.jobId = '';
    state.currentTurnResources = [];
    if (visibleUserMessage) state.messages.push({ id: 'user-' + submittedAt, role: 'user', text, complete: true, messageOrigin: messageOrigin || 'unknown' });
    if (visibleUserMessage && openingQuestion) state.openingQuestion = { sessionId: state.session.session_id, text: openingQuestion };
    const activity = ensureActivity(new Date(submittedAt).toISOString(), { newAttempt: true });
    upsertActivityStep(activity, { id: 'submitted', kind: 'submitted', status: 'complete', at: submittedAt });
    if (visibleUserMessage) { state.draft = ''; state.followUpDraft = null; STUDY_WORKSPACE.consume(expectedProjectId, expectedSessionId); STUDY_WORKSPACE.consumeSkill(expectedProjectId, state.session); }
    state.busy = true; state.error = ''; render();
    try {
      const payload = await api().sendPiCopilotMessage(state.session.session_id, {
        project_id: projectId(), message: text, allowed_actions: grants,
        ...(turnIntent ? { turn_intent: turnIntent } : {}),
        ...(ideaSource ? { idea_source: ideaSource } : {}),
        message_origin: messageOrigin,
      });
      if (!isCurrent()) return;
      if (ideaSource && IDEA_SOURCE) IDEA_SOURCE.consume();
      // The host consumed the attached-Skill draft with this message.
      if (visibleUserMessage && state.session && Array.isArray(state.session.draft_attached_skills)) state.session.draft_attached_skills = [];
      state.pendingEntryIntent = '';
      state.jobId = payload.job_id; watchJob(payload.job_id);
    } catch (error) {
      if (!isCurrent()) return;
      state.busy = false; finishActivity('error', null, 'failed');
      state.error = errorText(error);
      render();
    }
  }
  async function continueReplyExplanation(sourceErrorEntryId) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const recovery = state.session.reply_continuation;
    if (!recovery || !recovery.available || recovery.source_error_entry_id !== sourceErrorEntryId) return;
    const recoveryKind = recovery.recovery_kind || 'continue_explanation';
    if (!['retry_unstarted', 'continue_explanation'].includes(recoveryKind)) return;
    const sessionId = state.session.session_id;
    const expectedProjectId = projectId();
    const revision = state.sessionSelectionRevision;
    const isCurrent = () => state.session && state.session.session_id === sessionId
      && projectId() === expectedProjectId && state.sessionSelectionRevision === revision;
    state.busy = true; state.error = ''; state.currentTurnResources = [];
    state.continuingReplyEntryId = sourceErrorEntryId;
    state.recoveringReplyKind = recoveryKind;
    render();
    try {
      const recover = recoveryKind === 'retry_unstarted' ? api().retryPiCopilotWorkspaceReply : api().continuePiCopilotReply;
      const payload = await recover(sessionId, {
        project_id: expectedProjectId, source_error_entry_id: sourceErrorEntryId,
      });
      if (!isCurrent()) return;
      state.jobId = payload.job_id;
      watchJob(payload.job_id);
    } catch (error) {
      if (!isCurrent()) return;
      state.busy = false;
      state.continuingReplyEntryId = '';
      state.error = errorText(error);
      await refreshSession(false);
      render();
    }
  }
  async function regenerateMessage(userEntryId, text, regenerationIntent, targetMessageId) {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return;
    const entryId = String(userEntryId || '').trim();
    text = String(text || '').trim();
    if (!entryId || !text) return;
    const original = state.messages.find(row => row && row.role === 'user' && String(row.entryId || '') === entryId);
    const messageOrigin = regenerationIntent === 'user_edited_message' && original && original.text !== text
      ? 'typed' : (original && original.messageOrigin || 'unknown');
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
        message_origin: messageOrigin,
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
  // The researcher's opening question: the first message they typed in this
  // conversation on this page, sent as written. A starter card, a model
  // option, or a message carrying a reference or Skill is not one, and
  // messages restored from history do not say how they were written, so
  // none of them is used; the conversation then asks for the question once.
  function conversationResearchQuestion() {
    const opening = state.openingQuestion;
    const text = opening && state.session && opening.sessionId === state.session.session_id
      ? String(opening.text || '').trim() : '';
    return text.length >= 12 ? text : '';
  }
  async function continueAfterDataSourceConfirmation() {
    if (!state.session || state.busy || state.childJobId || sessionIsStale()) return false;
    // The carry waits on the study store. A conversation opened meanwhile,
    // this one reopened included, does not receive this continue message.
    const expectedSessionId = state.session.session_id;
    const expectedProjectId = projectId();
    const selectionRevision = state.sessionSelectionRevision;
    const isCurrent = () => state.session && state.session.session_id === expectedSessionId
      && projectId() === expectedProjectId && state.sessionSelectionRevision === selectionRevision;
    if (DATA_BINDING && typeof DATA_BINDING.carryQuestionIntoSetup === 'function') {
      await DATA_BINDING.carryQuestionIntoSetup();
      if (!isCurrent() || state.busy || state.childJobId || sessionIsStale()) return false;
    }
    await sendText(
      tr(
        'Continue this conversation after EasyICU confirmed the selected data source.',
        'EasyICU 已确认所选数据来源，请在当前对话中继续。',
      ),
      [],
      'advance_after_data_source_confirmation',
      false,
      'host_prompt',
    );
    return true;
  }
  async function sendMessage() {
    if (!state.session || state.projectLoading || state.childJobId || sessionIsStale()) return;
    const input = state.host.querySelector('[data-gpi-input]');
    const text = String((input && input.value) || state.draft || '').trim();
    if (!text) return;
    const followUp = state.followUpDraft;
    if (followUp) {
      // Editing a suggested question does not grant research approval or turn
      // it into the user's opening question. The typed-only queue cannot carry
      // its origin, so leave this draft for an explicit send after work ends.
      if (state.busy || followUp.sessionId !== state.session.session_id
        || followUp.projectId !== projectId() || followUp.selectionRevision !== state.sessionSelectionRevision) return;
      const prompt = STUDY_WORKSPACE.decorateMessage(
        STUDY_WORKSPACE.decorateSkillMessage(text, projectId(), state.session), projectId(), state.session.session_id,
      );
      await sendText(prompt, [], undefined, true, followUp.origin);
      return;
    }
    if (state.busy) {
      try { await MESSAGE_QUEUE.send(text); }
      catch (error) { state.error = errorText(error); render(); }
      return;
    }
    const referencing = STUDY_WORKSPACE.hasReference(projectId(), state.session.session_id);
    const taskScope = MODULES.optional('taskScope');
    if (agentMode() !== 'workspace' && !referencing && taskScope && taskScope.kind(text)) {
      state.taskDraft = text; state.draft = text;
      render(); return;
    }
    state.taskDraft = '';
    if (!referencing && await PLAN_ACTIONS.continueUserRequestedSystemProgression(text)) return;
    const intent = referencing ? undefined : state.pendingEntryIntent
      || (IDEA_SOURCE && IDEA_SOURCE.suggestsIdeaMining(text) ? 'idea_mining_entry' : undefined);
    const decorated = STUDY_WORKSPACE.decorateMessage(
      STUDY_WORKSPACE.decorateSkillMessage(text, projectId(), state.session), projectId(), state.session.session_id,
    );
    // The first message, typed and sent as written (no starter intent, no
    // reference or Skill around it), is the researcher's opening question.
    const opening = !intent && decorated === text && !state.messages.some(row => row && row.role === 'user') ? text : '';
    // Reference/Skill decoration mixes host-authored text with the user's
    // words. Until the transport separates them, this turn proves no wording.
    await sendText(decorated, undefined, intent, true, decorated === text ? 'typed' : 'host_prompt', opening);
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
    state.followUpDraft = null;
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
    await sendText(prompt, ['configure'], undefined, true, 'host_prompt');
  }
  async function startEntry() { return SESSION_NAVIGATION.startEntry(); }
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
    wire(); document.addEventListener('click', dismissHeaderOverflow, true);
    // Plans to review and finished runs, marked on their rows (owner: attention).
    const attention = MODULES.optional('attention');
    if (attention) attention.bind({
      projectId, selectedId: () => (state.session && state.session.session_id) || '', openSession: id => openSession(id),
      onChange: () => render(),
    });
    const capabilitiesReady = api().loadCapabilities
      ? Promise.resolve(api().loadCapabilities()).catch(() => null)
      : Promise.resolve(null);
    // Capabilities first: the status read needs to know a hosted instance.
    state.startupPromise = capabilitiesReady.then(() => loadStatus()).then(async () => {
      if (state.host) render(true);
      // Get Started's workflow demo asks once for the read-only replay (screens-help.js).
      let helpDemo = false;
      try {
        helpDemo = window.sessionStorage.getItem('easyicu.help.workflowDemo') === '1';
        window.sessionStorage.removeItem('easyicu.help.workflowDemo');
      } catch (_) {}
      if (helpDemo) { openDemo(); return; }
      openHostedConversation();
      const hubIntent = STUDY_WORKSPACE.hasHubIntent(), previousSessionId = state.session && state.session.session_id;
      if (hubIntent && projectId() && state.session && !state.creating) await createSession();
      if (hubIntent && previousSessionId && state.session && state.session.session_id === previousSessionId) return;
      if (hubIntent && state.session && (!previousSessionId || state.session.session_id !== previousSessionId)) state.draft = '';
      try {
        const suggestion = window.sessionStorage.getItem('easyicu.skillHub.question');
        if (suggestion && state.session && (!previousSessionId || state.session.session_id !== previousSessionId)) {
          state.draft = suggestion;
          window.sessionStorage.removeItem('easyicu.skillHub.question');
          // Attaching the chosen workflow repaints the composer once more; keep
          // the cursor at the end of the request through those first paints.
          state.focusComposerUntil = Date.now() + 3000;
          render();
        }
      } catch (_) {}
    }).finally(() => {
      state.startupPromise = null;
    });
    return state.startupPromise;
  }
  // A hosted clinician lands in a conversation (screens-guided-pi-hosted.js).
  const openHostedConversation = () => { const hosted = MODULES.optional('hosted'); if (hosted) hosted.openConversation(state, connectionReady(), createSession); };
  function unmount() {
    MESSAGE_QUEUE.stop();
    RUN_FILES.reset();
    document.removeEventListener('click', dismissHeaderOverflow, true);
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
