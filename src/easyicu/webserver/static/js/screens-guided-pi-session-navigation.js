/* Owner: project-scoped conversation navigation and selection lifecycle.
   Receives explicit UI dependencies; scientific state and execution remain
   with their existing owners. */
(function () {
  'use strict';
  const MODULES = window.EasyICU.guidedPi;
  function create(options) {
    const {
      state, api, projectId, connectionReady, closeSourceView, CONNECT,
      displayProjectTitle, tr, EFFORT_MENU, loadWorkflow, transcriptMessages,
      STUDY_WORKSPACE, hydrateProjectedJob, rememberSession, errorText, render,
      navigationSessionTitle, sessionMatchesUiLanguage, rememberedSession,
      openSession, uiLanguage, closeSource, closeChildSource, setShell,
    } = options;
    async function createSession() {
      if (state.creating || state.projectLoading || !projectId()) return;
      if (state.followUpDraft) { state.draft = ''; state.followUpDraft = null; }
      closeSourceView();
      const expectedProjectId = projectId();
      const selectionRevision = ++state.sessionSelectionRevision;
      // connectionReady() covers the account model and the Chat Completions API.
      if (!CONNECT.ask('start_conversation', createSession)) return;
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
          if (expectedProjectId !== projectId() || selectionRevision !== state.sessionSelectionRevision) return;
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
        const hubAttachment = STUDY_WORKSPACE.takeHubAttachment(expectedProjectId, state.session);
        const attachOwner = MODULES.optional('skillAttach');
        if (hubAttachment && attachOwner) {
          const session = state.session;
          await attachOwner.ensure(expectedProjectId, session, hubAttachment.source, hubAttachment.id);
          if (state.session !== session || expectedProjectId !== projectId()) return;
        }
        state.agentMode = state.session.agent_mode || state.agentMode;
        hydrateProjectedJob(state.workflow && state.workflow.active_job);
        state.projectInitialization = null;
        rememberSession(state.session.session_id);
        const projectOwner = MODULES.require('project');
        if (projectOwner && projectOwner.syncLocation) {
          projectOwner.syncLocation(expectedProjectId, state.session.session_id);
        }
        state.sessions = [state.session].concat(state.sessions.filter(row => row.session_id !== state.session.session_id));
      } catch (error) {
        if (expectedProjectId === projectId() && selectionRevision === state.sessionSelectionRevision) state.error = errorText(error);
      } finally {
        if (expectedProjectId === projectId() && selectionRevision === state.sessionSelectionRevision) { state.creating = false; render(); }
      }
    }

    async function renameSession(row) {
      if (!row || state.busy || state.childJobId || !api().renamePiCopilotSession) return;
      const expectedProjectId = projectId(), selectionRevision = state.sessionSelectionRevision;
      const isCurrent = () => expectedProjectId === projectId() && selectionRevision === state.sessionSelectionRevision;
      const current = navigationSessionTitle(row);
      const requested = window.prompt(tr('Rename this conversation', '重命名对话'), current);
      if (requested === null) return;
      const title = String(requested || '').replace(/\s+/g, ' ').trim().slice(0, 100);
      if (!title || title === current) return;
      try {
        const payload = await api().renamePiCopilotSession(row.session_id, {
          project_id: expectedProjectId, title,
        });
        if (!isCurrent()) return;
        const renamed = payload && payload.session;
        if (!renamed) return;
        state.sessions = state.sessions.map(item => item.session_id === renamed.session_id ? { ...item, ...renamed } : item);
        if (state.session && state.session.session_id === renamed.session_id) state.session = { ...state.session, ...renamed };
        state.error = '';
      } catch (error) {
        if (!isCurrent()) return;
        state.error = errorText(error);
      }
      render();
    }

    // The ••• menu asks twice (删除对话, then 确认删除); the conversation's
    // study results stay in the project.
    async function removeSession(row) {
      if (!row || !api().deletePiCopilotSession) return;
      const expectedProjectId = projectId(), selectionRevision = state.sessionSelectionRevision;
      const isCurrent = () => expectedProjectId === projectId() && selectionRevision === state.sessionSelectionRevision;
      const deletingSelected = Boolean(state.session && state.session.session_id === row.session_id);
      if (deletingSelected && (state.busy || state.childJobId)) return;
      try {
        await api().deletePiCopilotSession(row.session_id, {
          project_id: expectedProjectId, confirm: true,
        });
        if (!isCurrent()) return;
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
        if (!isCurrent()) return;
        state.error = errorText(error);
        render();
      }
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

    async function startEntry() {
      state.taskDraft = '';
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
      state.followUpDraft = null;
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
      // New conversation reuses an empty draft this project already has instead of adding
      // another one to the rail; a session is only created when none is free.
      const emptyDraft = state.sessions.find(row => STUDY_WORKSPACE.isEmptyConversation(row));
      if (emptyDraft) { await openSession(emptyDraft.session_id); return; }
      await createSession();
    }
    return { createSession, renameSession, removeSession, loadProjectSessions,
      reloadSessionsForLanguage, handleLanguageChange, startEntry };
  }
  MODULES.declare('sessionNavigation', { create });
})();
