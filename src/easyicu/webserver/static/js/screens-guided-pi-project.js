/* Copilot project preparation owner.
   Resolves the persisted project/study binding before the conversation owner
   loads sessions. Scientific workflow state remains server-owned. */
(function () {
  'use strict';

  const PROJECT_QUERY_KEY = 'pi_project';
  const SESSION_QUERY_KEY = 'pi_session';

  function queryValue(key) {
    try {
      return new URL(window.location.href).searchParams.get(key) || '';
    } catch (error) {
      return '';
    }
  }

  function requestedProjectId() {
    return queryValue(PROJECT_QUERY_KEY);
  }

  function requestedSessionId(projectId) {
    const requestedProject = requestedProjectId();
    const currentProject = String(projectId || '').trim();
    if (!requestedProject || requestedProject !== currentProject) return '';
    return queryValue(SESSION_QUERY_KEY);
  }

  function syncLocation(projectId, sessionId) {
    if (!window.location || !window.history || !window.history.replaceState) return;
    try {
      const next = new URL(window.location.href);
      const project = String(projectId || '').trim();
      const session = String(sessionId || '').trim();
      if (project) next.searchParams.set(PROJECT_QUERY_KEY, project);
      else next.searchParams.delete(PROJECT_QUERY_KEY);
      if (project && session) next.searchParams.set(SESSION_QUERY_KEY, session);
      else next.searchParams.delete(SESSION_QUERY_KEY);
      window.history.replaceState(null, '', `${next.pathname}${next.search}${next.hash}`);
    } catch (error) {}
  }

  async function prepare(options) {
    const {
      state, api, projectId, connectionReady, loadWorkflow,
      loadProjectSessions, render,
    } = options;
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    const bindingReceipt = state.project && state.project.binding_receipt;
    try {
      const initialized = await api().initializePiCopilotProject({
        project_id: expectedProjectId,
        title: (state.project && state.project.title) || expectedProjectId,
        confirm_initialization: false,
        binding_receipt: bindingReceipt || undefined,
      });
      if (expectedProjectId !== projectId()) return;
      state.projectIssue = '';
      state.projectInitialization = initialized || { status: 'ready' };
      if (bindingReceipt && initialized && initialized.binding_receipt) {
        state.project = { ...state.project, binding_receipt: null };
      }
      const workflowReady = loadWorkflow().then(render);
      if (connectionReady()) {
        await loadProjectSessions(false);
      } else {
        await workflowReady;
      }
    } catch (error) {
      if (expectedProjectId !== projectId()) return;
      if (error && error.code === 'pi_project_initialization_required') {
        state.projectIssue = '';
        state.projectInitialization = {
          required: true,
          missingRequired: (error.details && error.details.missing_required) || [],
        };
        state.error = '';
        render();
        return;
      }
      if (error && error.code === 'pi_project_study_context_missing') {
        state.projectIssue = error.code;
        state.error = '';
        render();
        return;
      }
      throw error;
    }
  }

  window.EU_GUIDED_PI_PROJECT = Object.freeze({
    prepare, requestedProjectId, requestedSessionId, syncLocation,
  });
})();
