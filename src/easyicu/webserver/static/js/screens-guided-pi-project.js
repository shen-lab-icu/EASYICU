/* Copilot project preparation owner.
   Resolves the persisted project/study binding before the conversation owner
   loads sessions. Scientific workflow state remains server-owned. */
(function () {
  'use strict';

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
      await loadWorkflow();
      if (connectionReady()) await loadProjectSessions();
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

  window.EU_GUIDED_PI_PROJECT = Object.freeze({ prepare });
})();
