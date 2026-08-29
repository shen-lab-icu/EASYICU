/* Guided Copilot data-source binding and extraction handoff owner.

   Owner: the three actions that connect a conversation to real data --
   confirming the user's data-source choice, persisting that binding, and
   handing the resulting receipt to the Data Extraction screen. Split out of
   screens-guided-pi.js, which was hundreds of lines past its size ratchet.

   The host state these actions reach crosses the seam as an explicit
   contract: they read the bound session and the busy flag, and write only the
   session, the error banner, and the workflow-receipt list. */
(function () {
  'use strict';

  function create(host) {
    const api = host.api;
    const render = host.render;
    const projectId = host.projectId;
    const loadWorkflow = host.loadWorkflow;
    const DATA_CONSENT = host.dataConsent;
    const errorText = host.errorText;
    const rememberSession = host.rememberSession;
    const continueAfterDataSourceConfirmation = host.continueAfterDataSourceConfirmation;

    async function authorizeDataSource(action, options) {
      if (!host.session() || host.busy() || !api().authorizePiCopilotDataSource) return;
      const database = String(options && options.database || '').trim();
      host.setError('');
      try {
        const payload = await api().authorizePiCopilotDataSource(
          host.session().session_id,
          { project_id: projectId(), action, ...(database ? { database } : {}) },
        );
        host.setSession(payload.session || host.session());
        rememberSession(host.session().session_id);
        if (action === 'begin_local_selection' || action === 'begin_full_data_selection') {
          const contextId = String(
            payload.resource && payload.resource.study_context_id
            || host.session() && host.session().binding && host.session().binding.study_context_id
            || ''
          );
          const store = window.EU_STUDY_CONTEXT;
          // A project may have been created after the page-level StudyContext
          // cache finished hydrating. Refresh the owner list before activating
          // the session-bound context so a brand-new project cannot look missing.
          if (store && typeof store.hydrate === 'function') await store.hydrate({ force: true });
          const active = store && typeof store.active === 'function' ? store.active() : null;
          if (contextId && (!active || active.id !== contextId) && store && typeof store.activate === 'function') {
            await store.activate(contextId);
          }
        }
        if (payload.resource && window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.open) {
          window.EU_GUIDED_PI_PREVIEW.open(payload.resource, projectId());
        }
        render();
      } catch (error) {
        host.setError(errorText(error));
        render();
      }
    }

    function notifyExtractionHandoff(receipt) {
      if (!receipt || typeof receipt !== 'object') return false;
      const id = String(receipt.id || ('extraction-' + Date.now()));
      const projected = {
        id,
        role: 'workflow_receipt',
        database: String(receipt.database || '').slice(0, 80),
        source_label: String(receipt.source_label || '').slice(0, 160),
        output_dir: String(receipt.output_dir || '').slice(0, 2048),
        data_file_count: Number(receipt.data_file_count || 0),
        support_file_count: Number(receipt.support_file_count || 0),
        total_rows: receipt.total_rows == null ? null : Number(receipt.total_rows),
        cohort_summary: String(receipt.cohort_summary || '').slice(0, 240),
        modules: Array.isArray(receipt.modules) ? receipt.modules.slice(0, 30).map(value => String(value).slice(0, 80)) : [],
        export_format: String(receipt.export_format || '').slice(0, 40),
        receipt_kind: receipt.receipt_kind === 'extraction_result' ? 'extraction_result' : 'extraction_setup',
        study_context_id: String(receipt.study_context_id || '').slice(0, 160),
        study_revision: Number(receipt.study_revision || 0),
      };
      host.setWorkflowReceipts(
        host.workflowReceipts().filter(row => row.id !== id).concat([projected]).slice(-3)
      );
      if (
        host.session()
        && DATA_CONSENT
        && DATA_CONSENT.selectionInProgress(host.session())
        && api().authorizePiCopilotDataSource
      ) {
        api().authorizePiCopilotDataSource(
          host.session().session_id,
          { project_id: projectId(), action: 'confirm_selected_source' },
        ).then(payload => {
          host.setSession(payload.session || host.session());
          rememberSession(host.session().session_id);
          loadWorkflow().then(render);
        }).catch(error => {
          host.setError(errorText(error));
          render();
        });
      }
      render();
      requestAnimationFrame(() => {
        const log = host.root() && host.root().querySelector('[data-gpi-log]');
        if (log) log.scrollTop = log.scrollHeight;
      });
      return true;
    }

    async function confirmDataSourceBinding(receipt) {
      if (
        !receipt
        || receipt.receipt_kind !== 'data_source_binding'
        || !host.session()
        || !DATA_CONSENT
        || !DATA_CONSENT.selectionInProgress(host.session())
        || !api().authorizePiCopilotDataSource
      ) return false;
      const payload = await api().authorizePiCopilotDataSource(
        host.session().session_id,
        { project_id: projectId(), action: 'confirm_selected_source' },
      );
      host.setSession(payload.session || host.session());
      rememberSession(host.session().session_id);
      document.dispatchEvent(new CustomEvent('easyicu:guided-projects-refresh'));
      if (payload.resource && window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.open) {
        window.EU_GUIDED_PI_PREVIEW.open(payload.resource, projectId());
      } else if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) {
        window.EU_GUIDED_PI_PREVIEW.close();
      }
      await loadWorkflow();
      render();
      if (payload.resource) return true;
      if (await continueAfterDataSourceConfirmation()) return true;
      requestAnimationFrame(() => {
        const composer = host.root() && host.root().querySelector('[data-gpi-input]');
        if (composer) composer.focus();
      });
      return true;
    }

    return { authorizeDataSource, notifyExtractionHandoff, confirmDataSourceBinding };
  }

  window.EU_GUIDED_PI_DATA_BINDING = { create };
})();
