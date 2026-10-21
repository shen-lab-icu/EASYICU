/* Owner: Guided Pi hosted-instance presentation. */
/* A hosted instance is one clinician's EasyICU on the hospital server, reached
   through the login gateway. The server reports it as
   /api/capabilities → deployment.hosted (EASYICU_HOSTED=1). The administrator
   connects the model and prepares the data sources there, so the page skips
   those steps: the clinician lands in their own project with a conversation
   open, never chooses a server folder or sees a server path, and is told whom
   to contact when the model is missing. This owner only reads the reported
   facts; provider authority, data roots and project creation stay on the
   server (provider_gate, EASYICU_DATA_ROOTS, guided_sessions). */
(function () {
  'use strict';

  // The project a hosted clinician gets on first use. Folder names keep
  // Chinese characters; the server derives the folder from this title.
  const DEFAULT_PROJECT_TITLE = '我的研究';

  function deployment() {
    const capabilities = window.EU_CAPABILITIES || {};
    const value = capabilities.deployment;
    return value && typeof value === 'object' ? value : {};
  }

  function active() {
    return deployment().hosted === true;
  }

  function providerProvisioned() {
    return active() && deployment().provider_provisioned === true;
  }

  // Administrator contact from the settings store (settings.support_contact);
  // empty when the administrator left it unset.
  function supportContact() {
    const value = (window.EU_SETTINGS || {}).support_contact;
    return typeof value === 'string' ? value.replace(/[\u0000-\u001f\u007f]/g, ' ').trim().slice(0, 200) : '';
  }

  function rowTime(row) {
    return String((row && (row.updated_at || row.created_at)) || '');
  }

  // The project a hosted clinician lands in when the address names none: the
  // most recently updated one, or null when the instance has none yet (the
  // caller then creates DEFAULT_PROJECT_TITLE).
  function landingProject(rows) {
    const list = Array.isArray(rows) ? rows.filter(row => row && row.id) : [];
    if (!list.length) return null;
    return list.slice().sort((a, b) => rowTime(b).localeCompare(rowTime(a)))[0];
  }

  // A hosted clinician has nothing to choose on the empty project page, so
  // the conversation opens by itself once the project is ready and the model
  // the administrator configured is usable.
  function opensConversation(state, connectionReady) {
    return active() && Boolean(connectionReady) && Boolean(state)
      && !state.session && !state.creating && !state.projectLoading
      && !state.demoMode && !state.projectIssue
      && Boolean(state.project && state.project.id);
  }

  function contactLine(tr, esc) {
    const contact = supportContact();
    return contact
      ? `<p class="gpi-hosted-contact">${tr('Administrator', '管理员')}：${esc(contact)}</p>`
      : '';
  }

  // Shown wherever the local build would ask the researcher to connect a
  // model: the administrator owns that connection on a hosted instance.
  function missingModelHtml(options) {
    const { tr, esc } = options;
    return `
      <div class="gpi-hosted-notice" data-gpi-hosted-model-missing>
        <p>${tr('The administrator sets up the model for this EasyICU, and it is not ready yet. Please contact the administrator.', '这里的模型由管理员统一配置，目前还没有配置好。请联系管理员。')}</p>
        ${contactLine(tr, esc)}
      </div>`;
  }

  function summaryHtml(options) {
    const { tr, esc, connectionReady } = options;
    if (connectionReady) {
      return `<div class="gpi-provider-summary ready" data-gpi-hosted-model><span class="gpi-dot"></span><div><strong>${tr('Model ready', '模型已就绪')}</strong><small>${tr('Set up by the administrator', '由管理员统一配置')}</small></div></div>`;
    }
    const contact = supportContact();
    return `<div class="gpi-provider-summary warn" data-gpi-hosted-model><span class="gpi-dot"></span><div><strong>${tr('Model not set up yet', '模型尚未配置')}</strong><small>${contact ? `${tr('Contact', '请联系')} ${esc(contact)}` : tr('Please contact the administrator', '请联系管理员')}</small></div></div>`;
  }

  window.EasyICU.guidedPi.declare('hosted', {
    DEFAULT_PROJECT_TITLE,
    active,
    providerProvisioned,
    supportContact,
    landingProject,
    opensConversation,
    missingModelHtml,
    summaryHtml,
  });
})();
