/* Copilot adapter for the native embedded data workbench.
   Rendering stays with the Patient/Cohort/Cross-DB visualization owners. */
(function () {
  'use strict';

  function owner() { return window.EU_VIZ_EMBEDDED_WORKBENCH; }

  function render(payload, view) {
    const workbench = owner();
    if (workbench && typeof workbench.render === 'function') {
      return workbench.render(payload || {}, String(view || 'cohort_summary'), {});
    }
    return '<div class="gpi-preview-state error">Native Data Workbench renderer unavailable.</div>';
  }

  function mount(host, payload, view) {
    const workbench = owner();
    if (workbench && typeof workbench.mount === 'function') {
      workbench.mount(host, payload || {}, String(view || 'cohort_summary'));
      return;
    }
    if (host) host.innerHTML = render(payload, view);
  }

  window.EasyICU.guidedPi.declare('dataPreview', { mount, render });
})();
