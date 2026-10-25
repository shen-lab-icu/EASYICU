/* Owner: Guided Pi data-preview adapter widget. */
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
    return `<div class="gpi-preview-state error">${window.EU_LANG === 'zh' ? '暂时无法显示数据预览，请重试。' : 'The data preview could not be displayed. Please retry.'}</div>`;
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
