/* Owner: result additions to one study; the server owns executable rosters. */
(function () {
  'use strict';
  function create({ tr, esc, api, projectId, host, session, canPreview, onError, errorText }) {
    let current = null;
    let busy = false;
    function render(run) {
      const valid = current && current.project === projectId() && current.run_id === run.run_id;
      const rows = valid ? current.options : [];
      const notice = valid && current.source_run_sha256
        ? tr('Additions use the same data version as this result.', '追加分析沿用这次结果的数据版本。') : '';
      return `<div class="gpi-result-append" data-append-run="${esc(run.run_id)}">${rows.length ? `<strong>${esc(tr('Continue this study', '继续这项研究'))}</strong><p>${esc(notice)}</p>` : ''}${rows.map(row => {
        const label = esc(tr(row.label_en, row.label_zh));
        return row.available
          ? `<button class="btn sm" type="button" data-gpi-result-append="${esc(row.kind)}"${busy || !canPreview() ? ' disabled' : ''}>${label}</button>`
          : `<p>${label}：${esc(tr(row.status_en, row.status_zh))} <button class="btn sm" type="button" data-gpi-append-owner="${esc(row.owner)}">${esc(tr('Open analysis settings', '进入分析设置'))}</button></p>`;
      }).join('')}</div>`;
    }
    async function load(run) {
      const expectedProject = projectId();
      if (!expectedProject || !run.run_id || !api().loadPiCopilotAppendOptions) return;
      if (current && current.project === expectedProject && current.run_id === run.run_id) return;
      const requested = { project: expectedProject, run_id: run.run_id, options: [] };
      current = requested;
      try {
        const result = await api().loadPiCopilotAppendOptions(expectedProject, run.run_id);
        if (current !== requested || projectId() !== expectedProject || result.run_id !== run.run_id) return;
        current = { ...requested, ...result, options: Array.isArray(result.options) ? result.options : [] };
      } catch (_error) {
        if (current !== requested) return;
        current.options = [];
      }
      const node = host && host()?.querySelector('.gpi-result-append');
      if (node && node.dataset.appendRun === run.run_id) node.outerHTML = render(run);
    }
    async function propose(kind) {
      if (busy || !canPreview() || !session() || !current || current.project !== projectId()
        || !current.options.some(row => row.kind === kind && row.available) || !current.source_run_sha256) return false;
      const expectedProject = projectId();
      busy = true;
      try {
        await api().proposePiCopilotAppend(session().session_id, {
          project_id: expectedProject, run_id: current.run_id, kind, expected_source_sha256: current.source_run_sha256,
        });
        return expectedProject === projectId();
      } catch (error) {
        if (expectedProject === projectId()) onError(`${tr('Could not prepare the addition', '无法准备追加分析')}: ${errorText(error)}`);
        return false;
      } finally { busy = false; }
    }
    return Object.freeze({ load, render, propose });
  }
  window.EasyICU.guidedPi.declare('resultAppend', { create });
})();
