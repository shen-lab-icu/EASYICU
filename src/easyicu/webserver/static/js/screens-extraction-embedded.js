/* Copilot right-preview adapter for the native Data Extraction owner. */
(function () {
  let host = null;
  let options = {};

  function jobSummary(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return '';
    const status = String(snapshot.status || 'running');
    const events = Array.isArray(snapshot.events) ? snapshot.events : [];
    const latest = events.slice().reverse().find(row => row && (row.message || row.step));
    const result = snapshot.result && typeof snapshot.result === 'object' ? snapshot.result : {};
    const tone = status === 'done' ? 'ok' : (status === 'failed' || status === 'cancelled' ? 'bad' : 'info');
    return `<div class="note ${tone} gpi-extraction-job" role="status">
      <div class="ico">${icon(status === 'done' ? 'check' : (tone === 'bad' ? 'alert' : 'activity'), 14)}</div>
      <div class="body"><div class="t">${escHtml(status === 'done' ? t('Extraction completed', '数据提取已完成') : status === 'running' ? t('Extraction is running', '数据提取正在运行') : t('Extraction task stopped', '数据提取任务已停止'))}</div>
      <div class="d">${escHtml((latest && (latest.message || latest.step)) || t('The live details also remain in the Copilot activity timeline.', '实时详情也会保留在 Copilot 活动时间线中。'))}</div>
      ${status === 'done' ? `<div class="gpi-extraction-job-metrics"><span>${t('Rows', '行数')} <b>${escHtml(result.total_rows == null ? '—' : result.total_rows)}</b></span><span>${t('Files', '文件')} <b>${escHtml(result.files_written == null ? '—' : result.files_written)}</b></span></div>` : ''}
      </div><button class="btn sm ghost" type="button" data-gpi-extraction-refresh>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
    </div>`;
  }

  function paint() {
    const owner = window.EU_EXTRACTION_NATIVE_OWNER;
    if (!host || !host.isConnected || !owner) return;
    const sourceId = String(options.sourceId || '').trim();
    host.innerHTML = `<div class="gpi-extraction-embed" data-gpi-extraction-embed>
      <div class="gpi-extraction-toolbar">
        <div><span>${t('Native Data Extraction', '原生数据提取')}</span><strong>${t('The same owner as Classic Workspace', '与经典工作台共用同一个功能 owner')}</strong></div>
        <div class="row gap-8">
          ${owner.isReal() ? '' : `<button class="btn sm" type="button" data-gpi-extraction-real>${icon('db', 12)} ${t('Use real local data', '使用本地真实数据')}</button>`}
          ${sourceId ? `<button class="btn sm" type="button" data-gpi-extraction-download>${icon('download', 12)} ${t('Download data package', '下载数据包')}</button>` : ''}
          <button class="btn sm primary" type="button" data-gpi-extraction-sync>${icon('agent', 12)} ${t('Sync back to Copilot', '同步回 Copilot')}</button>
        </div>
      </div>
      ${options.downloadError ? `<div class="note bad" role="alert"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Download blocked', '下载已阻止')}</div><div class="d">${escHtml(options.downloadError)}</div></div></div>` : ''}
      ${jobSummary(options.jobSnapshot)}
      <div class="gpi-extraction-native">${owner.render()}</div>
    </div>`;
    owner.bind(host);
    host.querySelectorAll('[data-study-handoff]').forEach(control => { control.hidden = true; });
    if (owner.isPreparedExport()) {
      host.querySelectorAll('[data-ex-run]').forEach(control => {
        control.disabled = true;
        control.innerHTML = `${icon('check', 13)} ${t('Prepared export is ready — sync to Copilot', '已准备导出可直接使用 — 同步回 Copilot')}`;
      });
    }
    const realButton = host.querySelector('[data-gpi-extraction-real]');
    if (realButton) realButton.addEventListener('click', () => { owner.useRealData(); paint(); });
    const syncButton = host.querySelector('[data-gpi-extraction-sync]');
    if (syncButton) syncButton.addEventListener('click', syncToCopilot);
    const downloadButton = host.querySelector('[data-gpi-extraction-download]');
    if (downloadButton) downloadButton.addEventListener('click', downloadExport);
    const refresh = host.querySelector('[data-gpi-extraction-refresh]');
    if (refresh) refresh.addEventListener('click', refreshJob);
  }

  function downloadExport(event) {
    const button = event.currentTarget;
    const sourceId = String(options.sourceId || '').trim();
    const api = window.EU_API;
    if (!sourceId || !api || typeof api.downloadRegisteredExport !== 'function') return;
    button.disabled = true;
    button.textContent = t('Preparing download…', '正在准备下载…');
    options.downloadError = '';
    api.downloadRegisteredExport(sourceId).then(() => {
      button.textContent = t('Download started', '已开始下载');
    }).catch(error => {
      options.downloadError = String(error && (error.message || error.code) || error);
      paint();
    });
  }

  function syncToCopilot(event) {
    const button = event.currentTarget;
    const store = window.EU_STUDY_CONTEXT;
    if (!store || typeof store.handoff !== 'function') return;
    button.disabled = true;
    button.textContent = t('Syncing…', '正在同步…');
    const handoff = store.handoff({ sourceRoute: 'extraction', targetRoute: 'guided' });
    Promise.resolve(handoff && handoff.persisted).then(() => {
      const copilot = window.EU_GUIDED_PI;
      return copilot && typeof copilot.rebind === 'function' ? copilot.rebind() : null;
    }).then(paint).catch(error => {
      options.syncError = String(error && error.message || error);
      paint();
    });
  }

  function refreshJob(event) {
    const button = event.currentTarget;
    const jobId = String(options.jobId || '').trim();
    const api = window.EU_API;
    if (!jobId || !api || typeof api.loadJobSnapshot !== 'function') return;
    button.disabled = true;
    api.loadJobSnapshot(jobId).then(snapshot => {
      options.jobSnapshot = snapshot;
      paint();
    }).catch(() => { button.disabled = false; });
  }

  window.EU_EXTRACTION_EMBEDDED_WORKSPACE = {
    mount(nextHost, nextOptions) {
      host = nextHost;
      options = Object.assign({}, nextOptions || {});
      paint();
    },
    unmount(nextHost) {
      if (!nextHost || nextHost === host) { host = null; options = {}; }
    },
    isMounted: () => !!(host && host.isConnected),
    repaint: paint,
  };
})();
