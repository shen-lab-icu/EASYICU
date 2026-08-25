/* Copilot right-preview adapter for the native Data Extraction owner. */
(function () {
  let host = null;
  let options = {};

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[character]);
  }

  function jobSummary(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return '';
    const status = String(snapshot.status || 'running');
    const events = Array.isArray(snapshot.events) ? snapshot.events : [];
    const latest = events.slice().reverse().find(row => row && (row.message || row.step));
    const result = snapshot.result && typeof snapshot.result === 'object' ? snapshot.result : {};
    const tone = status === 'done' ? 'ok' : (status === 'failed' || status === 'cancelled' ? 'bad' : 'info');
    return `<div class="note ${tone} gpi-extraction-job" role="status">
      <div class="ico">${icon(status === 'done' ? 'check' : (tone === 'bad' ? 'alert' : 'activity'), 14)}</div>
      <div class="body"><div class="t">${escapeHtml(status === 'done' ? t('Extraction completed', '数据提取已完成') : status === 'running' ? t('Extraction is running', '数据提取正在运行') : t('Extraction task stopped', '数据提取任务已停止'))}</div>
      <div class="d">${escapeHtml((latest && (latest.message || latest.step)) || t('The live details also remain in the Copilot activity timeline.', '实时详情也会保留在 Copilot 活动时间线中。'))}</div>
      ${status === 'done' ? `<div class="gpi-extraction-job-metrics"><span>${t('Rows', '行数')} <b>${escapeHtml(result.total_rows == null ? '—' : result.total_rows)}</b></span><span>${t('Data files', '数据文件')} <b>${escapeHtml(result.file_count == null ? (result.files_written == null ? '—' : result.files_written) : result.file_count)}</b></span></div>` : ''}
      </div><button class="btn sm ghost" type="button" data-gpi-extraction-refresh>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
    </div>`;
  }

  function projectCopilotSetup(root) {
    const express = root.querySelector('.express');
    const custom = root.querySelector('.ex2-custom');
    const owner = window.EU_EXTRACTION_NATIVE_OWNER;
    if (!express || !custom || !owner || typeof owner.setupSummary !== 'function') return;

    const summary = owner.setupSummary();
    const meta = [
      `${t('Cohort', '队列')} · <b>${escapeHtml(summary.cohort || '—')}</b>`,
      `${t('Modules', '模块')} · <b>${escapeHtml(summary.moduleCount == null ? 0 : summary.moduleCount)}</b>`,
      `${t('Concepts', '概念')} · <b>${escapeHtml(summary.conceptCount == null ? 0 : summary.conceptCount)}</b>`,
    ].map(row => `<span>${row}</span>`).join('');
    const run = custom.querySelector('[data-ex-run="custom"]');
    const runDisabled = !summary.runnable || !run || run.disabled;
    const compact = document.createElement('div');
    compact.className = 'gpi-extraction-compact';
    compact.setAttribute('aria-label', t('Current extraction setup', '当前抽取设置'));
    compact.innerHTML = `<div class="gpi-extraction-compact-main">
      <strong>${t('Current extraction setup', '当前抽取设置')}</strong>
      <div class="gpi-extraction-compact-meta">${meta}</div>
    </div>
    <button class="btn primary" data-ex-run="custom" ${runDisabled ? 'disabled' : ''}>${icon('play', 14)} ${t('Start extraction', '开始抽取')}</button>`;
    express.replaceWith(compact);

    const divider = root.querySelector('.ex2-divider');
    if (divider) divider.remove();
    custom.hidden = false;

    const layout = custom.querySelector('.ex2-layout > div');
    const moduleCard = layout && layout.querySelector('#exModGrid')
      ? layout.querySelector('#exModGrid').closest('.cfg')
      : null;
    if (layout && moduleCard) layout.prepend(moduleCard);

    const lead = root.querySelector('.page-head .lead');
    if (lead) lead.textContent = t(
      'Review feature modules and cohort criteria, then start extraction.',
      '确认特征模块与队列条件后开始抽取。'
    );
  }

  function paint() {
    const owner = window.EU_EXTRACTION_NATIVE_OWNER;
    if (!host || !host.isConnected || !owner) return;
    const previousScroller = host.querySelector('[data-gpi-extraction-embed]');
    const previousScrollTop = previousScroller ? previousScroller.scrollTop : 0;
    const sourceId = String(options.sourceId || '').trim();
    const currentReceipt = typeof owner.handoffReceipt === 'function' ? owner.handoffReceipt() : {};
    const resultReady = !!(currentReceipt && currentReceipt.output_dir);
    const syncLabel = resultReady
      ? t('Send result to Copilot', '将抽取结果交给 Copilot')
      : t('Save setup to Copilot', '保存配置到 Copilot');
    const syncReceiptMessage = options.syncReceipt ? t(
      'StudyContext revision ' + Number(options.syncReceipt.study_revision || 0)
        + ' now contains this ' + (options.syncReceipt.receipt_kind === 'extraction_result' ? 'extraction result' : 'extraction setup')
        + '. The next Copilot turn reads that typed state; the local folder path is not sent as model-authored text.',
      'StudyContext 第 ' + Number(options.syncReceipt.study_revision || 0)
        + ' 版已保存本次' + (options.syncReceipt.receipt_kind === 'extraction_result' ? '抽取结果' : '抽取配置')
        + '。下一轮 Copilot 会读取这份结构化状态；本机文件夹路径不会作为模型文字发送。'
    ) : '';
    host.innerHTML = `<div class="gpi-extraction-embed" data-gpi-extraction-embed>
      <div class="gpi-extraction-toolbar">
        <div><span>${t('Native Data Extraction', '原生数据提取')}</span><strong>${t('The same owner as Classic Workspace', '与经典工作台共用同一个功能 owner')}</strong></div>
        <div class="row gap-8">
          ${owner.isReal() ? '' : `<button class="btn sm" type="button" data-gpi-extraction-real>${icon('db', 12)} ${t('Use real local data', '使用本地真实数据')}</button>`}
          ${sourceId ? `<button class="btn sm" type="button" data-gpi-extraction-download>${icon('download', 12)} ${t('Download data package', '下载数据包')}</button>` : ''}
          <button class="btn sm primary" type="button" data-gpi-extraction-sync>${icon(options.syncReceipt ? 'check' : 'agent', 12)} ${options.syncReceipt ? t('Synced to Copilot', '已同步到 Copilot') : syncLabel}</button>
        </div>
      </div>
      ${options.syncReceipt ? `<div class="note ok gpi-extraction-sync-receipt" role="status"><div class="ico">${icon('check', 13)}</div><div class="body"><div class="t">${t('Synced to Copilot', '已同步到 Copilot')}</div><div class="d">${escapeHtml(syncReceiptMessage)}</div></div></div>` : ''}
      ${options.syncError ? `<div class="note bad" role="alert"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Copilot sync failed', 'Copilot 同步失败')}</div><div class="d">${escapeHtml(options.syncError)}</div></div></div>` : ''}
      ${options.downloadError ? `<div class="note bad" role="alert"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Download blocked', '下载已阻止')}</div><div class="d">${escapeHtml(options.downloadError)}</div></div></div>` : ''}
      ${jobSummary(options.jobSnapshot)}
      <div class="gpi-extraction-native">${owner.render()}</div>
    </div>`;
    projectCopilotSetup(host);
    owner.bind(host);
    const scroller = host.querySelector('[data-gpi-extraction-embed]');
    if (scroller && previousScrollTop > 0) scroller.scrollTop = previousScrollTop;
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

  function hydrateStudyContext() {
    const bridge = window.EU_EXTRACTION_STUDY_CONTEXT;
    if (!bridge || typeof bridge.hydrate !== 'function') return null;
    return bridge.hydrate(
      options.studyContext,
      options.resource && options.resource.expected_database,
    );
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
    const owner = window.EU_EXTRACTION_NATIVE_OWNER;
    if (!owner || typeof owner.syncToCopilot !== 'function') return;
    button.disabled = true;
    button.textContent = t('Syncing…', '正在同步…');
    options.syncError = '';
    return Promise.resolve(owner.syncToCopilot()).then(receipt => {
      const copilot = window.EU_GUIDED_PI;
      const rebound = copilot && typeof copilot.rebind === 'function' ? copilot.rebind() : null;
      return Promise.resolve(rebound).then(() => {
        if (copilot && typeof copilot.notifyExtractionHandoff === 'function') copilot.notifyExtractionHandoff(receipt);
        options.syncReceipt = receipt;
        paint();
        return receipt;
      });
    }).catch(error => {
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
      const owner = window.EU_EXTRACTION_NATIVE_OWNER;
      hydrateStudyContext();
      if (options.resource && options.resource.state === 'setup' && owner && !owner.isReal()) {
        owner.useRealData();
      }
      paint();
    },
    unmount(nextHost) {
      if (!nextHost || nextHost === host) { host = null; options = {}; }
    },
    isMounted: () => !!(host && host.isConnected),
    repaint: paint,
  };
})();
