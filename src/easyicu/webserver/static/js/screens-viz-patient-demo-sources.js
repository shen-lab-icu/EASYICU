/* Owner: Patient Review demo-source widget. */
/* ============================================================
   Official ICU demo-source owner.

   Owns PhysioNet demo discovery, prepare-job state, and source-card
   rendering shared by Patient Review, Cohort Statistics, and Cross-DB.
   It never handles arbitrary URLs or filesystem paths; the backend accepts
   only pinned source ids.
   ============================================================ */
(function () {
  'use strict';

  const OPENED_SOURCE_KEY = 'easyicu_patient_official_demo_v1';
  const ACTIVE_JOB_KEY = 'easyicu_patient_official_demo_job_v1';
  const TERMINAL_JOB_STATES = new Set(['done', 'completed', 'failed', 'cancelled']);
  const state = {
    catalog: null,
    error: null,
    loading: false,
    promise: null,
    job: null,
    // A failed preparation belongs to its source's card. It is kept apart
    // from `error` (the catalog's own failure), which a catalog refresh clears.
    failure: null,
    pollTimer: null,
  };

  function sourceById(sourceId) {
    return ((state.catalog && state.catalog.sources) || [])
      .find(source => source && source.id === sourceId) || null;
  }

  function registryLabel(source) {
    if (!source) return '';
    return `${source.title || source.id} v${source.version || ''}`.trim();
  }

  function registeredSources(registryRows) {
    const rows = Array.isArray(registryRows) ? registryRows : [];
    return ((state.catalog && state.catalog.sources) || []).map(source => {
      const label = registryLabel(source);
      const registry = rows.find(row => {
        if (!row || !row.ok || !row.path) return false;
        if (row.source_kind && row.source_kind !== 'public_demo') return false;
        if ('source_id' in row || 'archive_sha256' in row) {
          const digest = (source.download || {}).sha256;
          return row.source_id === source.id && (row.source_kind === 'public_demo'
            || !digest || row.archive_sha256 === digest);
        }
        // Legacy registrations have no release receipt; keep their title match.
        return String(row.label || '').trim().toLowerCase() === label.toLowerCase();
      });
      return registry ? {
        database: source.database || registry.database || '',
        label,
        path: registry.path,
        source_id: source.id,
        title: source.title || source.id,
        version: source.version || '',
      } : null;
    }).filter(Boolean);
  }

  function rememberOpened(sourceId) {
    const source = sourceById(sourceId);
    if (!source || !source.status || !source.status.active) return null;
    const provenance = source.provenance || {};
    const license = provenance.license || {};
    const value = {
      source_id: source.id,
      title: source.title,
      version: source.version || '',
      provider: provenance.provider || 'PhysioNet',
      license: license.name || 'ODbL 1.0',
      registry_label: registryLabel(source),
    };
    if (window.setDataModeContext) {
      window.setDataModeContext({
        display_mode: 'demo',
        processing_mode: 'real',
        kind: 'official_demo',
        source_id: value.source_id,
        source_label: value.registry_label,
      });
    }
    try { localStorage.setItem(OPENED_SOURCE_KEY, JSON.stringify(value)); } catch (error) {}
    return source;
  }

  function rememberPair(registryRows) {
    const matched = registeredSources(registryRows);
    if (matched.length < 2) return [];
    const pair = matched.slice(0, 2);
    if (window.setDataModeContext) {
      window.setDataModeContext({
        display_mode: 'demo',
        processing_mode: 'real',
        kind: 'official_demo_pair',
        source_id: pair.map(source => source.source_id).join(','),
        source_label: pair.map(source => `${source.title} ${source.version}`).join(' + '),
      });
    }
    return pair;
  }

  function activeMetadata(registrySources, activePath) {
    const active = (Array.isArray(registrySources) ? registrySources : [])
      .find(source => source && source.path === activePath);
    if (!active) return null;
    try {
      const value = JSON.parse(localStorage.getItem(OPENED_SOURCE_KEY) || 'null');
      return value && value.source_id && active.label === value.registry_label ? value : null;
    } catch (error) {
      return null;
    }
  }

  function hT(helpers, en, zh) {
    return helpers && helpers.t ? helpers.t(en, zh) : en;
  }

  function hEsc(helpers, value) {
    if (helpers && helpers.esc) return helpers.esc(value);
    return String(value == null ? '' : value).replace(/[&<>"']/g, character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    })[character]);
  }

  function sourceStatus(source) {
    return (source && source.status) || {};
  }

  function activeJob() {
    return state.job && !TERMINAL_JOB_STATES.has(state.job.status)
      ? state.job
      : null;
  }

  // What a researcher needs from the status is whether the demo can be opened
  // and, if not, whether anything is left to download; the cache stages
  // (downloaded, converted, export from another build) all read as "finish
  // preparing".
  function stateLabel(source, helpers) {
    const status = sourceStatus(source);
    const value = String(status.state || 'not_downloaded');
    if (value === 'prepared' && !status.export_update_required) return [hT(helpers, 'Ready', '已就绪'), 'ok'];
    if (value === 'not_downloaded') {
      return status.resume_available
        ? [hT(helpers, 'Download paused', '下载已暂停'), 'warn']
        : [hT(helpers, 'Not downloaded', '未下载'), 'neutral'];
    }
    return [hT(helpers, 'Downloaded · preparation unfinished', '已下载，待完成准备'), 'warn'];
  }

  // The release's own description is English; the reader's language names it.
  const DESCRIPTIONS = {
    mimic_iv_demo_v2_2: ['Deidentified MIMIC-IV demonstration release from PhysioNet; enough to try every step from extraction to figures.', 'PhysioNet 官方发布的去标识化 MIMIC-IV 演示数据，可完整体验从抽取到出图的流程。'],
    eicu_demo_v2_0_1: ['Deidentified multi-centre eICU demonstration release from PhysioNet.', 'PhysioNet 官方发布的去标识化 eICU 多中心演示数据。'],
    mimic_iii_demo_v1_4: ['Deidentified MIMIC-III demonstration release from PhysioNet; the same extraction-to-figures workflow on the older MIMIC structure.', 'PhysioNet 官方发布的去标识化 MIMIC-III 演示数据，可在旧版 MIMIC 结构上体验从抽取到出图的流程。'],
  };
  function describe(source, helpers) {
    const row = DESCRIPTIONS[source && source.id];
    return row ? hT(helpers, row[0], row[1]) : String((source && source.description) || '');
  }

  function scopeText(source, helpers) {
    const scope = (source && source.scope) || {};
    if (scope.summary) return String(scope.summary);
    if (scope.patients) return `${scope.patients} ${hT(helpers, 'patients', '名患者')}`;
    if (scope.icu_stays) return `${scope.icu_stays} ${hT(helpers, 'ICU stays', '次 ICU 住院')}`;
    return hT(helpers, 'bounded public cohort', '公开演示队列');
  }

  function activeJobFor(source) {
    const job = activeJob();
    return job && job.sourceId === source.id ? job : null;
  }

  function rememberActiveJob(job) {
    if (!job || !job.id || !job.sourceId) return;
    try {
      localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify({
        id: job.id,
        sourceId: job.sourceId,
        openAfterPrepare: Boolean(job.openAfterPrepare),
      }));
    } catch (error) {}
  }

  function clearRememberedJob() {
    try { localStorage.removeItem(ACTIVE_JOB_KEY); } catch (error) {}
  }

  function clearPollTimer() {
    if (state.pollTimer != null) clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function rememberedJob() {
    try {
      const value = JSON.parse(localStorage.getItem(ACTIVE_JOB_KEY) || 'null');
      return value && value.id && value.sourceId ? value : null;
    } catch (error) {
      clearRememberedJob();
      return null;
    }
  }

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : null;
  }

  function formatBytes(value) {
    const bytes = finiteNumber(value);
    if (bytes == null) return '';
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${Math.round(bytes)} B`;
  }

  function formatEta(value, helpers) {
    const seconds = finiteNumber(value);
    if (seconds == null || seconds <= 0) return '';
    if (seconds < 60) {
      return hT(helpers, `about ${Math.max(1, Math.ceil(seconds))}s left`, `预计还需 ${Math.max(1, Math.ceil(seconds))} 秒`);
    }
    const minutes = Math.ceil(seconds / 60);
    return hT(helpers, `about ${minutes} min left`, `预计还需 ${minutes} 分钟`);
  }

  function jobProgress(job, helpers) {
    const event = job && job.event;
    if (!event || event.phase !== 'download') return null;
    const received = finiteNumber(event.bytes_received);
    const total = finiteNumber(event.bytes_total);
    if (received == null || total == null || total <= 0) return null;
    const boundedReceived = Math.min(received, total);
    const percent = Math.max(0, Math.min(100, (boundedReceived / total) * 100));
    const parts = [
      `${Math.floor(percent)}%`,
      `${formatBytes(boundedReceived)} / ${formatBytes(total)}`,
    ];
    const rate = finiteNumber(event.download_rate_bps);
    if (rate != null && rate > 0) parts.push(`${formatBytes(rate)}/s`);
    const eta = formatEta(event.eta_seconds, helpers);
    if (eta) parts.push(eta);
    return { percent, text: parts.join(' · ') };
  }

  function phaseLabels(helpers) {
    return {
      download: hT(helpers, 'Downloading official archive', '正在下载官方数据包'),
      extract: hT(helpers, 'Verifying and extracting', '正在校验并解压'),
      convert: hT(helpers, 'Converting to local Parquet', '正在转换为本地 Parquet'),
      export: hT(helpers, 'Mapping EasyICU feature modules', '正在映射 EasyICU 特征模块'),
      register: hT(helpers, 'Registering Patient Review source', '正在注册患者审阅数据源'),
      prepare: hT(helpers, 'Preparing source', '正在准备数据源'),
    };
  }

  function jobText(job, helpers) {
    const event = job && job.event;
    const phase = String((event && event.phase) || (event && event.type) || 'prepare');
    const stage = String((event && event.stage) || '');
    const labels = phaseLabels(helpers);
    if (phase === 'download') {
      if (stage === 'resuming' || (
        stage === 'streaming' && finiteNumber(event && event.resume_from_bytes) > 0
      )) {
        return hT(helpers, 'Resuming verified official download', '正在断点续传已校验的官方数据包');
      }
      if (stage === 'restarting') {
        return hT(helpers, 'Range unavailable; safely restarting', '服务器未接受续传，已安全重新下载');
      }
      if (stage === 'reused') {
        return hT(helpers, 'Using verified local archive', '正在复用已校验的本地缓存');
      }
    }
    const current = Number(event && event.current);
    const total = Number(event && event.total);
    const progress = Number.isFinite(current) && Number.isFinite(total) && total > 0
      ? ` · ${current}/${total}`
      : '';
    return `${labels[phase] || phase}${progress}`;
  }

  // The job error is `demo_source_<phase>_failed: {detail}` from
  // demo_source_prepare.lower_layer_diagnostic; keep its phase and the
  // lower-layer message, and fall back to the raw text for anything else.
  function failureFromJob(sourceId, raw) {
    const text = String(raw || '').trim();
    const match = text.match(/^(demo_source_[a-z_]+):\s*(\{[\s\S]*\})$/);
    let phase = '';
    let cause = text;
    if (match) {
      try {
        const payload = JSON.parse(match[2]);
        const detail = (payload && payload.detail) || {};
        const lower = detail.cause || {};
        phase = String(detail.phase || '');
        cause = String((lower.detail && lower.detail.message) || lower.code || match[1]);
      } catch (error) {
        cause = text;
      }
    }
    return { sourceId, phase, cause };
  }

  function failureText(failure, helpers) {
    if (failure.phase === 'download') {
      return hT(helpers, 'Could not download the archive from PhysioNet. Check the network or proxy, then try again.', '无法从 PhysioNet 下载数据包。请检查网络或代理后重试。');
    }
    const label = phaseLabels(helpers)[failure.phase];
    return label
      ? hT(helpers, `Preparation stopped at: ${label}. Try again.`, `准备在「${label}」这一步中断，请重试。`)
      : hT(helpers, 'Preparation did not finish. Try again.', '准备没有完成，请重试。');
  }

  function renderFailure(failure, helpers) {
    return `
          <div class="note warn official-demo-failure" role="alert" data-demo-source-failure="${hEsc(helpers, failure.sourceId)}">
            <div class="body">
              <div class="d">${hEsc(helpers, failureText(failure, helpers))}</div>
              ${failure.cause ? `<details><summary>${hEsc(helpers, hT(helpers, 'Details', '详情'))}</summary><span class="mono">${hEsc(helpers, failure.cause)}</span></details>` : ''}
            </div>
          </div>`;
  }

  function renderSource(source, helpers, options) {
    const status = sourceStatus(source);
    const [statusText, tone] = stateLabel(source, helpers);
    const provenance = (source && source.provenance) || {};
    const license = provenance.license || {};
    const download = (source && source.download) || {};
    const job = activeJobFor(source);
    const progress = jobProgress(job, helpers);
    const anotherJob = activeJob() && !job;
    const title = source.title || source.id;
    const isPrepared = status.state === 'prepared' && status.registered;
    const isActive = isPrepared && status.active;
    const canResume = Boolean(status.resume_available);
    const scope = String(options && options.scope || 'patient');
    const canOpen = scope === 'crossdb' ? isPrepared : isActive;
    const openLabel = scope === 'cohort'
      ? hT(helpers, 'Open cohort statistics', '打开队列统计')
      : scope === 'crossdb'
        ? hT(helpers, 'Use for comparison', '用于跨库对比')
        : hT(helpers, 'Open patient review', '打开患者审阅');
    const prepareLabel = anotherJob
      ? hT(helpers, 'Waiting for current preparation', '等待当前准备任务')
      : isPrepared
        ? (scope === 'crossdb'
          ? hT(helpers, 'Activate for comparison', '激活以用于对比')
          : hT(helpers, 'Activate and open', '激活并打开'))
        : status.archive_ready
          ? hT(helpers, 'Continue preparation', '继续准备')
          : canResume
            ? hT(helpers, 'Resume download and prepare', '继续下载并准备')
            : hT(helpers, 'Download and prepare', '下载并准备');
    return `
      <article class="official-demo-source ${isPrepared ? 'ready' : ''}" data-demo-source="${hEsc(helpers, source.id)}">
        <div class="official-demo-source-head">
          <div>
            <span class="official-demo-source-kicker">${hEsc(helpers, provenance.provider || 'PhysioNet')} · ${hEsc(helpers, source.version || '')}</span>
            <h3>${hEsc(helpers, title)}</h3>
          </div>
          <span class="pill ${tone === 'ok' ? 'ok' : tone === 'warn' ? 'warn' : ''}">${hEsc(helpers, statusText)}</span>
        </div>
        <p>${hEsc(helpers, describe(source, helpers))}</p>
        <div class="official-demo-source-facts">
          <span>${hEsc(helpers, scopeText(source, helpers))}</span>
          <span>${hEsc(helpers, download.size_label || '')}</span>
          ${canResume ? `<span>${hEsc(helpers, hT(helpers, `${formatBytes(status.partial_bytes)} saved`, `已保存 ${formatBytes(status.partial_bytes)}`))}</span>` : ''}
        </div>
        <div class="official-demo-source-provenance">
          <a href="${hEsc(helpers, provenance.landing_page || '#')}" target="_blank" rel="noopener noreferrer">${hEsc(helpers, hT(helpers, 'Official dataset page', '官方数据集页面'))}</a>
          <span>${hEsc(helpers, license.name || 'ODbL 1.0')}</span>
        </div>
        ${job ? `
          <div class="official-demo-job" role="status">
            <div class="official-demo-job-line">
              <span class="spin accent"></span>
              <span class="official-demo-job-copy">
                <b>${hEsc(helpers, jobText(job, helpers))}</b>
                ${progress ? `<span>${hEsc(helpers, progress.text)}</span>` : ''}
              </span>
            </div>
            ${progress ? `<progress class="official-demo-progress" max="100" value="${progress.percent.toFixed(2)}" aria-label="${hEsc(helpers, hT(helpers, 'Download progress', '下载进度'))}"></progress>` : ''}
          </div>` : ''}
        ${!job && state.failure && state.failure.sourceId === source.id ? renderFailure(state.failure, helpers) : ''}
        <div class="official-demo-source-actions">
          ${canOpen
            ? `<button class="btn primary" type="button" data-demo-source-open="${hEsc(helpers, source.id)}">${hEsc(helpers, openLabel)}</button>`
            : `<button class="btn ${source.id.indexOf('mimic') >= 0 ? 'primary' : ''}" type="button" data-demo-source-prepare="${hEsc(helpers, source.id)}" ${isPrepared ? 'data-demo-source-open-after-prepare="true"' : ''} ${job || anotherJob ? 'disabled' : ''}>${hEsc(helpers, prepareLabel)}</button>`}
        </div>
      </article>`;
  }

  function render(helpers = {}, options = {}) {
    if (state.loading && !state.catalog) {
      return `<div class="official-demo-loading"><span class="spin accent"></span>${hEsc(helpers, hT(helpers, 'Checking official demo sources…', '正在检查官方演示数据源…'))}</div>`;
    }
    // pendingOnly lists the demos still to prepare: a prepared or registered
    // one is already a row in the page's prepared data, and nothing left
    // means no section.
    const registered = new Set(registeredSources(options.registryRows).map(row => row.source_id));
    const sources = ((state.catalog && state.catalog.sources) || []).filter(source => {
      if (!options.pendingOnly) return true;
      const status = sourceStatus(source);
      return !(status.state === 'prepared' && status.registered) && !registered.has(source.id);
    });
    if (options.pendingOnly && !sources.length && !state.error) return '';
    return `
      <div class="official-demo-sources" data-official-demo-sources data-demo-source-scope="${hEsc(helpers, options.scope || 'patient')}">
        <div class="official-demo-sources-head">
          <div>
            <div class="official-demo-sources-title">${hEsc(helpers, hT(helpers, 'Official public demo data', '官方公开演示数据'))}</div>
            <p>${hEsc(helpers, hT(
              helpers,
              'Deidentified real records. They stay on this computer and go through the same conversion, mapping and export as your own data.',
              '去标识化的真实记录，只保存在本机，并和你自己的数据走同样的转换、映射和导出流程。',
            ))}</p>
          </div>
        </div>
        ${state.error ? `<div class="note warn"><div class="body"><div class="d">${hEsc(helpers, state.error)}</div></div><button class="btn sm" type="button" data-demo-source-retry>${hEsc(helpers, hT(helpers, 'Retry', '重试'))}</button></div>` : ''}
        <div class="official-demo-source-grid">
          ${sources.map(source => renderSource(source, helpers, options)).join('')}
        </div>
      </div>`;
  }

  function loadCatalog(callback, force = false) {
    const api = window.EU_API;
    if (!api || typeof api.loadOfficialDemoSources !== 'function') {
      state.error = 'Official demo-source API is unavailable.';
      if (callback) callback(false);
      return Promise.resolve(false);
    }
    if (!force && state.catalog) return Promise.resolve(true);
    if (!force && state.error) return Promise.resolve(false);
    if (state.promise) return state.promise;
    state.loading = true;
    state.error = null;
    state.promise = api.loadOfficialDemoSources()
      .then(payload => {
        state.catalog = payload || { sources: [] };
        state.loading = false;
        state.promise = null;
        if (callback) callback(true);
        return true;
      })
      .catch(error => {
        state.loading = false;
        state.promise = null;
        state.error = String((error && error.message) || error);
        if (callback) callback(false);
        return false;
      });
    return state.promise;
  }

  function latestJobEvent(snapshot) {
    const events = Array.isArray(snapshot && snapshot.events) ? snapshot.events : [];
    return events.length ? events[events.length - 1] : null;
  }

  function pollJob(jobId, sourceId, config, openAfterPrepare) {
    const api = window.EU_API;
    if (!api || typeof api.loadJobSnapshot !== 'function') return;
    api.loadJobSnapshot(jobId).then(snapshot => {
      const current = activeJob();
      if (current && current.id && (current.id !== jobId || current.sourceId !== sourceId)) return;
      const status = String((snapshot && snapshot.status) || 'running').toLowerCase();
      state.job = {
        id: jobId,
        sourceId,
        status,
        event: latestJobEvent(snapshot),
        openAfterPrepare: Boolean(openAfterPrepare),
      };
      if (!TERMINAL_JOB_STATES.has(status)) {
        clearPollTimer();
        state.pollTimer = setTimeout(
          () => pollJob(jobId, sourceId, config, openAfterPrepare),
          700,
        );
        if (config && config.refresh) config.refresh();
        return;
      }
      // Clear the reconnect pointer before repainting. A repaint immediately
      // rebinds this owner; leaving the pointer in storage until afterwards
      // would resurrect the just-finished job and create an endless poll loop.
      clearPollTimer();
      clearRememberedJob();
      const succeeded = ['done', 'completed'].includes(status);
      state.failure = succeeded ? null : failureFromJob(sourceId, snapshot && snapshot.error);
      state.catalog = null;
      if (config && config.refresh) config.refresh();
      loadCatalog(() => {
        if (config && config.refresh) config.refresh();
        if (succeeded && openAfterPrepare && config && config.openPrepared) {
          config.openPrepared(sourceId);
        }
      }, true);
    }).catch(error => {
      state.error = String((error && error.message) || error);
      state.job = null;
      clearPollTimer();
      clearRememberedJob();
      if (config && config.refresh) config.refresh();
    });
  }

  function resumeRememberedJob(config) {
    if (activeJob()) return;
    const job = rememberedJob();
    if (!job) return;
    state.job = {
      id: String(job.id),
      sourceId: String(job.sourceId),
      status: 'running',
      event: { phase: 'prepare' },
      openAfterPrepare: Boolean(job.openAfterPrepare),
    };
    if (config && config.refresh) config.refresh();
    pollJob(state.job.id, state.job.sourceId, config, state.job.openAfterPrepare);
  }

  function prepare(sourceId, config, openAfterPrepare = false) {
    const api = window.EU_API;
    if (!api || typeof api.startOfficialDemoSourcePrepare !== 'function') return;
    if (activeJob()) return;
    clearPollTimer();
    state.error = null;
    state.failure = null;
    state.job = {
      sourceId,
      status: 'running',
      event: { phase: 'prepare' },
      openAfterPrepare: Boolean(openAfterPrepare),
    };
    if (config && config.refresh) config.refresh();
    api.startOfficialDemoSourcePrepare(sourceId).then(response => {
      if (!response || !response.job_id) throw new Error('Demo prepare job did not return an id.');
      state.job.id = String(response.job_id);
      rememberActiveJob(state.job);
      pollJob(response.job_id, sourceId, config, openAfterPrepare);
    }).catch(error => {
      state.failure = failureFromJob(sourceId, (error && error.message) || error);
      state.job = null;
      clearRememberedJob();
      if (config && config.refresh) config.refresh();
    });
  }

  function bind(root, config = {}) {
    if (!root || typeof root.querySelectorAll !== 'function') return;
    resumeRememberedJob(config);
    root.querySelectorAll('[data-demo-source-retry]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        state.error = null;
        loadCatalog(() => {
          if (config.refresh) config.refresh();
        }, true);
        if (config.refresh) config.refresh();
      });
    });
    root.querySelectorAll('[data-demo-source-prepare]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        prepare(
          button.getAttribute('data-demo-source-prepare'),
          config,
          button.getAttribute('data-demo-source-open-after-prepare') === 'true',
        );
      });
    });
    root.querySelectorAll('[data-demo-source-open]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        if (config.openPrepared) config.openPrepared(button.getAttribute('data-demo-source-open'));
      });
    });
  }

  const owner = {
    activeMetadata,
    describe,
    stateLabel,
    bind,
    ensureLoaded: loadCatalog,
    rememberOpened,
    rememberPair,
    registeredSources,
    registryLabel,
    render,
    source: sourceById,
    snapshot: () => ({
      catalog: state.catalog,
      error: state.error,
      failure: state.failure,
      job: state.job,
      loading: state.loading,
    }),
  };
  window.EU_OFFICIAL_DEMO_SOURCES = owner;
  window.EU_PATIENT_DEMO_SOURCES = owner;
})();
