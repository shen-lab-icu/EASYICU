/* Cross-DB job progress + cooperative-cancel presentation owner. */
(function () {
  'use strict';

  const state = {
    jobId: '',
    progress: null,
    progressBeforeCancel: null,
    cancelRequested: false,
    cancelSent: false,
    starting: false,
    jobKind: '',
    databaseOrder: [],
    databaseStates: Object.create(null),
  };

  function clear() {
    state.jobId = '';
    state.progress = null;
    state.progressBeforeCancel = null;
    state.cancelRequested = false;
    state.cancelSent = false;
    state.starting = false;
    state.jobKind = '';
    state.databaseOrder = [];
    state.databaseStates = Object.create(null);
  }

  function beginStart() {
    clear();
    state.starting = true;
  }

  function appliesTo(jobId) {
    return !!jobId && state.jobId === String(jobId);
  }

  function rememberDatabases(event) {
    const databases = Array.isArray(event && event.databases) ? event.databases : [];
    databases.forEach(database => {
      const key = String(database || '').trim();
      if (!key || state.databaseOrder.includes(key)) return;
      state.databaseOrder.push(key);
      state.databaseStates[key] = { database: key, status: 'pending' };
    });
    const database = String(event && event.database || '').trim();
    if (database && !state.databaseOrder.includes(database)) {
      state.databaseOrder.push(database);
      state.databaseStates[database] = { database, status: 'pending' };
    }
  }

  function applyProgress(event) {
    if (!event || typeof event !== 'object' || state.cancelRequested) return false;
    if (event.job_kind) state.jobKind = String(event.job_kind);
    rememberDatabases(event);
    state.progress = { ...event };
    const database = String(event.database || '').trim();
    if (database) {
      const previous = state.databaseStates[database] || { database, status: 'pending' };
      state.databaseStates[database] = {
        ...previous,
        database,
        label: String(event.database_label || previous.label || database),
        status: String(event.database_status || previous.status || 'loading'),
        chunkCurrent: Number(event.chunk_current || previous.chunkCurrent || 0),
        chunkTotal: Number(event.chunk_total || previous.chunkTotal || 0),
      };
    }
    return true;
  }

  function attach(jobId, progress) {
    state.jobId = String(jobId || '');
    state.starting = false;
    if (!state.cancelRequested) applyProgress(progress || { phase: 'queued' });
    return !!state.jobId;
  }

  function beginProbe(jobId, progress) {
    clear();
    state.jobId = String(jobId || '');
    state.starting = true;
    applyProgress(progress || { phase: 'reconnect' });
  }

  function resume(jobId, event, history) {
    state.jobId = String(jobId || '');
    state.starting = false;
    if (event && event.job_kind) state.jobKind = String(event.job_kind);
    if (Array.isArray(history) && history.length) {
      state.cancelRequested = false;
      state.cancelSent = false;
      state.progress = null;
      state.progressBeforeCancel = null;
      state.databaseOrder = [];
      state.databaseStates = Object.create(null);
      history.forEach(row => {
        if (row && row.type === 'progress') applyProgress(row);
        if (row && row.type === 'cancel_requested') applyCancelRequested(row);
      });
      return;
    }
    if (event && event.type === 'cancel_requested') {
      applyCancelRequested(event);
    } else {
      applyProgress(event || { phase: 'reconnect' });
    }
  }

  function applyCancelRequested(event) {
    if (!state.progressBeforeCancel && state.progress) state.progressBeforeCancel = { ...state.progress };
    state.cancelRequested = true;
    state.cancelSent = true;
    state.progress = {
      type: 'cancel_requested',
      phase: 'cancel',
      reason: String(event && event.reason || 'user_requested'),
    };
    return true;
  }

  function flushCancel(api, onError) {
    if (!state.cancelRequested || state.cancelSent || !state.jobId) return false;
    if (!api || typeof api.cancelJob !== 'function') {
      state.cancelRequested = false;
      state.cancelSent = false;
      state.progress = state.progressBeforeCancel;
      state.progressBeforeCancel = null;
      if (typeof onError === 'function') onError(new Error('Cross-DB cancel API unavailable'));
      return false;
    }
    state.cancelSent = true;
    api.cancelJob(state.jobId, 'user_requested').catch(error => {
      state.cancelRequested = false;
      state.cancelSent = false;
      state.progress = state.progressBeforeCancel;
      state.progressBeforeCancel = null;
      if (typeof onError === 'function') onError(error);
    });
    return true;
  }

  function requestCancel(options) {
    const opts = options || {};
    if (state.cancelRequested) return false;
    state.progressBeforeCancel = state.progress ? { ...state.progress } : null;
    state.cancelRequested = true;
    state.progress = { type: 'cancel_requested', phase: 'cancel', reason: 'user_requested' };
    if (typeof opts.onStateChange === 'function') opts.onStateChange();
    flushCancel(opts.api, opts.onError);
    return true;
  }

  function snapshot() {
    return {
      jobId: state.jobId,
      progress: state.progress ? { ...state.progress } : null,
      cancelRequested: state.cancelRequested,
      cancelSent: state.cancelSent,
      starting: state.starting,
      databases: state.databaseOrder.map(database => ({
        ...(state.databaseStates[database] || { database, status: 'pending' }),
      })),
    };
  }

  function render(options) {
    const opts = options || {};
    const esc = opts.esc || (value => String(value == null ? '' : value));
    const t = opts.t || ((english) => english);
    const fmtInt = opts.fmtInt || (value => String(value == null ? '—' : value));
    const progressMessage = opts.progressMessage || (value => String(value || ''));
    const statusLabel = opts.statusLabel || (value => String(value || ''));
    const icon = opts.icon || (() => '');
    const errorMessage = String(opts.errorMessage || '').trim();
    const p = state.progress || {};
    const registeredSummary = state.jobKind === 'crossdb-summary';
    const completedChunks = Number(p.completed_chunks || 0);
    const totalChunks = Number(p.total_chunks || 0);
    const completedDatabases = Number(p.current || 0);
    const databaseTotal = Number(p.database_total || p.total || state.databaseOrder.length || 0);
    const progressCurrent = totalChunks ? completedChunks : completedDatabases;
    const progressTotal = totalChunks || databaseTotal;
    const pct = progressTotal ? Math.max(0, Math.min(100, Math.round((progressCurrent / progressTotal) * 100))) : 0;
    const sample = opts.sampleProfile || {};
    const sampleText = sample.maxPatients && sample.sampleSize
      ? ` · ≤${fmtInt(sample.maxPatients)} ${t('entities/db', '实体/库')} · ≤${fmtInt(sample.sampleSize)} ${t('values/feature', '值/特征')}`
      : '';
    const currentDatabase = String(p.database_label || p.database || '').trim();
    const chunkCurrent = Number(p.chunk_current || 0);
    const chunkTotal = Number(p.chunk_total || 0);
    const detail = state.cancelRequested
      ? t('Cancellation requested. EasyICU will stop after the current bounded read returns.', '已请求取消。当前有界读取返回后，EasyICU 将立即停止。')
      : currentDatabase
        ? `${currentDatabase}${chunkTotal ? ` · ${t('chunk', '分块')} ${chunkCurrent || 1}/${chunkTotal}` : ''}`
        : progressMessage(p.message || (registeredSummary
          ? t('Starting registered-export Cross-DB summary job…', '正在启动注册导出的跨库摘要任务…')
          : t('Starting local raw Cross-DB density job…', '正在启动本地原始跨库密度任务…')));
    const rows = snapshot().databases;
    const rowHtml = rows.length ? `<ol class="crossdb-progress-databases" aria-label="${esc(t('Database progress', '数据库进度'))}" aria-live="off">
      ${rows.map(row => {
        const status = state.cancelRequested && row.status === 'loading' ? 'stopping' : row.status;
        const chunk = row.chunkTotal ? `${row.chunkCurrent || 1}/${row.chunkTotal}` : '';
        return `<li class="crossdb-progress-db is-${esc(status)}">
          <span class="crossdb-progress-db-name">${esc(row.label || row.database)}</span>
          <span class="crossdb-progress-db-chunk mono">${esc(chunk)}</span>
          <span class="crossdb-progress-db-status">${esc(statusLabel(status))}</span>
        </li>`;
      }).join('')}
    </ol>` : '';

    return `<div class="card pad crossdb-progress-card">
      <div class="load-strip">
        <span class="spin accent" aria-hidden="true"></span>
        <div class="grow">
          <div class="crossdb-progress-title">${esc(registeredSummary
            ? t('Loading aggregate summaries from registered exports…', '正在加载已注册导出的聚合摘要…')
            : t('Loading real feature densities from local databases…', '正在从本地数据库加载真实特征密度…'))}</div>
          <div class="crossdb-progress-meta mono">${esc(t('local-only · nothing uploaded', '仅本机 · 不上传任何内容'))}${p.phase ? ` · ${esc(statusLabel(p.phase))}` : ''}${sampleText}</div>
        </div>
        ${databaseTotal ? `<span class="crossdb-progress-count mono">${completedDatabases}/${databaseTotal}</span>` : ''}
        <button class="btn sm" type="button" data-crossdb-cancel ${state.cancelRequested ? 'aria-disabled="true"' : ''}>${icon('stop', 13)} ${state.cancelRequested ? t('Cancel requested', '已请求取消') : t('Cancel', '取消')}</button>
      </div>
      ${progressTotal ? `<progress class="crossdb-progress-bar" max="100" value="${pct}" aria-label="${esc(t('Cross-DB loading progress', '跨库加载进度'))}">${pct}%</progress>` : '<div class="indet mt-12"></div>'}
      <div class="crossdb-progress-detail" role="status" aria-live="polite" aria-atomic="true">${esc(detail)}</div>
      ${errorMessage ? `<div class="note warn mt-12 crossdb-progress-error" role="alert" aria-live="assertive"><div class="ico">${icon('alert', 14)}</div><div class="body"><div class="d mono">${esc(errorMessage)}</div></div></div>` : ''}
      ${rowHtml}
    </div>`;
  }

  window.EU_CROSSDB_PROGRESS = {
    appliesTo,
    applyCancelRequested,
    applyProgress,
    attach,
    beginProbe,
    beginStart,
    clear,
    flushCancel,
    render,
    requestCancel,
    resume,
    snapshot,
  };
})();
