/* Owner: Data Extraction job continuity widget. */
/* Data Extraction long-job continuity owner.
   Persists only bounded reconnect metadata; job progress/results always come
   back from the local FastAPI job snapshot or SSE stream. */
(function () {
  'use strict';

  const STORAGE_KEY = 'easyicu.extractionJob.v1';
  const JOB_ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
  const TERMINAL = new Set(['done', 'failed', 'cancelled']);
  const MAX_PATH = 4096;
  const MAX_DATABASE = 64;
  const MAX_MODULES = 64;
  const MAX_MODULE = 128;
  const MAX_STORAGE_CHARS = 16384;

  let generation = 0;
  let active = null;
  let running = false;
  let stream = null;
  let reconnectAttempts = 0;

  function boundedText(value, max) {
    return typeof value === 'string' ? value.slice(0, max) : '';
  }

  function cleanSource(value) {
    const source = value && typeof value === 'object' ? value : {};
    const path = boundedText(source.path, MAX_PATH).trim();
    if (!path) return null;
    return {
      path,
      database: boundedText(source.database, MAX_DATABASE).trim(),
    };
  }

  function cleanConfig(kind, value) {
    const config = value && typeof value === 'object' ? value : {};
    if (kind === 'convert') return {};
    const modules = Array.isArray(config.modules)
      ? config.modules
          .filter(module => typeof module === 'string')
          .slice(0, MAX_MODULES)
          .map(module => module.slice(0, MAX_MODULE))
      : [];
    const concepts = Object.fromEntries(modules.filter(module => Array.isArray(config.concepts?.[module]))
      .map(module => [module, config.concepts[module].slice(0, MAX_MODULES)
        .filter(id => typeof id === 'string' && id.length <= MAX_MODULE)]));
    const maxPatients = Number(config.max_patients);
    return {
      run_mode: config.run_mode === 'recommended' ? 'recommended' : 'custom',
      modules,
      ...(Object.keys(concepts).length ? { concepts } : {}),
      format: ['parquet', 'csv', 'excel'].includes(config.format) ? config.format : 'parquet',
      merge: config.merge === true,
      max_patients: Number.isFinite(maxPatients)
        ? Math.max(0, Math.min(10000000, Math.trunc(maxPatients)))
        : 0,
      out_dir: boundedText(config.out_dir, MAX_PATH),
    };
  }

  function cleanRecord(value, requireJobId) {
    if (!value || typeof value !== 'object') return null;
    const kind = value.kind === 'extract' || value.kind === 'convert' ? value.kind : null;
    const source = cleanSource(value.source);
    const jobId = boundedText(value.job_id, 64);
    if (!kind || !source || (requireJobId && !JOB_ID_RE.test(jobId))) return null;
    const record = {
      job_id: requireJobId ? jobId : '',
      kind,
      source,
      config: cleanConfig(kind, value.config),
    };
    return JSON.stringify(record).length <= MAX_STORAGE_CHARS ? record : null;
  }

  function readRecord() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw || raw.length > MAX_STORAGE_CHARS) return null;
      return cleanRecord(JSON.parse(raw), true);
    } catch (e) {
      return null;
    }
  }

  function writeRecord(record) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(record)); } catch (e) {}
  }

  function clearRecord() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
  }

  function closeStream() {
    if (!stream) return;
    try {
      stream.onmessage = null;
      stream.onerror = null;
      stream.close();
    } catch (e) {}
    stream = null;
  }

  function host() {
    return window.EU_EXTRACTION_JOB_HOST || null;
  }

  function continuityMessage(record, missing, error, t) {
    const kind = record && record.kind === 'convert' ? t('conversion', '转换') : t('extraction', '抽取');
    if (missing) {
      return t(
        `The saved ${kind} task is no longer available. The local server was restarted or its job history expired; start the task again. EasyICU did not infer a successful result.`,
        `已保存的${kind}任务已不可用。本地服务可能已重启，或任务历史已过期；请重新开始。EasyICU 没有推断任务成功。`
      );
    }
    const detail = error && error.message ? ` (${error.message})` : '';
    return t(
      `EasyICU could not verify the ${kind} task${detail}. Refresh to reconnect; this screen will not mark it complete without a terminal server result.`,
      `EasyICU 暂时无法核验${kind}任务${detail}。请刷新页面重新连接；没有服务端终态，本页面不会把任务标记为完成。`
    );
  }

  function isCurrent(record, requestGeneration) {
    return requestGeneration === generation && active &&
      active.job_id === record.job_id && active.kind === record.kind;
  }

  function surfaceMissing(record) {
    closeStream();
    clearRecord();
    generation += 1;
    active = null;
    running = false;
    const target = host();
    if (target && target.missing) target.missing(record);
  }

  function surfaceConnectionLoss(record, error) {
    const target = host();
    if (target && target.connectionLost) target.connectionLost(record, error);
  }

  function applySnapshot(record, snapshot, requestGeneration) {
    if (!isCurrent(record, requestGeneration)) return false;
    if (!snapshot || snapshot.id !== record.job_id || snapshot.kind !== record.kind ||
        !['running', 'done', 'failed', 'cancelled'].includes(snapshot.status)) {
      surfaceMissing(record);
      return false;
    }
    const target = host();
    if (!target || !target.applyEvent) return false;
    running = snapshot.status === 'running';
    (Array.isArray(snapshot.events) ? snapshot.events : []).forEach(event => {
      if (event && event.type !== 'end' && isCurrent(record, requestGeneration)) {
        target.applyEvent(record, event);
      }
    });
    if (TERMINAL.has(snapshot.status)) {
      target.applyEvent(record, {
        type: 'end',
        status: snapshot.status,
        result: snapshot.result,
        error: snapshot.error,
      });
      closeStream();
      return true;
    }
    connect(record, requestGeneration);
    return true;
  }

  function reconcile(record, requestGeneration) {
    const api = window.EU_API;
    if (!api || !api.loadJobSnapshot) {
      surfaceConnectionLoss(record, new Error('job snapshot API unavailable'));
      return Promise.resolve(false);
    }
    return api.loadJobSnapshot(record.job_id)
      .then(snapshot => applySnapshot(record, snapshot, requestGeneration))
      .catch(error => {
        if (!isCurrent(record, requestGeneration)) return false;
        // err.status, not the message. api.js puts the backend's human reason
        // in .message for 4xx and the "path -> HTTP 404" string in .technical,
        // and this route answers 404 with detail="unknown job" — so matching
        // /HTTP 404/ against .message never fired, and a job the server had
        // already forgotten was reported to the user as a dropped connection
        // they could resume.
        if (error && error.status === 404) {
          surfaceMissing(record);
        } else {
          closeStream();
          surfaceConnectionLoss(record, error);
        }
        return false;
      });
  }

  function connect(record, requestGeneration) {
    if (!isCurrent(record, requestGeneration)) return;
    closeStream();
    if (!window.EventSource) {
      surfaceConnectionLoss(record, new Error('event stream unavailable'));
      return;
    }
    const next = new EventSource('/api/jobs/' + encodeURIComponent(record.job_id) + '/events');
    stream = next;
    next.onmessage = event => {
      if (!isCurrent(record, requestGeneration) || stream !== next) return;
      let message;
      try { message = JSON.parse(event.data); } catch (e) { return; }
      const target = host();
      if (target && target.applyEvent) target.applyEvent(record, message);
      if (message && message.type === 'end') {
        running = false;
        closeStream();
      }
    };
    next.onerror = () => {
      if (!isCurrent(record, requestGeneration) || stream !== next) return;
      closeStream();
      reconnectAttempts += 1;
      if (reconnectAttempts > 3) {
        surfaceConnectionLoss(record, new Error('event stream reconnect limit reached'));
        return;
      }
      const delay = [250, 1000, 2500][reconnectAttempts - 1];
      setTimeout(() => {
        if (isCurrent(record, requestGeneration)) reconcile(record, requestGeneration);
      }, delay);
    };
  }

  function abandon() {
    generation += 1;
    closeStream();
    reconnectAttempts = 0;
    active = null;
    running = false;
    clearRecord();
  }

  function prepare(value) {
    abandon();
    const pending = cleanRecord(value, false);
    if (!pending) return null;
    return { generation, kind: pending.kind, source: pending.source, config: pending.config };
  }

  function attach(ticket, jobId) {
    const record = cleanRecord({
      job_id: jobId,
      kind: ticket && ticket.kind,
      source: ticket && ticket.source,
      config: ticket && ticket.config,
    }, true);
    if (!ticket || ticket.generation !== generation || !record) {
      if (JOB_ID_RE.test(String(jobId || '')) && window.EU_API && window.EU_API.cancelJob) {
        window.EU_API.cancelJob(jobId, 'source_changed_before_tracking').catch(() => null);
      }
      return null;
    }
    active = record;
    running = true;
    reconnectAttempts = 0;
    writeRecord(record);
    const target = host();
    if (target && target.begin) target.begin(record);
    connect(record, generation);
    return record;
  }

  function restore() {
    const record = readRecord();
    if (!record) {
      clearRecord();
      return Promise.resolve(false);
    }
    generation += 1;
    closeStream();
    active = record;
    running = true;
    reconnectAttempts = 0;
    writeRecord(record);
    const target = host();
    if (target && target.begin) target.begin(record);
    return reconcile(record, generation);
  }

  const owner = {
    capture: () => ({ generation }),
    prepare,
    attach,
    restore,
    abandon,
    continuityMessage,
    isPending: ticket => !!ticket && ticket.generation === generation,
    isRunning: () => !!(active && running),
    active: () => active ? JSON.parse(JSON.stringify(active)) : null,
  };
  window.EU_EXTRACTION_JOB_CONTINUITY = owner;
  owner.ready = Promise.resolve().then(restore);
})();
