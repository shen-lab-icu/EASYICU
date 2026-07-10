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
    const maxPatients = Number(config.max_patients);
    return {
      run_mode: config.run_mode === 'recommended' ? 'recommended' : 'custom',
      modules,
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

  function isCurrent(record, requestGeneration) {
    return requestGeneration === generation && active &&
      active.job_id === record.job_id && active.kind === record.kind;
  }

  function surfaceMissing(record) {
    closeStream();
    clearRecord();
    generation += 1;
    active = null;
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
        if (/HTTP\s+404\b/.test(String(error && error.message || error))) {
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
      if (message && message.type === 'end') closeStream();
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
    reconnectAttempts = 0;
    writeRecord(record);
    const target = host();
    if (target && target.begin) target.begin(record);
    return reconcile(record, generation);
  }

  const owner = {
    prepare,
    attach,
    restore,
    abandon,
    isPending: ticket => !!ticket && ticket.generation === generation,
    active: () => active ? JSON.parse(JSON.stringify(active)) : null,
  };
  window.EU_EXTRACTION_JOB_CONTINUITY = owner;
  owner.ready = Promise.resolve().then(restore);
})();
