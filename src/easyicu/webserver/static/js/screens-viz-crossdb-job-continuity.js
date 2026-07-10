/* Cross-DB raw-distribution job continuity: bounded local metadata + reconnect. */
(function () {
  'use strict';

  const STORAGE_KEY = 'easyicu_crossdb_raw_job_v1';
  const JOB_KIND = 'crossdb-raw-distribution';
  const SAMPLE_MODES = new Set(['quick', 'standard', 'deeper']);
  const JOB_ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
  const SOURCE_ID_RE = /^[a-z0-9_-]+(?:,[a-z0-9_-]+)*$/;
  let stream = null;
  let activeJobId = '';
  let restoreAttempted = false;
  let completion = null;

  function host() {
    return window.EU_CROSSDB_JOB_HOST || null;
  }

  function removeStored() {
    try { window.localStorage.removeItem(STORAGE_KEY); } catch (error) {}
  }

  function normalizeMeta(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const jobId = String(value.job_id || '').trim();
    const kind = String(value.kind || '').trim();
    const rawRoot = String(value.raw_root || '').trim();
    const sourceIdentity = String(value.source_identity || '').trim();
    const sampleMode = String(value.sample_mode || '').trim();
    if (!JOB_ID_RE.test(jobId) || kind !== JOB_KIND) return null;
    if (!rawRoot || rawRoot.length > 4096 || /[\u0000-\u001f]/.test(rawRoot)) return null;
    if (!sourceIdentity || sourceIdentity.length > 256 || !SOURCE_ID_RE.test(sourceIdentity)) return null;
    if (!SAMPLE_MODES.has(sampleMode)) return null;
    return {
      job_id: jobId,
      kind: JOB_KIND,
      raw_root: rawRoot,
      source_identity: sourceIdentity,
      sample_mode: sampleMode,
    };
  }

  function readStored() {
    let raw = '';
    try { raw = window.localStorage.getItem(STORAGE_KEY) || ''; } catch (error) { return null; }
    if (!raw || raw.length > 8192) {
      if (raw) removeStored();
      return null;
    }
    try {
      const meta = normalizeMeta(JSON.parse(raw));
      if (!meta) removeStored();
      return meta;
    } catch (error) {
      removeStored();
      return null;
    }
  }

  function writeStored(meta) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(meta));
      return true;
    } catch (error) {
      return false;
    }
  }

  function closeStream() {
    if (stream) {
      try { stream.close(); } catch (error) {}
    }
    stream = null;
  }

  function stillCurrent(meta) {
    const stored = readStored();
    return !!(stored && stored.job_id === meta.job_id && stored.raw_root === meta.raw_root && stored.source_identity === meta.source_identity);
  }

  function latestProgress(events) {
    const rows = Array.isArray(events) ? events : [];
    for (let index = rows.length - 1; index >= 0; index -= 1) {
      const event = rows[index] || {};
      if (event.type === 'progress' || event.type === 'cancel_requested') return event;
    }
    return null;
  }

  function finish(meta, status, result, error) {
    if (!stillCurrent(meta)) return false;
    const target = host();
    if (!target || (typeof target.matchesSource === 'function' && !target.matchesSource(meta))) {
      disconnect({ forget: true });
      return false;
    }
    closeStream();
    activeJobId = '';
    const applied = typeof target.onTerminal === 'function'
      ? target.onTerminal(meta, { id: meta.job_id, kind: JOB_KIND, status, result: result || null, error: error || null })
      : true;
    const callback = completion;
    completion = null;
    if (callback) callback(status === 'done' && applied !== false);
    if (applied === false) removeStored();
    return applied !== false;
  }

  function applyEvent(meta, event) {
    if (!event || activeJobId !== meta.job_id || !stillCurrent(meta)) return;
    const target = host();
    if (!target) return;
    if (event.type === 'progress' && typeof target.onProgress === 'function') {
      target.onProgress(meta, event);
    } else if (event.type === 'cancel_requested' && typeof target.onCancelRequested === 'function') {
      target.onCancelRequested(meta, event);
    } else if (event.type === 'end') {
      finish(meta, event.status, event.result, event.error);
    }
  }

  function openStream(meta) {
    closeStream();
    if (typeof window.EventSource !== 'function') {
      const target = host();
      if (target && typeof target.onConnectionError === 'function') target.onConnectionError(meta);
      const callback = completion;
      completion = null;
      if (callback) callback(false);
      return false;
    }
    activeJobId = meta.job_id;
    const current = new window.EventSource('/api/jobs/' + encodeURIComponent(meta.job_id) + '/events');
    stream = current;
    current.onmessage = event => {
      let payload;
      try { payload = JSON.parse(event.data); } catch (error) { return; }
      applyEvent(meta, payload);
    };
    current.onerror = () => {
      if (stream !== current) return;
      closeStream();
      probe(meta, true);
    };
    return true;
  }

  async function probe(meta, reconnecting) {
    const api = window.EU_API;
    const target = host();
    if (!stillCurrent(meta) || !target) return false;
    if (!api || typeof api.loadJobSnapshot !== 'function') {
      if (typeof target.onConnectionError === 'function') target.onConnectionError(meta);
      return false;
    }
    let snapshot;
    try {
      snapshot = await api.loadJobSnapshot(meta.job_id);
    } catch (error) {
      if (!stillCurrent(meta)) return false;
      const missing = /HTTP\s+404\b/.test(String(error && error.message || error));
      const callback = completion;
      if (missing) {
        disconnect({ forget: true });
        if (typeof target.onUnavailable === 'function') target.onUnavailable(meta);
      } else if (typeof target.onConnectionError === 'function') {
        target.onConnectionError(meta, reconnecting);
      }
      completion = null;
      if (callback) callback(false);
      return false;
    }
    if (!stillCurrent(meta)) return false;
    if (!snapshot || snapshot.id !== meta.job_id || snapshot.kind !== JOB_KIND) {
      disconnect({ forget: true });
      if (typeof target.onUnavailable === 'function') target.onUnavailable(meta);
      return false;
    }
    if (snapshot.status === 'running') {
      if (typeof target.onRunning === 'function') target.onRunning(meta, latestProgress(snapshot.events));
      openStream(meta);
      return true;
    }
    if (snapshot.status === 'done' || snapshot.status === 'failed' || snapshot.status === 'cancelled') {
      return finish(meta, snapshot.status, snapshot.result, snapshot.error);
    }
    disconnect({ forget: true });
    if (typeof target.onUnavailable === 'function') target.onUnavailable(meta);
    return false;
  }

  function restoreIfNeeded() {
    if (restoreAttempted) return Promise.resolve(false);
    const meta = readStored();
    const target = host();
    if (!meta || !target || (typeof target.canRestore === 'function' && !target.canRestore())) return Promise.resolve(false);
    restoreAttempted = true;
    if (typeof target.acceptResume === 'function' && !target.acceptResume(meta)) {
      disconnect({ forget: true });
      return Promise.resolve(false);
    }
    if (typeof target.onProbe === 'function') target.onProbe(meta);
    return probe(meta, false);
  }

  function start(value, progress, done) {
    const meta = normalizeMeta(value);
    if (!meta) return false;
    disconnect({ forget: true });
    if (!writeStored(meta)) return false;
    const target = host();
    if (!target || (typeof target.acceptResume === 'function' && !target.acceptResume(meta))) {
      disconnect({ forget: true });
      return false;
    }
    restoreAttempted = true;
    completion = typeof done === 'function' ? done : null;
    if (typeof target.onRunning === 'function') target.onRunning(meta, progress || null);
    return openStream(meta);
  }

  function onSourceChanged(rawRoot, sourceIdentity, sampleMode) {
    const meta = readStored();
    if (!meta) return;
    const nextRoot = String(rawRoot || '').trim();
    const nextIdentity = String(sourceIdentity || '').trim();
    const nextSampleMode = String(sampleMode || '').trim();
    if (nextRoot !== meta.raw_root
        || (nextIdentity && nextIdentity !== meta.source_identity)
        || (nextSampleMode && nextSampleMode !== meta.sample_mode)) {
      disconnect({ forget: true });
    }
  }

  function disconnect(options) {
    const opts = options || {};
    closeStream();
    activeJobId = '';
    completion = null;
    if (opts.forget) removeStored();
  }

  window.EU_CROSSDB_JOB_CONTINUITY = {
    disconnect,
    onSourceChanged,
    restoreIfNeeded,
    start,
  };
})();
