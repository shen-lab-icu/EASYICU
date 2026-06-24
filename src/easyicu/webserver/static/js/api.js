/* EasyICU real-data bridge.
   Replaces the mock catalog from data-catalog.js with live data served by the
   FastAPI backend (/api/catalog). Loaded after data-catalog.js, so window.EU_CATALOG
   already exists as a mock fallback; on success we merge real values in and
   re-render the current screen. This is the seam the migration grows along:
   each screen's data source moves from mock -> /api/* here. */
(function () {
  'use strict';

  window.EU_API = window.EU_API || {};

  async function getJSON(path) {
    const res = await fetch(path, { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(path + ' -> HTTP ' + res.status);
    return res.json();
  }

  async function postJSON(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let detail = '';
      try {
        const payload = await res.json();
        const d = payload && payload.detail;
        detail = typeof d === 'string' ? d : (d && (d.error || JSON.stringify(d)));
      } catch (e) {}
      throw new Error(path + ' -> HTTP ' + res.status + (detail ? ' · ' + detail : ''));
    }
    return res.json();
  }
  async function postBlob(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let detail = '';
      try {
        const payload = await res.json();
        const d = payload && payload.detail;
        detail = typeof d === 'string' ? d : (d && (d.error || JSON.stringify(d)));
      } catch (e) {}
      throw new Error(path + ' -> HTTP ' + res.status + (detail ? ' · ' + detail : ''));
    }
    const disposition = res.headers.get('content-disposition') || '';
    const match = disposition.match(/filename="([^"]+)"/i);
    return { blob: await res.blob(), filename: match ? match[1] : 'download.bin' };
  }

  async function hydrateSettings() {
    window.EU_SETTINGS = await getJSON('/api/settings');
    // keep the UI language in sync with persisted setting
    if (window.EU_SETTINGS.language) window.EU_LANG = window.EU_SETTINGS.language;
    return window.EU_SETTINGS;
  }

  // Persist a single setting; updates the local cache from the server reply.
  async function saveSetting(key, value) {
    const patch = {}; patch[key] = value;
    window.EU_SETTINGS = await postJSON('/api/settings', patch);
    return window.EU_SETTINGS;
  }

  async function hydrateCatalog() {
    const real = await getJSON('/api/catalog');
    const cat = (window.EU_CATALOG = window.EU_CATALOG || {});
    // Real backend owns these fields now; keep any mock-only extras (auditModules).
    cat.groups = real.groups;
    cat.groupConcepts = real.groupConcepts;
    cat.dict = real.dict;
    cat.cov = real.cov;
    cat.desc = real.desc;
    cat.totalConcepts = real.totalConcepts;
    cat.supportedDbs = real.supportedDbs;
    cat.__live = true;
    return cat;
  }

  async function hydrateWorkspaceRegistry() {
    const reg = await getJSON('/api/workspaces/registry');
    window.EU_WORKSPACE_REGISTRY = reg;
    return reg;
  }

  function registry() {
    return window.EU_WORKSPACE_REGISTRY || { sources: [], active_path: null, crossdb_paths: [] };
  }

  function activeSource() {
    const reg = registry();
    const path = reg.active_path;
    return (reg.sources || []).find(s => s.path === path) || (reg.sources || []).find(s => s.ok) || null;
  }

  function activePath() {
    const src = activeSource();
    return src && src.path ? src.path : null;
  }

  function crossdbPaths() {
    const reg = registry();
    const paths = Array.isArray(reg.crossdb_paths) ? reg.crossdb_paths.slice() : [];
    if (paths.length) return paths;
    const ok = (reg.sources || []).filter(s => s.ok && s.path).map(s => s.path);
    return ok.length >= 2 ? ok.slice(0, 2) : ok;
  }

  function setRegistry(reg) {
    window.EU_WORKSPACE_REGISTRY = reg || { sources: [], active_path: null, crossdb_paths: [] };
    return window.EU_WORKSPACE_REGISTRY;
  }

  async function saveWorkspaceRegistry(patch) {
    return setRegistry(await postJSON('/api/workspaces/registry', patch || {}));
  }

  async function registerWorkspaceSource(path, opts) {
    const body = { path: path, active: true, crossdb: true };
    Object.assign(body, opts || {});
    return setRegistry(await postJSON('/api/workspaces/register', body));
  }
  async function renameWorkspaceSource(path, label) {
    return setRegistry(await postJSON('/api/workspaces/rename', { path: path, label: label }));
  }
  async function removeWorkspaceSource(path) {
    return setRegistry(await postJSON('/api/workspaces/remove', { path: path }));
  }

  function rerender() {
    if (typeof window.__euRender === 'function') window.__euRender();
  }

  async function boot() {
    try {
      await Promise.all([hydrateCatalog(), hydrateSettings(), hydrateWorkspaceRegistry()]);
      rerender();
      console.info('[EasyICU] hydrated: %d concepts, settings loaded (ai_enabled=%s)',
        window.EU_CATALOG.totalConcepts, window.EU_SETTINGS.ai_enabled);
    } catch (err) {
      console.error('[EasyICU] live fetch failed, staying on mock:', err);
    }
  }

  // Data-extraction endpoints (Stage 3a): server-side folder picker + scan.
  function listDir(path) {
    return getJSON('/api/fs/list' + (path ? '?path=' + encodeURIComponent(path) : ''));
  }
  function scanPath(path, source) {
    return postJSON('/api/data/scan', { path: path, source: source || null });
  }
  function loadExtractionFilterOptions(body) {
    return postJSON('/api/extraction/filter-options', body || {});
  }
  function previewExtractionFilters(body) {
    return postJSON('/api/extraction/filter-preview', body || {});
  }
  function loadWorkspaceSummary(path) {
    return postJSON('/api/workspace/summary', { path: path });
  }
  function loadPatientReviewDrilldown(body) {
    return postJSON('/api/patient-review/drilldown', body || {});
  }
  function loadCohortReviewSummary(body) {
    return postJSON('/api/cohort-review/summary', body || {});
  }
  function loadCrossdbReviewSummary(body) {
    return postJSON('/api/crossdb-review/summary', body || {});
  }
  function loadCrossdbSummary(paths) {
    return postJSON('/api/workspaces/crossdb-summary', { paths: paths || [] });
  }
  function loadAgentProviderStatus(provider) {
    const q = provider ? '?provider=' + encodeURIComponent(provider) : '';
    return getJSON('/api/agent-runs/provider-status' + q);
  }
  function startAgentRun(body) {
    return postJSON('/api/jobs/agent-run', body || {});
  }
  function loadAgentRunReview(projectDir) {
    return postJSON('/api/agent-runs/review', { project_dir: projectDir });
  }
  function signoffAgentRun(projectDir, body) {
    const payload = Object.assign({}, body || {}, { project_dir: projectDir });
    return postJSON('/api/agent-runs/signoff', payload);
  }
  function loadAgentRunHistory(body) {
    return postJSON('/api/agent-runs/history', body || {});
  }
  function loadAgentRunArtifact(projectDir, artifact) {
    return postJSON('/api/agent-runs/artifact', { project_dir: projectDir, artifact: artifact });
  }
  async function downloadAgentRunArtifact(projectDir, artifact) {
    const file = await postBlob('/api/agent-runs/download-artifact', { project_dir: projectDir, artifact: artifact });
    triggerDownload(file.blob, file.filename);
    return file;
  }
  async function downloadAgentRunBundle(projectDir) {
    const file = await postBlob('/api/agent-runs/download-bundle', { project_dir: projectDir });
    triggerDownload(file.blob, file.filename);
    return file;
  }
  function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'download.bin';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  window.EU_API.getJSON = getJSON;
  window.EU_API.postJSON = postJSON;
  window.EU_API.postBlob = postBlob;
  window.EU_API.hydrateCatalog = hydrateCatalog;
  window.EU_API.hydrateSettings = hydrateSettings;
  window.EU_API.hydrateWorkspaceRegistry = hydrateWorkspaceRegistry;
  window.EU_API.saveSetting = saveSetting;
  window.EU_API.saveWorkspaceRegistry = saveWorkspaceRegistry;
  window.EU_API.registerWorkspaceSource = registerWorkspaceSource;
  window.EU_API.renameWorkspaceSource = renameWorkspaceSource;
  window.EU_API.removeWorkspaceSource = removeWorkspaceSource;
  window.EU_API.listDir = listDir;
  window.EU_API.scanPath = scanPath;
  window.EU_API.loadExtractionFilterOptions = loadExtractionFilterOptions;
  window.EU_API.previewExtractionFilters = previewExtractionFilters;
  window.EU_API.loadWorkspaceSummary = loadWorkspaceSummary;
  window.EU_API.loadPatientReviewDrilldown = loadPatientReviewDrilldown;
  window.EU_API.loadCohortReviewSummary = loadCohortReviewSummary;
  window.EU_API.loadCrossdbReviewSummary = loadCrossdbReviewSummary;
  window.EU_API.loadCrossdbSummary = loadCrossdbSummary;
  window.EU_API.loadAgentProviderStatus = loadAgentProviderStatus;
  window.EU_API.startAgentRun = startAgentRun;
  window.EU_API.loadAgentRunReview = loadAgentRunReview;
  window.EU_API.signoffAgentRun = signoffAgentRun;
  window.EU_API.loadAgentRunHistory = loadAgentRunHistory;
  window.EU_API.loadAgentRunArtifact = loadAgentRunArtifact;
  window.EU_API.downloadAgentRunArtifact = downloadAgentRunArtifact;
  window.EU_API.downloadAgentRunBundle = downloadAgentRunBundle;

  window.EU_SOURCES = window.EU_SOURCES || {};
  window.EU_SOURCES.registry = registry;
  window.EU_SOURCES.activeSource = activeSource;
  window.EU_SOURCES.activePath = activePath;
  window.EU_SOURCES.crossdbPaths = crossdbPaths;
  window.EU_SOURCES.setRegistry = setRegistry;
  window.EU_SOURCES.reload = hydrateWorkspaceRegistry;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
