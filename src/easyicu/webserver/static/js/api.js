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
    if (!res.ok) {
      let d = null;
      try { const payload = await res.json(); d = payload && payload.detail; } catch (e) {}
      throw apiError(path, res, d);
    }
    return res.json();
  }

  /* Backend validation errors ship a human sentence in detail.reason next to
     the machine detail.error code. Clinicians should read the sentence, not
     'path -> HTTP 400 · pdf_too_large' — so for 4xx the Error message is the
     human reason (code as fallback) and the transport string moves to
     err.technical; 5xx keeps the technical prefix (it IS the story there). */
  function apiError(path, res, d) {
    const technical = path + ' -> HTTP ' + res.status;
    const reason = d && typeof d === 'object' ? (d.reason || '') : '';
    const code = d && typeof d === 'object' ? (d.error || '') : (typeof d === 'string' ? d : '');
    const human = reason || code;
    const err = new Error(res.status < 500 && human ? human : technical + (human ? ' · ' + human : ''));
    err.technical = technical; err.status = res.status; err.code = code || null;
    err.details = d && typeof d === 'object' && d.details && typeof d.details === 'object' ? d.details : {};
    return err;
  }
  async function postJSON(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let d = null;
      try { const payload = await res.json(); d = payload && payload.detail; } catch (e) {}
      throw apiError(path, res, d);
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
      let d = null;
      try { const payload = await res.json(); d = payload && payload.detail; } catch (e) {}
      throw apiError(path, res, d);
    }
    const disposition = res.headers.get('content-disposition') || '';
    const match = disposition.match(/filename="([^"]+)"/i);
    return { blob: await res.blob(), filename: match ? match[1] : 'download.bin' };
  }

  async function hydrateSettings() {
    window.EU_SETTINGS = await getJSON('/api/settings');
    // Browser language/data-mode selection is the source of truth for the
    // current UI. If it differs from the backend settings file, keep the UI
    // responsive now and persist the browser choice in the background.
    let browserLang = null;
    let browserMode = null;
    try { browserLang = localStorage.getItem('easyicu_lang'); } catch (e) {}
    try { browserMode = localStorage.getItem('easyicu_home_data'); } catch (e) {}
    const serverLang = window.EU_SETTINGS.language;
    const serverMode = window.EU_SETTINGS.data_mode;
    const validLang = l => l === 'en' || l === 'zh';
    const validMode = m => m === 'demo' || m === 'real';
    const lang = validLang(browserLang) ? browserLang : (validLang(serverLang) ? serverLang : (validLang(window.EU_LANG) ? window.EU_LANG : 'en'));
    const mode = validMode(browserMode) ? browserMode : (validMode(serverMode) ? serverMode : (validMode(window.EU_DATA) ? window.EU_DATA : 'demo'));
    window.EU_LANG = lang;
    window.EU_DATA = mode;
    window.EU_SETTINGS.language = lang;
    window.EU_SETTINGS.data_mode = mode;
    if (window.applySettingsState) window.applySettingsState(window.EU_SETTINGS);
    const syncPatch = {};
    if (validLang(browserLang) && browserLang !== serverLang) syncPatch.language = browserLang;
    if (validMode(browserMode) && browserMode !== serverMode) syncPatch.data_mode = browserMode;
    if (Object.keys(syncPatch).length) {
      postJSON('/api/settings', syncPatch).then(settings => {
        window.EU_SETTINGS = Object.assign({}, window.EU_SETTINGS || {}, settings || {}, syncPatch);
        if (window.applySettingsState) window.applySettingsState(window.EU_SETTINGS);
      }).catch(err => console.warn('[EasyICU] language setting sync failed', err));
    }
    if (!validLang(browserLang)) {
      try { localStorage.setItem('easyicu_lang', lang); } catch (e) {}
    }
    if (!validMode(browserMode)) {
      try { localStorage.setItem('easyicu_home_data', mode); } catch (e) {}
    }
    return window.EU_SETTINGS;
  }

  async function hydrateCapabilities() {
    window.EU_CAPABILITIES = await getJSON('/api/capabilities');
    return window.EU_CAPABILITIES;
  }

  // Persist a single setting; updates the local cache from the server reply.
  async function saveSetting(key, value) {
    const patch = {}; patch[key] = value;
    window.EU_SETTINGS = await postJSON('/api/settings', patch);
    if (window.applySettingsState) window.applySettingsState(window.EU_SETTINGS);
    try {
      await hydrateCapabilities();
    } catch (err) {
      console.warn('[EasyICU] capability refresh failed after setting save', err);
    }
    return window.EU_SETTINGS;
  }

  async function resetSettings() {
    window.EU_SETTINGS = await postJSON('/api/settings/reset', {});
    if (window.applySettingsState) window.applySettingsState(window.EU_SETTINGS, { syncStorage: true });
    try {
      await hydrateCapabilities();
    } catch (err) {
      console.warn('[EasyICU] capability refresh failed after reset', err);
    }
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
    if (real.conceptCoverage) cat.conceptCoverage = real.conceptCoverage;
    else delete cat.conceptCoverage;
    if (real.coverageSummary) cat.coverageSummary = real.coverageSummary;
    else delete cat.coverageSummary;
    if (real.activeExportCoverage) cat.activeExportCoverage = real.activeExportCoverage;
    else delete cat.activeExportCoverage;
    cat.desc = real.desc;
    cat.totalConcepts = real.totalConcepts;
    cat.supportedDbs = real.supportedDbs;
    cat.__live = true;
    return cat;
  }

  async function loadConceptLineage(conceptId) {
    return getJSON('/api/catalog/lineage/' + encodeURIComponent(String(conceptId || '')));
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
      try {
        await hydrateCapabilities();
      } catch (err) {
        console.warn('[EasyICU] capability fetch failed:', err);
      }
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
  function createDir(path) {
    return postJSON('/api/fs/mkdir', { path: path });
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
  function startExtractionJob(body) {
    return postJSON('/api/jobs/extract', body || {});
  }
  function loadOfficialDemoSources() {
    return getJSON('/api/demo-sources');
  }
  function startOfficialDemoSourcePrepare(sourceId) {
    return postJSON('/api/jobs/demo-source-prepare', { source_id: sourceId });
  }
  function loadWorkspaceSummary(path) {
    return postJSON('/api/workspace/summary', { path: path });
  }
  function loadPatientReviewSources(body) {
    return postJSON('/api/patient-review/sources', body || {});
  }
  function loadPatientReviewDrilldown(body) {
    return postJSON('/api/patient-review/drilldown', body || {});
  }
  function loadPatientReviewEntities(body) {
    return postJSON('/api/patient-review/entities', body || {});
  }
  function loadPatientReviewEntity(body) {
    return postJSON('/api/patient-review/entity', body || {});
  }
  function loadPatientReviewTablePreview(body) {
    return postJSON('/api/patient-review/table-preview', body || {});
  }
  function loadPatientReviewFeature(body) {
    return postJSON('/api/patient-review/feature', body || {});
  }
  function loadCohortReviewSummary(body) {
    return postJSON('/api/cohort-review/summary', body || {});
  }
  function loadCrossdbRawDistribution(body) {
    return postJSON('/api/crossdb-review/raw-distribution', body || {});
  }
  function scanCrossdbRawRoot(body) {
    return postJSON('/api/crossdb-review/raw-root-scan', body || {});
  }
  function startCrossdbRawDistributionJob(body) {
    return postJSON('/api/jobs/crossdb-raw-distribution', body || {});
  }
  function startCrossdbReviewSummaryJob(body) {
    return postJSON('/api/jobs/crossdb-summary', body || {});
  }
  function loadCrossdbDemoDistribution(body) {
    return postJSON('/api/crossdb-review/demo-distribution', body || {});
  }
  function loadAgentProviderStatus(provider) {
    const q = provider ? '?provider=' + encodeURIComponent(provider) : '';
    return getJSON('/api/agent-runs/provider-status' + q);
  }
  function loadCodexAuthStatus() {
    return getJSON('/api/agent-runs/codex-auth/status');
  }
  function startCodexAuthLogin() {
    return postJSON('/api/agent-runs/codex-auth/login', {});
  }
  function cancelCodexAuthLogin() {
    return postJSON('/api/agent-runs/codex-auth/cancel', {});
  }
  function logoutCodexAuth() {
    return postJSON('/api/agent-runs/codex-auth/logout', {});
  }
  function saveAgentProviderConfig(body) {
    return postJSON('/api/agent-runs/provider-config', body || {});
  }
  function startAgentRun(body) {
    return postJSON('/api/jobs/agent-run', body || {});
  }
  function submitAgentRunReview(body) {
    return postJSON('/api/jobs/agent-run-review', body || {});
  }
  function loadActiveStudyContext() {
    return getJSON('/api/study-contexts/active');
  }
  function listStudyContexts() {
    return getJSON('/api/study-contexts');
  }
  function loadStudyContext(id) {
    return getJSON('/api/study-contexts/' + encodeURIComponent(id || ''));
  }
  function saveStudyContext(body) {
    return postJSON('/api/study-contexts', body || {});
  }
  function handoffStudyContext(body) {
    return postJSON('/api/study-contexts/handoff', body || {});
  }
  function loadJobSnapshot(jobId) {
    return getJSON('/api/jobs/' + encodeURIComponent(jobId || ''));
  }
  function cancelJob(jobId, reason) {
    return postJSON('/api/jobs/' + encodeURIComponent(jobId || '') + '/cancel', { reason: reason || 'user_requested' });
  }
  function openExtractionOutput(jobId, file) {
    const body = file ? { file: file } : {};
    return postJSON('/api/jobs/' + encodeURIComponent(jobId || '') + '/open-output', body);
  }
  function loadAgentRunReview(projectDir) {
    return postJSON('/api/agent-runs/review', { project_dir: projectDir });
  }
  function loadAgentScienceWorkbench(body) {
    return postJSON('/api/agent-runs/science-workbench', body || {});
  }
  function loadCapabilities() {
    return hydrateCapabilities();
  }
  async function loadExtensions() {
    window.EU_EXTENSIONS = await getJSON('/api/extensions');
    return window.EU_EXTENSIONS;
  }
  async function installExtensionSkill(body) {
    const result = await postJSON('/api/extensions/skills/install', body || {});
    if (result && result.extensions) window.EU_EXTENSIONS = result.extensions;
    return result;
  }
  async function installExtensionMcp(body) {
    const result = await postJSON('/api/extensions/mcp/install', body || {});
    if (result && result.extensions) window.EU_EXTENSIONS = result.extensions;
    return result;
  }
  async function setExtensionState(body) {
    const result = await postJSON('/api/extensions/state', body || {});
    if (result && result.extensions) window.EU_EXTENSIONS = result.extensions;
    return result;
  }
  async function removeExtension(body) {
    const result = await postJSON('/api/extensions/remove', body || {});
    if (result && result.extensions) window.EU_EXTENSIONS = result.extensions;
    return result;
  }
  function testExtensionMcp(body) {
    return postJSON('/api/extensions/mcp/test', body || {});
  }
  function checkCapabilityTool(body) {
    return postJSON('/api/capabilities/tool-check', body || {});
  }
  function searchZotero(body) {
    return postJSON('/api/capabilities/zotero/search', body || {});
  }
  function testZoteroConnection(body) {
    return postJSON('/api/capabilities/zotero/test', body || {});
  }
  function zoteroSource(body) {
    return postJSON('/api/capabilities/zotero/source', body || {});
  }
  function importZoteroSource(body) {
    return postJSON('/api/capabilities/zotero/import', body || {});
  }
  function loadCapabilityAuditEvents(body) {
    return postJSON('/api/capabilities/audit-events', body || {});
  }
  function signoffAgentRun(projectDir, body) {
    const payload = Object.assign({}, body || {}, { project_dir: projectDir });
    return postJSON('/api/agent-runs/signoff', payload);
  }
  function loadAgentRunHistory(body) {
    return postJSON('/api/agent-runs/history', body || {});
  }
  function createGuidedDraft(body) {
    return postJSON('/api/guided/drafts', body || {});
  }
  function loadGuidedDrafts(body) {
    return postJSON('/api/guided/drafts/list', body || {});
  }
  function removeGuidedDraft(body) {
    return postJSON('/api/guided/drafts/remove', body || {});
  }
  function createGuidedSession(body) {
    return postJSON('/api/guided/session', body || {});
  }
  function openGuidedProject(body) {
    return postJSON('/api/guided/project/open', body || {});
  }
  function sendGuidedMessage(body) {
    return postJSON('/api/guided/message', body || {});
  }
  function runGuidedAction(body) {
    return postJSON('/api/guided/action', body || {});
  }
  function saveGuidedSlots(body) {
    return postJSON('/api/guided/action', Object.assign({ action: 'update_slots' }, body || {}));
  }
  function loadGuidedSessions(body) {
    return postJSON('/api/guided/sessions/list', body || {});
  }
  function loadPiCopilotStatus() {
    return getJSON('/api/copilot/pi/status');
  }
  function savePiCopilotProviderConfig(body) {
    return postJSON('/api/copilot/pi/provider-config', body || {});
  }
  function loadPiCopilotCodexStatus() {
    return getJSON('/api/copilot/pi/research-provider/codex/status');
  }
  function startPiCopilotCodexLogin(flow) {
    return postJSON('/api/copilot/pi/research-provider/codex/login', { flow: flow || 'browser' });
  }
  function cancelPiCopilotCodexLogin() {
    return postJSON('/api/copilot/pi/research-provider/codex/cancel', {});
  }
  function logoutPiCopilotCodex() {
    return postJSON('/api/copilot/pi/research-provider/codex/logout', {});
  }
  function loadPiCopilotCodexModels() {
    return getJSON('/api/copilot/pi/research-provider/codex/models');
  }
  function createPiCopilotSession(body) {
    return postJSON('/api/copilot/pi/sessions', body || {});
  }
  function initializePiCopilotProject(body) {
    return postJSON('/api/copilot/pi/projects/initialize', body || {});
  }
  function loadPiCopilotProjectWorkflow(projectId) {
    return getJSON('/api/copilot/pi/projects/' + encodeURIComponent(projectId || '') + '/workflow');
  }
  function loadPiCopilotLiteratureSource(pmid) {
    return getJSON('/api/copilot/pi/literature/sources/' + encodeURIComponent(pmid || ''));
  }
  function loadPiCopilotSessions(limit, projectId, agentMode) {
    const n = Math.max(1, Math.min(100, Number(limit) || 30));
    const mode = agentMode === 'research' || agentMode === 'workspace' ? '&agent_mode=' + encodeURIComponent(agentMode) : '';
    return getJSON('/api/copilot/pi/sessions?project_id=' + encodeURIComponent(projectId || '') + '&limit=' + encodeURIComponent(n) + mode);
  }
  function loadPiCopilotSession(sessionId, projectId, options) {
    const opts = options || {};
    const query = new URLSearchParams({ project_id: String(projectId || '') });
    if (opts.transcriptCursor != null) query.set('transcript_cursor', String(opts.transcriptCursor));
    if (opts.transcriptLimit != null) query.set('transcript_limit', String(opts.transcriptLimit));
    if (opts.replayCursor != null) query.set('replay_cursor', String(opts.replayCursor));
    if (opts.replayLimit != null) query.set('replay_limit', String(opts.replayLimit));
    return getJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '?' + query.toString());
  }
  function sendPiCopilotMessage(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/message', body || {});
  }
  function confirmPiCopilotCohortEligibility(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/cohort-eligibility-selection', body || {});
  }
  function confirmPiCopilotPlanDecision(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/plan-decision-selection', body || {});
  }
  function regeneratePiCopilotMessage(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/regenerate', body || {});
  }
  function authorizePiCopilotDataSource(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/data-source-authorization', body || {});
  }
  function rebindPiCopilotSession(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/rebind', body || {});
  }
  function pinPiCopilotPresentation(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/presentation', body || {});
  }
  function archivePiCopilotChildJob(sessionId, jobId, body) {
    return postJSON(
      '/api/copilot/pi/sessions/' + encodeURIComponent(sessionId)
      + '/child-jobs/' + encodeURIComponent(jobId) + '/archive',
      body || {},
    );
  }
  function recordPiCopilotHostAction(sessionId, body) {
    return postJSON(
      '/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/host-actions',
      body || {},
    );
  }
  function abortPiCopilotSession(sessionId, body) {
    return postJSON('/api/copilot/pi/sessions/' + encodeURIComponent(sessionId) + '/abort', body || {});
  }
  function loadPiCopilotWorkspaceFile(projectId, file) {
    return getJSON('/api/copilot/pi/projects/' + encodeURIComponent(projectId) + '/workspace/file?file=' + encodeURIComponent(file));
  }
  function piCopilotWorkspacePreviewUrl(projectId, file, checkedSha256) {
    return '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/workspace/preview?file=' + encodeURIComponent(file)
      + '&checked_sha256=' + encodeURIComponent(checkedSha256 || '');
  }
  function loadPiCopilotResearchArtifact(projectId, runId, artifact, expectedSha256) {
    const digest = String(expectedSha256 || '').trim().toLowerCase();
    return getJSON(
      '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/runs/' + encodeURIComponent(runId)
      + '/artifacts/' + encodeURIComponent(artifact)
      + (/^[a-f0-9]{64}$/.test(digest)
        ? '?expected_sha256=' + encodeURIComponent(digest) : '')
    );
  }
  function loadPiCopilotResearchEvidence(projectId, runId, evidenceId, expectedSha256) {
    return getJSON(
      '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/runs/' + encodeURIComponent(runId)
      + '/evidence/' + encodeURIComponent(evidenceId)
      + '?expected_sha256=' + encodeURIComponent(expectedSha256)
    );
  }
  function loadPiCopilotDataPackageReview(projectId, studyRevision, reviewSha256) {
    return getJSON(
      '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/data-package-review?study_revision=' + encodeURIComponent(studyRevision)
      + '&review_sha256=' + encodeURIComponent(reviewSha256)
    );
  }
  function preparePiCopilotDataPackageReview(projectId) {
    return postJSON(
      '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/data-package-review/prepare',
      {},
    );
  }
  function preparePiCopilotDataWorkbenchSnapshot(projectId) {
    return postJSON(
      '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/data-workbench-snapshot/prepare',
      {},
    );
  }
  function loadPiCopilotDataWorkbenchSnapshot(projectId, snapshotSha256) {
    return getJSON(
      '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/data-workbench-snapshot?snapshot_sha256=' + encodeURIComponent(snapshotSha256)
    );
  }
  function piCopilotResearchDocumentUrl(projectId, runId, documentName) {
    return '/api/copilot/pi/projects/' + encodeURIComponent(projectId)
      + '/runs/' + encodeURIComponent(runId)
      + '/documents/' + encodeURIComponent(documentName);
  }
  function createPageGuideSession(body) {
    return postJSON('/api/page-guide/sessions', body || {});
  }
  function sendPageGuideMessage(body) {
    return postJSON('/api/page-guide/message', body || {});
  }
  function runPageGuideAction(body) {
    return postJSON('/api/page-guide/action', body || {});
  }
  function loadPageGuideSessions(body) {
    return postJSON('/api/page-guide/sessions/list', body || {});
  }
  function createCopilotSession(body) {
    return postJSON('/api/copilot/sessions', body || {});
  }
  function sendCopilotMessage(body) {
    return postJSON('/api/copilot/message', body || {});
  }
  function runCopilotAction(body) {
    return postJSON('/api/copilot/action', body || {});
  }
  function loadCopilotSessions(body) {
    return postJSON('/api/copilot/sessions/list', body || {});
  }
  function mineIdeas(body) {
    return postJSON('/api/ideas/mine', body || {});
  }
  function resolveIdeaSource(body) {
    return postJSON('/api/ideas/resolve-source', body || {});
  }
  function discoverIdeas(body) {
    return postJSON('/api/ideas/discover', body || {});
  }
  function ingestIdeaPdf(body) {
    return postJSON('/api/ideas/ingest-pdf', body || {});
  }
  function scanIdeaLiteratureFolder(body) {
    return postJSON('/api/ideas/literature-folder', body || {});
  }
  function checkIdeaPriorArt(body) {
    return postJSON('/api/ideas/prior-art', body || {});
  }
  function planIdea(body) {
    return postJSON('/api/ideas/plan', body || {});
  }
  function checkIdeaSampleFeasibility(body) {
    return postJSON('/api/ideas/bounded-feasibility', body || {});
  }
  function handoffIdea(body) {
    return postJSON('/api/ideas/handoff', body || {});
  }
  function createIdeaAgentProject(body) {
    return postJSON('/api/ideas/create-agent-project', body || {});
  }
  function loadIdeaAgentProjects(body) {
    return postJSON('/api/ideas/agent-projects', body || {});
  }
  function loadIdeaHistory(body) {
    return postJSON('/api/ideas/history', body || {});
  }
  function loadIdeaRun(body) {
    return postJSON('/api/ideas/run', body || {});
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
  async function downloadRegisteredExport(sourceId) {
    const file = await postBlob('/api/workspaces/download', { source_id: sourceId });
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
  window.EU_API.loadConceptLineage = loadConceptLineage;
  window.EU_API.hydrateSettings = hydrateSettings;
  window.EU_API.hydrateCapabilities = hydrateCapabilities;
  window.EU_API.hydrateWorkspaceRegistry = hydrateWorkspaceRegistry;
  window.EU_API.saveSetting = saveSetting;
  window.EU_API.resetSettings = resetSettings;
  window.EU_API.saveWorkspaceRegistry = saveWorkspaceRegistry;
  window.EU_API.registerWorkspaceSource = registerWorkspaceSource;
  window.EU_API.renameWorkspaceSource = renameWorkspaceSource;
  window.EU_API.removeWorkspaceSource = removeWorkspaceSource;
  window.EU_API.listDir = listDir;
  window.EU_API.createDir = createDir;
  window.EU_API.scanPath = scanPath;
  window.EU_API.loadExtractionFilterOptions = loadExtractionFilterOptions;
  window.EU_API.previewExtractionFilters = previewExtractionFilters;
  window.EU_API.startExtractionJob = startExtractionJob;
  window.EU_API.loadOfficialDemoSources = loadOfficialDemoSources;
  window.EU_API.startOfficialDemoSourcePrepare = startOfficialDemoSourcePrepare;
  window.EU_API.loadWorkspaceSummary = loadWorkspaceSummary;
  window.EU_API.loadPatientReviewSources = loadPatientReviewSources;
  window.EU_API.loadPatientReviewDrilldown = loadPatientReviewDrilldown;
  window.EU_API.loadPatientReviewEntities = loadPatientReviewEntities;
  window.EU_API.loadPatientReviewEntity = loadPatientReviewEntity;
  window.EU_API.loadPatientReviewTablePreview = loadPatientReviewTablePreview;
  window.EU_API.loadPatientReviewFeature = loadPatientReviewFeature;
  window.EU_API.loadCohortReviewSummary = loadCohortReviewSummary;
  window.EU_API.loadCrossdbRawDistribution = loadCrossdbRawDistribution;
  window.EU_API.scanCrossdbRawRoot = scanCrossdbRawRoot;
  window.EU_API.startCrossdbRawDistributionJob = startCrossdbRawDistributionJob;
  window.EU_API.startCrossdbReviewSummaryJob = startCrossdbReviewSummaryJob;
  window.EU_API.loadCrossdbDemoDistribution = loadCrossdbDemoDistribution;
  window.EU_API.loadAgentProviderStatus = loadAgentProviderStatus;
  window.EU_API.loadCodexAuthStatus = loadCodexAuthStatus;
  window.EU_API.startCodexAuthLogin = startCodexAuthLogin;
  window.EU_API.cancelCodexAuthLogin = cancelCodexAuthLogin;
  window.EU_API.logoutCodexAuth = logoutCodexAuth;
  window.EU_API.saveAgentProviderConfig = saveAgentProviderConfig;
  window.EU_API.startAgentRun = startAgentRun;
  window.EU_API.submitAgentRunReview = submitAgentRunReview;
  window.EU_API.loadActiveStudyContext = loadActiveStudyContext;
  window.EU_API.listStudyContexts = listStudyContexts;
  window.EU_API.loadStudyContext = loadStudyContext;
  window.EU_API.saveStudyContext = saveStudyContext;
  window.EU_API.handoffStudyContext = handoffStudyContext;
  window.EU_API.loadJobSnapshot = loadJobSnapshot;
  window.EU_API.cancelJob = cancelJob;
  window.EU_API.openExtractionOutput = openExtractionOutput;
  window.EU_API.loadAgentRunReview = loadAgentRunReview;
  window.EU_API.loadAgentScienceWorkbench = loadAgentScienceWorkbench;
  window.EU_API.loadCapabilities = loadCapabilities;
  window.EU_API.loadExtensions = loadExtensions;
  window.EU_API.installExtensionSkill = installExtensionSkill;
  window.EU_API.installExtensionMcp = installExtensionMcp;
  window.EU_API.setExtensionState = setExtensionState;
  window.EU_API.removeExtension = removeExtension;
  window.EU_API.testExtensionMcp = testExtensionMcp;
  window.EU_API.checkCapabilityTool = checkCapabilityTool;
  window.EU_API.searchZotero = searchZotero;
  window.EU_API.testZoteroConnection = testZoteroConnection;
  window.EU_API.zoteroSource = zoteroSource;
  window.EU_API.importZoteroSource = importZoteroSource;
  window.EU_API.loadCapabilityAuditEvents = loadCapabilityAuditEvents;
  window.EU_API.signoffAgentRun = signoffAgentRun;
  window.EU_API.loadAgentRunHistory = loadAgentRunHistory;
  window.EU_API.createGuidedDraft = createGuidedDraft;
  window.EU_API.loadGuidedDrafts = loadGuidedDrafts;
  window.EU_API.removeGuidedDraft = removeGuidedDraft;
  window.EU_API.createGuidedSession = createGuidedSession;
  window.EU_API.openGuidedProject = openGuidedProject;
  window.EU_API.sendGuidedMessage = sendGuidedMessage;
  window.EU_API.runGuidedAction = runGuidedAction;
  window.EU_API.saveGuidedSlots = saveGuidedSlots;
  window.EU_API.loadGuidedSessions = loadGuidedSessions;
  window.EU_API.loadPiCopilotStatus = loadPiCopilotStatus;
  window.EU_API.savePiCopilotProviderConfig = savePiCopilotProviderConfig;
  window.EU_API.loadPiCopilotCodexStatus = loadPiCopilotCodexStatus;
  window.EU_API.startPiCopilotCodexLogin = startPiCopilotCodexLogin;
  window.EU_API.cancelPiCopilotCodexLogin = cancelPiCopilotCodexLogin;
  window.EU_API.logoutPiCopilotCodex = logoutPiCopilotCodex;
  window.EU_API.loadPiCopilotCodexModels = loadPiCopilotCodexModels;
  window.EU_API.createPiCopilotSession = createPiCopilotSession;
  window.EU_API.initializePiCopilotProject = initializePiCopilotProject;
  window.EU_API.loadPiCopilotProjectWorkflow = loadPiCopilotProjectWorkflow;
  window.EU_API.loadPiCopilotLiteratureSource = loadPiCopilotLiteratureSource;
  window.EU_API.loadPiCopilotSessions = loadPiCopilotSessions;
  window.EU_API.loadPiCopilotSession = loadPiCopilotSession;
  window.EU_API.sendPiCopilotMessage = sendPiCopilotMessage;
  window.EU_API.confirmPiCopilotCohortEligibility = confirmPiCopilotCohortEligibility;
  window.EU_API.confirmPiCopilotPlanDecision = confirmPiCopilotPlanDecision;
  window.EU_API.regeneratePiCopilotMessage = regeneratePiCopilotMessage;
  window.EU_API.authorizePiCopilotDataSource = authorizePiCopilotDataSource;
  window.EU_API.rebindPiCopilotSession = rebindPiCopilotSession;
  window.EU_API.pinPiCopilotPresentation = pinPiCopilotPresentation;
  window.EU_API.archivePiCopilotChildJob = archivePiCopilotChildJob;
  window.EU_API.recordPiCopilotHostAction = recordPiCopilotHostAction;
  window.EU_API.abortPiCopilotSession = abortPiCopilotSession;
  window.EU_API.loadPiCopilotWorkspaceFile = loadPiCopilotWorkspaceFile;
  window.EU_API.piCopilotWorkspacePreviewUrl = piCopilotWorkspacePreviewUrl;
  window.EU_API.loadPiCopilotResearchArtifact = loadPiCopilotResearchArtifact;
  window.EU_API.loadPiCopilotResearchEvidence = loadPiCopilotResearchEvidence;
  window.EU_API.loadPiCopilotDataPackageReview = loadPiCopilotDataPackageReview;
  window.EU_API.preparePiCopilotDataPackageReview = preparePiCopilotDataPackageReview;
  window.EU_API.preparePiCopilotDataWorkbenchSnapshot = preparePiCopilotDataWorkbenchSnapshot;
  window.EU_API.loadPiCopilotDataWorkbenchSnapshot = loadPiCopilotDataWorkbenchSnapshot;
  window.EU_API.piCopilotResearchDocumentUrl = piCopilotResearchDocumentUrl;
  window.EU_API.createPageGuideSession = createPageGuideSession;
  window.EU_API.sendPageGuideMessage = sendPageGuideMessage;
  window.EU_API.runPageGuideAction = runPageGuideAction;
  window.EU_API.loadPageGuideSessions = loadPageGuideSessions;
  window.EU_API.createCopilotSession = createCopilotSession;
  window.EU_API.sendCopilotMessage = sendCopilotMessage;
  window.EU_API.runCopilotAction = runCopilotAction;
  window.EU_API.loadCopilotSessions = loadCopilotSessions;
  window.EU_API.mineIdeas = mineIdeas;
  window.EU_API.resolveIdeaSource = resolveIdeaSource;
  window.EU_API.discoverIdeas = discoverIdeas;
  window.EU_API.ingestIdeaPdf = ingestIdeaPdf;
  window.EU_API.scanIdeaLiteratureFolder = scanIdeaLiteratureFolder;
  window.EU_API.checkIdeaPriorArt = checkIdeaPriorArt;
  window.EU_API.planIdea = planIdea;
  window.EU_API.checkIdeaSampleFeasibility = checkIdeaSampleFeasibility;
  window.EU_API.handoffIdea = handoffIdea;
  window.EU_API.createIdeaAgentProject = createIdeaAgentProject;
  window.EU_API.loadIdeaAgentProjects = loadIdeaAgentProjects;
  window.EU_API.loadIdeaHistory = loadIdeaHistory;
  window.EU_API.loadIdeaRun = loadIdeaRun;
  window.EU_API.loadAgentRunArtifact = loadAgentRunArtifact;
  window.EU_API.downloadAgentRunArtifact = downloadAgentRunArtifact;
  window.EU_API.downloadAgentRunBundle = downloadAgentRunBundle;
  window.EU_API.downloadRegisteredExport = downloadRegisteredExport;

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
