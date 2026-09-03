/* Digest-pinned artifact-set loader for synthetic Guided Copilot reports. */
(function () {
  'use strict';

  async function load(api, projectId, runId, names, resource, optionalNames) {
    const anchor = String(resource && resource.sha256 || '').trim().toLowerCase();
    if (!/^[a-f0-9]{64}$/.test(anchor)) {
      throw new Error('The report artifact registry is not digest pinned.');
    }
    const ledger = await api.loadPiCopilotResearchArtifact(
      projectId, runId, 'evidence_ledger.json', anchor,
    );
    const artifacts = ledger && ledger.payload && Array.isArray(ledger.payload.artifacts)
      ? ledger.payload.artifacts : [];
    const digests = new Map(artifacts
      .filter(row => row && /^[a-f0-9]{64}$/.test(String(row.sha256 || '')))
      .map(row => [String(row.name || ''), String(row.sha256)]));
    const optional = new Set(Array.isArray(optionalNames) ? optionalNames : []);
    const loaded = await Promise.all(names.map(async name => {
      const digest = digests.get(name);
      if (!digest) {
        if (optional.has(name)) return [name, null];
        throw new Error(`The report registry has no digest for ${name}.`);
      }
      try {
        return [name, await api.loadPiCopilotResearchArtifact(projectId, runId, name, digest)];
      } catch (error) {
        if (optional.has(name)) return [name, null];
        throw error;
      }
    }));
    return Object.fromEntries(loaded);
  }

  window.EU_GUIDED_PI_REPORT_ARTIFACTS = Object.freeze({ load });
})();
