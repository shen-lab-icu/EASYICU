/* Guided Pi conversation-resource transport owner.
   Keeps resource identity and DOM projection out of the already-large screen
   shell. It carries coordinates only; governed payloads load in the preview. */
(function () {
  'use strict';

  function create(deps) {
    const { esc } = deps;

    function name(resource) {
      return resource && String(resource.label || resource.artifact || resource.file || '').trim();
    }
    function key(resource) {
      if (!resource) return '';
      if (resource.kind === 'literature_source') return `literature:${resource.pmid || resource.doi || resource.url || ''}`;
      if (resource.kind === 'demo_artifact') return `demo:${resource.artifact || ''}`;
      if (resource.kind === 'demo_document') return `demo-document:${resource.artifact || ''}`;
      if (resource.kind === 'data_package_review') return `data-package:${resource.study_context_id || ''}:${resource.study_revision || 0}:${resource.review_sha256 || ''}`;
      if (resource.kind === 'data_workbench_snapshot') return `data-workbench:${resource.view || ''}:${resource.snapshot_sha256 || ''}`;
      if (resource.kind === 'native_workspace') return `native-workspace:${resource.route || ''}:${resource.study_context_id || ''}:${resource.job_id || resource.state || ''}`;
      return resource.kind === 'research_artifact' || resource.kind === 'research_document' || resource.kind === 'system_validation_document'
        ? `research:${resource.run_id || ''}:${resource.artifact || ''}`
        : `${resource.kind || 'file'}:${resource.file || ''}`;
    }
    function label(resource) {
      if (!resource) return '';
      if (resource.kind === 'demo_artifact' && window.EU_GUIDED_PI_DEMO && typeof window.EU_GUIDED_PI_DEMO.artifactLabel === 'function') {
        return window.EU_GUIDED_PI_DEMO.artifactLabel(resource.artifact || resource.label || '');
      }
      if (resource.kind === 'research_artifact' && window.AGENT_RENDER && typeof window.AGENT_RENDER.artifactTitle === 'function') {
        return window.AGENT_RENDER.artifactTitle(resource.artifact || resource.label || '');
      }
      return name(resource);
    }
    function kind(resource) {
      const supported = new Set([
        'demo_document', 'demo_artifact', 'data_package_review',
        'data_workbench_snapshot', 'native_workspace', 'system_validation_document',
        'research_document', 'research_artifact', 'literature_source', 'webpage',
      ]);
      return supported.has(resource && resource.kind) ? resource.kind : 'file';
    }
    function button(resource, overrideLabel) {
      if (!resource) return '';
      return `<button class="gpi-resource-link" type="button"
        data-gpi-resource-kind="${esc(kind(resource))}"
        data-gpi-resource-file="${esc(resource.file || '')}"
        data-gpi-resource-run="${esc(resource.run_id || '')}"
        data-gpi-resource-artifact="${esc(resource.artifact || '')}"
        data-gpi-resource-label="${esc(resource.kind === 'demo_artifact' ? (resource.title || label(resource)) : (resource.label || resource.artifact || resource.file || ''))}"
        data-gpi-resource-media="${esc(resource.media_type || 'text/plain')}"
        data-gpi-resource-url="${esc(resource.url || '')}"
        data-gpi-resource-title="${esc(resource.title || resource.label || '')}"
        data-gpi-resource-year="${esc(resource.year || '')}"
        data-gpi-resource-venue="${esc(resource.venue || '')}"
        data-gpi-resource-relevance="${esc(resource.relevance || '')}"
        data-gpi-resource-doi="${esc(resource.doi || '')}"
        data-gpi-resource-pmid="${esc(resource.pmid || '')}"
        data-gpi-resource-study="${esc(resource.study_context_id || '')}"
        data-gpi-resource-revision="${esc(resource.study_revision == null ? '' : resource.study_revision)}"
        data-gpi-resource-route="${esc(resource.route || '')}"
        data-gpi-resource-state="${esc(resource.state || '')}"
        data-gpi-resource-job="${esc(resource.job_id || '')}"
        data-gpi-resource-view="${esc(resource.view || '')}"
        data-gpi-resource-digest="${esc(resource.snapshot_sha256 || resource.review_sha256 || resource.checked_sha256 || resource.sha256 || '')}">${esc(overrideLabel || label(resource))}</button>`;
    }
    function fromButton(element) {
      if (!element) return null;
      return {
        file: element.dataset.gpiResourceFile,
        kind: element.dataset.gpiResourceKind,
        run_id: element.dataset.gpiResourceRun,
        artifact: element.dataset.gpiResourceArtifact,
        label: element.dataset.gpiResourceLabel,
        media_type: element.dataset.gpiResourceMedia,
        url: element.dataset.gpiResourceUrl,
        title: element.dataset.gpiResourceTitle,
        year: element.dataset.gpiResourceYear,
        venue: element.dataset.gpiResourceVenue,
        relevance: element.dataset.gpiResourceRelevance,
        doi: element.dataset.gpiResourceDoi,
        pmid: element.dataset.gpiResourcePmid,
        study_context_id: element.dataset.gpiResourceStudy,
        study_revision: element.dataset.gpiResourceRevision,
        route: element.dataset.gpiResourceRoute,
        state: element.dataset.gpiResourceState,
        job_id: element.dataset.gpiResourceJob,
        view: element.dataset.gpiResourceView,
        snapshot_sha256: element.dataset.gpiResourceDigest,
        review_sha256: element.dataset.gpiResourceDigest,
        checked_sha256: element.dataset.gpiResourceDigest,
        sha256: element.dataset.gpiResourceDigest,
      };
    }

    return { name, key, label, button, fromButton };
  }

  window.EU_GUIDED_PI_RESOURCES = { create };
})();
