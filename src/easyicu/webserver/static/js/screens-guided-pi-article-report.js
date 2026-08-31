/* Guided Copilot article-report owner.
   Adds the verified presentation gallery to the existing evidence-bound
   manuscript reader without changing claim or evidence bindings. */
(function () {
  'use strict';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  async function load(api, projectId, runId) {
    if (!api || typeof api.loadPiCopilotResearchArtifact !== 'function') {
      throw new Error(tr('The research artifact API is unavailable.', '研究产物接口不可用。'));
    }
    const [provenance, gallery] = await Promise.all([
      api.loadPiCopilotResearchArtifact(projectId, runId, 'manuscript_provenance.json'),
      api.loadPiCopilotResearchArtifact(projectId, runId, 'figure_gallery.json').catch(() => null),
    ]);
    return {
      payload: {
        ...((provenance && provenance.payload) || {}),
        figure_gallery: (gallery && gallery.payload) || {},
        report_variant: 'article_with_figures',
      },
      governance: (provenance && provenance.governance) || null,
    };
  }

  function render(payload) {
    const renderer = window.AGENT_RENDER;
    if (!renderer || typeof renderer.manuscriptProvenanceView !== 'function') {
      throw new Error(tr('The evidence-bound article renderer is unavailable.', '证据绑定文章渲染器不可用。'));
    }
    return renderer.manuscriptProvenanceView(payload || {});
  }

  window.EU_GUIDED_PI_ARTICLE_REPORT = { load, render };
})();
