/* Guided Copilot article-report owner.
   Adds the verified presentation gallery to the existing evidence-bound
   manuscript reader without changing claim or evidence bindings. */
(function () {
  'use strict';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  async function load(api, projectId, runId, resource) {
    if (!api || typeof api.loadPiCopilotResearchArtifact !== 'function') {
      throw new Error(tr('The research artifact API is unavailable.', '研究产物接口不可用。'));
    }
    const loader = window.EasyICU.guidedPi.require('reportArtifacts');
    if (!loader || typeof loader.load !== 'function') throw new Error('The report artifact loader is unavailable.');
    const rows = await loader.load(
      api, projectId, runId,
      ['manuscript_provenance.json', 'figure_gallery.json'],
      resource, ['figure_gallery.json'],
    );
    const provenance = rows['manuscript_provenance.json'];
    const gallery = rows['figure_gallery.json'];
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

  window.EasyICU.guidedPi.declare('articleReport', { load, render });
})();
