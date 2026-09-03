/* Shared visualization context contract.
   The large route owner injects its private state once; Copilot and
   StudyContext consume only this dependency-neutral public surface. */
(function () {
  'use strict';

  const ROUTES = new Set(['patient', 'cohort', 'crossdb']);

  function unique(values) {
    return Array.from(new Set((values || []).filter(Boolean)));
  }

  function init(host) {
    const owner = host || {};
    window.EU_VIZ_CONTEXT = {
      snapshot(route) {
        const activePath = owner.activePath ? owner.activePath() : '';
        const sources = owner.sources ? owner.sources() : [];
        const source = sources.find(row => row.path === activePath) || sources[0] || {};
        const dataSource = {
          path: source.path || activePath || (window.EU_DATA === 'real' && owner.defaultExportPath ? owner.defaultExportPath() : ''),
          label: source.label || (window.EU_DATA === 'real' ? 'Local EasyICU export' : 'Demo data'),
          database: source.database || (window.EU_DATA === 'real' ? '' : 'demo'),
        };
        if (route === 'patient') {
          const drill = owner.patient ? (owner.patient() || {}) : {};
          const summary = drill.summary || {};
          return {
            data_source: dataSource,
            cohort: {
              entity_count: summary.entities,
              full_entity_count: summary.entities,
              review_entities: summary.review_entities,
              review_entity_cap: summary.review_entity_cap,
              review_scope: summary.review_scope,
              module_count: summary.modules,
            },
            modules: unique((drill.module_profiles || []).map(row => row.module || row.id)),
          };
        }
        if (route === 'cohort') {
          const review = owner.cohort ? (owner.cohort() || {}) : {};
          const selected = review.feature_selection && review.feature_selection.selected || [];
          const catalog = review.feature_catalog && review.feature_catalog.modules || [];
          return {
            data_source: dataSource,
            cohort: {
              cohort_size: review.summary && review.summary.cohort_size,
              comparison: owner.cohortComparison ? owner.cohortComparison() : 'outcome',
            },
            modules: unique(selected.map(row => row.module).concat(catalog.map(row => row.module))),
            outcome: owner.cohortOutcome ? owner.cohortOutcome() : 'mort_28d',
            comparator: owner.cohortComparison ? owner.cohortComparison() : 'outcome',
          };
        }
        const crossdb = owner.crossdb ? (owner.crossdb() || {}) : {};
        return {
          data_source: null,
          crossdb_selection: crossdb.selection_receipt || {},
          cohort: {
            source_count: crossdb.source_count,
            source_type: crossdb.source_type,
            comparison_mode: crossdb.compatibility_gate && crossdb.compatibility_gate.comparison_mode,
          },
          modules: unique(crossdb.shared_modules || []),
          comparator: 'cross_database_descriptive',
        };
      },
      hydratePreview(route, payload) {
        const cleanRoute = String(route || '');
        const hydrators = owner.hydrate || {};
        const legacy = typeof hydrators === 'function';
        const hydrate = legacy ? hydrators : hydrators[cleanRoute];
        if (!ROUTES.has(cleanRoute) || !payload || typeof payload !== 'object' || typeof hydrate !== 'function') return false;
        if (legacy) hydrate(cleanRoute, payload); else hydrate(payload);
        return true;
      },
      renderCohortPanel(payload, panel) {
        const panels = owner.cohortPanels || {};
        const render = panels[String(panel || 'groups')] || panels.groups;
        if (owner.cohortBegin) owner.cohortBegin();
        return typeof render === 'function' ? render(payload || {}) : '';
      },
      mountCohortCharts(root) {
        return typeof owner.cohortMount === 'function' ? owner.cohortMount(root) : 0;
      },
      patientSeriesHelpers() {
        return typeof owner.patientSeriesHelpers === 'function' ? owner.patientSeriesHelpers() : (owner.patientSeriesHelpers || {});
      },
      crossdbResultsConfig(repaint) {
        return typeof owner.crossdbResultsConfig === 'function' ? owner.crossdbResultsConfig(repaint) : { repaint };
      },
    };
    return window.EU_VIZ_CONTEXT;
  }

  window.EU_VIZ_CONTEXT_OWNER = { init };
})();
