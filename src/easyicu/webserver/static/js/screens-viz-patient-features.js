/* ============================================================
   screens-viz-patient-features.js — Patient Review feature-lane
   ownership boundary.

   Public contract:
     window.EU_PATIENT_FEATURES.signalKey(signal) -> string
     window.EU_PATIENT_FEATURES.catalogLanes(lanes) -> lane[]

   The owner reads the immutable catalog view exposed by VIZ_DEMO.
   Rendering remains in screens-viz-patient-series.js and route state
   remains in screens-viz.js.
   ============================================================ */
(function () {
  'use strict';

  function signalKey(signal) {
    return String((signal && (signal.feature || signal.key || signal.name)) || '').toLowerCase();
  }

  function signalAvailability(signal) {
    if (!signal) return 'metadata_only';
    const numericCount = ((signal && signal.values) || [])
      .filter(value => value != null && value !== '')
      .map(Number)
      .filter(Number.isFinite)
      .length;
    return numericCount >= 2 ? 'numeric_trajectory' : 'observed_categorical';
  }

  function catalogLanes(lanes, coverage, lazyStateFor) {
    const demoCatalog = window.VIZ_DEMO;
    const sourceLanes = Array.isArray(lanes) ? lanes : [];
    const byFeature = new Map();
    const coverageByFeature = new Map();
    ((coverage && coverage.modules) || []).forEach(module => {
      ((module && module.features) || []).forEach(feature => {
        if (feature && feature.feature) coverageByFeature.set(feature.feature, feature);
      });
    });

    sourceLanes.forEach(lane => {
      ((lane && lane.signals) || []).forEach(signal => {
        const key = signalKey(signal);
        if (key && !byFeature.has(key)) byFeature.set(key, signal);
      });
    });

    const catalogLanes = demoCatalog.demoCatalogModules().map(moduleRow => {
      const existing = sourceLanes.find(row => row && row.lane === moduleRow.module) || {};
      const features = moduleRow.features.map(feature => {
        const meta = demoCatalog.catalogFeatureMeta(feature);
        const signal = byFeature.get(feature);
        const availability = signalAvailability(signal);
        const exportCoverage = coverageByFeature.get(feature) || {};
        const lazy = typeof lazyStateFor === 'function'
          ? (lazyStateFor(feature) || {})
          : {};
        let status = availability;
        if (!signal && exportCoverage.status === 'observed') {
          status = exportCoverage.trajectory_candidate
            ? 'available_unloaded'
            : (exportCoverage.numeric ? 'observed_static' : 'observed_categorical');
        } else if (!signal && exportCoverage.status) {
          status = exportCoverage.status;
        }
        if (!signal && lazy.payload) {
          status = lazy.payload.status === 'unavailable'
            ? 'selected_entity_unavailable'
            : String(lazy.payload.status || status);
        }
        return {
          feature,
          name: meta.name || feature,
          unit: meta.unit || '',
          observed: Boolean(signal) || exportCoverage.status === 'observed',
          export_observed: exportCoverage.status === 'observed',
          status,
          trajectory: availability === 'numeric_trajectory',
          trajectory_candidate: Boolean(exportCoverage.trajectory_candidate),
          loadable: Boolean(exportCoverage.loadable),
          materialized: exportCoverage.materialized,
          non_null_count: exportCoverage.non_null_count,
          reason_code: (lazy.payload && lazy.payload.reason_code) || exportCoverage.reason_code,
          observation: (lazy.payload && lazy.payload.observation) || null,
          loading: Boolean(lazy.loading),
          load_error: lazy.error || '',
          lazy_loaded: Boolean(lazy.loaded),
        };
      });
      const signals = moduleRow.features.map(feature => byFeature.get(feature)).filter(Boolean);
      const numericTrajectoryCount = features.filter(feature => feature.trajectory).length;
      return {
        ...existing,
        lane: moduleRow.module,
        label: moduleRow.label,
        signal_count: features.length,
        available_signal_count: numericTrajectoryCount,
        observed_signal_count: signals.length,
        export_observed_count: features.filter(feature => feature.export_observed).length,
        numeric_trajectory_count: numericTrajectoryCount,
        features,
        signals,
        status: numericTrajectoryCount ? 'ready' : (signals.length ? 'observed' : 'metadata_only'),
      };
    });

    const catalogFeatureKeys = new Set(
      catalogLanes.flatMap(row => row.features.map(feature => feature.feature)),
    );
    const uncatalogued = [];
    sourceLanes.forEach(lane => {
      const signals = ((lane && lane.signals) || [])
        .filter(signal => !catalogFeatureKeys.has(signalKey(signal)));
      if (signals.length) {
        uncatalogued.push({ ...lane, signals, signal_count: signals.length });
      }
    });
    return catalogLanes.concat(uncatalogued);
  }

  window.EU_PATIENT_FEATURES = Object.freeze({
    signalKey,
    signalAvailability,
    catalogLanes,
  });
})();
