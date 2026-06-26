/* ============================================================
   screens-viz-demo.js — demo / fixture data layer for the
   visualization screens (patient / cohort / cross-DB).

   Split out of screens-viz.js 2026-06-26 as the first owner-file
   carve-out (see file-size budget rule in CLAUDE.md / AGENTS.md).
   Pure deterministic demo-data generators + their DEMO_* constants;
   the only external dependency is the global window.t (i18n).

   Exposed via window.VIZ_DEMO; screens-viz.js rebinds the names at
   the top of its IIFE so call sites stay unchanged. This file MUST
   load before screens-viz.js in index.html.
   ============================================================ */
(function () {
  const t = window.t;

  // Catalog metadata accessors — read-only lookups over the global
  // window.EU_CATALOG; shared by the demo generators and the main viz
  // screens (rebound there). Pure: only depend on EU_CATALOG + t.
  function catalogModuleLabel(key) {
    const hit = ((window.EU_CATALOG || {}).groups || []).find(row => row[0] === key);
    return hit ? t(hit[1], hit[2] || hit[1]) : key;
  }
  function catalogFeatureMeta(key) {
    const hit = ((window.EU_CATALOG || {}).dict || {})[key];
    return { name: hit ? hit[0] : key, unit: hit ? hit[2] : '' };
  }

  const DEMO_ENTITY_COUNT = 48;
  const DEMO_DURATION_HOURS = 48;
  const DEMO_CLINICAL_LANES = {
    vitals: ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    labs: ['lact', 'crea', 'bili', 'plt', 'hgb', 'wbc', 'inr_pt', 'glu', 'k', 'na', 'alb', 'crp', 'tnt', 'ph', 'po2', 'pco2'],
    interventions: ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'fio2', 'peep', 'ins', 'abx', 'cort', 'rrt'],
    scores: ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'mews', 'news', 'pafi', 'safi'],
  };
  const DEMO_THRESHOLDS = {
    hr: [[60, 'Bradycardia'], [100, 'Tachycardia']],
    map: [[65, 'Hypotension']],
    sbp: [[90, 'Hypotension'], [140, 'Hypertension']],
    spo2: [[94, 'Hypoxemia']],
    temp: [[36, 'Hypothermia'], [38, 'Fever']],
    resp: [[12, 'Bradypnea'], [20, 'Tachypnea']],
    lact: [[2, 'Elevated']],
    crea: [[1.2, 'Elevated']],
    ph: [[7.35, 'Acidosis'], [7.45, 'Alkalosis']],
    glu: [[70, 'Hypoglycemia'], [180, 'Hyperglycemia']],
    k: [[3.5, 'Hypokalemia'], [5, 'Hyperkalemia']],
    na: [[135, 'Hyponatremia'], [145, 'Hypernatremia']],
    plt: [[150, 'Thrombocytopenia']],
    hgb: [[7, 'Severe anemia']],
    inr_pt: [[1.5, 'Coagulopathy']],
    pafi: [[300, 'Mild ARDS'], [200, 'Moderate ARDS'], [100, 'Severe ARDS']],
    bili: [[1.2, 'Elevated']],
    sofa: [[2, 'Organ dysfunction']],
    sofa2: [[2, 'Organ dysfunction']],
    gcs: [[8, 'Severe impairment']],
    qsofa: [[2, 'Positive qSOFA']],
  };

  function demoCatalogModules() {
    const cat = window.EU_CATALOG || {};
    const groups = Array.isArray(cat.groups) ? cat.groups : [];
    const byGroup = cat.groupConcepts || {};
    return groups.map(row => {
      const key = row && row[0];
      const features = (byGroup[key] || []).filter(Boolean);
      return { module: key, label: catalogModuleLabel(key), features };
    }).filter(row => row.module && row.features.length);
  }
  function demoIsTimeIndexed(module) {
    return !['demographics', 'outcome', 'sepsis3_sofa1', 'sepsis3_sofa2'].includes(module);
  }
  function demoFeatureModule(feature) {
    const modules = demoCatalogModules();
    const hit = modules.find(m => (m.features || []).includes(feature));
    return hit ? hit.module : 'catalog';
  }
  function demoCoverageForFeature(feature, offset) {
    const cov = ((window.EU_CATALOG || {}).conceptCoverage || {})[feature] || {};
    const dbs = Number(cov.databases);
    const support = Number.isFinite(dbs) ? dbs : (cov.kind === 'derived' ? 5 : 4);
    return Math.max(52, Math.min(100, 58 + support * 6 + ((offset || 0) % 7)));
  }
  function demoRowsForModule(module, featureCount, coverage) {
    const entityRows = Math.max(1, Math.round(DEMO_ENTITY_COUNT * (coverage || 90) / 100));
    if (!demoIsTimeIndexed(module)) return entityRows;
    const density = Math.max(1, Math.min(6, Math.ceil(featureCount / 8)));
    return entityRows * DEMO_DURATION_HOURS * density;
  }
  function demoReviewStatus(coverage, featureCount) {
    if (!featureCount) return 'empty';
    if (coverage >= 80) return 'ready';
    if (coverage >= 50) return 'partial';
    return 'sparse';
  }
  function demoQualityStatus(missing, outlier, duplicate) {
    if ((missing || 0) >= 50 || (outlier || 0) >= 5 || (duplicate || 0) >= 2) return 'bad';
    if ((missing || 0) >= 20 || (outlier || 0) >= 1 || (duplicate || 0) >= 0.5) return 'warn';
    return 'ok';
  }
  function demoRateTone(value, warn, danger) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 'neutral';
    if (n >= danger) return 'bad';
    if (n >= warn) return 'warn';
    return 'ok';
  }
  function demoThresholds(feature) {
    return (DEMO_THRESHOLDS[feature] || []).map(row => ({ value: row[0], label: row[1] }));
  }
  function demoBaseValue(feature, entityIndex) {
    const idx = entityIndex || 0;
    const map = {
      hr: 88, map: 74, sbp: 112, dbp: 61, pulse_pressure: 51, temp: 37.1, spo2: 96, resp: 20,
      lact: 2.3, crea: 1.25, bili: 1.1, plt: 168, hgb: 10.2, wbc: 11.5, inr_pt: 1.25,
      glu: 146, k: 4.1, na: 138, alb: 3.0, crp: 82, tnt: 0.03, ph: 7.38, po2: 86, pco2: 42,
      fio2: 42, peep: 8, norepi_rate: 0.08, epi_rate: 0.02, dopa_rate: 3.0, dobu_rate: 2.0, ins: 2.4,
      sofa: 6, sofa2: 7, qsofa: 2, sirs: 3, gcs: 12, mews: 5, news: 6, pafi: 214, safi: 246,
      abx: 1, cort: 1, rrt: 0, mech_vent: 1, vent_ind: 1,
    };
    const base = Object.prototype.hasOwnProperty.call(map, feature) ? map[feature] : 1 + ((feature || '').length % 9);
    return base + (idx % 4) * 0.45;
  }
  function demoSignal(feature, entityIndex) {
    const meta = catalogFeatureMeta(feature);
    const values = Array.from({ length: 12 }, (_, i) => {
      const base = demoBaseValue(feature, entityIndex);
      if (meta.unit === 'boolean') return (i + (entityIndex || 0)) % 5 === 0 ? 0 : 1;
      const wave = Math.sin((i + 1 + (entityIndex || 0)) / 2.1);
      const drift = (i - 5) * 0.06;
      const scale = Math.max(0.08, Math.abs(base) * 0.045);
      return Number((base + wave * scale + drift).toFixed(feature === 'ph' || base < 1 ? 2 : 1));
    });
    return {
      key: feature,
      feature,
      name: meta.name || feature,
      unit: meta.unit || '',
      values,
      point_count: values.length,
      current: values[values.length - 1],
      min: Math.min(...values),
      max: Math.max(...values),
      mean: Number((values.reduce((a, b) => a + b, 0) / values.length).toFixed(2)),
      thresholds: demoThresholds(feature),
      bounded: true,
      max_points: 12,
    };
  }
  function demoTimeLanes(entityIndex) {
    const catFeatures = new Set(Object.values((window.EU_CATALOG || {}).groupConcepts || {}).flat());
    return Object.entries(DEMO_CLINICAL_LANES).map(([lane, features]) => {
      const signals = features.filter(f => catFeatures.has(f)).map(f => demoSignal(f, entityIndex));
      return {
        lane,
        label: lane.replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase()),
        signal_count: signals.length,
        signals,
        status: signals.length ? 'ready' : 'unavailable',
      };
    });
  }
  function demoSignalDelta(values) {
    return values && values.length > 1 ? Number((values[values.length - 1] - values[0]).toFixed(2)) : null;
  }
  function demoFeatureTone(feature, value) {
    if (['sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd'].includes(feature)) return value >= 1 ? 'bad' : 'ok';
    if (['mech_vent', 'vent_ind', 'rrt', 'vaso_ind'].includes(feature)) return value >= 1 ? 'warn' : 'ok';
    if (['sofa', 'sofa2', 'qsofa', 'sirs', 'mews', 'news'].includes(feature)) {
      return value < 6 ? 'ok' : (value < 10 ? 'warn' : 'bad');
    }
    return 'neutral';
  }
  function demoCategorySection(id, title, features, signalIndex) {
    const cards = features.map(feature => {
      const signal = signalIndex[feature];
      if (!signal) return null;
      return {
        feature,
        label: signal.name || feature,
        unit: signal.unit || '',
        current: signal.current,
        delta: demoSignalDelta(signal.values || []),
        tone: demoFeatureTone(feature, signal.current),
        values: (signal.values || []).slice(0, 12),
        thresholds: signal.thresholds || [],
      };
    }).filter(Boolean);
    return { id, title, available_count: cards.length, cards };
  }
  function demoQualityPanelRows(rows, metric) {
    return rows.slice(0, 8).map(row => ({
      feature: row.feature,
      name: row.name,
      module: row.module,
      value: row[metric],
      records: row.records,
      entities: row.entities,
      status: row.status,
    }));
  }

  window.VIZ_DEMO = {
    catalogModuleLabel, catalogFeatureMeta,
    DEMO_ENTITY_COUNT, DEMO_DURATION_HOURS, DEMO_CLINICAL_LANES, DEMO_THRESHOLDS,
    demoCatalogModules, demoIsTimeIndexed, demoFeatureModule, demoCoverageForFeature,
    demoRowsForModule, demoReviewStatus, demoQualityStatus, demoRateTone,
    demoThresholds, demoBaseValue, demoSignal, demoTimeLanes, demoSignalDelta,
    demoFeatureTone, demoCategorySection, demoQualityPanelRows,
  };
})();
