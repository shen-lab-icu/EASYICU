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
  const DEMO_CHART_HOURS = [0.2, 1, 2, 3.4, 6, 8, 12, 18, 24, 30, 36, 48];
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
    const density = Math.max(1, Math.min(3, Math.ceil(featureCount / 12)));
    return entityRows * DEMO_CHART_HOURS.length * density;
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
  const DEMO_BOOLEAN_FEATURES = new Set([
    'abx', 'cort', 'rrt', 'mech_vent', 'vent_ind', 'vaso_ind',
    'sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd', 'death',
    'mort_icu', 'mort_hosp', 'mort_28d',
  ]);
  const DEMO_INTEGER_TOTAL_SCORES = new Set(['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'mews', 'news']);
  const DEMO_SCORED_COMPONENTS = new Set([
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal',
    'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
  ]);
  function demoIsBooleanFeature(feature, unit) {
    const key = String(feature || '').toLowerCase();
    return String(unit || '').toLowerCase() === 'boolean'
      || DEMO_BOOLEAN_FEATURES.has(key)
      || key.endsWith('_ind')
      || key.endsWith('60')
      || key.endsWith('90');
  }
  function demoIsIntegerFeature(feature) {
    const key = String(feature || '').toLowerCase();
    return DEMO_INTEGER_TOTAL_SCORES.has(key) || DEMO_SCORED_COMPONENTS.has(key);
  }
  function demoClamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }
  function demoNormalizeValue(feature, raw, unit) {
    const key = String(feature || '').toLowerCase();
    if (demoIsBooleanFeature(key, unit)) return raw >= 0.5 ? 1 : 0;
    if (DEMO_SCORED_COMPONENTS.has(key)) return demoClamp(Math.round(raw), 0, 4);
    if (key === 'qsofa') return demoClamp(Math.round(raw), 0, 3);
    if (key === 'sirs') return demoClamp(Math.round(raw), 0, 4);
    if (key === 'gcs') return demoClamp(Math.round(raw), 3, 15);
    if (key === 'sofa' || key === 'sofa2') return demoClamp(Math.round(raw), 0, 24);
    if (key === 'mews' || key === 'news') return demoClamp(Math.round(raw), 0, 20);
    if (demoIsIntegerFeature(key)) return Math.round(raw);
    if (key === 'ph') return Number(demoClamp(raw, 6.8, 7.8).toFixed(2));
    if (key === 'spo2') return Number(demoClamp(raw, 70, 100).toFixed(1));
    if (key === 'fio2') return Number(demoClamp(raw, 21, 100).toFixed(1));
    if (key === 'pafi' || key === 'safi') return Number(demoClamp(raw, 40, 600).toFixed(1));
    if (key.includes('rate') || key.includes('dur') || key === 'peep' || key === 'ins') {
      return Number(Math.max(0, raw).toFixed(Math.abs(raw) < 1 ? 2 : 1));
    }
    const nonNegativeUnits = ['mg/dl', 'mmol/l', 'x10', 'mcg/kg/min', 'hours', 'ratio', 'bpm', 'mmhg', '%', 'breaths/min'];
    const unitKey = String(unit || '').toLowerCase();
    const bounded = nonNegativeUnits.some(part => unitKey.includes(part)) || Math.abs(raw) < 1;
    const value = bounded ? Math.max(0, raw) : raw;
    if (Math.abs(value) < 1) return Number(value.toFixed(2));
    return Number(value.toFixed(1));
  }
  function demoCharttimeAt(rowIndex) {
    const idx = Math.max(0, Number(rowIndex) || 0);
    const cycle = Math.floor(idx / DEMO_CHART_HOURS.length);
    const hour = DEMO_CHART_HOURS[idx % DEMO_CHART_HOURS.length] + cycle * DEMO_DURATION_HOURS;
    return Number(hour.toFixed(2));
  }
  function demoBaselineDrift(feature, base, entityIndex) {
    const idx = Math.max(0, Number(entityIndex) || 0);
    const key = String(feature || '').toLowerCase();
    if (demoIsIntegerFeature(key)) return (idx % 3) - 1;
    if (key === 'ph') return ((idx % 4) - 1.5) * 0.02;
    if (key === 'temp') return ((idx % 4) - 1.5) * 0.08;
    if (Math.abs(base) < 1) return ((idx % 4) - 1.5) * 0.01;
    if (['spo2', 'fio2', 'peep'].includes(key)) return (idx % 4) * 0.4;
    return (idx % 4) * 0.45;
  }
  function demoBaseValue(feature, entityIndex) {
    const idx = entityIndex || 0;
    const key = String(feature || '').toLowerCase();
    const map = {
      hr: 88, map: 74, sbp: 112, dbp: 61, pulse_pressure: 51, temp: 37.1, spo2: 96, resp: 20,
      lact: 2.3, crea: 1.25, bili: 1.1, plt: 168, hgb: 10.2, wbc: 11.5, inr_pt: 1.25,
      glu: 146, k: 4.1, na: 138, alb: 3.0, crp: 82, tnt: 0.03, ph: 7.38, po2: 86, pco2: 42,
      fio2: 42, peep: 8, norepi_rate: 0.08, epi_rate: 0.02, dopa_rate: 3.0, dobu_rate: 2.0, ins: 2.4,
      sofa: 6, sofa2: 7, qsofa: 2, sirs: 3, gcs: 12, mews: 5, news: 6, pafi: 214, safi: 246,
      sofa_resp: 2, sofa_coag: 1, sofa_liver: 1, sofa_cardio: 3, sofa_cns: 1, sofa_renal: 2,
      sofa2_resp: 2, sofa2_coag: 1, sofa2_liver: 1, sofa2_cardio: 3, sofa2_cns: 1, sofa2_renal: 2,
      abx: 1, cort: 1, rrt: 0, mech_vent: 1, vent_ind: 1,
      sep3_sofa1: 1, sep3_sofa2: 1, susp_inf: 1, infection_icd: 1, death: 0,
    };
    const base = Object.prototype.hasOwnProperty.call(map, key) ? map[key] : 1 + (key.length % 9);
    const drift = demoBaselineDrift(key, base, idx);
    return demoNormalizeValue(key, base + drift, catalogFeatureMeta(key).unit);
  }
  function demoTableValue(feature, entityIndex) {
    const meta = catalogFeatureMeta(feature);
    const key = String(feature || '').toLowerCase();
    if (demoIsBooleanFeature(key, meta.unit)) {
      return ((entityIndex || 0) + key.length) % 3 === 0;
    }
    return demoBaseValue(key, entityIndex);
  }
  function demoSignal(feature, entityIndex) {
    const meta = catalogFeatureMeta(feature);
    const key = String(feature || '').toLowerCase();
    const values = Array.from({ length: 12 }, (_, i) => {
      const base = demoBaseValue(feature, entityIndex);
      if (demoIsBooleanFeature(key, meta.unit)) return (i + (entityIndex || 0)) % 5 === 0 ? 0 : 1;
      const wave = Math.sin((i + 1 + (entityIndex || 0)) / 2.1);
      const smallDose = Math.abs(base) < 1;
      const drift = (i - 5) * (smallDose ? 0.006 : 0.06);
      const scale = Math.max(smallDose ? 0.008 : 0.08, Math.abs(base) * 0.045);
      return demoNormalizeValue(key, base + wave * scale + drift, meta.unit);
    });
    return {
      key: feature,
      feature,
      name: meta.name || feature,
      unit: meta.unit || '',
      values,
      times: DEMO_CHART_HOURS.slice(0, values.length),
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
    DEMO_ENTITY_COUNT, DEMO_DURATION_HOURS, DEMO_CHART_HOURS, DEMO_CLINICAL_LANES, DEMO_THRESHOLDS,
    demoCatalogModules, demoIsTimeIndexed, demoFeatureModule, demoCoverageForFeature,
    demoRowsForModule, demoReviewStatus, demoQualityStatus, demoRateTone,
    demoThresholds, demoBaseValue, demoTableValue, demoCharttimeAt, demoSignal, demoTimeLanes, demoSignalDelta,
    demoFeatureTone, demoCategorySection, demoQualityPanelRows,
  };
})();
