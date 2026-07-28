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
    const name = hit
      ? (window.EU_LANG === 'zh' ? (hit[1] || hit[0]) : hit[0])
      : key;
    return { name, unit: hit ? hit[2] : '' };
  }

  const DEMO_ENTITY_COUNT = 48;
  const DEMO_DURATION_HOURS = 48;
  // Irregular, bounded offsets mirror how ICU exports are actually sampled:
  // dense near admission, then progressively wider gaps. These are synthetic
  // relative hours; no real patient timestamps or rows are embedded here.
  const DEMO_CHART_HOURS = [-1, 0.5, 2, 4, 7, 11, 16, 22, 29, 36, 43, 48];
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
  const DEMO_SCENARIOS = [
    'shock_recovery',
    'respiratory_recovery',
    'late_deterioration',
    'renal_dominant',
    'stable_recovery',
  ];
  const DEMO_HIGH_FREQUENCY = new Set([
    'hr', 'map', 'sbp', 'dbp', 'pulse_pressure', 'spo2', 'o2sat', 'sao2', 'resp',
    'shock_index', 'modified_shock_index', 'diastolic_shock_index',
  ]);
  const DEMO_RESPIRATORY_FEATURES = new Set([
    'pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'o2sat', 'sao2', 'mech_vent',
    'ett_gcs', 'adv_resp', 'oxygenation_index', 'peep', 'tidal_vol', 'tidal_vol_set',
    'pip', 'plateau_pres', 'mean_airway_pres', 'minute_vol', 'vent_rate', 'etco2',
    'compliance', 'driving_pres', 'ps', 'driving_pres_controlled',
  ]);
  const DEMO_BLOOD_GAS_FEATURES = new Set(['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2']);
  const DEMO_SCORE_FEATURES = new Set([
    'sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'mews', 'news',
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal',
    'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
  ]);
  const DEMO_INTERVENTION_FEATURES = new Set(DEMO_CLINICAL_LANES.interventions);

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
    'abx', 'cort', 'rrt', 'vent_ind', 'vaso_ind',
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
      || key.endsWith('_ind');
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
    if (raw == null) return null;
    if (typeof raw === 'string') return raw;
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
  function demoHash(feature, entityIndex, timeIndex) {
    const token = `${feature}|${Number(entityIndex) || 0}|${Number(timeIndex) || 0}`;
    let hash = 2166136261;
    for (let i = 0; i < token.length; i += 1) {
      hash ^= token.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0) / 4294967295;
  }
  function demoJitter(feature, entityIndex, timeIndex, amplitude) {
    return (demoHash(feature, entityIndex, timeIndex) * 2 - 1) * amplitude;
  }
  function demoStep(value, increment) {
    const step = Math.max(Number(increment) || 1, 0.0001);
    return Math.round(value / step) * step;
  }
  function demoScenarioName(entityIndex) {
    const index = Math.max(0, Number(entityIndex) || 0);
    return DEMO_SCENARIOS[index % DEMO_SCENARIOS.length];
  }
  function demoScenarioState(entityIndex, hour) {
    const index = Math.max(0, Number(entityIndex) || 0);
    const scenario = demoScenarioName(index);
    const h = demoClamp(Number(hour) || 0, 0, DEMO_DURATION_HOURS);
    const progress = h / DEMO_DURATION_HOURS;
    const recovery = 1 - progress;
    const lateRamp = demoClamp((h - 16) / 32, 0, 1);
    const entityShift = ((index % 7) - 3) * 0.018;
    let state;
    if (scenario === 'shock_recovery') {
      state = {
        shock: 0.92 * recovery + 0.08,
        respiratory: 0.68 * recovery + 0.22,
        renal: 0.56 - 0.12 * progress,
        inflammation: 0.88 * recovery + 0.16,
        neuro: 0.55 * recovery + 0.12,
        hepatic: 0.42 * recovery + 0.12,
        coagulation: 0.46 * recovery + 0.1,
      };
    } else if (scenario === 'respiratory_recovery') {
      state = {
        shock: 0.27 * recovery + 0.08,
        respiratory: 0.96 * recovery + 0.28,
        renal: 0.3 * recovery + 0.14,
        inflammation: 0.62 * recovery + 0.14,
        neuro: 0.34 * recovery + 0.08,
        hepatic: 0.22 * recovery + 0.08,
        coagulation: 0.25 * recovery + 0.08,
      };
    } else if (scenario === 'late_deterioration') {
      state = {
        shock: 0.18 + 0.78 * lateRamp,
        respiratory: 0.3 + 0.58 * lateRamp,
        renal: 0.2 + 0.62 * lateRamp,
        inflammation: 0.38 + 0.48 * lateRamp,
        neuro: 0.16 + 0.58 * lateRamp,
        hepatic: 0.16 + 0.48 * lateRamp,
        coagulation: 0.18 + 0.54 * lateRamp,
      };
    } else if (scenario === 'renal_dominant') {
      state = {
        shock: 0.42 * recovery + 0.16,
        respiratory: 0.38 * recovery + 0.2,
        renal: 0.7 + 0.2 * progress,
        inflammation: 0.52 * recovery + 0.18,
        neuro: 0.26 + 0.14 * progress,
        hepatic: 0.22 + 0.12 * progress,
        coagulation: 0.25 + 0.12 * progress,
      };
    } else {
      state = {
        shock: 0.24 * recovery + 0.06,
        respiratory: 0.28 * recovery + 0.1,
        renal: 0.24 * recovery + 0.1,
        inflammation: 0.36 * recovery + 0.1,
        neuro: 0.2 * recovery + 0.05,
        hepatic: 0.16 * recovery + 0.05,
        coagulation: 0.18 * recovery + 0.05,
      };
    }
    Object.keys(state).forEach(key => {
      state[key] = demoClamp(state[key] + entityShift, 0.02, 1);
    });
    state.scenario = scenario;
    state.hour = h;
    state.progress = progress;
    return state;
  }
  function demoSofaComponents(state, version) {
    const offset = version === 2 ? 0.08 : 0;
    return {
      resp: demoClamp(Math.round((state.respiratory + offset) * 4), 0, 4),
      coag: demoClamp(Math.round(state.coagulation * 4), 0, 4),
      liver: demoClamp(Math.round(state.hepatic * 4), 0, 4),
      cardio: demoClamp(Math.round((state.shock + offset) * 4), 0, 4),
      cns: demoClamp(Math.round(state.neuro * 4), 0, 4),
      renal: demoClamp(Math.round(state.renal * 4), 0, 4),
    };
  }
  function demoCorePhysiology(entityIndex, timeIndex) {
    const index = Math.max(0, Number(entityIndex) || 0);
    const ti = demoClamp(Math.round(Number(timeIndex) || 0), 0, DEMO_CHART_HOURS.length - 1);
    const hour = DEMO_CHART_HOURS[ti];
    const state = demoScenarioState(index, hour);
    const hr = 70 + 44 * state.shock + 12 * state.inflammation + demoJitter('hr', index, ti, 3.2);
    const map = 88 - 39 * state.shock + demoJitter('map', index, ti, 2.2);
    const sbp = map + 34 + 8 * (1 - state.shock) + demoJitter('sbp', index, ti, 2.4);
    const dbp = map - 15 + demoJitter('dbp', index, ti, 1.7);
    const temp = 36.65 + 1.55 * state.inflammation - 0.25 * state.shock + demoJitter('temp', index, ti, 0.12);
    const resp = 14 + 15 * state.respiratory + 3 * state.shock + demoJitter('resp', index, ti, 1.1);
    const fio2 = demoStep(21 + 67 * state.respiratory, 5);
    const spo2 = 98.5 - 10.5 * state.respiratory + demoJitter('spo2', index, ti, 0.55);
    const pafi = 430 - 330 * state.respiratory + demoJitter('pafi', index, ti, 10);
    const gcs = demoClamp(15 - Math.round(9 * state.neuro), 3, 15);
    const lactate = 0.8 + 5.3 * state.shock + 0.35 * state.inflammation + demoJitter('lact', index, ti, 0.13);
    const creatinine = 0.65 + 2.75 * state.renal + demoJitter('crea', index, ti, 0.08);
    const bilirubin = 0.45 + 4.1 * state.hepatic + demoJitter('bili', index, ti, 0.1);
    const platelets = 275 - 190 * state.coagulation + demoJitter('plt', index, ti, 5);
    const wbc = 6.2 + 12.5 * state.inflammation + demoJitter('wbc', index, ti, 0.6);
    const sofa1 = demoSofaComponents(state, 1);
    const sofa2 = demoSofaComponents(state, 2);
    return {
      state, hr, map, sbp, dbp, temp, resp, fio2, spo2, pafi, gcs, lactate,
      creatinine, bilirubin, platelets, wbc, sofa1, sofa2,
    };
  }
  function demoClinicalValue(feature, entityIndex, timeIndex) {
    const key = String(feature || '').toLowerCase();
    const meta = catalogFeatureMeta(key);
    const p = demoCorePhysiology(entityIndex, timeIndex);
    const s = p.state;
    const t = s.progress;
    const sofa1 = p.sofa1;
    const sofa2 = p.sofa2;
    const sofaTotal = Object.values(sofa1).reduce((sum, value) => sum + value, 0);
    const sofa2Total = Object.values(sofa2).reduce((sum, value) => sum + value, 0);
    const qsofa = Number(p.map < 65) + Number(p.resp >= 22) + Number(p.gcs < 15);
    const sirs = Number(p.hr > 90) + Number(p.temp < 36 || p.temp > 38)
      + Number(p.resp > 20) + Number(p.wbc < 4 || p.wbc > 12);
    const values = {
      hr: p.hr,
      map: p.map,
      sbp: p.sbp,
      dbp: p.dbp,
      pulse_pressure: p.sbp - p.dbp,
      cvp: 6 + 7 * s.shock + demoJitter('cvp', entityIndex, timeIndex, 0.8),
      temp: p.temp,
      spo2: p.spo2,
      o2sat: p.spo2,
      sao2: p.spo2 + 0.4,
      resp: p.resp,
      shock_index: p.hr / p.sbp,
      modified_shock_index: p.hr / p.map,
      diastolic_shock_index: p.hr / p.dbp,
      lact: p.lactate,
      crea: p.creatinine,
      bili: p.bilirubin,
      plt: p.platelets,
      hgb: 11.3 - 1.2 * s.renal - 0.35 * t + demoJitter('hgb', entityIndex, timeIndex, 0.25),
      wbc: p.wbc,
      inr_pt: 0.95 + 1.25 * s.hepatic + 0.45 * s.coagulation + demoJitter('inr_pt', entityIndex, timeIndex, 0.04),
      glu: 98 + 82 * s.inflammation + 26 * s.shock + demoJitter('glu', entityIndex, timeIndex, 5),
      k: 3.75 + 0.95 * s.renal + demoJitter('k', entityIndex, timeIndex, 0.12),
      na: 140 - 4.5 * s.shock + demoJitter('na', entityIndex, timeIndex, 0.7),
      alb: 3.65 - 1.45 * s.inflammation - 0.25 * s.hepatic + demoJitter('alb', entityIndex, timeIndex, 0.06),
      crp: 8 + 205 * s.inflammation + demoJitter('crp', entityIndex, timeIndex, 5),
      tnt: 0.008 + 0.21 * s.shock + demoJitter('tnt', entityIndex, timeIndex, 0.008),
      ph: 7.43 - 0.18 * s.shock - 0.04 * s.respiratory + demoJitter('ph', entityIndex, timeIndex, 0.008),
      po2: p.pafi * (p.fio2 / 100),
      pco2: 38 + 16 * s.respiratory + demoJitter('pco2', entityIndex, timeIndex, 2),
      be: 1.5 - 9 * s.shock + demoJitter('be', entityIndex, timeIndex, 0.7),
      bicar: 24 - 8 * s.shock + demoJitter('bicar', entityIndex, timeIndex, 0.8),
      tco2: 25 - 7 * s.shock + demoJitter('tco2', entityIndex, timeIndex, 0.8),
      pafi: p.pafi,
      safi: 100 * p.spo2 / p.fio2,
      fio2: p.fio2,
      peep: demoStep(5 + 11 * s.respiratory, 1),
      tidal_vol: demoStep(470 - 70 * s.respiratory + demoJitter('tidal_vol', entityIndex, timeIndex, 18), 10),
      tidal_vol_set: demoStep(460 - 60 * s.respiratory, 10),
      pip: demoStep(18 + 17 * s.respiratory, 1),
      plateau_pres: demoStep(15 + 14 * s.respiratory, 1),
      mean_airway_pres: demoStep(8 + 12 * s.respiratory, 1),
      minute_vol: 6.2 + 5.2 * s.respiratory,
      vent_rate: demoStep(12 + 12 * s.respiratory, 1),
      etco2: 35 + 10 * s.respiratory,
      compliance: 62 - 40 * s.respiratory,
      driving_pres: 9 + 9 * s.respiratory,
      driving_pres_controlled: s.respiratory > 0.72 ? 9 + 9 * s.respiratory : null,
      ps: demoStep(5 + 9 * s.respiratory, 1),
      norepi_rate: demoStep(Math.max(0, (s.shock - 0.22) * 0.48), 0.02),
      norepi_equiv: demoStep(Math.max(0, (s.shock - 0.2) * 0.5), 0.02),
      epi_rate: demoStep(Math.max(0, (s.shock - 0.72) * 0.18), 0.01),
      dopa_rate: s.shock > 0.82 && (Number(entityIndex) || 0) % 4 === 1 ? 5 : 0,
      dobu_rate: s.shock > 0.72 && (Number(entityIndex) || 0) % 3 === 0 ? demoStep(1 + 4 * s.shock, 0.5) : 0,
      ins: demoStep(Math.max(0, (98 + 82 * s.inflammation - 125) / 25), 0.5),
      sofa: sofaTotal,
      sofa2: sofa2Total,
      sofa_resp: sofa1.resp,
      sofa_coag: sofa1.coag,
      sofa_liver: sofa1.liver,
      sofa_cardio: sofa1.cardio,
      sofa_cns: sofa1.cns,
      sofa_renal: sofa1.renal,
      sofa2_resp: sofa2.resp,
      sofa2_coag: sofa2.coag,
      sofa2_liver: sofa2.liver,
      sofa2_cardio: sofa2.cardio,
      sofa2_cns: sofa2.cns,
      sofa2_renal: sofa2.renal,
      qsofa,
      sirs,
      gcs: p.gcs,
      mews: demoClamp(Math.round(1 + 3 * s.shock + 3 * s.respiratory + 2 * s.neuro), 0, 14),
      news: demoClamp(Math.round(1 + 4 * s.shock + 4 * s.respiratory + 2 * s.neuro), 0, 20),
      abx: s.inflammation > 0.22 ? 1 : 0,
      cort: s.shock > 0.58 ? 1 : 0,
      rrt: s.renal > 0.78 ? 1 : 0,
      mech_vent: s.respiratory > 0.56
        ? 'invasive'
        : (s.respiratory > 0.35 ? 'noninvasive' : null),
      vent_ind: s.respiratory > 0.35 ? 1 : 0,
      supp_o2: s.respiratory > 0.35 || p.fio2 > 21 ? 1 : 0,
      adv_resp: s.respiratory > 0.35 ? 1 : 0,
      vaso_ind: s.shock > 0.28 ? 1 : 0,
      sep3_sofa1: sofaTotal >= 2 && s.inflammation > 0.28 ? 1 : 0,
      sep3_sofa2: sofa2Total >= 2 && s.inflammation > 0.28 ? 1 : 0,
      susp_inf: s.inflammation > 0.28 ? 1 : 0,
      infection_icd: s.inflammation > 0.34 ? 1 : 0,
      death: s.scenario === 'late_deterioration' ? 1 : 0,
      mort_icu: s.scenario === 'late_deterioration' ? 1 : 0,
      mort_hosp: s.scenario === 'late_deterioration' ? 1 : 0,
      mort_28d: s.scenario === 'late_deterioration' ? 1 : 0,
      mort_90d: s.scenario === 'late_deterioration' ? 1 : 0,
      mort_365d: s.scenario === 'late_deterioration' ? 1 : 0,
      age: 44 + ((Number(entityIndex) || 0) * 7) % 38,
      bmi: 22 + ((Number(entityIndex) || 0) % 6) * 1.7,
      height: 158 + ((Number(entityIndex) || 0) % 8) * 3,
      weight: 61 + ((Number(entityIndex) || 0) % 9) * 4,
      los_icu: 3.6 + s.renal * 4 + s.respiratory * 3,
      los_hosp: 8 + s.renal * 6 + s.respiratory * 5,
    };
    if (key === 'sex') return (Number(entityIndex) || 0) % 2 ? 'F' : 'M';
    if (key === 'adm') return (Number(entityIndex) || 0) % 3 ? 'Emergency' : 'Urgent';
    if (key === 'vent_mode') {
      if (s.respiratory > 0.72) return 'volume';
      if (s.respiratory > 0.35) return 'pressure';
      return 'standby';
    }
    if (key === 'vent_breath_seq') {
      if (s.respiratory > 0.72) return 'controlled';
      if (s.respiratory > 0.56) return 'assisted';
      if (s.respiratory > 0.35) return 'spontaneous';
      return 'standby';
    }
    if (Object.prototype.hasOwnProperty.call(values, key)) {
      return demoNormalizeValue(key, values[key], meta.unit);
    }
    // Unmodelled catalog concepts remain unavailable instead of receiving a
    // plausible-looking magic number. The catalog metadata is still visible.
    return null;
  }
  function demoHasClinicalModel(feature) {
    return demoClinicalValue(feature, 0, 0) != null;
  }
  function demoCadenceIndices(feature, entityIndex) {
    const key = String(feature || '').toLowerCase();
    let indices;
    if (DEMO_HIGH_FREQUENCY.has(key)) indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    else if (key === 'temp') indices = [0, 1, 3, 5, 7, 9, 10, 11];
    else if (DEMO_BLOOD_GAS_FEATURES.has(key)) indices = [0, 2, 4, 6, 8, 10, 11];
    else if (DEMO_RESPIRATORY_FEATURES.has(key) || DEMO_INTERVENTION_FEATURES.has(key)) indices = [0, 1, 3, 5, 7, 9, 11];
    else if (DEMO_SCORE_FEATURES.has(key)) indices = [0, 3, 6, 9, 11];
    else indices = [0, 3, 6, 9, 11];
    // Some low-frequency concepts deliberately miss one scheduled collection.
    // The rule is deterministic and never affects the high-frequency vital set.
    if (indices.length > 5 && !DEMO_HIGH_FREQUENCY.has(key) && demoHash(key, entityIndex, 99) < 0.28) {
      indices = indices.filter((_, idx) => idx !== 2);
    }
    return indices;
  }
  function demoTypicalPointCount(feature, entityIndex) {
    const key = String(feature || '').toLowerCase();
    const offset = Math.floor(demoHash(key, entityIndex, 77) * 4);
    if (DEMO_HIGH_FREQUENCY.has(key)) return 38 + offset;
    if (key === 'temp') return 11 + offset;
    if (DEMO_BLOOD_GAS_FEATURES.has(key)) return 7 + offset;
    if (DEMO_RESPIRATORY_FEATURES.has(key) || DEMO_INTERVENTION_FEATURES.has(key)) return 9 + offset;
    if (DEMO_SCORE_FEATURES.has(key)) return 6 + offset;
    return 6 + offset;
  }
  function demoBaseValue(feature, entityIndex) {
    return demoClinicalValue(feature, entityIndex, 1);
  }
  function demoTableValue(feature, entityIndex, timeIndex) {
    const meta = catalogFeatureMeta(feature);
    const key = String(feature || '').toLowerCase();
    const value = demoClinicalValue(key, entityIndex, timeIndex);
    if (demoIsBooleanFeature(key, meta.unit)) return Boolean(value);
    return value;
  }
  function demoSignal(feature, entityIndex) {
    const meta = catalogFeatureMeta(feature);
    const cadence = demoCadenceIndices(feature, entityIndex);
    const values = cadence.map(timeIndex => demoClinicalValue(feature, entityIndex, timeIndex));
    const numericValues = values
      .filter(value => value != null && value !== '')
      .map(Number)
      .filter(Number.isFinite);
    return {
      key: feature,
      feature,
      name: meta.name || feature,
      unit: meta.unit || '',
      values,
      times: cadence.map(timeIndex => DEMO_CHART_HOURS[timeIndex]),
      time_axis: {
        kind: 'relative_hours',
        label_en: 'ICU hour',
        label_zh: 'ICU 入科后小时',
        unit: 'hour',
        source_column: 'synthetic_relative_hour',
      },
      point_count: demoTypicalPointCount(feature, entityIndex),
      current: values[values.length - 1],
      min: numericValues.length ? Math.min(...numericValues) : null,
      max: numericValues.length ? Math.max(...numericValues) : null,
      mean: numericValues.length
        ? Number((numericValues.reduce((a, b) => a + b, 0) / numericValues.length).toFixed(2))
        : null,
      thresholds: demoThresholds(feature),
      bounded: true,
      max_points: 12,
      cadence: DEMO_HIGH_FREQUENCY.has(String(feature || '').toLowerCase()) ? 'high_frequency_bounded' : 'scheduled_sparse',
      scenario: demoScenarioName(entityIndex),
      synthetic: true,
    };
  }
  function demoTimeLanes(entityIndex) {
    return demoCatalogModules().map(moduleRow => {
      const features = moduleRow.features.map(feature => {
        const meta = catalogFeatureMeta(feature);
        const trajectory = demoIsTimeIndexed(moduleRow.module) && demoHasClinicalModel(feature);
        return {
          feature,
          name: meta.name || feature,
          unit: meta.unit || '',
          modeled: demoHasClinicalModel(feature),
          trajectory,
          status: trajectory ? 'trajectory' : (demoHasClinicalModel(feature) ? 'static' : 'metadata_only'),
        };
      });
      const signals = features.filter(row => row.trajectory).map(row => demoSignal(row.feature, entityIndex));
      return {
        lane: moduleRow.module,
        label: moduleRow.label,
        signal_count: features.length,
        available_signal_count: signals.length,
        features,
        signals,
        status: signals.length ? 'ready' : 'metadata_only',
      };
    });
  }
  function demoSignalDelta(values) {
    if (!values || values.length < 2) return null;
    const firstRaw = values[0];
    const lastRaw = values[values.length - 1];
    if (firstRaw == null || firstRaw === '' || lastRaw == null || lastRaw === '') return null;
    const first = Number(firstRaw);
    const last = Number(lastRaw);
    return Number.isFinite(first) && Number.isFinite(last)
      ? Number((last - first).toFixed(2))
      : null;
  }
  function demoFeatureTone(feature, value) {
    if (['sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd'].includes(feature)) return value >= 1 ? 'bad' : 'ok';
    if (feature === 'mech_vent') return String(value || '').toLowerCase() === 'invasive' ? 'warn' : 'ok';
    if (['vent_ind', 'rrt', 'vaso_ind'].includes(feature)) return value >= 1 ? 'warn' : 'ok';
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
    demoHasClinicalModel, demoScenarioName, demoTypicalPointCount,
    demoFeatureTone, demoCategorySection, demoQualityPanelRows,
  };
})();
