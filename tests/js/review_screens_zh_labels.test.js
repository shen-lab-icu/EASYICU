/* Executable contract: on a Chinese page, Patient Review and Cohort Statistics
   name catalog concepts and modules by the catalog's Chinese names, show
   clinical values in Chinese, and keep internal provenance words (manifest,
   payload, path hash, receipts, tokens) and picker emoji out of the visible
   text. A hosted page never says "本地". English pages keep their labels. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  console,
  Date,
  Math,
  Map,
  Number,
  Set,
  JSON,
  setTimeout() {},
  EU_LANG: 'zh',
  EU_DATA: 'real',
  localStorage: { getItem() { return null; }, setItem() {} },
  location: { hash: '#patient' },
  EU_PALETTE: { color: () => '#2563eb', dashPattern: () => null, series: () => ['#2563eb'] },
  gloss: (term, label) => label || term,
};
context.t = (en, zh) => (context.EU_LANG === 'zh' && zh != null ? zh : en);
context.window = context;
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => {
  vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path });
  if (/data-catalog\.js$/.test(path)) {
    // The served catalog prefixes group names with a picker emoji.
    context.EU_CATALOG.groups = context.EU_CATALOG.groups.map(([key, en, zh]) => [key, `⭐ ${en}`, `⭐ ${zh}`]);
  }
});

const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[ch]);
const visible = html => String(html).replace(/<[^>]*>/g, ' ').replace(/&amp;/g, '&').replace(/\s+/g, ' ');
const dict = context.EU_CATALOG.dict;
const EMOJI = /\p{Extended_Pictographic}/u;

function fakeElement(store, key) {
  const handlers = {};
  const element = {
    handlers,
    querySelectorAll: selector => store[`all:${selector}`] || [],
    querySelector: selector => store[selector] || null,
    addEventListener(type, fn) { handlers[type] = fn; },
    set innerHTML(value) { store[`${key}.html`] = value; },
    get innerHTML() { return store[`${key}.html`] || ''; },
  };
  return element;
}

const series = (feature, name, unit, values) => ({
  feature, name, label: name, unit, values, times: values.map((_, index) => index),
});
const drilldown = {
  mode: 'real',
  demo: false,
  source: { database: 'miiv', path_hash: '410efec930fa' },
  summary: { entities: 3, modules: 2, total_rows: 120, mean_age: 61, female_pct: 50, mortality: 33.3, median_sofa2: 5, sepsis_pct: 40 },
  data_tables: {
    loaded_summary: { entities: 3, review_features: 4, module_count: 2, observed_features: 3 },
    module_picker: { default_module: 'vitals' },
    modules: [{ module: 'vitals', label: 'Vital Signs', label_i18n: { en: 'Vital Signs', zh: '生命体征' }, rows: 60 }],
    table_previews: [{
      module: 'vitals',
      display_columns: ['entity', 'hr'],
      display_column_labels: [
        { column: 'entity', label_en: 'Pseudonymous entity', label_zh: '伪匿名实体' },
        { column: 'hr', label_en: 'Heart Rate', label_zh: '心率' },
      ],
      rows: [{ entity: 'a1b2', hr: 80 }],
      rows_total: 1,
    }],
    detail_gate: {
      title: 'Bounded local table previews',
      reason: 'The browser renders capped module table previews with pseudonymous entity tokens; direct identifiers and full tables stay on disk.',
    },
  },
  module_profiles: [{ module: 'vitals', label: 'Vital Signs', label_i18n: { en: 'Vital Signs', zh: '生命体征' }, rows: 60, entities: 3, coverage_pct: 100 }],
  selected: { ref: 'r1', label: 'Entity 1', ordinal: 1, signals: [] },
  entities: [{ ref: 'r1', label: 'Entity 1', ordinal: 1 }, { ref: 'r2', label: 'Entity 2', ordinal: 2 }],
  trajectory_review: {
    lanes: [{
      lane: 'vitals',
      label: 'Vitals',
      signals: [
        series('hr', 'Heart Rate', 'bpm', [80, 92, 88]),
        series('map', 'Mean Arterial Pressure', 'mmHg', [70, 63, 72]),
        series('sofa2', 'SOFA-2 Score (Total)', 'points', [3, 5, 6]),
        series('sofa2_resp', 'SOFA-2 Respiratory', '0-4', [1, 2, 2]),
      ],
    }],
    contract: [{ index: '01', label: 'Entity scope', detail: '3 pseudonymous options exposed', status: 'ready' }],
    modes: [],
    multi_entity_comparison: {
      feature: 'hr', label: 'Heart Rate', unit: 'bpm', module: 'vitals', module_label: 'Vital Signs',
      time_axis: { kind: 'relative_hours', label_en: 'ICU hour', label_zh: 'ICU 入科后小时', unit: 'hour' },
      traces: [
        { ref: 'r1', label: 'Entity 1', values: [80, 90], times: [0, 1] },
        { ref: 'r2', label: 'Entity 2', values: [70, 75], times: [0, 1] },
      ],
    },
    single_entity: { selected_label: 'Entity 1', signals: [series('hr', 'Heart Rate', 'bpm', [80, 92, 88, 85])] },
  },
  patient_overview: {
    dashboard: {
      summary_cards: [
        { label: 'Age / sex', value: '72 / Female' },
        { label: 'Sepsis-3', value: 'Negative' },
        { label: 'Outcome', value: 'Deceased' },
        { label: 'ICU LOS', value: '5.6 d' },
      ],
    },
    category_view: { sections: [] },
    data_table: {},
  },
  quality: [
    { module: 'blood_gas', coverage_pct: 80, quality_status: 'ok', rows: 10 },
    { module: 'chemistry', rows: 20 },
  ],
  quality_metrics: {
    summary: { concept_count: 3, denominator_entities: 3, total_records: 30, weighted_missing_pct: 5, weighted_out_of_physio_pct: 1, weighted_duplicate_time_pct: 0 },
    features: [{ feature: 'hr', name: 'Heart Rate', module: 'vitals', missing_pct: 10, records: 30 }],
    top_issues: [{ feature: 'lact', name: 'Lactate', module: 'blood_gas', records: 5, missing_pct: 50 }],
  },
  quality_review: {
    contract: [{ index: '01', label: 'Local concept scope', detail: '3 concepts · 3 entities · 30 records', status: 'ready' }],
    panels: [{ id: 'missingness', label: 'Missingness', rows: [{ feature: 'hr', name: 'Heart Rate', value: 10, records: 30 }] }],
  },
  privacy: { max_points_per_signal: 24 },
  blocked_features: [
    { id: 'raw_identifier_table', reason: 'Patient Review returns aggregates and one pseudonymous entity only.' },
    { id: 'arbitrary_entity_search', reason: 'Search by direct clinical identifiers is not exposed in the native API.' },
  ],
};

context.EU_VIZ_PATIENT.init({ t: context.t, esc, icon: () => '', fmtInt: String, fmtNum: v => String(v), fmtPct: v => `${v}%` });

function renderPatientTabs() {
  context.EU_VIZ_PATIENT.hydrate(JSON.parse(JSON.stringify(drilldown)));
  const store = {};
  store['#ptabs'] = fakeElement(store, 'tabs');
  store['#ptbody'] = fakeElement(store, 'body');
  const modeButtons = ['single', 'compare'].map(mode => Object.assign(fakeElement(store, mode), { dataset: { patientSeriesMode: mode } }));
  store['all:[data-patient-series-mode]'] = modeButtons;
  context.document = { querySelector: selector => store[selector] || null };
  const out = { page: context.SCREENS.patient.render() };
  context.SCREENS.patient.afterRender(fakeElement(store, 'root'));
  const openTab = tab => store['#ptabs'].handlers.click({ target: { closest: () => ({ dataset: { ptab: tab } }) } });
  openTab('series');
  out.series = store['body.html'];
  for (const button of modeButtons) {
    button.handlers.click({ preventDefault() {} });
    out[button.dataset.patientSeriesMode] = store['body.html'];
  }
  modeButtons[0].handlers.click({ preventDefault() {} });
  for (const tab of ['patient', 'quality', 'tables']) {
    openTab(tab);
    out[tab] = store['body.html'];
  }
  return out;
}

// --- Chinese, hosted ---------------------------------------------------------
context.EU_CAPABILITIES = { deployment: { hosted: true } };
const zh = renderPatientTabs();
const zhText = Object.values(zh).map(visible).join(' ');
for (const concept of ['hr', 'map', 'sofa2', 'sofa2_resp', 'lact']) {
  const [en, zhName] = dict[concept];
  assert(!zhText.includes(en), `Chinese Patient Review shows the English catalog name "${en}"`);
  assert(zhText.includes(zhName), `Chinese Patient Review should name ${concept} "${zhName}"`);
}
for (const word of ['Entity 1', 'Female', 'Deceased', 'Negative', 'blood_gas', 'chemistry',
  'Bounded local table previews', 'Local concept scope', 'token', 'payload', '伪匿名', '载荷', '清洗回执', '回执核验', '本地', 'points',
  'raw_identifier_table', 'arbitrary_entity_search', 'pseudonymous entity only', ' ready', 'unknown', '三种审阅视图', '时间序列工作区']) {
  assert(!zhText.includes(word), `Chinese hosted Patient Review still shows "${word}"`);
}
assert(!EMOJI.test(zhText), 'module headings must not carry the catalog picker emoji');
// Patient-level rows are not downloadable on a hosted instance (user decision 2026-10-22).
assert(!zh.page.includes('data-patient-export'), 'a hosted Patient Review offers no patient-level Export');
for (const word of ['病例 1（当前）', '病例 2', '女', '死亡', '阴性', '5.6 天', '血气分析', '本次核查的特征', '匿名编号', 'ICU 入科后小时 (h)']) {
  assert(zhText.includes(word), `Chinese Patient Review should show "${word}"`);
}

// --- English, local: existing labels stay ------------------------------------
context.EU_LANG = 'en';
context.EU_CAPABILITIES = { deployment: { hosted: false } };
const en = renderPatientTabs();
const enText = Object.values(en).map(visible).join(' ');
for (const word of ['Heart Rate', 'Entity 1', '72 / Female', 'Deceased', 'Local export patient drilldown ready', 'Bounded local table previews']) {
  assert(enText.includes(word), `English Patient Review should keep "${word}"`);
}
assert(en.page.includes('data-patient-export'), 'a local Patient Review keeps its Export');

// --- Cohort Statistics -------------------------------------------------------
const review = {
  summary: { cohort_size: 100, mortality_pct: 20, age: { median: 60 }, sofa2: { median: 5 } },
  source: { label: 'MIMIC-IV', database: 'miiv', path_hash: '410efec930fa' },
  provenance: { payload_scope: 'cohort_aggregate_only' },
  coverage: [
    { module: 'blood_gas', coverage_pct: 80, covered_entities: 80, quality_status: 'ok', rows: 100, column_count: 5 },
    { module: 'vitals', coverage_pct: 100, covered_entities: 100, quality_status: 'ok', rows: 900, column_count: 9 },
  ],
  quality: { modules_ok: 2 },
  feature_catalog: {
    total_modules: 2,
    total_features: 4,
    modules: [
      { module: 'blood_gas', label: 'Blood Gas Analysis', feature_count: 1, features: [{ id: 'blood_gas:lact', module: 'blood_gas', column: 'lact', label: 'Lact' }] },
      { module: 'sofa2_score', label: 'SOFA-2', feature_count: 1, features: [{ id: 'sofa2_score:sofa2_cardio_observed', module: 'sofa2_score', column: 'sofa2_cardio_observed', label: 'Sofa2 Cardio Observed', audience: 'internal' }] },
      { module: 'vitals', label: 'Vital Signs', feature_count: 2, features: [
        { id: 'vitals:hr', module: 'vitals', column: 'hr', label: 'HR' },
        { id: 'vitals:map', module: 'vitals', column: 'map', label: 'Map' },
      ] },
    ],
  },
  feature_selection: { selected: [{ id: 'vitals:hr', module: 'vitals', column: 'hr', label: 'HR' }], default_ids: [], selected_count: 1 },
  groups: {
    supported: [{
      id: 'outcome',
      label: 'Survived vs Deceased',
      groups: [{ label: 'Survived', count: 80, pct: 80 }, { label: 'Deceased', count: 20, pct: 20 }, { label: 'Unknown', count: 0, pct: 0 }],
      profile: {
        columns: ['Survived', 'Deceased', 'Unknown'],
        rows: [
          { metric: 'N', kind: 'count', values: [80, 20, 0] },
          { metric: 'Median HR', feature_id: 'vitals:hr', column: 'hr', kind: 'numeric', values: [80, 90, null] },
          { metric: 'Sofa2 Cardio Available %', feature_id: 'sofa2_score:sofa2_cardio_available', column: 'sofa2_cardio_available', kind: 'percent', values: [90, 95, null] },
          { metric: 'Death Time available %', feature_id: 'outcome:death_time', column: 'death_time', kind: 'percent', values: [0, 100, null] },
        ],
      },
    }],
    blocked: [],
  },
};
context.EU_VIZ_COHORT.init({
  t: context.t, icon: () => '', esc, fmtInt: String, fmtNum: v => String(v), fmtPct: v => `${v}%`, fmtP: String,
  displayDataMode: () => 'real', officialDemoContext: () => null, registryActivePath: () => '/data/export',
  bindSourceRegistry() {}, sourceModeSelector: () => '', sourceRegistryBlock: () => '', repaintScreen() {},
  vizRail: () => '', workspaceSamplingNote: () => '', cohortCharts: null,
});

function renderCohort() {
  context.EU_VIZ_COHORT.hydrate(JSON.parse(JSON.stringify(review)));
  return [
    context.SCREENS.cohort.render(),
    context.EU_VIZ_COHORT.panels.coverage(review),
    context.EU_VIZ_COHORT.panels.profile(review),
  ].map(visible).join(' ');
}

context.EU_LANG = 'zh';
const cohortZh = renderCohort();
for (const word of ['Analysis readiness', '分析准备情况', 'manifest', 'payload', '载荷', '路径哈希', '410efec930fa', '草稿核验',
  'Blood Gas Analysis', 'Vital Signs', 'blood_gas', 'vitals', 'Heart Rate', 'KM/log-rank', 'Table One', 'Cardio', 'Death Time', 'coverage']) {
  assert(!cohortZh.includes(word), `Chinese Cohort Statistics still shows "${word}"`);
}
// An empty "unknown" group is no column or card; every row is descriptive, so no status column;
// the conversation has its current name; no sublines that repeat the tabs.
// A bookkeeping column is not offered in the feature picker, and its module chip goes with it.
assert(!cohortZh.includes('SOFA-2心血管评分 已观测') && !cohortZh.includes('Sofa2 Cardio Observed'), 'internal features are not offered');
assert(!/data-cohort-feature-module="sofa2_score"/.test(context.EU_VIZ_COHORT.panels.profile ? context.SCREENS.cohort.render() : ''), 'a module with only internal features has no chip');
assert(!/未知\s+0\b/.test(cohortZh) && !/死亡\s+未知/.test(cohortZh), 'an empty unknown group is neither a card nor a column');
for (const word of ['Table 1', '研究引导', '当前导出已加载', '分析前审阅模块覆盖率', '为保证大型本地导出']) {
  assert(!cohortZh.includes(word), `Chinese Cohort Statistics should not show "${word}"`);
}
assert.strictEqual(cohortZh.split('描述性').length - 1, 1, 'only the split chooser says 描述性 (no per-row status pill)');
assert(cohortZh.includes('在研究对话中继续'), 'the next step names the research conversation');
assert(!/\bHR\b/.test(cohortZh), 'Chinese Cohort Statistics should not name heart rate by its column code');
for (const word of ['基线特征表', '血气分析', '生命体征', '心率', '心率 中位数', '乳酸', ' 可用 %', '院内死亡时间 有记录比例 %']) {
  assert(cohortZh.includes(word), `Chinese Cohort Statistics should show "${word}"`);
}

context.EU_LANG = 'en';
const cohortEn = renderCohort();
for (const word of ['Median HR', 'HR', 'Sofa2 Cardio Available %']) {
  assert(cohortEn.includes(word), `English Cohort Statistics should keep "${word}"`);
}

console.log(JSON.stringify({ ok: true }));
