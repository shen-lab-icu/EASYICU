'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = { console, EU_LANG: 'zh', EU_CATALOG: { dict: {} }, VIZ_DEMO: { demoCatalogModules: () => [], catalogModuleText: x => x } };
context.window = context;
const sandbox = vm.createContext(context);
for (const file of ['src/easyicu/webserver/static/js/screens-viz-cohort-survival.js', 'src/easyicu/webserver/static/js/screens-viz-cohort-view.js']) {
  vm.runInContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: file });
}
const review = {
  summary: {
    cohort_size: 2520, mortality_pct: 8.5,
    mortality: { deceased_count: 212, known_count: 2492, unknown_count: 28 },
    age: { median: 66, count: 2516 }, sex: { female_pct: 40, female_count: 1000, known_count: 2500, unknown_count: 20 }, sofa2: { median: 3, count: 2481 },
  },
  groups: { supported: [{ id: 'sepsis', label: 'Sepsis vs Non-sepsis',
    groups: [{ label: 'Non-sepsis', count: 2357, pct: 93.5 }, { label: 'Sepsis', count: 163, pct: 6.5 }],
    profile: { columns: ['Non-sepsis', 'Sepsis'], rows: [{ metric: 'N', kind: 'count', values: [2357, 163] }] } }] },
  survival_analysis: {
    default_outcome: 'hospital_death', default_group: 'age',
    outcomes: [{ id: 'hospital_death', status: 'ready', label: 'Hospital mortality', event_summary: { status: 'available', event_rate_pct: 8.5, event_count: 212, denominator: 2492 } }],
    group_options: [
      { id: 'sepsis', label: 'Sepsis vs Non-sepsis', status: 'blocked', reason: 'Sepsis events may occur after the survival time origin; a verified baseline group or time-varying analysis is required.' },
      { id: 'sofa2', label: 'SOFA-2 <= 3 vs > 3', status: 'blocked', reason: 'The whole-stay maximum SOFA-2 is not a verified baseline measurement at the survival time origin.' },
      { id: 'age', label: 'Age <65 vs >=65', status: 'ready', groups: [{ count: 10 }, { count: 10 }] },
    ],
    curves: [{ outcome_id: 'hospital_death', group_id: 'age', label: 'Hospital mortality by Age <65 vs >=65',
      display_horizon_days: 30, groups: [{ label: '<65', points: [{ time: 5, survival: 80 }] }, { label: '>=65', points: [{ time: 30, survival: 70 }] }], logrank: {}, number_at_risk: {} }],
  },
};
const state = { compare: 'sepsis', survivalGroup: 'sepsis', survivalOutcome: 'hospital_death' };
const fmt = v => v == null ? '—' : String(v);
context.EU_VIZ_COHORT_VIEW.init({ state, t: (_en, zh) => zh, icon: () => '', esc: fmt,
  fmtInt: fmt, fmtNum: fmt, fmtPct: v => v == null ? '—' : `${v}%`, fmtP: fmt,
  cohortCharts: { survivalSlot: () => '<chart />' }, cohortReview: () => review, workspaceSamplingNote: () => '' });
const profile = context.EU_VIZ_COHORT_VIEW.snapshotBody(review);
assert.match(profile, /1000 \/ 2500 性别已知 · 20 未知/);
const groups = context.EU_VIZ_COHORT_VIEW.groupsBody(review);
assert.match(groups, /描述性分组汇总/);
assert.doesNotMatch(groups, /基线特征表|Sepsis-3 阳性/);
assert.match(groups, /212 \/ 2492 结局已知 · 28 未知/);
assert.match(groups, /入住期间的最高值/);
assert.match(groups, /未记录事件不代表确证阴性/);
assert.match(groups, /各指标的有效例数尚未提供/);
const survival = context.EU_VIZ_COHORT_VIEW.survivalBody(review);
assert.equal(state.survivalGroup, 'sepsis'); // selection is resolved locally, never using the blocked curve
assert.match(survival, /年龄 &lt;65|年龄 <65/);
assert.match(survival, /脓毒症事件可能发生在生存时间起点之后/);
assert.match(survival, /整个入住期间的最高 SOFA-2 并非/);
assert.match(survival, /同一时点可能有多例/);
assert.match(survival, /并非全因 30 天生存/);
assert.doesNotMatch(survival, /绝对风险差|30 天生存:/);

// Render the real patient owner with mixed full-cohort and bounded-review scopes.
context.EU_PATIENT_FEATURES = { signalKey: x => x, catalogLanes: () => [], entityLabel: x => x, clinicalValueText: x => x };
context.VIZ_DEMO.catalogConceptLabel = x => x;
context.VIZ_DEMO.catalogUnitText = x => x;
vm.runInContext(fs.readFileSync('src/easyicu/webserver/static/js/screens-viz-patient.js', 'utf8'), sandbox);
context.EU_VIZ_PATIENT.init({ t: (_en, zh) => zh, fmtInt: fmt, fmtNum: fmt,
  fmtPct: v => v == null ? '—' : `${v}%`, esc: fmt });
context.EU_VIZ_PATIENT.hydrate({
  summary: { entities: 2520, review_entities: 500, review_scope: 'browser_bounded_entity_sample', modules: 19, total_rows: 1000,
    mortality: 8.5, mortality_known_count: 2492, mortality_unknown_count: 28, female_pct: 40, female_known_count: 2500, female_unknown_count: 20, median_sofa2: 2 },
  module_profiles: [], data_tables: {},
});
const patient = context.SCREENS.patient.render();
assert.match(patient, /完整队列 2520 次入住/);
assert.match(patient, /SOFA-2 使用本页审阅的 500 次入住/);
assert.match(patient, /结局已知 2492 · 未知 28/);
assert.match(patient, /已知 2500 · 未知 20/);
assert.match(patient, /并非标准 Sepsis-3/);
context.EU_VIZ_PATIENT.hydrate({ summary: { entities: 3, review_entities: 3 }, module_profiles: [], data_tables: {} });
assert.match(context.SCREENS.patient.render(), /当前服务仍在运行旧统计逻辑/);
