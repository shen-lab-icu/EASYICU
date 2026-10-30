'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const c = { console, EU_LANG: 'zh', location: { hash: '#cohort' }, VIZ_DEMO: { demoCatalogModules: () => [], catalogModuleText: x => x }, EU_CATALOG: { dict: {} } };
c.window = c;
for (const file of ['screens-viz-cohort-survival.js', 'screens-viz-cohort-view.js', 'screens-viz-cohort.js']) vm.runInNewContext(fs.readFileSync(`src/easyicu/webserver/static/js/${file}`, 'utf8'), c);
const fmt = value => value == null ? '—' : String(value);
let source = '/demo';
let posted = [];
let deferred;
c.EU_API = { loadCohortReviewSummary(body) { posted.push(body); return new Promise((resolve, reject) => { deferred = { resolve, reject }; }); } };
c.EU_VIZ_COHORT.init({ t: (_en, zh) => zh, icon: () => '', esc: fmt, fmtInt: fmt, fmtNum: fmt, fmtPct: fmt, fmtP: fmt,
  officialDemoContext: () => null, registryActivePath: () => source, bindSourceRegistry() {}, officialDemoOffers: () => '',
  activeSourceKind: () => null, sourceRegistryBlock: () => '', repaintScreen() {}, vizRail: () => '', workspaceSamplingNote: () => '' });
const featureCatalog = { modules: [{ features: [
  { id: 'demographics:age', module: 'demographics', column: 'age', label: 'Age', kind: 'numeric' },
  { id: 'demographics:sex', module: 'demographics', column: 'sex', label: 'Sex', kind: 'categorical' },
  { id: 'vitals:hr', module: 'vitals', column: 'hr', label: 'Heart Rate', kind: 'numeric' },
  { id: 'hematology:pt', module: 'hematology', column: 'pt', label: 'Prothrombin time', kind: 'numeric' },
  { id: 'outcome:death', module: 'outcome', column: 'death', label: 'Death', kind: 'binary' },
] }] };
const payload = { summary: { cohort_size: 3 }, feature_catalog: featureCatalog,
  custom_group_options: { version: '1', categories: [{ feature_id: 'demographics:adm', label: 'Admission', values: ['medical', 'surgical', 'other'] }] },
  groups: { supported: [] }, survival_analysis: { group_options: [] } };
c.EU_VIZ_COHORT.hydrate(payload);
let html = c.SCREENS.cohort.render();
assert.match(html, /当前服务仍在运行旧统计逻辑/);
assert.match(html, /≤ 阈值和 > 阈值/);
assert.match(html, /不改变科研计划或执行/);
assert.match(html, /vitals:hr/);
assert.doesNotMatch(html.split('</section>')[0], /outcome:death/);
function element(value = '') { return { value, handlers: {}, addEventListener(type, fn) { this.handlers[type] = fn; } }; }
function rootFor({ kind = 'numeric_threshold', feature = 'demographics:age', threshold = '', a = '', b = '', landmark = '24' } = {}) {
  const elements = { '[data-custom-kind]': element(kind), '[data-custom-feature]': element(feature), '[data-custom-threshold]': element(threshold), '[data-custom-value-a]': element(a), '[data-custom-value-b]': element(b), '[data-custom-landmark]': element(landmark) };
  const apply = element();
  const root = { querySelector(sel) { return elements[sel] || null; }, querySelectorAll(sel) { return sel === '[data-custom-apply]' ? [apply] : []; } };
  c.SCREENS.cohort.afterRender(root);
  return { elements, apply };
}
async function tick() { await Promise.resolve(); await Promise.resolve(); }
(async () => {
  let form = rootFor();
  form.apply.handlers.click();
  assert.equal(posted.length, 0, 'an empty threshold is not coerced to zero');
  assert.match(c.SCREENS.cohort.render(), /输入有效的有限阈值/);
  form = rootFor({ threshold: '0' });
  form.apply.handlers.click();
  assert.deepEqual(JSON.parse(JSON.stringify(posted[0].custom_group)), { kind: 'numeric_threshold', feature_id: 'demographics:age', threshold: 0 });
  deferred.resolve({ ...payload, custom_group_applied: {kind: 'numeric_threshold', feature_id: 'demographics:age', threshold: 0}, groups: { supported: [{ id: 'custom', kind: 'numeric_threshold', feature_id: 'demographics:age', label: 'Age · custom', groups: [{ label: '≤ 0', count: 1 }, { label: '> 0', count: 2 }], profile: { columns: ['≤ 0', '> 0'], rows: [] } }] }, survival_analysis: { group_options: [] } });
  await tick();
  html = c.SCREENS.cohort.render();
  assert.match(html, /当前预览分组：年龄 · ≤ 0 \/ > 0/);
  assert.match(html, /年龄 · 自定义/);
  assert.equal(c.EU_VIZ_COHORT_VIEW.text('Hospital mortality by Age · custom'), '院内死亡 · 按 年龄 · 自定义 分组');
  assert.doesNotMatch(html, /当前预览分组：demographics:age/);
  const blocked = c.EU_VIZ_COHORT.panels.groups({ ...payload, groups: { supported: [{
    id: 'custom', status: 'blocked', label: 'Age · custom', reason: 'Custom split does not produce two non-empty groups.',
    groups: [{ label: '≤ 0', count: 0 }, { label: '> 0', count: 3 }, { label: 'Unknown', count: 0 }, { label: 'Other', count: 0 }],
  }] } });
  assert.match(blocked, /没有产生两个非空组/);
  assert.match(blocked, /未知/);
  assert.match(blocked, /其他未选类别/);
  assert.doesNotMatch(blocked, /data-cohort-table-one/);

  form = rootFor({ kind: 'categorical_values', feature: 'demographics:adm', a: 'medical', b: 'surgical' });
  form.elements['[data-custom-kind]'].handlers.change();
  const categoryHtml = c.SCREENS.cohort.render();
  assert.match(categoryHtml, /demographics:adm/);
  assert.match(categoryHtml, /medical/);
  assert.match(categoryHtml, /surgical/);
  form = rootFor({ kind: 'categorical_values', feature: 'demographics:adm', a: 'medical', b: 'surgical' });
  form.apply.handlers.click();
  assert.deepEqual(JSON.parse(JSON.stringify(posted[1].custom_group)), { kind: 'categorical_values', feature_id: 'demographics:adm', values: ['medical', 'surgical'] });
  deferred.reject(Object.assign(new Error('failed'), { code: 'custom_group_category_unavailable' }));
  await tick();
  assert.match(c.SCREENS.cohort.render(), /所选类别不在当前导出/);
  assert.equal(c.EU_VIZ_COHORT.snapshot().customGroup.threshold, 0, 'failed edit retains last successful grouping');
  form = rootFor({ threshold: '65' });
  form.apply.handlers.click();
  deferred.resolve(payload); // Old or mixed server omits the echoed grouping.
  await tick();
  assert.equal(c.EU_VIZ_COHORT.snapshot().customGroup.threshold, 0);
  assert.match(c.SCREENS.cohort.render(), /服务未确认请求的自定义分组/);
  form = rootFor({ threshold: '65' });
  form.apply.handlers.click();
  deferred.resolve({ ...payload, custom_group_applied: { kind: 'numeric_threshold', feature_id: 'demographics:age', threshold: 70 }, groups: { supported: [{ id: 'custom', kind: 'numeric_threshold', feature_id: 'demographics:age' }] } });
  await tick();
  assert.equal(c.EU_VIZ_COHORT.snapshot().customGroup.threshold, 0, 'a mismatched threshold cannot be accepted');
  form = rootFor({ threshold: '65' });
  form.apply.handlers.click();
  source = '/new';
  c.EU_VIZ_COHORT.resetForSourceChange();
  deferred.resolve(payload);
  await tick();
  assert.equal(c.EU_VIZ_COHORT.snapshot().customGroup, null, 'old-source result cannot reapply the grouping');
  assert.equal(c.EU_VIZ_COHORT.review(), null);
  source = '/demo';
  c.EU_VIZ_COHORT.hydrate(payload);
  c.__euCohortPanel = 'survival';
  c.SCREENS.cohort.render();
  form = rootFor({ feature: 'hematology:pt', threshold: '100' });
  form.elements['[data-custom-feature]'].handlers.change();
  html = c.SCREENS.cohort.render();
  assert.match(html, /当前服务尚未启用按时间窗口的数值生存分组/);
  assert.match(html, /data-custom-apply disabled/);
  const supported = { ...payload, custom_group_options: { ...payload.custom_group_options, landmark_hours: [6, 12, 24, 48] } };
  c.EU_VIZ_COHORT.hydrate(supported);
  html = c.SCREENS.cohort.render();
  assert.match(html, /ICU 入科后 24 小时内；第 24 小时起算/);
  assert.match(html, /仅支持经核验的 eICU 来源、ICU 死亡及 ICU 随访/);
  assert.doesNotMatch(html.split('</section>')[0], /vitals:hr/, 'non-laboratory numeric fields are not offered as timed laboratory groups');
  assert.match(html.split('</section>')[0], /hematology:pt/);
  form = rootFor({ feature: 'hematology:pt', threshold: '100', landmark: '24' });
  form.apply.handlers.click();
  assert.deepEqual(JSON.parse(JSON.stringify(posted.at(-1).custom_group)), { kind: 'numeric_threshold', feature_id: 'hematology:pt', threshold: 100, landmark_hours: 24 });
  deferred.resolve({ ...supported, custom_group_applied: { kind: 'numeric_threshold', feature_id: 'hematology:pt', threshold: 100 }, groups: { supported: [{ id: 'custom', kind: 'numeric_threshold', feature_id: 'hematology:pt' }] } });
  await tick();
  assert.equal(c.EU_VIZ_COHORT.snapshot().customGroup, null, 'a mixed service that ignores the landmark is rejected');
  assert.match(c.SCREENS.cohort.render(), /服务未确认请求的自定义分组/);
  form = rootFor({ feature: 'hematology:pt', threshold: '100', landmark: '24' });
  form.apply.handlers.click();
  deferred.resolve({ ...supported, custom_group_applied: { kind: 'numeric_threshold', feature_id: 'hematology:pt', threshold: 100, landmark_hours: 24 },
    groups: { supported: [{ id: 'custom', kind: 'numeric_threshold', feature_id: 'hematology:pt' }] },
    survival_analysis: { default_outcome: 'hospital_death', default_group: 'custom', outcomes: [{id: 'icu_death', status: 'ready'}],
      group_options: [{ id: 'custom', status: 'ready', label: 'HR · custom', groups: [{count: 1}, {count: 2}] }],
      curves: [{ outcome_id: 'icu_death', group_id: 'custom', time_label: 'Days since ICU landmark', landmark_hours: 24, measurement_unit: 's', event_time_basis: 'icu_discharge_status_and_exit_time_proxy', measurement_aggregation: 'first', measurement_window_hours: [0, 24], measurement_window_end_inclusive: false, landmark_counts: {included: 3, missing_measurement: 2}, groups: [] }] } });
  await tick();
  html = c.SCREENS.cohort.render();
  assert.match(html, /全队列，非 landmark 风险集/);
  assert.match(html, /生存起点为 ICU 入科后第 24 小时/);
  assert.match(html, /ICU 标志时间点后的天数/);
  assert.doesNotMatch(html, /Days since ICU landmark/);
  assert.match(html, /纳入 3 · 窗口内无有效测量 2/);
  assert.match(html, /声明单位（需核对来源） s/);
  assert.match(html, /事件时间使用 ICU 出科时点作为代理，并非记录的死亡时刻/);
  assert.equal(c.EU_VIZ_COHORT.snapshot().survivalOutcome, 'icu_death', 'landmark custom groups cannot use hospital-origin time');
  const staleCurve = c.EU_VIZ_COHORT.panels.survival({ survival_analysis: {
    default_outcome: 'icu_death', default_group: 'custom', outcomes: [{ id: 'icu_death', status: 'ready' }],
    group_options: [{ id: 'custom', status: 'ready', groups: [] }],
    curves: [{ outcome_id: 'icu_death', group_id: 'custom', landmark_hours: 12, groups: [] }],
  } });
  assert.match(staleCurve, /服务没有返回所选时间起点的 ICU 生存曲线/);
  assert.doesNotMatch(staleCurve, /surv-card mt-14/, 'mismatched landmark must not be plotted');
  const blockedSurvival = c.EU_VIZ_COHORT.panels.survival({ survival_analysis: {
    default_outcome: 'hospital_death', default_group: 'age', outcomes: [{ id: 'hospital_death', status: 'ready' }],
    group_options: [{ id: 'age', status: 'ready', label: 'Age', groups: [] }, { id: 'custom', status: 'blocked', label: 'PT · custom', basis: 'icu_landmark_first_measurement', measurement_unit: 'sec', reason: 'The landmark risk set does not produce two non-empty groups.', groups: [{label: '≤ 1', count: 0}, {label: '> 1', count: 492}] }],
    curves: [{ outcome_id: 'hospital_death', group_id: 'age', groups: [] }],
  } });
  assert.match(blockedSurvival, /自定义生存分组不可用/);
  assert.match(blockedSurvival, /下方曲线使用当前可用的分组，并非这个自定义分组/);
  assert.match(blockedSurvival, /所选生存起点的风险人群不能形成两个非空组/);
  assert.match(blockedSurvival, /生存起点风险集分组：≤ 1 = 0 · > 1 = 492 · 声明单位（需核对来源） sec/);
  assert.ok(blockedSurvival.indexOf('自定义生存分组不可用') < blockedSurvival.indexOf('surv-card'), 'reason appears before the fallback chart');
  c.EU_VIZ_COHORT.hydrate({ ...payload, custom_group_options: undefined });
  const oldServer = c.SCREENS.cohort.render();
  assert.match(oldServer, /当前服务版本尚未加载自定义分组，请重启服务后使用/);
  assert.doesNotMatch(oldServer, /data-custom-apply/);
  console.log('custom cohort grouping contracts passed');
})().catch(err => { console.error(err); process.exitCode = 1; });
