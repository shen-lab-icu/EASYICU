'use strict';

// The Data map lays out the host's decisions: a study type the data cannot
// support says what is missing, a substitute says what stands in, variables
// page through the host's catalogue, and every database is marked as not yet
// accepted on real data until the host says otherwise. Synthetic fixtures
// are not counted as registrations.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve)); };

function row(database, survival, extra = {}) {
  return {
    database, profile: { reference_release: '3.1' }, real_data_acceptance: 'not_accepted', sources: extra.sources || [], source_next_offset: null,
    studies: [{ family: 'survival', status: survival.status, decision: survival.decision }, { family: 'descriptive_epidemiology', status: 'exact', decision: { status: 'exact' } }],
    concepts: [{ concept: 'age', unit: 'years', status: extra.ageStatus || 'available', substitute: extra.ageStatus === 'substituted' ? { kind: 'capped_or_shifted' } : null }],
  };
}
const VIEW = {
  concept_total: 41, concept_next_offset: 40,
  databases: [
    row('miiv', { status: 'exact', decision: {} }, { sources: [{ kind: 'public_demo' }, { kind: 'unrecorded' }, { kind: 'synthetic' }] }),
    row('eicu', { status: 'refused', decision: { missing: [{ capability: 'event_time', code: 'event_time_unavailable' }], substitutions: [] } }, { ageStatus: 'substituted' }),
    row('aumc', { status: 'substitution', decision: { missing: [], substitutions: [{ capability: 'event_time', substitute: 'date', source_field: { table: 'admissions', column: 'dateofdeath' } }] } }),
  ],
};

function harness(overrides = {}) {
  const calls = [];
  const detail = { scrolled: [], scrollIntoView(options) { this.scrolled.push(options); } };
  const host = { innerHTML: '', dataset: {}, listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; },
    querySelector: selector => (selector === '.dmap-detail' && host.innerHTML.includes('dmap-detail') ? detail : null) };
  const window = {
    EU_LANG: overrides.language || 'zh',
    EU_DATABASE_NAMES: { name: key => ({ miiv: 'MIMIC-IV', eicu: 'eICU-CRD', aumc: 'AmsterdamUMCdb' }[key] || ''), abbr: key => ({ miiv: 'M-IV', eicu: 'eICU', aumc: 'AUMC' }[key] || '') },
    EU_CATALOG: { dict: { age: ['Age', '年龄', 'years'], samp: ['Microbiology sample', '微生物标本', ''] } },
    EU_API: { loadDataMap: async page => { calls.push(JSON.parse(JSON.stringify(page))); if (overrides.failure) throw overrides.failure; return overrides.view || VIEW; } },
  };
  const document = { querySelector: selector => (selector === '[data-dmap-root]' ? host : null) };
  const context = vm.createContext({ window, document });
  for (const file of ['html-escape.js', 'study-family-names.js', 'screens-datamap.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const screen = window.SCREENS.datamap;
  const root = { querySelector: () => host };
  return { screen, root, host, calls, detail };
}
const clickOn = (host, attribute, value, extra = {}) => host.listeners.click({ target: { closest: selector => (selector === `[${attribute}]` ? { dataset: value, ...extra } : null) } });

test('the map loads one page and names databases, study types and variables', async () => {
  const { screen, root, host, calls } = harness();
  assert.match(screen.render(), /<h1>数据地图<\/h1>/);
  screen.afterRender(root);
  await settle();
  assert.deepEqual(calls, [{ concept_offset: 0, concept_limit: 40 }]);
  assert.match(host.innerHTML, /<h3>MIMIC-IV<\/h3>\s*<p>参考版本 3\.1<\/p>\s*<p>本机已登记：自有 1 · 公开 Demo 1 · 合成测试数据 1<\/p>\s*<span class="dmap-chip is-pending">尚未在真实数据上验收<\/span>/);
  assert.match(host.innerHTML, /<p>本机未登记<\/p>/);
  // Known families sort first in the owner's order; the column heads are the databases.
  assert.match(host.innerHTML, /<th scope="row">描述性研究<\/th>.*<th scope="row">生存分析<\/th>/s);
  assert.match(host.innerHTML, /<th scope="col" title="eICU-CRD">eICU<\/th>/);
  assert.match(host.innerHTML, /<code>age<\/code><span>年龄<\/span><\/th><td class="dmap-unit">years<\/td><td><span class="dmap-chip is-available">可用<\/span><\/td><td title="超过上限的值被截断或平移"><span class="dmap-chip is-substituted">替代<\/span><\/td>/);
  assert.match(host.innerHTML, /1–1 \/ 共 41 个/);
});

test('a decision explains itself when opened; an exact one has nothing to open', async () => {
  const { screen, root, host, detail } = harness();
  screen.afterRender(root);
  await settle();
  assert.doesNotMatch(host.innerHTML, /data-dmap-study="miiv\|survival"/);
  clickOn(host, 'data-dmap-study', { dmapStudy: 'eicu|survival' });
  assert.match(host.innerHTML, /<h3>生存分析 · eICU-CRD <span class="dmap-chip is-refused">不满足<\/span><\/h3>\s*<ul><li>没有记录事件发生时间。<\/li><\/ul>/);
  // The explanation is below the matrix, so opening it brings it into view.
  assert.deepEqual(JSON.parse(JSON.stringify(detail.scrolled)), [{ block: 'nearest' }]);
  clickOn(host, 'data-dmap-study', { dmapStudy: 'aumc|survival' });
  assert.match(host.innerHTML, /<li>事件发生时间使用只有日期（admissions\.dateofdeath）。<\/li>/);
  assert.match(host.innerHTML, /会写明替代内容，需要你确认后才运行/);
});

test('paging asks the host for the next page; a failed read offers a retry', async () => {
  const { screen, root, host, calls } = harness();
  screen.afterRender(root);
  await settle();
  clickOn(host, 'data-dmap-page', { dmapPage: 'next' }, { disabled: false });
  await settle();
  assert.deepEqual(calls[1], { concept_offset: 40, concept_limit: 40 });

  const failed = harness({ failure: new Error('x') });
  failed.screen.afterRender(failed.root);
  await settle();
  assert.match(failed.host.innerHTML, /暂时读不到数据地图。.*data-dmap-retry/s);
});

test('default mortality refusals retain the changed meaning, date and missing follow-up', async () => {
  for (const [language, meaning, date, missing] of [
    ['zh', '出 ICU 后 72 小时内死亡', '只有日期', '没有记录删失时间'],
    ['en', 'death up to 72 h after ICU discharge', 'a date only', 'No censoring time'],
  ]) {
    const view = { ...VIEW, databases: [row('aumc', { status: 'refused', decision: {
      missing: [{ capability: 'censor_time' }], substitutions: [
        { capability: 'outcome_meaning', substitute: 'death_within_72h_after_icu_discharge', source_field: { table: 'admissions', column: 'dateofdeath' } },
        { capability: 'event_time', substitute: 'date' },
      ],
    } })] };
    const { screen, root, host } = harness({ view, language });
    screen.afterRender(root);
    await settle();
    clickOn(host, 'data-dmap-study', { dmapStudy: 'aumc|survival' });
    for (const text of [meaning, date, missing, 'admissions.dateofdeath']) assert.ok(host.innerHTML.includes(text));
    assert.doesNotMatch(host.innerHTML, /outcome_meaning|death_within_72h_after_icu_discharge/);
  }
});

test('HiRID default mortality keeps its ICU meaning and observation proxy explicit', async () => {
  for (const [language, meaning, timing] of [
    ['zh', 'ICU 内死亡', '最后一次观测时间作为近似'],
    ['en', 'death in the ICU', 'the last observation time as a proxy'],
  ]) {
    const view = { ...VIEW, databases: [row('hirid', { status: 'refused', decision: {
      missing: [{ capability: 'censor_time' }], substitutions: [
        { capability: 'outcome_meaning', substitute: 'icu_death' },
        { capability: 'event_time', substitute: 'last_observation_proxy', source_field: { table: 'observations', column: 'datetime' } },
      ],
    } })] };
    const { screen, root, host } = harness({ view, language });
    screen.afterRender(root);
    await settle();
    clickOn(host, 'data-dmap-study', { dmapStudy: 'hirid|survival' });
    for (const text of [meaning, timing, 'observations.datetime']) assert.ok(host.innerHTML.includes(text));
    assert.doesNotMatch(host.innerHTML, /outcome_meaning|icu_death|last_observation_proxy/);
  }
});

test('supported mortality offers name the endpoint, clock, censoring and required confirmation', async () => {
  for (const language of ['en', 'zh']) {
    for (const [endpoint, timing, censor, labels] of [
      ['icu_death', 'icu_discharge_proxy', 'icu_discharge', ['death in the ICU', 'ICU 内死亡']],
      ...[28, 90, 365].map(days => [`mort_${days}d`, `followup_days_${days}d`, `followup_days_${days}d`,
        [`death within ${days} days`, `${days} 天内死亡`]]),
    ]) {
      const view = { ...VIEW, databases: [row('synthetic', { status: 'substitution', decision: {
        missing: [], substitutions: [
          { capability: 'outcome_meaning', substitute: endpoint },
          { capability: 'event_time', substitute: timing, source_field: { table: 'synthetic', column: 'event' } },
          { capability: 'censor_time', substitute: censor, source_field: { table: 'synthetic', column: 'end' } },
        ],
      } })] };
      const { screen, root, host } = harness({ view, language });
      screen.afterRender(root);
      await settle();
      clickOn(host, 'data-dmap-study', { dmapStudy: 'synthetic|survival' });
      assert.ok(host.innerHTML.includes(labels[language === 'zh' ? 1 : 0]));
      assert.ok(host.innerHTML.includes('synthetic.event'));
      assert.ok(host.innerHTML.includes('synthetic.end'));
      assert.ok(host.innerHTML.includes(language === 'zh' ? '需要你确认后才运行' : 'needs your confirmation before it runs'));
      assert.doesNotMatch(host.innerHTML, /icu_death|icu_discharge_proxy|followup_days_|mort_\d+d/);
    }
  }
});

test('a binned variable names its band width and top band; a missing derived one names its unavailable inputs', async () => {
  for (const [language, age, weight, missing] of [
    ['zh', '每 5 岁记为一段；90 岁及以上为一段', '每 5 kg 记为一段', '缺少的输入：微生物标本、gcs_raw'],
    ['en', 'Recorded in 5-year bands; 90 and above form one band', 'Recorded in 5-kg bands', 'Missing inputs: Microbiology sample, gcs_raw'],
  ]) {
    const concepts = [
      { concept: 'age', unit: 'years', status: 'substituted', capability: {}, substitute: { kind: 'binned', bin_width: 5, top_code: 90 } },
      { concept: 'weight', unit: 'kg', status: 'substituted', capability: {}, substitute: { kind: 'binned', bin_width: 5, top_code: null } },
      { concept: 'susp_inf', unit: 'boolean', status: 'missing', capability: { missing_dependencies: ['samp', 'gcs_raw'] }, substitute: null },
      { concept: 'lact', unit: 'mmol/L', status: 'missing', capability: { missing_dependencies: [] }, substitute: null },
    ];
    const view = { ...VIEW, databases: [{ ...row('hirid', { status: 'exact', decision: {} }), concepts }] };
    const { screen, root, host } = harness({ view, language });
    screen.afterRender(root);
    await settle();
    const titles = [...host.innerHTML.matchAll(/<td title="([^"]*)"><span class="dmap-chip/g)].map(match => match[1]);
    assert.deepEqual(titles, [age, weight, missing]);
  }
});
