'use strict';

// The data dictionary's study-type view lays out the host's decisions: a
// study type the data cannot support says what is missing, a substitute says
// what stands in, and nothing claims acceptance on real data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve)); };

function row(database, survival) {
  return {
    database, profile: { reference_release: '3.1' }, sources: [], source_next_offset: null,
    studies: [{ family: 'survival', status: survival.status, decision: survival.decision }, { family: 'descriptive_epidemiology', status: 'exact', decision: { status: 'exact' } }],
    concepts: [],
  };
}
const VIEW = {
  concept_total: 41, concept_next_offset: 1,
  databases: [
    row('miiv', { status: 'exact', decision: {} }),
    row('eicu', { status: 'refused', decision: { missing: [{ capability: 'event_time', code: 'event_time_unavailable' }], substitutions: [] } }),
    row('aumc', { status: 'substitution', decision: { missing: [], substitutions: [{ capability: 'event_time', substitute: 'date', source_field: { table: 'admissions', column: 'dateofdeath' } }] } }),
  ],
};

function harness(overrides = {}) {
  const calls = [];
  const detail = { scrolled: [], scrollIntoView(options) { this.scrolled.push(options); } };
  const host = { innerHTML: '', dataset: {}, listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; },
    querySelector: selector => (selector === '.dst-detail' && host.innerHTML.includes('dst-detail') ? detail : null) };
  const window = {
    EU_LANG: overrides.language || 'zh',
    EU_DATABASE_NAMES: { name: key => ({ miiv: 'MIMIC-IV', eicu: 'eICU-CRD', aumc: 'AmsterdamUMCdb' }[key] || ''), abbr: key => ({ miiv: 'M-IV', eicu: 'eICU', aumc: 'AUMC' }[key] || '') },
    EU_API: { loadDataMap: async page => { calls.push(JSON.parse(JSON.stringify(page))); if (overrides.failure) throw overrides.failure; return overrides.view || VIEW; } },
  };
  const context = vm.createContext({ window });
  for (const file of ['html-escape.js', 'study-family-names.js', 'screens-dict-studies.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const view = window.EU_DICT_STUDIES;
  const root = { querySelector: selector => (selector === '[data-dst-root]' ? host : null) };
  return { view, root, host, calls, detail };
}
const clickOn = (host, attribute, value, extra = {}) => host.listeners.click({ target: { closest: selector => (selector === `[${attribute}]` ? { dataset: value, ...extra } : null) } });

test('the view reads the decisions once and lays out study types against databases', async () => {
  const { view, root, host, calls } = harness();
  assert.match(view.html(), /data-dst-root/);
  view.bind(root);
  await settle();
  assert.deepEqual(calls, [{ concept_offset: 0, concept_limit: 1 }]);
  // Known families sort first in the owner's order; the column heads are the databases.
  assert.match(host.innerHTML, /<th scope="row">描述性研究<\/th>.*<th scope="row">生存分析<\/th>/s);
  assert.match(host.innerHTML, /<th scope="col" title="eICU-CRD">eICU<\/th>/);
  // Nothing claims acceptance on real data, and no variable table repeats the dictionary.
  assert.doesNotMatch(host.innerHTML, /验收|accepted|<code>/);
});

test('a decision explains itself when opened; an exact one has nothing to open', async () => {
  const { view, root, host, detail } = harness();
  view.bind(root);
  await settle();
  assert.doesNotMatch(host.innerHTML, /data-dst-study="miiv\|survival"/);
  clickOn(host, 'data-dst-study', { dstStudy: 'eicu|survival' });
  assert.match(host.innerHTML, /<h3>生存分析 · eICU-CRD <span class="dst-chip is-refused">不满足<\/span><\/h3>\s*<ul><li>没有记录事件发生时间。<\/li><\/ul>/);
  // The explanation is below the matrix, so opening it brings it into view.
  assert.deepEqual(JSON.parse(JSON.stringify(detail.scrolled)), [{ block: 'nearest' }]);
  clickOn(host, 'data-dst-study', { dstStudy: 'aumc|survival' });
  assert.match(host.innerHTML, /<li>事件发生时间使用只有日期（admissions\.dateofdeath）。<\/li>/);
  assert.match(host.innerHTML, /会写明替代内容，需要你确认后才运行/);
});

test('a failed read offers a retry', async () => {
  const failed = harness({ failure: new Error('x') });
  failed.view.bind(failed.root);
  await settle();
  assert.match(failed.host.innerHTML, /暂时读不到研究类型。.*data-dst-retry/s);
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
    const { view: studies, root, host } = harness({ view, language });
    studies.bind(root);
    await settle();
    clickOn(host, 'data-dst-study', { dstStudy: 'aumc|survival' });
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
    const { view: studies, root, host } = harness({ view, language });
    studies.bind(root);
    await settle();
    clickOn(host, 'data-dst-study', { dstStudy: 'hirid|survival' });
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
      const { view: studies, root, host } = harness({ view, language });
      studies.bind(root);
      await settle();
      clickOn(host, 'data-dst-study', { dstStudy: 'synthetic|survival' });
      assert.ok(host.innerHTML.includes(labels[language === 'zh' ? 1 : 0]));
      assert.ok(host.innerHTML.includes('synthetic.event'));
      assert.ok(host.innerHTML.includes('synthetic.end'));
      assert.ok(host.innerHTML.includes(language === 'zh' ? '需要你确认后才运行' : 'needs your confirmation before it runs'));
      assert.doesNotMatch(host.innerHTML, /icu_death|icu_discharge_proxy|followup_days_|mort_\d+d/);
    }
  }
});

test('a mortality meaning reads in words for the dictionary', () => {
  assert.equal(harness().view.meaning('icu_death'), 'ICU 内死亡');
  assert.equal(harness({ language: 'en' }).view.meaning('death_within_72h_after_icu_discharge'), 'death up to 72 h after ICU discharge');
  assert.equal(harness().view.meaning('unknown_key'), '');
});
