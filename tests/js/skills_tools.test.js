'use strict';

// The data-definition and planning Skills show the host's answers: a
// definition's availability and parts for one database, extraction modules by
// mapping status, and a sample size computed by the host from typed
// assumptions (percentages sent as proportions, refusals in plain words).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve)); };

const DEFINITIONS = {
  items: [
    { concept: 'sofa', description: 'sequential organ failure assessment score', time_window: { study_observation_window: 'requires_explicit_selection' },
      clinical_contract: { definition: 'SOFA', version: '1996', source_id: 'Vincent 1996', status: 'source_bound_golden' },
      databases: { eicu: { availability: { status: 'degraded', degraded_dependencies: ['sofa_cns'], missing_dependencies: [] },
        component_availability: { sofa_cns: { status: 'degraded' }, sofa_coag: { status: 'full' } } } } },
    { concept: 'kdigo_aki', description: 'KDIGO acute kidney injury stage', databases: { eicu: { availability: { status: 'blocked', missing_dependencies: ['urine24'] } } } },
  ],
};

function harness(overrides = {}) {
  const calls = [];
  const listeners = {};
  const nodes = [];
  const window = {
    EU_LANG: 'zh',
    EU_DATABASE_NAMES: { keys: () => ['eicu', 'miiv'], name: key => ({ eicu: 'eICU-CRD', miiv: 'MIMIC-IV' }[key] || '') },
    EU_CATALOG: { groups: [['vitals', '💓 Vital Signs', '💓 生命体征']] },
    EU_API: {
      loadClinicalDefinitions: async db => { calls.push(['definitions', db]); return DEFINITIONS; },
      loadExtractionProjection: async db => { calls.push(['extraction', db]); return { databases: [{ database: db, modules: { vitals: { availability: { hr: { status: 'full' }, map: { status: 'degraded' }, x: { status: 'blocked' } } } } }] }; },
      calculateSampleSize: async body => { calls.push(['calculate', JSON.parse(JSON.stringify(body))]); if (overrides.refusal) throw overrides.refusal; return { method: 'two_proportions_cohen_h', implementation: 'statsmodels/0.14.6', group_sizes: [292, 292], total_n: 584, achieved_power: 0.8004 }; },
    },
  };
  const document = {
    addEventListener(name, handler) { listeners[name] = handler; },
    querySelectorAll: selector => nodes.filter(node => selector === `[data-skt-tool="${node.tool}"]`),
    activeElement: null,
  };
  const context = vm.createContext({ window, document, location: { hash: '' } });
  for (const file of ['html-escape.js', 'screens-skills-tools.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const mount = tool => {
    const parts = {};
    const node = { tool, innerHTML: window.EU_SKILL_TOOLS.html(tool), contains: () => false,
      querySelector: selector => parts[selector] || (parts[selector] = { innerHTML: '' }) };
    nodes.push(node);
    return node;
  };
  return { tools: window.EU_SKILL_TOOLS, calls, listeners, mount };
}

test('only the three tool Skills get a tool', () => {
  const { tools } = harness();
  assert.equal(tools.has('clinical-concept-definitions'), true);
  assert.equal(tools.has('sample-size-and-power'), true);
  assert.equal(tools.has('treatment-patterns'), false);
  assert.equal(tools.html('treatment-patterns'), '');
});

test('a definition shows its availability, parts and source for the chosen database', async () => {
  const { calls, listeners, mount } = harness();
  const node = mount('clinical-concept-definitions');
  await settle();
  assert.deepEqual(calls, [['definitions', 'eicu']]);
  assert.match(node.innerHTML, /按 EasyICU 的变量映射判断，不代表数据的实际完整程度/);
  assert.match(node.innerHTML, /<strong>sofa<\/strong><span>sequential organ failure assessment score<\/span><span class="eusk-tool-status is-degraded">部分可用<\/span>/);
  assert.match(node.innerHTML, /<strong>kdigo_aki<\/strong>.*is-blocked">不可用/s);
  listeners.click({ target: { closest: () => ({ dataset: { sktConcept: 'sofa' } }) } });
  const detail = node.querySelector('[data-skt-detail]').innerHTML;
  assert.match(detail, /<dt>部分可用的输入<\/dt><dd><code>sofa_cns<\/code><\/dd>/);
  assert.match(detail, /<dt>时间窗<\/dt><dd>在研究计划中指定<\/dd>/);
  assert.match(detail, /<td><code>sofa_coag<\/code><\/td><td><span class="eusk-tool-status is-full">可用<\/span><\/td>/);
  assert.match(detail, /<h4>SOFA <small>1996<\/small><\/h4>\s*<p>Vincent 1996<\/p>\s*<p>已按出处核对<\/p>/);
  listeners.input({ target: { dataset: { sktQuery: '' }, value: 'kdigo' } });
  assert.doesNotMatch(node.querySelector('[data-skt-list]').innerHTML, /<strong>sofa<\/strong>/);
});

test('extraction modules are counted by mapping status and named by the catalog', async () => {
  const { calls, mount } = harness();
  const node = mount('six-database-cohort-extraction');
  await settle();
  assert.deepEqual(calls, [['extraction', 'eicu']]);
  assert.match(node.innerHTML, /<tr><th scope="row">生命体征<\/th><td>1<\/td><td>1<\/td><td>1<\/td><\/tr>/);
  assert.doesNotMatch(node.innerHTML, /data-skt-open-extraction/);
});

test('a sample size is the host’s answer to the typed assumptions, percentages sent as proportions', async () => {
  const { calls, listeners, mount } = harness();
  const node = mount('sample-size-and-power');
  assert.match(node.innerHTML, /组 1 发生率 \(%\)/);
  listeners.input({ target: { dataset: { sktValue: 'p1' }, value: '40' } });
  listeners.submit({ target: { matches: () => true }, preventDefault() {} });
  await settle();
  assert.deepEqual(calls, [['calculate', { calculation: 'two_proportions', p1: 0.4, p2: 0.2, alpha: 0.05, power: 0.8, allocation_ratio: 1 }]]);
  assert.match(node.innerHTML, /<dt>每组人数<\/dt><dd>292 \/ 292<\/dd>.*<dt>合计<\/dt><dd>584<\/dd>.*<dt>实际把握度<\/dt><dd>80%<\/dd>/s);
  assert.match(node.innerHTML, /用于研究规划，不是研究结果/);
});

test('an empty value is never sent, and a host refusal reads as words', async () => {
  const blank = harness();
  const node = blank.mount('sample-size-and-power');
  blank.listeners.input({ target: { dataset: { sktValue: 'p2' }, value: '' } });
  blank.listeners.submit({ target: { matches: () => true }, preventDefault() {} });
  await settle();
  assert.deepEqual(blank.calls, []);
  assert.match(node.innerHTML, /请为每一项填写数字/);

  const refused = harness({ refusal: Object.assign(new Error('x'), { status: 422, code: 'planning.null_effect' }) });
  const other = refused.mount('sample-size-and-power');
  refused.listeners.submit({ target: { matches: () => true }, preventDefault() {} });
  await settle();
  assert.match(other.innerHTML, /两组假设相同，没有可检出的差异/);
});
