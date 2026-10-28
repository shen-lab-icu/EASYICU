/* Executable contract: an outcome EasyICU computes in no database yet reads
   "Not supported yet" with what it would need, on the row and in the detail
   panel, never "No mapping", on the desktop and on a hosted instance alike. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  console, JSON, Object, Number, String, Array, Set, Math,
  EU_LANG: 'zh',
  icon: () => '',
  __euRender() {},
};
context.t = (en, zh) => (context.EU_LANG === 'zh' && zh != null ? zh : en);
context.window = context;
const supportedDbs = ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'];
const unsupported = {
  kind: 'unsupported', databases: 0, declared_databases: [], basis: 'concept_dictionary',
  source: 'outcomes_loader', unavailable_reason: 'outcome_concept_structurally_unavailable',
  audience: 'clinical', instance_databases_covered: ['miiv'],
};
context.EU_CATALOG = {
  groups: [['outcome', 'Outcomes', '结局']],
  groupConcepts: { outcome: ['mort_28d', 'icu_free_days_28', 'icu_readmission', 'new_outcome'] },
  dict: {
    mort_28d: ['28-day mortality', '28 天死亡', 'boolean'],
    icu_free_days_28: ['ICU-free Days (to day 28)', 'ICU-free 天数 (至第28天)', 'days'],
    icu_readmission: ['ICU Readmission (same hospitalisation)', 'ICU 再入 (同次住院)', 'boolean'],
    new_outcome: ['New outcome', '新结局', ''],
  },
  desc: {},
  cov: {},
  totalConcepts: 4,
  supportedDbs,
  conceptCoverage: {
    mort_28d: { kind: 'derived', databases: 2, declared_databases: ['miiv', 'sic'], audience: 'clinical', instance_databases_covered: ['miiv'] },
    icu_free_days_28: unsupported,
    icu_readmission: unsupported,
    new_outcome: unsupported,
  },
  coverageSummary: { supportedDatabases: 6, mappedAll: 0, mappedPartial: 1, unmapped: 0, unsupported: 3, derived: 1 },
};
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path }));

const visible = markup => markup.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const rowOf = (markup, key) => markup.split('data-dict-concept="').find(part => part.startsWith(`${key}"`)) || '';

for (const hosted of [false, true]) {
  context.EU_CAPABILITIES = { deployment: { hosted } };
  context.EU_CATALOG.instanceDatabases = hosted ? ['miiv'] : undefined;
  const markup = context.SCREENS.dictionary.render();
  const where = hosted ? 'hosted' : 'desktop';
  const freeDays = visible(rowOf(markup, 'icu_free_days_28'));
  assert(freeDays.includes('暂不支持') && freeDays.includes('需完整 ICU 入住记录和第 28 天生存'), `${where}: the row says it is not supported and what it needs`);
  assert(!freeDays.includes('未声明映射') && !freeDays.includes('MIMIC-IV') && !freeDays.includes('本实例无'), `${where}: no mapping or instance claim on an unsupported outcome`);
  assert(visible(rowOf(markup, 'icu_readmission')).includes('需同次住院的全部 ICU 入住'), `${where}: each outcome names its own need`);
  assert(visible(rowOf(markup, 'new_outcome')).includes('暂未在任何库中计算'), `${where}: an outcome without its own entry reads the general reason`);
  assert(!visible(rowOf(markup, 'mort_28d')).includes('暂不支持'), `${where}: a supported outcome keeps its coverage`);
  assert(rowOf(markup, 'icu_free_days_28').includes('title="EasyICU 暂不计算这个结局。'), `${where}: the full reason is on the badge`);
}

// Selecting the concept shows the full reason in the detail panel.
let handler = null;
const root = {
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: (type, fn) => { if (type === 'click') handler = fn; },
};
context.document = { getElementById: () => null };
const lineageCalls = [];
context.EU_API = { loadConceptLineage: async key => { lineageCalls.push(key); return { concept: key, lanes: [] }; } };
context.SCREENS.dictionary.afterRender(root);
handler({ target: { closest: sel => (sel === '[data-dict-concept]' ? { dataset: { dictConcept: 'icu_free_days_28' }, getAttribute: () => 'icu_free_days_28' } : null) } });
const panel = visible(context.SCREENS.dictionary.render());
assert(panel.includes('它需要患者的全部 ICU 入住记录和第 28 天时是否存活'), 'the detail panel gives the full reason');
assert(!panel.includes('未声明原始来源映射'), 'the detail panel does not call it an unmapped source');
assert.deepEqual(lineageCalls, [], 'an unsupported outcome fetches no lineage');

context.EU_LANG = 'en';
const english = visible(rowOf(context.SCREENS.dictionary.render(), 'icu_free_days_28'));
assert(english.includes('Not supported yet') && english.includes('Needs complete ICU stays and day-28 survival'), 'the English row reads the same');

console.log(JSON.stringify({ unsupported_outcome_badge: true }));
