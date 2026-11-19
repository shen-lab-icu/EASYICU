/* Executable contract: a database the score owner marks structurally
   unavailable for a concept (SICdb has no GCS, so no SOFA-2 CNS, no SOFA-2
   total and no SOFA-2 based sepsis) is shown as unavailable and never imputed,
   on the dictionary row and in the derived lineage panel, never as a mapped
   (green) database. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  console, JSON, Object, Number, String, Array, Set, Math, Promise,
  EU_LANG: 'en',
  icon: () => '',
  __euRender() {},
};
context.t = (en, zh) => (context.EU_LANG === 'zh' && zh != null ? zh : en);
context.window = context;
const supportedDbs = ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'];
context.EU_CATALOG = {
  groups: [['sofa2_score', 'SOFA-2 Scores', 'SOFA-2 评分']],
  groupConcepts: { sofa2_score: ['sofa2_cns', 'sofa2_coag'] },
  dict: {
    sofa2_cns: ['SOFA-2 Central Nervous System', 'SOFA-2 中枢神经系统', '0-4'],
    sofa2_coag: ['SOFA-2 Coagulation', 'SOFA-2 凝血', '0-4'],
  },
  desc: {},
  cov: {},
  totalConcepts: 2,
  supportedDbs,
  conceptCoverage: {
    sofa2_cns: { kind: 'derived', databases: 5, declared_databases: ['miiv', 'mimic', 'eicu', 'aumc', 'hirid'], structurally_unavailable: ['sic'], audience: 'clinical' },
    sofa2_coag: { kind: 'derived', databases: 6, declared_databases: supportedDbs, audience: 'clinical' },
  },
  coverageSummary: { supportedDatabases: 6, mappedAll: 1, mappedPartial: 1, unmapped: 0, unsupported: 0, derived: 2 },
};
context.EU_CAPABILITIES = { deployment: { hosted: false } };
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path }));

const rowOf = (markup, key) => markup.split('data-dict-concept="').find(part => part.startsWith(`${key}"`)) || '';
const visible = markup => markup.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

const markup = context.SCREENS.dictionary.render();
const cns = rowOf(markup, 'sofa2_cns');
assert(/<i class="na" title="[^"]*structurally unavailable[^"]*never imputed[^"]*">SIC<\/i>/.test(cns), 'SICdb is marked unavailable on the SOFA-2 CNS row');
assert(!/<i class="on[^"]*">SIC<\/i>/.test(cns), 'SICdb is not shown as mapped for SOFA-2 CNS');
assert(/<i class="on">SIC<\/i>/.test(rowOf(markup, 'sofa2_coag')), 'a component SICdb does record stays mapped');

let handler = null;
const root = {
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: (type, fn) => { if (type === 'click') handler = fn; },
};
context.document = { getElementById: () => null };
context.EU_API = { loadConceptLineage: async key => ({ concept: key, lanes: [], inputs: ['gcs'] }) };
context.SCREENS.dictionary.afterRender(root);
handler({ target: { closest: sel => (sel === '[data-dict-concept]' ? { dataset: { dictConcept: 'sofa2_cns' }, getAttribute: () => 'sofa2_cns' } : null) } });

setTimeout(() => {
  const panel = context.SCREENS.dictionary.render();
  assert(panel.includes('data-dict-structural'), 'the lineage panel carries the structural note');
  assert(visible(panel).includes('SICdb: structurally unavailable'), 'the note names SICdb and says unavailable');
  context.EU_LANG = 'zh';
  assert(visible(context.SCREENS.dictionary.render()).includes('结构性不可得'), 'the Chinese note reads the same');
  console.log(JSON.stringify({ structural_unavailability: true }));
}, 0);
