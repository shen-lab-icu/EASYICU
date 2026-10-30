/* Executable contract: the Data Dictionary is the one page for what each
   database records. A database that holds a variable only as a substitute
   (age in bands, ICU death for death) marks it in the variable's row and its
   lineage, and the study-type view opens from a tab and from an old Data map
   link. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  console, JSON, Object, Number, String, Array, Set, Math, Promise, setImmediate,
  EU_LANG: 'zh',
  icon: () => '',
  rendered: 0,
  __euRender() { context.rendered += 1; },
};
context.t = (en, zh) => (context.EU_LANG === 'zh' && zh != null ? zh : en);
context.window = context;
const all = ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'];
context.EU_CATALOG = {
  groups: [['demo', 'Demographics', '人口学']],
  groupConcepts: { demo: ['age', 'weight', 'death', 'hr', 'los_icu'] },
  dict: { age: ['Age', '年龄', 'years'], weight: ['Weight', '体重', 'kg'], death: ['In-hospital Mortality', '院内死亡', 'boolean'], hr: ['Heart rate', '心率', 'bpm'], los_icu: ['ICU stay length', 'ICU住院时长', 'days'] },
  desc: {},
  variants: {},
  cov: {},
  totalConcepts: 4,
  supportedDbs: all,
  conceptCoverage: {
    age: { kind: 'mapped', databases: 6, declared_databases: all, audience: 'clinical', substitutes: {
      miiv: { kind: 'capped_or_shifted', exact_through: 89 },
      aumc: { kind: 'binned', bin_labels: ['18-39', '40-49', '80+'] },
      hirid: { kind: 'binned', bin_width: 5, top_code: 90 },
      sic: { kind: 'approximate', top_code: 90 },
    } },
    weight: { kind: 'mapped', databases: 6, declared_databases: all, audience: 'clinical', substitutes: { sic: { kind: 'approximate' } } },
    death: { kind: 'mapped', databases: 6, declared_databases: all, audience: 'clinical', substitutes: { hirid: { meaning: 'icu_death', time_kind: 'last_observation_proxy' } } },
    hr: { kind: 'mapped', databases: 6, declared_databases: all, audience: 'clinical' },
    los_icu: { kind: 'mapped', databases: 6, declared_databases: all, audience: 'clinical', source_semantics: {
      sic: { kind: 'care_unit_to_case_closure_proxy', exact_icu_los: false },
    } },
  },
};
context.EU_CAPABILITIES = { deployment: { hosted: false } };
let lineageFor = null;
context.EU_API = {
  loadConceptLineage: async concept => ({ concept, name: concept, canonical: {}, lanes: [
    { database: 'miiv', label: 'MIMIC-IV', mappings: [] }, { database: 'hirid', label: 'HiRID', mappings: [] },
  ] }),
  loadDataMap: async () => ({ databases: [] }),
};
context.EU_PALETTE = { color: () => '#000' };
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path }));

const dictionary = context.SCREENS.dictionary;
const results = { innerHTML: '' };
const lineage = { innerHTML: '' };
context.document = { getElementById: id => (id === 'dictResults' ? results : id === 'dictLineage' ? lineage : null) };
const marks = markup => [...markup.matchAll(/<i class="on sub" title="([^"]*)">([^<]*)<\/i>/g)].map(match => [match[2], match[1]]);
const rowOf = (markup, concept) => markup.split('data-dict-concept="').find(part => part.startsWith(`${concept}"`)) || '';

(async () => {
  // Variables: each substituted database is marked in the row and says how.
  let page = dictionary.render();
  assert(rowOf(page, 'los_icu').includes('首次转入 ICU 或中间监护病房到病例闭合'));
  assert(rowOf(page, 'los_icu').includes('不等于精确的 ICU 住院时长'));
  assert.deepEqual(marks(rowOf(page, 'los_icu')).map(([name]) => name), ['SIC']);
  assert(page.includes('不代表各库临床口径相同、具备精确研究能力'));
  assert.deepEqual(marks(rowOf(page, 'age')).map(([, title]) => title), [
    'MIMIC-IV：89 岁以上的值被截断或平移',
    'AmsterdamUMCdb：按区间记录：18-39、40-49、80+',
    'HiRID：每 5 岁记为一段；90 岁及以上为一段',
    'SICdb：数值经过取整，精确区间边界未说明',
  ]);
  assert.deepEqual(marks(rowOf(page, 'weight')).map(([, title]) => title), ['SICdb：数值经过取整，精确区间边界未说明']);
  assert.deepEqual(marks(rowOf(page, 'death')).map(([, title]) => title), ['HiRID：口径为 ICU 内死亡']);
  assert.deepEqual(marks(rowOf(page, 'hr')), []);
  assert(!/验收|accepted on real data/.test(page), 'nothing claims acceptance on real data');

  // The lineage of a substituted variable says it for the open database.
  let click = null;
  const root = {
    querySelector: selector => (selector === '#dictSearchInput' ? { addEventListener() {} } : null),
    querySelectorAll: () => [],
    addEventListener: (type, fn) => { if (type === 'click') click = fn; },
  };
  dictionary.afterRender(root);
  await click({ target: { closest: selector => (selector === '[data-dict-concept]' ? { dataset: { dictConcept: 'death' } } : null) } });
  assert(!lineage.innerHTML.includes('data-dict-substitute'), 'MIMIC-IV holds death itself');
  await click({ target: { closest: selector => (selector === '[data-dict-lineage-db]' ? { dataset: { dictLineageDb: 'hirid' } } : null) } });
  assert(lineage.innerHTML.includes('HiRID：口径为 ICU 内死亡'), 'the HiRID lane names its endpoint');

  // Study types: a tab on the same page, and where an old Data map link lands.
  assert(page.includes('data-dict-view="studies"') && page.includes('aria-selected="true" data-dict-view="concepts"'));
  let tabs = [];
  dictionary.afterRender({ ...root, querySelectorAll: selector => (selector === '[data-dict-view]' ? tabs : []) });
  tabs = [{ dataset: { dictView: 'studies' }, addEventListener(type, fn) { this.fn = fn; } }];
  dictionary.afterRender({ ...root, querySelectorAll: selector => (selector === '[data-dict-view]' ? tabs : []) });
  tabs[0].fn();
  assert.equal(context.rendered, 1, 'choosing the tab repaints the page');
  page = dictionary.render();
  assert(page.includes('data-dst-root') && !page.includes('dictSearchInput'), 'the study view replaces the variable list');
  context.__euDictView = null;
  tabs = [{ dataset: { dictView: 'concepts' }, addEventListener(type, fn) { this.fn = fn; } }];
  dictionary.afterRender({ ...root, querySelector: () => null, querySelectorAll: selector => (selector === '[data-dict-view]' ? tabs : []) });
  tabs[0].fn();
  assert(dictionary.render().includes('dictSearchInput'), 'the variables tab brings the list back');
  context.__euDictView = 'studies';
  assert(dictionary.render().includes('data-dst-root'), 'an old Data map link opens the study view');
  assert.equal(context.__euDictView, null, 'the link is read once');
  console.log(JSON.stringify({ ok: true }));
})().catch(error => { console.error(error); process.exit(1); });
