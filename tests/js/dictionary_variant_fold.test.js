/* Executable contract: a concept with several definitions reads as one
   Data Dictionary entry. Its variants fold under it, open when the entry is
   picked, and still answer a search by their own names. */
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
const clinical = { kind: 'mapped', databases: 6, declared_databases: ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'], audience: 'clinical' };
context.EU_CATALOG = {
  groups: [['renal', '🚰 Renal', '🚰 肾脏与尿量']],
  groupConcepts: { renal: ['crea', 'aki_stage_reference', 'aki_reference', 'aki_stage_uo_reference', 'aki_reference_status'] },
  dict: {
    crea: ['Creatinine', '肌酐', 'mg/dL'],
    aki_stage_reference: ['AKI Stage (KDIGO)', 'AKI 分期（KDIGO）', '0-3'],
    aki_reference: ['AKI (KDIGO Stage ≥1)', 'AKI（KDIGO ≥1 期）', 'boolean'],
    aki_stage_uo_reference: ['AKI Stage by Urine Output (KDIGO)', 'AKI 分期·尿量标准（KDIGO）', '0-3'],
    aki_reference_status: ['AKI Reference Evaluation Status', 'AKI参考口径评估状态', 'category'],
  },
  desc: { aki_stage_uo_reference: ['Stage from urine output alone', '只按每公斤体重尿量分期'] },
  variants: { aki_stage_reference: ['aki_reference', 'aki_stage_uo_reference'] },
  cov: {},
  totalConcepts: 5,
  supportedDbs: ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'],
  conceptCoverage: {
    crea: clinical, aki_stage_reference: clinical, aki_reference: clinical, aki_stage_uo_reference: clinical,
    aki_reference_status: { ...clinical, audience: 'internal' },
  },
};
context.EU_CAPABILITIES = { deployment: { hosted: false } };
context.EU_API = { loadConceptLineage: async concept => ({ concept, lanes: [] }) };
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path }));

const visible = markup => markup.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const dictionary = context.SCREENS.dictionary;
const results = { innerHTML: '' };
const lineage = { innerHTML: '' };
context.document = { getElementById: id => (id === 'dictResults' ? results : id === 'dictLineage' ? lineage : null) };
const input = { value: '', listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }, focus() {} };
let click = null;
dictionary.afterRender({
  querySelector: selector => (selector === '#dictSearchInput' ? input : null),
  querySelectorAll: () => [],
  addEventListener: (type, fn) => { if (type === 'click') click = fn; },
});
const pick = concept => click({ target: { closest: selector => (selector === '[data-dict-concept]' ? { dataset: { dictConcept: concept } } : null) } });

// Folded: one entry, its variant count, the variants' names out of sight.
let text = visible(dictionary.render() + dictionary.rail());
assert(text.includes('AKI 分期（KDIGO）') && text.includes('2 个相关变量'), 'the entry names how many variables fold under it');
assert(!text.includes('AKI（KDIGO ≥1 期）') && !text.includes('尿量标准'), 'variants stay folded');
assert(text.includes('2 个概念') && text.includes('全部 2'), 'counts name entries, not variants');
assert(dictionary.render().includes('aria-expanded="false"'), 'the entry says it is folded');

(async () => {
  // Picking the entry opens its variants beneath it.
  await pick('aki_stage_reference');
  text = visible(results.innerHTML);
  assert(text.includes('AKI（KDIGO ≥1 期）') && text.includes('AKI 分期·尿量标准（KDIGO）'), 'the variants open');
  assert(text.indexOf('AKI 分期（KDIGO）') < text.indexOf('AKI（KDIGO ≥1 期）'), 'variants follow their entry');
  assert(text.includes('2 个概念'), 'opened variants do not change the count');
  assert(results.innerHTML.includes('aria-expanded="true"') && results.innerHTML.includes('is-variant'), 'the entry says it is open');

  // Picking a variant keeps the family open; a second press on the entry folds it.
  await pick('aki_stage_uo_reference');
  assert(visible(results.innerHTML).includes('AKI（KDIGO ≥1 期）'), 'a picked variant keeps its family open');
  await pick('aki_stage_reference');
  await pick('aki_stage_reference');
  assert(!visible(results.innerHTML).includes('AKI（KDIGO ≥1 期）'), 'a second press folds the variants');

  // A search finds a variant by its own words and shows it under its entry.
  input.value = '尿量';
  input.listeners.input();
  text = visible(results.innerHTML);
  assert(text.includes('AKI 分期（KDIGO）') && text.includes('AKI 分期·尿量标准（KDIGO）'), 'a matching variant shows under its entry');
  assert(!text.includes('AKI（KDIGO ≥1 期）'), 'a variant that does not match stays folded');
  assert(!text.includes('肌酐'), 'other entries still filter');
  console.log(JSON.stringify({ ok: true }));
})().catch(error => { console.error(error); process.exit(1); });
