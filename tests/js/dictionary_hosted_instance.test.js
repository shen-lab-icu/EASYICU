/* Executable contract: the Data Dictionary lists the concepts a clinician
   picks, keeps bookkeeping concepts (receipts, evidence flags, helper series)
   behind one toggle, and on a hosted instance counts coverage over the
   databases this instance has instead of the six the dictionary maps. */
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
context.EU_CATALOG = {
  groups: [['vitals', '❤️ Vital signs', '❤️ 生命体征'], ['renal', '🫘 Renal', '🫘 肾脏']],
  groupConcepts: { vitals: ['hr'], renal: ['crea', 'aki_reference_status'] },
  dict: {
    hr: ['Heart rate', '心率', 'bpm'],
    crea: ['Creatinine', '肌酐', 'mg/dL'],
    aki_reference_status: ['AKI reference status', 'AKI参考口径评估状态', ''],
  },
  desc: {},
  cov: { hr: 6, crea: 5, aki_reference_status: 6 },
  totalConcepts: 3,
  supportedDbs: ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'],
  instanceDatabases: ['eicu', 'miiv'],
  conceptCoverage: {
    hr: { kind: 'mapped', databases: 6, declared_databases: ['miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic'], data_audit_databases: 6, audience: 'clinical', instance_databases_covered: ['eicu', 'miiv'] },
    crea: { kind: 'mapped', databases: 5, declared_databases: ['miiv', 'mimic', 'eicu', 'aumc', 'hirid'], data_audit_databases: 4, audience: 'clinical', instance_databases_covered: ['miiv'] },
    aki_reference_status: { kind: 'audited', databases: 6, audience: 'internal', instance_databases_covered: ['eicu', 'miiv'] },
  },
};
const sandbox = vm.createContext(context);
process.argv.slice(2).forEach(path => vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, { filename: path }));

const visible = markup => markup.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const page = () => visible(context.SCREENS.dictionary.render() + context.SCREENS.dictionary.rail());

// Hosted: the instance's databases, no six-database fractions, no catalog note.
context.EU_CAPABILITIES = { deployment: { hosted: true } };
let text = page();
assert(text.includes('心率') && text.includes('肌酐'), 'clinical concepts are listed');
assert(!text.includes('AKI参考口径评估状态'), 'an internal concept is not listed by default');
assert(text.includes('显示内部字段（1）'), 'one toggle offers the hidden fields');
assert(text.includes('本实例数据库'), 'the coverage column names this instance');
assert(text.includes('eICU-CRD · MIMIC-IV') && text.includes(' MIMIC-IV '), 'coverage lists the instance databases a concept has');
assert(!/\d\/6/.test(text) && !text.includes('6 库'), 'no six-database counts on a hosted instance');
assert(!text.includes('这一列显示字典把每个特征映射到了哪些'), 'no explanatory note on a hosted instance');
assert(text.includes('全部 2'), 'the All chip counts listed concepts only');

// The toggle brings the internal concept back.
const markup = context.SCREENS.dictionary.render();
assert(markup.includes('data-dict-internal'), 'the toggle is a button');
let handler = null;
const root = {
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: (type, fn) => { if (type === 'click') handler = fn; },
};
context.document = { getElementById: () => null };
context.SCREENS.dictionary.afterRender(root);
handler({ target: { closest: sel => (sel === '[data-dict-internal]' ? {} : null) } });
text = page();
assert(text.includes('AKI参考口径评估状态') && text.includes('隐藏内部字段'), 'shown after the toggle');
handler({ target: { closest: sel => (sel === '[data-dict-internal]' ? {} : null) } });

// Local desktop: the dictionary audit view stays, internal concepts still behind the toggle.
context.EU_CAPABILITIES = { deployment: { hosted: false } };
text = page();
assert(!text.includes('AKI参考口径评估状态'), 'internal concepts are hidden on the desktop too');
// Local desktop: each concept names the databases the dictionary maps it to.
assert(text.includes('M-IV') && text.includes('HiRID') && text.includes('这一列显示字典把每个特征映射到了哪些'), 'the desktop names the mapped databases');
assert(text.includes('数据核对 4/5'), 'an earlier audit that confirmed fewer databases is marked');
assert(!text.includes('无字典审计'), 'no concept is left as unaudited');

console.log(JSON.stringify({ ok: true }));
