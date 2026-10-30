'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../src/easyicu/webserver/static/js/screens-dict.js'), 'utf8');
function deferred() { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; }
function fixture(load) {
  const ctx = { console, JSON, Object, Number, String, Array, Set, Math, Promise,
    EU_LANG: 'en', icon: () => '', __euRender() {}, t: en => en };
  ctx.window = ctx;
  ctx.EU_CATALOG = { groups: [['lab','Lab','实验室']], groupConcepts: { lab: ['crea','lact'] },
    dict: { crea: ['Creatinine','肌酐','mg/dL'], lact: ['Lactate','乳酸','mmol/L'] },
    desc: {}, variants: {}, cov: {}, totalConcepts: 2, supportedDbs: ['aumc'],
    conceptCoverage: { crea: {kind:'mapped',declared_databases:['aumc']}, lact: {kind:'mapped',declared_databases:['aumc']} } };
  ctx.EU_CAPABILITIES = { deployment: { hosted:false } };
  ctx.EU_PALETTE = { color: () => '#000' };
  ctx.EU_DATABASE_NAMES = { key: value => value, name: value => value, abbr: value => value, keys: () => ['aumc'] };
  ctx.EU_API = { loadConceptLineage: load };
  const result = { innerHTML:'' }, panel = { innerHTML:'' };
  ctx.document = { getElementById: id => id === 'dictResults' ? result : id === 'dictLineage' ? panel : null };
  vm.runInNewContext(source, ctx);
  let click;
  ctx.SCREENS.dictionary.afterRender({ querySelector:()=>null, querySelectorAll:()=>[], addEventListener:(type,fn)=>{if(type==='click')click=fn;} });
  const select = id => click({target:{closest: selector => selector === '[data-dict-concept]' ? {dataset:{dictConcept:id}} : null}});
  return {select,panel,result};
}
test('all row selectors and per-source raw fields are inspectable', async () => {
  const f = fixture(async concept => ({ concept, name:'Creatinine', canonical:{unit:'mg/dL'}, lanes:[{database:'aumc',label:'AmsterdamUMCdb',mappings:[
    {table:'numericitems',selector_field:'itemid',selector_values:[1,2,3,4,5,6],value_field:'value',unit_field:'unit',time_field:'measuredat',callback:'convert'},
    {table:'other',selector_field:'name',selector_values:[7,8,9,10,11,12],value_field:'other_value',unit_field:'other_unit',time_field:'other_time'}
  ]}]}));
  await f.select('crea');
  assert.match(f.panel.innerHTML, /5, 6/,'the full list beyond the first four is accessible');
  assert.match(f.panel.innerHTML, /other_value/,'each table has its own source fields');
  assert.match(f.panel.innerHTML, /other_time/);
  assert.match(f.panel.innerHTML, /numericitems/);
  const track = f.panel.innerHTML.split('data-dict-audit-track>')[1].split('</div>\n      <details')[0];
  assert.match(track, /name ∈ \{7, 8, 9, 10, …\}/, 'later mappings stay compact in the audit track');
  assert.doesNotMatch(track, /11, 12/, 'the complete later selector stays in the expandable details');
  assert.match(f.panel.innerHTML, /11, 12/, 'the expanded mappings contain the full later selector');
});
test('an earlier response cannot replace the later selected concept', async () => {
  const first = deferred(), second = deferred();
  const f=fixture(concept => concept==='crea' ? first.promise : second.promise);
  const a=f.select('crea'), b=f.select('lact');
  second.resolve({concept:'lact',name:'Lactate',canonical:{unit:'mmol/L'},lanes:[{database:'aumc',label:'AmsterdamUMCdb',mappings:[]} ]});
  await b;
  first.resolve({concept:'crea',name:'Creatinine',canonical:{unit:'mg/dL'},lanes:[{database:'aumc',label:'AmsterdamUMCdb',mappings:[]} ]});
  await a;
  assert.match(f.panel.innerHTML,/Lactate/);
  assert.doesNotMatch(f.panel.innerHTML,/Creatinine/);
});
test('an earlier failure cannot hide a later concept that is still loading', async () => {
  let reject;
  const first = new Promise((_resolve, fail) => { reject = fail; });
  const second = deferred();
  const f = fixture(concept => concept === 'crea' ? first : second.promise);
  const a = f.select('crea'), b = f.select('lact');
  reject(new Error('old error'));
  await a;
  assert.match(f.panel.innerHTML, /Loading declared lineage/);
  assert.doesNotMatch(f.panel.innerHTML, /old error/);
  second.resolve({concept:'lact',name:'Lactate',canonical:{unit:'mmol/L'},lanes:[{database:'aumc',label:'AmsterdamUMCdb',mappings:[]} ]});
  await b;
  assert.match(f.panel.innerHTML,/Lactate/);
});
