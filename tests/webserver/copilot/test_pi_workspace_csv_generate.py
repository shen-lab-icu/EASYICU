"""Explicit first-export actions stay inside the current sandboxed page."""

from __future__ import annotations

import json
import shutil

import pytest

from easyicu.webserver.pi_copilot.workspace_csv_download import csv_preview_bridge
from tests.support.node import run_node


def _run(body: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    enabled, wrapper = csv_preview_bridge("a" * 64, generate_csv=True)
    disabled, default_wrapper = csv_preview_bridge("a" * 64)
    assert wrapper == default_wrapper, "generation must not alter the Host transport"
    scripts = [source.removeprefix("<script>").removesuffix("</script>")
               for source in (enabled, disabled)]
    harness = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
function setup(options={}){
  const clicked=[],messages=[],frames=[],timers=new Map(),events=new Map();let nextTimer=0,queries=0;
  class Anchor {click(){throw new Error('A CSV should use the bridge, not native navigation');}}
  const URL={createObjectURL(){return 'blob:null/current';},revokeObjectURL(){}};
  const document={readyState:options.readyState||'loading',
    addEventListener(name,fn,settings){
      const rows=events.get(name)||[];rows.push({fn,once:Boolean(settings&&settings.once)});events.set(name,rows);
    },
    removeEventListener(name,fn){events.set(name,(events.get(name)||[]).filter(row=>row.fn!==fn));},
    querySelectorAll(selector){
      queries++;
      if(selector==='script[data-easyicu-runtime-qa]')return options.declarations||[];
      if(selector==='[data-easyicu-export="csv"]')return controls.filter(row=>row.getAttribute('data-easyicu-export')==='csv');
      if(selector==='button,input[type="button"],input[type="submit"],a[download]')return controls;
      if(selector==='#dl'||selector==='#download'||selector==='#declared')return controls.filter(row=>row.id===selector.slice(1));
      if(selector==='.export')return controls.filter(row=>row.exportClass);
      if(selector==='[invalid')throw new Error('Invalid selector');
      return [];
    },
  };
  const context={URL,Blob,TextEncoder,TextDecoder,Map,Set,Date,HTMLAnchorElement:Anchor,document,
    parent:{postMessage(value){messages.push(value);}},
    requestAnimationFrame(fn){frames.push(fn);},
    setTimeout(fn,ms){assert.equal(ms,15000);const id=++nextTimer;timers.set(id,fn);return id;},
    clearTimeout(id){timers.delete(id);},
    getComputedStyle(element){return element.style;},
  };
  context.window=context;context.__easyicuRuntimeQaRunning=Boolean(options.qaRunning);
  const controls=(options.controls||[]).map((value,index)=>({
    tagName:'BUTTON',type:'button',id:value.id||String(index),textContent:value.text||'',value:'',
    style:{display:value.hidden?'none':'block',visibility:value.invisible?'hidden':'visible',opacity:value.transparent?'0':'1'},
    disabled:Boolean(value.disabled),exportClass:Boolean(value.exportClass),
    getAttribute(name){return (value.attrs||{})[name]||null;},
    hasAttribute(name){return Object.hasOwn(value.attrs||{},name);},
    matches(selector){return selector===':disabled'&&Boolean(value.parentDisabled);},
    closest(){return value.inert?{}:null;},
    getClientRects(){return value.zeroArea?[]:[{width:100,height:30}];},
    click(){
      clicked.push(this.id);
      if(value.csv){
        const anchor=new Anchor();anchor.href=URL.createObjectURL(new Blob([
          'id,filter\nSYN-001,'+(context.currentFilter||'all')+'\n'],{type:'text/csv'}));
        anchor.download='synthetic.csv';anchor.click();URL.revokeObjectURL(anchor.href);
      }
    },
  }));
  vm.runInNewContext(scripts[options.enabled===false?1:0],context);
  function emit(name){
    const rows=events.get(name)||[];events.set(name,rows.filter(row=>!row.once));
    rows.forEach(row=>row.fn());
  }
  return {context,clicked,messages,frames,timers,events,controls,
    queryCount:()=>queries,ready:()=>emit('DOMContentLoaded'),finished:()=>emit('easyicu:runtime-qa-finished'),
    frame(){const fn=frames.shift();assert.ok(fn,'Expected the next animation frame');fn();},
    expire(){for(const fn of [...timers.values()])fn();},
  };
}
function contract(selector){return {type:'application/json',textContent:JSON.stringify({
  schema:'easyicu.web-runtime-qa/1',checks:[{id:'csv',kind:'download_control',selector}],
})};}
"""
    program = f"const scripts={json.dumps(scripts)};\n" + harness + "\n" + body
    completed = run_node(node, program, check=True)
    assert completed.stdout.strip() == "passed"


def test_generation_is_opt_in_and_waits_two_frames_after_dom_ready() -> None:
    _run(r"""
const no=setup({enabled:false,controls:[{id:'download',text:'下载合成 CSV'}]});
no.ready();assert.equal(no.clicked.length,0);assert.equal(no.frames.length,0);assert.equal(no.queryCount(),0);
const yes=setup({controls:[{id:'download',text:'下载合成 CSV',csv:true}]});
assert.equal(yes.frames.length,0);yes.ready();assert.equal(yes.frames.length,1);assert.equal(yes.clicked.length,0);
yes.frame();assert.equal(yes.clicked.length,0);yes.frame();assert.deepEqual(yes.clicked,['download']);
yes.ready();assert.equal(yes.frames.length,0);
setImmediate(()=>{
  assert.equal(yes.messages.length,1);assert.equal(yes.messages[0].nonce,'a'.repeat(64));
  assert.equal(yes.messages[0].filename,'synthetic.csv');assert.equal(yes.messages[0].csv,'id,filter\nSYN-001,all\n');
  console.log('passed');
});
""")


def test_declared_and_marked_controls_precede_legacy_semantic_detection() -> None:
    _run(r"""
const declared=setup({readyState:'complete',declarations:[contract('#declared')],controls:[
  {id:'declared',text:'Download CSV'},
  {id:'download',text:'下载 CSV',attrs:{'data-easyicu-export':'csv'}},
]});
declared.frame();declared.frame();assert.deepEqual(declared.clicked,['declared']);
const marked=setup({controls:[{id:'icon',attrs:{'data-easyicu-export':'csv'}},{id:'download',text:'下载 CSV'}]});
marked.ready();marked.frame();marked.frame();assert.deepEqual(marked.clicked,['icon']);
for(const [id,text] of [['dl','下载合成 CSV'],['download','Download synthetic CSV'],['download','Export CSV']]){
  const legacy=setup({controls:[{id,text}]});legacy.ready();legacy.frame();legacy.frame();assert.deepEqual(legacy.clicked,[id]);
}
console.log('passed');
""")


def test_missing_ambiguous_and_invalid_controls_preserve_manual_export() -> None:
    _run(r"""
for(const options of [
  {controls:[]},
  {controls:[{text:'Download CSV'},{text:'Export CSV'}]},
  {controls:[{text:'CSV preview'},{text:'Download PDF'},{text:'Delete file'}]},
  {declarations:[contract('#missing')],controls:[{text:'Download CSV'}]},
  {declarations:[contract('[invalid')],controls:[{text:'Download CSV'}]},
  {declarations:[contract('.export')],controls:[{text:'Download CSV',exportClass:true},{text:'Export CSV',exportClass:true}]},
  {declarations:[contract('#dl'),contract('#dl')],controls:[{id:'dl',text:'Download CSV'}]},
  {declarations:[{type:'application/json',textContent:'invalid'}],controls:[{text:'Download CSV'}]},
  {controls:[{attrs:{'data-easyicu-export':'csv'}},{attrs:{'data-easyicu-export':'csv'}}]},
  {controls:[{disabled:true,attrs:{'data-easyicu-export':'csv'}},{text:'Download CSV'}]},
]){
  const page=setup(options);page.ready();page.frame();page.frame();assert.equal(page.clicked.length,0);
}
console.log('passed');
""")


def test_only_visible_enabled_exports_are_candidates() -> None:
    _run(r"""
const page=setup({controls:[
  {id:'hidden',text:'Download CSV',hidden:true},{id:'disabled',text:'Download CSV',disabled:true},
  {id:'fieldset',text:'Download CSV',parentDisabled:true},{id:'aria',text:'Download CSV',attrs:{'aria-disabled':'true'}},
  {id:'inert',text:'Download CSV',inert:true},{id:'invisible',text:'Download CSV',invisible:true},
  {id:'transparent',text:'Download CSV',transparent:true},{id:'empty',text:'Download CSV',zeroArea:true},
  {id:'dl',text:'下载合成 CSV'},
]});
page.ready();page.frame();page.frame();assert.deepEqual(page.clicked,['dl']);console.log('passed');
""")


def test_generation_waits_for_qa_to_restore_filters_then_two_frames() -> None:
    _run(r"""
const page=setup({qaRunning:true,controls:[{id:'dl',text:'Download CSV',csv:true}]});
page.context.currentFilter='temporary female';page.ready();assert.equal(page.frames.length,0);
assert.equal(page.timers.size,1);assert.equal(page.clicked.length,0);
page.context.currentFilter='all';page.context.__easyicuRuntimeQaRunning=false;page.finished();
assert.equal(page.timers.size,0);assert.equal(page.frames.length,1);page.frame();assert.equal(page.clicked.length,0);
page.frame();assert.deepEqual(page.clicked,['dl']);page.finished();assert.equal(page.frames.length,0);
setImmediate(()=>{assert.equal(page.messages[0].csv,'id,filter\nSYN-001,all\n');console.log('passed');});
""")


def test_qa_timeout_early_event_and_qa_start_race_fail_closed() -> None:
    _run(r"""
const timed=setup({qaRunning:true,controls:[{id:'dl',text:'Download CSV'}]});
timed.ready();timed.expire();timed.context.__easyicuRuntimeQaRunning=false;timed.finished();
assert.equal(timed.frames.length,0);assert.equal(timed.clicked.length,0);
const early=setup({qaRunning:true,controls:[{id:'dl',text:'Download CSV'}]});
early.ready();early.finished();early.context.__easyicuRuntimeQaRunning=false;early.finished();
assert.equal(early.frames.length,0);assert.equal(early.clicked.length,0);
const race=setup({controls:[{id:'dl',text:'Download CSV'}]});race.ready();race.frame();
race.context.__easyicuRuntimeQaRunning=true;race.frame();assert.equal(race.clicked.length,0);assert.equal(race.timers.size,1);
race.context.__easyicuRuntimeQaRunning=false;race.finished();race.frame();race.frame();assert.deepEqual(race.clicked,['dl']);
console.log('passed');
""")
