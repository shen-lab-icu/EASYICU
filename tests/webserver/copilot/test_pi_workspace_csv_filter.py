"""The passive nested-frame CSV bridge preserves live filter state and consent."""

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.workspace_csv_download import csv_preview_bridge
from tests.support.node import run_node


def _run(body: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    child, wrapper = csv_preview_bridge("a" * 64, allow_downloads=False)
    host = (Path(__file__).parents[3] / "src/easyicu/webserver/static/js/screens-guided-pi-csv-download.js").read_text()
    scripts = [source.removeprefix("<script>").removesuffix("</script>") for source in (child, wrapper)]
    harness = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
function setup(options={}){
  let owner,receiveHost,receiveWrapper,receiveChild,allowed=false,key='project/file/'+'b'.repeat(64),now=10000;
  let clicks=0,nativeClicks=0,seq=0,nextTimer=0,loaded;
  const calls=[],commands=[],messages=[],revoked=[],frames=[],tasks=[],timers=new Map(),events=new Map();
  const filters={sex:'all',unit:'all'};
  const sexControl={get value(){return filters.sex==='all'?'':filters.sex;},set value(value){filters.sex=value||'all';}};
  const unitControl={get value(){return filters.unit==='all'?'':filters.unit;},set value(value){filters.unit=value||'all';}};
  class Element{
    constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.isConnected=true;this.handlers={};}
    setAttribute(name,value){this[name]=value;}
    replaceChildren(...values){this.children=values;}
    addEventListener(name,fn){this.handlers[name]=fn;}
  }
  const frame=new Element('iframe'),slot=new Element('span');
  const hostProxy={postMessage(data){messages.push(data);tasks.push(receiveHost({data,source:wrapperProxy,origin:'null'}));}};
  const wrapperProxy={postMessage(data){commands.push(data);receiveWrapper({data,source:hostProxy,origin:'http://localhost'});}};
  const wrapperChildProxy={postMessage(data){receiveWrapper({data,source:childProxy,origin:'null'});}};
  const childProxy={postMessage(data){receiveChild({data,source:wrapperChildProxy,origin:'null'});}};
  frame.contentWindow=wrapperProxy;
  const hostContext={Blob,TextEncoder,TextDecoder,URLSearchParams,AbortController,Headers,ReadableStream,
    Date:{now:()=>now},setTimeout(fn){const id=++nextTimer;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);},
    document:{createElement:tag=>new Element(tag),dispatchEvent(){}},CustomEvent:function(name,value){this.detail=value.detail;},
    crypto:require('node:crypto').webcrypto,btoa:value=>Buffer.from(value,'binary').toString('base64'),
    URL:{createObjectURL(){return 'blob:host/'+(++seq);},revokeObjectURL(value){revoked.push(value);}},
    async fetch(url,options){
      calls.push({url,options});const bytes=Buffer.from(options.body.get('csv_base64'),'base64');
      return {ok:true,headers:new Headers({'Content-Type':'application/octet-stream','X-EasyICU-Source-SHA256':'b'.repeat(64)}),
        body:new ReadableStream({start(controller){controller.enqueue(bytes);controller.close();}})};
    },
    addEventListener(name,fn){receiveHost=fn;},EasyICU:{guidedPi:{declare(name,value){owner=value;}}},
  };hostContext.window=hostContext;
  vm.runInNewContext(hostScript,hostContext);
  const bridge=owner.create(()=>allowed?key:'');
  const nonce=bridge.prepare(key,{passive:true});
  // The real URL carries a freshly generated transport nonce, not an export grant.
  const rendered=scripts.map(source=>source.replaceAll('a'.repeat(64),nonce));
  const childURL={createObjectURL(){return 'blob:child/'+(++seq);},revokeObjectURL(){}};
  class Anchor{click(){nativeClicks++;}}
  function exportCsv(){
    clicks++;const rows=filters.sex==='F'&&filters.unit==='SICU'?5:24;
    const anchor=new Anchor();anchor.download='synthetic.csv';
    anchor.href=childURL.createObjectURL(new Blob(['id,sex,unit\n'+Array.from({length:rows},(_,i)=>
      `${i+1},${filters.sex},${filters.unit}\n`).join('')],{type:'text/csv'}));
    anchor.click();childURL.revokeObjectURL(anchor.href);
  }
  const control={tagName:'BUTTON',disabled:false,type:'button',textContent:'Download CSV',
    getAttribute(){return null;},hasAttribute(){return false;},matches(){return false;},closest(){return null;},
    getClientRects(){return [{width:100,height:30}];},click:exportCsv};
  const childContext={Blob,TextEncoder,TextDecoder,URL:childURL,HTMLAnchorElement:Anchor,Date:{now:()=>now},
    parent:wrapperChildProxy,addEventListener(name,fn){receiveChild=fn;},
    document:{readyState:'complete',addEventListener(name,fn){const values=events.get(name)||[];values.push(fn);events.set(name,values);},
      querySelectorAll(selector){return selector==='script[data-easyicu-runtime-qa]'?[]:
        selector==='input,select,textarea'?[sexControl,unitControl]:[control];}},
    requestAnimationFrame(fn){frames.push(fn);},setTimeout(){throw new Error('No QA wait is expected');},clearTimeout(){},
    getComputedStyle(){return {display:'block',visibility:'visible',opacity:'1'};},
  };childContext.window=childContext;
  vm.runInNewContext(rendered[0],childContext);
  // Frozen from the real 24-row synthetic HTML, SHA-256 8e8b4586...3385ddb.
  // Its programmatic reset intentionally emits neither input nor change.
  let resetHandler;
  childContext.$=selector=>selector==='#sex'?sexControl:selector==='#icu'?unitControl:
    {addEventListener(name,fn){resetHandler=fn;}};
  childContext.render=()=>{};
  vm.runInNewContext("$('#reset').addEventListener('click',()=>{$('#sex').value='';$('#icu').value='';render()});",childContext);
  vm.runInNewContext(rendered[1],{TextEncoder,Date:{now:()=>now},parent:hostProxy,
    document:{getElementById(){return {contentWindow:childProxy,addEventListener(name,fn){loaded=fn;}};}},
    addEventListener(name,fn){receiveWrapper=fn;}});
  if(!options.loading)loaded();
  bridge.bind(frame,slot,{projectId:'project',file:'demo.html',sha256:'b'.repeat(64)});
  const api={bridge,slot,frame,filters,calls,commands,messages,revoked,childContext,hostContext,
    nonce,hostProxy,wrapperProxy,wrapperChildProxy,childProxy,
    clicks:()=>clicks,nativeClicks:()=>nativeClicks,
    manualExport:exportCsv,
    allow(){allowed=true;return bridge.generate();},generate(){now+=1001;return bridge.generate();},
    revoke(){allowed=false;bridge.revoke();},change(sex,unit){
      filters.sex=sex;filters.unit=unit;for(const fn of events.get('change')||[])fn();
    },
    changedHash(){key='project/file/'+'c'.repeat(64);},
    clickWithHandler(handler){
      for(const fn of events.get('click')||[])fn({target:{closest(){return null;}}});
      handler(sexControl,unitControl);
    },
    resetPage(){api.clickWithHandler(resetHandler);},
    loaded(){loaded();},
    host(event){return receiveHost(event);},wrapper(event){receiveWrapper(event);},child(event){receiveChild(event);},
    async settle(){while(frames.length)frames.shift()();await new Promise(resolve=>setImmediate(resolve));await Promise.all(tasks);},
    csv(){return Buffer.from(calls.at(-1).options.body.get('csv_base64'),'base64').toString();},
    cleanup(){bridge.reset();},
  };return api;
}
"""
    program = f"const scripts={json.dumps(scripts)},hostScript={json.dumps(host)};\n" + harness + body
    assert run_node(node, program, check=True).stdout.strip() == "passed"


def test_opt_in_generates_five_current_rows_without_navigation_or_native_download() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.change('F','SICU');
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  assert.ok(page.allow());await page.settle();
  assert.deepEqual(page.filters,{sex:'F',unit:'SICU'});
  assert.equal(page.clicks(),1);assert.equal(page.calls.length,1);
  assert.equal(page.csv().trim().split('\n').length,6);
  assert.equal(page.slot.children[0].tag,'a');assert.ok(page.slot.children[0].textContent.includes('click to save'));
  assert.equal(page.nativeClicks(),0);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_passive_preview_cannot_forward_or_save_a_csv_before_explicit_opt_in() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.change('F','SICU');page.manualExport();await page.settle();
  assert.equal(page.nativeClicks(),0);assert.equal(page.calls.length,0);
  assert.equal(page.messages.filter(data=>data.type==='easyicu:workspace-csv').length,0);
  const data={type:'easyicu:workspace-csv-child',nonce:page.nonce,epoch:'',selection_version:1,
    filename:'synthetic.csv',media_type:'text/csv',csv:'id\n1\n'};
  page.wrapper({data,source:page.childProxy,origin:'null'});await page.settle();
  assert.equal(page.calls.length,0);assert.equal(page.slot.children.length,0);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_filter_change_invalidates_old_save_link_and_new_generation_has_fresh_epoch() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.allow();await page.settle();
  assert.equal(page.csv().trim().split('\n').length,25);
  const oldLink=page.slot.children[0],oldEpoch=page.commands.at(-1).epoch;
  page.change('F','SICU');await page.settle();
  assert.ok(page.revoked.includes(oldLink.href));assert.equal(page.slot.children[0].tag,'span');
  let prevented=false;oldLink.handlers.click({preventDefault(){prevented=true;}});assert.ok(prevented);
  assert.ok(page.generate());assert.notEqual(page.commands.at(-1).epoch,oldEpoch);await page.settle();
  assert.equal(page.csv().trim().split('\n').length,6);
  assert.equal(page.slot.children[0].tag,'a');page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_sources_nonces_old_epochs_repeated_commands_and_acks_fail_closed() -> None:
    _run(r"""
(async()=>{
  const page=setup(),control={type:'easyicu:workspace-csv-control',nonce:page.nonce,action:'generate',epoch:'d'.repeat(64)};
  page.wrapper({data:control,source:{},origin:'http://localhost'});
  page.wrapper({data:{...control,nonce:'e'.repeat(64)},source:page.hostProxy,origin:'http://localhost'});
  page.child({data:control,source:{},origin:'null'});
  page.child({data:control,source:page.wrapperChildProxy,origin:'http://localhost'});
  await page.settle();assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  const csv={type:'easyicu:workspace-csv',nonce:page.nonce,epoch:'d'.repeat(64),selection_version:0,
    filename:'synthetic.csv',media_type:'text/csv',csv:'id\n1\n'};
  await page.host({data:csv,source:page.wrapperProxy,origin:'null'});assert.equal(page.calls.length,0);
  page.allow();const epoch=page.commands.at(-1).epoch;await page.settle();
  const calls=page.calls.length,clicks=page.clicks(),link=page.slot.children[0];
  page.wrapper({data:{...control,epoch},source:page.hostProxy,origin:'http://localhost'});
  await page.host({data:{type:'easyicu:workspace-csv-ack',nonce:page.nonce,epoch,status:'manual'},source:{},origin:'null'});
  await page.host({data:{type:'easyicu:workspace-csv-ack',nonce:page.nonce,epoch,status:'manual'},source:page.wrapperProxy,origin:'null'});
  await page.settle();assert.equal(page.clicks(),clicks);assert.equal(page.slot.children[0],link);
  page.revoke();await page.host({data:{...csv,epoch},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,calls);assert.equal(page.slot.children.length,0);
  page.allow();const newEpoch=page.commands.at(-1).epoch;assert.notEqual(epoch,newEpoch);
  await page.host({data:{...csv,epoch},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,calls);page.changedHash();
  await page.host({data:{...csv,epoch:newEpoch},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,calls);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_qa_active_export_does_not_click_temporary_filters() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.childContext.__easyicuRuntimeQaRunning=true;
  page.allow();await page.settle();
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  assert.ok(page.slot.children[0].textContent.includes('checks are still running'));
  page.childContext.__easyicuRuntimeQaRunning=false;page.change('F','SICU');
  page.generate();await page.settle();assert.equal(page.csv().trim().split('\n').length,6);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_initial_loading_queues_generation_and_repeated_ready_does_not_repeat_export() -> None:
    _run(r"""
(async()=>{
  const page=setup({loading:true});page.change('F','SICU');page.allow();await page.settle();
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  const ready={data:{type:'easyicu:workspace-csv-ready-control',nonce:page.nonce},source:page.wrapperProxy,origin:'null'};
  await page.host({...ready,source:{}});await page.host({...ready,data:{...ready.data,nonce:'e'.repeat(64)}});
  await page.host(ready);await page.host(ready);page.loaded();await page.settle();
  assert.equal(page.clicks(),1);assert.equal(page.calls.length,1);
  assert.equal(page.csv().trim().split('\n').length,6);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_filter_change_and_disable_abort_inflight_host_reply_and_old_payloads() -> None:
    _run(r"""
(async()=>{
  const page=setup(),original=page.hostContext.fetch;
  let finish,pendingSignal;
  page.hostContext.fetch=(url,options)=>{pendingSignal=options.signal;return new Promise(resolve=>{
    finish=async()=>resolve(await original(url,options));
  });};
  page.allow();
  // Wait for the child's Blob promise and Host request without awaiting its deferred response.
  await new Promise(resolve=>setImmediate(resolve));
  const epoch=page.commands.at(-1).epoch;
  // Generation still needs its two animation frames.
  const settling=page.settle();await new Promise(resolve=>setImmediate(resolve));
  assert.ok(finish);page.change('F','SICU');assert.ok(pendingSignal.aborted);
  await finish();await settling;assert.equal(page.slot.children[0].tag,'span');
  const previousCalls=page.calls.length;
  const payload={type:'easyicu:workspace-csv',nonce:page.nonce,epoch,selection_version:0,
    filename:'synthetic.csv',media_type:'text/csv',csv:'id\n24\n'};
  await page.host({data:payload,source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,previousCalls);
  page.hostContext.fetch=original;page.generate();await page.settle();
  assert.equal(page.csv().trim().split('\n').length,6);
  page.revoke();assert.equal(page.slot.children.length,0);
  await page.host({data:{...payload,epoch:page.commands.at(-2).epoch,selection_version:1},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.slot.children.length,0);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_real_programmatic_reset_invalidates_ready_csv_at_first_microtask_without_raw_values() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.change('F','SICU');page.allow();await page.settle();
  assert.equal(page.csv().trim().split('\n').length,6);
  const link=page.slot.children[0];
  page.clickWithHandler(()=>page.manualExport());await Promise.resolve();
  assert.equal(page.slot.children[0],link);await page.settle();
  assert.equal(page.slot.children[0],link,'Export click with unchanged filters must preserve its ready link');
  page.resetPage();await Promise.resolve();
  assert.deepEqual(page.filters,{sex:'all',unit:'all'});
  assert.ok(page.revoked.includes(link.href),'A synchronous reset must revoke the old link before the next user task');
  assert.equal(page.slot.children[0].tag,'span');
  const changes=page.messages.filter(data=>data.type==='easyicu:workspace-csv-changed');
  assert.equal(changes.length,1);
  assert.deepEqual(Object.keys(changes[0]).sort(),['epoch','nonce','selection_version','type']);
  await page.settle();assert.equal(page.messages.filter(data=>data.type==='easyicu:workspace-csv-changed').length,1);
  page.generate();await page.settle();assert.equal(page.csv().trim().split('\n').length,25);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")
