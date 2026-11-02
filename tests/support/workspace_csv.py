"""Node harness for the real workspace CSV filtering bridge."""
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
    host = (Path(__file__).parents[2] / "src/easyicu/webserver/static/js/screens-guided-pi-csv-download.js").read_text()
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
  function exportCsv(original=false){
    clicks++;const rows=!original&&filters.sex==='F'&&filters.unit==='SICU'?5:24;
    const anchor=new Anchor();anchor.download=original?'original.csv':'synthetic.csv';
    anchor.href=childURL.createObjectURL(new Blob(['id,sex,unit\n'+Array.from({length:rows},(_,i)=>
      `${i+1},${filters.sex},${filters.unit}\n`).join('')],{type:'text/csv'}));
    anchor.click();childURL.revokeObjectURL(anchor.href);
  }
  const total=options.count||(options.multi?2:1);
  const controls=Array.from({length:total},(_,index)=>({
    id:index===0?'downloadFiltered':index===1?'downloadOriginal':'extra'+index,
    role:index===0?'filtered':index===1?'original':'extra'+index,
    tagName:'BUTTON',disabled:false,isConnected:true,type:'button',
    textContent:options.labels?.[index]||(total===1?'Download CSV':index===0?'下载当前筛选 CSV':index===1?'下载原始 CSV':'Export CSV '+(index+1)),
    getAttribute(name){return name==='data-export'?this.role:null;},hasAttribute(){return false;},matches(){return false;},closest(){return null;},
    getClientRects(){return [{width:100,height:30}];},click(){exportCsv(index===1);},
  }));
  const declaration={type:'application/json',textContent:JSON.stringify({schema:'easyicu.web-runtime-qa/1',
    checks:controls.map((control,index)=>({id:'csv'+index,kind:'download_control',
      selector:index<2?'button[data-export='+control.role+']':'#'+control.id}))})};
  const childContext={Blob,TextEncoder,TextDecoder,crypto:require('node:crypto').webcrypto,URL:childURL,HTMLAnchorElement:Anchor,Date:{now:()=>now},
    parent:wrapperChildProxy,addEventListener(name,fn){receiveChild=fn;},
    document:{readyState:'complete',addEventListener(name,fn){const values=events.get(name)||[];values.push(fn);events.set(name,values);},
      querySelectorAll(selector){
        if(selector==='script[data-easyicu-runtime-qa]')return total>1&&!options.noContract?[declaration]:[];
        if(selector==='input,select,textarea')return [sexControl,unitControl];
        if(selector==='[data-easyicu-export="csv"]')return [];
        if(selector.startsWith('button[data-export='))return controls.filter(control=>'button[data-export='+control.role+']'===selector);
        if(selector.startsWith('#'))return controls.filter(control=>'#'+control.id===selector);
        return controls;
      }},
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
  const api={bridge,slot,frame,filters,calls,commands,messages,revoked,childContext,hostContext,controls,
    nonce,hostProxy,wrapperProxy,wrapperChildProxy,childProxy,
    clicks:()=>clicks,nativeClicks:()=>nativeClicks,
    manualExport:exportCsv,
    pick(index){slot.children[index+1].handlers.click({preventDefault(){},stopPropagation(){}});},
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
