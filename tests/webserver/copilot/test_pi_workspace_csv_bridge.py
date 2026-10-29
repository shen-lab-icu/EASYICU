"""Runtime contracts for the isolated CSV bridge and Host download links."""

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.workspace_csv_download import csv_preview_bridge
from tests.support.node import run_node


def _run(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    assert run_node(node, script, check=True).stdout.strip() == "passed"


def test_child_bridge_preserves_bom_and_immediately_revoked_disconnected_csv_blob():
    child, _ = csv_preview_bridge("a" * 64)
    script = child.removeprefix("<script>").removesuffix("</script>")
    _run(f"""
      const assert = require('node:assert/strict'), vm = require('node:vm');
      let nativeClicks=0, seq=0; const messages=[];
      class Anchor {{ click() {{ nativeClicks++; }} }}
      const URL={{createObjectURL(){{return 'blob:null/' + (++seq);}},revokeObjectURL(){{}}}};
      const context={{URL,Blob,TextEncoder,TextDecoder,Map,Date,HTMLAnchorElement:Anchor,
        document:{{addEventListener(){{}}}},parent:{{postMessage(value){{messages.push(value);}}}}}};
      vm.runInNewContext({json.dumps(script)},context);
      (async()=>{{
        const csv='\\uFEFFid,标记\\nSYN-001,演示\\n', anchor=new Anchor();
        anchor.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));
        anchor.download='synthetic.csv'; anchor.click(); URL.revokeObjectURL(anchor.href);
        await new Promise(resolve=>setImmediate(resolve));
        assert.equal(nativeClicks,0); assert.equal(messages.length,1);
        assert.equal(messages[0].csv,csv);
        assert.deepEqual(Buffer.from(messages[0].csv),Buffer.from(csv));
        const other=new Anchor(); other.href=URL.createObjectURL(new Blob(['<p>x</p>'],{{type:'text/html'}}));
        other.download='x.csv';other.click(); assert.equal(nativeClicks,1);
        console.log('passed');
      }})().catch(error=>{{console.error(error);process.exit(1);}});
    """)


def test_wrapper_rejects_wrong_child_nonce_origin_and_limited_csv_messages():
    _, wrapper = csv_preview_bridge("a" * 64)
    script = wrapper.removeprefix("<script>").removesuffix("</script>")
    _run(f"""
      const assert=require('node:assert/strict'),vm=require('node:vm');
      const source={{}},messages=[];let handler,now=10000;
      vm.runInNewContext({json.dumps(script)},{{TextEncoder,Date:{{now:()=>now}},
        document:{{getElementById:()=>({{contentWindow:source}})}},
        addEventListener(name,value){{handler=value;}},parent:{{postMessage(value){{messages.push(value);}}}}}});
      const data={{type:'easyicu:workspace-csv-child',nonce:'a'.repeat(64),filename:'synthetic.csv',media_type:'text/csv',csv:'id\\n1\\n'}};
      const event={{data,source,origin:'null'}};
      handler({{...event,source:{{}}}});handler({{...event,origin:'http://localhost'}});
      handler({{...event,data:{{...data,nonce:'b'.repeat(64)}}}});
      handler({{...event,data:{{...data,filename:'../private.csv'}}}});
      handler({{...event,data:{{...data,filename:'synthetic.csv\\n'}}}});
      handler({{...event,data:{{...data,csv:'中'.repeat(400000)}}}});
      assert.equal(messages.length,0);handler(event);handler(event);assert.equal(messages.length,1);
      for(let i=0;i<9;i++){{now+=1001;handler(event);}}
      assert.equal(messages.length,5);console.log('passed');
    """)


def test_host_bridge_uses_cors_origin_and_validates_bounded_snapshot_before_real_download():
    source = (Path(__file__).parents[3] / "src/easyicu/webserver/static/js/screens-guided-pi-csv-download.js").read_text()
    _run(f"""
      const assert=require('node:assert/strict');let owner,receive,now=10000,current='project/file/hash';
      const calls=[],revoked=[],blobs=[];
      class Element {{
        constructor(tag){{this.tag=tag;this.children=[];this.dataset={{}};this.isConnected=true;this.handlers={{}};}}
        setAttribute(name,value){{this[name]=value;}}
        replaceChildren(...nodes){{this.children=nodes;}}
        addEventListener(name,fn){{this.handlers[name]=fn;}}
      }}
      global.document={{createElement:tag=>new Element(tag)}};
      const csv='\\uFEFFid,标记\\nSYN-001,演示\\n',bytes=new TextEncoder().encode(csv);
      function response(hash='a'.repeat(64),body=bytes,length=body.length,encoding='identity') {{
        return {{ok:true,headers:new Headers({{'Content-Type':'application/octet-stream','X-EasyICU-Source-SHA256':hash,'Content-Length':String(length),'Content-Encoding':encoding}}),
          body:new ReadableStream({{start(controller){{controller.enqueue(body);controller.close();}}}})}};
      }}
      global.window={{crypto:require('node:crypto').webcrypto,btoa:value=>Buffer.from(value,'binary').toString('base64'),
        URL:{{createObjectURL(blob){{blobs.push(blob);return 'blob:http://localhost/host-'+blobs.length;}},revokeObjectURL(url){{revoked.push(url);}}}},
        async fetch(url,options){{calls.push({{url,options}});return response();}},
        addEventListener(name,fn){{receive=fn;}},EasyICU:{{guidedPi:{{declare(name,value){{owner=value;}}}}}}}};
      const realDate=Date;global.Date={{now:()=>now}};
      eval({json.dumps(source)});
      const bridge=owner.create(()=>current),slot=new Element('span'),frame=new Element('iframe');frame.contentWindow={{}};
      function setup(){{const nonce=bridge.prepare(current);bridge.bind(frame,slot,{{projectId:'project-a',file:'demo.html',sha256:'a'.repeat(64)}});return nonce;}}
      const nonce=setup(),data={{type:'easyicu:workspace-csv',nonce,filename:'synthetic.csv',media_type:'text/csv',csv}};
      const event={{data,source:frame.contentWindow,origin:'null'}};
      (async()=>{{
        await receive({{...event,source:{{}}}});await receive({{...event,origin:'http://localhost'}});
        await receive({{...event,data:{{...data,nonce:'x'}}}});
        await receive({{...event,data:{{...data,filename:'synthetic.csv\\n'}}}});
        await receive({{...event,data:{{...data,csv:'中'.repeat(400000)}}}});
        assert.equal(calls.length,0);await receive(event);assert.equal(calls.length,1);
        const {{url,options}}=calls[0];assert.equal(url,'/api/copilot/pi/projects/project-a/workspace/download-csv');
        assert.equal(options.method,'POST');assert.equal(options.mode,'cors');assert.equal(options.credentials,'same-origin');
        assert.equal(options.redirect,'error');assert.equal(options.referrerPolicy,'no-referrer');
        assert.equal(options.body.get('file'),'demo.html');assert.equal(options.body.get('expected_sha256'),'a'.repeat(64));
        assert.equal(Buffer.from(options.body.get('csv_base64'),'base64').toString(),csv);
        const [link]=slot.children;assert.equal(link.tag,'a');assert.equal(link.download,'synthetic.csv');
        assert.ok(link.href.startsWith('blob:http://localhost/'));assert.ok(link.textContent.includes('CSV ready'));
        assert.deepEqual(Buffer.from(await blobs[0].arrayBuffer()),Buffer.from(csv));
        let prevented=false;current='changed hash';link.handlers.click({{preventDefault(){{prevented=true;}}}});assert.ok(prevented);
        bridge.reset();assert.equal(slot.children.length,0);assert.equal(revoked.length,1);
        current='project/file/hash';let late;window.fetch=(url,options)=>{{calls.push({{url,options}});return new Promise(resolve=>late=resolve);}};
        const nonce2=setup(),work=receive({{...event,data:{{...data,nonce:nonce2}}}});
        assert.equal(slot.children[0].tag,'span');bridge.reset();assert.ok(calls.at(-1).options.signal.aborted);
        late(response());await work;assert.equal(slot.children.length,0);assert.equal(blobs.length,1);
        let bodyController;window.fetch=async()=>({{ok:true,headers:new Headers({{'Content-Type':'application/octet-stream','X-EasyICU-Source-SHA256':'a'.repeat(64)}}),
          body:new ReadableStream({{start(controller){{bodyController=controller;}}}})}});
        const streamNonce=setup(),streamWork=receive({{...event,data:{{...data,nonce:streamNonce}}}});
        await new Promise(resolve=>setImmediate(resolve));bridge.reset();bodyController.enqueue(bytes);bodyController.close();
        await streamWork;assert.equal(slot.children.length,0);assert.equal(blobs.length,1);
        window.fetch=async()=>response('b'.repeat(64));const nonce3=setup();await receive({{...event,data:{{...data,nonce:nonce3}}}});
        assert.equal(slot.children[0].tag,'span');assert.ok(slot.children[0].textContent.includes('failed'));
        window.fetch=async()=>response('a'.repeat(64),new Uint8Array(1048577),0);
        const nonce4=setup();await receive({{...event,data:{{...data,nonce:nonce4}}}});
        assert.ok(slot.children[0].textContent.includes('failed'));assert.equal(blobs.length,1);
        const compressedCsv=csv+'SYN-002,演示\\n'.repeat(200),decoded=new TextEncoder().encode(compressedCsv);
        const compressed=require('node:zlib').gzipSync(decoded);assert.ok(compressed.length<decoded.length);
        window.fetch=async()=>response('a'.repeat(64),decoded,compressed.length,'gzip');
        const nonce5=setup();await receive({{...event,data:{{...data,nonce:nonce5,csv:compressedCsv}}}});
        assert.equal(slot.children[0].tag,'a');assert.equal(blobs.length,2);
        assert.deepEqual(Buffer.from(await blobs[1].arrayBuffer()),Buffer.from(compressedCsv));
        const changed=bytes.slice();changed[0]^=1;window.fetch=async()=>response('a'.repeat(64),changed);
        const nonce6=setup();await receive({{...event,data:{{...data,nonce:nonce6}}}});
        assert.ok(slot.children[0].textContent.includes('failed'));assert.equal(blobs.length,2);
        window.fetch=async()=>response('a'.repeat(64),bytes.subarray(0,bytes.length-1));
        const nonce7=setup();await receive({{...event,data:{{...data,nonce:nonce7}}}});
        assert.ok(slot.children[0].textContent.includes('failed'));assert.equal(blobs.length,2);
        global.Date=realDate;console.log('passed');
      }})().catch(error=>{{console.error(error);process.exit(1);}});
    """)
