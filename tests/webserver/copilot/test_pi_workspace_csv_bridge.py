"""Runtime contracts for the isolated CSV bridge and real Host form delivery."""

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


def test_host_bridge_requires_active_frame_and_real_scoped_form_submission():
    source = (Path(__file__).parents[3] / "src/easyicu/webserver/static/js/screens-guided-pi-csv-download.js").read_text()
    _run(f"""
      const assert=require('node:assert/strict');let owner,receive,now=10000,current='project/file/hash';
      class Element {{
        constructor(tag){{this.tag=tag;this.children=[];this.dataset={{}};this.isConnected=true;this.handlers={{}};}}
        setAttribute(name,value){{this[name]=value;}}
        appendChild(node){{node.parentNode=this;this.children.push(node);}}
        removeChild(node){{this.children=this.children.filter(item=>item!==node);node.parentNode=null;}}
        replaceChildren(...nodes){{this.children.forEach(node=>node.parentNode=null);this.children=[];nodes.forEach(node=>this.appendChild(node));}}
        addEventListener(name,fn){{this.handlers[name]=fn;}}
      }}
      global.document={{createElement:tag=>new Element(tag)}};
      global.window={{crypto:require('node:crypto').webcrypto,btoa:value=>Buffer.from(value,'binary').toString('base64'),
        addEventListener(name,fn){{receive=fn;}},EasyICU:{{guidedPi:{{declare(name,value){{owner=value;}}}}}}}};
      const realDate=Date;global.Date={{now:()=>now}};
      eval({json.dumps(source)});
      const bridge=owner.create(()=>current),nonce=bridge.prepare(current),slot=new Element('span'),frame=new Element('iframe');
      frame.contentWindow={{}};
      bridge.bind(frame,slot,{{projectId:'project-a',file:'demo.html',sha256:'a'.repeat(64)}});
      const csv='\\uFEFFid,标记\\nSYN-001,演示\\n';
      const data={{type:'easyicu:workspace-csv',nonce,filename:'synthetic.csv',media_type:'text/csv',csv}};
      const event={{data,source:frame.contentWindow,origin:'null'}};
      receive({{...event,source:{{}}}});receive({{...event,origin:'http://localhost'}});
      receive({{...event,data:{{...data,nonce:'x'}}}});receive({{...event,data:{{...data,filename:'a:b.csv'}}}});
      receive({{...event,data:{{...data,filename:'synthetic.csv\\n'}}}});
      receive({{...event,data:{{...data,csv:'中'.repeat(400000)}}}});
      assert.equal(slot.children.length,0);
      receive(event);assert.equal(slot.children.length,2);
      const [destination,form]=slot.children;
      assert.equal(destination.sandbox,'allow-downloads');assert.equal(form.method,'post');
      assert.equal(form.action,'/api/copilot/pi/projects/project-a/workspace/download-csv');
      assert.equal(form.target,destination.name);
      const fields=Object.fromEntries(form.children.filter(node=>node.tag==='textarea').map(node=>[node.name,node.value]));
      assert.equal(fields.file,'demo.html');assert.equal(fields.expected_sha256,'a'.repeat(64));
      assert.equal(Buffer.from(fields.csv_base64,'base64').toString(),csv);
      assert.ok(form.children.at(-1).textContent.includes('CSV ready'));
      receive(event);assert.equal(slot.children[1],form);
      form.handlers.submit({{preventDefault(){{throw new Error('Valid submission blocked');}}}});
      assert.equal(slot.children[0],destination);assert.equal(form.children.at(-1).disabled,true);
      now+=1001;receive(event);assert.equal(slot.children.length,3);assert.equal(slot.children[0],destination);
      assert.equal(destination.parentNode,slot);
      assert.notEqual(slot.children[1].name,destination.name);
      assert.equal(slot.children[2].target,slot.children[1].name);
      let prevented=false;current='other hash';slot.children.at(-1).handlers.submit({{preventDefault(){{prevented=true;}}}});assert.ok(prevented);
      bridge.reset();assert.equal(slot.children.length,0);receive(event);assert.equal(slot.children.length,0);
      current='project/file/hash';assert.notEqual(bridge.prepare(current),nonce);
      global.Date=realDate;console.log('passed');
    """)
