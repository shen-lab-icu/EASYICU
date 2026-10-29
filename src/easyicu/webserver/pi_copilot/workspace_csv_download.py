"""Bounded CSV delivery from an explicitly enabled, isolated HTML preview."""

from __future__ import annotations

import base64
import binascii
import json
import re
from urllib.parse import parse_qs, urlsplit

from fastapi import HTTPException, Request

from easyicu.webserver.host_security import public_origin

MAX_CSV_BYTES = 1024 * 1024
MAX_FORM_BYTES = MAX_CSV_BYTES * 4 + 4096
CSV_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,115}\.csv", re.ASCII)


async def read_csv_download_form(request: Request) -> dict[str, str]:
    """Accept only a same-origin Host form, before allocating an unbounded body."""

    origin = request.headers.get("origin", "")
    try:
        source = urlsplit(origin)
        target = public_origin() or urlsplit(str(request.base_url))
        same_origin = (
            source.scheme in {"http", "https"}
            and (source.scheme, source.hostname, source.port or (443 if source.scheme == "https" else 80))
            == (target.scheme, target.hostname, target.port or (443 if target.scheme == "https" else 80))
            and not (source.username or source.password or source.query or source.fragment)
            and source.path == ""
        )
    except ValueError:
        same_origin = False
    if not same_origin:
        raise HTTPException(403, detail={"error": "pi_workspace_csv_origin_blocked"})
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/x-www-form-urlencoded":
        raise HTTPException(415, detail={"error": "pi_workspace_csv_form_required"})
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_FORM_BYTES:
            raise HTTPException(413, detail={"error": "pi_workspace_csv_too_large"})
        body.extend(chunk)
    try:
        encoded = body.decode("ascii")
        if re.search(r"%(?![0-9a-fA-F]{2})", encoded):
            raise ValueError("Invalid form escape")
        fields = parse_qs(encoded, keep_blank_values=True, strict_parsing=True,
                          encoding="utf-8", errors="strict", max_num_fields=5)
        if set(fields) != {"file", "expected_sha256", "filename", "media_type", "csv_base64"}:
            raise ValueError("Unexpected form fields")
        if any(len(values) != 1 for values in fields.values()):
            raise ValueError("Repeated form fields")
        result = {key: values[0] for key, values in fields.items()}
        if not 1 <= len(result["file"]) <= 240 or not re.fullmatch(r"[a-f0-9]{64}", result["expected_sha256"]):
            raise ValueError("Invalid source coordinates")
        if not CSV_FILENAME.fullmatch(result["filename"]) or result["media_type"] != "text/csv":
            raise ValueError("Only a safe CSV filename is accepted")
        # Form newline normalization must not change the exported CSV bytes.
        csv_bytes = base64.b64decode(result.pop("csv_base64"), validate=True)
        result["csv"] = csv_bytes.decode("utf-8")
        if not csv_bytes or b"\0" in csv_bytes:
            raise ValueError("Invalid CSV text")
        if len(csv_bytes) > MAX_CSV_BYTES:
            raise HTTPException(413, detail={"error": "pi_workspace_csv_too_large"})
        return result
    except (UnicodeError, ValueError, binascii.Error) as exc:
        raise HTTPException(422, detail={"error": "pi_workspace_csv_form_invalid"}) from exc


def csv_preview_bridge(nonce: str, *, generate_csv: bool = False) -> tuple[str, str]:
    """Supply CSV bytes without Host capability; optionally invoke one page export.

    ``generate_csv`` is for an explicit Host user action on an enabled preview.
    It invokes only the page's own unambiguous export control, leaving the
    existing nonce, snapshot checks, and Host save action in place.
    """

    configuration = json.dumps(nonce)
    child = r"""<script>(function(){
      'use strict';
      const nonce=NONCE, generateCsv=GENERATE_CSV, limit=1048576, objects=new Map();
      const create=URL.createObjectURL.bind(URL), revoke=URL.revokeObjectURL.bind(URL);
      const safeName=/^[A-Za-z0-9][A-Za-z0-9_.-]{0,115}\.csv$/;
      let busy=false, count=0, last=0;
      URL.createObjectURL=function(blob){
        const url=create(blob);
        if(blob instanceof Blob && blob.size<=limit && /^text\/csv(?:;charset=utf-8)?$/i.test(blob.type) && objects.size<8) objects.set(url,blob);
        return url;
      };
      URL.revokeObjectURL=function(url){objects.delete(url); return revoke(url);};
      function deliver(anchor){
        const blob=objects.get(anchor.href), name=anchor.download;
        if(!blob || !safeName.test(name) || !name.endsWith('.csv')) return false;
        if(busy || count>=5 || Date.now()-last<1000) return true;
        busy=true; count++; last=Date.now();
        blob.arrayBuffer().then(function(buffer){
          const csv=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(buffer);
          if(csv.length && new TextEncoder().encode(csv).length<=limit && !csv.includes('\0'))
            parent.postMessage({type:'easyicu:workspace-csv-child',nonce,filename:name,media_type:'text/csv',csv},'*');
        }).catch(function(){}).finally(function(){busy=false;});
        return true;
      }
      const click=HTMLAnchorElement.prototype.click;
      HTMLAnchorElement.prototype.click=function(){if(!deliver(this)) return click.call(this);};
      document.addEventListener('click',function(event){
        const anchor=event.target.closest && event.target.closest('a[download]');
        if(anchor && deliver(anchor)){event.preventDefault();event.stopImmediatePropagation();}
      },true);
      if(generateCsv){
        let attempted=false,expired=false,waiting=false,waitTimer=0;
        function usable(element){
          const tag=String(element.tagName||'').toLowerCase();
          if(tag!=='button' && !(tag==='input' && /^(button|submit)$/i.test(element.type||''))
            && !(tag==='a' && element.hasAttribute('download'))) return false;
          if(element.disabled || element.matches(':disabled') || element.getAttribute('aria-disabled')==='true'
            || element.closest('[hidden],[inert],[aria-hidden="true"]')) return false;
          const style=getComputedStyle(element);
          return style.display!=='none' && style.visibility==='visible' && Number(style.opacity)!==0
            && Array.from(element.getClientRects()).some(rect=>rect.width>0 && rect.height>0);
        }
        function csvAction(element){
          const text=[element.textContent,element.value,element.getAttribute('aria-label'),
            element.getAttribute('title'),element.getAttribute('download')].join(' ');
          return /\bcsv\b/i.test(text) && /download|export|下载|导出/i.test(text);
        }
        function declaredControls(){
          const declarations=document.querySelectorAll('script[data-easyicu-runtime-qa]');
          if(!declarations.length) return null;
          if(declarations.length!==1 || String(declarations[0].type).toLowerCase()!=='application/json'
            || declarations[0].textContent.length>16384) return [];
          try{
            const contract=JSON.parse(declarations[0].textContent);
            if(!contract || contract.schema!=='easyicu.web-runtime-qa/1' || !Array.isArray(contract.checks)
              || contract.checks.length>32) return [];
            const checks=contract.checks.filter(check=>check && check.kind==='download_control');
            if(!checks.length) return null;
            const controls=new Set();
            for(const check of checks){
              if(typeof check.selector!=='string' || !check.selector.length || check.selector.length>256
                || /[\u0000-\u001f]/.test(check.selector)) return [];
              const matches=Array.from(document.querySelectorAll(check.selector)).filter(usable);
              if(matches.length!==1) return [];
              controls.add(matches[0]);
            }
            return Array.from(controls);
          }catch(_){return [];}
        }
        function generate(){
          if(attempted || expired) return;
          if(window.__easyicuRuntimeQaRunning===true){waitForQa();return;}
          attempted=true;
          let controls=declaredControls();
          if(controls===null){
            const marked=document.querySelectorAll('[data-easyicu-export="csv"]');
            controls=marked.length ? Array.from(marked).filter(usable)
              : Array.from(document.querySelectorAll('button,input[type="button"],input[type="submit"],a[download]'))
                .filter(element=>usable(element) && csvAction(element));
          }
          if(controls.length===1){try{controls[0].click();}catch(_){/* Manual export stays available. */}}
        }
        function qaFinished(){
          if(expired) return;
          waiting=false;clearTimeout(waitTimer);
          if(window.__easyicuRuntimeQaRunning===true){expired=true;return;}
          schedule();
        }
        function waitForQa(){
          if(waiting || expired) return;
          waiting=true;
          document.addEventListener('easyicu:runtime-qa-finished',qaFinished,{once:true});
          waitTimer=setTimeout(function(){
            expired=true;document.removeEventListener('easyicu:runtime-qa-finished',qaFinished);
          },15000);
        }
        function schedule(){
          if(attempted || expired) return;
          if(window.__easyicuRuntimeQaRunning===true){waitForQa();return;}
          requestAnimationFrame(()=>requestAnimationFrame(generate));
        }
        if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',schedule,{once:true});
        else schedule();
      }
    })();</script>""".replace("NONCE", configuration).replace(
        "GENERATE_CSV", json.dumps(generate_csv is True)
    )
    wrapper = r"""<script>(function(){
      'use strict';
      const nonce=NONCE, frame=document.getElementById('easyicu-workspace-preview-content');
      let count=0,last=0;
      addEventListener('message',function(event){
        const data=event.data;
        if(event.source!==frame.contentWindow || event.origin!=='null' || !data || data.type!=='easyicu:workspace-csv-child' || data.nonce!==nonce) return;
        if(count>=5 || Date.now()-last<1000 || data.media_type!=='text/csv' || typeof data.csv!=='string' || !data.csv.length || data.csv.length>1048576 || new TextEncoder().encode(data.csv).length>1048576 || data.csv.includes('\0') || typeof data.filename!=='string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,115}\.csv$/.test(data.filename) || !data.filename.endsWith('.csv')) return;
        count++;last=Date.now();
        parent.postMessage({type:'easyicu:workspace-csv',nonce,filename:data.filename,media_type:'text/csv',csv:data.csv},'*');
      });
    })();</script>""".replace("NONCE", configuration)
    return child, wrapper
