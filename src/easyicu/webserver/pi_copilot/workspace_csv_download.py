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


def csv_preview_bridge(nonce: str) -> tuple[str, str]:
    """The generated page can supply CSV bytes only; it receives no Host capability."""

    configuration = json.dumps(nonce)
    child = r"""<script>(function(){
      'use strict';
      const nonce=NONCE, limit=1048576, objects=new Map();
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
    })();</script>""".replace("NONCE", configuration)
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
