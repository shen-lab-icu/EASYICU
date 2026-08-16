"""Capability policy for the native research workbench.

This module backs the Settings capability switches with concrete backend
behavior: local status probes, allow/blocked decisions, and a small audit log.
It does not execute external tools or remote compute by itself.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from easyicu.research_agent.publication_skills import PUBLICATION_SKILLS
from easyicu.webserver import state_paths
from easyicu.webserver import settings as settings_store

_STATE_DIR = state_paths.state_root()
_AUDIT_PATH = _STATE_DIR / "capability_tool_audit.jsonl"
_ZOTERO_LOCAL_API = "http://127.0.0.1:23119/api/users/0/items"
_ZOTERO_TIMEOUT_SECONDS = 0.35
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:18|19|20|21|22)\d{2}\b")

CAPABILITY_SETTINGS = {
    "science_skills_enabled": True,
    "nature_figure_skill_enabled": True,
    "nature_writing_skill_enabled": True,
    "connector_pubmed_enabled": True,
    "connector_zotero_enabled": False,
    "mcp_tools_enabled": False,
    "prompt_contracts_enabled": True,
    "tool_audit_enabled": True,
    "remote_compute_enabled": False,
}

MCP_TOOL_REGISTRY = [
    {
        "id": "pubmed_metadata_search",
        "label": "PubMed metadata search",
        "scope": "external_metadata",
        "setting": "connector_pubmed_enabled",
    },
    {
        "id": "zotero_local_search",
        "label": "Zotero local library search",
        "scope": "local_connector",
        "setting": "connector_zotero_enabled",
    },
    {
        "id": "agent_artifact_reader",
        "label": "Agent artifact reader",
        "scope": "local_artifact",
        "setting": None,
    },
    {
        "id": "idea_run_reader",
        "label": "Idea Mining run reader",
        "scope": "local_artifact",
        "setting": None,
    },
    {
        "id": "remote_compute_submit",
        "label": "Remote compute submit",
        "scope": "remote_compute",
        "setting": "remote_compute_enabled",
    },
]

PROMPT_CONTRACT_RULES = [
    {
        "id": "case_neutral_global_prompt",
        "label": "Global prompts stay case-neutral",
        "enforced": True,
    },
    {
        "id": "case_rules_live_in_protocol",
        "label": "Case-specific study rules live in protocols or rubrics",
        "enforced": True,
    },
    {
        "id": "patient_rows_never_in_provider_prompt",
        "label": "Patient rows never enter provider prompts",
        "enforced": True,
    },
]


def _zotero_author(data: Dict[str, Any]) -> str:
    creators = data.get("creators") if isinstance(data.get("creators"), list) else []
    if not creators:
        return ""
    first = creators[0] if isinstance(creators[0], dict) else {}
    return " ".join(
        part for part in [first.get("firstName"), first.get("lastName")] if part
    )


def _zotero_year(value: Any) -> str:
    text = str(value or "")
    for idx in range(max(0, len(text) - 3)):
        chunk = text[idx : idx + 4]
        if chunk.isdigit() and 1800 <= int(chunk) <= 2200:
            return chunk
    return ""


def _zotero_item_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    data = row.get("data") if isinstance(row.get("data"), dict) else row
    data = data if isinstance(data, dict) else {}
    key = row.get("key") if isinstance(row, dict) else None
    key = key or data.get("key")
    abstract = data.get("abstractNote") or data.get("abstract") or ""
    journal = (
        data.get("publicationTitle")
        or data.get("journalAbbreviation")
        or data.get("journal")
        or data.get("proceedingsTitle")
        or data.get("publisher")
    )
    doi = data.get("DOI") or data.get("doi")
    year = data.get("year") or _zotero_year(data.get("date"))
    return {
        "key": key,
        "title": data.get("title") or "Untitled",
        "item_type": data.get("itemType"),
        "date": data.get("date"),
        "year": year,
        "doi": doi,
        "url": data.get("url"),
        "journal": journal,
        "abstract": abstract,
        "first_author": _zotero_author(data),
        "citation_key": data.get("citation_key") or key,
    }


def _zotero_source_payload(
    item: Dict[str, Any], source_origin: str = "zotero_desktop"
) -> Dict[str, Any]:
    title = str(item.get("title") or "Untitled").strip()
    abstract = str(item.get("abstract") or "").strip()
    origin = source_origin or "zotero_desktop"
    origin_label = (
        "Pasted literature metadata"
        if origin == "pasted_literature"
        else "Zotero Desktop"
    )
    return {
        "source_type": "zotero",
        "source_origin": origin,
        "source_origin_label": origin_label,
        "topic": title,
        "title": title,
        "journal": item.get("journal") or "",
        "year": item.get("year") or item.get("date") or "",
        "doi": item.get("doi") or "",
        "url": item.get("url") or "",
        "excerpt": abstract[:420],
        "abstract": abstract,
        "citation_key": item.get("citation_key") or item.get("key") or "",
        "zotero_key": item.get("key") or "",
    }


def _compact_text(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip() if limit and len(text) > limit else text


def _strip_bib_value(value: str) -> str:
    text = str(value or "").strip().rstrip(",").strip()
    if len(text) >= 2 and (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith('"') and text.endswith('"'))
    ):
        text = text[1:-1]
    return _compact_text(text)


def _bibtex_field(text: str, names: List[str]) -> str:
    for name in names:
        match = re.search(
            rf"(?is)\b{re.escape(name)}\s*=\s*"
            r"(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)",
            text,
        )
        if match:
            return _strip_bib_value(match.group(1))
    return ""


def _bibtex_key(text: str) -> str:
    match = re.search(r"(?is)@\w+\s*\{\s*([^,\s]+)", text)
    return _compact_text(match.group(1), limit=120) if match else ""


def _ris_fields(text: str) -> Dict[str, List[str]]:
    fields: Dict[str, List[str]] = {}
    for raw_line in str(text or "").splitlines():
        match = re.match(r"^([A-Z0-9]{2})\s{2}-\s*(.*)$", raw_line.strip())
        if not match:
            continue
        tag, value = match.groups()
        fields.setdefault(tag, []).append(_compact_text(value))
    return fields


def _first_ris(fields: Dict[str, List[str]], tags: List[str]) -> str:
    for tag in tags:
        values = [value for value in fields.get(tag, []) if value]
        if values:
            return values[0]
    return ""


def _extract_doi(text: str) -> str:
    match = _DOI_RE.search(str(text or ""))
    return match.group(0).rstrip(".,;)") if match else ""


def _extract_year(text: str) -> str:
    match = _YEAR_RE.search(str(text or ""))
    return match.group(0) if match else ""


def _first_author_from_bibtex(value: str) -> str:
    first = re.split(r"\s+and\s+", str(value or ""), maxsplit=1, flags=re.IGNORECASE)[0]
    return _compact_text(first)


def _pasted_source_item(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    bib_key = _bibtex_key(raw)
    ris = _ris_fields(raw)
    title = (
        _bibtex_field(raw, ["title"])
        or _first_ris(ris, ["TI", "T1", "TT"])
    )
    journal = (
        _bibtex_field(raw, ["journal", "journaltitle", "booktitle"])
        or _first_ris(ris, ["JO", "JF", "JA", "T2"])
    )
    doi = _bibtex_field(raw, ["doi"]) or _first_ris(ris, ["DO"]) or _extract_doi(raw)
    url = _bibtex_field(raw, ["url"]) or _first_ris(ris, ["UR", "L1", "L2"])
    year = (
        _extract_year(_bibtex_field(raw, ["year", "date"]))
        or _extract_year(_first_ris(ris, ["PY", "Y1", "DA"]))
        or _extract_year(raw)
    )
    abstract = (
        _bibtex_field(raw, ["abstract"])
        or _compact_text(" ".join(ris.get("AB", []) or ris.get("N2", [])))
    )
    first_author = (
        _first_author_from_bibtex(_bibtex_field(raw, ["author"]))
        or _first_ris(ris, ["AU", "A1"])
    )

    if not title:
        lines = [_compact_text(line) for line in raw.splitlines() if _compact_text(line)]
        non_doi_lines = [line for line in lines if not _DOI_RE.fullmatch(line)]
        title = (non_doi_lines or lines or ["Pasted literature source"])[0]
        if not abstract:
            body = non_doi_lines[1:] if len(non_doi_lines) > 1 else lines[1:]
            abstract = _compact_text(" ".join(body), limit=1800)

    return {
        "key": bib_key or "",
        "title": _compact_text(title, limit=500) or "Pasted literature source",
        "item_type": "pasted_literature_source",
        "date": year,
        "year": year,
        "doi": doi,
        "url": url,
        "journal": journal,
        "abstract": _compact_text(abstract, limit=1800),
        "first_author": first_author,
        "citation_key": bib_key or doi or "",
    }


def capability_settings() -> Dict[str, bool]:
    settings = settings_store.load_settings()
    return {
        key: bool(settings.get(key, fallback))
        for key, fallback in CAPABILITY_SETTINGS.items()
    }


def zotero_status(settings: Dict[str, bool] | None = None) -> Dict[str, Any]:
    settings = settings or capability_settings()
    if not settings.get("connector_zotero_enabled", False):
        return {
            "enabled": False,
            "available": False,
            "status": "disabled",
            "reason": "connector_zotero_enabled_false",
            "local_api": _ZOTERO_LOCAL_API,
        }
    try:
        req = urllib.request.Request(
            f"{_ZOTERO_LOCAL_API}?limit=1",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_ZOTERO_TIMEOUT_SECONDS) as res:
            status_code = int(getattr(res, "status", 200) or 200)
            ok = 200 <= status_code < 300
            return {
                "enabled": True,
                "available": ok,
                "status": "available" if ok else "unavailable",
                "reason": "local_zotero_api_ready" if ok else "local_zotero_api_http_error",
                "http_status": status_code,
                "local_api": _ZOTERO_LOCAL_API,
            }
    except Exception as exc:  # noqa: BLE001 - status probe must fail closed
        return {
            "enabled": True,
            "available": False,
            "status": "unavailable",
            "reason": "local_zotero_api_unavailable",
            "detail": exc.__class__.__name__,
            "local_api": _ZOTERO_LOCAL_API,
        }


def test_zotero_connection() -> Dict[str, Any]:
    settings = capability_settings()
    status = zotero_status(settings)
    record_tool_event(
        "zotero_connection_test",
        {
            "enabled": bool(status.get("enabled")),
            "available": bool(status.get("available")),
            "status": status.get("status"),
            "reason": status.get("reason"),
        },
    )
    return {"ok": True, "status": status}


def search_zotero(query: str, limit: int = 5) -> Dict[str, Any]:
    text = str(query or "").strip()
    settings = capability_settings()
    status = zotero_status(settings)
    if not settings.get("connector_zotero_enabled", False):
        return {"ok": True, "blocked": True, "status": status, "items": []}
    if not status.get("available"):
        return {"ok": True, "blocked": True, "status": status, "items": []}
    if not text:
        return {
            "ok": False,
            "blocked": False,
            "status": status,
            "error": "query_required",
            "items": [],
        }

    params = urllib.parse.urlencode(
        {"q": text, "limit": max(1, min(20, int(limit or 5)))}
    )
    req = urllib.request.Request(
        f"{_ZOTERO_LOCAL_API}?{params}",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as res:
            payload = json.loads(res.read().decode("utf-8") or "[]")
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": True,
            "blocked": True,
            "status": {
                **status,
                "available": False,
                "reason": "local_zotero_search_failed",
                "detail": exc.__class__.__name__,
            },
            "items": [],
        }

    items = []
    for row in payload if isinstance(payload, list) else []:
        if isinstance(row, dict):
            items.append(_zotero_item_from_row(row))
    record_tool_event(
        "zotero_local_search",
        {"query": text, "result_count": len(items), "blocked": False},
    )
    return {"ok": True, "blocked": False, "status": status, "items": items}


def zotero_source(item: Dict[str, Any] | None = None, item_key: str = "") -> Dict[str, Any]:
    settings = capability_settings()
    status = zotero_status(settings)
    if not settings.get("connector_zotero_enabled", False):
        return {"ok": True, "blocked": True, "status": status, "suggested_payload": {}}
    if item and isinstance(item, dict):
        mapped = _zotero_item_from_row(item)
    else:
        if not status.get("available"):
            return {"ok": True, "blocked": True, "status": status, "suggested_payload": {}}
        key = str(item_key or "").strip()
        if not key:
            return {
                "ok": False,
                "blocked": False,
                "status": status,
                "error": "zotero_item_required",
                "suggested_payload": {},
            }
        req = urllib.request.Request(
            f"{_ZOTERO_LOCAL_API}/{urllib.parse.quote(key)}",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as res:
                row = json.loads(res.read().decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": True,
                "blocked": True,
                "status": {
                    **status,
                    "available": False,
                    "reason": "local_zotero_item_fetch_failed",
                    "detail": exc.__class__.__name__,
                },
                "suggested_payload": {},
            }
        mapped = _zotero_item_from_row(row if isinstance(row, dict) else {})

    suggested = _zotero_source_payload(mapped, source_origin="zotero_desktop")
    record_tool_event(
        "zotero_source_selected",
        {
            "key": mapped.get("key"),
            "title": mapped.get("title"),
            "blocked": False,
        },
    )
    return {
        "ok": True,
        "blocked": False,
        "status": status,
        "item": mapped,
        "suggested_payload": suggested,
        "source_adapter": {
            "status": "literature_source_ready",
            "source_type": "zotero",
            "source_origin": "zotero_desktop",
            "display_status": "Literature source ready / 文献来源已就绪",
            "display_reason": (
                "Selected from Zotero Desktop metadata; no full text is stored by EasyICU. "
                "/ 已从 Zotero Desktop 元数据选择；EasyICU 不保存全文。"
            ),
            "network_calls": 0,
            "external_llm_calls": 0,
            "full_text_stored": False,
            "reason": "Selected metadata from the local Zotero library; no full text is stored by EasyICU.",
        },
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "uploads": 0,
        },
    }


def import_zotero_source(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {
            "ok": False,
            "blocked": False,
            "error": "zotero_paste_required",
            "suggested_payload": {},
        }
    mapped = _pasted_source_item(raw)
    suggested = _zotero_source_payload(mapped, source_origin="pasted_literature")
    record_tool_event(
        "zotero_paste_import",
        {
            "title": mapped.get("title"),
            "doi": mapped.get("doi"),
            "citation_key": mapped.get("citation_key"),
            "blocked": False,
        },
    )
    return {
        "ok": True,
        "blocked": False,
        "status": {
            "enabled": True,
            "available": True,
            "status": "literature_source_ready",
            "reason": "literature_source_ready",
        },
        "item": mapped,
        "suggested_payload": suggested,
        "source_adapter": {
            "status": "literature_source_ready",
            "source_type": "zotero",
            "source_origin": "pasted_literature",
            "display_status": "Literature source ready / 文献来源已就绪",
            "display_reason": (
                "Parsed locally from pasted DOI, BibTeX, RIS, or title/abstract metadata; "
                "no Zotero setup is required. / 已从粘贴的 DOI、BibTeX、RIS 或标题摘要元数据本地解析；"
                "不需要配置 Zotero。"
            ),
            "network_calls": 0,
            "external_llm_calls": 0,
            "full_text_stored": False,
            "reason": (
                "Parsed locally from pasted DOI, BibTeX, RIS, or title/abstract metadata; "
                "no Zotero setup is required. / 已从粘贴的 DOI、BibTeX、RIS 或标题摘要元数据本地解析；"
                "不需要配置 Zotero。"
            ),
        },
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "uploads": 0,
        },
    }


def mcp_tool_policy(settings: Dict[str, bool] | None = None) -> Dict[str, Any]:
    settings = settings or capability_settings()
    mcp_enabled = settings.get("mcp_tools_enabled", False)
    tools = []
    for row in MCP_TOOL_REGISTRY:
        required = row.get("setting")
        allowed = bool(mcp_enabled or row.get("scope") == "local_artifact")
        if required:
            allowed = allowed and bool(settings.get(str(required), False))
        reason = "allowed" if allowed else "blocked_by_setting"
        if not mcp_enabled and row.get("scope") != "local_artifact":
            reason = "mcp_tools_enabled_false"
        tools.append({**row, "allowed": allowed, "reason": reason})
    return {
        "enabled": bool(mcp_enabled),
        "tools": tools,
        "allowed_tools": [row["id"] for row in tools if row["allowed"]],
        "blocked_tools": [row["id"] for row in tools if not row["allowed"]],
    }


def check_tool_allowed(tool_id: str) -> Dict[str, Any]:
    policy = mcp_tool_policy()
    row = next((item for item in policy["tools"] if item["id"] == tool_id), None)
    if not row:
        return {
            "ok": True,
            "tool_id": tool_id,
            "allowed": False,
            "reason": "unknown_tool",
            "policy": policy,
        }
    return {"ok": True, "tool_id": tool_id, **row, "policy": policy}


def remote_compute_status(settings: Dict[str, bool] | None = None) -> Dict[str, Any]:
    settings = settings or capability_settings()
    enabled = bool(settings.get("remote_compute_enabled", False))
    return {
        "enabled": enabled,
        "available": False,
        "status": "adapter_not_configured" if enabled else "disabled",
        "reason": (
            "remote_compute_adapter_not_configured"
            if enabled
            else "remote_compute_enabled_false"
        ),
    }


def validate_compute_target(body: Dict[str, Any]) -> Dict[str, Any]:
    target = str(body.get("compute_target") or body.get("compute") or "local").strip()
    if target in {"", "local", "browser", "native"}:
        return {"ok": True, "compute_target": "local"}
    status = remote_compute_status()
    if not status.get("enabled"):
        return {
            "ok": False,
            "compute_target": target,
            "error": "remote_compute_disabled",
            "status": status,
        }
    return {
        "ok": False,
        "compute_target": target,
        "error": "remote_compute_adapter_not_configured",
        "status": status,
    }


def prompt_contract_status(settings: Dict[str, bool] | None = None) -> Dict[str, Any]:
    settings = settings or capability_settings()
    enabled = bool(settings.get("prompt_contracts_enabled", True))
    return {
        "enabled": enabled,
        "status": "enforced" if enabled else "reminders_disabled",
        "rules": PROMPT_CONTRACT_RULES if enabled else [],
    }


def record_tool_event(event_type: str, detail: Dict[str, Any] | None = None) -> Dict[str, Any]:
    settings = capability_settings()
    if not settings.get("tool_audit_enabled", True):
        return {"recorded": False, "reason": "tool_audit_enabled_false"}
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": str(event_type or "tool_event"),
        "detail": detail or {},
    }
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        with _AUDIT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        # A read-only home directory must not take down the business endpoint
        # that triggered the audit write. Fail closed by reporting the audit
        # gap so callers can surface it, but keep the audit path free of PHI.
        return {
            "recorded": False,
            "reason": "tool_audit_write_failed",
            "path": str(_AUDIT_PATH),
            "error": type(exc).__name__,
            "event": event,
        }
    return {"recorded": True, "path": str(_AUDIT_PATH), "event": event}


def audit_events(limit: int = 20) -> Dict[str, Any]:
    max_items = max(1, min(100, int(limit or 20)))
    try:
        lines = _AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    events: List[Dict[str, Any]] = []
    for line in lines[-max_items:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return {"ok": True, "path": str(_AUDIT_PATH), "count": len(events), "events": events}


def capability_status() -> Dict[str, Any]:
    settings = capability_settings()
    skills_master_enabled = bool(settings["science_skills_enabled"])
    publication_skills = [
        skill.to_dict(
            enabled=(
                skills_master_enabled
                and bool(settings.get(skill.setting_key, skill.default_enabled))
            )
        )
        for skill in PUBLICATION_SKILLS
    ]
    zotero = zotero_status(settings)
    mcp = mcp_tool_policy(settings)
    remote = remote_compute_status(settings)
    prompt_contracts = prompt_contract_status(settings)
    audit = {
        "enabled": bool(settings.get("tool_audit_enabled", True)),
        "path": str(_AUDIT_PATH),
        "event_count": audit_events(limit=100)["count"],
    }
    return {
        "ok": True,
        "settings": settings,
        "capabilities": {
            "science_skills": {
                "enabled": skills_master_enabled,
                "status": "enabled" if skills_master_enabled else "disabled",
                "behavior": (
                    "controls_default_agent_publication_skills_and_filters_"
                    "reusable_protocol_registry"
                ),
            },
            "publication_skills": {
                "enabled": skills_master_enabled,
                "status": "enabled" if skills_master_enabled else "disabled",
                "items": publication_skills,
                "active_skill_ids": [
                    row["id"] for row in publication_skills if row["enabled"]
                ],
            },
            "pubmed_connector": {
                "enabled": settings["connector_pubmed_enabled"],
                "status": (
                    "enabled" if settings["connector_pubmed_enabled"] else "disabled"
                ),
                "behavior": "requires_source_opt_in_and_backend_network_gate",
            },
            "zotero_connector": zotero,
            "mcp_tools": mcp,
            "prompt_contracts": prompt_contracts,
            "tool_audit": audit,
            "remote_compute": remote,
        },
    }
