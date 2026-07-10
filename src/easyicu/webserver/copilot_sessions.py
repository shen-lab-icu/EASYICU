"""Local-first Page guide / Copilot sessions for the native WebApp.

This module keeps routing deterministic while assigning collision-safe session
IDs. It gives the floating Page guide and future Guided Copilot shells a real
backend contract without constructing an external model client, reading patient
rows, or creating manuscript artifacts.
The backend owns session metadata, shortcut classification, allowed UI actions,
and fail-closed blockers.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_CONFIG_DIR = Path.home() / ".easyicu"
_CONFIG_PATH = _CONFIG_DIR / "webserver_copilot_sessions.json"
_PROJECTS_ROOT = Path.home() / "easyicu" / "projects"
_MAX_SESSIONS = 80
_MAX_MESSAGES = 120
_LOCK = threading.RLock()

_VALID_ROUTES = {
    "entry",
    "ideas",
    "extraction",
    "patient",
    "cohort",
    "crossdb",
    "agent",
    "settings",
    "dictionary",
    "states",
    "tutorial",
    "guided",
}
_VALID_SCOPES = {"page_guide", "quick_help", "guided"}
_ROW_LEVEL_KEYS = {"tableRows", "series", "patient", "stay_id", "subject_id", "hadm_id"}


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _clean_text(value: Any, fallback: str = "", max_len: int = 500) -> str:
    text = " ".join(str(value or fallback or "").split())
    return text[:max_len]


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else fallback


def _scope(value: Any) -> str:
    scope = _choice(value, _VALID_SCOPES, "page_guide")
    return "page_guide" if scope == "quick_help" else scope


def _slug(value: Any, fallback: str = "copilot-session") -> str:
    text = str(value or fallback or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return (text or fallback)[:64].strip("-._") or fallback


def _session_id(seed: str) -> str:
    return "copilot_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


def _read_raw() -> Dict[str, Any]:
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_raw(data: Dict[str, Any]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CONFIG_PATH.with_suffix(_CONFIG_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_CONFIG_PATH)


def _row_level_markers(value: Any, markers: List[str] | None = None) -> List[str]:
    found = markers if markers is not None else []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in _ROW_LEVEL_KEYS:
                found.append(str(key))
            _row_level_markers(child, found)
    elif isinstance(value, list):
        for child in value:
            _row_level_markers(child, found)
    return found


def _project_dir_for(session_id: str, route: str, scope: str) -> Path:
    prefix = "page-guide" if scope == "page_guide" else "guided-copilot"
    return _PROJECTS_ROOT / f"{prefix}-{_slug(route)}-{session_id[-12:]}"


def _sanitize_context(raw: Any) -> Dict[str, Any]:
    context = raw if isinstance(raw, dict) else {}
    route = _choice(context.get("route"), _VALID_ROUTES, "entry")
    data_mode = _choice(context.get("data_mode"), {"demo", "real"}, "demo")
    language = _choice(context.get("language"), {"en", "zh"}, "en")
    selected_source = (
        context.get("selected_source")
        if isinstance(context.get("selected_source"), dict)
        else {}
    )
    source_meta: Dict[str, Any] = {}
    if selected_source.get("label"):
        source_meta["label"] = _clean_text(selected_source.get("label"), max_len=100)
    if selected_source.get("database"):
        source_meta["database"] = _clean_text(
            selected_source.get("database"), max_len=40
        )
    path = str(selected_source.get("path") or "").strip()
    if path:
        source_meta["path_hash"] = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    return {
        "route": route,
        "data_mode": data_mode,
        "language": language,
        "selected_source": source_meta or None,
    }


def _load_sessions() -> List[Dict[str, Any]]:
    raw = _read_raw()
    sessions = raw.get("sessions") if isinstance(raw.get("sessions"), list) else []
    return [row for row in sessions if isinstance(row, dict) and row.get("id")]


def _save_sessions(sessions: List[Dict[str, Any]]) -> None:
    _write_raw(
        {
            "schema_version": 1,
            "updated_at": _now(),
            "sessions": sessions[:_MAX_SESSIONS],
        }
    )


def _find_session(session_id: str) -> Dict[str, Any] | None:
    for row in _load_sessions():
        if row.get("id") == session_id:
            return row
    return None


def _upsert_session(session: Dict[str, Any]) -> None:
    sessions = [row for row in _load_sessions() if row.get("id") != session.get("id")]
    sessions.insert(0, session)
    _save_sessions(sessions)


def _route_intro(route: str) -> Dict[str, str]:
    intros: Dict[str, Dict[str, str]] = {
        "extraction": {
            "en": "This page guide can explain extraction, open Guided Copilot, or take you to data workspace controls. It will not start a job without a user action.",
            "zh": "这个页面指南可以解释抽取、打开 Guided Copilot，或带你到数据工作台控件。没有你的动作，它不会自动启动任务。",
        },
        "patient": {
            "en": "Patient Review is for bounded drilldown and visual checks. This guide can open the relevant workspace or explain the tabs.",
            "zh": "患者审阅用于有界明细和可视化检查。这个指南可以打开相关工作区，或解释各个标签页。",
        },
        "crossdb": {
            "en": "Cross-DB Compare needs compatible registered sources. This guide can open the comparison page or explain why a comparison is blocked.",
            "zh": "跨库比较需要兼容的已注册数据源。这个指南可以打开比较页，或解释为什么比较被阻断。",
        },
        "agent": {
            "en": "Agent Projects can run an auditable local analysis. Drafts stay locked until evidence and human checks pass.",
            "zh": "研究项目可以运行可审计的本地分析。证据和人工检查通过前，草稿保持锁定。",
        },
        "settings": {
            "en": "Settings are local-first. Only controls backed by native runtime logic are editable.",
            "zh": "设置是本地优先的。只有已由原生运行逻辑承接的控件可以编辑。",
        },
    }
    return intros.get(
        route,
        {
            "en": "This page guide can explain the current screen, navigate the workspace, or open Guided Copilot. Actions are local and bounded.",
            "zh": "这个页面指南可以解释当前页面、导航工作区，或打开 Guided Copilot。所有动作都是本地且有边界的。",
        },
    )


def _chips_for_route(route: str) -> List[Dict[str, str]]:
    common = [
        {
            "label_en": "Start guided study",
            "label_zh": "开始研究引导",
            "action": "open_guided",
        },
        {
            "label_en": "Privacy boundary",
            "label_zh": "隐私边界",
            "action": "explain_privacy",
        },
    ]
    route_chips: Dict[str, List[Dict[str, str]]] = {
        "extraction": [
            {
                "label_en": "Explain extraction",
                "label_zh": "解释抽取",
                "action": "explain_extraction",
            },
            {
                "label_en": "Open Data Extraction",
                "label_zh": "打开数据抽取",
                "action": "open_extraction",
            },
        ],
        "patient": [
            {
                "label_en": "Open Patient Review",
                "label_zh": "打开患者审阅",
                "action": "open_patient",
            },
            {
                "label_en": "Explain review tabs",
                "label_zh": "解释审阅标签",
                "action": "explain_patient",
            },
        ],
        "crossdb": [
            {
                "label_en": "Open Cross-DB Compare",
                "label_zh": "打开跨库比较",
                "action": "open_crossdb",
            },
            {
                "label_en": "Explain compatibility",
                "label_zh": "解释兼容性",
                "action": "explain_crossdb",
            },
        ],
        "agent": [
            {
                "label_en": "Open Agent Projects",
                "label_zh": "打开研究项目",
                "action": "open_agent",
            },
            {
                "label_en": "Why draft is locked",
                "label_zh": "为什么草稿锁定",
                "action": "explain_gate",
            },
        ],
        "settings": [
            {
                "label_en": "Open Settings",
                "label_zh": "打开设置",
                "action": "open_settings",
            },
            {
                "label_en": "Explain local settings",
                "label_zh": "解释本地设置",
                "action": "explain_settings",
            },
        ],
    }
    return route_chips.get(route, common[:1]) + common


def _reply_for_intent(intent: str, route: str) -> Dict[str, Any]:
    answers: Dict[str, Dict[str, str]] = {
        "privacy": {
            "en": "EasyICU is local-first. Patient rows are not uploaded. This Page guide backend stores metadata-only session state and returns bounded UI actions.",
            "zh": "EasyICU 是本地优先。患者行不会上传。这个 Page guide 后端只保存元数据会话状态，并返回有边界的 UI 动作。",
        },
        "gate": {
            "en": "The manuscript draft remains locked until evidence binding, numeric checks, privacy checks, and human signoff pass.",
            "zh": "稿件草稿会保持锁定，直到 evidence binding、数值审计、隐私检查和人工确认都通过。",
        },
        "extraction": {
            "en": "Extraction turns a registered ICU data folder or export into analysis-ready tables plus a manifest. Unsupported filters fail closed.",
            "zh": "抽取会把已注册 ICU 数据文件夹或 export 转成可分析数据表和 manifest。不支持的筛选会 fail-closed。",
        },
        "patient": {
            "en": "Patient Review should use bounded backend aggregates and pseudonymous entity refs, not fake seeded panels in Real mode.",
            "zh": "患者审阅应使用有界后端聚合和伪匿名 entity ref，真实模式下不能显示假的 seeded 面板。",
        },
        "crossdb": {
            "en": "Cross-DB Compare only runs when registered sources share compatible core modules. Otherwise the UI should show a blocker, not a synthetic result.",
            "zh": "跨库比较只在已注册 source 共享兼容核心模块时运行。否则 UI 应显示阻断原因，而不是合成结果。",
        },
        "settings": {
            "en": "Only settings consumed by the native backend/runtime are editable. Roadmap or per-workflow items are locked as status rows.",
            "zh": "只有原生后端/运行时真正消费的设置可编辑。路线图项或工作流专属项会锁定为状态行。",
        },
        "how": {
            "en": "Use Idea Mining to form a feasible question, Data Workspace to extract/review, then Agent Projects to run an evidence-gated analysis.",
            "zh": "先用 Idea Mining 形成可行问题，再用数据工作台抽取/审阅，最后用研究项目运行受证据核验约束的分析。",
        },
    }
    if intent == "route":
        return {"reply": _route_intro(route), "chips": _chips_for_route(route)}
    return {
        "reply": answers.get(intent, answers["how"]),
        "chips": _chips_for_route(route),
    }


def _classify(text: str, route: str) -> tuple[str, List[Dict[str, Any]]]:
    value = text.lower()
    actions: List[Dict[str, Any]] = []
    if any(
        token in value
        for token in ("guided", "引导", "全流程", "带我", "run it", "whole")
    ):
        actions.append(
            {"type": "navigate", "target": "guided", "requires_user_confirm": False}
        )
        return "route", actions
    if any(
        token in value
        for token in ("privacy", "upload", "local", "phi", "隐私", "上传", "本地")
    ):
        return "privacy", actions
    if any(
        token in value
        for token in ("gate", "lock", "draft", "sign", "证据", "草稿", "锁定")
    ):
        return "gate", actions
    if any(token in value for token in ("extract", "export", "抽取", "导出")):
        actions.append(
            {"type": "navigate", "target": "extraction", "requires_user_confirm": False}
        )
        return "extraction", actions
    if any(token in value for token in ("patient", "患者", "drill")):
        actions.append(
            {"type": "navigate", "target": "patient", "requires_user_confirm": False}
        )
        return "patient", actions
    if any(token in value for token in ("cross", "database", "跨库", "数据库")):
        actions.append(
            {"type": "navigate", "target": "crossdb", "requires_user_confirm": False}
        )
        return "crossdb", actions
    if any(token in value for token in ("agent", "run", "analysis", "项目", "分析")):
        actions.append(
            {"type": "navigate", "target": "agent", "requires_user_confirm": False}
        )
        return "gate", actions
    if any(token in value for token in ("setting", "设置")):
        actions.append(
            {"type": "navigate", "target": "settings", "requires_user_confirm": False}
        )
        return "settings", actions
    if route in {"extraction", "patient", "crossdb", "agent", "settings"}:
        return route if route != "agent" else "gate", actions
    return "how", actions


def create_session(body: Dict[str, Any]) -> Dict[str, Any]:
    context = _sanitize_context(body.get("context"))
    scope = _scope(body.get("scope"))
    now = _now()
    session_id = _session_id(
        "|".join([now, scope, context["route"], secrets.token_hex(16)])
    )
    project_dir = _project_dir_for(session_id, context["route"], scope)
    project_kind = (
        "page_guide_session_folder"
        if scope == "page_guide"
        else "guided_copilot_session_folder"
    )
    session = {
        "id": session_id,
        "scope": scope,
        "status": "active",
        "context": context,
        "project_dir": str(project_dir),
        "project_kind": project_kind,
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
    }
    markers = _row_level_markers(session)
    session["privacy"] = {
        "no_patient_rows_persisted": not markers,
        "row_level_markers": markers,
        "scan": (
            "page_guide_session_json_keys"
            if scope == "page_guide"
            else "copilot_session_json_keys"
        ),
    }
    project_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = (
        "page_guide_session.json" if scope == "page_guide" else "copilot_session.json"
    )
    (project_dir / artifact_name).write_text(
        json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with _LOCK:
        _upsert_session(session)
    intro = _reply_for_intent("route", context["route"])
    return {
        "ok": True,
        "session": _public_session(session),
        "reply": intro["reply"],
        "chips": intro["chips"],
        "actions": [],
        "storage": "metadata_only",
    }


def _public_session(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": session.get("id"),
        "scope": session.get("scope"),
        "status": session.get("status"),
        "context": session.get("context"),
        "project_dir": session.get("project_dir"),
        "project_kind": session.get("project_kind"),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "local_first": session.get("local_first"),
        "privacy": session.get("privacy"),
    }


def post_message(body: Dict[str, Any]) -> Dict[str, Any]:
    session_id = str(body.get("session_id") or "")
    text = _clean_text(body.get("message"), max_len=500)
    context = _sanitize_context(body.get("context"))
    with _LOCK:
        session = _find_session(session_id) if session_id else None
        if session is None:
            created = create_session(
                {"scope": body.get("scope") or "page_guide", "context": context}
            )
            session = _find_session(created["session"]["id"])
        if session is None:
            return {"ok": False, "error": "session_create_failed"}

        route = (
            context.get("route")
            or (session.get("context") or {}).get("route")
            or "entry"
        )
        intent, actions = _classify(text, route)
        response = _reply_for_intent(intent, route)
        now = _now()
        messages = (
            session.get("messages") if isinstance(session.get("messages"), list) else []
        )
        messages.extend(
            [
                {"role": "user", "text": text, "created_at": now},
                {
                    "role": "assistant",
                    "intent": intent,
                    "reply": response["reply"],
                    "actions": actions,
                    "created_at": now,
                },
            ]
        )
        session["messages"] = messages[-_MAX_MESSAGES:]
        session["context"] = context
        session["updated_at"] = now
        markers = _row_level_markers(session)
        session["privacy"] = {
            "no_patient_rows_persisted": not markers,
            "row_level_markers": markers,
            "scan": (
                "page_guide_session_json_keys"
                if session.get("scope") == "page_guide"
                else "copilot_session_json_keys"
            ),
        }
        Path(str(session["project_dir"])).mkdir(parents=True, exist_ok=True)
        artifact_name = (
            "page_guide_session.json"
            if session.get("scope") == "page_guide"
            else "copilot_session.json"
        )
        (Path(str(session["project_dir"])) / artifact_name).write_text(
            json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _upsert_session(session)
    return {
        "ok": True,
        "session": _public_session(session),
        "reply": response["reply"],
        "chips": response["chips"],
        "actions": actions,
        "storage": "metadata_only",
    }


def execute_action(body: Dict[str, Any]) -> Dict[str, Any]:
    action = str(body.get("action") or body.get("type") or "").strip()
    context = _sanitize_context(body.get("context"))
    allowed = {
        "open_guided": {"type": "navigate", "target": "guided"},
        "open_extraction": {"type": "navigate", "target": "extraction"},
        "open_patient": {"type": "navigate", "target": "patient"},
        "open_cohort": {"type": "navigate", "target": "cohort"},
        "open_crossdb": {"type": "navigate", "target": "crossdb"},
        "open_agent": {"type": "navigate", "target": "agent"},
        "open_settings": {"type": "navigate", "target": "settings"},
    }
    if action in allowed:
        return {
            "ok": True,
            "action": action,
            "result": allowed[action],
            "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
        }
    explain = {
        "explain_privacy": "privacy",
        "explain_gate": "gate",
        "explain_extraction": "extraction",
        "explain_patient": "patient",
        "explain_crossdb": "crossdb",
        "explain_settings": "settings",
    }.get(action)
    if explain:
        response = _reply_for_intent(explain, context["route"])
        return {
            "ok": True,
            "action": action,
            "result": {
                "type": "reply",
                "reply": response["reply"],
                "chips": response["chips"],
            },
            "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
        }
    return {
        "ok": False,
        "blocked": True,
        "error": "unsupported_page_guide_action",
        "reason": "This Page guide backend only executes whitelisted local UI actions.",
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
    }


def list_sessions(limit: int = 20) -> Dict[str, Any]:
    with _LOCK:
        rows = _load_sessions()
    rows.sort(
        key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""),
        reverse=True,
    )
    cap = max(1, min(int(limit or 20), 100))
    return {
        "ok": True,
        "sessions": [_public_session(row) for row in rows[:cap]],
        "count": len(rows),
        "config_path": str(_CONFIG_PATH),
        "storage": "metadata_only",
    }
