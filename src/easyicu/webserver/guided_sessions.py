"""Metadata-only guided study drafts for the native Research Copilot.

Guided drafts are not Agent runs. They persist just enough local metadata in a
real local project folder for the left rail to show that a conversation was
started, while preserving the same local-first boundary as the rest of the
native WebApp: no patient rows, no table previews, no external calls, and no
manuscript unlock state.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_CONFIG_DIR = Path.home() / ".easyicu"
_CONFIG_PATH = _CONFIG_DIR / "webserver_guided_drafts.json"
_PROJECTS_ROOT = Path.home() / "easyicu" / "projects"

_VALID_BRANCHES = {"predict", "crossdb", "quality"}
_VALID_DEPTHS = {"extract", "review", "full"}
_VALID_DATA_MODES = {"demo", "real"}
_ROW_LEVEL_KEYS = {"tableRows", "series", "patient", "stay_id", "subject_id", "hadm_id"}
_MAX_DRAFTS = 80
_MAX_SESSIONS = 80
_MAX_MESSAGES = 120
_LOCK = threading.RLock()

_VALID_GOALS = {"idea_mining", "data_extraction", "review_data", "run_agent"}
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
_MAX_SLOT_DEPTH = 5
_MAX_SLOT_ITEMS = 80
_MAX_SLOT_TEXT = 700
_GOAL_META = {
    "idea_mining": {
        "target_route": "ideas",
        "label_en": "Find a Study Idea",
        "label_zh": "找研究想法",
        "summary_en": "Turn a paper, review topic, or hunch into an auditable idea ledger.",
        "summary_zh": "把文章、综述主题或初步想法转成可审计 idea ledger。",
    },
    "data_extraction": {
        "target_route": "extraction",
        "label_en": "Prepare Data",
        "label_zh": "准备/抽取数据",
        "summary_en": "Open the native extraction workspace and prefill the starting intent.",
        "summary_zh": "打开原生抽取工作台，并预填起始意图。",
    },
    "review_data": {
        "target_route": "patient",
        "label_en": "Review Data",
        "label_zh": "审阅已有数据",
        "summary_en": "Start from bounded patient, cohort, or Cross-DB review screens.",
        "summary_zh": "从有界患者、队列或跨库审阅页面开始。",
    },
    "run_agent": {
        "target_route": "agent",
        "label_en": "Run a Research Project",
        "label_zh": "运行研究项目",
        "summary_en": "Create or open an Agent project after the plan is confirmed.",
        "summary_zh": "计划确认后创建或打开 Agent 项目。",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _clean_text(value: Any, fallback: str = "", max_len: int = 220) -> str:
    text = " ".join(str(value or fallback or "").split())
    return text[:max_len]


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else fallback


def _source_meta(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "").strip()
    meta: Dict[str, Any] = {}
    if raw.get("id"):
        meta["id"] = _clean_text(raw.get("id"), max_len=80)
    if raw.get("label"):
        meta["label"] = _clean_text(raw.get("label"), max_len=100)
    if raw.get("database"):
        meta["database"] = _clean_text(raw.get("database"), max_len=40)
    if path:
        meta["path_hash"] = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    return meta or None


def _slot_key(raw: Any) -> str | None:
    key = str(raw or "").strip()
    if not key or key in _ROW_LEVEL_KEYS or len(key) > 48:
        return None
    if not re.match(r"^[A-Za-z0-9_.:-]+$", key):
        return None
    return key


def _bounded_slot_value(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_SLOT_DEPTH:
        return None
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for raw_key, child in list(value.items())[:_MAX_SLOT_ITEMS]:
            key = _slot_key(raw_key)
            if key is None:
                continue
            bounded = _bounded_slot_value(child, depth + 1)
            if bounded is not None:
                out[key] = bounded
        return out
    if isinstance(value, list):
        rows = []
        for child in value[:_MAX_SLOT_ITEMS]:
            bounded = _bounded_slot_value(child, depth + 1)
            if bounded is not None:
                rows.append(bounded)
        return rows
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    return _clean_text(value, max_len=_MAX_SLOT_TEXT)


def _merge_slots(current: Any, patch: Any) -> Dict[str, Any]:
    base = current if isinstance(current, dict) else {}
    bounded = _bounded_slot_value(patch if isinstance(patch, dict) else {})
    update = bounded if isinstance(bounded, dict) else {}

    def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    return merge_dict(base, update)


def _session_id(seed: str) -> str:
    return "guided_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _sanitize_context(raw: Any) -> Dict[str, Any]:
    context = raw if isinstance(raw, dict) else {}
    route = _choice(context.get("route"), _VALID_ROUTES, "guided")
    data_mode = _choice(context.get("data_mode"), _VALID_DATA_MODES, "demo")
    language = _choice(context.get("language"), {"en", "zh"}, "en")
    source = _source_meta(context.get("selected_source"))
    summary = context.get("summary") if isinstance(context.get("summary"), dict) else {}
    bounded_summary: Dict[str, Any] = {}
    for key in ("stays", "modules", "database", "label"):
        if key in summary and not isinstance(summary.get(key), (dict, list)):
            bounded_summary[key] = _clean_text(summary.get(key), max_len=80)
    return {
        "route": route,
        "data_mode": data_mode,
        "language": language,
        "selected_source": source,
        "summary": bounded_summary or None,
    }


def _load_sessions() -> List[Dict[str, Any]]:
    raw = _read_raw()
    rows = raw.get("sessions") if isinstance(raw.get("sessions"), list) else []
    return [row for row in rows if isinstance(row, dict) and row.get("id")]


def _save_sessions(sessions: List[Dict[str, Any]]) -> None:
    raw = _read_raw()
    raw["schema_version"] = 2
    raw["updated_at"] = _now()
    raw["sessions"] = sessions[:_MAX_SESSIONS]
    if not isinstance(raw.get("drafts"), list):
        raw["drafts"] = []
    _write_raw(raw)


def _find_session(session_id: str) -> Dict[str, Any] | None:
    for row in _load_sessions():
        if row.get("id") == session_id:
            return row
    return None


def _upsert_session(session: Dict[str, Any]) -> None:
    sessions = [row for row in _load_sessions() if row.get("id") != session.get("id")]
    sessions.insert(0, session)
    _save_sessions(sessions)


def _safe_project_dir(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        candidate = Path(text).expanduser().resolve()
        root = _PROJECTS_ROOT.expanduser().resolve()
    except OSError:
        return None
    if candidate == root or root in candidate.parents:
        return candidate
    return None


def _find_session_by_project_dir(project_dir: Path) -> Dict[str, Any] | None:
    resolved = str(project_dir)
    for row in _load_sessions():
        if str(row.get("project_dir") or "") == resolved:
            return row
    return None


def _session_file(project_dir: Path) -> Path:
    return project_dir / "guided_copilot_session.json"


def _read_project_session(project_dir: Path) -> Dict[str, Any] | None:
    path = _session_file(project_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("id") else None


def _refresh_privacy(session: Dict[str, Any]) -> None:
    markers = _row_level_markers(session)
    session["privacy"] = {
        "no_patient_rows_persisted": not markers,
        "row_level_markers": markers,
        "scan": "guided_copilot_session_json_keys",
    }


def _persist_session(session: Dict[str, Any]) -> None:
    project_dir = _safe_project_dir(session.get("project_dir")) or _project_dir_for_session(str(session["id"]))
    session["project_dir"] = str(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    _refresh_privacy(session)
    _session_file(project_dir).write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")


def _goal_cards() -> List[Dict[str, str]]:
    return [
        {
            "goal": goal,
            "label_en": meta["label_en"],
            "label_zh": meta["label_zh"],
            "summary_en": meta["summary_en"],
            "summary_zh": meta["summary_zh"],
            "target_route": meta["target_route"],
        }
        for goal, meta in _GOAL_META.items()
    ]


def _public_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = session.get("messages") if isinstance(session.get("messages"), list) else []
    public: List[Dict[str, Any]] = []
    for row in rows[-40:]:
        if not isinstance(row, dict):
            continue
        role = _choice(row.get("role"), {"user", "assistant", "system"}, "assistant")
        item: Dict[str, Any] = {
            "role": role,
            "created_at": row.get("created_at"),
        }
        for key in ("intent", "goal", "action"):
            if row.get(key):
                item[key] = _clean_text(row.get(key), max_len=80)
        if row.get("text"):
            item["text"] = _clean_text(row.get("text"), max_len=500)
        reply = row.get("reply") if isinstance(row.get("reply"), dict) else None
        if reply:
            item["reply"] = {
                "en": _clean_text(reply.get("en"), max_len=700),
                "zh": _clean_text(reply.get("zh"), max_len=700),
            }
        public.append(item)
    return public


def _public_session(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": session.get("id"),
        "kind": "guided_copilot_session",
        "mode": session.get("mode"),
        "status": session.get("status"),
        "step": session.get("step"),
        "goal": session.get("goal"),
        "slots": session.get("slots") or {},
        "handoff": session.get("handoff"),
        "context": session.get("context"),
        "project_dir": session.get("project_dir"),
        "project_kind": session.get("project_kind"),
        "project_title": session.get("project_title"),
        "draft_id": session.get("draft_id"),
        "memory_scope": session.get("memory_scope"),
        "messages": _public_messages(session),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "local_first": session.get("local_first"),
        "privacy": session.get("privacy"),
    }


def _project_dir_for_session(session_id: str) -> Path:
    return _PROJECTS_ROOT / f"guided-copilot-{session_id[-6:]}"


def _prefill_for(goal: str, session: Dict[str, Any]) -> Dict[str, Any]:
    context = session.get("context") if isinstance(session.get("context"), dict) else {}
    slots = session.get("slots") if isinstance(session.get("slots"), dict) else {}
    return {
        "source": "guided_copilot",
        "goal": goal,
        "mode": session.get("mode") or "local",
        "data_mode": context.get("data_mode") or "demo",
        "question_hint": slots.get("question_hint") or "",
        "cohort_hint": slots.get("cohort_hint") or "",
        "module_hint": slots.get("module_hint") or "",
        "route_source": context.get("route") or "guided",
    }


def _handoff_for(goal: str, session: Dict[str, Any]) -> Dict[str, Any]:
    meta = _GOAL_META[goal]
    return {
        "type": "module_handoff",
        "status": "ready",
        "goal": goal,
        "target_route": meta["target_route"],
        "label_en": meta["label_en"],
        "label_zh": meta["label_zh"],
        "prefill": _prefill_for(goal, session),
        "requires_user_confirm": True,
    }


def _infer_goal(text: str) -> str | None:
    value = text.lower()
    if any(token in value for token in ("idea", "paper", "pdf", "article", "frontier", "选题", "想法", "文章", "论文", "综述", "前沿", "挖掘")):
        return "idea_mining"
    if any(
        token in value
        for token in (
            "review data",
            "patient review",
            "cohort review",
            "cross-db",
            "crossdb",
            "compare",
            "patient",
            "visual",
            "visualiz",
            "drill",
            "审阅",
            "查看",
            "可视",
            "患者",
            "跨库",
            "比较",
        )
    ):
        return "review_data"
    if any(token in value for token in ("extract", "export", "data", "module", "feature", "cohort", "抽取", "导出", "数据", "特征", "队列")):
        return "data_extraction"
    if any(token in value for token in ("agent", "analysis", "run", "model", "draft", "manuscript", "项目", "分析", "建模", "草稿", "运行")):
        return "run_agent"
    return None


def _reply_choose_goal() -> Dict[str, Any]:
    return {
        "reply": {
            "en": "Pick a goal card first. Local mode can route and prefill the right workspace, but it does not pretend to understand every free-form research request.",
            "zh": "请先选一个目标卡片。本地模式可以路由并预填正确工作区，但不会假装理解所有自由研究请求。",
        },
        "goal_cards": _goal_cards(),
        "chips": [
            {"label_en": "Find a Study Idea", "label_zh": "找研究想法", "action": "choose_goal", "goal": "idea_mining"},
            {"label_en": "Prepare Data", "label_zh": "准备数据", "action": "choose_goal", "goal": "data_extraction"},
            {"label_en": "Run a Research Project", "label_zh": "运行研究项目", "action": "choose_goal", "goal": "run_agent"},
        ],
    }


def _reply_goal_ready(goal: str, session: Dict[str, Any]) -> Dict[str, Any]:
    meta = _GOAL_META[goal]
    return {
        "reply": {
            "en": f"Good. I will hand this to {meta['label_en']} and let that module own the detailed configuration.",
            "zh": f"好的。我会把它交接到“{meta['label_zh']}”，由那个模块继续负责详细配置。",
        },
        "goal_cards": _goal_cards(),
        "handoff": _handoff_for(goal, session),
        "chips": [
            {"label_en": f"Open {meta['label_en']}", "label_zh": f"打开{meta['label_zh']}", "action": "handoff_to_module", "goal": goal},
            {"label_en": "Choose another goal", "label_zh": "重选目标", "action": "reset_goal"},
        ],
    }


def create_guided_session(body: Dict[str, Any]) -> Dict[str, Any]:
    payload = body if isinstance(body, dict) else {}
    context = _sanitize_context(payload.get("context"))
    mode = _choice(payload.get("mode"), {"local", "ai"}, "local")
    now = _now()
    session_id = _session_id("|".join([now, mode, context["route"]]))
    project_dir = _project_dir_for_session(session_id)
    session = {
        "id": session_id,
        "kind": "guided_copilot_session",
        "mode": mode,
        "status": "active",
        "step": "choose_goal",
        "goal": None,
        "slots": {},
        "handoff": None,
        "context": context,
        "project_dir": str(project_dir),
        "project_kind": "guided_copilot_session_folder",
        "project_title": _clean_text(payload.get("title"), "Guided Copilot session", max_len=90),
        "memory_scope": "guided_frontdoor_session_folder",
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
    }
    _persist_session(session)
    with _LOCK:
        _upsert_session(session)
    return {
        "ok": True,
        "session": _public_session(session),
        "storage": "metadata_only",
        **_reply_choose_goal(),
    }


def open_guided_project(body: Dict[str, Any]) -> Dict[str, Any]:
    payload = body if isinstance(body, dict) else {}
    project_dir = _safe_project_dir(payload.get("project_dir"))
    if project_dir is None:
        return {
            "ok": False,
            "blocked": True,
            "error": "invalid_guided_project_dir",
            "reason": "Guided project memory can only be opened from the local EasyICU projects folder.",
            "storage": "metadata_only",
        }
    if not project_dir.exists():
        return {
            "ok": False,
            "blocked": True,
            "error": "guided_project_folder_not_found",
            "reason": "The selected local project folder does not exist on this machine.",
            "storage": "metadata_only",
        }
    context = _sanitize_context(payload.get("context"))
    mode = _choice(payload.get("mode"), {"local", "ai"}, "local")
    title = _clean_text(payload.get("title"), "Guided project", max_len=90)
    draft_id = _clean_text(payload.get("draft_id"), max_len=80)
    draft_path = project_dir / "guided_draft.json"
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        if isinstance(draft, dict):
            title = _clean_text(draft.get("title"), title, max_len=90)
            draft_id = _clean_text(draft.get("id"), draft_id, max_len=80)
    except (FileNotFoundError, json.JSONDecodeError):
        draft = None

    with _LOCK:
        session = _read_project_session(project_dir) or _find_session_by_project_dir(project_dir)
        now = _now()
        if session is None:
            session = {
                "id": _session_id("project|" + str(project_dir)),
                "kind": "guided_copilot_session",
                "mode": mode,
                "status": "active",
                "step": "choose_goal",
                "goal": None,
                "slots": {},
                "handoff": None,
                "context": context,
                "project_dir": str(project_dir),
                "project_kind": "guided_project_memory",
                "project_title": title,
                "draft_id": draft_id or None,
                "memory_scope": "project_folder",
                "messages": [],
                "created_at": now,
                "updated_at": now,
                "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
            }
        else:
            session["mode"] = _choice(session.get("mode"), {"local", "ai"}, mode)
            session["status"] = session.get("status") or "active"
            session["step"] = session.get("step") or "choose_goal"
            session["context"] = context
            session["project_dir"] = str(project_dir)
            session["project_kind"] = "guided_project_memory"
            session["project_title"] = title
            session["draft_id"] = draft_id or session.get("draft_id")
            session["memory_scope"] = "project_folder"
            session["local_first"] = {"uploads": 0, "tokens": 0, "external_calls": 0}
            session["updated_at"] = now
        _persist_session(session)
        _upsert_session(session)
    return {
        "ok": True,
        "session": _public_session(session),
        "opened": True,
        "storage": "metadata_only",
        "messages_restored": len(_public_messages(session)),
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
        **_reply_choose_goal(),
    }


def post_guided_message(body: Dict[str, Any]) -> Dict[str, Any]:
    payload = body if isinstance(body, dict) else {}
    session_id = str(payload.get("session_id") or "")
    text = _clean_text(payload.get("message"), max_len=500)
    context = _sanitize_context(payload.get("context"))
    with _LOCK:
        session = _find_session(session_id) if session_id else None
        if session is None:
            created = create_guided_session({"mode": payload.get("mode") or "local", "context": context})
            session = _find_session(created["session"]["id"])
        if session is None:
            return {"ok": False, "error": "session_create_failed"}

        now = _now()
        goal = _infer_goal(text)
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        messages.append({"role": "user", "text": text, "created_at": now})
        session["context"] = context
        if goal:
            session["goal"] = goal
            session["step"] = "handoff_ready"
            session["slots"] = {
                **(session.get("slots") if isinstance(session.get("slots"), dict) else {}),
                "question_hint": text,
            }
            session["handoff"] = _handoff_for(goal, session)
            response = _reply_goal_ready(goal, session)
            messages.append({
                "role": "assistant",
                "intent": "goal_detected",
                "goal": goal,
                "reply": response["reply"],
                "created_at": now,
            })
        else:
            session["step"] = "choose_goal"
            session["handoff"] = None
            response = _reply_choose_goal()
            messages.append({
                "role": "assistant",
                "intent": "choose_goal_fallback",
                "reply": response["reply"],
                "created_at": now,
            })
        session["messages"] = messages[-_MAX_MESSAGES:]
        session["updated_at"] = now
        _persist_session(session)
        _upsert_session(session)
    return {
        "ok": True,
        "session": _public_session(session),
        "storage": "metadata_only",
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
        **response,
    }


def execute_guided_action(body: Dict[str, Any]) -> Dict[str, Any]:
    payload = body if isinstance(body, dict) else {}
    action = str(payload.get("action") or "").strip()
    goal = str(payload.get("goal") or "").strip()
    session_id = str(payload.get("session_id") or "")
    with _LOCK:
        session = _find_session(session_id) if session_id else None
        if session is None and action == "update_slots":
            return {
                "ok": False,
                "blocked": True,
                "error": "guided_project_session_required",
                "reason": "Guided Copilot slot updates require an existing project-folder session.",
                "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
            }
        if session is None:
            created = create_guided_session({"mode": payload.get("mode") or "local", "context": payload.get("context")})
            session = _find_session(created["session"]["id"])
        if session is None:
            return {"ok": False, "error": "session_create_failed"}
        if action == "update_slots":
            if session.get("memory_scope") != "project_folder":
                return {
                    "ok": False,
                    "blocked": True,
                    "error": "guided_project_memory_required",
                    "reason": "Required Copilot configuration is only persisted inside a local project folder.",
                    "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
                }
            now = _now()
            if payload.get("context") is not None:
                session["context"] = _sanitize_context(payload.get("context"))
            if goal in _VALID_GOALS:
                session["goal"] = goal
            step = _clean_text(payload.get("step"), max_len=60)
            if step:
                session["step"] = step
            session["slots"] = _merge_slots(session.get("slots"), payload.get("slots"))
            session["updated_at"] = now
            _persist_session(session)
            _upsert_session(session)
            return {
                "ok": True,
                "action": action,
                "session": _public_session(session),
                "storage": "metadata_only",
                "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
            }
        if action == "reset_goal":
            messages = session.get("messages") if isinstance(session.get("messages"), list) else []
            now = _now()
            session["goal"] = None
            session["handoff"] = None
            session["step"] = "choose_goal"
            messages.append({
                "role": "assistant",
                "intent": "reset_goal",
                "reply": _reply_choose_goal()["reply"],
                "created_at": now,
            })
            session["messages"] = messages[-_MAX_MESSAGES:]
            session["updated_at"] = now
            _persist_session(session)
            _upsert_session(session)
            return {"ok": True, "session": _public_session(session), **_reply_choose_goal()}
        if action in {"choose_goal", "handoff_to_module"}:
            if goal not in _VALID_GOALS:
                return {
                    "ok": False,
                    "blocked": True,
                    "error": "unsupported_guided_goal",
                    "reason": "Guided Copilot only routes to whitelisted native modules.",
                    "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
                }
            now = _now()
            if payload.get("context") is not None:
                session["context"] = _sanitize_context(payload.get("context"))
            session["goal"] = goal
            session["step"] = "handoff_ready"
            session["handoff"] = _handoff_for(goal, session)
            session["updated_at"] = now
            response = _reply_goal_ready(goal, session)
            messages = session.get("messages") if isinstance(session.get("messages"), list) else []
            messages.append({
                "role": "user",
                "action": action,
                "goal": goal,
                "text": response["handoff"]["label_en"],
                "created_at": now,
            })
            messages.append({
                "role": "assistant",
                "intent": "goal_detected" if action == "choose_goal" else "module_handoff",
                "goal": goal,
                "reply": response["reply"],
                "created_at": now,
            })
            session["messages"] = messages[-_MAX_MESSAGES:]
            _persist_session(session)
            _upsert_session(session)
            result: Dict[str, Any] = {
                "type": "guided_goal" if action == "choose_goal" else "navigate",
                "goal": goal,
                "handoff": response["handoff"],
            }
            if action == "handoff_to_module":
                result["target"] = response["handoff"]["target_route"]
                result["prefill"] = response["handoff"]["prefill"]
            return {
                "ok": True,
                "action": action,
                "result": result,
                "session": _public_session(session),
                "storage": "metadata_only",
                "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
                **response,
            }
    return {
        "ok": False,
        "blocked": True,
        "error": "unsupported_guided_action",
        "reason": "Guided Copilot only executes whitelisted local routing actions.",
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
    }


def list_guided_sessions(limit: int = 20) -> Dict[str, Any]:
    with _LOCK:
        rows = _load_sessions()
    rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    cap = max(1, min(int(limit or 20), 100))
    return {
        "ok": True,
        "sessions": [_public_session(row) for row in rows[:cap]],
        "count": len(rows),
        "config_path": str(_CONFIG_PATH),
        "storage": "metadata_only",
    }


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


def _draft_id(seed: str) -> str:
    return "draft_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _slug(value: Any, fallback: str = "guided-study") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return (text or fallback)[:64].strip("-._") or fallback


def _project_dir_for(draft_id: str, title: str, requested_slug: Any = None) -> Path:
    slug = _slug(requested_slug or title)
    suffix = str(draft_id or "")[-6:] or hashlib.sha1(title.encode("utf-8")).hexdigest()[:6]
    return _PROJECTS_ROOT / f"guided-{slug}-{suffix}"


def _normalise_draft(body: Dict[str, Any], existing_id: str | None = None) -> Dict[str, Any]:
    created = _now()
    title = _clean_text(body.get("title") or body.get("study_id"), "Untitled guided study", max_len=90)
    branch = _choice(body.get("branch"), _VALID_BRANCHES, "predict")
    depth = _choice(body.get("depth"), _VALID_DEPTHS, "full")
    data_mode = _choice(body.get("data_mode"), _VALID_DATA_MODES, "demo")
    draft_id = existing_id or _draft_id("|".join([created, title, branch, depth]))
    project_dir = _project_dir_for(draft_id, title, body.get("folder_slug"))
    payload: Dict[str, Any] = {
        "id": draft_id,
        "kind": "guided_draft",
        "status": "metadata_only",
        "title": title,
        "branch": branch,
        "depth": depth,
        "data_mode": data_mode,
        "question": _clean_text(body.get("question"), max_len=260),
        "cohort_hint": _clean_text(body.get("cohort_hint"), max_len=120),
        "module_hint": _clean_text(body.get("module_hint"), max_len=120),
        "source": _source_meta(body.get("source")),
        "agent_run_created": False,
        "project_dir": str(project_dir),
        "project_kind": "guided_draft_folder",
        "project_artifact": "guided_draft.json",
        "run_id": None,
        "reportable": False,
        "draft_unlocked": False,
        "local_first": {"uploads": 0, "tokens": 0, "external_calls": 0},
        "created_at": created,
        "updated_at": created,
    }
    markers = _row_level_markers(payload)
    payload["privacy"] = {
        "no_patient_rows_persisted": not markers,
        "row_level_markers": markers,
        "scan": "draft_metadata_json_keys",
    }
    return payload


def list_guided_drafts(limit: int = 20) -> Dict[str, Any]:
    raw = _read_raw()
    rows = raw.get("drafts") if isinstance(raw.get("drafts"), list) else []
    drafts = [row for row in rows if isinstance(row, dict) and row.get("id")]
    drafts.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    cap = max(1, min(int(limit or 20), 100))
    return {
        "ok": True,
        "drafts": drafts[:cap],
        "count": len(drafts),
        "config_path": str(_CONFIG_PATH),
        "storage": "metadata_only",
    }


def create_guided_draft(body: Dict[str, Any]) -> Dict[str, Any]:
    draft = _normalise_draft(body if isinstance(body, dict) else {})
    project_dir = Path(str(draft["project_dir"]))
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "guided_draft.json").write_text(
        json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raw = _read_raw()
    current = raw.get("drafts") if isinstance(raw.get("drafts"), list) else []
    drafts = [row for row in current if isinstance(row, dict) and row.get("id") != draft["id"]]
    drafts.insert(0, draft)
    drafts = drafts[:_MAX_DRAFTS]
    raw["schema_version"] = max(2, int(raw.get("schema_version") or 1))
    raw["updated_at"] = _now()
    raw["drafts"] = drafts
    if not isinstance(raw.get("sessions"), list):
        raw["sessions"] = []
    _write_raw(raw)
    return {
        "ok": True,
        "draft": draft,
        "storage": "metadata_only",
        "storage_detail": "local_project_folder_indexed_by_registry",
        "persisted": True,
    }
