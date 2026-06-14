"""Research Copilot — study-state lifecycle + session manifest persistence.

Extracted from `llm_chat.py` (Phase-6 split, 4th batch). Covers the
default/ensure/reset study-state factory trio plus the on-disk *study session*
layer: manifest read/write, session listing, title normalisation, message
sanitisation, and legacy-study migration. All functions operate on `state` /
`study` mappings passed in as arguments and on the filesystem — **none reads the
Streamlit `st` handle** — so they are unaffected by the test suite's
`monkeypatch.setattr(llm_chat, "st", ...)` contract and move cleanly.

The few `COPILOT_*` constants and the `_repo_root` helper these need stay in
`llm_chat.py` and are imported lazily inside the using functions (routing.py
pattern), so this module never imports `llm_chat` at load time — no cycle.
`llm_chat.py` re-imports every name below, so all call sites keep working.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from pathlib import Path

from easyicu.webapp import copilot_engine as _copilot_engine


def _default_copilot_study_state(state: MutableMapping[str, object] | None = None) -> dict[str, object]:
    from easyicu.webapp.llm_chat import COPILOT_DEFAULT_MODULES  # lazy: avoid import cycle
    state = state or {}
    patient_n = int(state.get("demo_mode_patients") or 10)
    return {
        "branch": None,
        "step": "question",
        "depth": _copilot_engine.DEFAULT_DEPTH,
        "data_mode": "real",
        "patient_n": patient_n,
        "db_count": 6,
        "outcome": "a prespecified ICU outcome",
        "window": "first 24h",
        "exposure": "",
        "modules": COPILOT_DEFAULT_MODULES[:],
        "question": "",
        "cohort_phase": "ready",
        "cohort_filters": [],
        "cohort_configured": False,
        "concepts_configured": False,
        "draft_signed": False,
        "last_update": datetime.now().isoformat(timespec="seconds"),
    }


def _copilot_is_legacy_default_question(question: str) -> bool:
    text = " ".join((question or "").split()).strip().lower()
    if not text:
        return False
    legacy_exact = {
        "among sepsis-3 patients, do first-24h bedside features predict in-hospital mortality, and does adding lactate improve the model?",
    }
    return text in legacy_exact


def _copilot_normalize_legacy_study(study: MutableMapping[str, object]) -> None:
    """Clean old default examples from persisted chat state after UI copy changes."""
    if str(study.get("branch") or "predict") != "predict":
        return
    if _copilot_is_legacy_default_question(str(study.get("question") or "")):
        study["question"] = ""
        if str(study.get("outcome") or "").strip().lower() in {"in-hospital mortality", "院内死亡"}:
            study["outcome"] = "a prespecified ICU outcome"
        if str(study.get("exposure") or "").strip().lower() in {"lactate", "乳酸"}:
            study["exposure"] = ""


def _ensure_copilot_study_state(state: MutableMapping[str, object]) -> dict[str, object]:
    from easyicu.webapp.llm_chat import COPILOT_BRANCH_CONFIG  # lazy: avoid import cycle
    study = state.get("_copilot_guided_study")
    if not isinstance(study, dict):
        study = _default_copilot_study_state(state)
        state["_copilot_guided_study"] = study
    for key, value in _default_copilot_study_state(state).items():
        study.setdefault(key, value)
    branch_hint = str(state.get("_copilot_entry_branch_hint") or "").strip()
    if branch_hint in COPILOT_BRANCH_CONFIG and not str(study.get("branch") or "").strip():
        study["branch"] = branch_hint
    _copilot_normalize_legacy_study(study)
    return study


def _remember_copilot_guided_study_resume(
    state: MutableMapping[str, object],
    study: MutableMapping[str, object],
) -> dict[str, object]:
    """Persist the signed guided study for the entry-page resume card."""
    from easyicu.webapp.llm_chat import COPILOT_BRANCH_CONFIG, COPILOT_DEFAULT_MODULES  # lazy
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    modules = [str(item) for item in list(study.get("modules") or COPILOT_DEFAULT_MODULES)]
    selected_concepts = [
        str(item)
        for item in list(study.get("selected_concepts") or [])
        if str(item).strip()
    ] or [
        str(item)
        for item in list(config.get("selected_concepts") or [])
        if str(item).strip()
    ]
    try:
        patient_n = int(study.get("patient_n") or 10)
    except (TypeError, ValueError):
        patient_n = 10
    record: dict[str, object] = {
        "branch": branch,
        "data_mode": str(study.get("data_mode") or "real"),
        "patient_n": patient_n,
        "modules": modules,
        "selected_concepts": selected_concepts,
        "question": str(study.get("question") or config.get("question_en") or config.get("chip") or branch),
        "step": str(study.get("step") or "draft"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    state["_eu_last_study_resume"] = record
    state["easyicu_study"] = record
    return record


def _reset_copilot_study_state(state: MutableMapping[str, object]) -> dict[str, object]:
    study = _default_copilot_study_state(state)
    state["_copilot_guided_study"] = study
    state.pop("_copilot_data_source_choice", None)
    state.pop("_copilot_data_source_notice", None)
    return study


def _copilot_study_sessions_root(state: Mapping[str, object] | None = None) -> Path:
    """Return the local directory that stores Copilot study session manifests."""
    from easyicu.webapp.llm_chat import _repo_root  # lazy: avoid import cycle
    raw_root = str((state or {}).get("copilot_study_root") or "").strip()
    if raw_root:
        return Path(raw_root).expanduser()
    return _repo_root() / "research_output" / "copilot_studies"


def _copilot_study_manifest_path(workdir: Path) -> Path:
    return workdir / "copilot_session.json"


def _copilot_session_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _copilot_session_fallback_title(lang: str) -> str:
    return "Untitled study" if lang == "en" else "未命名研究"


def _copilot_session_title_from_state(
    state: Mapping[str, object],
    study: Mapping[str, object],
    lang: str,
) -> str:
    question = str(study.get("question") or "").strip()
    if question:
        return question[:84]
    messages = state.get("llm_messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping):
                continue
            if str(message.get("role") or "").lower() != "user":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                return content[:84]
    title = str(state.get("_copilot_current_session_title") or "").strip()
    return title or _copilot_session_fallback_title(lang)


def _copilot_jsonable(value: object) -> object:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _copilot_sanitized_messages(messages: object) -> list[dict[str, object]]:
    from easyicu.webapp.llm_chat import COPILOT_SESSION_MESSAGE_SAVE_LIMIT  # lazy
    if not isinstance(messages, list):
        return []
    sanitized: list[dict[str, object]] = []
    for message in messages[-COPILOT_SESSION_MESSAGE_SAVE_LIMIT:]:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        item: dict[str, object] = {
            "role": role,
            "content": str(message.get("content") or ""),
        }
        for key in ("actions", "workflow_snapshot"):
            if key in message:
                item[key] = _copilot_jsonable(message.get(key))
        sanitized.append(item)
    return sanitized


def _invalidate_copilot_session_cache(state: MutableMapping[str, object]) -> None:
    state.pop("_copilot_sessions_cache", None)


def _write_copilot_study_session_manifest(
    state: MutableMapping[str, object],
    lang: str,
    *,
    created_at: str | None = None,
) -> dict[str, object] | None:
    raw_workdir = str(state.get("_copilot_current_session_dir") or "").strip()
    session_id = str(state.get("_copilot_current_session_id") or "").strip()
    if not raw_workdir or not session_id:
        return None
    workdir = Path(raw_workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    agent_runs_dir = workdir / "agent_runs"
    agent_runs_dir.mkdir(parents=True, exist_ok=True)
    study = dict(_ensure_copilot_study_state(state))
    existing_created = created_at
    manifest_path = _copilot_study_manifest_path(workdir)
    if existing_created is None and manifest_path.exists():
        existing = _read_copilot_study_session_manifest(manifest_path)
        if isinstance(existing, dict):
            existing_created = str(existing.get("created_at") or "") or None
    now = _copilot_session_now()
    title = _copilot_session_title_from_state(state, study, lang)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id": session_id,
        "title": title,
        "status": "active",
        "created_at": existing_created or now,
        "updated_at": now,
        "workdir": str(workdir.resolve()),
        "agent_runs_dir": str(agent_runs_dir.resolve()),
        "study": _copilot_jsonable(study),
        "messages": _copilot_sanitized_messages(state.get("llm_messages")),
    }
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)
    state["_copilot_current_session_title"] = title
    state["research_agent_workdir"] = str(agent_runs_dir.resolve())
    _invalidate_copilot_session_cache(state)
    return manifest


def _read_copilot_study_session_manifest(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not str(data.get("id") or "").strip():
        return None
    return data


def _copilot_normalize_session_title(manifest: Mapping[str, object], lang: str) -> str:
    """Session-list title: prefer the question, skip generic chip labels.

    A session created via the "New study / workspace" chip stored that label as
    its first user message, leaking into the saved manifest title. Normalize any
    empty/generic title down to the real question (or "Untitled study").
    """
    from easyicu.webapp.llm_chat import _COPILOT_GENERIC_SESSION_TITLES  # lazy
    raw = str(manifest.get("title") or "").strip()
    if raw and raw.lower() not in _COPILOT_GENERIC_SESSION_TITLES:
        return raw[:84]
    study = manifest.get("study") if isinstance(manifest.get("study"), Mapping) else {}
    cand = str(study.get("question") or "").strip()
    if not cand:
        messages = manifest.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                if str(message.get("role") or "").lower() != "user":
                    continue
                content = str(message.get("content") or "").strip()
                if content and content.lower() not in _COPILOT_GENERIC_SESSION_TITLES:
                    cand = content
                    break
    return cand[:84] if cand else _copilot_session_fallback_title(lang)


def _copilot_list_study_sessions(state: Mapping[str, object]) -> list[dict[str, object]]:
    root = _copilot_study_sessions_root(state)
    cached = state.get("_copilot_sessions_cache")
    if isinstance(cached, Mapping) and str(cached.get("root") or "") == str(root):
        sessions_cached = cached.get("sessions")
        if isinstance(sessions_cached, list):
            return [
                dict(session)
                for session in sessions_cached
                if isinstance(session, Mapping)
            ]
    if not root.exists():
        return []
    lang = str(state.get("language") or "en")
    sessions: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("*/copilot_session.json")):
        manifest = _read_copilot_study_session_manifest(manifest_path)
        if not manifest:
            continue
        workdir = Path(str(manifest.get("workdir") or manifest_path.parent)).expanduser()
        sessions.append({
            "id": str(manifest.get("id") or manifest_path.parent.name),
            "title": _copilot_normalize_session_title(manifest, lang),
            "updated_at": str(manifest.get("updated_at") or manifest.get("created_at") or ""),
            "created_at": str(manifest.get("created_at") or ""),
            "workdir": str(workdir),
            "agent_runs_dir": str(manifest.get("agent_runs_dir") or (workdir / "agent_runs")),
            "study": manifest.get("study") if isinstance(manifest.get("study"), dict) else {},
            "messages": manifest.get("messages") if isinstance(manifest.get("messages"), list) else [],
        })
    sessions.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    if isinstance(state, MutableMapping):
        state["_copilot_sessions_cache"] = {
            "root": str(root),
            "sessions": [dict(session) for session in sessions],
        }
    return sessions


def _start_new_copilot_study_session(
    state: MutableMapping[str, object],
    lang: str,
    *,
    carry_messages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    root = _copilot_study_sessions_root(state)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = 0
    while True:
        session_id = f"study_{stamp}" if suffix == 0 else f"study_{stamp}_{suffix}"
        workdir = root / session_id
        if not workdir.exists():
            break
        suffix += 1
    (workdir / "agent_runs").mkdir(parents=True, exist_ok=True)
    _reset_copilot_study_state(state)
    state["llm_messages"] = list(carry_messages or [])
    state["_copilot_current_session_id"] = session_id
    state["_copilot_current_session_dir"] = str(workdir.resolve())
    state["_copilot_current_session_title"] = _copilot_session_fallback_title(lang)
    state["research_agent_workdir"] = str((workdir / "agent_runs").resolve())
    state.pop("_ai_pending_question", None)
    state.pop("_copilot_data_source_form", None)
    _write_copilot_study_session_manifest(state, lang, created_at=_copilot_session_now())
    return {
        "id": session_id,
        "workdir": str(workdir.resolve()),
        "agent_runs_dir": str((workdir / "agent_runs").resolve()),
    }


def _open_copilot_study_session(
    state: MutableMapping[str, object],
    session_id: str,
    lang: str,
) -> bool:
    for session in _copilot_list_study_sessions(state):
        if str(session.get("id") or "") != session_id:
            continue
        study = session.get("study")
        state["_copilot_guided_study"] = dict(study) if isinstance(study, Mapping) else _default_copilot_study_state(state)
        state["llm_messages"] = [
            dict(message)
            for message in session.get("messages", [])
            if isinstance(message, Mapping)
        ]
        state["_copilot_current_session_id"] = str(session.get("id") or session_id)
        state["_copilot_current_session_dir"] = str(session.get("workdir") or "")
        state["_copilot_current_session_title"] = str(session.get("title") or _copilot_session_fallback_title(lang))
        agent_runs_dir = str(session.get("agent_runs_dir") or "").strip()
        if agent_runs_dir:
            state["research_agent_workdir"] = agent_runs_dir
        state.pop("_ai_pending_question", None)
        _invalidate_copilot_session_cache(state)
        return True
    return False


def _touch_current_copilot_study_session(state: MutableMapping[str, object], lang: str) -> None:
    if str(state.get("_copilot_current_session_id") or "").strip():
        _write_copilot_study_session_manifest(state, lang)
