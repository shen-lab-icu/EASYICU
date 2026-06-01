"""Shell-A redesign · Research Agent live workbench.

Faithful implementation of ``page-agent-workbench.jsx``: a three-column
live run view —

* Left   — step sequence (status + retry branch for each pipeline step)
* Center — step review panel (review summary + output/code/issues/history tabs)
* Right  — result gallery (mini charts) + evidence list
* Bottom — timeline scrubber across all steps

Data binding
------------
The workbench reads its state from ``st.session_state['_agent_workbench']``
only after a live run, imported manifest, or explicit local history selection
binds a real run. Use :func:`build_workbench_state_from_manifest` to bind a
real research-agent ``manifest.json`` / ``manifest_partial.json`` into that
state.

Expected shape of ``_agent_workbench`` (all keys optional)::

    {
      "title": str, "subtitle": str, "status": "running"|"done"|...,
      "steps": [{"label","sub","status","step_id"}],
      "code": str,                          # current step source
      "code_path": str,
      "trace": [{"t","msg","level"}],       # level in ok/info/warn/err
      "autopatch": {"from","to","ago"} | None,
      "results": [{"kind","title","metric","kind_svg"}],
      "evidence": [{"label","sub","tag"}],  # tag in data/paper/code/test/fix
      "step_details": [{"code","trace","results","evidence","step_contract"}],
      "audit_tasks": [{"title","detail","tone","action"}],
      "review_decisions": [{"label","detail","state"}],
      "timeline": [{"label","t","d","status"}],
      "elapsed": float, "total": float, "tokens": int,
    }

This module renders the visual surface and the manifest-to-workbench
adapter. Starting the pipeline itself still lives in ``research_agent.py``.
"""

from __future__ import annotations

import html
import json
import re
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from easyicu.webapp import cohort_charts as cc


def _T(lang: str, en: str, zh: str) -> str:
    return en if lang == "en" else zh


def _esc(v: object) -> str:
    return html.escape(str(v))


_STATUS_COLOR = {
    "ok": "var(--ink)",
    "fail": "var(--bad)",
    "retry": "var(--warn)",
    "running": "var(--accent)",
    "pending": "var(--ink-4)",
}

_TERMINAL_OK = {"ok", "complete", "completed", "passed", "success"}
_TERMINAL_FAIL = {
    "blocked",
    "blocked_by_concept_audit",
    "coder_failed",
    "error",
    "execution_failed",
    "fail",
    "failed",
    "repair_failed",
}

_ACKED_FINDINGS_KEY = "_eu_wb_findings_acked"
_ACKED_FINDINGS_RUN_KEY = "_eu_wb_findings_acked_run_dir"
_FINDING_REVIEW_STATE_FILE = "finding_review_state.json"
_REVIEW_DETAILS_EXPANDED_KEY = "_eu_wb_review_details_expanded"
_APPROVED_REVIEW_DECISIONS = {"approved", "accept", "accepted", "signed_off", "ready"}


def _file_fingerprint(path: Path) -> tuple[str, int, int]:
    """Cheap (path, mtime_ns, size) tuple — used to key cached reads.

    Returns zero-mtime/size on stat failure so the cache stays valid
    when the file is missing (caller's read will fail and return {}).
    """
    try:
        s = path.stat()
        return (str(path), s.st_mtime_ns, s.st_size)
    except OSError:
        return (str(path), 0, 0)


@st.cache_data(show_spinner=False, max_entries=512)
def _cached_read_json(fingerprint: tuple[str, int, int]) -> dict[str, Any]:
    """JSON manifest cache keyed by mtime+size. Cleared automatically when files change."""
    path_str, _mtime, _size = fingerprint
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    return _cached_read_json(_file_fingerprint(path))


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_truncated_text(fingerprint: tuple[str, int, int], *, limit: int = 7000) -> str:
    """Cached text-file read for the Code-tab snippet lookup."""
    path_str, _mtime, _size = fingerprint
    if not path_str:
        return ""
    try:
        return Path(path_str).read_text(encoding="utf-8")[:limit]
    except OSError:
        return ""


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_artifact_bytes(fingerprint: tuple[str, int, int]) -> bytes:
    """Result-download byte cache keyed by mtime+size.

    Before this, every Streamlit rerun re-read each result artifact from
    disk to build the `st.download_button(data=…)` payload — that made
    the Workbench feel sluggish on every click. The cache key is the
    file fingerprint so an edited artifact invalidates automatically.
    """
    path_str, _mtime, _size = fingerprint
    if not path_str:
        return b""
    try:
        return Path(path_str).read_bytes()
    except OSError:
        return b""


def _compact_label(value: object, *, max_len: int = 72) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _step_label(step_id: object) -> str:
    raw = str(step_id or "").strip() or "step"
    text = re.sub(r"^\d+[_\-. ]*", "", raw)
    text = text.replace("_", " ").replace("-", " ").strip()
    return _compact_label(text.title() if text else raw, max_len=34)


def _dedupe_step_labels(labels: Sequence[str]) -> list[str]:
    """Keep repeated manifest step labels distinguishable in review maps."""
    totals: dict[str, int] = {}
    for label in labels:
        totals[label] = totals.get(label, 0) + 1
    seen: dict[str, int] = {}
    resolved: list[str] = []
    for label in labels:
        if totals.get(label, 0) <= 1:
            resolved.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        resolved.append(_compact_label(f"{label} {seen[label]}", max_len=34))
    return resolved


def _step_id_to_first_index(steps: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Map repeated step IDs to their first visible occurrence."""
    mapping: dict[str, int] = {}
    for idx, step in enumerate(steps):
        step_id = str(step.get("step_id") or step.get("id") or "")
        if step_id:
            mapping.setdefault(step_id, idx)
    return mapping


def _finding_target_step_id(finding: dict[str, Any], step_ids: Sequence[str]) -> str:
    """Infer a step target from validator metadata or message text."""
    explicit = str(finding.get("step_id") or finding.get("step") or "").strip()
    if explicit in step_ids:
        return explicit

    text = " ".join(
        str(finding.get(key) or "")
        for key in ("message", "detail", "title", "evidence_id", "path")
    )
    if not text:
        return ""
    for step_id in sorted((sid for sid in step_ids if sid), key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(step_id)}(?![A-Za-z0-9_])"
        if re.search(pattern, text):
            return step_id
    return ""


def _finding_review_id(finding: dict[str, Any]) -> str:
    if finding.get("id"):
        return str(finding["id"])
    return f"{finding.get('validator', '?')}|{finding.get('severity', '')}|{finding.get('message', '')[:80]}"


def _reviewable_findings(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        finding
        for finding in (audit.get("findings") or [])
        if isinstance(finding, dict)
        and str(finding.get("severity") or "info").lower() in {"error", "warning"}
    ]


def _reviewed_finding_ids(state: dict[str, Any]) -> set[str]:
    return {str(item) for item in (state.get("reviewed_finding_ids") or []) if item}


def _finding_review_state_path(run_dir: str | Path | None) -> Path | None:
    if not run_dir:
        return None
    try:
        return Path(str(run_dir)).expanduser() / _FINDING_REVIEW_STATE_FILE
    except (TypeError, ValueError):
        return None


def _load_reviewed_finding_ids(run_dir: str | Path | None) -> list[str]:
    path = _finding_review_state_path(run_dir)
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_ids = payload.get("reviewed_finding_ids") if isinstance(payload, dict) else payload
    if not isinstance(raw_ids, list):
        return []
    return sorted({str(item) for item in raw_ids if item})


def _write_reviewed_finding_ids(
    run_dir: str | Path | None,
    reviewed_ids: Sequence[str],
    *,
    run_id: object | None = None,
) -> Path | None:
    path = _finding_review_state_path(run_dir)
    if path is None:
        return None
    ids = sorted({str(item) for item in reviewed_ids if item})
    payload = {
        "run_id": str(run_id or Path(str(run_dir)).name),
        "reviewed_finding_ids": ids,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "easyicu_web_research_agent",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return None
    return path


def _sync_reviewed_findings_to_session(state: dict[str, Any]) -> set[str]:
    """Keep the review queue scoped to the currently opened real run."""
    run_dir = str(state.get("run_dir") or "").strip()
    if not run_dir or state.get("is_demo"):
        return _reviewed_finding_ids(state) or {
            str(item) for item in (st.session_state.get(_ACKED_FINDINGS_KEY) or []) if item
        }
    if st.session_state.get(_ACKED_FINDINGS_RUN_KEY) != run_dir:
        loaded = set(_load_reviewed_finding_ids(run_dir))
        st.session_state[_ACKED_FINDINGS_KEY] = sorted(loaded)
        st.session_state[_ACKED_FINDINGS_RUN_KEY] = run_dir
        return loaded
    return {str(item) for item in (st.session_state.get(_ACKED_FINDINGS_KEY) or []) if item}


def _store_reviewed_findings_for_state(state: dict[str, Any], reviewed_ids: set[str]) -> None:
    ids = sorted({str(item) for item in reviewed_ids if item})
    run_dir = str(state.get("run_dir") or "").strip()
    st.session_state[_ACKED_FINDINGS_KEY] = ids
    if run_dir:
        st.session_state[_ACKED_FINDINGS_RUN_KEY] = run_dir
        _write_reviewed_finding_ids(run_dir, ids, run_id=state.get("run_id"))
    state["reviewed_finding_ids"] = ids
    existing = st.session_state.get("_agent_workbench")
    if isinstance(existing, dict) and str(existing.get("run_dir") or "") == run_dir:
        existing["reviewed_finding_ids"] = ids
        st.session_state["_agent_workbench"] = existing


def _finding_queue_rows(
    state: dict[str, Any],
    *,
    reviewed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare reviewable findings for the Workbench queue.

    This is intentionally a UI adapter: it keeps the manifest finding order,
    attaches review state, and resolves only the step target used by the
    Open-step control.
    """
    audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    step_id_to_idx = _step_id_to_first_index(steps)
    step_ids = list(step_id_to_idx)
    reviewed_ids = set(reviewed_ids or _reviewed_finding_ids(state))
    rows: list[dict[str, Any]] = []
    for idx, finding in enumerate(_reviewable_findings(audit)):
        fid = _finding_review_id(finding)
        severity = str(finding.get("severity") or "info").lower()
        target_step = _finding_target_step_id(finding, step_ids)
        target_idx = step_id_to_idx.get(target_step) if target_step else None
        target_label = ""
        if target_idx is not None and 0 <= target_idx < len(steps):
            step_label = steps[target_idx].get("label") or steps[target_idx].get("step_id") or target_step
            target_label = f"{target_idx + 1:02d} · {_compact_label(step_label, max_len=28)}"
        rows.append({
            "index": idx,
            "review_id": fid,
            "finding": finding,
            "severity": severity,
            "validator": finding.get("validator") or "?",
            "message": finding.get("message") or "",
            "reviewed": fid in reviewed_ids,
            "target_step": target_step,
            "target_index": target_idx,
            "target_label": target_label,
        })
    return rows


def _finding_queue_stats(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "reviewed": sum(1 for row in rows if row.get("reviewed")),
        "errors": sum(1 for row in rows if str(row.get("severity") or "").lower() == "error"),
        "warnings": sum(1 for row in rows if str(row.get("severity") or "").lower() == "warning"),
        "linked": sum(1 for row in rows if row.get("target_index") is not None),
    }


def _wb_status(raw: object, *, partial: bool = False) -> str:
    status = str(raw or "").strip().lower()
    if status in _TERMINAL_OK:
        return "ok"
    if status in _TERMINAL_FAIL or "fail" in status or "error" in status or "blocked" in status:
        return "fail"
    if "repair" in status or "retry" in status:
        return "retry"
    if "skip" in status:
        return "pending"
    if partial:
        return "running"
    return "pending"


def _gate_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "ok", "ready", "complete", "satisfied"):
            if isinstance(value.get(key), bool):
                return bool(value[key])
        status = str(value.get("status", "")).strip().lower()
    else:
        status = str(value or "").strip().lower()
    if status in {"pass", "passed", "ok", "ready", "complete", "true", "yes"}:
        return True
    if status in {"fail", "failed", "blocked", "incomplete", "false", "no"}:
        return False
    return None


_INTERNAL_READINESS_GATES = {
    "manuscript_generated",
    "writer_probe_mode",
}


def _readiness_gate_label(name: object) -> str | None:
    """Human-facing readiness label, hiding internal run bookkeeping gates."""
    raw = str(name or "").strip()
    if not raw:
        return None
    normalized = raw.lower().replace(" ", "_").replace("-", "_")
    if normalized in _INTERNAL_READINESS_GATES:
        return None
    return raw.replace("_", " ")


def _evidence_tag(record: dict[str, Any]) -> str:
    kind = str(record.get("kind") or "").lower()
    rel = str(record.get("relative_path") or record.get("path") or "").lower()
    if kind in {"figure", "table", "dataset", "data"}:
        return "data"
    if kind in {"code", "script"} or rel.endswith((".py", ".r", ".sql")):
        return "code"
    if kind in {"paper", "citation", "literature"}:
        return "paper"
    if "audit" in kind or "test" in kind or "validator" in kind:
        return "test"
    if "repair" in kind or "fix" in kind:
        return "fix"
    if rel.endswith((".json", ".jsonl", ".log", ".txt")):
        return "test"
    return "data"


def _evidence_label(record: dict[str, Any]) -> str:
    for key in ("title", "description", "relative_path", "path", "evidence_id"):
        value = record.get(key)
        if value:
            if key == "description":
                return _compact_label(value, max_len=44)
            return _compact_label(Path(str(value)).name or value, max_len=44)
    return "evidence"


def _evidence_sub(record: dict[str, Any]) -> str:
    bits = []
    kind = record.get("kind")
    if kind:
        bits.append(str(kind))
    producer = record.get("producer") or record.get("produced_by_step")
    if producer:
        bits.append(str(producer))
    rel = record.get("relative_path") or record.get("path")
    if rel and Path(str(rel)).name not in bits:
        bits.append(str(rel))
    return _compact_label(" · ".join(bits), max_len=72)


def _cost_tokens(manifest: dict[str, Any]) -> int:
    total = 0
    for item in manifest.get("cost_records") or []:
        if isinstance(item, dict):
            try:
                total += int(item.get("total_tokens") or 0)
            except Exception:
                pass
    repro = manifest.get("reproducibility")
    if total == 0 and isinstance(repro, dict):
        for call in repro.get("calls") or repro.get("llm_calls") or []:
            if isinstance(call, dict):
                try:
                    total += int(call.get("total_tokens") or 0)
                except Exception:
                    pass
    return total


def _step_subtitle(record: dict[str, Any], manifest: dict[str, Any]) -> str:
    parts: list[str] = []
    mode = record.get("generation_mode")
    if mode:
        parts.append(str(mode))
    rc = record.get("returncode")
    if rc is not None:
        parts.append(f"rc={rc}")
    repairs = record.get("code_repair_attempts")
    if repairs:
        parts.append(f"{repairs} repair")
    evidence_ids = record.get("evidence_ids") or []
    produced = [
        ev for ev in manifest.get("evidence", []) or []
        if isinstance(ev, dict) and ev.get("produced_by_step") == record.get("step_id")
    ]
    ev_n = len(set(map(str, evidence_ids))) + len(produced)
    if ev_n:
        parts.append(f"{ev_n} evidence")
    if record.get("diagnostic_only"):
        parts.append("diagnostic")
    status = record.get("status")
    if status and str(status).lower() not in {"ok", "complete", "completed"}:
        parts.append(str(status))
    return _compact_label(" · ".join(parts) or "recorded", max_len=60)


def _artifact_path_for_preview(run_dir: Path | None, record: dict[str, Any]) -> Path | None:
    raw = record.get("relative_path") or record.get("path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else ((run_dir / path) if run_dir else path)


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_figure_preview_html(fingerprint: tuple[str, int, int], name: str) -> str:
    """Render the preview HTML once per (path, mtime, size); cheap on rerun."""
    path_str, _mtime, _size = fingerprint
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return ""
    suffix = path.suffix.lower()
    try:
        if suffix == ".svg":
            svg = path.read_text(encoding="utf-8")[:500_000]
            if "<svg" not in svg.lower() or "<script" in svg.lower():
                return ""
            return f'<div class="eu-result-artifact-preview real">{svg}</div>'
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix)
        if mime:
            data = base64.b64encode(path.read_bytes()[:4_000_000]).decode("ascii")
            return (
                '<div class="eu-result-artifact-preview real">'
                f'<img src="data:{mime};base64,{data}" alt="{_esc(name)}" />'
                '</div>'
            )
    except Exception:
        return ""
    return ""


def _figure_file_preview_html(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return _cached_figure_preview_html(_file_fingerprint(path), path.name)


def _artifact_slot_html(kind: str, *, lang: str, real: bool) -> str:
    if real:
        title = _T(lang, "Registered artifact", "已注册产物")
        detail = _T(lang, "Preview opens only when the generated file can be rendered safely.", "仅当生成文件可安全渲染时才显示预览。")
    else:
        title = _T(lang, "No generated output", "尚无生成输出")
        detail = _T(lang, "Run the agent or open a manifest to populate this slot.", "运行 agent 或打开 manifest 后填充此处。")
    return (
        '<div class="eu-result-artifact-preview empty">'
        f'<b>{_esc(kind)}</b>'
        f'<span>{_esc(title)}</span>'
        f'<small>{_esc(detail)}</small>'
        '</div>'
    )


def _result_cards_from_evidence(
    evidence: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
    lang: str = "en",
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for record in evidence:
        if not isinstance(record, dict):
            continue
        kind = str(record.get("kind") or "").lower()
        if kind not in {"figure", "table"}:
            continue
        path = _artifact_path_for_preview(run_dir, record)
        preview_html = _figure_file_preview_html(path) if kind == "figure" else ""
        rel = str(record.get("relative_path") or record.get("path") or "")
        resolved_path = str(path) if path is not None and path.exists() and path.is_file() else ""
        cards.append({
            "kind": kind,
            "title": _evidence_label(record),
            "metric": _T(lang, "rendered", "已渲染") if preview_html else _T(lang, "registered", "已注册"),
            "sub": rel or _evidence_sub(record),
            "relative_path": rel,
            "path": rel,
            "artifact_path": resolved_path or rel,
            "sha256": str(record.get("sha256") or ""),
            "evidence_id": str(record.get("evidence_id") or ""),
            "preview_html": preview_html or _artifact_slot_html(kind.title(), lang=lang, real=True),
            "svg": "",
        })
        if len(cards) >= 4:
            break
    return cards


def _code_from_evidence(run_dir: Path, evidence: list[dict[str, Any]]) -> tuple[str, str] | None:
    candidates = [
        ev for ev in evidence
        if str(ev.get("relative_path") or ev.get("path") or "").lower().endswith((".py", ".r", ".sql"))
    ]
    for ev in candidates:
        rel = ev.get("relative_path") or ev.get("path")
        if not rel:
            continue
        path = run_dir / str(rel)
        if path.exists() and path.is_file():
            try:
                return _cached_truncated_text(_file_fingerprint(path)), str(rel)
            except Exception:
                continue
    return None


def _code_from_run(
    run_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    preferred_step_id: str | None = None,
) -> tuple[str, str]:
    evidence = [r for r in manifest.get("evidence", []) or [] if isinstance(r, dict)]
    latest_step = preferred_step_id or ""
    if not latest_step:
        for rec in reversed(records):
            if rec.get("step_id"):
                latest_step = str(rec["step_id"])
                break
    candidates: list[dict[str, Any]] = []
    if latest_step:
        candidates.extend([
            ev for ev in evidence
            if ev.get("produced_by_step") == latest_step
            and str(ev.get("relative_path") or ev.get("path") or "").lower().endswith((".py", ".r", ".sql"))
        ])
    candidates.extend([
        ev for ev in evidence
        if str(ev.get("relative_path") or ev.get("path") or "").lower().endswith((".py", ".r", ".sql"))
    ])
    found = _code_from_evidence(run_dir, candidates)
    if found:
        return found
    run_id = manifest.get("run_id") or run_dir.name
    code = (
        f"# EasyICU Research Agent run: {run_id}\n"
        "# No executable script artifact is selected for preview.\n"
        "# Open the evidence list or detailed report for all generated files.\n"
    )
    return code, str(run_dir)


def _evidence_for_step(manifest: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = [r for r in manifest.get("evidence", []) or [] if isinstance(r, dict)]
    by_id = {
        str(rec.get("evidence_id")): rec
        for rec in evidence
        if rec.get("evidence_id") not in (None, "")
    }
    step_id = str(record.get("step_id") or "")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for evidence_id in record.get("evidence_ids") or []:
        rec = by_id.get(str(evidence_id))
        if rec is None:
            continue
        key = str(rec.get("evidence_id") or id(rec))
        if key not in seen:
            seen.add(key)
            out.append(rec)
    for rec in evidence:
        key = str(rec.get("evidence_id") or id(rec))
        if step_id and rec.get("produced_by_step") == step_id and key not in seen:
            seen.add(key)
            out.append(rec)
    return out


def _finding_level(finding: dict[str, Any]) -> str:
    severity = str(finding.get("severity") or "").lower()
    if severity == "error":
        return "err"
    if severity in {"warning", "warn"}:
        return "warn"
    return "ok"


def _step_trace_from_record(record: dict[str, Any], step_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []

    def add(t: str, msg: object, level: str = "ok") -> None:
        text = _compact_label(msg, max_len=108)
        if text:
            trace.append({"t": t, "msg": text, "level": level})

    add("state", f"{record.get('step_id') or 'step'} · {record.get('status') or 'recorded'}")
    if record.get("intent"):
        add("goal", record.get("intent"), "info")
    if record.get("generation_mode"):
        add("executor", record.get("generation_mode"), "info")
    if record.get("returncode") not in (None, 0, "0"):
        add("error", record.get("returncode"), "err")

    summary = record.get("step_summary")
    if isinstance(summary, dict):
        for key in ("n_rows", "n_columns", "target_outcome", "outcome_rate", "error"):
            if summary.get(key) not in (None, "", []):
                add("result", f"{key}: {summary.get(key)}", "err" if key == "error" else "ok")
        if isinstance(summary.get("plots"), dict):
            add("figures", ", ".join(map(str, summary["plots"].keys())), "ok")
        if isinstance(summary.get("high_missingness_variables"), list) and summary["high_missingness_variables"]:
            add("quality", ", ".join(map(str, summary["high_missingness_variables"][:6])), "warn")
    elif summary:
        add("result", summary, "ok")

    finding_fields = (
        "usage_findings",
        "clinical_findings",
        "stat_findings",
        "visual_findings",
        "guard_findings",
        "contract_findings",
    )
    for field in finding_fields:
        for finding in record.get(field) or []:
            if isinstance(finding, dict):
                add(finding.get("validator") or field, finding.get("message") or finding, _finding_level(finding))
            else:
                add(field, finding, "warn")
            if len(trace) >= 10:
                break
        if len(trace) >= 10:
            break
    if step_evidence:
        add("evidence", f"{len(step_evidence)} artifact(s) linked to this step", "ok")
    return trace[:10]


def _contract_status(ok: bool | None) -> str:
    if ok is True:
        return "ok"
    if ok is False:
        return "bad"
    return "wait"


def _artifact_display_path(record: dict[str, Any]) -> str:
    return _compact_label(
        record.get("relative_path")
        or record.get("path")
        or record.get("evidence_id")
        or record.get("kind")
        or "artifact",
        max_len=82,
    )


def _method_binding_for_step(
    *,
    step_id: str,
    record: dict[str, Any],
    step_evidence: list[dict[str, Any]],
    lang: str,
) -> dict[str, str]:
    text = " ".join(
        str(v or "")
        for v in (
            step_id,
            record.get("intent"),
            record.get("generation_mode"),
            record.get("step_summary"),
        )
    ).lower()
    evidence_kinds = sorted({
        str(ev.get("kind") or "artifact").lower()
        for ev in step_evidence
        if isinstance(ev, dict)
    })
    if any(token in text for token in ("probe", "profile", "cohort", "frame")):
        label = _T(lang, "Cohort / variable probe", "队列 / 变量探查")
        audit = _T(lang, "denominator, available variables, missingness", "分母、可用变量、缺失情况")
    elif any(token in text for token in ("missing", "qc", "quality", "audit")):
        label = _T(lang, "Missingness and quality audit", "缺失与质量审计")
        audit = _T(lang, "high-missingness flags, zero artefacts, cohort loss", "高缺失标记、零值伪影、队列损耗")
    elif any(token in text for token in ("model", "association", "regression", "glm", "mortality")):
        label = _T(lang, "Statistical association model", "统计关联模型")
        audit = _T(lang, "estimand, covariates, complete-case risk", "估计目标、协变量、完整病例风险")
    elif any(token in text for token in ("figure", "plot", "publication", "export")):
        label = _T(lang, "Publication figure export", "发表图件导出")
        audit = _T(lang, "source data, vector/bitmap outputs, rendering QA", "源数据、矢量/位图输出、渲染质检")
    elif any(token in text for token in ("manuscript", "draft", "writer", "report")):
        label = _T(lang, "Manuscript gate", "手稿关口")
        audit = _T(lang, "claim ledger, evidence refs, guarded conclusion", "主张账本、证据引用、保守结论")
    else:
        label = _T(lang, "Review step", "复核步骤")
        audit = _T(lang, "registered outputs and evidence checks", "注册产物与证据检查")
    outputs = ", ".join(evidence_kinds[:4]) if evidence_kinds else _T(lang, "no artifact yet", "暂无产物")
    return {
        "label": label,
        "sub": _T(lang, "Bound method template", "已绑定方法模板"),
        "audit": audit,
        "outputs": outputs,
    }


def _step_contract_from_record(
    *,
    run_path: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    step_evidence: list[dict[str, Any]],
    step_number: int,
    lang: str,
) -> dict[str, Any]:
    step_id = str(record.get("step_id") or f"step_{step_number:02d}")
    method = _method_binding_for_step(
        step_id=step_id,
        record=record,
        step_evidence=step_evidence,
        lang=lang,
    )
    inputs: list[dict[str, Any]] = []
    for key, label in (
        ("context_path", _T(lang, "Research context", "研究上下文")),
        ("plan_path", _T(lang, "Analysis plan", "分析计划")),
    ):
        if manifest.get(key):
            inputs.append({
                "path": _compact_label(manifest[key], max_len=72),
                "meta": label,
                "ok": True,
            })
    if record.get("intent"):
        inputs.append({
            "path": _compact_label(record.get("intent"), max_len=72),
            "meta": _T(lang, "step intent", "步骤意图"),
            "ok": True,
        })
    if not inputs:
        inputs.append({
            "path": _compact_label(run_path, max_len=72),
            "meta": _T(lang, "run directory", "运行目录"),
            "ok": True,
        })

    outputs: list[dict[str, Any]] = []
    for ev in step_evidence[:6]:
        outputs.append({
            "path": _artifact_display_path(ev),
            "meta": _compact_label(ev.get("kind") or "artifact", max_len=28),
            "ok": True,
        })
    if not outputs:
        outputs.append({
            "path": _T(lang, "No step artifact registered", "未注册步骤产物"),
            "meta": _T(lang, "expected after execution", "执行后预期"),
            "ok": None,
        })

    returncode = record.get("returncode")
    status = str(record.get("status") or "").lower()
    failure = status in _TERMINAL_FAIL or returncode not in (None, 0, "0")
    checkpoints: list[dict[str, Any]] = [
        {
            "label": _T(lang, "Step status", "步骤状态"),
            "detail": _compact_label(record.get("status") or "recorded", max_len=60),
            "ok": not failure,
        },
        {
            "label": _T(lang, "Evidence bound", "证据已绑定"),
            "detail": _T(lang, f"{len(step_evidence)} artifact(s)", f"{len(step_evidence)} 个产物"),
            "ok": bool(step_evidence),
        },
    ]
    if returncode is not None:
        checkpoints.append({
            "label": _T(lang, "Return code", "返回码"),
            "detail": str(returncode),
            "ok": returncode in (0, "0"),
        })
    if record.get("code_repair_attempts"):
        checkpoints.append({
            "label": _T(lang, "Repair attempts", "修复尝试"),
            "detail": str(record.get("code_repair_attempts")),
            "ok": False if failure else None,
        })
    for field, label in (
        ("contract_findings", _T(lang, "Contract audit", "契约审计")),
        ("clinical_findings", _T(lang, "Clinical audit", "临床审计")),
        ("stat_findings", _T(lang, "Stat audit", "统计审计")),
    ):
        findings = [f for f in record.get(field) or [] if isinstance(f, dict)]
        if findings:
            worst = next((f for f in findings if f.get("severity") == "error"), findings[0])
            checkpoints.append({
                "label": label,
                "detail": _compact_label(worst.get("message") or worst.get("validator") or field, max_len=70),
                "ok": False if worst.get("severity") == "error" else None,
            })
    return {
        "method": method,
        "inputs": inputs[:4],
        "outputs": outputs[:6],
        "checkpoints": checkpoints[:6],
    }


def _evidence_rows_from_records(evidence: list[dict[str, Any]], *, fallback_label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in evidence[:12]:
        sha = str(record.get("sha256") or "")
        raw_path = str(
            record.get("relative_path")
            or record.get("path")
            or record.get("artifact_path")
            or record.get("file")
            or ""
        )
        rows.append({
            "label": _evidence_label(record),
            "sub": _evidence_sub(record),
            "tag": _evidence_tag(record),
            "sha8": sha[:8] if sha else "",
            "sha256": sha,
            "evidence_id": str(record.get("evidence_id") or ""),
            "relative_path": raw_path,
            "path": raw_path,
        })
    if not rows:
        rows.append({
            "label": fallback_label,
            "sub": "no step-specific evidence artifact",
            "tag": "test",
            "sha8": "",
            "sha256": "",
            "evidence_id": "",
            "relative_path": "",
            "path": "",
        })
    return rows


def _step_detail_from_record(
    *,
    run_path: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    step_number: int,
    total_steps: int,
    lang: str,
) -> dict[str, Any]:
    step_id = str(record.get("step_id") or f"step_{step_number:02d}")
    step_evidence = _evidence_for_step(manifest, record)
    step_code = _code_from_evidence(run_path, step_evidence)
    if step_code is None:
        code = (
            f"# Step {step_number:02d}: {step_id}\n"
            f"# {record.get('intent') or 'No executable source was attached to this step.'}\n"
            "# This step is represented by the trace and evidence panel.\n"
        )
        code_path = step_id
    else:
        code, code_path = step_code

    return {
        "step_id": step_id,
        "label": _step_label(step_id),
        "code": code,
        "code_path": code_path,
        "trace": _step_trace_from_record(record, step_evidence),
        "results": _result_cards_from_evidence(step_evidence, run_dir=run_path, lang=lang),
        "evidence": _evidence_rows_from_records(step_evidence, fallback_label=step_id),
        "step_contract": _step_contract_from_record(
            run_path=run_path,
            manifest=manifest,
            record=record,
            step_evidence=step_evidence,
            step_number=step_number,
            lang=lang,
        ),
        "subtitle_short": f"step {step_number}/{total_steps} · {len(step_evidence)} evidence",
        "autopatch": {
            "from": _compact_label(record.get("runner_repair") or "LLM code", max_len=42),
            "to": _compact_label(record.get("deterministic_code_fallback") or "repaired step", max_len=42),
            "ago": "Repair recorded in this step",
        } if (record.get("runner_repair") or record.get("deterministic_code_fallback")) else None,
    }


def _audit_payload(
    *,
    manifest: dict[str, Any],
    run_dir: Path,
    partial: bool,
) -> dict[str, Any]:
    findings = [f for f in manifest.get("findings", []) or [] if isinstance(f, dict)]
    run_status = _read_json(run_dir / "run_status.json")
    gates = manifest.get("readiness") if isinstance(manifest.get("readiness"), dict) else {}
    if not gates and isinstance(run_status.get("gates"), dict):
        gates = run_status["gates"]
    gate_rows = []
    for name, value in (gates or {}).items():
        ok = _gate_passed(value)
        if ok is None:
            continue
        label = _readiness_gate_label(name)
        if label is None:
            continue
        gate_rows.append({"label": label, "ok": ok})
    counts = {
        "errors": sum(1 for f in findings if f.get("severity") == "error"),
        "warnings": sum(1 for f in findings if f.get("severity") == "warning"),
        "info": sum(1 for f in findings if f.get("severity") == "info"),
    }
    repro = manifest.get("reproducibility")
    repro_bits: list[str] = []
    if isinstance(repro, dict):
        for key in ("provider", "model", "requested_seed", "seed", "temperature"):
            if repro.get(key) not in (None, ""):
                repro_bits.append(f"{key}={repro[key]}")
        calls = repro.get("calls") or repro.get("llm_calls") or []
        if isinstance(calls, list) and calls:
            repro_bits.append(f"{len(calls)} call hashes")
    elif manifest.get("used_mock_llm"):
        repro_bits.append("deterministic fallback · no external LLM")
    return {
        "partial": partial,
        "counts": counts,
        "gates": gate_rows[:8],
        "findings": findings,
        "reproducibility": " · ".join(repro_bits),
        "run_status": run_status.get("status") or ("partial" if partial else "complete"),
        "review_decision": _read_json(run_dir / "review_decision.json"),
    }


def _write_summary_review_decision(
    run_dir: Path,
    *,
    decision: str,
    note: str,
    run_id: object | None = None,
) -> dict[str, Any]:
    """Persist the Summary gate decision using the history-page schema."""
    path = Path(run_dir) / "review_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": str(decision),
        "note": str(note or ""),
        "run_id": str(run_id or Path(run_dir).name),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "easyicu_web_research_agent",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _sync_review_decision_to_workbench_state(
    review_decision: dict[str, Any],
    *,
    lang: str,
) -> None:
    """Refresh the in-memory Workbench/Summary state after a local decision save."""
    existing = st.session_state.get("_agent_workbench")
    if not isinstance(existing, dict):
        return
    audit = existing.get("audit") if isinstance(existing.get("audit"), dict) else {}
    audit = dict(audit)
    audit["review_decision"] = dict(review_decision)
    existing["audit"] = audit
    existing["review_decisions"] = _review_decisions_from_audit(
        audit,
        lang=lang,
        is_demo=bool(existing.get("is_demo")),
    )
    st.session_state["_agent_workbench"] = existing


def _prime_summary_draft_setup(state: dict[str, Any]) -> None:
    """Route the Summary CTA into the existing force-manuscript setup path."""
    run_dir_text = str(state.get("run_dir") or "").strip()
    run_id = str(state.get("run_id") or Path(run_dir_text).name or "").strip()
    if not run_id:
        return
    st.session_state["_active_main_page"] = "research_agent"
    st.session_state["research_agent_resume_run_id"] = run_id
    st.session_state["research_agent_force_manuscript"] = True
    st.session_state["research_agent_resume_mode"] = "force_manuscript"
    st.session_state["research_agent_resume_notes"] = ""
    st.session_state["research_agent_resume_relax_probe"] = False
    prior_question = str(state.get("research_question") or "").strip()
    if prior_question:
        st.session_state["research_agent_question"] = prior_question
    st.session_state["_ra_view"] = "setup"
    st.session_state["_research_agent_expand_history"] = False


def _summary_outputs_from_manifest(
    *,
    manifest: dict[str, Any],
    evidence: list[dict[str, Any]],
    lang: str,
) -> list[dict[str, str]]:
    """Build the output-summary gallery from registered run artifacts."""
    outputs: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(kind: str, title: object, sub: object, badge: str = "") -> None:
        label = _compact_label(title, max_len=58)
        if not label:
            return
        key = f"{kind}:{label}:{sub}"
        if key in seen:
            return
        seen.add(key)
        outputs.append({
            "kind": _compact_label(kind, max_len=20),
            "title": label,
            "sub": _compact_label(sub, max_len=80),
            "badge": badge,
        })

    for rec in evidence:
        if not isinstance(rec, dict):
            continue
        kind = str(rec.get("kind") or "").lower()
        if kind in {"figure", "table", "dataset", "statistic", "report"}:
            add(kind or "artifact", _evidence_label(rec), _evidence_sub(rec), _evidence_tag(rec))
        if len(outputs) >= 6:
            break

    for key, label, kind in (
        ("report_path", _T(lang, "Results report", "结果报告"), "report"),
        ("manuscript_path", _T(lang, "Manuscript scaffold", "手稿草稿"), "draft"),
        ("plan_path", _T(lang, "Study plan", "研究方案"), "plan"),
        ("context_path", _T(lang, "Research context", "研究上下文"), "context"),
    ):
        if manifest.get(key) and len(outputs) < 8:
            add(kind, label, manifest[key], key)

    if not outputs:
        outputs.append({
            "kind": _T(lang, "preview", "预览"),
            "title": _T(lang, "No generated artifact yet", "尚无生成产物"),
            "sub": _T(lang, "Run or open a manifest to populate this gallery.", "运行或打开 manifest 后会填充此画廊。"),
            "badge": _T(lang, "empty", "空"),
        })
    return outputs


def _agent_ref_status_class(status: object) -> str:
    value = str(status or "pending").lower()
    if value == "ok":
        return "ready"
    if value == "running":
        return "running"
    if value in {"fail", "error", "blocked"}:
        return "gated"
    if value == "retry":
        return "running"
    return "queued"


def _agent_ref_status_text(status: object, lang: str) -> str:
    value = str(status or "pending").lower()
    if value == "ok":
        return _T(lang, "done", "完成")
    if value == "running":
        return _T(lang, "running", "运行中")
    if value == "retry":
        return _T(lang, "repairing", "修复中")
    if value in {"fail", "error", "blocked"}:
        return _T(lang, "review", "复核")
    return _T(lang, "queued", "排队")


def _countish(value: Any) -> int:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _agent_ref_step_meta(step: dict[str, Any], lang: str) -> str:
    """Human-facing step subtitle for the Claude-style Workbench overview."""
    evidence_count = _countish(step.get("evidence_count"))
    repair_count = _countish(step.get("repair_count"))
    parts: list[str] = []
    if step.get("diagnostic_only"):
        parts.append(_T(lang, "diagnostic", "诊断"))
    if repair_count:
        parts.append(_T(lang, f"{repair_count} repair logged", f"{repair_count} 次修复记录"))
    if evidence_count:
        parts.append(_T(lang, f"{evidence_count} evidence", f"{evidence_count} 条证据"))
    status = str(step.get("status") or "").lower()
    if status in {"fail", "error", "blocked"}:
        parts.append(_T(lang, "needs review", "需要复核"))
    if not parts:
        parts.append(_T(lang, "reviewable step", "可复核步骤"))
    return _compact_label(" · ".join(parts), max_len=54)


def _agent_ref_run_meta(
    *,
    done: int,
    total: int,
    evidence_count: int,
    errors: int,
    warnings: int,
    is_demo: bool,
    lang: str,
) -> str:
    if is_demo:
        pieces = [
            _T(lang, f"{done} of {total} preview steps", f"{done}/{total} 个预览步骤"),
            _T(lang, f"{evidence_count} preview slots", f"{evidence_count} 个预览槽位"),
            _T(lang, "no LLM call", "不调用 LLM"),
        ]
    else:
        pieces = [
            _T(lang, f"{done} of {total} steps", f"{done}/{total} 个步骤"),
            _T(lang, f"{evidence_count} evidence", f"{evidence_count} 条证据"),
        ]
        if errors or warnings:
            pieces.append(_T(lang, f"{errors} error(s)", f"{errors} 个错误"))
            pieces.append(_T(lang, f"{warnings} warning(s)", f"{warnings} 个警告"))
        else:
            pieces.append(_T(lang, "checks clear", "检查通过"))
    return " · ".join(pieces)


def _agent_ref_ledger_icon(kind: str) -> str:
    if kind == "list":
        path = (
            '<path d="M8 6h8M8 10h8M8 14h8"/>'
            '<path d="M5 6h.01M5 10h.01M5 14h.01"/>'
        )
    elif kind == "layers":
        path = '<path d="M12 4 20 8 12 12 4 8 12 4Z"/><path d="M4 12l8 4 8-4"/><path d="M4 16l8 4 8-4"/>'
    elif kind == "lock":
        path = '<rect x="6" y="10" width="12" height="9" rx="2"/><path d="M8 10V8a4 4 0 0 1 8 0v2"/>'
    else:
        path = '<path d="M7 3h7l3 3v15H7z"/><path d="M14 3v4h4"/>'
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" '
        'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        f'{path}'
        '</svg>'
    )


def _agent_ref_thumb(kind: object, title: object = "") -> str:
    kind_l = str(kind or "").lower()
    title_l = str(title or "").lower()
    if "figure" in kind_l or "roc" in title_l or "plot" in title_l:
        return (
            '<svg viewBox="0 0 120 70" role="img" aria-label="figure preview">'
            '<line x1="14" y1="58" x2="106" y2="58" stroke="var(--hair-3)"/>'
            '<line x1="14" y1="58" x2="14" y2="8" stroke="var(--hair-3)"/>'
            '<line x1="14" y1="58" x2="106" y2="8" stroke="var(--hair-2)" stroke-dasharray="2 3"/>'
            '<path d="M14 58 Q 32 32 60 22 Q 88 13 106 10" stroke="var(--accent)" '
            'stroke-width="1.8" fill="none"/>'
            '</svg>'
        )
    if "table" in kind_l:
        rows = "".join(
            f'<rect x="12" y="{8 + r * 12}" width="38" height="5" '
            f'fill="{"var(--ink-3)" if r == 0 else "var(--hair-3)"}" rx="1"/>'
            f'<rect x="56" y="{8 + r * 12}" width="22" height="5" fill="var(--hair-2)" rx="1"/>'
            f'<rect x="84" y="{8 + r * 12}" width="22" height="5" fill="var(--hair-2)" rx="1"/>'
            for r in range(5)
        )
        return f'<svg viewBox="0 0 120 70" role="img" aria-label="table preview">{rows}</svg>'
    if "stat" in kind_l or "num" in kind_l:
        return '<div class="eu-ref-number">10</div>'
    cells = []
    for r in range(7):
        for c in range(11):
            missing = ((r * 7 + c * 3) % 10) > 7
            cells.append(
                f'<rect x="{10 + c * 9}" y="{8 + r * 7.5}" width="6.5" height="5.5" '
                f'fill="{"var(--bad)" if missing else "var(--hair-3)"}" '
                f'opacity="{0.65 if missing else 1}" rx="0.5"/>'
            )
    return f'<svg viewBox="0 0 120 70" role="img" aria-label="artifact preview">{"".join(cells)}</svg>'


def _agent_reference_workbench_html(state: dict[str, Any], lang: str) -> str:
    """Claude Design-compatible overview surface for the Workbench tab."""
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    outputs = [o for o in state.get("summary_outputs", []) if isinstance(o, dict)]
    evidence = [e for e in state.get("evidence", []) if isinstance(e, dict)]
    evidence_total = _countish(state.get("evidence_total")) or len(evidence)
    audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    errors = int(counts.get("errors") or 0)
    warnings = int(counts.get("warnings") or 0)
    finding_rows = _finding_queue_rows(state) if not state.get("is_demo") else []
    finding_stats = _finding_queue_stats(finding_rows)
    finding_total = int(finding_stats.get("total") or 0)
    finding_reviewed = int(finding_stats.get("reviewed") or 0)
    finding_linked = int(finding_stats.get("linked") or 0)
    blocked_gates = [
        gate for gate in audit.get("gates") or []
        if isinstance(gate, dict) and gate.get("ok") is False
    ]
    running = any(str(s.get("status") or "") == "running" for s in steps)
    done = sum(1 for s in steps if str(s.get("status") or "") == "ok")
    total = max(len(steps), 1)
    pct = max(0, min(100, int(round(done / total * 100))))
    if state.get("is_demo"):
        status_label = _T(lang, "Static preview", "静态预览")
        status_class = "ready"
    elif running:
        status_label = _T(lang, "Running", "运行中")
        status_class = "warn"
    elif errors:
        status_label = _T(lang, "Review blocked", "复核阻断")
        status_class = "gated"
    elif warnings:
        status_label = _T(lang, "Review needed", "需要复核")
        status_class = "review"
    elif blocked_gates:
        status_label = _T(lang, "Gate follow-up", "关口跟进")
        status_class = "warn"
    else:
        status_label = _T(lang, "Ready for review", "等待复核")
        status_class = "ok"

    task_rows = []
    for i, step in enumerate(steps[:8], start=1):
        status = step.get("status") or "pending"
        cls = _agent_ref_status_class(status)
        label = _compact_label(step.get("label") or step.get("step_id") or f"step {i}", max_len=34)
        sub = _agent_ref_step_meta(step, lang)
        task_rows.append(
            f'<div class="eu-ref-plan-item {cls}">'
            f'<div class="eu-ref-pi-n mono">{i:02d}</div>'
            '<div class="eu-ref-pi-node"></div>'
            '<div class="eu-ref-pi-body">'
            f'<div class="eu-ref-pi-t">{_esc(label)}</div>'
            f'<div class="eu-ref-pi-d">{_esc(sub)}</div>'
            '</div>'
            '<div class="eu-ref-pi-tag">'
            f'<span class="eu-ref-pill {cls}"><span class="dot"></span>{_esc(_agent_ref_status_text(status, lang))}</span>'
            '</div>'
            '</div>'
        )
    if len(steps) > 8:
        task_rows.append(
            '<div class="eu-ref-plan-more mono">'
            f'+ {len(steps) - 8} {_T(lang, "more step(s) in detailed inspector", "个步骤在详细检查器中")}'
            '</div>'
        )

    if state.get("is_demo"):
        ledger_rows = [
            (_T(lang, "Preview structure", "预览结构"), _T(lang, "no manifest opened", "未打开 manifest"), "file"),
            (_T(lang, "Step contracts", "步骤契约"), f"{len(steps)} {_T(lang, 'preview steps', '个预览步骤')}", "list"),
            (_T(lang, "Template binding", "模板绑定"), _T(lang, "expected slots only", "仅预期槽位"), "layers"),
            (_T(lang, "Review gate", "复核关口"), _T(lang, "draft locked", "草稿锁定"), "lock"),
        ]
    else:
        if finding_total:
            review_gate_sub = _T(
                lang,
                f"{finding_reviewed}/{finding_total} reviewed · {errors} error(s) · {warnings} warning(s)",
                f"{finding_reviewed}/{finding_total} 已复核 · {errors} 错误 · {warnings} 警告",
            )
        else:
            review_gate_sub = _T(lang, f"{errors} error(s) · {warnings} warning(s)", f"{errors} 错误 · {warnings} 警告")
        ledger_rows = [
            (_T(lang, "Run manifest", "运行 manifest"), state.get("run_id") or _T(lang, "current run", "当前运行"), "file"),
            (_T(lang, "Step contracts", "步骤契约"), f"{len(steps)} {_T(lang, 'steps', '步')} · {evidence_total} evidence", "list"),
            (_T(lang, "Template binding", "模板绑定"), _T(lang, "expected tables + checks", "预期表格与检查"), "layers"),
            (_T(lang, "Review gate", "复核关口"), review_gate_sub, "lock"),
        ]
    ledger_html = "".join(
        '<div class="eu-ref-ledger-row">'
        f'<span class="eu-ref-ledger-ico">{_agent_ref_ledger_icon(icon)}</span>'
        '<div>'
        f'<div class="eu-ref-ledger-title">{_esc(title)}</div>'
        f'<div class="eu-ref-ledger-sub">{_esc(str(sub))}</div>'
        '</div>'
        '</div>'
        for title, sub, icon in ledger_rows
    )

    output_rows = []
    for output in outputs[:4]:
        kind = output.get("kind", "")
        title = output.get("title", "")
        output_rows.append(
            '<div class="eu-ref-out-tile">'
            f'<div class="eu-ref-out-thumb">{_agent_ref_thumb(kind, title)}</div>'
            '<div class="eu-ref-out-meta">'
            f'<div class="mono eu-ref-out-kind">{_esc(str(kind))}</div>'
            f'<div class="eu-ref-out-title">{_esc(_compact_label(title, max_len=42))}</div>'
            f'<div class="eu-ref-out-sub">{_esc(_compact_label(output.get("sub"), max_len=60))}</div>'
            '</div>'
            '</div>'
        )
    if not output_rows:
        output_rows.append(
            '<div class="eu-ref-out-tile">'
            '<div class="eu-ref-out-thumb"><div class="eu-ref-number">0</div></div>'
            '<div class="eu-ref-out-meta">'
            f'<div class="eu-ref-out-title">{_T(lang, "No outputs yet", "尚无产物")}</div>'
            f'<div class="eu-ref-out-sub">{_T(lang, "Open or run a manifest to populate this gallery.", "打开或运行 manifest 后填充。")}</div>'
            '</div></div>'
        )

    finding = ""
    for row in [r for r in finding_rows if not r.get("reviewed")] + finding_rows:
        finding = _compact_label(row.get("message") or row.get("validator"), max_len=160)
        if finding and row.get("target_label"):
            finding = _compact_label(f"{row['target_label']}: {finding}", max_len=180)
        if finding:
            break
    if not finding:
        for item in audit.get("findings") or []:
            if isinstance(item, dict):
                finding = _compact_label(item.get("message") or item.get("validator"), max_len=180)
                if finding:
                    break
    review_meta_html = ""
    if finding_total and not state.get("is_demo"):
        review_meta = _T(
            lang,
            f"{finding_reviewed}/{finding_total} findings reviewed · {finding_linked}/{finding_total} linked to a step",
            f"{finding_reviewed}/{finding_total} 条发现已复核 · {finding_linked}/{finding_total} 条可定位到步骤",
        )
        review_meta_html = f'<div class="eu-ref-note-meta mono">{_esc(review_meta)}</div>'
    if state.get("is_demo"):
        note_class = "demo"
        note_icon = "i"
        note_title = _T(lang, "Preview only", "仅预览")
        note_pill_class = "queued"
        note_pill = _T(lang, "static guide", "静态导览")
        if not finding:
            finding = _T(
                lang,
                "No findings have been generated in this static preview. Open a real manifest before reviewing or drafting.",
                "静态预览不会生成发现；请打开真实 manifest 后再复核或起草。",
            )
    elif errors:
        note_class = "warn"
        note_icon = "!"
        note_title = _T(lang, "Findings · resolve before drafting", "发现 · 写作前处理")
        note_pill_class = "gated"
        note_pill = (
            _T(lang, f"{finding_reviewed}/{finding_total} reviewed", f"{finding_reviewed}/{finding_total} 已复核")
            if finding_total else _T(lang, "blocked", "已阻断")
        )
        if not finding:
            finding = _T(
                lang,
                "At least one fail-closed evidence gate is blocked. Resolve the gate before drafting.",
                "至少一个失败即拦截的证据关口被阻断；请先处理后再写作。",
            )
    elif warnings:
        note_class = "warn"
        note_icon = "!"
        note_title = _T(lang, "Findings · review before drafting", "发现 · 写作前复核")
        note_pill_class = "review"
        note_pill = (
            _T(lang, f"{finding_reviewed}/{finding_total} reviewed", f"{finding_reviewed}/{finding_total} 已复核")
            if finding_total else _T(lang, "review needed", "需要复核")
        )
    elif blocked_gates:
        note_class = "warn"
        note_icon = "!"
        note_title = _T(lang, "Gate follow-up required", "需要关口跟进")
        note_pill_class = "queued"
        note_pill = _T(lang, "gate follow-up", "关口跟进")
        if not finding:
            finding = _T(
                lang,
                "Backend readiness gates need follow-up. Inspect Audit tasks before promoting this run to drafting.",
                "后端 readiness 关口需要跟进；进入写作前请先检查 Audit tasks。",
            )
    else:
        note_class = "ok"
        note_icon = "i"
        note_title = _T(lang, "Evidence gate clear", "证据关口已清空")
        note_pill_class = "ok"
        note_pill = _T(lang, "ready", "就绪")
        if not finding:
            finding = _T(
                lang,
                "No blocking finding is recorded. Review the evidence ledger before drafting.",
                "暂无阻断性发现；写作前仍需复核证据 ledger。",
            )

    return (
        '<div class="eu-ref-workbench">'
        '<div class="eu-ref-run-strip">'
        f'<span class="eu-ref-pill {status_class}"><span class="dot"></span>{_esc(status_label)}</span>'
        '<span class="mono eu-ref-run-meta">'
        f'{_esc(_agent_ref_run_meta(done=done, total=len(steps), evidence_count=evidence_total, errors=errors, warnings=warnings, is_demo=bool(state.get("is_demo")), lang=lang))}'
        '</span>'
        '<div class="eu-ref-runbar"><i style="width:'
        f'{pct}%'
        '"></i></div>'
        f'<span class="mono eu-ref-run-pct">{pct}%</span>'
        '</div>'
        '<div class="eu-ref-split">'
        '<div class="eu-ref-card eu-ref-pad">'
        '<div class="eu-ref-card-head">'
        f'<div class="eu-ref-eyebrow">{_T(lang, "Task map", "任务地图")}</div>'
        f'<span class="mono">{_T(lang, "deterministic · repairs logged", "确定性 · 修复有记录")}</span>'
        '</div>'
        f'<div class="eu-ref-planlist">{"".join(task_rows)}</div>'
        '</div>'
        '<div class="eu-ref-card eu-ref-pad">'
        f'<div class="eu-ref-eyebrow">{_T(lang, "Evidence ledger", "证据 ledger")}</div>'
        f'<div class="eu-ref-ledger-list">{ledger_html}</div>'
        '</div>'
        '</div>'
        f'<div class="eu-ref-section-label">{_T(lang, "Analysis outputs", "分析产出")}</div>'
        f'<div class="eu-ref-out-grid">{"".join(output_rows)}</div>'
        f'<div class="eu-ref-note {note_class}">'
        f'<div class="eu-ref-note-ico">{_esc(note_icon)}</div>'
        '<div class="eu-ref-note-body">'
        '<div class="eu-ref-note-head">'
        f'<b>{_esc(note_title)}</b>'
        f'<span class="eu-ref-pill {note_pill_class}">{_esc(note_pill)}</span>'
        '</div>'
        f'<p>{_esc(finding)}</p>'
        f'{review_meta_html}'
        '</div>'
        '</div>'
        '</div>'
    )


def _review_gate_actions_from_audit(audit: dict[str, Any], *, lang: str, is_demo: bool = False) -> list[dict[str, str]]:
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    errors = int(counts.get("errors") or 0)
    warnings = int(counts.get("warnings") or 0)
    blocked_gates = [
        gate for gate in audit.get("gates") or []
        if isinstance(gate, dict) and gate.get("ok") is False
    ]
    if is_demo:
        return [
            {
                "label": _T(lang, "Switch to Real Data", "切换到真实数据"),
                "state": "ready",
                "detail": _T(lang, "Demo mode shows structure only; it does not invent metrics.", "Demo 只展示结构，不编造指标。"),
            },
            {
                "label": _T(lang, "Open Workbench", "打开工作台"),
                "state": "ready",
                "detail": _T(lang, "Inspect the step-by-step agent surface without token use.", "无需 token 即可查看逐步 agent 工作台。"),
            },
        ]
    if errors:
        return [
            {
                "label": _T(lang, "Resolve audit blockers", "处理审计阻断"),
                "state": "blocked",
                "detail": _T(
                    lang,
                    f"{errors} error finding(s).",
                    f"{errors} 个 error 级发现。",
                ),
            },
            {
                "label": _T(lang, "Keep manuscript locked", "保持手稿锁定"),
                "state": "blocked",
                "detail": _T(lang, "Drafting remains a second-stage action.", "写作仍是第二阶段动作。"),
            },
        ]
    if warnings:
        return [
            {
                "label": _T(lang, "Review warnings", "复核警告"),
                "state": "review",
                "detail": _T(lang, f"{warnings} warning(s) require human confirmation.", f"{warnings} 个 warning 需要人工确认。"),
            },
            {
                "label": _T(lang, "Draft after confirmation", "确认后生成草稿"),
                "state": "review",
                "detail": _T(lang, "The evidence gate can be advanced after review.", "人工复核后可推进证据关口。"),
            },
        ]
    if blocked_gates:
        return [
            {
                "label": _T(lang, "Follow up readiness gates", "跟进 readiness 关口"),
                "state": "warning",
                "detail": _T(
                    lang,
                    f"{len(blocked_gates)} backend gate(s) need follow-up.",
                    f"{len(blocked_gates)} 个后端关口需要跟进。",
                ),
            },
            {
                "label": _T(lang, "Keep manuscript locked", "保持手稿锁定"),
                "state": "blocked",
                "detail": _T(lang, "Drafting remains a second-stage action.", "写作仍是第二阶段动作。"),
            },
        ]
    return [
        {
            "label": _T(lang, "Analysis ready", "分析就绪"),
            "state": "ready",
            "detail": _T(lang, "Evidence gates are clear for a manuscript draft.", "证据关口已清空，可以进入手稿草稿。"),
        },
        {
            "label": _T(lang, "Draft methods + results", "生成方法与结果草稿"),
            "state": "ready",
            "detail": _T(lang, "Drafting stays traceable to the manifest.", "草稿继续绑定 manifest 溯源。"),
        },
    ]


def _review_decisions_from_audit(audit: dict[str, Any], *, lang: str, is_demo: bool = False) -> list[dict[str, str]]:
    review = audit.get("review_decision") if isinstance(audit.get("review_decision"), dict) else {}
    if review:
        decision = str(review.get("decision") or "reviewed")
        note = str(review.get("note") or review.get("updated_at") or "")
        tone = "selected"
        if decision in {"repair_requested", "locked"}:
            tone = "warning"
        elif decision == "blocked":
            tone = "danger"
        return [
            {
                "label": _T(lang, f"Saved: {decision}", f"已保存: {decision}"),
                "state": tone,
                "detail": _compact_label(note, max_len=96) or _T(lang, "Local review decision recorded.", "已记录本地审核决定。"),
            }
        ]
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    errors = int(counts.get("errors") or 0)
    warnings = int(counts.get("warnings") or 0)
    failed_gates = [
        gate for gate in audit.get("gates") or []
        if isinstance(gate, dict) and gate.get("ok") is False
    ]
    if is_demo:
        return [
            {
                "label": _T(lang, "Preview only", "仅预览"),
                "state": "selected",
                "detail": _T(lang, "No manuscript decision is written in Demo Mode.", "Demo 模式不写入手稿决策。"),
            },
            {
                "label": _T(lang, "Open real data", "打开真实数据"),
                "state": "idle",
                "detail": _T(lang, "Use a manifest before approving outputs.", "使用 manifest 后再批准输出。"),
            },
        ]
    if errors:
        return [
            {
                "label": _T(lang, "Keep locked", "保持锁定"),
                "state": "selected",
                "detail": _T(lang, "Audit blockers remain unresolved.", "审计阻断尚未解决。"),
            },
            {
                "label": _T(lang, "Request repair", "请求修复"),
                "state": "warning",
                "detail": _T(lang, "Send the selected step back to repair/re-run.", "将当前步骤退回修复或重跑。"),
            },
            {
                "label": _T(lang, "Mark blocked", "标记阻塞"),
                "state": "danger",
                "detail": _T(lang, "Record external dependency or data issue.", "记录外部依赖或数据问题。"),
            },
        ]
    if warnings:
        return [
            {
                "label": _T(lang, "Conditional approve", "附条件通过"),
                "state": "selected",
                "detail": _T(lang, "Allow analysis-only output with caveats.", "允许带 caveat 的 analysis-only 输出。"),
            },
            {
                "label": _T(lang, "Request clarification", "请求澄清"),
                "state": "warning",
                "detail": _T(lang, "Ask the agent to bind or explain warning evidence.", "要求 agent 绑定或解释 warning 证据。"),
            },
            {
                "label": _T(lang, "Keep manuscript locked", "保持手稿锁定"),
                "state": "idle",
                "detail": _T(lang, "Do not promote to manuscript until reviewed.", "复核前不提升到手稿。"),
            },
        ]
    if failed_gates:
        return [
            {
                "label": _T(lang, "Keep locked", "保持锁定"),
                "state": "selected",
                "detail": _T(lang, "Readiness gates need follow-up before drafting.", "写作前仍需跟进 readiness 关口。"),
            },
            {
                "label": _T(lang, "Request gate follow-up", "请求关口跟进"),
                "state": "warning",
                "detail": _T(lang, "Ask the agent to explain or repair the failed gate.", "要求 agent 解释或修复失败关口。"),
            },
        ]
    return [
        {
            "label": _T(lang, "Approve analysis", "通过分析"),
            "state": "selected",
            "detail": _T(lang, "Unlock manuscript drafting from this manifest.", "从该 manifest 解锁手稿草稿。"),
        },
        {
            "label": _T(lang, "Approve figure bundle", "通过图件包"),
            "state": "idle",
            "detail": _T(lang, "Mark generated figures as review-ready.", "将生成图件标记为可审阅。"),
        },
    ]


def _audit_tasks_from_audit(audit: dict[str, Any], *, lang: str, is_demo: bool = False) -> list[dict[str, str]]:
    if is_demo:
        return [
            {
                "title": _T(lang, "Open a real manifest", "打开真实 manifest"),
                "detail": _T(lang, "Demo tasks are placeholders and do not imply generated results.", "Demo 任务只是占位，不代表已生成结果。"),
                "tone": "info",
                "action": _T(lang, "Load run", "加载 run"),
            },
            {
                "title": _T(lang, "Confirm cohort inputs", "确认队列输入"),
                "detail": _T(lang, "Use real cohort/context before execution.", "执行前使用真实队列与上下文。"),
                "tone": "info",
                "action": _T(lang, "Setup", "配置"),
            },
        ]
    tasks: list[dict[str, str]] = []
    failed_gates = [
        gate
        for gate in (audit.get("gates") or [])
        if isinstance(gate, dict) and gate.get("ok") is False
    ]
    if failed_gates:
        labels = [
            _compact_label(gate.get("label"), max_len=28)
            for gate in failed_gates[:4]
            if gate.get("label")
        ]
        detail = (
            _T(
                lang,
                f"{len(failed_gates)} readiness gate(s) still block promotion: {', '.join(labels)}.",
                f"{len(failed_gates)} 个 readiness gate 仍阻止提升：{', '.join(labels)}。",
            )
            if labels else
            _T(lang, "Readiness gates still block promotion.", "readiness gate 仍阻止提升。")
        )
        tasks.append({
            "title": _T(lang, "Review readiness gates", "复核 readiness gates"),
            "detail": detail,
            "tone": "danger",
            "action": _T(lang, "Gate follow-up", "关口跟进"),
        })
    reviewable = _reviewable_findings(audit)
    error_findings = [
        finding
        for finding in reviewable
        if str(finding.get("severity") or "").lower() == "error"
    ]
    warning_findings = [
        finding
        for finding in reviewable
        if str(finding.get("severity") or "").lower() == "warning"
    ]
    if error_findings:
        first = error_findings[0]
        validator = _compact_label(first.get("validator") or "audit", max_len=32)
        message = _compact_label(first.get("message") or "", max_len=88)
        tasks.append({
            "title": _T(lang, f"{len(error_findings)} error finding(s)", f"{len(error_findings)} 个 error 级发现"),
            "detail": f"{validator}: {message}",
            "tone": "danger",
            "action": _T(lang, "Resolve error", "处理 error"),
        })
    if warning_findings:
        tasks.append({
            "title": _T(lang, "Review warning queue", "复核 warning 队列"),
            "detail": _T(
                lang,
                f"{len(warning_findings)} warning finding(s) require human review before Summary sign-off.",
                f"{len(warning_findings)} 个 warning finding 需要人工复核后才能 Summary 签字。",
            ),
            "tone": "warning",
            "action": _T(lang, "Finding queue", "发现队列"),
        })
    if tasks:
        return tasks[:3]
    if not tasks:
        tasks.append({
            "title": _T(lang, "Draft guarded conclusion", "生成保守结论"),
            "detail": _T(lang, "No blocking audit task is recorded for this manifest.", "该 manifest 暂无阻断性审计任务。"),
            "tone": "ok",
            "action": _T(lang, "Draft", "草稿"),
        })
    return tasks[:6]


def build_workbench_state_from_manifest(
    run_dir: str | Path,
    manifest: dict[str, Any] | None,
    *,
    lang: str = "en",
    partial: bool | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map a real research-agent run manifest to Workbench UI state."""
    run_path = Path(run_dir)
    manifest = dict(manifest or {})
    if partial is None:
        partial = (run_path / "manifest_partial.json").exists() and not (run_path / "manifest.json").exists()
    run_id = str(manifest.get("run_id") or run_path.name or "run")
    records = [r for r in manifest.get("per_step_records", []) or [] if isinstance(r, dict)]
    evidence = [r for r in manifest.get("evidence", []) or [] if isinstance(r, dict)]
    findings = [f for f in manifest.get("findings", []) or [] if isinstance(f, dict)]
    progress_events = [e for e in (progress_events or []) if isinstance(e, dict)]

    steps: list[dict[str, Any]] = []
    step_details: list[dict[str, Any]] = []
    display_labels = _dedupe_step_labels([
        _step_label(record.get("step_id") or f"step_{idx:02d}")
        for idx, record in enumerate(records, start=1)
    ])
    for idx, record in enumerate(records, start=1):
        status = _wb_status(record.get("status"), partial=bool(partial))
        step_id = str(record.get("step_id") or f"step_{idx:02d}")
        step_evidence = _evidence_for_step(manifest, record)
        display_label = display_labels[idx - 1] if idx - 1 < len(display_labels) else _step_label(step_id)
        steps.append({
            "label": display_label,
            "sub": _step_subtitle(record, manifest),
            "status": status,
            "step_id": step_id,
            "record_index": idx - 1,
            "evidence_count": len(step_evidence),
            "repair_count": _countish(record.get("code_repair_attempts")),
            "diagnostic_only": bool(record.get("diagnostic_only")),
        })
        step_details.append(_step_detail_from_record(
            run_path=run_path,
            manifest=manifest,
            record=record,
            step_number=idx,
            total_steps=max(len(records), 1),
            lang=lang,
        ) | {"label": display_label})
    if not steps and progress_events:
        for idx, event in enumerate(progress_events[-6:], start=1):
            step_id = str(event.get("step_id") or event.get("stage") or f"event_{idx:02d}")
            steps.append({
                "label": _step_label(step_id),
                "sub": _compact_label(event.get("message"), max_len=60),
                "status": _wb_status(event.get("status"), partial=True),
                "step_id": step_id,
                "record_index": idx - 1,
            })
            step_details.append({
                "step_id": step_id,
                "label": _step_label(step_id),
                "code": f"# Live event: {step_id}\n# {_compact_label(event.get('message'), max_len=120)}\n",
                "code_path": "progress event",
                "trace": [{
                    "t": str(event.get("timestamp", ""))[11:19] or "event",
                    "msg": _compact_label(event.get("message"), max_len=96),
                    "level": "err" if event.get("status") == "error" else "ok",
                }],
                "results": [],
                "evidence": [{"label": step_id, "sub": "live progress event", "tag": "test"}],
                "subtitle_short": f"event {idx}/{len(progress_events[-6:])}",
                "autopatch": None,
            })
    if partial and progress_events:
        latest = progress_events[-1]
        step_id = latest.get("step_id") or latest.get("stage")
        if step_id and not any(s["label"] == _step_label(step_id) and s["status"] == "running" for s in steps):
            if not records or str(step_id) not in {str(r.get("step_id")) for r in records}:
                steps.append({
                    "label": _step_label(step_id),
                    "sub": _compact_label(latest.get("message"), max_len=60),
                    "status": "running",
                    "step_id": str(step_id),
                    "record_index": len(step_details),
                })
                step_details.append({
                    "step_id": str(step_id),
                    "label": _step_label(step_id),
                    "code": f"# Running: {step_id}\n# {_compact_label(latest.get('message'), max_len=120)}\n",
                    "code_path": "progress event",
                    "trace": [{
                        "t": str(latest.get("timestamp", ""))[11:19] or "live",
                        "msg": _compact_label(latest.get("message"), max_len=96),
                        "level": "ok",
                    }],
                    "results": [],
                    "evidence": [{"label": str(step_id), "sub": "live progress event", "tag": "test"}],
                    "subtitle_short": f"event {len(step_details) + 1}/{len(steps)}",
                    "autopatch": None,
                })
    if not steps:
        steps = [{
            "label": _T(lang, "Awaiting plan", "等待计划"),
            "sub": run_id,
            "status": "pending",
            "step_id": "awaiting_plan",
            "record_index": 0,
        }]
        step_details = [{
            "step_id": "awaiting_plan",
            "label": _T(lang, "Awaiting plan", "等待计划"),
            "code": "# Waiting for a Research Agent plan.\n",
            "code_path": str(run_path),
            "trace": [{"t": "wait", "msg": "no manifest step records yet", "level": "info"}],
            "results": [],
            "evidence": [{"label": "manifest", "sub": run_id, "tag": "test"}],
            "subtitle_short": "awaiting manifest",
            "autopatch": None,
        }]

    n_ok = sum(1 for step in steps if step.get("status") == "ok")
    n_fail = sum(1 for step in steps if step.get("status") == "fail")
    is_running = bool(partial) or any(step.get("status") == "running" for step in steps)
    gates = _audit_payload(manifest=manifest, run_dir=run_path, partial=bool(partial))
    errors = gates["counts"]["errors"]
    warnings = gates["counts"]["warnings"]
    status = "running" if is_running else ("blocked" if errors or n_fail else "done")
    status_step = (
        _T(lang, f"running · {n_ok}/{len(steps)} steps", f"运行中 · {n_ok}/{len(steps)} 步")
        if is_running else
        _T(lang, f"review · {errors} errors / {warnings} warnings", f"复核 · {errors} 错误 / {warnings} 警告")
    )

    code, code_path = _code_from_run(run_path, manifest, records)
    result_cards = _result_cards_from_evidence(evidence, run_dir=run_path, lang=lang)
    evidence_rows = [
        {
            "label": _evidence_label(record),
            "sub": _evidence_sub(record),
            "tag": _evidence_tag(record),
        }
        for record in evidence[:12]
    ]
    if not evidence_rows:
        evidence_rows = [{"label": "manifest", "sub": run_id, "tag": "test"}]

    trace = []
    for event in progress_events[-8:]:
        trace.append({
            "t": str(event.get("timestamp", ""))[11:19] or "--:--:--",
            "msg": _compact_label(event.get("message"), max_len=96),
            "level": "err" if event.get("status") == "error" else ("warn" if event.get("status") in {"paused", "skipped"} else "ok"),
        })
    if not trace:
        for finding in findings[:6]:
            trace.append({
                "t": str(finding.get("severity") or "info"),
                "msg": _compact_label(finding.get("message"), max_len=96),
                "level": "err" if finding.get("severity") == "error" else "warn",
            })
    if not trace:
        trace.append({"t": "run", "msg": "manifest loaded", "level": "ok"})

    manifest_rows = []
    for op, key in (("READ", "context_path"), ("READ", "plan_path"), ("WRITE", "report_path"), ("WRITE", "manuscript_path")):
        if manifest.get(key):
            manifest_rows.append({"op": op, "path": str(manifest[key]), "note": key})
    artifact_paths = manifest.get("artifact_paths") if isinstance(manifest.get("artifact_paths"), dict) else {}
    canonical = manifest.get("canonical_outputs") if isinstance(manifest.get("canonical_outputs"), dict) else {}
    for name, rel in {**artifact_paths, **canonical}.items():
        manifest_rows.append({"op": "WRITE", "path": str(rel), "note": str(name)})
    if not manifest_rows:
        manifest_rows.append({"op": "READ", "path": "manifest_partial.json" if partial else "manifest.json", "note": run_id})

    review_rules = [
        _T(lang, "numeric claims require evidence refs", "数值主张必须有证据引用"),
        _T(lang, "error-severity findings block the manuscript gate", "error 级问题会阻止手稿关口"),
        _T(lang, "drafts stay locked until review gates pass", "复核关口通过前草稿保持锁定"),
    ]
    if errors:
        review_rules.insert(0, _T(lang, f"{errors} error finding(s) require review", f"{errors} 个 error 级问题需要复核"))

    total = max(float(len(steps) or 1), 1.0)
    timeline = [
        {"label": _compact_label(step["label"], max_len=16), "t": float(i), "d": 0.9, "status": step["status"]}
        for i, step in enumerate(steps)
    ]
    elapsed = min(total, float(n_ok + (0.5 if is_running else 0)))
    architecture = [
        {"label": _T(lang, "Input lock", "输入锁定"), "short": _T(lang, "Input", "输入"), "sub": "context", "status": "ok"},
        {"label": _T(lang, "Plan", "计划"), "short": _T(lang, "Plan", "计划"), "sub": "analysis plan", "status": "ok" if manifest.get("plan_path") or records else "pending"},
        {"label": _T(lang, "Execute", "执行"), "short": _T(lang, "Exec", "执行"), "sub": "steps", "status": "running" if is_running else ("pending" if n_fail else "ok")},
        {"label": "EvidenceStore", "short": _T(lang, "Evidence", "证据"), "sub": f"{len(evidence)} records", "status": "ok" if evidence else "pending"},
        {"label": _T(lang, "Draft gate", "草稿闸门"), "short": _T(lang, "Gate", "闸门"), "sub": "review", "status": "pending" if errors or warnings or is_running else "ok"},
    ]
    state_lanes = [
        {"key": "staging", "label": _T(lang, "Setup", "配置"), "desc": _T(lang, "input / plan", "输入/计划")},
        {"key": "running", "label": _T(lang, "Execution", "执行"), "desc": _T(lang, "step running", "步骤运行")},
        {"key": "issue", "label": _T(lang, "Needs review", "需要复核"), "desc": _T(lang, "repair or block", "修复或拦截")},
        {"key": "review", "label": _T(lang, "Audit gate", "审计关口"), "desc": _T(lang, "review gate", "复核关口")},
        {"key": "approved", "label": _T(lang, "Approved", "已通过"), "desc": _T(lang, "draft allowed", "允许草稿")},
    ]
    state_segments = []
    for i, step in enumerate(steps):
        lane = "running"
        if step["status"] in {"fail", "retry"}:
            lane = "issue"
        elif step["status"] == "pending":
            lane = "review"
        elif step["status"] == "ok" and i == len(steps) - 1 and not errors and not warnings and not is_running:
            lane = "approved"
        elif i == 0:
            lane = "staging"
        state_segments.append({"lane": lane, "start": float(i), "end": float(i) + 0.9, "label": _compact_label(step["label"], max_len=14)})

    autopatch = None
    repair_record = next((r for r in reversed(records) if r.get("runner_repair") or r.get("deterministic_code_fallback")), None)
    if repair_record:
        autopatch = {
            "from": _compact_label(repair_record.get("runner_repair") or "LLM code", max_len=42),
            "to": _compact_label(repair_record.get("deterministic_code_fallback") or "repaired step", max_len=42),
            "ago": _T(lang, "Repair recorded in this run", "本次运行记录了修复"),
        }

    research_question = str(manifest.get("research_question") or "").strip()
    question = _compact_label(research_question or run_id, max_len=72)
    subtitle_bits = [run_id, f"{len(steps)} steps", f"{len(evidence)} evidence", f"{len(findings)} findings"]
    summary_outputs = _summary_outputs_from_manifest(
        manifest=manifest,
        evidence=evidence,
        lang=lang,
    )
    reviewed_finding_ids = _load_reviewed_finding_ids(run_path)
    execution_contract = {
        "cohort": _compact_label(manifest.get("context_path") or manifest.get("cohort_path") or run_path, max_len=72),
        "provider": _compact_label(
            (manifest.get("reproducibility") or {}).get("provider")
            if isinstance(manifest.get("reproducibility"), dict) else "",
            max_len=44,
        ) or _T(lang, "recorded in run", "运行中记录"),
        "workdir": str(run_path),
        "gate": gates.get("run_status") or ("partial" if partial else "complete"),
    }
    return {
        "run_id": run_id,
        "run_dir": str(run_path),
        "title": question or _T(lang, "Research Agent run", "研究智能体运行"),
        "research_question": research_question,
        "subtitle": " · ".join(subtitle_bits),
        "subtitle_short": (
            f"{len(evidence)} evidence · "
            + (
                f"{errors} error(s) · {warnings} warning(s)"
                if errors or warnings else
                _T(lang, "checks clear", "检查通过")
            )
        ),
        "evidence_total": len(evidence),
        "status": status,
        "status_step": status_step,
        "steps": steps,
        "code_path": code_path,
        "code": code,
        "autopatch": autopatch,
        "trace": trace,
        "results": result_cards,
        "evidence": evidence_rows,
        "step_details": step_details,
        "timeline": timeline,
        "elapsed": elapsed,
        "total": total,
        "playhead": elapsed,
        "tokens": _cost_tokens(manifest),
        "architecture": architecture,
        "manifest": manifest_rows,
        "review_rules": review_rules,
        "summary_outputs": summary_outputs,
        "artifact_counts": _artifact_counts_from_records(evidence),
        "execution_contract": execution_contract,
        "review_gate_actions": _review_gate_actions_from_audit(gates, lang=lang),
        "review_decisions": _review_decisions_from_audit(gates, lang=lang),
        "audit_tasks": _audit_tasks_from_audit(gates, lang=lang),
        "state_lanes": state_lanes,
        "state_segments": state_segments,
        "audit": gates,
        "reviewed_finding_ids": reviewed_finding_ids,
        "finding_review_state_path": str(_finding_review_state_path(run_path) or ""),
        "source_label": _T(lang, "Real manifest", "真实 manifest"),
        "is_demo": False,
    }


# ---------------------------------------------------------------------
# Demo fallback state (mirrors the design canvas)
# ---------------------------------------------------------------------

def _demo_state(lang: str) -> dict[str, Any]:
    state = {
        "title": _T(lang, "Research workflow preview", "研究流程预览"),
        "subtitle": _T(lang, "Demo structure only · no cohort loaded · no metrics generated", "仅 Demo 结构 · 未加载队列 · 未生成指标"),
        "status": "preview",
        "status_step": _T(lang, "preview only", "仅预览"),
        "steps": [
            {"label": _T(lang, "Cohort summary", "队列总结"), "sub": _T(lang, "preview structure", "预览结构"), "status": "ok"},
            {"label": "Table 1", "sub": _T(lang, "artifact slot only", "仅产物槽位"), "status": "ok"},
            {"label": _T(lang, "Missingness audit", "缺失审计"), "sub": _T(lang, "audit slot only", "仅审计槽位"), "status": "ok"},
            {"label": _T(lang, "Model step", "模型步骤"), "sub": _T(lang, "real run required", "需要真实运行"), "status": "pending"},
            {"label": _T(lang, "ROC · Calibration", "ROC · 校准"), "sub": _T(lang, "real run required", "需要真实运行"), "status": "pending"},
            {"label": _T(lang, "Reviewer gate", "复核关口"), "sub": _T(lang, "locked until evidence", "证据前锁定"), "status": "pending"},
        ],
        "code_path": "demo/preview_only.py",
        "code": (
            "# Demo preview only\n"
            "# No cohort is loaded, no model is fitted, and no files are written here.\n"
            "# Open a real manifest to inspect the bound script, table, figure, and audit log.\n\n"
            "execution_contract = {\n"
            '    "cohort": "selected in Setup",\n'
            '    "provider": "selected in Setup",\n'
            '    "outputs": ["table", "figure", "audit", "report"],\n'
            '    "gate": "human review before manuscript draft",\n'
            "}\n"
        ),
        "autopatch": {
            "from": _T(lang, "example failing code", "示例失败代码"),
            "to": _T(lang, "example repaired code", "示例修复代码"),
            "ago": _T(lang, "Retry branch preview", "重试分支预览"),
        },
        "trace": [
            {"t": "demo", "msg": "preview mode: no cohort loaded", "level": "info"},
            {"t": "demo", "msg": "preview mode: no model run", "level": "info"},
            {"t": "demo", "msg": "preview mode: no table or figure file written", "level": "warn"},
        ],
        "results": [
            {
                "kind": _T(lang, "table slot", "表格槽位"),
                "title": _T(lang, "Generated table", "生成表格"),
                "metric": _T(lang, "not generated", "未生成"),
                "sub": _T(lang, "Real run required for table preview.", "需要真实 run 才能预览表格。"),
                "preview_html": _artifact_slot_html(_T(lang, "Table", "表格"), lang=lang, real=False),
            },
            {
                "kind": _T(lang, "figure slot", "图件槽位"),
                "title": _T(lang, "Generated figure", "生成图件"),
                "metric": _T(lang, "not generated", "未生成"),
                "sub": _T(lang, "Real manifest required for figure preview.", "需要真实 manifest 才能预览图件。"),
                "preview_html": _artifact_slot_html(_T(lang, "Figure", "图件"), lang=lang, real=False),
            },
        ],
        "evidence": [
            {"label": _T(lang, "cohort input", "队列输入"), "sub": _T(lang, "waiting for real setup", "等待真实配置"), "tag": "data"},
            {"label": _T(lang, "analysis plan", "分析计划"), "sub": _T(lang, "created after launch", "启动后创建"), "tag": "test"},
            {"label": _T(lang, "table artifact", "表格产物"), "sub": _T(lang, "no file in demo mode", "Demo 模式无文件"), "tag": "data"},
            {"label": _T(lang, "figure artifact", "图件产物"), "sub": _T(lang, "no file in demo mode", "Demo 模式无文件"), "tag": "data"},
        ],
        "timeline": [
            {"label": "cohort", "t": 0.0, "d": 0.1, "status": "ok"},
            {"label": "table", "t": 0.1, "d": 0.1, "status": "ok"},
            {"label": "audit", "t": 0.2, "d": 0.1, "status": "ok"},
            {"label": "model", "t": 0.3, "d": 0.1, "status": "pending"},
            {"label": "figure", "t": 0.4, "d": 0.1, "status": "pending"},
            {"label": "gate", "t": 0.5, "d": 0.1, "status": "pending"},
        ],
        "elapsed": 0.0,
        "total": 1.0,
        "playhead": 0.0,
        "tokens": 0,
        "architecture": [
            {
                "label": _T(lang, "Input lock", "输入锁定"),
                "short": _T(lang, "Input", "输入"),
                "sub": _T(lang, "cohort + concept manifest", "队列 + 概念清单"),
                "status": "ok",
            },
            {
                "label": _T(lang, "Plan", "计划"),
                "short": _T(lang, "Plan", "计划"),
                "sub": _T(lang, "7 executable steps", "7 个可执行步骤"),
                "status": "ok",
            },
            {
                "label": _T(lang, "Execute", "执行"),
                "short": _T(lang, "Exec", "执行"),
                "sub": _T(lang, "sandboxed scripts", "沙箱脚本"),
                "status": "running",
            },
            {
                "label": "EvidenceStore",
                "short": _T(lang, "Evidence", "证据"),
                "sub": _T(lang, "tables, logs, hashes", "表格、日志、哈希"),
                "status": "running",
            },
            {
                "label": _T(lang, "Draft gate", "草稿闸门"),
                "short": _T(lang, "Gate", "闸门"),
                "sub": _T(lang, "human review required", "需要人工复核"),
                "status": "pending",
            },
        ],
        "manifest": [
            {"op": "CONFIG", "path": _T(lang, "cohort selected in Setup", "在配置页选择队列"), "note": "required"},
            {"op": "CONFIG", "path": _T(lang, "LLM provider selected in Setup", "在配置页选择模型提供方"), "note": "required"},
            {"op": "WRITE", "path": _T(lang, "none in demo mode", "Demo 模式不写入"), "note": "preview"},
        ],
        "review_rules": [
            _T(lang, "numeric claims require evidence refs", "数值主张必须有证据引用"),
            _T(lang, "missingness warning blocks clean clinical claims", "缺失警告阻止干净临床结论"),
            _T(lang, "manuscript draft waits for human confirmation", "主文草稿等待人工确认"),
        ],
        "state_lanes": [
            {"key": "staging", "label": _T(lang, "Setup", "配置"), "desc": _T(lang, "checks inputs", "检查输入")},
            {"key": "running", "label": _T(lang, "Execution", "执行"), "desc": _T(lang, "step running", "步骤运行")},
            {"key": "issue", "label": _T(lang, "Needs review", "需要复核"), "desc": _T(lang, "repair or retry", "修复或重试")},
            {"key": "review", "label": _T(lang, "Audit gate", "审计关口"), "desc": _T(lang, "human gate", "人工关口")},
            {"key": "approved", "label": _T(lang, "Approved", "已通过"), "desc": _T(lang, "unlocks draft", "解锁草稿")},
        ],
        "state_segments": [
            {"lane": "staging", "start": 0.0, "end": 0.2, "label": "lock"},
            {"lane": "running", "start": 0.2, "end": 0.4, "label": "run"},
            {"lane": "issue", "start": 0.4, "end": 0.5, "label": "issue"},
            {"lane": "staging", "start": 0.5, "end": 0.6, "label": "retry"},
            {"lane": "running", "start": 0.6, "end": 0.8, "label": "outputs"},
            {"lane": "review", "start": 0.8, "end": 1.0, "label": "gate"},
        ],
    }
    steps = state["steps"]
    for i, step in enumerate(steps):
        step.setdefault("step_id", f"demo_step_{i + 1:02d}")
        step.setdefault("record_index", i)
    state["step_details"] = [
        {
            "step_id": step["step_id"],
            "label": step["label"],
            "code": (
                state["code"] if i in {5, 6, 7} else
                f"# Demo step {i + 1:02d}: {step['label']}\n"
                f"# {step['sub']}\n"
                "# Preview only. Open a real run to inspect the bound script.\n"
            ),
            "code_path": state["code_path"] if i in {5, 6, 7} else step["step_id"],
            "trace": [
                {"t": "demo", "msg": f"{step['label']} · {step['status']}", "level": "ok" if step["status"] == "ok" else "warn"},
                {"t": "note", "msg": step["sub"], "level": "info"},
            ],
            "results": state["results"] if i in {6, 7} else [],
            "evidence": state["evidence"][:4] if i in {6, 7} else [{
                "label": step["label"],
                "sub": _T(lang, "preview step, no artifact", "预览步骤，无产物"),
                "tag": "test",
            }],
            "step_contract": {
                "method": {
                    "label": _T(lang, "Demo method slot", "Demo 方法槽"),
                    "sub": _T(lang, "No real method is executed", "未执行真实方法"),
                    "audit": _T(lang, "Open a real manifest for checkpoints.", "打开真实 manifest 查看检查点。"),
                    "outputs": _T(lang, "preview only", "仅预览"),
                },
                "inputs": [
                    {
                        "path": _T(lang, "Demo cohort structure", "Demo 队列结构"),
                        "meta": _T(lang, "sample only", "仅样例"),
                        "ok": None,
                    }
                ],
                "outputs": [
                    {
                        "path": _T(lang, "No file written", "未写入文件"),
                        "meta": _T(lang, "demo mode", "Demo 模式"),
                        "ok": None,
                    }
                ],
                "checkpoints": [
                    {
                        "label": _T(lang, "No fabricated metrics", "不编造指标"),
                        "detail": _T(lang, "Real outputs require a manifest.", "真实输出需要 manifest。"),
                        "ok": True,
                    }
                ],
            },
            "subtitle_short": f"preview step {i + 1}/{len(steps)}",
            "autopatch": state["autopatch"] if i == 4 else None,
        }
        for i, step in enumerate(steps)
    ]
    state["source_label"] = _T(lang, "Demo structure only", "仅 Demo 结构")
    state["is_demo"] = True
    state["summary_outputs"] = [
        {
            "kind": _T(lang, "plan", "计划"),
            "title": _T(lang, "Study plan slot", "研究方案位置"),
            "sub": _T(lang, "Generated only after a real or recorded run is opened.", "只有真实或历史 run 打开后才生成。"),
            "badge": _T(lang, "preview", "预览"),
        },
        {
            "kind": _T(lang, "table", "表格"),
            "title": _T(lang, "Cohort table slot", "队列表位置"),
            "sub": _T(lang, "Demo mode does not fabricate cohort metrics.", "Demo 模式不伪造队列指标。"),
            "badge": _T(lang, "empty", "空"),
        },
        {
            "kind": _T(lang, "figure", "图件"),
            "title": _T(lang, "Analysis figure slot", "分析图位置"),
            "sub": _T(lang, "Open a real manifest to preview generated figures.", "打开真实 manifest 后预览生成图件。"),
            "badge": _T(lang, "empty", "空"),
        },
        {
            "kind": _T(lang, "audit", "审计"),
            "title": _T(lang, "Evidence manifest slot", "证据清单位置"),
            "sub": _T(lang, "Evidence links appear only when artifacts exist.", "只有产物存在时才显示证据链接。"),
            "badge": _T(lang, "gate", "关口"),
        },
    ]
    state["execution_contract"] = {
        "cohort": _T(lang, "Demo structure only", "仅 Demo 结构"),
        "provider": _T(lang, "No LLM call", "不调用 LLM"),
        "workdir": _T(lang, "No files written", "不写入文件"),
        "gate": _T(lang, "No metrics generated", "不生成指标"),
    }
    state["review_gate_actions"] = _review_gate_actions_from_audit(
        {"counts": {"errors": 0, "warnings": 0, "info": 0}, "gates": []},
        lang=lang,
        is_demo=True,
    )
    state["review_decisions"] = _review_decisions_from_audit(
        {"counts": {"errors": 0, "warnings": 0, "info": 0}, "gates": []},
        lang=lang,
        is_demo=True,
    )
    state["audit_tasks"] = _audit_tasks_from_audit(
        {"counts": {"errors": 0, "warnings": 0, "info": 0}, "gates": []},
        lang=lang,
        is_demo=True,
    )
    state["demo_notice"] = _T(
        lang,
        "Demo mode shows the agent surface without generating new metrics.",
        "Demo 模式只展示 agent 界面，不生成或编造新指标。",
    )
    return state


# ---------------------------------------------------------------------
# Column renderers (return HTML strings)
# ---------------------------------------------------------------------

def _process_graph_html(state: dict[str, Any], lang: str) -> str:
    steps = state.get("steps", [])
    n_ok = sum(1 for s in steps if s["status"] == "ok")
    n_retry = sum(1 for s in steps if s["status"] == "retry")
    n_run = sum(1 for s in steps if s["status"] == "running")

    rows = []
    for i, s in enumerate(steps):
        status = s["status"]
        status_label = {
            "ok": "done",
            "fail": "issue",
            "retry": "retry",
            "running": "run",
            "pending": "next",
        }.get(status, status)
        is_retry_branch = status == "retry"
        rows.append(
            f'<div class="eu-agent-step {status}{" branch" if is_retry_branch else ""}">'
            f'<div class="eu-agent-step-num mono">{i + 1:02d}</div>'
            '<div class="eu-agent-step-copy">'
            f'<div class="eu-agent-step-label">{_esc(s["label"])}</div>'
            f'<div class="eu-agent-step-sub mono">{_esc(s["sub"])}</div>'
            '</div>'
            f'<div class="eu-agent-step-status mono">{_esc(status_label)}</div>'
            '</div>'
        )

    legend = "".join(
        f'<span style="display:flex;align-items:center;gap:4px">'
        f'<span style="width:7px;height:7px;border-radius:999px;background:{color}"></span>{label}</span>'
        for label, color in [("ok", "var(--ink)"), ("retry", "var(--warn)"),
                             ("running", "var(--accent)"), ("fail", "var(--bad)")]
    )
    return (
        '<div style="padding:14px 16px;height:100%;display:flex;flex-direction:column">'
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">'
        '<div>'
        f'<div style="font-size:12.5px;font-weight:600">{_T(lang, "Run queue", "运行队列")} · {len(steps)} {_T(lang, "steps", "步")}</div>'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4)">{n_ok} ok · {n_retry} retry · {n_run} running</div>'
        '</div></div>'
        '<div class="eu-agent-queue" style="flex:1;overflow:auto">'
        '<div class="eu-agent-start mono">START</div>'
        + "".join(rows)
        + '</div>'
        f'<div style="display:flex;gap:10px;margin-top:8px;font-size:10.5px;color:var(--ink-3)">{legend}</div>'
        '</div>'
    )


def _step_status_label(status: str, lang: str) -> str:
    return {
        "ok": _T(lang, "done", "完成"),
        "fail": _T(lang, "issue", "问题"),
        "retry": _T(lang, "retry", "重试"),
        "running": _T(lang, "run", "运行"),
        "pending": _T(lang, "next", "等待"),
    }.get(status, status)


def _step_status_action_label(status: str, lang: str) -> str:
    return {
        "ok": _T(lang, "DONE", "完成"),
        "fail": _T(lang, "NEEDS FIX", "需修复"),
        "retry": _T(lang, "RETRYING", "重试中"),
        "running": _T(lang, "RUNNING", "运行中"),
        "pending": _T(lang, "QUEUED", "排队中"),
    }.get(status, status.upper())


def _step_legend_html(lang: str) -> str:
    items = [
        ("ok", _T(lang, "Done", "完成")),
        ("running", _T(lang, "Running", "运行中")),
        ("pending", _T(lang, "Queued", "等待")),
        ("fail", _T(lang, "Needs fix", "需修复")),
        ("retry", _T(lang, "Retrying", "重试中")),
    ]
    chips = "".join(
        '<span class="eu-agent-step-legend-chip '
        f'{status}"><i></i><b>{_esc(label)}</b></span>'
        for status, label in items
    )
    return (
        '<div class="eu-agent-step-legend" aria-label="Step status legend">'
        + chips
        + '</div>'
    )


def _step_button_label(step: dict[str, Any], idx: int, lang: str) -> str:
    status = str(step.get("status") or "pending")
    label = _compact_label(step.get("label") or step.get("step_id") or f"step {idx + 1}", max_len=32)
    sub = _agent_ref_step_meta(step, lang)
    status_label = _agent_ref_status_text(status, lang)
    if sub:
        return f"{idx + 1:02d}  {label}\n{status_label} · {sub}"
    return f"{idx + 1:02d}  {label}\n{status_label}"


def _step_button_key(state: dict[str, Any], idx: int, status: str, selected: bool) -> str:
    raw = str(state.get("run_id") or "demo")
    safe_run = re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:54]
    safe_status = re.sub(r"[^a-z0-9_]+", "_", status.lower())[:24] or "pending"
    suffix = "selected" if selected else "idle"
    return f"_eu_wb_step_btn_{safe_run}_{idx:02d}_{safe_status}_{suffix}"


def _set_selected_step(select_key: str, idx: int) -> None:
    st.session_state[select_key] = idx
    st.session_state[_REVIEW_DETAILS_EXPANDED_KEY] = True


def _step_select_key(state: dict[str, Any]) -> str:
    raw = str(state.get("run_id") or "demo")
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:70]
    return f"_eu_wb_step_select_{safe}"


def _default_selected_step(steps: list[dict[str, Any]]) -> int:
    for i, step in enumerate(steps):
        if step.get("status") == "running":
            return i
    for i, step in enumerate(steps):
        if step.get("status") == "fail":
            return i
    for i in range(len(steps) - 1, -1, -1):
        if steps[i].get("status") == "ok":
            return i
    return 0


def _resolve_selected_step(state: dict[str, Any]) -> tuple[str, int]:
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    key = _step_select_key(state)
    default = _default_selected_step(steps) if steps else 0
    current = st.session_state.get(key, default)
    try:
        selected = int(current)
    except Exception:
        selected = default
    if not steps:
        selected = 0
    else:
        selected = max(0, min(selected, len(steps) - 1))
    if key in st.session_state and selected != current:
        st.session_state.pop(key, None)
        selected = default
    st.session_state[key] = selected
    return key, selected


def _state_for_selected_step(state: dict[str, Any], selected_idx: int) -> dict[str, Any]:
    details = [d for d in state.get("step_details", []) if isinstance(d, dict)]
    if selected_idx < 0 or selected_idx >= len(details):
        return state
    detail = details[selected_idx]
    view_state = dict(state)
    for key in ("code", "code_path", "trace", "results", "evidence", "subtitle_short", "autopatch", "step_contract"):
        if key in detail and detail[key] is not None:
            view_state[key] = detail[key]
    view_state["active_step"] = {
        "index": selected_idx,
        "label": detail.get("label") or (state.get("steps") or [{}])[selected_idx].get("label", ""),
        "step_id": detail.get("step_id", ""),
    }
    return view_state


def _render_process_graph_controls(
    state: dict[str, Any],
    lang: str,
    *,
    selected_idx: int,
    select_key: str,
) -> int:
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    st.markdown(
        '<div class="eu-agent-process-head">'
        f'<b>{_T(lang, "Review steps", "复核步骤")} · {len(steps)} {_T(lang, "steps", "步")}</b>'
        f'<span class="mono">{_esc(_review_steps_summary_text(steps, lang))}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if steps:
        st.markdown(
            '<div class="eu-agent-step-rail-note mono">'
            f'{_T(lang, "Select a step to review checklist, outputs, evidence, and activity.", "选择步骤复核检查清单、产物、证据和活动记录。")}'
            '</div>',
            unsafe_allow_html=True,
        )
    cols_per_row = 2 if len(steps) > 1 else 1
    for row_start in range(0, len(steps), cols_per_row):
        row_steps = steps[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="small")
        for offset, step in enumerate(row_steps):
            i = row_start + offset
            status = str(step.get("status") or "pending")
            is_selected = i == selected_idx
            with cols[offset]:
                st.button(
                    _step_button_label(step, i, lang),
                    key=_step_button_key(state, i, status, is_selected),
                    use_container_width=True,
                    on_click=_set_selected_step,
                    args=(select_key, i),
                )
    return selected_idx


def _review_steps_summary_text(steps: list[dict[str, Any]], lang: str) -> str:
    """Summarize visible step review status without losing recorded repairs."""
    n_ok = sum(1 for s in steps if s.get("status") == "ok")
    n_run = sum(1 for s in steps if s.get("status") == "running")
    repair_events = sum(_countish(s.get("repair_count")) for s in steps)
    retry_without_count = sum(
        1
        for s in steps
        if s.get("status") == "retry" and not _countish(s.get("repair_count"))
    )
    n_repair = repair_events + retry_without_count
    return (
        f'{n_ok} {_T(lang, "done", "完成")} · '
        f'{n_repair} {_T(lang, "repair", "修复")} · '
        f'{n_run} {_T(lang, "running", "运行中")}'
    )


def _step_flow_html(state: dict[str, Any], lang: str, *, selected_idx: int = 0) -> str:
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    if not steps:
        return ""
    nodes = [
        (
            '<div class="eu-agent-flow-node root">'
            f'<span class="mono">{_T(lang, "INPUT", "输入")}</span>'
            f'<b>{_T(lang, "Question + cohort", "问题 + 队列")}</b>'
            '</div>'
        )
    ]
    for idx, step in enumerate(steps):
        status = str(step.get("status") or "pending")
        label = _compact_label(step.get("label") or step.get("step_id") or f"step {idx + 1}", max_len=28)
        sub = _compact_label(step.get("sub") or _step_status_label(status, lang), max_len=42)
        selected = " selected" if idx == selected_idx else ""
        branch = " branch" if status in {"retry", "fail"} else ""
        nodes.append(
            f'<div class="eu-agent-flow-node {status}{selected}{branch}">'
            f'<span class="mono">{idx + 1:02d} · {_esc(_step_status_label(status, lang))}</span>'
            f'<b>{_esc(label)}</b>'
            f'<small>{_esc(sub)}</small>'
            '</div>'
        )
    nodes.append(
        (
            '<div class="eu-agent-flow-node gate">'
            f'<span class="mono">{_T(lang, "GATE", "闸门")}</span>'
            f'<b>{_T(lang, "Evidence review", "证据复核")}</b>'
            '</div>'
        )
    )
    return (
        '<div class="eu-agent-flow-wrap">'
        '<div class="eu-agent-flow-rail"></div>'
        '<div class="eu-agent-flow-nodes">'
        + "".join(nodes)
        + '</div>'
        '</div>'
    )


def _live_code_html(state: dict[str, Any], lang: str) -> str:
    tabs = [
        (_T(lang, "Code", "代码"), True, ""),
        (_T(lang, "Output", "输出"), False, ""),
        (_T(lang, "Errors", "错误"), False, "1"),
        (_T(lang, "History", "历史"), False, str(len(state.get("steps", [])))),
    ]
    tab_parts = []
    for title, active, count in tabs:
        count_html = (
            '<span class="mono" style="font-size:10px;color:var(--ink-4)">'
            + _esc(count)
            + '</span>'
            if count else ""
        )
        tab_parts.append(
            f'<div style="padding:10px 12px;font-size:12px;'
            f'color:{"var(--ink)" if active else "var(--ink-3)"};'
            f'border-bottom:{"2px solid var(--ink)" if active else "2px solid transparent"};'
            f'margin-bottom:-1px;font-weight:{500 if active else 400};display:flex;align-items:center;gap:6px">'
            f'{_esc(title)}{count_html}</div>'
        )
    tab_html = "".join(tab_parts)

    autopatch = state.get("autopatch")
    autopatch_html = ""
    if autopatch:
        autopatch_html = (
            '<div style="margin:10px 14px 0;padding:10px 12px;border-radius:var(--r-2);'
            'background:var(--warn-soft);border:1px solid oklch(86% 0.05 75);'
            'display:flex;align-items:flex-start;gap:10px;font-size:12px;color:oklch(30% 0.10 75)">'
            '<span style="margin-top:1px">✦</span>'
            '<div style="flex:1">'
            f'<div style="font-weight:500">{_esc(autopatch["ago"])}</div>'
            '<div style="margin-top:2px">'
            f'<span class="mono" style="color:var(--bad);text-decoration:line-through">{_esc(autopatch["from"])}</span>'
            ' → '
            f'<span class="mono" style="color:var(--ok)">{_esc(autopatch["to"])}</span>'
            '</div></div></div>'
        )

    code = state.get("code", "")
    code_lines = code.split("\n")
    gutter = "".join(f'<div style="line-height:18px">{i + 1}</div>' for i in range(len(code_lines)))
    code_html = "".join(
        '<div style="line-height:18px;min-height:18px;white-space:pre">'
        f'{_highlight_python_html(line) if line else "&nbsp;"}'
        '</div>'
        for line in code_lines
    )

    trace = state.get("trace", [])
    trace_color = {"ok": "#B8D7A3", "info": "#9BD2F4", "warn": "#FFC580", "err": "#F4A6A6"}
    trace_html = "".join(
        f'<div><span style="color:#7A8A99">{_esc(t["t"])}</span>  {_esc(t["msg"])} '
        f'<span style="color:{trace_color.get(t.get("level", "ok"), "#B8D7A3")}">'
        f'{"" if t.get("level") == "info" else ""}</span></div>'
        for t in trace
    )

    return (
        '<div style="display:flex;flex-direction:column;height:100%">'
        '<div style="flex:none;border-bottom:1px solid var(--hair);display:flex;align-items:center;padding:0 12px">'
        + tab_html
        + '<span style="margin-left:auto;display:flex;align-items:center;gap:6px">'
        '<span class="eu-pill"><span class="dot eu-pulse" style="background:var(--accent)"></span>'
        f'{_T(lang, "streaming", "流式")}</span></span>'
        '</div>'
        + autopatch_html
        + '<div style="flex:1;padding:10px 14px;display:flex;flex-direction:column;min-height:0">'
        '<div style="flex:1;min-height:0;display:flex;overflow:auto">'
        '<div class="mono" style="flex:none;width:32px;padding:4px 6px 4px 4px;font-size:11px;'
        'color:var(--ink-4);text-align:right;border-right:1px solid var(--hair)">'
        + gutter
        + '</div>'
        '<div class="mono" style="margin:0;padding:4px 12px;flex:1;min-width:0;font-size:11.5px;'
        'line-height:18px;background:transparent;color:var(--ink);overflow:visible">'
        + code_html
        + '</div></div>'
        '<div class="mono" style="flex:none;margin-top:10px;background:var(--ink);color:#E8E6DD;'
        'border-radius:var(--r-3);padding:10px 12px;font-size:11px;line-height:1.55;max-height:130px;overflow:auto">'
        + trace_html
        + '</div></div></div>'
    )


def _result_evidence_html(state: dict[str, Any], lang: str) -> str:
    results = state.get("results", [])
    result_cards = []
    for r in results:
        preview_html = r.get("preview_html") or r.get("svg") or _artifact_slot_html(
            str(r.get("title") or r.get("kind") or "artifact"),
            lang=lang,
            real=not bool(state.get("is_demo")),
        )
        result_cards.append(
            '<div class="eu-card" style="padding:10px">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<div class="mono" style="font-size:10px;color:var(--ink-4);letter-spacing:0.06em;'
            f'text-transform:uppercase">{_esc(r["kind"])}</div>'
            f'<span class="mono" style="font-size:11px;color:var(--ink)">{_esc(r["metric"])}</span>'
            '</div>'
            f'<div class="eu-result-card-title">{_esc(r.get("title", ""))}</div>'
            f'<div style="margin-top:7px">{preview_html}</div>'
            + (
                '<div class="mono" style="font-size:10px;color:var(--ink-4);margin-top:4px">'
                + _esc(r.get("sub", ""))
                + '</div>'
                if r.get("sub") else ""
            )
            + '</div>'
        )

    evidence = state.get("evidence", [])
    tag_style = {
        "fix": ("var(--warn-soft)", "oklch(40% 0.10 75)"),
        "data": ("var(--accent-soft)", "var(--accent-ink)"),
        "paper": ("var(--surface-2)", "var(--ink-3)"),
        "code": ("var(--surface-2)", "var(--ink-3)"),
        "test": ("var(--surface-2)", "var(--ink-3)"),
    }
    ev_rows = []
    for i, e in enumerate(evidence):
        bg, fg = tag_style.get(e["tag"], ("var(--surface-2)", "var(--ink-3)"))
        sha8 = str(e.get("sha8") or "")
        ev_id = str(e.get("evidence_id") or "")
        prov_bits = []
        if sha8:
            prov_bits.append(f"sha:{sha8}")
        if ev_id:
            prov_bits.append(ev_id)
        prov_html = (
            f'<div class="mono" style="font-size:10px;color:var(--ink-4);'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'margin-top:2px">{_esc(" · ".join(prov_bits))}</div>'
        ) if prov_bits else ""
        ev_rows.append(
            f'<div style="padding:8px 12px;display:grid;grid-template-columns:20px 1fr auto;gap:8px;'
            f'align-items:center;{"border-top:1px solid var(--hair);" if i else ""}">'
            f'<span style="color:var(--ink-3);display:flex;align-items:center;justify-content:center">'
            f'{_evidence_icon_svg(e["tag"])}</span>'
            '<div style="min-width:0">'
            f'<div class="mono" style="font-size:11.5px;color:var(--ink);white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{_esc(e["label"])}</div>'
            f'<div style="font-size:10.5px;color:var(--ink-4)">{_esc(e["sub"])}</div>'
            f'{prov_html}'
            '</div>'
            f'<span class="eu-chip mono" style="font-size:9.5px;padding:0 5px;background:{bg};color:{fg}">'
            f'{_esc(e["tag"])}</span>'
            '</div>'
        )

    return (
        '<div style="padding:14px;display:flex;flex-direction:column;gap:12px;height:100%;overflow:auto">'
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        '<div>'
        f'<div style="font-size:12.5px;font-weight:500">{_T(lang, "Results", "结果")} · {len(results)}</div>'
        f'<div class="mono" style="font-size:11px;color:var(--ink-4)">{_esc(state.get("subtitle_short", ""))}</div>'
        '</div></div>'
        + "".join(result_cards)
        + f'<div class="eu-section-label" style="padding:0;margin-top:4px">'
          f'<span>{_T(lang, "Evidence", "证据")} · {len(evidence)}</span></div>'
        '<div class="eu-card" style="padding:0;overflow:hidden">'
        + "".join(ev_rows)
        + '</div></div>'
    )


def _agent_architecture_html(state: dict[str, Any], lang: str) -> str:
    """PlanAgent-inspired architecture strip, styled in the EasyICU shell."""
    stages = state.get("architecture") or []
    if not stages:
        return ""
    cards = []
    for i, stage in enumerate(stages):
        status = stage.get("status", "pending")
        cards.append(
            f'<div class="eu-agent-arch-card {status}">'
            f'<div class="idx mono">{i + 1:02d}</div>'
            '<div>'
            f'<div class="label">{_esc(stage.get("label", ""))}</div>'
            f'<div class="sub mono">{_esc(stage.get("sub", ""))}</div>'
            '</div>'
            '</div>'
        )
    return (
        '<div class="eu-agent-arch">'
        '<div class="eu-agent-arch-head">'
        f'<span class="mono">{_T(lang, "Agent architecture", "Agent 架构")}</span>'
        f'<span class="mono muted">{_T(lang, "EasyICU export -> gated draft", "EasyICU 导出 -> 闸门草稿")}</span>'
        '</div>'
        f'<div class="eu-agent-arch-grid">{"".join(cards)}</div>'
        '</div>'
    )


def _agent_command_strip_html(state: dict[str, Any], lang: str) -> str:
    """Compact first-screen command strip.

    The earlier version stacked architecture, ledger, preflight files and
    review rules as separate bands. That was faithful to the source concepts
    but too cramped in the actual Streamlit shell, so the default surface now
    shows only the live architecture + ledger summary; detailed contract
    rows stay available behind a native details disclosure.
    """
    stages = state.get("architecture") or []
    steps = state.get("steps", [])
    results = state.get("results", [])
    evidence = state.get("evidence", [])
    manifest = state.get("manifest", [])
    rules = state.get("review_rules", [])
    elapsed = float(state.get("elapsed", 0) or 0)
    total = float(state.get("total", 1) or 1)
    pct = max(0, min(100, int(round(elapsed / max(total, 0.01) * 100))))
    ok_steps = sum(1 for s in steps if s.get("status") == "ok")
    running = next((s for s in steps if s.get("status") == "running"), None)

    stage_html = "".join(
        f'<span class="eu-cmd-node {stage.get("status", "pending")}">'
        f'<i class="mono">{i + 1:02d}</i>'
        f'<b>{_esc(stage.get("short") or stage.get("label", ""))}</b>'
        '</span>'
        for i, stage in enumerate(stages)
    )
    ledger_items = [
        (_T(lang, "In", "输入"), _T(lang, "locked", "锁定"), "ok"),
        (_T(lang, "Run", "执行"), f"{ok_steps}/{len(steps)}", "warn" if running else "ok"),
        (_T(lang, "Out", "输出"), str(len(results)), "ok" if results else "warn"),
        (_T(lang, "Ev", "证据"), str(len(evidence)), "ok" if evidence else "warn"),
    ]
    ledger_html = "".join(
        f'<span class="eu-cmd-metric {tone}">'
        f'<em>{_esc(label)}</em><b>{_esc(value)}</b>'
        '</span>'
        for label, value, tone in ledger_items
    )
    manifest_rows = "".join(
        '<div class="eu-manifest-row">'
        f'<span class="op mono">{_esc(row.get("op", ""))}</span>'
        f'<span class="path mono">{_esc(row.get("path", ""))}</span>'
        f'<span class="note">{_esc(row.get("note", ""))}</span>'
        '</div>'
        for row in manifest[:6]
    )
    rule_rows = "".join(
        '<div class="eu-review-rule">'
        '<span></span>'
        f'<p>{_esc(rule)}</p>'
        '</div>'
        for rule in rules[:4]
    )
    current = running.get("label") if running else _T(lang, "Ready for review", "等待复核")
    return (
        '<div class="eu-agent-command">'
        '<div class="eu-agent-command-run">'
        '<div class="left">'
        f'<span class="idx mono">{_esc(state.get("run_id", "RUN-003"))}</span>'
        f'<b>{_esc(state.get("title", "Research Agent"))}</b>'
        f'<span class="eu-pill">{_esc(state.get("source_label", ""))}</span>'
        f'<span class="eu-pill"><span class="dot eu-pulse" style="background:var(--accent)"></span>{_esc(state.get("status", "running"))}</span>'
        '</div>'
        '<div class="mid">'
        f'<span class="mono">{pct}%</span>'
        '<div class="bar"><i style="width:'
        f'{pct}%'
        '"></i></div>'
        f'<span class="mono">{elapsed:.1f}s / {total:.1f}s</span>'
        f'<span class="mono">{len(steps)} steps</span>'
        f'<span class="mono">{int(state.get("tokens", 0)):,} tok</span>'
        '</div>'
        '</div>'
        '<div class="eu-agent-command-line">'
        '<div class="eu-agent-command-now">'
        f'<span class="mono">{_T(lang, "Phase overview", "阶段概览")}</span>'
        f'<b>{_esc(current)}</b>'
        '</div>'
        f'<div class="eu-cmd-stage-row">{stage_html}</div>'
        f'<div class="eu-cmd-ledger-row">{ledger_html}</div>'
        '</div>'
        '<details class="eu-agent-contract-details">'
        f'<summary>{_T(lang, "Open preflight contract and review rules", "展开执行前契约与复核规则")}</summary>'
        '<div class="eu-agent-contract compact">'
        '<div class="eu-contract-col">'
        f'<div class="eu-contract-title mono">{_T(lang, "Preflight contract", "执行前契约")}</div>'
        f'<div class="eu-manifest-list">{manifest_rows}</div>'
        '</div>'
        '<div class="eu-contract-col">'
        f'<div class="eu-contract-title mono">{_T(lang, "Review rules", "复核规则")}</div>'
        f'<div class="eu-review-rule-list">{rule_rows}</div>'
        '</div>'
        '</div>'
        '</details>'
        '</div>'
    )


def _timeline_html(state: dict[str, Any], lang: str) -> str:
    steps = state.get("timeline", [])
    total = state.get("total", 20) or 20
    elapsed = state.get("elapsed", 0)
    playhead = state.get("playhead", elapsed)
    tokens = state.get("tokens", 0)

    blocks = []
    for s in steps:
        left = (s["t"] / total) * 100
        w = (s["d"] / total) * 100
        c = _STATUS_COLOR.get(s["status"], "var(--hair-3)")
        pulse = " eu-pulse" if s["status"] == "running" else ""
        opacity = "0.5" if s["status"] == "pending" else "1"
        blocks.append(
            f'<div class="{pulse.strip()}" style="position:absolute;left:{left:.1f}%;top:14px;'
            f'height:8px;width:{w:.1f}%;background:{c};border-radius:2px;opacity:{opacity}"></div>'
            f'<div class="mono" style="position:absolute;left:{left:.1f}%;top:26px;font-size:9px;'
            f'color:var(--ink-4);white-space:nowrap">{_esc(s["label"])}</div>'
        )
    ph_left = (playhead / total) * 100
    done_n = sum(1 for s in steps if s["status"] == "ok")

    return (
        '<div style="padding:10px 16px;border-top:1px solid var(--hair);background:var(--surface);'
        'display:flex;align-items:center;gap:12px">'
        '<div style="display:flex;flex-direction:column;font-size:11px;color:var(--ink-3);'
        'line-height:1.2;min-width:110px">'
        f'<span class="mono" style="font-weight:500;color:var(--ink-2)">⏱ {elapsed:.1f}s / {total:.1f}s</span>'
        f'<span class="mono" style="font-size:10px;color:var(--ink-4)">{done_n} of {len(steps)} · {tokens:,} tok</span>'
        '</div>'
        '<div style="flex:1;position:relative;height:40px">'
        '<div style="position:absolute;left:0;right:0;top:14px;height:8px;background:var(--surface-2);border-radius:4px"></div>'
        + "".join(blocks)
        + f'<div style="position:absolute;left:{ph_left:.1f}%;top:4px;bottom:0">'
        '<div style="width:2px;height:28px;background:var(--accent)"></div>'
        '<div style="width:8px;height:8px;background:var(--accent);border-radius:999px;margin-top:-22px;margin-left:-3px"></div>'
        '</div></div></div>'
    )


def _state_track_html(state: dict[str, Any], lang: str) -> str:
    """Compact state-lane visualization adapted from agentdesign page 08."""
    lanes = state.get("state_lanes") or []
    segments = state.get("state_segments") or []
    if not lanes or not segments:
        return _timeline_html(state, lang)

    total = float(state.get("total", 20) or 20)
    elapsed = float(state.get("elapsed", 0) or 0)
    tokens = int(state.get("tokens", 0) or 0)
    lane_index = {lane["key"]: i for i, lane in enumerate(lanes)}
    row_h = 34
    track_h = 24 + (len(lanes) * row_h)

    label_rows = []
    for lane in lanes:
        label_rows.append(
            '<div class="eu-state-lane-label">'
            f'<span class="eu-state-dot {lane["key"]}"></span>'
            '<div>'
            f'<b>{_esc(lane.get("label", ""))}</b>'
            f'<small>{_esc(lane.get("desc", ""))}</small>'
            '</div>'
            '</div>'
        )

    lane_lines = "".join(
        f'<div style="position:absolute;left:0;right:0;top:{24 + i * row_h}px;height:{row_h}px;'
        f'border-top:1px dashed var(--hair)"></div>'
        for i in range(len(lanes) + 1)
    )
    axis = "".join(
        f'<span style="left:{p}%">{(total * p / 100):.0f}s</span>'
        for p in (0, 25, 50, 75, 100)
    )
    bars = []
    for seg in segments:
        lane = str(seg.get("lane", "running"))
        top = 31 + lane_index.get(lane, 0) * row_h
        left = max(0, min(100, float(seg.get("start", 0)) / total * 100))
        right = max(0, min(100, float(seg.get("end", 0)) / total * 100))
        width = max(2.2, right - left)
        bars.append(
            f'<div class="eu-state-segment {lane}" style="left:{left:.1f}%;top:{top}px;width:{width:.1f}%">'
            f'{_esc(seg.get("label", ""))}</div>'
        )
    playhead = max(0, min(100, elapsed / total * 100))

    return (
        '<div class="eu-agent-state-track">'
        '<div class="eu-agent-state-head">'
        '<div>'
        f'<b>{_T(lang, "Review timeline", "复核时间线")}</b>'
        f'<span class="mono">{elapsed:.1f}s / {total:.1f}s · {tokens:,} tok</span>'
        '</div>'
        f'<span class="mono muted">{_T(lang, "step timing · review states", "步骤耗时 · 复核状态")}</span>'
        '</div>'
        '<div class="eu-state-grid">'
        f'<div class="eu-state-labels">{"".join(label_rows)}</div>'
        f'<div class="eu-state-canvas" style="height:{track_h}px">'
        f'<div class="eu-state-axis">{axis}</div>'
        f'{lane_lines}{"".join(bars)}'
        f'<div class="eu-state-playhead" style="left:{playhead:.1f}%"></div>'
        '</div></div></div>'
    )


def _audit_review_html(state: dict[str, Any], lang: str) -> str:
    audit = state.get("audit")
    if not isinstance(audit, dict):
        return ""
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    errors = int(counts.get("errors") or 0)
    warnings = int(counts.get("warnings") or 0)
    infos = int(counts.get("info") or 0)
    gate_rows = []
    for gate in audit.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        ok = bool(gate.get("ok"))
        gate_rows.append(
            '<div class="eu-audit-gate">'
            f'<span class="{ "ok" if ok else "bad" }"></span>'
            f'<b>{_esc(gate.get("label", ""))}</b>'
            f'<small>{_T(lang, "pass", "通过") if ok else _T(lang, "blocked", "拦截")}</small>'
            '</div>'
        )
    note_rows = []
    for finding in audit.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        sev = str(finding.get("severity") or "info").lower()
        if sev in {"error", "warning"}:
            continue
        note_rows.append(
            f'<div class="eu-audit-finding {sev}">'
            f'<span></span>'
            '<div>'
            f'<b class="mono">{_esc(finding.get("validator", "?"))}</b>'
            f'<p>{_esc(finding.get("message", ""))}</p>'
            '</div>'
            '</div>'
        )
    repro = audit.get("reproducibility")
    repro_html = (
        f'<div class="eu-audit-repro mono">{_esc(repro)}</div>'
        if repro else
        f'<div class="eu-audit-repro mono">{_T(lang, "No LLM reproducibility envelope recorded", "未记录 LLM 可复现性信封")}</div>'
    )
    gate_body = "".join(gate_rows) or '<p class="muted">No fail-closed gates recorded.</p>'
    note_body = "".join(note_rows[:4]) or (
        f'<p class="muted">{_T(lang, "Reviewable findings are handled in the Finding queue below.", "可复核 finding 在下方 Finding queue 中处理。")}</p>'
    )
    return (
        '<div class="eu-agent-audit">'
        '<div class="eu-audit-head">'
        f'<b>{_T(lang, "Review gate", "复核关口")}</b>'
        f'<span class="mono">{_esc(audit.get("run_status", ""))}</span>'
        '</div>'
        '<div class="eu-audit-metrics">'
        f'<div class="err"><span>{_T(lang, "Errors", "错误")}</span><b>{errors}</b></div>'
        f'<div class="warn"><span>{_T(lang, "Warnings", "警告")}</span><b>{warnings}</b></div>'
        f'<div><span>{_T(lang, "Info", "信息")}</span><b>{infos}</b></div>'
        '</div>'
        '<div class="eu-audit-review-grid">'
        f'{_review_decisions_html(state, lang)}'
        f'{_audit_tasks_html(state, lang)}'
        '</div>'
        f'<div class="eu-audit-gates">{gate_body}</div>'
        f'{repro_html}'
        '<div class="eu-audit-findings">'
        f'<div class="eu-contract-title mono">{_T(lang, "Audit notes", "审计备注")}</div>'
        f'{note_body}'
        '</div>'
        '</div>'
    )


def _summary_bundle_counts(state: dict[str, Any]) -> dict[str, int]:
    evidence = [e for e in state.get("evidence", []) if isinstance(e, dict)]
    outputs = [o for o in state.get("summary_outputs", []) if isinstance(o, dict)]
    evidence_total = _countish(state.get("evidence_total")) or len(evidence)
    artifact_counts = state.get("artifact_counts")
    if isinstance(artifact_counts, dict):
        counts = {
            "figures": _countish(artifact_counts.get("figures")),
            "tables": _countish(artifact_counts.get("tables")),
            "code": _countish(artifact_counts.get("code")),
            "evidence": _countish(artifact_counts.get("evidence")) or evidence_total,
        }
    else:
        counts = {"figures": 0, "tables": 0, "code": 0, "evidence": evidence_total}
        for rec in evidence:
            kind = str(rec.get("kind") or rec.get("tag") or "").lower()
            if kind in {"figure", "fig"}:
                counts["figures"] += 1
                continue
            if kind == "table":
                counts["tables"] += 1
                continue
            if kind in {"code", "script"}:
                counts["code"] += 1
                continue
            if kind:
                continue
            text = " ".join(
                str(rec.get(key) or "")
                for key in ("kind", "tag", "relative_path", "label", "title", "sub")
            ).lower()
            if "figure" in text or text.endswith((".png", ".svg", ".jpg", ".jpeg", ".webp")):
                counts["figures"] += 1
            if "table" in text or ".csv" in text or ".tsv" in text or ".tex" in text:
                counts["tables"] += 1
            if "code" in text or ".py" in text or ".ipynb" in text:
                counts["code"] += 1
    output_counts = {"figures": 0, "tables": 0, "code": 0}
    for rec in outputs:
        text = " ".join(
            str(rec.get(key) or "")
            for key in ("kind", "badge", "title", "sub")
        ).lower()
        if "figure" in text or "plot" in text or "roc" in text:
            output_counts["figures"] += 1
        if "table" in text or "csv" in text or "tex" in text:
            output_counts["tables"] += 1
        if "code" in text or ".py" in text or "notebook" in text:
            output_counts["code"] += 1
    for key in output_counts:
        counts[key] = max(counts[key], output_counts[key])
    return counts


def _artifact_counts_from_records(evidence: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"figures": 0, "tables": 0, "code": 0, "evidence": len(evidence)}
    for rec in evidence:
        kind = str(rec.get("kind") or rec.get("tag") or "").lower()
        if kind in {"figure", "fig"}:
            counts["figures"] += 1
            continue
        if kind == "table":
            counts["tables"] += 1
            continue
        if kind in {"code", "script"}:
            counts["code"] += 1
            continue
        if kind:
            continue
        text = " ".join(
            str(rec.get(key) or "")
            for key in ("kind", "tag", "relative_path", "label", "title", "sub")
        ).lower()
        if "figure" in text or text.endswith((".png", ".svg", ".jpg", ".jpeg", ".webp")):
            counts["figures"] += 1
        if "table" in text or ".csv" in text or ".tsv" in text or ".tex" in text:
            counts["tables"] += 1
        if "code" in text or ".py" in text or ".ipynb" in text:
            counts["code"] += 1
    return counts


def _summary_bundle_icon(kind: str) -> str:
    if kind == "figure":
        path = '<path d="M4 19h16"/><path d="M7 16l3-4 3 2 4-7"/><path d="M7 7h.01M17 7h.01"/>'
    elif kind == "table":
        path = '<path d="M4 6h16M4 12h16M4 18h16"/><path d="M9 6v12M15 6v12"/>'
    elif kind == "evidence":
        path = '<path d="M12 3 20 7v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7l8-4Z"/><path d="m9 12 2 2 4-5"/>'
    else:
        path = '<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>'
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" '
        'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        f'{path}'
        '</svg>'
    )


def _summary_cohort_denominators_resolved(state: dict[str, Any]) -> bool:
    """Return whether the run carries enough cohort/table-one evidence for review.

    Normal agent runs often express the denominator step as ``Table One`` or
    ``Outcome Incidence`` rather than a literal "cohort" step. The Summary gate
    should follow the registered evidence contract instead of a display label.
    """
    denominator_tokens = (
        "cohort",
        "cohort_locked",
        "cohort.parquet",
        "denominator",
        "table_one",
        "table one",
        "outcome_incidence",
        "outcome incidence",
    )
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    for step in steps:
        status = str(step.get("status") or "").lower()
        if status not in _TERMINAL_OK:
            continue
        haystack = " ".join(
            str(step.get(field) or "")
            for field in ("label", "step_id", "sub", "subtitle_short")
        ).lower()
        if any(token in haystack for token in denominator_tokens):
            return True

    evidence = [e for e in state.get("evidence", []) if isinstance(e, dict)]
    for rec in evidence:
        haystack = " ".join(
            str(rec.get(field) or "")
            for field in (
                "evidence_id",
                "label",
                "title",
                "relative_path",
                "path",
                "artifact_path",
                "file",
            )
        ).lower()
        if any(token in haystack for token in denominator_tokens):
            return True
    return False


def _summary_review_checks(state: dict[str, Any], lang: str) -> list[dict[str, object]]:
    if state.get("is_demo"):
        return [
            {
                "label": _T(lang, "Demo context structure is visible", "Demo 上下文结构已可见"),
                "ok": True,
            },
            {
                "label": _T(lang, "No LLM call or token use", "不调用 LLM、不消耗 token"),
                "ok": True,
            },
            {
                "label": _T(lang, "Artifact slots are labelled as preview", "产物槽位已标注为预览"),
                "ok": True,
            },
            {
                "label": _T(lang, "Draft gate remains locked", "草稿关口保持锁定"),
                "ok": True,
            },
            {
                "label": _T(lang, "Real run or manifest bound", "已绑定真实 run 或 manifest"),
                "ok": False,
            },
        ]
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    evidence = [e for e in state.get("evidence", []) if isinstance(e, dict)]
    audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    bundle_counts = _summary_bundle_counts(state)
    review = audit.get("review_decision") if isinstance(audit.get("review_decision"), dict) else {}
    review_status = str(review.get("decision") or review.get("status") or "").lower()
    reviewed_ids = _reviewed_finding_ids(state)
    reviewable_findings = _reviewable_findings(audit)
    error_findings = [
        finding
        for finding in reviewable_findings
        if str(finding.get("severity") or "").lower() == "error"
    ]
    warning_findings = [
        finding
        for finding in reviewable_findings
        if str(finding.get("severity") or "").lower() == "warning"
    ]
    reviewed_warnings = sum(
        1
        for finding in warning_findings
        if _finding_review_id(finding) in reviewed_ids
    )
    cohort_ok = _summary_cohort_denominators_resolved(state)
    findings_reviewed = not error_findings and reviewed_warnings == len(warning_findings)
    findings_status = (
        _T(lang, "error unresolved", "错误未处理")
        if error_findings else
        _T(lang, f"{reviewed_warnings}/{len(warning_findings)} reviewed", f"已复核 {reviewed_warnings}/{len(warning_findings)}")
        if warning_findings else
        _T(lang, "passed", "通过")
    )
    return [
        {
            "label": _T(lang, "Cohort denominators resolved", "队列分母已确认"),
            "ok": cohort_ok,
        },
        {
            "label": _T(lang, "Evidence manifest attached", "证据 manifest 已绑定"),
            "ok": bool(evidence),
        },
        {
            "label": _T(lang, "Tables and figures registered", "表格与图件已注册"),
            "ok": bool(bundle_counts["tables"] or bundle_counts["figures"]),
        },
        {
            "label": _T(lang, "Validator findings reviewed", "校验发现已复核"),
            "ok": findings_reviewed,
            "status": findings_status,
        },
        {
            "label": _T(lang, "Reviewer sign-off", "审核者签字"),
            "ok": review_status in _APPROVED_REVIEW_DECISIONS,
        },
    ]


def _summary_bundle_index_download(state: dict[str, Any], lang: str) -> tuple[str, str]:
    bundle_counts = _summary_bundle_counts(state)
    payload = {
        "run_id": state.get("run_id"),
        "run_dir": state.get("run_dir"),
        "source_label": state.get("source_label"),
        "status": state.get("status"),
        "output_bundle": bundle_counts,
        "review_checks": [
            {
                "label": check["label"],
                "ok": bool(check["ok"]),
                "status": check.get("status"),
            }
            for check in _summary_review_checks(state, lang)
        ],
        "evidence": [
            {
                "kind": rec.get("kind") or rec.get("tag"),
                "label": rec.get("label") or rec.get("title"),
                "relative_path": rec.get("relative_path"),
                "sha": rec.get("sha") or rec.get("sha256"),
            }
            for rec in (state.get("evidence") or [])
            if isinstance(rec, dict)
        ],
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    encoded = base64.b64encode(data).decode("ascii")
    filename = _compact_label(state.get("run_id") or "easyicu_summary", max_len=48)
    return f"data:application/json;base64,{encoded}", f"{filename}_bundle_index.json"


def _output_summary_html(state: dict[str, Any], lang: str) -> str:
    audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    is_demo = bool(state.get("is_demo"))
    bundle_counts = _summary_bundle_counts(state)
    checks = _summary_review_checks(state, lang)
    passed = sum(1 for check in checks if check["ok"] is True)
    total = len(checks)
    has_blocker = passed < total
    pending_count = max(0, total - passed)
    pending_labels = [str(check.get("label") or "") for check in checks if check.get("ok") is not True]
    only_reviewer_pending = pending_count == 1 and any("Reviewer sign-off" in label for label in pending_labels)
    error_count = int(counts.get("errors") or 0)
    warning_count = int(counts.get("warnings") or 0)
    draft_title = (
        _T(lang, "Manuscript draft is locked until checks pass", "证据检查通过前锁定手稿草稿")
        if has_blocker else
        _T(lang, "Manuscript draft can move to reviewer sign-off", "手稿草稿可进入审核签字")
    )
    draft_status = (
        _T(lang, "One reviewer sign-off outstanding", "仍需一位审核者确认")
        if has_blocker and only_reviewer_pending else
        _T(
            lang,
            f"{pending_count} review checks outstanding",
            f"仍有 {pending_count} 项复核检查待确认",
        )
        if has_blocker else
        _T(lang, "Reviewer gate ready", "审核关口就绪")
    )
    draft_detail = (
        _T(
            lang,
            "Resolve the pending checks before preparing methods and results.",
            "请先处理待确认检查，再准备方法与结果草稿。",
        )
        if has_blocker else
        _T(lang, "The draft can now be prepared from logged artifacts.", "现在可以基于已记录产物生成草稿。")
    )
    findings_detail = _T(
        lang,
        f"{error_count} error(s) · {warning_count} warning(s)",
        f"{error_count} 个错误 · {warning_count} 个警告",
    )
    gate_rows = []
    for check in checks:
        ok = check["ok"] is True
        status_text = str(
            check.get("status")
            or (_T(lang, "passed", "通过") if ok else _T(lang, "pending", "待确认"))
        )
        gate_rows.append(
            f'<div class="eu-summary-check-row {"passed" if ok else "pending"}">'
            '<span></span>'
            f'<b>{_esc(str(check["label"]))}</b>'
            f'<em>{_esc(status_text)}</em>'
            '</div>'
        )
    if is_demo:
        bundle_rows = [
            (_T(lang, "Figure slot", "图件槽位"), _T(lang, "not generated in demo", "Demo 中不生成"), "figure"),
            (_T(lang, "Table slot", "表格槽位"), _T(lang, "not generated in demo", "Demo 中不生成"), "table"),
            (_T(lang, "Evidence ledger", "证据 ledger"), _T(lang, "requires a real manifest", "需要真实 manifest"), "evidence"),
            (_T(lang, "Repro code", "复现代码"), _T(lang, "no files written in demo", "Demo 中不写文件"), "code"),
        ]
    else:
        bundle_rows = [
            (
                _T(lang, f"{bundle_counts['figures']} figures", f"{bundle_counts['figures']} 个图件"),
                _T(lang, "registered figure artifacts", "已注册图件产物"),
                "figure",
            ),
            (
                _T(lang, f"{bundle_counts['tables']} tables", f"{bundle_counts['tables']} 张表格"),
                _T(lang, "CSV / table evidence", "CSV / 表格证据"),
                "table",
            ),
            (
                _T(lang, "Evidence ledger", "证据 ledger"),
                _T(lang, f"{bundle_counts['evidence']} manifest rows", f"{bundle_counts['evidence']} 条 manifest 记录"),
                "evidence",
            ),
            (
                _T(lang, "Repro code", "复现代码"),
                _T(lang, f"{bundle_counts['code']} code artifacts", f"{bundle_counts['code']} 个代码产物"),
                "code",
            ),
        ]
    bundle_html = "".join(
        '<div class="eu-summary-bundle-row">'
        f'<span class="eu-summary-bundle-ico">{_summary_bundle_icon(icon)}</span>'
        '<div>'
        f'<b>{_esc(title)}</b>'
        f'<p>{_esc(detail)}</p>'
        '</div>'
        '</div>'
        for title, detail, icon in bundle_rows
    )
    bundle_href, bundle_filename = _summary_bundle_index_download(state, lang)
    demo_notice = (
        f'<div class="eu-summary-demo-note">{_esc(state.get("demo_notice", ""))}</div>'
        if is_demo and state.get("demo_notice") else ""
    )
    bundle_action = (
        f'<span class="eu-summary-bundle-button disabled">{_T(lang, "Real run required", "需要真实运行")}</span>'
        if is_demo else
        f'<a class="eu-summary-bundle-button" href="{_esc(bundle_href)}" download="{_esc(bundle_filename)}">{_T(lang, "Export bundle index", "导出 bundle 索引")}</a>'
    )

    return (
        '<div class="eu-summary-page eu-summary-reference">'
        f'{demo_notice}'
        '<div class="eu-summary-reference-grid">'
        '<div class="eu-summary-review eu-summary-review-main">'
        f'<div class="eu-section-label">{_T(lang, "Review gate", "复核关口")}</div>'
        '<div class="eu-summary-gate-head">'
        f'<b>{_esc(draft_title)}</b>'
        f'<p>{_T(lang, "Drafting is intentionally a second-stage action. The draft stays evidence-bound — every claim traces to a logged artifact.", "草稿生成是第二阶段动作；草稿保持证据绑定，每条主张都追溯到已记录产物。")}</p>'
        '</div>'
        f'<div class="eu-summary-checklist">{"".join(gate_rows)}</div>'
        f'<div class="eu-summary-manuscript {"locked" if has_blocker else "ready"}">'
        f'<span class="mono">{passed} / {total} {_T(lang, "checks", "项检查")}</span>'
        '<div>'
        f'<b>{_esc(draft_status)}</b>'
        f'<p>{_esc(draft_detail)} {_esc(findings_detail)}</p>'
        '</div>'
        '<div class="eu-summary-manuscript-actions">'
        f'<span class="eu-summary-action-token disabled">{_T(lang, "Decline", "退回")}</span>'
        f'<span class="eu-summary-action-token {"disabled" if has_blocker else "ready"}">{_T(lang, "Draft methods + results", "生成方法与结果草稿")}</span>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="eu-summary-bundle">'
        f'<div class="eu-section-label">{_T(lang, "Output bundle", "输出包")}</div>'
        f'<div class="eu-summary-bundle-list">{bundle_html}</div>'
        f'{bundle_action}'
        '</div>'
        '</div>'
        '</div>'
    )


def _summary_empty_html(lang: str) -> str:
    checks = [
        _T(lang, "Real run manifest selected", "已选择真实运行 manifest"),
        _T(lang, "Cohort denominators resolved", "队列分母已确认"),
        _T(lang, "Evidence manifest attached", "证据 manifest 已绑定"),
        _T(lang, "Tables and figures registered", "表格与图件已注册"),
        _T(lang, "Reviewer sign-off", "审核者签字"),
    ]
    gate_rows = "".join(
        '<div class="eu-summary-check-row pending">'
        '<span></span>'
        f'<b>{_esc(label)}</b>'
        f'<em>{_T(lang, "pending", "待确认")}</em>'
        '</div>'
        for label in checks
    )
    bundle_rows = [
        (_T(lang, "Study plan", "研究计划"), _T(lang, "not generated", "尚未生成"), "P"),
        (_T(lang, "Cohort table", "队列表格"), _T(lang, "not generated", "尚未生成"), "C"),
        (_T(lang, "Tables + figures", "表格 + 图件"), _T(lang, "not generated", "尚未生成"), "T"),
        (_T(lang, "Evidence manifest", "证据 manifest"), _T(lang, "not attached", "尚未绑定"), "E"),
        (_T(lang, "Optional draft", "可选草稿"), _T(lang, "locked until review", "复核前锁定"), "D"),
    ]
    bundle_html = "".join(
        '<div class="eu-summary-bundle-row">'
        f'<span class="eu-summary-bundle-ico mono">{_esc(icon)}</span>'
        '<div>'
        f'<b>{_esc(title)}</b>'
        f'<p>{_esc(detail)}</p>'
        '</div>'
        '</div>'
        for title, detail, icon in bundle_rows
    )
    note = (
        _T(
            lang,
            "Demo Mode does not create a fake analysis package. Open a local manifest or run a real analysis to populate this summary.",
            "演示模式不会伪造分析包。请打开本机 manifest 或启动真实分析后再填充这里。",
        )
        if st.session_state.get("entry_mode") == "demo" else
        _T(
            lang,
            "Open a local manifest or run an analysis from Setup to populate this summary.",
            "请从配置页运行分析，或打开本机 manifest 后再填充这里。",
        )
    )
    return (
        '<div class="eu-summary-page eu-summary-reference">'
        f'<div class="eu-summary-demo-note">{_esc(note)}</div>'
        '<div class="eu-summary-reference-grid">'
        '<div class="eu-summary-review eu-summary-review-main">'
        f'<div class="eu-section-label">{_T(lang, "Review gate", "复核关口")}</div>'
        '<div class="eu-summary-gate-head">'
        f'<b>{_T(lang, "No draft until evidence is bound", "证据绑定前不生成草稿")}</b>'
        f'<p>{_T(lang, "Summary is intentionally a second-stage surface. It should explain what is missing before any manuscript-like output appears.", "Summary 是第二阶段界面；在出现类似手稿的输出前，它应明确说明还缺哪些证据。")}</p>'
        '</div>'
        f'<div class="eu-summary-checklist">{gate_rows}</div>'
        '<div class="eu-summary-manuscript locked">'
        f'<span class="mono">0 / {len(checks)} {_T(lang, "checks", "项检查")}</span>'
        '<div>'
        f'<b>{_T(lang, "Draft gate locked", "草稿关口已锁定")}</b>'
        f'<p>{_T(lang, "Run or import a manifest before preparing methods and results text.", "请先运行或导入 manifest，再准备方法与结果文本。")}</p>'
        '</div>'
        '<div class="eu-summary-manuscript-actions">'
        f'<button disabled>{_T(lang, "Draft methods + results", "生成方法与结果草稿")}</button>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="eu-summary-bundle">'
        f'<div class="eu-section-label">{_T(lang, "Output bundle", "输出包")}</div>'
        f'<div class="eu-summary-bundle-list">{bundle_html}</div>'
        f'<span class="eu-summary-bundle-button disabled">{_T(lang, "Bundle index unavailable", "Bundle 索引尚不可用")}</span>'
        '</div>'
        '</div>'
        '</div>'
    )


def _render_summary_empty_state(lang: str, *, show_header: bool = True) -> None:
    if show_header:
        st.markdown(
            cc.render_design_page_header(
                kicker=_T(lang, "Research Agent · summary gate", "研究智能体 · 摘要关口"),
                title_en=_T(lang, "Review gate waiting for a real run", "Review gate waiting for a real run"),
                title_zh=_T(lang, "Review gate waiting for a real run", "复核关口等待真实运行"),
                desc=_T(
                    lang,
                    "The Summary tab stays locked until a run or local manifest provides evidence-bound artifacts.",
                    "Summary 页在真实 run 或本机 manifest 提供证据绑定产物前保持锁定。",
                ),
                right_html=f'<span class="eu-pill">{_T(lang, "No active run", "暂无运行")}</span>',
                lang=lang,
            ),
            unsafe_allow_html=True,
        )
    st.markdown(_summary_empty_html(lang), unsafe_allow_html=True)


def _render_summary_review_controls(state: dict[str, Any], lang: str) -> None:
    """Render real reviewer actions that write the local review decision."""
    if state.get("is_demo") or not state.get("run_dir"):
        return
    run_dir = Path(str(state.get("run_dir")))
    if not run_dir.exists():
        return
    checks = _summary_review_checks(state, lang)
    if not checks:
        return
    non_reviewer_ready = all(check.get("ok") is True for check in checks[:-1])
    reviewer_ready = checks[-1].get("ok") is True
    audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
    review = audit.get("review_decision") if isinstance(audit.get("review_decision"), dict) else {}
    safe_run = re.sub(r"[^A-Za-z0-9_]+", "_", str(state.get("run_id") or run_dir.name))
    note_key = f"_eu_summary_review_note_{safe_run}"
    default_note = str(review.get("note") or "")
    if note_key not in st.session_state:
        st.session_state[note_key] = default_note
    passed = sum(1 for check in checks if check.get("ok") is True)
    total = len(checks)
    if reviewer_ready:
        control_title = _T(lang, "Reviewer sign-off saved", "审核签字已保存")
        control_detail = _T(
            lang,
            f"Decision: {review.get('decision', 'approved')} · {review.get('updated_at', '')}",
            f"决定：{review.get('decision', 'approved')} · {review.get('updated_at', '')}",
        )
        control_tone = "ready"
    elif non_reviewer_ready:
        control_title = _T(lang, "One reviewer sign-off outstanding", "仍需一位审核者签字")
        control_detail = _T(
            lang,
            "The review decision writes to local review_decision.json and unlocks the draft gate.",
            "审核决定会写入本地 review_decision.json，并解锁草稿关口。",
        )
        control_tone = "pending"
    else:
        control_title = _T(lang, "Reviewer sign-off locked", "审核签字仍锁定")
        control_detail = _T(
            lang,
            "Resolve evidence and validator checks before writing local review_decision.json.",
            "请先处理证据与验证器检查，再写入本地 review_decision.json。",
        )
        control_tone = "locked"

    with st.container(key=f"_eu_summary_review_panel_{safe_run}"):
        st.markdown(
            (
                '<div class="eu-summary-review-control-head">'
                f'<span class="mono {control_tone}">{passed} / {total} {_T(lang, "checks", "项检查")}</span>'
                '<div>'
                f'<b>{_esc(control_title)}</b>'
                f'<p>{_esc(control_detail)}</p>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        note = st.text_area(
            _T(lang, "Reviewer note", "审核备注"),
            key=note_key,
            height=68,
            placeholder=_T(
                lang,
                "Optional note for the run audit trail.",
                "可选：写入该 run 的审核记录。",
            ),
            disabled=not non_reviewer_ready,
        )
        approve_col, lock_col = st.columns([1, 1])
        with approve_col:
            if st.button(
                _T(lang, "Sign off review", "签字通过复核"),
                key=f"_eu_summary_review_approve_{safe_run}",
                type="primary",
                use_container_width=True,
                disabled=(not non_reviewer_ready or reviewer_ready),
                help=_T(
                    lang,
                    "Available after non-reviewer evidence checks pass.",
                    "非审核者证据检查通过后可用。",
                ),
            ):
                payload = _write_summary_review_decision(
                    run_dir,
                    decision="approved",
                    note=note or _T(lang, "Approved from Summary gate.", "从 Summary gate 签字通过。"),
                    run_id=state.get("run_id"),
                )
                _sync_review_decision_to_workbench_state(payload, lang=lang)
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "summary"
                st.success(_T(lang, "Review sign-off saved.", "审核签字已保存。"))
                st.rerun()
        with lock_col:
            if st.button(
                _T(lang, "Keep locked", "保持锁定"),
                key=f"_eu_summary_review_lock_{safe_run}",
                use_container_width=True,
                disabled=not non_reviewer_ready,
                help=_T(
                    lang,
                    "Records that the draft should remain locked for this run.",
                    "记录该 run 的草稿仍需保持锁定。",
                ),
            ):
                payload = _write_summary_review_decision(
                    run_dir,
                    decision="locked",
                    note=note or _T(lang, "Reviewer kept the draft gate locked.", "审核者保持草稿关口锁定。"),
                    run_id=state.get("run_id"),
                )
                _sync_review_decision_to_workbench_state(payload, lang=lang)
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "summary"
                st.warning(_T(lang, "Review gate kept locked.", "复核关口已保持锁定。"))
                st.rerun()
        if st.button(
            _T(lang, "Draft methods + results", "生成方法与结果草稿"),
            key=f"_eu_summary_review_draft_{safe_run}",
            type="primary",
            use_container_width=True,
            disabled=not reviewer_ready,
            help=_T(
                lang,
                "Opens Setup in force-manuscript mode using this reviewed run.",
                "使用该已复核 run 打开 Setup 的强制写稿模式。",
            ),
        ):
            _prime_summary_draft_setup(state)
            st.rerun()
        if reviewer_ready:
            st.success(
                _T(
                    lang,
                    f"Saved decision: {review.get('decision', 'approved')} · {review.get('updated_at', '')}",
                    f"已保存决定：{review.get('decision', 'approved')} · {review.get('updated_at', '')}",
                )
            )


def render_agent_output_summary(lang: str, *, show_header: bool = True) -> None:
    """Render the analysis-first Research Agent summary page."""
    state = _resolve_workbench_state(lang)
    if not state.get("steps"):
        _render_summary_empty_state(lang, show_header=show_header)
        return
    state = dict(state)
    state["reviewed_finding_ids"] = sorted(_sync_reviewed_findings_to_session(state))
    state.setdefault("source_label", _T(lang, "Real manifest", "真实 manifest") if not state.get("is_demo") else _T(lang, "Sample workflow", "示例流程"))
    actions = (
        (
            f'<span class="eu-pill">{_T(lang, "Runs", "运行")} · {_T(lang, "preview", "预览")}</span>'
            f'<span class="eu-pill">{_T(lang, "No LLM call", "不调用 LLM")}</span>'
        )
        if state.get("is_demo") else
        (
            f'<span class="eu-pill">{_T(lang, "Runs", "运行")} · {len(state.get("steps") or [])}</span>'
            f'<span class="eu-pill">{_esc(state.get("source_label", ""))}</span>'
        )
    )
    if show_header:
        st.markdown(
            cc.render_design_page_header(
                kicker=_T(lang, "Research Agent · research workflow", "研究智能体 · 研究工作流"),
                title_en="EasyICU Research Agent",
                title_zh="EasyICU Research Agent",
                desc=_T(
                    lang,
                    "An auditable, evidence-bound workflow — plan, run, review, then draft.",
                    "一个可审计、证据绑定的工作流：先计划、运行、复核，再进入草稿。",
                ),
                right_html=actions,
                lang=lang,
            ),
            unsafe_allow_html=True,
        )
    st.markdown(_output_summary_html(state, lang), unsafe_allow_html=True)
    _render_summary_review_controls(state, lang)


def _resolve_workbench_state(lang: str) -> dict[str, Any]:
    existing = st.session_state.get("_agent_workbench")
    if isinstance(existing, dict) and existing.get("steps"):
        if existing.get("is_demo"):
            return {}
        return existing
    return {}


def prime_agent_workbench_state(lang: str) -> None:
    """Compatibility shim; Workbench no longer auto-opens latest runs."""
    existing = st.session_state.get("_agent_workbench")
    if isinstance(existing, dict) and existing.get("steps"):
        if existing.get("is_demo"):
            st.session_state.pop("_agent_workbench", None)
        return


def _workbench_empty_html(lang: str) -> str:
    return (
        '<div class="eu-agent-empty">'
        '<div class="eu-agent-empty-glyph" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="7" height="7" rx="1"/>'
        '<rect x="14" y="3" width="7" height="7" rx="1"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1"/>'
        '<rect x="14" y="14" width="7" height="7" rx="1"/>'
        '</svg>'
        '</div>'
        f'<h2>{_T(lang, "No active run", "暂无当前运行")}</h2>'
        f'<p>{_T(lang, "Open Setup or choose a saved manifest.", "打开配置页，或选择本机保存的 manifest。")}</p>'
        f'<small>{_T(lang, "The Workbench stays empty until real evidence-bound artifacts are selected. Local run history stays on this machine and is opened only when you choose a manifest.", "只有选择真实、证据绑定的产物后，工作台才会填充。本机运行历史只保留在这台机器上，并且只会在你选择 manifest 时打开。")}</small>'
        '</div>'
    )


def _route_to_agent_empty_state_target(view: str) -> None:
    """Keep empty-state actions inside the Research Agent workspace."""
    if st.session_state.get("entry_mode", "none") == "none":
        st.session_state["entry_mode"] = "real"
        st.session_state["use_mock_data"] = False
    st.session_state["_active_main_page"] = "research_agent"
    st.session_state["_ra_view"] = view


def _render_workbench_empty_state(lang: str, *, summary: bool = False, show_header: bool = True) -> None:
    if show_header:
        st.markdown(
            cc.render_design_page_header(
                kicker=_T(lang, "Research Agent", "研究智能体"),
                title_en=_T(lang, "Agent project workspace", "Agent project workspace"),
                title_zh=_T(lang, "Agent project workspace", "智能体项目工作区"),
                desc=_T(
                    lang,
                    "Choose a research question, cohort, and local saved run before reviewing agent outputs.",
                    "先选择研究问题、队列和本机历史运行，再复核智能体输出。",
                ),
                right_html=f'<span class="eu-pill">{_T(lang, "No active run", "暂无运行")}</span>',
                lang=lang,
            ),
            unsafe_allow_html=True,
        )
    with st.container(key=f"eu_wb_empty_panel_{summary}"):
        st.markdown(_workbench_empty_html(lang), unsafe_allow_html=True)
        with st.container(key=f"eu_wb_empty_actions_{summary}"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button(_T(lang, "Open setup", "打开配置"), key=f"_eu_wb_empty_setup_{summary}", type="primary", use_container_width=True):
                    _route_to_agent_empty_state_target("setup")
                    st.rerun()
            with c2:
                if st.button(_T(lang, "Open local saved runs", "查看本机历史运行"), key=f"_eu_wb_empty_history_{summary}", use_container_width=True):
                    _route_to_agent_empty_state_target("history")
                    st.session_state.pop("_research_agent_expand_history", None)
                    st.rerun()


def _step_contract_html(state: dict[str, Any], lang: str) -> str:
    contract = state.get("step_contract") if isinstance(state.get("step_contract"), dict) else {}
    if not contract:
        return ""
    method = contract.get("method") if isinstance(contract.get("method"), dict) else {}

    def rows(items: object, empty: str) -> str:
        out = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            status = _contract_status(item.get("ok") if item.get("ok") in {True, False, None} else None)
            out.append(
                f'<div class="eu-step-contract-row {status}">'
                '<span></span>'
                '<div>'
                f'<b>{_esc(item.get("path", ""))}</b>'
                f'<small>{_esc(item.get("meta", ""))}</small>'
                '</div>'
                '</div>'
            )
        if not out:
            out.append(f'<p class="muted">{_esc(empty)}</p>')
        return "".join(out)

    checkpoints = []
    for item in contract.get("checkpoints") or []:
        if not isinstance(item, dict):
            continue
        status = _contract_status(item.get("ok") if item.get("ok") in {True, False, None} else None)
        checkpoints.append(
            f'<div class="eu-step-checkpoint {status}">'
            '<span></span>'
            '<div>'
            f'<b>{_esc(item.get("label", ""))}</b>'
            f'<small>{_esc(item.get("detail", ""))}</small>'
            '</div>'
            '</div>'
        )
    if not checkpoints:
        checkpoints.append(f'<p class="muted">{_T(lang, "No checkpoints recorded.", "未记录检查点。")}</p>')

    return (
        '<div class="eu-step-contract">'
        '<div class="eu-step-contract-method">'
        f'<span class="mono">{_T(lang, "Method binding", "方法绑定")}</span>'
        f'<b>{_esc(method.get("label", ""))}</b>'
        f'<p>{_esc(method.get("audit", ""))}</p>'
        f'<small class="mono">{_T(lang, "Expected", "预期")} · {_esc(method.get("outputs", ""))}</small>'
        '</div>'
        '<div class="eu-step-contract-grid">'
        '<div>'
        f'<h4>{_T(lang, "Inputs", "输入")}</h4>'
        f'{rows(contract.get("inputs"), _T(lang, "No input contract recorded.", "未记录输入契约。"))}'
        '</div>'
        '<div>'
        f'<h4>{_T(lang, "Outputs", "输出")}</h4>'
        f'{rows(contract.get("outputs"), _T(lang, "No output contract recorded.", "未记录输出契约。"))}'
        '</div>'
        '<div>'
        f'<h4>{_T(lang, "Checkpoints", "检查点")}</h4>'
        f'{"".join(checkpoints)}'
        '</div>'
        '</div>'
        '</div>'
    )


def _review_decisions_html(state: dict[str, Any], lang: str) -> str:
    decisions = [d for d in state.get("review_decisions", []) if isinstance(d, dict)]
    if not decisions:
        return ""
    rows = []
    for d in decisions:
        rows.append(
            f'<div class="eu-review-decision { _esc(d.get("state", "idle")) }">'
            '<span></span>'
            '<div>'
            f'<b>{_esc(d.get("label", ""))}</b>'
            f'<small>{_esc(d.get("detail", ""))}</small>'
            '</div>'
            '</div>'
        )
    return (
        '<div class="eu-review-decision-panel">'
        f'<div class="eu-contract-title mono">{_T(lang, "Reviewer decision", "审核决定")}</div>'
        f'{"".join(rows)}'
        '</div>'
    )


def _audit_tasks_html(state: dict[str, Any], lang: str) -> str:
    tasks = [t for t in state.get("audit_tasks", []) if isinstance(t, dict)]
    if not tasks:
        return ""
    rows = []
    for task in tasks:
        rows.append(
            f'<div class="eu-audit-task { _esc(task.get("tone", "info")) }">'
            '<div>'
            f'<b>{_esc(task.get("title", ""))}</b>'
            f'<p>{_esc(task.get("detail", ""))}</p>'
            '</div>'
            f'<span class="mono">{_esc(task.get("action", ""))}</span>'
            '</div>'
        )
    return (
        '<div class="eu-audit-task-panel">'
        f'<div class="eu-contract-title mono">{_T(lang, "Audit tasks", "审计任务")}</div>'
        f'{"".join(rows)}'
        '</div>'
    )


def _workbench_action_panel_html(state: dict[str, Any], lang: str) -> str:
    panel = st.session_state.get("_eu_wb_action_panel")
    if not panel:
        return ""
    if panel == "summary":
        audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
        counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
        active = state.get("active_step") if isinstance(state.get("active_step"), dict) else {}
        outputs = [o for o in state.get("summary_outputs", []) if isinstance(o, dict)]
        output_html = "".join(
            '<div>'
            f'<span>{_esc(o.get("kind", ""))}</span>'
            f'<b>{_esc(o.get("title", ""))}</b>'
            f'<small>{_esc(o.get("sub", ""))}</small>'
            '</div>'
            for o in outputs[:4]
        )
        evidence = [e for e in state.get("evidence", []) if isinstance(e, dict)]
        evidence_html = "".join(
            '<div>'
            f'<span>{_esc(e.get("tag", ""))}</span>'
            f'<b>{_esc(e.get("label", ""))}</b>'
            f'<small>{_esc(e.get("sub", ""))}</small>'
            '</div>'
            for e in evidence[:4]
        )
        return (
            '<div class="eu-wb-action-panel">'
            f'<b>{_T(lang, "Run summary", "运行摘要")}</b>'
            f'<span class="mono">{_esc(state.get("run_id", ""))}</span>'
            f'<p>{_esc(state.get("subtitle", ""))}</p>'
            '<div class="eu-wb-action-grid">'
            f'<div><span>{_T(lang, "Source", "来源")}</span><b>{_esc(state.get("source_label", ""))}</b></div>'
            f'<div><span>{_T(lang, "Active step", "当前步骤")}</span><b>{_esc(active.get("label", ""))}</b></div>'
            f'<div><span>{_T(lang, "Findings", "发现")}</span><b>{int(counts.get("errors") or 0)} error(s) · {int(counts.get("warnings") or 0)} warning(s)</b></div>'
            '</div>'
            f'<div class="eu-wb-manifest-mini">{output_html}</div>'
            f'<div class="eu-wb-manifest-mini">{evidence_html}</div>'
            f'{_step_contract_html(state, lang)}'
            '</div>'
        )
    if panel == "plan":
        rules = state.get("review_rules") or []
        manifest = state.get("manifest") or []
        contract = state.get("execution_contract") if isinstance(state.get("execution_contract"), dict) else {}
        rule_html = "".join(f'<li>{_esc(rule)}</li>' for rule in rules[:5])
        manifest_html = "".join(
            f'<div><span class="mono">{_esc(row.get("op", ""))}</span>'
            f'<b class="mono">{_esc(row.get("path", ""))}</b>'
            f'<small>{_esc(row.get("note", ""))}</small></div>'
            for row in manifest[:6]
        )
        contract_html = "".join(
            f'<div><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'
            for label, value in [
                (_T(lang, "Cohort", "队列"), contract.get("cohort", "")),
                (_T(lang, "Provider", "模型提供方"), contract.get("provider", "")),
                (_T(lang, "Workdir", "工作目录"), contract.get("workdir", "")),
                (_T(lang, "Gate", "关口"), contract.get("gate", "")),
            ]
        )
        return (
            '<div class="eu-wb-action-panel">'
            f'<b>{_T(lang, "Plan and gate contract", "计划与关口契约")}</b>'
            f'<ul>{rule_html}</ul>'
            f'<div class="eu-wb-action-grid">{contract_html}</div>'
            f'<div class="eu-wb-manifest-mini">{manifest_html}</div>'
            f'{_step_contract_html(state, lang)}'
            '</div>'
        )
    if panel == "pause":
        return (
            '<div class="eu-wb-action-panel">'
            f'<b>{_T(lang, "Display paused", "显示已暂停")}</b>'
            f'<p>{_T(lang, "The selected run is kept in view; live pipeline control remains in Setup.", "当前 run 保持在视图中；真实 pipeline 控制仍在配置页。")}</p>'
            '</div>'
        )
    return ""


def _review_gate_html(state: dict[str, Any], lang: str) -> str:
    """Compact PlanAgent-style execution / review ledger."""
    steps = state.get("steps", [])
    results = state.get("results", [])
    evidence = state.get("evidence", [])
    manifest = state.get("manifest", [])
    rules = state.get("review_rules", [])
    ok_steps = sum(1 for s in steps if s.get("status") == "ok")
    running = next((s for s in steps if s.get("status") == "running"), None)
    gate_rows = [
        (
            _T(lang, "Inputs", "输入"),
            _T(lang, "cohort + concept map", "队列 + 概念映射"),
            "ok",
        ),
        (
            _T(lang, "Execution", "执行"),
            f"{ok_steps}/{len(steps)} " + _T(lang, "steps passed", "步骤通过"),
            "warn" if running else "ok",
        ),
        (
            _T(lang, "Outputs", "输出"),
            f"{len(results)} " + _T(lang, "results staged", "结果已暂存"),
            "ok" if results else "warn",
        ),
        (
            _T(lang, "Review gate", "复核关口"),
            f"{len(evidence)} " + _T(lang, "evidence links", "证据链接"),
            "warn" if running else "ok",
        ),
    ]
    cards = []
    for label, value, tone in gate_rows:
        cards.append(
            f'<div class="eu-gate-card {tone}">'
            f'<div class="k">{_esc(label)}</div>'
            f'<div class="v">{_esc(value)}</div>'
            '</div>'
        )
    manifest_rows = "".join(
        '<div class="eu-manifest-row">'
        f'<span class="op mono">{_esc(row.get("op", ""))}</span>'
        f'<span class="path mono">{_esc(row.get("path", ""))}</span>'
        f'<span class="note">{_esc(row.get("note", ""))}</span>'
        '</div>'
        for row in manifest[:6]
    )
    rule_rows = "".join(
        '<div class="eu-review-rule">'
        '<span></span>'
        f'<p>{_esc(rule)}</p>'
        '</div>'
        for rule in rules[:4]
    )
    current = running.get("label") if running else _T(lang, "Ready for review", "等待复核")
    return (
        '<div class="eu-agent-gate">'
        '<div class="eu-agent-gate-head">'
        f'<span class="mono">{_T(lang, "Execution ledger", "执行账本")}</span>'
        f'<span class="mono muted">{_esc(current)}</span>'
        '</div>'
        f'<div class="eu-agent-gate-grid">{"".join(cards)}</div>'
        '<div class="eu-agent-contract">'
        '<div class="eu-contract-col">'
        f'<div class="eu-contract-title mono">{_T(lang, "Preflight contract", "执行前契约")}</div>'
        f'<div class="eu-manifest-list">{manifest_rows}</div>'
        '</div>'
        '<div class="eu-contract-col">'
        f'<div class="eu-contract-title mono">{_T(lang, "Review rules", "复核规则")}</div>'
        f'<div class="eu-review-rule-list">{rule_rows}</div>'
        '</div>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------
# Real interactive helpers — promote decorative HTML to working callbacks
# (2026-05-25 wired-vs-decorative audit fix for items 1-5).
# ---------------------------------------------------------------------

def _live_code_only_html(state: dict[str, Any], lang: str) -> str:
    """Code + autopatch + line gutter. Used by the Code tab."""
    autopatch = state.get("autopatch")
    autopatch_html = ""
    if autopatch:
        autopatch_html = (
            '<div style="margin:8px 0 10px;padding:8px 10px;border-radius:var(--r-2);'
            'background:var(--warn-soft);border:1px solid oklch(86% 0.05 75);'
            'display:flex;align-items:flex-start;gap:10px;font-size:12px;color:oklch(30% 0.10 75)">'
            '<span style="margin-top:1px">✦</span>'
            '<div style="flex:1">'
            f'<div style="font-weight:500">{_esc(autopatch.get("ago", ""))}</div>'
            '<div style="margin-top:2px">'
            f'<span class="mono" style="color:var(--bad);text-decoration:line-through">{_esc(autopatch.get("from", ""))}</span>'
            ' → '
            f'<span class="mono" style="color:var(--ok)">{_esc(autopatch.get("to", ""))}</span>'
            '</div></div></div>'
        )
    code = state.get("code", "") or ""
    code_lines = code.split("\n")
    gutter = "".join(f'<div style="line-height:18px">{i + 1}</div>' for i in range(len(code_lines)))
    code_html = "".join(
        '<div style="line-height:18px;min-height:18px;white-space:pre">'
        f'{_highlight_python_html(line) if line else "&nbsp;"}'
        '</div>'
        for line in code_lines
    )
    return (
        '<div style="padding:8px 12px">'
        + autopatch_html
        + '<div style="display:flex;overflow:auto;max-height:520px">'
        '<div class="mono" style="flex:none;width:32px;padding:4px 6px 4px 4px;font-size:11px;'
        'color:var(--ink-4);text-align:right;border-right:1px solid var(--hair)">'
        + gutter
        + '</div>'
        '<div class="mono" style="margin:0;padding:4px 12px;flex:1;min-width:0;font-size:11.5px;'
        'line-height:18px;background:transparent;color:var(--ink);overflow:visible">'
        + code_html
        + '</div></div></div>'
    )


def _live_trace_block_html(trace: list, lang: str, *, level_filter: set | None = None) -> str:
    trace_color = {"ok": "#B8D7A3", "info": "#9BD2F4", "warn": "#FFC580", "err": "#F4A6A6"}
    rows = []
    for t in trace or []:
        if level_filter and t.get("level") not in level_filter:
            continue
        rows.append(
            f'<div><span style="color:#7A8A99">{_esc(t.get("t", ""))}</span>  '
            f'<span style="color:{trace_color.get(t.get("level", "ok"), "#B8D7A3")}">'
            f'[{_esc(t.get("level", "ok"))}]</span>  '
            f'{_esc(t.get("msg", ""))}</div>'
        )
    if not rows:
        empty = _T(lang, "No log entries.", "暂无日志。")
        return (
            '<div class="mono" style="padding:12px;background:var(--ink);color:var(--ink-4);'
            'border-radius:var(--r-3);font-size:11px;text-align:center">'
            f'{_esc(empty)}</div>'
        )
    return (
        '<div class="mono" style="margin:8px 12px;background:var(--ink);color:#E8E6DD;'
        'border-radius:var(--r-3);padding:10px 12px;font-size:11px;line-height:1.55;'
        'max-height:520px;overflow:auto">'
        + "".join(rows)
        + '</div>'
    )


def _live_history_block_html(full_state: dict[str, Any], lang: str) -> str:
    steps = [s for s in full_state.get("steps", []) if isinstance(s, dict)]
    if not steps:
        return (
            '<div style="padding:16px;color:var(--ink-4);font-size:12px;text-align:center">'
            f'{_esc(_T(lang, "No steps recorded yet.", "尚未记录步骤。"))}</div>'
        )
    rows = []
    for i, s in enumerate(steps):
        status = str(s.get("status") or "pending")
        rows.append(
            '<div style="display:grid;grid-template-columns:34px 1fr auto;gap:10px;'
            'align-items:center;padding:8px 12px;border-top:'
            + ("1px solid var(--hair)" if i else "0")
            + '">'
            f'<span class="mono" style="font-size:11px;color:var(--ink-4)">{i + 1:02d}</span>'
            f'<div style="min-width:0">'
            f'<div style="font-size:12px;font-weight:500;color:var(--ink);'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
            f'{_esc(_compact_label(s.get("label") or s.get("step_id") or f"step {i + 1}"))}</div>'
            f'<div class="mono" style="font-size:10.5px;color:var(--ink-4)">{_esc(_agent_ref_step_meta(s, lang))}</div>'
            '</div>'
            f'<span class="mono" style="font-size:10.5px;color:var(--ink-3)">{_esc(_step_status_label(status, lang))}</span>'
            '</div>'
        )
    return '<div style="padding:0">' + "".join(rows) + '</div>'


# ---------------------------------------------------------------------
# Visual-fidelity helpers (2026-05-26)
#
# Three small helpers that close the remaining design gaps vs.
# page-agent-workbench.jsx without changing any wired behavior:
#   _process_minimap_svg_html : retry-branch DAG mini-map (left column)
#   _highlight_python_html    : oklch-tokenised code lines (center column)
#   _evidence_icon_svg        : 13×13 tag icons for the right column list
# ---------------------------------------------------------------------

_MINIMAP_STATUS_COLOR = {
    "ok": "var(--ink)",
    "fail": "var(--bad)",
    "retry": "var(--warn)",
    "running": "var(--accent)",
    "pending": "var(--ink-4)",
}


def _process_minimap_svg_html(steps: list[dict[str, Any]], lang: str) -> str:
    """Compact read-only DAG of step statuses with a recovery-branch curve.

    The left column's button rail still drives selection. This panel sits
    above it and brings back the .jsx visual narrative — a vertical spine
    with status dots, and dashed Bezier curves whenever a ``fail`` step is
    immediately followed by a ``retry`` / ``running`` / ``ok`` recovery
    (the typical repair/re-run arc).
    """
    if not steps:
        return ""
    # Cap render to keep the panel a fixed visual size; the button rail
    # below still handles the full list.
    visible = steps[:12]
    n = len(visible)
    top = 14
    spacing = 22
    height = top + spacing * (n - 1) + 28 if n else 60
    spine_x = 22
    dots = []
    branches = []
    labels = []
    for i, s in enumerate(visible):
        status = str(s.get("status") or "pending")
        color = _MINIMAP_STATUS_COLOR.get(status, "var(--ink-4)")
        y = top + i * spacing
        # connector segment (skip first)
        if i > 0:
            prev_status = str(visible[i - 1].get("status") or "pending")
            # Solid for normal flow, dashed if previous was a failed step
            # being recovered.
            dashed = prev_status == "fail" and status in {"retry", "running", "ok"}
            if dashed:
                # Curved retry branch: spine → offset → spine
                branches.append(
                    f'<path d="M {spine_x} {y - spacing} '
                    f'C {spine_x + 26} {y - spacing + 4}, '
                    f'{spine_x + 26} {y - 4}, {spine_x} {y}" '
                    f'stroke="var(--warn)" stroke-width="1.2" fill="none" '
                    f'stroke-dasharray="3 3" />'
                )
            else:
                branches.append(
                    f'<line x1="{spine_x}" y1="{y - spacing + 6}" '
                    f'x2="{spine_x}" y2="{y - 6}" '
                    f'stroke="var(--hair-2)" stroke-width="1" />'
                )
        # status dot
        pulse_cls = ' class="eu-pulse"' if status == "running" else ""
        dots.append(
            f'<circle cx="{spine_x}" cy="{y}" r="4.5" '
            f'fill="{"transparent" if status == "pending" else color}" '
            f'stroke="{color}" stroke-width="1.2"{pulse_cls} />'
        )
        # inline checkmark / × / ↻ glyph on filled dots
        if status == "ok":
            dots.append(
                f'<path d="M {spine_x - 2.4} {y} L {spine_x - 0.4} {y + 2} '
                f'L {spine_x + 2.6} {y - 2.2}" stroke="#fff" stroke-width="1.2" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round" />'
            )
        elif status == "fail":
            dots.append(
                f'<path d="M {spine_x - 2.2} {y - 2.2} L {spine_x + 2.2} {y + 2.2} '
                f'M {spine_x + 2.2} {y - 2.2} L {spine_x - 2.2} {y + 2.2}" '
                f'stroke="#fff" stroke-width="1.2" />'
            )
        elif status == "retry":
            dots.append(
                f'<text x="{spine_x}" y="{y + 2.4}" font-size="7" fill="#fff" '
                f'text-anchor="middle" font-family="var(--font-mono)">↻</text>'
            )
        # short label to the right
        raw_label = s.get("label") or s.get("step_id") or f"step {i + 1}"
        label = _compact_label(raw_label, max_len=22)
        active_weight = 500 if status in {"running", "fail"} else 400
        labels.append(
            f'<text x="{spine_x + 12}" y="{y + 3}" font-size="9.5" '
            f'fill="var(--ink-2)" font-family="var(--font-sans)" '
            f'font-weight="{active_weight}">'
            f'{_esc(label)}</text>'
        )
    overflow_note = ""
    if len(steps) > n:
        overflow_note = (
            f'<text x="{spine_x + 12}" y="{top + spacing * n + 6}" '
            f'font-size="9" fill="var(--ink-4)" font-family="var(--font-mono)">'
            f'+ {len(steps) - n} more</text>'
        )
        height += 16
    title = _T(lang, "Analysis flow", "分析流程")
    sub = _T(lang, "real run steps · repair trail", "真实步骤 · 修复轨迹")
    return (
        '<div class="eu-agent-minimap" '
        'style="padding:8px 10px 6px;margin-bottom:6px;'
        'background:var(--surface);border:1px solid var(--hair);'
        'border-radius:var(--r-2)">'
        '<div style="display:flex;justify-content:space-between;'
        'align-items:baseline;margin-bottom:2px">'
        f'<div class="mono" style="font-size:10px;color:var(--ink-4);'
        f'letter-spacing:0.06em;text-transform:uppercase">{_esc(title)}</div>'
        f'<div class="mono" style="font-size:9.5px;color:var(--ink-4)">{_esc(sub)}</div>'
        '</div>'
        f'<svg width="100%" height="{height}" viewBox="0 0 240 {height}" '
        'preserveAspectRatio="xMinYMin meet" style="display:block">'
        + "".join(branches)
        + "".join(dots)
        + "".join(labels)
        + overflow_note
        + '</svg>'
        '</div>'
    )


# Python syntax highlighting -----------------------------------------------
# Re-creates the .jsx Tok colors. Single-line regex tokenizer; deliberately
# minimal (we just want visual differentiation, not a full lexer).

_PY_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
    "pass", "raise", "return", "try", "while", "with", "yield", "True",
    "False", "None",
}
_PY_BUILTINS = {
    "len", "range", "print", "int", "float", "str", "list", "dict", "set",
    "tuple", "open", "enumerate", "zip", "map", "filter", "sorted", "min",
    "max", "sum", "abs", "round", "bool", "isinstance", "type",
}
# Color tokens — match the .jsx oklch constants exactly.
_TOK_C_KEY = "oklch(75% 0.10 80)"
_TOK_C_STR = "oklch(75% 0.10 145)"
_TOK_C_FN = "oklch(75% 0.10 220)"
_TOK_C_NUM = "oklch(80% 0.08 30)"
_TOK_C_CMT = "oklch(60% 0.02 240)"

# One alternation regex with named groups; first match wins.
_PY_TOKEN_RE = re.compile(
    r"(?P<cmt>\#[^\n]*)"
    r"|(?P<str>(?:[rRbBuUfF]{0,2})(?:'''.*?'''|\"\"\".*?\"\"\"|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"))"
    r"|(?P<num>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)"
    r"|(?P<id>[A-Za-z_][A-Za-z_0-9]*)",
    re.DOTALL,
)


def _highlight_python_html(line: str) -> str:
    """Tokenize one line of Python and return color-spanned HTML.

    Strings that span lines (triple-quoted blocks) will only highlight
    correctly inside the line where the opener / closer appears; this is
    a deliberate trade-off to keep the renderer line-stateless.
    """
    if not line:
        return "&nbsp;"
    out: list[str] = []
    pos = 0
    for m in _PY_TOKEN_RE.finditer(line):
        if m.start() > pos:
            out.append(_esc(line[pos:m.start()]))
        kind = m.lastgroup
        text = m.group()
        if kind == "cmt":
            out.append(f'<span style="color:{_TOK_C_CMT};font-style:italic">{_esc(text)}</span>')
        elif kind == "str":
            out.append(f'<span style="color:{_TOK_C_STR}">{_esc(text)}</span>')
        elif kind == "num":
            out.append(f'<span style="color:{_TOK_C_NUM}">{_esc(text)}</span>')
        elif kind == "id":
            if text in _PY_KEYWORDS:
                out.append(f'<span style="color:{_TOK_C_KEY};font-weight:500">{_esc(text)}</span>')
            elif text in _PY_BUILTINS:
                out.append(f'<span style="color:{_TOK_C_FN}">{_esc(text)}</span>')
            else:
                # peek next non-space char for "(" → function-call-ish
                tail = line[m.end():m.end() + 2]
                if tail.lstrip().startswith("("):
                    out.append(f'<span style="color:{_TOK_C_FN}">{_esc(text)}</span>')
                else:
                    out.append(_esc(text))
        else:  # pragma: no cover
            out.append(_esc(text))
        pos = m.end()
    if pos < len(line):
        out.append(_esc(line[pos:]))
    return "".join(out)


# Evidence-row icons --------------------------------------------------------
# 13×13 inline SVGs keyed by ``tag``. The 20px column hosting them was the
# strongest type-recognition signal in the .jsx mock; bringing it back keeps
# the list scannable even after rows pile up.

def _evidence_icon_svg(tag: str) -> str:
    paths = {
        "data": (
            '<ellipse cx="6.5" cy="3" rx="5" ry="1.6" />'
            '<path d="M 1.5 3 V 7 C 1.5 7.9 3.7 8.6 6.5 8.6 C 9.3 8.6 11.5 7.9 11.5 7 V 3" />'
            '<path d="M 1.5 7 V 10 C 1.5 10.9 3.7 11.6 6.5 11.6 C 9.3 11.6 11.5 10.9 11.5 10 V 7" />'
        ),
        "paper": (
            '<path d="M 6.5 1.5 L 6.5 7" />'
            '<ellipse cx="6.5" cy="7.5" rx="2.6" ry="1.4" />'
            '<path d="M 3.9 7.5 L 3.9 10 C 3.9 10.7 5 11.3 6.5 11.3 C 8 11.3 9.1 10.7 9.1 10 L 9.1 7.5" />'
        ),
        "code": (
            '<rect x="1.5" y="2" width="10" height="3" rx="0.6" />'
            '<rect x="1.5" y="5.5" width="10" height="3" rx="0.6" />'
            '<rect x="1.5" y="9" width="10" height="2" rx="0.6" />'
        ),
        "test": (
            '<line x1="2" y1="11" x2="2" y2="7" />'
            '<line x1="5" y1="11" x2="5" y2="4" />'
            '<line x1="8" y1="11" x2="8" y2="6" />'
            '<line x1="11" y1="11" x2="11" y2="2" />'
        ),
        "fix": (
            '<path d="M 2 11 L 6 7 L 5 6 L 9 2 L 11 4 L 7 8 L 6 7" />'
            '<circle cx="3" cy="10" r="0.8" />'
        ),
        "paper_": (
            '<rect x="2.5" y="1.5" width="7.5" height="10" rx="0.8" />'
            '<line x1="4" y1="4" x2="8.5" y2="4" />'
            '<line x1="4" y1="6" x2="8.5" y2="6" />'
            '<line x1="4" y1="8" x2="7" y2="8" />'
        ),
    }
    # fall back to file-glyph for unknown tags
    fallback = (
        '<path d="M 3 1.5 H 7.5 L 10 4 V 11.5 H 3 Z" />'
        '<path d="M 7.5 1.5 V 4 H 10" />'
    )
    body = paths.get(tag, fallback)
    return (
        f'<svg width="13" height="13" viewBox="0 0 13 13" fill="none" '
        f'stroke="currentColor" stroke-width="1" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


def _step_review_html(active_state: dict[str, Any], full_state: dict[str, Any], lang: str) -> str:
    """Design-first step review surface shown before raw code."""
    contract = active_state.get("step_contract") if isinstance(active_state.get("step_contract"), dict) else {}
    method = contract.get("method") if isinstance(contract.get("method"), dict) else {}
    checkpoints = [c for c in contract.get("checkpoints") or [] if isinstance(c, dict)]
    results = [r for r in active_state.get("results", []) if isinstance(r, dict)]
    evidence = [e for e in active_state.get("evidence", []) if isinstance(e, dict)]
    trace = [t for t in active_state.get("trace", []) if isinstance(t, dict)]
    active_step = active_state.get("active_step") if isinstance(active_state.get("active_step"), dict) else {}
    step_idx = active_step.get("index")
    steps = [s for s in full_state.get("steps", []) if isinstance(s, dict)]
    selected_step = steps[step_idx] if isinstance(step_idx, int) and 0 <= step_idx < len(steps) else {}
    status = str(selected_step.get("status") or active_state.get("status") or "pending")
    run_id = _compact_label(full_state.get("run_id") or "", max_len=36)
    step_label = _compact_label(
        active_step.get("label")
        or selected_step.get("label")
        or active_state.get("label")
        or active_state.get("step_id")
        or "Step",
        max_len=64,
    )
    subtitle = active_state.get("subtitle_short") or ""

    def ok_label(value: Any) -> tuple[str, str]:
        if value is True:
            return ("ok", _T(lang, "passed", "通过"))
        if value is False:
            return ("blocked", _T(lang, "needs review", "需复核"))
        return ("pending", _T(lang, "pending", "待确认"))

    checkpoint_rows = []
    for c in checkpoints[:6]:
        tone, label = ok_label(c.get("ok"))
        checkpoint_rows.append(
            '<div class="eu-wb-step-check">'
            f'<span class="{tone}"></span>'
            '<div>'
            f'<b>{_esc(c.get("label", ""))}</b>'
            f'<small>{_esc(c.get("detail", label))}</small>'
            '</div>'
            f'<em>{_esc(label)}</em>'
            '</div>'
        )
    if not checkpoint_rows:
        checkpoint_rows.append(
            '<div class="eu-wb-step-check">'
            '<span class="pending"></span><div>'
            f'<b>{_T(lang, "No step checklist", "暂无步骤检查清单")}</b>'
            f'<small>{_T(lang, "This step has not registered audit checkpoints yet.", "该步骤尚未注册审计检查点。")}</small>'
            '</div><em>pending</em></div>'
        )

    output_rows = []
    for r in results[:4]:
        output_rows.append(
            '<div class="eu-wb-step-artifact">'
            f'<span class="mono">{_esc(r.get("kind", ""))}</span>'
            '<div>'
            f'<b>{_esc(r.get("title", ""))}</b>'
            f'<small>{_esc(r.get("sub", ""))}</small>'
            '</div>'
            '</div>'
        )
    if not output_rows:
        output_rows.append(
            '<div class="eu-wb-step-artifact empty">'
            f'<span class="mono">{_T(lang, "none", "无")}</span>'
            '<div>'
            f'<b>{_T(lang, "No rendered output yet", "暂无可渲染产物")}</b>'
            f'<small>{_T(lang, "Use Activity or Script tabs for lower-level details.", "可在活动或脚本标签中查看底层细节。")}</small>'
            '</div>'
            '</div>'
        )

    trace_rows = []
    for t in trace[:4]:
        level = str(t.get("level") or "info")
        trace_rows.append(
            f'<div class="eu-wb-step-note { _esc(level) }">'
            f'<span class="mono">{_esc(t.get("t", ""))}</span>'
            f'<p>{_esc(t.get("msg", ""))}</p>'
            '</div>'
        )

    method_label = method.get("label") or _T(lang, "Review step", "复核步骤")
    method_audit = method.get("audit") or _T(lang, "registered outputs and evidence checks", "注册产物与证据检查")
    method_outputs = method.get("outputs") or _T(lang, "outputs and evidence", "产物与证据")
    trace_html = (
        "".join(trace_rows)
        if trace_rows else
        f"<p>{_T(lang, 'No activity notes recorded for this step.', '该步骤暂无活动记录。')}</p>"
    )
    return (
        '<div class="eu-wb-step-review">'
        '<div class="eu-wb-step-review-head">'
        '<div>'
        f'<div class="eu-section-label">{_T(lang, "Step review", "步骤复核")}</div>'
        f'<h3>{_esc(step_label)}</h3>'
        f'<p>{_esc(subtitle)}</p>'
        '</div>'
        f'<span class="eu-wb-step-status { _esc(status) }">{_esc(_step_status_label(status, lang))}</span>'
        '</div>'
        '<div class="eu-wb-step-method">'
        f'<span class="mono">{_esc(run_id)}</span>'
        f'<b>{_esc(method_label)}</b>'
        f'<p>{_esc(method_audit)}</p>'
        f'<small>{_T(lang, "Outputs", "产物")}: {_esc(method_outputs)}</small>'
        '</div>'
        '<div class="eu-wb-step-review-grid">'
        '<div class="eu-wb-step-card">'
        f'<div class="eu-section-label">{_T(lang, "Evidence checklist", "证据检查")}</div>'
        f'{"".join(checkpoint_rows)}'
        '</div>'
        '<div class="eu-wb-step-card">'
        f'<div class="eu-section-label">{_T(lang, "Registered outputs", "注册产物")}</div>'
        f'{"".join(output_rows)}'
        '</div>'
        '</div>'
        '<div class="eu-wb-step-card compact">'
        f'<div class="eu-section-label">{_T(lang, "Activity notes", "活动记录")}</div>'
        f'{trace_html}'
        '</div>'
        f'<div class="eu-wb-step-footnote">{len(evidence)} {_T(lang, "evidence item(s) linked to the selected step", "条证据已关联到当前步骤")}</div>'
        '</div>'
    )


def _render_code_panel_tabs(active_state: dict[str, Any], full_state: dict[str, Any], lang: str) -> None:
    """Replace the static 4-tab HTML strip in _live_code_html with real st.tabs.

    Item 2 of the workbench wiring audit (2026-05-25).
    """
    trace = active_state.get("trace") or []
    err_trace = [t for t in trace if t.get("level") in {"err", "warn"}]
    steps = [s for s in full_state.get("steps", []) if isinstance(s, dict)]
    tab_labels = [
        f"{_T(lang, 'Review', '复核')}",
        f"{_T(lang, 'Activity', '活动')} · {len(trace)}",
        f"{_T(lang, 'Script', '脚本')}",
        f"{_T(lang, 'Issues', '问题')} · {len(err_trace)}",
        f"{_T(lang, 'All steps', '全部步骤')} · {len(steps)}",
    ]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        st.markdown(_step_review_html(active_state, full_state, lang), unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(_live_trace_block_html(trace, lang), unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(_live_code_only_html(active_state, lang), unsafe_allow_html=True)
    with tabs[3]:
        st.markdown(
            _live_trace_block_html(trace, lang, level_filter={"err", "warn"}),
            unsafe_allow_html=True,
        )
    with tabs[4]:
        st.markdown(_live_history_block_html(full_state, lang), unsafe_allow_html=True)


def _render_evidence_drilldown(
    active_state: dict[str, Any],
    lang: str,
    *,
    key_suffix: str = "",
) -> None:
    """Item 1: real Open / Copy / Show-SHA actions for the right-column evidence.

    The evidence list is still rendered as design-fidelity HTML above; this
    helper sits below it so users can actually inspect what each row points
    to. Selection persists in session_state per step so re-runs do not
    reset the choice.
    """
    evidence = [e for e in active_state.get("evidence", []) if isinstance(e, dict)]
    if not evidence:
        return
    step_id = str(active_state.get("step_id") or active_state.get("run_id") or "wb")
    select_state_key = f"_eu_wb_evidence_pick_{step_id}_{key_suffix}"
    options = list(range(len(evidence)))
    labels = [
        f"{i + 1:02d} · {(e.get('label') or e.get('tag') or 'evidence')[:48]}"
        for i, e in enumerate(evidence)
    ]
    st.markdown(
        '<div class="eu-section-label" style="padding:0;margin:10px 0 4px">'
        f'{_esc(_T(lang, "Inspect evidence", "查看证据"))}</div>',
        unsafe_allow_html=True,
    )
    picked = st.selectbox(
        _T(lang, "Evidence row", "证据行"),
        options=options,
        format_func=lambda i: labels[i],
        key=select_state_key,
        label_visibility="collapsed",
    )
    rec = evidence[picked] if 0 <= picked < len(evidence) else {}
    raw_path = (
        rec.get("relative_path")
        or rec.get("path")
        or rec.get("artifact_path")
        or rec.get("file")
        or ""
    )
    sha = str(rec.get("sha256") or rec.get("sha8") or "")
    ev_id = str(rec.get("evidence_id") or "")

    # path display + copy (st.code adds a built-in copy button)
    if raw_path:
        st.code(str(raw_path), language="text")
    else:
        st.caption(_T(lang, "No path on this evidence record.", "该证据未携带文件路径。"))

    cols = st.columns([1, 1, 1], gap="small")
    with cols[0]:
        if st.button(
            _T(lang, "Open", "打开"),
            key=f"_eu_wb_ev_open_{step_id}_{key_suffix}",
            use_container_width=True,
            disabled=not raw_path,
            help=_T(lang, "Open the selected evidence path from the run directory.", "从 run 目录打开所选证据路径。"),
        ):
            import os
            import subprocess
            import sys
            target = str(raw_path)
            run_dir = active_state.get("run_dir")
            if run_dir and not os.path.isabs(target):
                target = os.path.join(str(run_dir), target)
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", target], check=False)
                elif sys.platform.startswith("linux"):
                    subprocess.run(["xdg-open", target], check=False)
                elif sys.platform == "win32":
                    os.startfile(target)  # type: ignore[attr-defined]
                st.toast(_T(lang, "Opened.", "已打开。"))
            except Exception as exc:  # pragma: no cover - desktop only
                st.warning(f"open failed: {exc}")
    with cols[1]:
        if st.button(
            _T(lang, "SHA", "SHA"),
            key=f"_eu_wb_ev_sha_{step_id}_{key_suffix}",
            use_container_width=True,
            disabled=not sha,
            help=_T(lang, "Show the full checksum for this evidence record.", "查看该证据记录的完整校验值。"),
        ):
            st.session_state[f"_eu_wb_ev_sha_show_{step_id}_{key_suffix}"] = True
    with cols[2]:
        if st.button(
            _T(lang, "Copy ID", "复制 ID"),
            key=f"_eu_wb_ev_id_{step_id}_{key_suffix}",
            use_container_width=True,
            disabled=not ev_id,
            help=_T(lang, "Reveal the evidence identifier so it can be copied.", "显示证据 ID 以便复制。"),
        ):
            st.session_state[f"_eu_wb_ev_id_show_{step_id}_{key_suffix}"] = True

    if st.session_state.get(f"_eu_wb_ev_sha_show_{step_id}_{key_suffix}"):
        st.code(sha, language="text")
    if st.session_state.get(f"_eu_wb_ev_id_show_{step_id}_{key_suffix}"):
        st.code(ev_id, language="text")


def _render_result_downloads(active_state: dict[str, Any], lang: str) -> None:
    """Item 5: per-result download buttons (figure / CSV / artifact).

    The HTML tile grid stays for design fidelity; this row makes the
    artifacts actually accessible.
    """
    results = [r for r in active_state.get("results", []) if isinstance(r, dict)]
    if not results:
        return
    run_dir = active_state.get("run_dir")
    run_dir_path = Path(str(run_dir)) if run_dir else None
    rows_emitted = False
    path_hints_seen = False
    st.markdown(
        '<div class="eu-section-label" style="padding:0;margin:14px 0 4px">'
        f'{_esc(_T(lang, "Download results", "下载结果"))}</div>',
        unsafe_allow_html=True,
    )
    for i, r in enumerate(results):
        raw_path = (
            r.get("artifact_path")
            or r.get("path")
            or r.get("relative_path")
            or r.get("file")
            or ""
        )
        if not raw_path:
            continue
        path_hints_seen = True
        path = Path(str(raw_path))
        if not path.is_absolute() and run_dir_path is not None:
            path = run_dir_path / path
        if not path.exists() or not path.is_file():
            continue
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".tiff": "image/tiff",
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".json": "application/json",
            ".parquet": "application/octet-stream",
            ".md": "text/markdown",
        }.get(suffix, "application/octet-stream")
        try:
            data = _cached_artifact_bytes(_file_fingerprint(path))
        except Exception:
            continue
        if not data:
            continue
        title = r.get("title") or r.get("kind") or path.name
        cols = st.columns([3, 1.4])
        with cols[0]:
            st.markdown(
                f'<div style="font-size:12px;color:var(--ink);font-weight:500">{_esc(title)}</div>'
                f'<div class="mono" style="font-size:10.5px;color:var(--ink-4);'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(path.name)}'
                f' · {len(data) / 1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.download_button(
                _T(lang, "Download", "下载"),
                data=data,
                file_name=path.name,
                mime=mime,
                key=f"_eu_wb_dl_{i}_{path.name}",
                use_container_width=True,
            )
        rows_emitted = True
    if not rows_emitted:
        if active_state.get("is_demo"):
            caption = _T(
                lang,
                "Demo preview does not write downloadable result files. Open a real run to download bound artifacts.",
                "Demo 预览不会写入可下载结果文件。打开真实运行后可下载绑定产物。",
            )
        elif path_hints_seen:
            caption = _T(
                lang,
                "Registered result paths are not available on disk from this run directory.",
                "已注册的结果路径在当前 run 目录下不可用。",
            )
        else:
            caption = _T(
                lang,
                "No downloadable result files are registered for this selected step.",
                "当前步骤没有注册可下载的结果文件。",
            )
        st.caption(caption)


def _render_timeline_jump(state: dict[str, Any], lang: str, select_key: str) -> None:
    """Item 3: jump-to-step segmented radio below the timeline / state track."""
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    if len(steps) < 2:
        return
    options = list(range(len(steps)))
    current = int(st.session_state.get(select_key, 0) or 0)
    if current >= len(steps):
        current = 0

    def _label(i: int) -> str:
        s = steps[i]
        return f"{i + 1:02d}·{_compact_label(s.get('label') or s.get('step_id') or f'step {i + 1}', max_len=14)}"

    st.markdown(
        '<div class="eu-section-label" style="padding:0;margin:14px 0 4px">'
        f'{_esc(_T(lang, "Jump to step", "跳到步骤"))}</div>',
        unsafe_allow_html=True,
    )
    # Keep this widget keyed to the selected step. A stable radio key keeps
    # Streamlit's old value after the step buttons change, which makes the
    # timeline claim step 01 while the detail panel is showing another step.
    picked = st.radio(
        _T(lang, "Step", "步骤"),
        options=options,
        format_func=_label,
        index=current,
        horizontal=True,
        key=f"_eu_wb_timeline_jump_{select_key}_{current}",
        label_visibility="collapsed",
    )
    if picked != current:
        _set_selected_step(select_key, int(picked))
        st.rerun()


def _render_audit_actions(state: dict[str, Any], lang: str, select_key: str) -> None:
    """Item 4: per-finding Open-step + Mark-reviewed actions."""
    acked = set(_sync_reviewed_findings_to_session(state))
    rows = _finding_queue_rows(state, reviewed_ids=acked)
    if not rows:
        return
    stats = _finding_queue_stats(rows)
    linked = stats["linked"]
    unlinked = stats["total"] - linked
    reviewed = stats["reviewed"]
    total = stats["total"]
    progress_label = _T(
        lang,
        f"{reviewed}/{total} reviewed",
        f"已复核 {reviewed}/{total}",
    )
    severity_label = _T(
        lang,
        f"{stats['errors']} error · {stats['warnings']} warning",
        f"{stats['errors']} 错误 · {stats['warnings']} 警告",
    )
    link_label = _T(
        lang,
        f"{linked} linked · {unlinked} manual",
        f"{linked} 可跳转 · {unlinked} 需人工定位",
    )

    st.markdown(
        '<div class="eu-finding-queue-head">'
        '<div>'
        f'<div class="eu-section-label">{_esc(_T(lang, "Finding queue", "校验发现队列"))}</div>'
        f'<b>{_esc(_T(lang, "Review warnings before Summary sign-off", "Summary 签字前复核 warning"))}</b>'
        f'<p>{_esc(_T(lang, "Open linked steps when possible; mark a finding reviewed only after checking the evidence trail.", "能定位到步骤时先打开步骤；确认过证据链后再标记已复核。"))}</p>'
        '</div>'
        '<div class="eu-finding-queue-metrics">'
        f'<span>{_esc(progress_label)}</span>'
        f'<span>{_esc(severity_label)}</span>'
        f'<span>{_esc(link_label)}</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    rerun_needed = False
    for row in rows:
        fid = str(row["review_id"])
        idx = int(row["index"])
        is_acked = bool(row["reviewed"])
        sev = str(row["severity"] or "warning").lower()
        target_idx = row.get("target_index")
        target_label = str(row.get("target_label") or _T(lang, "manual review", "人工定位"))
        review_label = _T(lang, "reviewed", "已复核") if is_acked else _T(lang, "needs review", "待复核")
        target_tone = "linked" if target_idx is not None else "manual"
        with st.container(key=f"_eu_wb_finding_row_{idx}_{fid[:18]}"):
            cols = st.columns([6.3, 1.18, 1.35], gap="small")
            with cols[0]:
                st.markdown(
                    f'<div class="eu-finding-card { _esc(sev) } { "reviewed" if is_acked else "" }">'
                    '<div class="eu-finding-card-dot"></div>'
                    '<div class="eu-finding-card-copy">'
                    '<div class="eu-finding-card-meta">'
                    f'<span class="eu-finding-sev">{_esc(sev)}</span>'
                    f'<span class="mono">{_esc(row.get("validator") or "?")}</span>'
                    f'<span class="eu-finding-target {target_tone}">{_esc(target_label)}</span>'
                    f'<span class="eu-finding-status">{_esc(review_label)}</span>'
                    '</div>'
                    f'<p>{_esc(row.get("message") or "")}</p>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button(
                    _T(lang, "Open step", "打开步骤"),
                    key=f"_eu_wb_finding_open_{idx}_{fid[:32]}",
                    use_container_width=True,
                    disabled=target_idx is None,
                ):
                    _set_selected_step(select_key, int(target_idx))
                    st.session_state["_active_main_page"] = "research_agent"
                    st.session_state["_ra_view"] = "workbench"
                    st.session_state[_REVIEW_DETAILS_EXPANDED_KEY] = True
                    rerun_needed = True
            with cols[2]:
                label = (
                    _T(lang, "Unmark", "取消标记")
                    if is_acked
                    else _T(lang, "Mark reviewed", "标记已查阅")
                )
                if st.button(
                    label,
                    key=f"_eu_wb_finding_ack_{idx}_{fid[:32]}",
                    use_container_width=True,
                ):
                    if is_acked:
                        acked.discard(fid)
                    else:
                        acked.add(fid)
                    _store_reviewed_findings_for_state(state, acked)
                    st.session_state["_active_main_page"] = "research_agent"
                    st.session_state["_ra_view"] = "workbench"
                    st.session_state[_REVIEW_DETAILS_EXPANDED_KEY] = True
                    rerun_needed = True
    if rerun_needed:
        st.rerun()


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------

def render_agent_workbench(lang: str, *, show_header: bool = True) -> None:
    """Render the Research Agent Workbench with a Claude Design-style overview."""
    state = _resolve_workbench_state(lang)
    if not state.get("steps"):
        _render_workbench_empty_state(lang, show_header=show_header)
        return
    # carry a short subtitle into the results column
    state.setdefault("subtitle_short", "")
    state.setdefault("source_label", _T(lang, "Real manifest", "真实 manifest") if not state.get("is_demo") else _T(lang, "Sample workflow", "示例流程"))
    state["reviewed_finding_ids"] = sorted(_sync_reviewed_findings_to_session(state))
    select_key, selected_idx = _resolve_selected_step(state)
    active_state = _state_for_selected_step(state, selected_idx)

    # Header: match the Claude polish reference by keeping the page identity
    # stable and moving the run-specific question into the overview cards.
    arm_label = _T(lang, "ICU-aware · default arm", "ICU-aware · 默认实验臂")
    runs_label = _T(lang, "Runs", "运行")
    actions = (
        (
            f'<span class="eu-pill">{_esc(runs_label)} · {_T(lang, "preview", "预览")}</span>'
            f'<span class="eu-pill">{_T(lang, "No LLM call", "不调用 LLM")}</span>'
        )
        if state.get("is_demo") else
        (
            f'<span class="eu-pill">{_esc(runs_label)} · {len(state.get("steps", []) or [])}</span>'
            f'<span class="eu-pill" title="{_esc(_T(lang, "Web UI runs the ICU-aware arm only; the naive ablation is exposed via the CLI --arms flag.", "Web 端只跑 ICU-aware 实验臂；naive 消融需通过 CLI --arms 显式触发。"))}">'
            f'<span class="dot" style="background:var(--accent)"></span>{_esc(arm_label)}</span>'
        )
    )
    if show_header:
        st.markdown(
            cc.render_design_page_header(
                kicker=_T(lang, "Research Agent · research workflow", "Research Agent · 研究工作流"),
                title_en="EasyICU Research Agent",
                title_zh="EasyICU Research Agent",
                desc=_T(
                    lang,
                    "An auditable, evidence-bound workflow — plan, run, review, then draft.",
                    "可审计、证据绑定的工作流：计划、运行、复核，然后再写作。",
                ),
                right_html=actions,
                lang=lang,
            ),
            unsafe_allow_html=True,
        )

    st.markdown(_agent_reference_workbench_html(state, lang), unsafe_allow_html=True)

    details_expanded = bool(st.session_state.get(_REVIEW_DETAILS_EXPANDED_KEY))
    with st.expander(_T(lang, "Review details", "复核详情"), expanded=details_expanded):
        c1, c2, c3 = st.columns([1, 1, 1], gap="small")
        with c1:
            if st.button(_T(lang, "Summary", "摘要"), key="_eu_wb_summary", use_container_width=True):
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "summary"
                st.rerun()
        with c2:
            if st.button(
                _T(lang, "Run setup", "运行配置"),
                key="_eu_wb_run_controls",
                use_container_width=True,
                help=_T(
                    lang,
                    "Open Setup, where live runs are configured and launched.",
                    "打开配置页；实时 run 在那里配置和启动。",
                ),
            ):
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "setup"
                st.rerun()
        with c3:
            if st.button(
                _T(lang, "Contract", "契约"),
                key="_eu_wb_adjust",
                use_container_width=True,
            ):
                st.session_state["_eu_wb_action_panel"] = "plan"
        panel_html = _workbench_action_panel_html(active_state, lang)
        if panel_html:
            st.markdown(panel_html, unsafe_allow_html=True)

        with st.container(border=True):
            selected_idx = _render_process_graph_controls(
                state,
                lang,
                selected_idx=selected_idx,
                select_key=select_key,
            )
        active_state = _state_for_selected_step(state, selected_idx)
        col_c, col_r = st.columns([1.75, 1], gap="medium")
        with col_c:
            # Item 2 of the workbench wire-up audit: real st.tabs instead of
            # the decorative 4-tab <div> strip that lived inside _live_code_html.
            with st.container(border=True):
                _render_code_panel_tabs(active_state, state, lang)
        with col_r:
            st.markdown(
                '<div class="eu-agent-panel" style="padding:0;overflow:hidden;max-height:620px;overflow-y:auto">'
                + _result_evidence_html(active_state, lang)
                + '</div>',
                unsafe_allow_html=True,
            )
            # Items 1 + 5: real drill-down actions for the right-column cards.
            _render_evidence_drilldown(active_state, lang, key_suffix=str(selected_idx))
            _render_result_downloads(active_state, lang)

        st.markdown(
            '<div class="eu-agent-timeline">' + _state_track_html(state, lang) + '</div>',
            unsafe_allow_html=True,
        )
        _render_timeline_jump(state, lang, select_key)

        audit_html = _audit_review_html(state, lang)
        if audit_html:
            st.markdown(audit_html, unsafe_allow_html=True)
            # Item 4: real Open-step + Mark-reviewed actions per finding.
            _render_audit_actions(state, lang, select_key)


def render_agent_live_workbench(lang: str) -> None:
    """Render a widget-free live Workbench snapshot for in-run refreshes."""
    state = _resolve_workbench_state(lang)
    if not state.get("steps"):
        _render_workbench_empty_state(lang)
        return
    state.setdefault("subtitle_short", "")
    state.setdefault("source_label", _T(lang, "Live run", "实时运行"))
    selected_idx = _default_selected_step([s for s in state.get("steps", []) if isinstance(s, dict)])
    active_state = _state_for_selected_step(state, selected_idx)
    arm_label = _T(lang, "ICU-aware · default arm", "ICU-aware · 默认实验臂")
    actions = (
        f'<span class="eu-pill" title="{_esc(_T(lang, "Web UI runs the ICU-aware arm only; the naive ablation is exposed via the CLI --arms flag.", "Web 端只跑 ICU-aware 实验臂；naive 消融需通过 CLI --arms 显式触发。"))}">'
        f'<span class="dot" style="background:var(--accent)"></span>{_esc(arm_label)}</span>'
        '<span class="eu-pill"><span class="dot eu-pulse" style="background:var(--accent)"></span>'
        f'{_esc(state.get("status_step", ""))}</span>'
    )
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Live Workbench", "实时工作台"),
            title_en=state.get("title", "Research Agent"),
            title_zh=state.get("title", "研究智能体"),
            desc=state.get("subtitle", ""),
            right_html=actions,
            lang=lang,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(_agent_command_strip_html(state, lang), unsafe_allow_html=True)
    st.markdown('<div class="eu-agent-panel-spacer"></div>', unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1.05, 1.7, 1.15], gap="medium")
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    n_ok = sum(1 for s in steps if s.get("status") == "ok")
    n_retry = sum(1 for s in steps if s.get("status") == "retry")
    n_run = sum(1 for s in steps if s.get("status") == "running")
    with col_l:
        st.markdown(
            '<div class="eu-agent-panel" style="padding:14px 16px;min-height:620px;overflow:auto">'
            '<div class="eu-agent-process-head">'
            f'<b>{_T(lang, "Step sequence", "步骤流程")} · {len(steps)} {_T(lang, "steps", "步")}</b>'
            f'<span class="mono">{n_ok} ok · {n_retry} retry · {n_run} running</span>'
            '</div>'
            + _step_flow_html(state, lang, selected_idx=selected_idx)
            + '</div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            '<div class="eu-agent-panel" style="padding:0;overflow:hidden;height:620px">'
            + _live_code_html(active_state, lang)
            + '</div>',
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            '<div class="eu-agent-panel" style="padding:0;overflow:hidden;height:620px">'
            + _result_evidence_html(active_state, lang)
            + '</div>',
            unsafe_allow_html=True,
        )
    with st.expander(_T(lang, "Detailed trace and audit actions", "详细轨迹与复核操作"), expanded=False):
        st.markdown(
            '<div class="eu-agent-timeline">' + _state_track_html(state, lang) + '</div>',
            unsafe_allow_html=True,
        )
        audit_html = _audit_review_html(state, lang)
        if audit_html:
            st.markdown(audit_html, unsafe_allow_html=True)
