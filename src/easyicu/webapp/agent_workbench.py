"""Shell-A redesign · Research Agent live workbench.

Faithful implementation of ``page-agent-workbench.jsx``: a three-column
live run view —

* Left   — process mind-map (DAG of steps with status + retry branch)
* Center — live code panel (tabs + auto-patch banner + code + run trace)
* Right  — result gallery (mini charts) + evidence list
* Bottom — timeline scrubber across all steps

Data binding
------------
The workbench reads its state from ``st.session_state['_agent_workbench']``
when present, and otherwise falls back to a representative snapshot so
the layout is never empty. Use :func:`build_workbench_state_from_manifest`
to bind a real research-agent ``manifest.json`` / ``manifest_partial.json``
into that state.

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
      "step_details": [{"code","trace","results","evidence"}],
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
from pathlib import Path
from typing import Any

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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _result_cards_from_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for record in evidence:
        if not isinstance(record, dict):
            continue
        kind = str(record.get("kind") or "").lower()
        if kind not in {"figure", "table"}:
            continue
        rel = str(record.get("relative_path") or record.get("path") or "")
        cards.append({
            "kind": kind,
            "title": _evidence_label(record),
            "metric": "staged",
            "sub": rel or _evidence_sub(record),
            "svg": cc.render_tile_calibration() if kind == "figure" else "",
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
                return path.read_text(encoding="utf-8")[:7000], str(rel)
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

    add("status", f"{record.get('step_id') or 'step'} · {record.get('status') or 'recorded'}")
    if record.get("intent"):
        add("intent", record.get("intent"), "info")
    if record.get("generation_mode"):
        add("mode", record.get("generation_mode"), "info")
    if record.get("returncode") not in (None, 0, "0"):
        add("returncode", record.get("returncode"), "err")

    summary = record.get("step_summary")
    if isinstance(summary, dict):
        for key in ("n_rows", "n_columns", "target_outcome", "outcome_rate", "error"):
            if summary.get(key) not in (None, "", []):
                add("summary", f"{key}: {summary.get(key)}", "err" if key == "error" else "ok")
        if isinstance(summary.get("plots"), dict):
            add("plots", ", ".join(map(str, summary["plots"].keys())), "ok")
        if isinstance(summary.get("high_missingness_variables"), list) and summary["high_missingness_variables"]:
            add("missing", ", ".join(map(str, summary["high_missingness_variables"][:6])), "warn")
    elif summary:
        add("summary", summary, "ok")

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
        add("evidence", f"{len(step_evidence)} artifact(s) bound to this step", "ok")
    return trace[:10]


def _evidence_rows_from_records(evidence: list[dict[str, Any]], *, fallback_label: str) -> list[dict[str, str]]:
    rows = [
        {
            "label": _evidence_label(record),
            "sub": _evidence_sub(record),
            "tag": _evidence_tag(record),
        }
        for record in evidence[:12]
    ]
    if not rows:
        rows.append({"label": fallback_label, "sub": "no step-specific evidence artifact", "tag": "test"})
    return rows


def _step_detail_from_record(
    *,
    run_path: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    step_number: int,
    total_steps: int,
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
        "results": _result_cards_from_evidence(step_evidence),
        "evidence": _evidence_rows_from_records(step_evidence, fallback_label=step_id),
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
        gate_rows.append({"label": str(name).replace("_", " "), "ok": ok})
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
        repro_bits.append("mock LLM")
    return {
        "partial": partial,
        "counts": counts,
        "gates": gate_rows[:8],
        "findings": findings[:10],
        "reproducibility": " · ".join(repro_bits),
        "run_status": run_status.get("status") or ("partial" if partial else "complete"),
    }


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
    for idx, record in enumerate(records, start=1):
        status = _wb_status(record.get("status"), partial=bool(partial))
        step_id = str(record.get("step_id") or f"step_{idx:02d}")
        steps.append({
            "label": _step_label(step_id),
            "sub": _step_subtitle(record, manifest),
            "status": status,
            "step_id": step_id,
            "record_index": idx - 1,
        })
        step_details.append(_step_detail_from_record(
            run_path=run_path,
            manifest=manifest,
            record=record,
            step_number=idx,
            total_steps=max(len(records), 1),
        ))
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
    result_cards = _result_cards_from_evidence(evidence)
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
        {"key": "staging", "label": _T(lang, "Staging", "准备中"), "desc": _T(lang, "input / plan", "输入/计划")},
        {"key": "running", "label": _T(lang, "Running", "执行中"), "desc": _T(lang, "script active", "脚本运行")},
        {"key": "issue", "label": _T(lang, "Issue", "发现问题"), "desc": _T(lang, "repair / block", "修复/拦截")},
        {"key": "review", "label": _T(lang, "Review", "等待确认"), "desc": _T(lang, "audit gate", "审计关口")},
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

    question = _compact_label(manifest.get("research_question") or run_id, max_len=72)
    subtitle_bits = [run_id, f"{len(steps)} steps", f"{len(evidence)} evidence", f"{len(findings)} findings"]
    return {
        "run_id": run_id,
        "run_dir": str(run_path),
        "title": question or _T(lang, "Research Agent run", "Research Agent 运行"),
        "subtitle": " · ".join(subtitle_bits),
        "subtitle_short": f"{len(evidence)} evidence · {errors}E/{warnings}W",
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
        "state_lanes": state_lanes,
        "state_segments": state_segments,
        "audit": gates,
        "source_label": _T(lang, "Real manifest", "真实 manifest"),
        "is_demo": False,
    }


# ---------------------------------------------------------------------
# Demo fallback state (mirrors the design canvas)
# ---------------------------------------------------------------------

def _demo_state(lang: str) -> dict[str, Any]:
    state = {
        "title": _T(lang, "Sepsis mortality predictors", "脓毒症死亡预测因子"),
        "subtitle": "sepsis_mortality_v3 · 2,481 stays · gpt-oss-20b · seed 42",
        "status": "running",
        "status_step": _T(lang, "running · step 6 of 7", "运行中 · 第 6 / 7 步"),
        "steps": [
            {"label": _T(lang, "Cohort summary", "队列总结"), "sub": "2.1s · n=2,481", "status": "ok"},
            {"label": "Table 1", "sub": _T(lang, "3.4s · 11 features", "3.4s · 11 特征"), "status": "ok"},
            {"label": _T(lang, "Missingness audit", "缺失审计"), "sub": "1.8s · 8.4%", "status": "ok"},
            {"label": "LR · base", "sub": "0.6s · ValueError", "status": "fail"},
            {"label": _T(lang, "Fix: aggregate lactate", "修复:聚合 lactate"), "sub": _T(lang, "auto-patch · 0.4s", "自动修复 · 0.4s"), "status": "retry"},
            {"label": "LR · base (retry)", "sub": "4.2s · AUC 0.815", "status": "ok"},
            {"label": "LR + lactate", "sub": "3.6s · AUC 0.842", "status": "ok"},
            {"label": _T(lang, "ROC + calibration", "ROC + 校准"), "sub": _T(lang, "running… ~1.2s", "运行中… ~1.2s"), "status": "running"},
            {"label": _T(lang, "Findings", "结论"), "sub": _T(lang, "queued", "排队中"), "status": "pending"},
        ],
        "code_path": "easyicu/agent/runs/sepsis_mortality_v3/step_06_roc.py",
        "code": (
            "# auto-generated · seed=42 · step 6 of 7\n\n"
            "from easyicu.research import cohort, model, viz\n"
            "from sklearn.metrics import roc_curve, auc, brier_score_loss\n\n"
            'c = cohort.load("sepsis_mortality_v3")\n'
            'y = c.outcomes["died_hosp"]\n\n'
            "# Build features matching the LR + lactate model\n"
            'X = c.features([\n'
            '    "sofa_max", "age", "map_min",\n'
            '    "lactate_max", "is_sepsis",\n'
            '], window="first_24h", agg="per_stay")\n\n'
            'm = model.load("lr_sepsis_lact_v2")\n'
            "p = m.predict_proba(X)[:, 1]\n\n"
            "fpr, tpr, _ = roc_curve(y, p)\n"
            "auc_val = auc(fpr, tpr)            # -> 0.842\n"
            "brier   = brier_score_loss(y, p)   # -> 0.108\n\n"
            'viz.roc(fpr, tpr, save="04_roc.svg")\n'
            'viz.calibration(y, p, bins=10, save="05_calib.svg")\n'
        ),
        "autopatch": {
            "from": "features['lactate_max']",
            "to": "features.groupby(stay_id)['lactate'].max()",
            "ago": _T(lang, "Auto-patched 1 step ago", "1 步前已自动修复"),
        },
        "trace": [
            {"t": "14:24:11", "msg": "loading cohort sepsis_mortality_v3 (2,481 stays)…", "level": "ok"},
            {"t": "14:24:11", "msg": "building feature matrix [5 cols × 2,481 rows]…", "level": "ok"},
            {"t": "14:24:11", "msg": "loading model lr_sepsis_lact_v2…", "level": "ok"},
            {"t": "14:24:12", "msg": "scoring p̂… 0.842 AUC · 0.108 Brier", "level": "info"},
            {"t": "14:24:12", "msg": "rendering 04_roc.svg…", "level": "ok"},
            {"t": "14:24:12", "msg": "rendering 05_calib.svg…", "level": "warn"},
        ],
        "results": [
            {"kind": "04 · roc", "title": "ROC", "metric": "AUC 0.842",
             "sub": "95% CI 0.81–0.87 · n=2,481", "svg": cc.render_tile_roc()},
            {"kind": "05 · calibration", "title": "Calibration", "metric": _T(lang, "rendering…", "渲染中…"),
             "sub": "Brier 0.108", "svg": cc.render_tile_calibration()},
        ],
        "evidence": [
            {"label": "cohort.parquet", "sub": "2,481 rows · MIMIC-IV demo", "tag": "data"},
            {"label": "labs.parquet", "sub": "lactate · 24,103 rows", "tag": "data"},
            {"label": "sepsis-3 definition", "sub": "Singer 2016 · A41 R65.20", "tag": "paper"},
            {"label": "SOFA components", "sub": "easyicu/concepts/sofa.py", "tag": "code"},
            {"label": "χ² Sepsis × Mortality", "sub": "p < .001, df=1", "tag": "test"},
            {"label": "auto-patch · 14:24:08", "sub": _T(lang, "aggregate lactate fix", "聚合 lactate 修复"), "tag": "fix"},
        ],
        "timeline": [
            {"label": "cohort", "t": 0, "d": 2.1, "status": "ok"},
            {"label": "table1", "t": 2.1, "d": 3.4, "status": "ok"},
            {"label": "missing", "t": 5.5, "d": 1.8, "status": "ok"},
            {"label": "LR base", "t": 7.3, "d": 0.6, "status": "fail"},
            {"label": "fix", "t": 7.9, "d": 0.4, "status": "retry"},
            {"label": "LR retry", "t": 8.3, "d": 4.2, "status": "ok"},
            {"label": "LR + lact", "t": 12.5, "d": 3.6, "status": "ok"},
            {"label": "ROC", "t": 16.1, "d": 1.3, "status": "running"},
            {"label": "findings", "t": 17.4, "d": 2.0, "status": "pending"},
        ],
        "elapsed": 16.1,
        "total": 19.4,
        "playhead": 16.1,
        "tokens": 12408,
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
            {"op": "READ", "path": "cohort.parquet", "note": "2,481 stays"},
            {"op": "READ", "path": "concept_manifest.json", "note": "19 modules"},
            {"op": "WRITE", "path": "results/04_roc.svg", "note": "staged"},
            {"op": "WRITE", "path": "audit/numeric_audit.json", "note": "required"},
        ],
        "review_rules": [
            _T(lang, "numeric claims require evidence refs", "数值主张必须有证据引用"),
            _T(lang, "missingness warning blocks clean clinical claims", "缺失警告阻止干净临床结论"),
            _T(lang, "manuscript draft waits for human confirmation", "主文草稿等待人工确认"),
        ],
        "state_lanes": [
            {"key": "staging", "label": _T(lang, "Staging", "准备中"), "desc": _T(lang, "checks inputs", "检查输入")},
            {"key": "running", "label": _T(lang, "Running", "执行中"), "desc": _T(lang, "script active", "脚本运行")},
            {"key": "issue", "label": _T(lang, "Issue", "发现问题"), "desc": _T(lang, "repair / retry", "修复重试")},
            {"key": "review", "label": _T(lang, "Review", "等待确认"), "desc": _T(lang, "human gate", "人工关口")},
            {"key": "approved", "label": _T(lang, "Approved", "已通过"), "desc": _T(lang, "unlocks draft", "解锁草稿")},
        ],
        "state_segments": [
            {"lane": "staging", "start": 0.0, "end": 2.1, "label": "lock"},
            {"lane": "running", "start": 2.1, "end": 7.3, "label": "tables"},
            {"lane": "issue", "start": 7.3, "end": 7.9, "label": "error"},
            {"lane": "staging", "start": 7.9, "end": 8.3, "label": "patch"},
            {"lane": "running", "start": 8.3, "end": 16.1, "label": "models"},
            {"lane": "review", "start": 16.1, "end": 19.4, "label": "draft gate"},
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
                "# This is sample content. Open a real run to inspect the bound script.\n"
            ),
            "code_path": state["code_path"] if i in {5, 6, 7} else step["step_id"],
            "trace": [
                {"t": "demo", "msg": f"{step['label']} · {step['status']}", "level": "ok" if step["status"] == "ok" else "warn"},
                {"t": "note", "msg": step["sub"], "level": "info"},
            ],
            "results": state["results"] if i in {6, 7} else [],
            "evidence": state["evidence"][:4] if i in {6, 7} else [{"label": step["label"], "sub": "sample step", "tag": "test"}],
            "subtitle_short": f"sample step {i + 1}/{len(steps)}",
            "autopatch": state["autopatch"] if i == 4 else None,
        }
        for i, step in enumerate(steps)
    ]
    state["source_label"] = _T(lang, "Sample workflow", "示例流程")
    state["is_demo"] = True
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
        f'<span style="width:7px;height:7px;border-radius:999px;background:{c}"></span>{l}</span>'
        for l, c in [("ok", "var(--ink)"), ("retry", "var(--warn)"),
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
    return key, selected


def _state_for_selected_step(state: dict[str, Any], selected_idx: int) -> dict[str, Any]:
    details = [d for d in state.get("step_details", []) if isinstance(d, dict)]
    if selected_idx < 0 or selected_idx >= len(details):
        return state
    detail = details[selected_idx]
    view_state = dict(state)
    for key in ("code", "code_path", "trace", "results", "evidence", "subtitle_short", "autopatch"):
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
    n_ok = sum(1 for s in steps if s.get("status") == "ok")
    n_retry = sum(1 for s in steps if s.get("status") == "retry")
    n_run = sum(1 for s in steps if s.get("status") == "running")
    st.markdown(
        '<div class="eu-agent-process-head">'
        f'<b>{_T(lang, "Run queue", "运行队列")} · {len(steps)} {_T(lang, "steps", "步")}</b>'
        f'<span class="mono">{n_ok} ok · {n_retry} retry · {n_run} running</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eu-agent-start mono">START</div>', unsafe_allow_html=True)
    options = list(range(len(steps)))

    def fmt(i: int) -> str:
        step = steps[i]
        return (
            f"{i + 1:02d}  {step.get('label', '')}  · "
            f"{_step_status_label(str(step.get('status') or ''), lang)}  —  "
            f"{_compact_label(step.get('sub'), max_len=46)}"
        )

    if options:
        selected_idx = int(st.radio(
            _T(lang, "Select step", "选择步骤"),
            options,
            index=selected_idx,
            format_func=fmt,
            key=select_key,
            label_visibility="collapsed",
        ))
    if steps:
        active = steps[selected_idx]
        st.markdown(
            '<div class="eu-agent-step-readout">'
            f'<span class="mono">{selected_idx + 1:02d}</span>'
            f'<b>{_esc(active.get("label", ""))}</b>'
            f'<small>{_esc(active.get("sub", ""))}</small>'
            '</div>',
            unsafe_allow_html=True,
        )
    return selected_idx


def _live_code_html(state: dict[str, Any], lang: str) -> str:
    tabs = [
        (_T(lang, "Code", "代码"), True, ""),
        (_T(lang, "Output", "输出"), False, ""),
        (_T(lang, "Errors", "错误"), False, "1"),
        (_T(lang, "History", "历史"), False, str(len(state.get("steps", [])))),
    ]
    tab_html = "".join(
        f'<div style="padding:10px 12px;font-size:12px;'
        f'color:{"var(--ink)" if a else "var(--ink-3)"};'
        f'border-bottom:{"2px solid var(--ink)" if a else "2px solid transparent"};'
        f'margin-bottom:-1px;font-weight:{500 if a else 400};display:flex;align-items:center;gap:6px">'
        f'{_esc(t)}{f"<span class=\"mono\" style=\"font-size:10px;color:var(--ink-4)\">{c}</span>" if c else ""}</div>'
        for t, a, c in tabs
    )

    autopatch = state.get("autopatch")
    autopatch_html = ""
    if autopatch:
        autopatch_html = (
            '<div style="margin:10px 14px 0;padding:10px 12px;border-radius:8px;'
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
        f'{_esc(line) if line else "&nbsp;"}'
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
        'border-radius:8px;padding:10px 12px;font-size:11px;line-height:1.55;max-height:130px;overflow:auto">'
        + trace_html
        + '</div></div></div>'
    )


def _result_evidence_html(state: dict[str, Any], lang: str) -> str:
    results = state.get("results", [])
    result_cards = []
    for r in results:
        result_cards.append(
            '<div class="eu-card" style="padding:10px">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<div class="mono" style="font-size:10px;color:var(--ink-4);letter-spacing:0.06em;'
            f'text-transform:uppercase">{_esc(r["kind"])}</div>'
            f'<span class="mono" style="font-size:11px;color:var(--ink)">{_esc(r["metric"])}</span>'
            '</div>'
            f'<div style="margin-top:4px">{r.get("svg", "")}</div>'
            f'{f"<div class=\"mono\" style=\"font-size:10px;color:var(--ink-4);margin-top:4px\">{_esc(r.get(chr(115)+chr(117)+chr(98), str()))}</div>" if r.get("sub") else ""}'
            '</div>'
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
        ev_rows.append(
            f'<div style="padding:8px 12px;display:grid;grid-template-columns:1fr auto;gap:8px;'
            f'align-items:center;{"border-top:1px solid var(--hair);" if i else ""}">'
            '<div style="min-width:0">'
            f'<div class="mono" style="font-size:11.5px;color:var(--ink);white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{_esc(e["label"])}</div>'
            f'<div style="font-size:10.5px;color:var(--ink-4)">{_esc(e["sub"])}</div>'
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
        f'<span class="mono">{_T(lang, "PlanAgent route", "PlanAgent 路线")}</span>'
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
        f'<b>{_T(lang, "State track", "状态轨迹")}</b>'
        f'<span class="mono">{elapsed:.1f}s / {total:.1f}s · {tokens:,} tok</span>'
        '</div>'
        f'<span class="mono muted">{_T(lang, "PlanAgent lanes, EasyICU queue retained", "PlanAgent 状态轨道 + EasyICU 队列保留")}</span>'
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
    finding_rows = []
    for finding in audit.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        sev = str(finding.get("severity") or "info").lower()
        finding_rows.append(
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
        f'<div class="eu-audit-gates">{"".join(gate_rows) or "<p class=\"muted\">No fail-closed gates recorded.</p>"}</div>'
        f'{repro_html}'
        f'<div class="eu-audit-findings">{"".join(finding_rows[:8]) or "<p class=\"muted\">No validator findings recorded.</p>"}</div>'
        '</div>'
    )


def _manifest_path_for_run(run_dir: Path) -> Path | None:
    final_path = run_dir / "manifest.json"
    partial_path = run_dir / "manifest_partial.json"
    if final_path.exists():
        return final_path
    if partial_path.exists():
        return partial_path
    return None


def _recent_manifest_paths_from_root(root: Path, *, limit: int = 80) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    if root.name.startswith("run_"):
        direct = _manifest_path_for_run(root)
        if direct is not None:
            return [direct]
    try:
        for path in root.rglob("manifest.json"):
            if path.parent.name.startswith("run_"):
                paths.append(path)
        for path in root.rglob("manifest_partial.json"):
            if path.parent.name.startswith("run_") and not (path.parent / "manifest.json").exists():
                paths.append(path)
    except Exception:
        return []
    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[:limit]


def _candidate_manifest_paths() -> list[Path]:
    cwd = Path.cwd()
    roots: list[Path] = []
    workdir = st.session_state.get("research_agent_workdir")
    if workdir:
        roots.append(Path(str(workdir)).expanduser())
    roots.extend([
        cwd / "research_output" / "webapp",
        cwd / "research_output",
        cwd / "pilot_runs",
        cwd.parent / "easyicu写作" / "00_当前投稿_20260516" / "v19_benchmark_runs",
    ])
    seen: set[str] = set()
    paths: list[Path] = []
    for root in roots:
        for path in _recent_manifest_paths_from_root(root):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_real_workbench_state(lang: str) -> dict[str, Any] | None:
    for manifest_path in _candidate_manifest_paths()[:60]:
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        run_dir = manifest_path.parent
        try:
            state = build_workbench_state_from_manifest(
                run_dir,
                manifest,
                lang=lang,
                partial=manifest_path.name == "manifest_partial.json",
            )
        except Exception:
            continue
        st.session_state["_agent_workbench_source_run_dir"] = str(run_dir)
        state["source_label"] = _T(lang, "Real manifest", "真实 manifest")
        state["is_demo"] = False
        return state
    return None


def _resolve_workbench_state(lang: str) -> dict[str, Any]:
    existing = st.session_state.get("_agent_workbench")
    if isinstance(existing, dict) and existing.get("steps"):
        return existing
    latest = _latest_real_workbench_state(lang)
    if latest:
        st.session_state["_agent_workbench"] = latest
        return latest
    return _demo_state(lang)


def prime_agent_workbench_state(lang: str) -> None:
    """Preload a real run so app chrome can label Workbench correctly."""
    existing = st.session_state.get("_agent_workbench")
    if isinstance(existing, dict) and existing.get("steps"):
        return
    latest = _latest_real_workbench_state(lang)
    if latest:
        st.session_state["_agent_workbench"] = latest


def _workbench_action_panel_html(state: dict[str, Any], lang: str) -> str:
    panel = st.session_state.get("_eu_wb_action_panel")
    if not panel:
        return ""
    if panel == "summary":
        audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
        counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
        active = state.get("active_step") if isinstance(state.get("active_step"), dict) else {}
        return (
            '<div class="eu-wb-action-panel">'
            f'<b>{_T(lang, "Run summary", "运行摘要")}</b>'
            f'<span class="mono">{_esc(state.get("run_id", ""))}</span>'
            f'<p>{_esc(state.get("subtitle", ""))}</p>'
            '<div class="eu-wb-action-grid">'
            f'<div><span>{_T(lang, "Source", "来源")}</span><b>{_esc(state.get("source_label", ""))}</b></div>'
            f'<div><span>{_T(lang, "Active step", "当前步骤")}</span><b>{_esc(active.get("label", ""))}</b></div>'
            f'<div><span>{_T(lang, "Findings", "发现")}</span><b>{int(counts.get("errors") or 0)}E / {int(counts.get("warnings") or 0)}W</b></div>'
            '</div>'
            '</div>'
        )
    if panel == "plan":
        rules = state.get("review_rules") or []
        manifest = state.get("manifest") or []
        rule_html = "".join(f'<li>{_esc(rule)}</li>' for rule in rules[:5])
        manifest_html = "".join(
            f'<div><span class="mono">{_esc(row.get("op", ""))}</span>'
            f'<b class="mono">{_esc(row.get("path", ""))}</b>'
            f'<small>{_esc(row.get("note", ""))}</small></div>'
            for row in manifest[:6]
        )
        return (
            '<div class="eu-wb-action-panel">'
            f'<b>{_T(lang, "Plan and gate contract", "计划与关口契约")}</b>'
            f'<ul>{rule_html}</ul>'
            f'<div class="eu-wb-manifest-mini">{manifest_html}</div>'
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
# Public entrypoint
# ---------------------------------------------------------------------

def render_agent_workbench(lang: str) -> None:
    """Render the live agent workbench (3 columns + timeline)."""
    state = _resolve_workbench_state(lang)
    # carry a short subtitle into the results column
    state.setdefault("subtitle_short", "")
    state.setdefault("source_label", _T(lang, "Real manifest", "真实 manifest") if not state.get("is_demo") else _T(lang, "Sample workflow", "示例流程"))
    select_key, selected_idx = _resolve_selected_step(state)
    active_state = _state_for_selected_step(state, selected_idx)

    # Header
    actions = (
        '<span class="eu-pill"><span class="dot eu-pulse" style="background:var(--accent)"></span>'
        f'{_esc(state.get("status_step", ""))}</span>'
    )
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Agent workbench", "实时工作台"),
            title_en=state.get("title", "Research Agent"),
            title_zh=state.get("title", "研究 Agent"),
            desc=state.get("subtitle", ""),
            right_html=actions,
            lang=lang,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(_agent_command_strip_html(state, lang), unsafe_allow_html=True)

    # Real action buttons (Streamlit) so callbacks work
    c1, c2, c3, c4 = st.columns([7.2, 1.7, 1.1, 1.7])
    with c2:
        if st.button(_T(lang, "View summary", "查看摘要"), key="_eu_wb_summary", use_container_width=True):
            st.session_state["_eu_wb_action_panel"] = "summary"
    with c3:
        if st.button(_T(lang, "Pause", "暂停"), key="_eu_wb_pause", use_container_width=True):
            st.session_state["_eu_wb_action_panel"] = "pause"
    with c4:
        if st.button(
            _T(lang, "Adjust plan", "调整计划"),
            key="_eu_wb_adjust",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["_eu_wb_action_panel"] = "plan"
    panel_html = _workbench_action_panel_html(active_state, lang)
    if panel_html:
        st.markdown(panel_html, unsafe_allow_html=True)

    # Three columns
    st.markdown('<div class="eu-agent-panel-spacer"></div>', unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1.05, 1.7, 1.15], gap="medium")
    with col_l:
        with st.container(border=True):
            selected_idx = _render_process_graph_controls(
                state,
                lang,
                selected_idx=selected_idx,
                select_key=select_key,
            )
    active_state = _state_for_selected_step(state, selected_idx)
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

    # Timeline scrubber
    st.markdown(
        '<div class="eu-agent-timeline" style="margin-top:18px">' + _state_track_html(state, lang) + '</div>',
        unsafe_allow_html=True,
    )
    audit_html = _audit_review_html(state, lang)
    if audit_html:
        st.markdown(audit_html, unsafe_allow_html=True)
