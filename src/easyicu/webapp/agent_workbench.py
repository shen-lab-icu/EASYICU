"""Shell-A redesign · Research Agent live workbench.

Faithful implementation of ``page-agent-workbench.jsx``: a three-column
live run view —

* Left   — step sequence (status + retry branch for each pipeline step)
* Center — live code panel (tabs + auto-patch banner + code + run trace)
* Right  — result gallery (mini charts) + evidence list
* Bottom — timeline scrubber across all steps

Data binding
------------
The workbench reads its state from ``st.session_state['_agent_workbench']``
only after a live run, imported manifest, or explicit history selection binds
a real run. Use :func:`build_workbench_state_from_manifest` to bind a real
research-agent ``manifest.json`` / ``manifest_partial.json`` into that state.

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


def _artifact_path_for_preview(run_dir: Path | None, record: dict[str, Any]) -> Path | None:
    raw = record.get("relative_path") or record.get("path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else ((run_dir / path) if run_dir else path)


def _figure_file_preview_html(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
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
                f'<img src="data:{mime};base64,{data}" alt="{_esc(path.name)}" />'
                '</div>'
            )
    except Exception:
        return ""
    return ""


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
        cards.append({
            "kind": kind,
            "title": _evidence_label(record),
            "metric": _T(lang, "rendered", "已渲染") if preview_html else _T(lang, "registered", "已注册"),
            "sub": rel or _evidence_sub(record),
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
        label = _T(lang, "Execution step", "执行步骤")
        audit = _T(lang, "script, log, and registered artifacts", "脚本、日志、注册产物")
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
        rows.append({
            "label": _evidence_label(record),
            "sub": _evidence_sub(record),
            "tag": _evidence_tag(record),
            "sha8": sha[:8] if sha else "",
            "evidence_id": str(record.get("evidence_id") or ""),
        })
    if not rows:
        rows.append({
            "label": fallback_label,
            "sub": "no step-specific evidence artifact",
            "tag": "test",
            "sha8": "",
            "evidence_id": "",
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
        "review_decision": _read_json(run_dir / "review_decision.json"),
    }


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
    if errors or blocked_gates:
        return [
            {
                "label": _T(lang, "Resolve audit blockers", "处理审计阻断"),
                "state": "blocked",
                "detail": _T(
                    lang,
                    f"{errors} error finding(s), {len(blocked_gates)} failed gate(s).",
                    f"{errors} 个 error 级发现，{len(blocked_gates)} 个关口失败。",
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
    if errors or failed_gates:
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
    for gate in audit.get("gates") or []:
        if isinstance(gate, dict) and gate.get("ok") is False:
            label = _compact_label(gate.get("label"), max_len=52)
            tasks.append({
                "title": _T(lang, f"Resolve gate: {label}", f"处理关口: {label}"),
                "detail": _T(lang, "Failed readiness gate blocks promotion.", "失败的 readiness gate 会阻止提升。"),
                "tone": "danger",
                "action": _T(lang, "Inspect", "检查"),
            })
    for finding in audit.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        severity = str(finding.get("severity") or "info").lower()
        if severity not in {"error", "warning"}:
            continue
        validator = _compact_label(finding.get("validator") or "audit", max_len=32)
        message = _compact_label(finding.get("message") or "", max_len=88)
        tasks.append({
            "title": _T(lang, f"{validator}: review finding", f"{validator}: 复核发现"),
            "detail": message,
            "tone": "danger" if severity == "error" else "warning",
            "action": _T(lang, "Open evidence", "打开证据"),
        })
        if len(tasks) >= 6:
            break
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
            lang=lang,
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
    summary_outputs = _summary_outputs_from_manifest(
        manifest=manifest,
        evidence=evidence,
        lang=lang,
    )
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
        "summary_outputs": summary_outputs,
        "execution_contract": execution_contract,
        "review_gate_actions": _review_gate_actions_from_audit(gates, lang=lang),
        "review_decisions": _review_decisions_from_audit(gates, lang=lang),
        "audit_tasks": _audit_tasks_from_audit(gates, lang=lang),
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
        "title": _T(lang, "Research workflow preview", "研究流程预览"),
        "subtitle": _T(lang, "Demo structure only · no cohort loaded · no metrics generated", "仅 Demo 结构 · 未加载队列 · 未生成指标"),
        "status": "preview",
        "status_step": _T(lang, "preview only", "仅预览"),
        "steps": [
            {"label": _T(lang, "Cohort summary", "队列总结"), "sub": _T(lang, "structure slot", "结构槽位"), "status": "ok"},
            {"label": "Table 1", "sub": _T(lang, "table artifact slot", "表格产物槽位"), "status": "ok"},
            {"label": _T(lang, "Missingness audit", "缺失审计"), "sub": _T(lang, "audit slot", "审计槽位"), "status": "ok"},
            {"label": _T(lang, "Model step", "模型步骤"), "sub": _T(lang, "example blocked state", "示例阻断状态"), "status": "fail"},
            {"label": _T(lang, "Repair branch", "修复分支"), "sub": _T(lang, "example retry state", "示例重试状态"), "status": "retry"},
            {"label": _T(lang, "Model rerun", "模型重跑"), "sub": _T(lang, "method slot", "方法槽位"), "status": "ok"},
            {"label": _T(lang, "Comparison step", "比较步骤"), "sub": _T(lang, "result slot", "结果槽位"), "status": "ok"},
            {"label": _T(lang, "Figure export", "图件导出"), "sub": _T(lang, "example running state", "示例运行状态"), "status": "running"},
            {"label": _T(lang, "Findings", "结论"), "sub": _T(lang, "queued", "排队中"), "status": "pending"},
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
            {"label": "blocked", "t": 0.3, "d": 0.1, "status": "fail"},
            {"label": "retry", "t": 0.4, "d": 0.1, "status": "retry"},
            {"label": "rerun", "t": 0.5, "d": 0.1, "status": "ok"},
            {"label": "compare", "t": 0.6, "d": 0.1, "status": "ok"},
            {"label": "figure", "t": 0.7, "d": 0.1, "status": "running"},
            {"label": "findings", "t": 0.8, "d": 0.1, "status": "pending"},
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
            {"key": "staging", "label": _T(lang, "Staging", "准备中"), "desc": _T(lang, "checks inputs", "检查输入")},
            {"key": "running", "label": _T(lang, "Running", "执行中"), "desc": _T(lang, "script active", "脚本运行")},
            {"key": "issue", "label": _T(lang, "Issue", "发现问题"), "desc": _T(lang, "repair / retry", "修复重试")},
            {"key": "review", "label": _T(lang, "Review", "等待确认"), "desc": _T(lang, "human gate", "人工关口")},
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
    sub = _compact_label(step.get("sub"), max_len=42)
    status_label = _step_status_action_label(status, lang)
    if sub:
        return f"{idx + 1:02d}  {label} · {status_label}\n{sub}"
    return f"{idx + 1:02d}  {label}\n{status_label}"


def _step_button_key(state: dict[str, Any], idx: int, status: str, selected: bool) -> str:
    raw = str(state.get("run_id") or "demo")
    safe_run = re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:54]
    safe_status = re.sub(r"[^a-z0-9_]+", "_", status.lower())[:24] or "pending"
    suffix = "selected" if selected else "idle"
    return f"_eu_wb_step_btn_{safe_run}_{idx:02d}_{safe_status}_{suffix}"


def _set_selected_step(select_key: str, idx: int) -> None:
    st.session_state[select_key] = idx


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
    st.markdown(_step_legend_html(lang), unsafe_allow_html=True)

    if steps:
        st.markdown(
            '<div class="eu-agent-step-rail-note mono">'
            f'{_T(lang, "Click a step to inspect its code, outputs, evidence, and review state.", "点击步骤查看对应代码、输出、证据和审阅状态。")}'
            '</div>',
            unsafe_allow_html=True,
        )
    for i, step in enumerate(steps):
        status = str(step.get("status") or "pending")
        is_selected = i == selected_idx
        st.button(
            _step_button_label(step, i, lang),
            key=_step_button_key(state, i, status, is_selected),
            use_container_width=True,
            on_click=_set_selected_step,
            args=(select_key, i),
        )
    return selected_idx


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
            f'<div style="padding:8px 12px;display:grid;grid-template-columns:1fr auto;gap:8px;'
            f'align-items:center;{"border-top:1px solid var(--hair);" if i else ""}">'
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
        '<div class="eu-audit-review-grid">'
        f'{_review_decisions_html(state, lang)}'
        f'{_audit_tasks_html(state, lang)}'
        '</div>'
        f'<div class="eu-audit-gates">{"".join(gate_rows) or "<p class=\"muted\">No fail-closed gates recorded.</p>"}</div>'
        f'{repro_html}'
        f'<div class="eu-audit-findings">{"".join(finding_rows[:8]) or "<p class=\"muted\">No validator findings recorded.</p>"}</div>'
        '</div>'
    )


def _output_summary_html(state: dict[str, Any], lang: str) -> str:
    outputs = [o for o in state.get("summary_outputs", []) if isinstance(o, dict)]
    steps = [s for s in state.get("steps", []) if isinstance(s, dict)]
    actions = [a for a in state.get("review_gate_actions", []) if isinstance(a, dict)]
    contract = state.get("execution_contract") if isinstance(state.get("execution_contract"), dict) else {}
    audit = state.get("audit") if isinstance(state.get("audit"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    is_demo = bool(state.get("is_demo"))

    cohort_stats = [
        (_T(lang, "Cohort", "队列"), contract.get("cohort") or state.get("run_dir") or _T(lang, "not selected", "未选择")),
        (_T(lang, "Provider", "模型"), contract.get("provider") or _T(lang, "recorded in run", "运行中记录")),
        (_T(lang, "Evidence", "证据"), f"{len(state.get('evidence', []) or [])}"),
        (_T(lang, "Findings", "发现"), f"{int(counts.get('errors') or 0)}E / {int(counts.get('warnings') or 0)}W"),
    ]
    cohort_html = "".join(
        '<div class="eu-summary-stat">'
        f'<span>{_esc(label)}</span>'
        f'<b>{_esc(value)}</b>'
        '</div>'
        for label, value in cohort_stats
    )
    concept_chips = [
        _T(lang, "vitals", "生命体征"),
        _T(lang, "labs", "化验"),
        "SOFA",
        _T(lang, "outcomes", "转归"),
        _T(lang, "evidence", "证据"),
    ]
    chips_html = "".join(f'<span class="eu-chip mono">{_esc(chip)}</span>' for chip in concept_chips)

    step_html = "".join(
        f'<span class="eu-summary-step {s.get("status", "pending")}">'
        f'{_esc(_compact_label(s.get("label"), max_len=28))}'
        f'<em>{_esc(_step_status_label(str(s.get("status") or ""), lang))}</em>'
        '</span>'
        for s in steps[:8]
    )
    if not step_html:
        step_html = f'<span class="eu-summary-step pending">{_T(lang, "Plan not generated yet", "计划尚未生成")}</span>'

    output_html = "".join(
        '<div class="eu-summary-output">'
        '<div class="eu-summary-output-preview">'
        f'<span class="mono">{_esc(o.get("kind", ""))}</span>'
        '</div>'
        '<div class="eu-summary-output-copy">'
        f'<small class="mono">{_esc(o.get("badge", ""))}</small>'
        f'<b>{_esc(o.get("title", ""))}</b>'
        f'<p>{_esc(o.get("sub", ""))}</p>'
        '</div>'
        '</div>'
        for o in outputs[:8]
    )

    findings = []
    for finding in (audit.get("findings") or [])[:4]:
        if isinstance(finding, dict):
            findings.append(
                '<div class="eu-summary-finding">'
                f'<span class="{_esc(str(finding.get("severity") or "info").lower())}"></span>'
                '<div>'
                f'<b>{_esc(finding.get("validator", "?"))}</b>'
                f'<p>{_esc(finding.get("message", ""))}</p>'
                '</div>'
                '</div>'
            )
    if not findings:
        findings.append(
            '<div class="eu-summary-finding">'
            '<span class="info"></span>'
            '<div>'
            f'<b>{_T(lang, "No validator findings", "无校验发现")}</b>'
            f'<p>{_T(lang, "A real manifest will populate this section.", "真实 manifest 会填充这里。")}</p>'
            '</div>'
            '</div>'
        )
    action_html = "".join(
        f'<div class="eu-summary-action {a.get("state", "ready")}">'
        f'<b>{_esc(a.get("label", ""))}</b>'
        f'<p>{_esc(a.get("detail", ""))}</p>'
        '</div>'
        for a in actions[:3]
    )
    demo_notice = (
        f'<div class="eu-summary-demo-note">{_esc(state.get("demo_notice", ""))}</div>'
        if is_demo and state.get("demo_notice") else ""
    )

    return (
        '<div class="eu-summary-page">'
        f'{demo_notice}'
        '<div class="eu-summary-top">'
        '<div class="eu-summary-cohort">'
        f'<div class="eu-section-label">{_T(lang, "Inbound cohort", "输入队列")}</div>'
        f'<h3>{_esc(state.get("run_id") or _T(lang, "Current EasyICU session", "当前 EasyICU 会话"))}</h3>'
        f'<p>{_esc(state.get("source_label", ""))} · {_esc(state.get("status", ""))}</p>'
        f'<div class="eu-summary-stat-grid">{cohort_html}</div>'
        f'<div class="eu-summary-chip-row">{chips_html}</div>'
        '</div>'
        '<div class="eu-summary-question">'
        f'<div class="eu-section-label">{_T(lang, "Research question", "研究问题")}</div>'
        f'<h2>{_esc(state.get("title", "Research Agent"))}</h2>'
        f'<p>{_T(lang, "Analysis-first: generated outputs are reviewed before manuscript drafting.", "分析优先：生成产物先复核，再进入手稿草稿。")}</p>'
        f'<div class="eu-summary-step-row">{step_html}</div>'
        '</div>'
        '</div>'
        '<div class="eu-summary-output-block">'
        '<div class="eu-summary-block-head">'
        f'<b>{_T(lang, "Analysis outputs", "分析产出")}</b>'
        f'<span class="mono">{len(outputs)} {_T(lang, "items", "项")}</span>'
        '</div>'
        f'<div class="eu-summary-output-grid">{output_html}</div>'
        '</div>'
        '<div class="eu-summary-bottom">'
        '<div class="eu-summary-findings">'
        f'<div class="eu-section-label">{_T(lang, "Findings", "主要发现")}</div>'
        f'{"".join(findings)}'
        '</div>'
        '<div class="eu-summary-review">'
        f'<div class="eu-section-label">{_T(lang, "Review gate", "复核关口")}</div>'
        f'<div class="eu-summary-action-grid">{action_html}</div>'
        '<div class="eu-summary-manuscript">'
        f'<b>{_T(lang, "Manuscript stays behind the gate", "手稿保留在关口之后")}</b>'
        f'<p>{_T(lang, "Draft methods and results only after evidence checks are accepted.", "只有证据检查通过后，才生成方法和结果草稿。")}</p>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )


def render_agent_output_summary(lang: str) -> None:
    """Render the analysis-first Research Agent summary page."""
    state = _resolve_workbench_state(lang)
    if not state.get("steps"):
        _render_workbench_empty_state(lang, summary=True)
        return
    state.setdefault("source_label", _T(lang, "Real manifest", "真实 manifest") if not state.get("is_demo") else _T(lang, "Sample workflow", "示例流程"))
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Research Agent", "研究代理"),
            title_en=state.get("title", "Research Agent"),
            title_zh=state.get("title", "研究 Agent"),
            desc=_T(
                lang,
                "Analysis-first output summary. Manuscript drafting remains gated by evidence review.",
                "分析优先的输出总览。手稿草稿仍受证据复核关口控制。",
            ),
            right_html=f'<span class="eu-pill">{_esc(state.get("source_label", ""))}</span>',
            lang=lang,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(_output_summary_html(state, lang), unsafe_allow_html=True)


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
    is_demo = st.session_state.get("entry_mode") == "demo"
    title = (
        _T(lang, "Demo guide has no active agent run", "Demo 导览没有当前 Agent run")
        if is_demo else
        _T(lang, "No active agent run is open", "尚未打开当前 Agent run")
    )
    body = (
        _T(
            lang,
            "Demo Mode explains the Research Agent without creating a fake queue. Switch to Real Data Mode, start a run from Setup, or explicitly open a historical manifest.",
            "Demo 模式只解释 Research Agent，不再伪造运行队列。请切换到真实数据模式，在配置页启动 run，或明确打开历史 manifest。",
        )
        if is_demo else
        _T(
            lang,
            "Start from Setup to configure the question, cohort, and LLM, or explicitly open a historical manifest from the run history. Workbench will not silently load the newest old run.",
            "请先在配置页设置研究问题、队列和 LLM，或从历史记录中明确打开某个 manifest。工作台不会再自动加载最新旧 run。",
        )
    )
    return (
        '<div class="eu-agent-empty">'
        '<div>'
        f'<span class="eu-section-label">{_T(lang, "Project entry", "项目入口")}</span>'
        f'<h2>{_esc(title)}</h2>'
        f'<p>{_esc(body)}</p>'
        '</div>'
        '<div class="eu-agent-empty-grid">'
        '<div>'
        f'<b>{_T(lang, "1. Configure", "1. 配置")}</b>'
        f'<small>{_T(lang, "Question, data recipe, model, workdir", "研究问题、数据来源、模型、写入目录")}</small>'
        '</div>'
        '<div>'
        f'<b>{_T(lang, "2. Run or import", "2. 运行或导入")}</b>'
        f'<small>{_T(lang, "Launch a real run or open an existing manifest", "启动真实 run 或打开已有 manifest")}</small>'
        '</div>'
        '<div>'
        f'<b>{_T(lang, "3. Review", "3. 复核")}</b>'
        f'<small>{_T(lang, "Step IO, checkpoints, evidence, review tasks", "步骤输入输出、检查点、证据和复核任务")}</small>'
        '</div>'
        '</div>'
        '</div>'
    )


def _render_workbench_empty_state(lang: str, *, summary: bool = False) -> None:
    st.markdown(
        cc.render_design_page_header(
            kicker=_T(lang, "Research Agent", "研究代理"),
            title_en=_T(lang, "Agent project workspace", "Agent 项目工作区"),
            title_zh=_T(lang, "Agent project workspace", "Agent 项目工作区"),
            desc=_T(
                lang,
                "Choose a research request, cohort, and run manifest before reviewing agent outputs.",
                "先选择研究请求、队列和 run manifest，再复核 agent 输出。",
            ),
            right_html=f'<span class="eu-pill">{_T(lang, "No active run", "无当前 run")}</span>',
            lang=lang,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(_workbench_empty_html(lang), unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1.4, 1.55, 1.4, 5.0])
    with c1:
        if st.button(_T(lang, "Configure new run", "配置新 run"), key=f"_eu_wb_empty_setup_{summary}", type="primary", use_container_width=True):
            st.session_state["_ra_view"] = "setup"
            st.rerun()
    with c2:
        if st.button(_T(lang, "Open run history", "打开历史记录"), key=f"_eu_wb_empty_history_{summary}", use_container_width=True):
            st.session_state["_ra_view"] = "setup"
            st.session_state["_research_agent_expand_history"] = True
            st.rerun()
    with c3:
        if st.button(_T(lang, "Open latest run", "打开最近 run"), key=f"_eu_wb_empty_latest_{summary}", use_container_width=True):
            latest = _latest_real_workbench_state(lang)
            if latest:
                st.session_state["_agent_workbench"] = latest
                st.session_state["_agent_workbench_is_active_selection"] = True
                st.session_state["_ra_view"] = "summary" if summary else "workbench"
                st.rerun()
            st.warning(_T(lang, "No local run manifest found.", "没有找到本地 run manifest。"))


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
            f'<div><span>{_T(lang, "Findings", "发现")}</span><b>{int(counts.get("errors") or 0)}E / {int(counts.get("warnings") or 0)}W</b></div>'
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
# Public entrypoint
# ---------------------------------------------------------------------

def render_agent_workbench(lang: str) -> None:
    """Render the live agent workbench (3 columns + timeline)."""
    state = _resolve_workbench_state(lang)
    if not state.get("steps"):
        _render_workbench_empty_state(lang)
        return
    # carry a short subtitle into the results column
    state.setdefault("subtitle_short", "")
    state.setdefault("source_label", _T(lang, "Real manifest", "真实 manifest") if not state.get("is_demo") else _T(lang, "Sample workflow", "示例流程"))
    select_key, selected_idx = _resolve_selected_step(state)
    active_state = _state_for_selected_step(state, selected_idx)

    # Header
    arm_label = _T(lang, "ICU-aware · default arm", "ICU-aware · 默认实验臂")
    actions = (
        f'<span class="eu-pill" title="{_esc(_T(lang, "Web UI runs the ICU-aware arm only; the naive ablation is exposed via the CLI --arms flag.", "Web 端只跑 ICU-aware 实验臂；naive 消融需通过 CLI --arms 显式触发。"))}">'
        f'<span class="dot" style="background:var(--accent)"></span>{_esc(arm_label)}</span>'
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
    c1, c2, c3, c4 = st.columns([7.0, 1.8, 1.8, 1.8])
    with c2:
        if st.button(_T(lang, "View summary", "查看摘要"), key="_eu_wb_summary", use_container_width=True):
            st.session_state["_ra_view"] = "summary"
            st.rerun()
    with c3:
        if st.button(
            _T(lang, "Run controls", "运行控制"),
            key="_eu_wb_run_controls",
            use_container_width=True,
            help=_T(
                lang,
                "Open Setup, where live runs are configured and launched.",
                "打开配置页；实时 run 在那里配置和启动。",
            ),
        ):
            st.session_state["_ra_view"] = "setup"
            st.rerun()
    with c4:
        if st.button(
            _T(lang, "Plan contract", "计划契约"),
            key="_eu_wb_adjust",
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
            title_zh=state.get("title", "研究 Agent"),
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
    st.markdown(
        '<div class="eu-agent-timeline" style="margin-top:18px">' + _state_track_html(state, lang) + '</div>',
        unsafe_allow_html=True,
    )
    audit_html = _audit_review_html(state, lang)
    if audit_html:
        st.markdown(audit_html, unsafe_allow_html=True)
