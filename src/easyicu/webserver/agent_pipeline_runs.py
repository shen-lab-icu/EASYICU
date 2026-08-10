"""Native Web bridge to the real Research Agent scientific pipeline.

Pi Copilot and the Web route own conversation and job lifecycle only.  This
module delegates concept selection/materialisation to the data-foundation
owner and Plan -> Execute -> Validate -> Write to ``ResearchAgentPipeline``.
It then creates a bounded, path-free Web projection of the pipeline's own
artifacts; it never computes a scientific result itself.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from easyicu.webserver import provider_adapter

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_MANUSCRIPT_PREVIEW = 24_000
_MAX_FIGURE_EMBED_BYTES = 420_000
_MAX_FIGURE_EMBED_TOTAL = 1_400_000
_MAX_TABLE_ROWS = 30
_MAX_TABLE_COLUMNS = 12
_MAX_PENDING = 16
_PENDING_LOCK = threading.RLock()
_UNSAFE_PROJECTION_PATTERNS = (
    re.compile(
        r"(?:file://|(?<![A-Za-z0-9])/(?:Users|home|private|tmp|var|etc|opt|Volumes)/|\b[A-Za-z]:\\)",
        re.I,
    ),
    re.compile(
        r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{8,}|"
        r"\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+|"
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
        re.I,
    ),
    re.compile(
        r"[\"']?(?:subject_id|stay_id|hadm_id|patient_id|mrn)[\"']?"
        r"\s*[:,=]\s*[\"']?[A-Za-z0-9-]+",
        re.I,
    ),
)


class ResearchPipelineRunError(RuntimeError):
    """Stable Web-facing failure from the true Research Agent bridge."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass
class _PendingRun:
    pipeline: Any
    pending: Any
    wrapper_dir: Path
    study: Dict[str, Any]
    provider: Dict[str, Any]
    acquisition: Any
    created_at: float


_PENDING: Dict[str, _PendingRun] = {}


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "study")).strip("-.")
    return text[:96] or "study"


def _clean_text(value: Any, limit: int = 1_200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_relative(root: Path, raw: Any) -> Optional[Path]:
    text = str(raw or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or "\0" in text:
        return None
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    candidate = root.joinpath(*parts)
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if candidate.is_symlink() or not resolved.is_file():
        return None
    return resolved


def _target_outcome(study: Mapping[str, Any]) -> Optional[str]:
    value = _clean_text(study.get("outcome"), 160)
    if value.lower() in {
        "none",
        "n/a",
        "na",
        "not applicable",
        "descriptive only",
        "无",
        "不适用",
        "仅描述",
    }:
        return None
    return value or None


def _cohort_window(study: Mapping[str, Any]) -> tuple[float, float]:
    raw = study.get("time_window")
    window = raw if isinstance(raw, Mapping) else {}
    value = window.get("hours") or window.get("observation_hours") or 24
    try:
        hours = float(value)
    except (TypeError, ValueError):
        hours = 24.0
    if not 0 < hours <= 24 * 365:
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_invalid",
            "The configured study time window is outside the supported range.",
        )
    return (0.0, hours)


def _configured_modules(study: Mapping[str, Any]) -> tuple[str, ...]:
    raw = study.get("modules")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(
            str(module).strip().lower()
            for module in raw
            if isinstance(module, str) and str(module).strip()
        )
    )


def _data_foundation_profile(
    *,
    export_path: str,
    study: Mapping[str, Any],
    target: Optional[str],
) -> Dict[str, Any]:
    """Compile StudyContext modules into one typed materialization request."""

    from easyicu.research_agent.acquisition.catalog import build_available_catalog

    modules = _configured_modules(study)
    if not modules:
        raise ResearchPipelineRunError(
            "research_pipeline_modules_required",
            "A full Research Agent run requires configured feature modules.",
        )
    allowed = set(modules)
    catalog = build_available_catalog(Path(export_path).expanduser())
    concepts = [
        concept
        for concept in catalog.concepts
        if Path(concept.file_name).stem.lower() in allowed
    ]
    by_id = {concept.concept_id: concept for concept in concepts}
    demographic_values = [
        concept.concept_id
        for concept in concepts
        if Path(concept.file_name).stem.lower() == "demographics"
        and (not concept.typed_metadata or concept.column_role == "value")
    ]
    preferred_base = [
        concept for concept in ("age", "sex") if concept in demographic_values
    ]
    static_concepts = preferred_base or demographic_values[:1]
    if not static_concepts:
        raise ResearchPipelineRunError(
            "research_pipeline_stay_denominator_unavailable",
            "The configured modules do not provide a stay-level denominator concept.",
        )

    outcome_concepts: List[str] = []
    required_feature_concepts: List[str] = []
    require_outcome = False
    if target:
        target_meta = by_id.get(target)
        if target_meta is None:
            raise ResearchPipelineRunError(
                "research_pipeline_target_outside_configured_modules",
                "The configured outcome is not available in the selected feature modules.",
            )
        target_module = Path(target_meta.file_name).stem.lower()
        if target_meta.column_role == "event_status":
            outcome_concepts.append(target)
            require_outcome = True
        elif target_module in {"demographics", "outcome"}:
            static_concepts.append(target)
        else:
            required_feature_concepts.append(target)

    return {
        "allowed_modules": modules,
        "static_concepts": tuple(dict.fromkeys(static_concepts)),
        "outcome_concepts": tuple(outcome_concepts),
        "required_feature_concepts": tuple(required_feature_concepts),
        "require_outcome": require_outcome,
    }


def _research_user_preferences(study: Mapping[str, Any]) -> Dict[str, Any]:
    """Compile StudyContext into the existing strict preference contract."""

    preferences: Dict[str, Any] = {}
    purpose = _clean_text(study.get("purpose"), 1_200)
    analysis_goal = _clean_text(study.get("analysis_goal"), 1_200)
    comparator = _clean_text(study.get("comparator"), 800)
    if purpose:
        preferences["extra_notes"] = purpose
    if analysis_goal:
        preferences["must_have_outputs"] = analysis_goal
    if comparator:
        preferences["subgroup_sensitivity"] = comparator

    time_window = study.get("time_window")
    if isinstance(time_window, Mapping) and time_window:
        preferences["timing_and_design"] = json.dumps(
            dict(time_window), ensure_ascii=False, sort_keys=True
        )[:1_000]
    constraints: Dict[str, Any] = {}
    cohort = study.get("cohort")
    confirmations = study.get("confirmations")
    if isinstance(cohort, Mapping) and cohort:
        constraints["cohort"] = dict(cohort)
    if isinstance(confirmations, Mapping) and confirmations:
        constraints["confirmations"] = dict(confirmations)
    if constraints:
        preferences["data_constraints"] = json.dumps(
            constraints, ensure_ascii=False, sort_keys=True
        )[:2_400]
    return preferences


def _inclusion_criteria(study: Mapping[str, Any]) -> List[str]:
    raw = study.get("cohort")
    cohort = raw if isinstance(raw, Mapping) else {}
    rows: List[str] = []
    review = _clean_text(cohort.get("review") or cohort.get("label"), 500)
    if review:
        rows.append(review)
    age_min = cohort.get("age_min")
    age_max = cohort.get("age_max")
    if age_min is not None or age_max is not None:
        rows.append(
            f"age range: {age_min if age_min is not None else '*'} to {age_max if age_max is not None else '*'}"
        )
    minimum_los = cohort.get("min_icu_los_hours")
    if minimum_los is not None:
        rows.append(f"minimum ICU length of stay: {minimum_los} hours")
    if cohort.get("exclude_readmissions") is True:
        rows.append("first eligible ICU stay per patient")
    for key, prefix in (
        ("include_diagnoses", "include diagnoses"),
        ("exclude_diagnoses", "exclude diagnoses"),
    ):
        values = cohort.get(key)
        if isinstance(values, list):
            clean = [_clean_text(item, 120) for item in values[:20]]
            clean = [item for item in clean if item]
            if clean:
                rows.append(f"{prefix}: {', '.join(clean)}")
    return rows[:32]


def _progress(job: Any, *, step: str, label: str, **extra: Any) -> None:
    event: Dict[str, Any] = {
        "type": "progress",
        "step": _clean_text(step, 80),
        "label": _clean_text(label, 240),
    }
    for key in ("current", "total", "status", "run_id"):
        if extra.get(key) is not None:
            event[key] = extra[key]
    job.emit(event)
    if getattr(job, "cancel_requested", False):
        raise ResearchPipelineRunError(
            "research_pipeline_cancelled",
            "The Research Agent run was cancelled by the user.",
        )


def _pipeline_progress(job: Any, event: Mapping[str, Any]) -> None:
    _progress(
        job,
        step=str(event.get("step") or event.get("phase") or "research_pipeline"),
        label=str(
            event.get("label") or event.get("message") or "Research Agent working"
        ),
        current=event.get("current"),
        total=event.get("total"),
        status=event.get("status"),
        run_id=event.get("run_id"),
    )


def _safe_claims(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "claim_ledger.csv"
    try:
        if not path.is_file() or path.stat().st_size > 512_000:
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))[:40]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    claims: List[Dict[str, Any]] = []
    for row in rows:
        text = _clean_text(row.get("claim_text"), 2_000)
        if not text:
            continue
        refs = [
            _clean_text(item, 160)
            for item in re.split(r"[;,|]", str(row.get("evidence_refs") or ""))[:40]
            if _clean_text(item, 160)
        ]
        claims.append(
            {
                "id": _clean_text(row.get("claim_id"), 160),
                "text": text,
                "evidence_ids": refs,
                "status": _clean_text(row.get("status"), 120) or "evidence_bound_draft",
                "note": _clean_text(row.get("note"), 800),
            }
        )
    return claims


def _figure_projection(run_dir: Path) -> Dict[str, Any]:
    source = _read_json(run_dir / "figure_gallery.json", {})
    figures = source.get("figures") if isinstance(source, Mapping) else []
    public: List[Dict[str, Any]] = []
    embedded_total = 0
    for row in figures[:24] if isinstance(figures, list) else []:
        if not isinstance(row, Mapping):
            continue
        relative = _clean_text(row.get("relative_path"), 300)
        item: Dict[str, Any] = {
            "label": _clean_text(row.get("label") or row.get("figure_id"), 240),
            "name": Path(relative).name if relative else "figure",
            "relative_path": relative,
            "status": _clean_text(row.get("status"), 120) or "available",
            "tier": _clean_text(row.get("tier"), 120),
            "panel_roles": [
                _clean_text(value, 120)
                for value in list(row.get("panel_roles") or [])[:12]
                if _clean_text(value, 120)
            ],
            "chart_types": [
                _clean_text(value, 120)
                for value in list(row.get("chart_types") or [])[:12]
                if _clean_text(value, 120)
            ],
        }
        path = _safe_relative(run_dir, relative)
        if path is not None and path.suffix.lower() == ".png":
            try:
                size = path.stat().st_size
                if (
                    0 < size <= _MAX_FIGURE_EMBED_BYTES
                    and embedded_total + size <= _MAX_FIGURE_EMBED_TOTAL
                ):
                    item["data_url"] = "data:image/png;base64," + base64.b64encode(
                        path.read_bytes()
                    ).decode("ascii")
                    embedded_total += size
            except OSError:
                pass
        public.append(item)
    return {
        "kind": "figure_gallery",
        "schema_version": "easyicu.web-pipeline-figure-gallery/1",
        "status": _clean_text(
            source.get("status") if isinstance(source, Mapping) else "", 120
        )
        or ("available" if public else "no_figures"),
        "figures": public,
        "primary_count": int(source.get("primary_count") or 0)
        if isinstance(source, Mapping)
        else 0,
        "supporting_count": int(source.get("supporting_count") or 0)
        if isinstance(source, Mapping)
        else len(public),
        "embedded_count": sum(1 for row in public if row.get("data_url")),
    }


def _identifier_column(name: Any) -> bool:
    token = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    return token in {
        "stayid",
        "subjectid",
        "patientid",
        "hadmid",
        "icustayid",
        "patientunitstayid",
        "recordid",
    }


def _table_projection(run_dir: Path) -> Dict[str, Any]:
    evidence = _read_json(run_dir / "evidence" / "evidence_index.json", [])
    tables: List[Dict[str, Any]] = []
    skipped_sensitive = 0
    for record in evidence if isinstance(evidence, list) else []:
        if not isinstance(record, Mapping) or str(record.get("kind")) != "table":
            continue
        path = _safe_relative(run_dir, record.get("relative_path"))
        if path is None or path.suffix.lower() != ".csv":
            continue
        try:
            if path.stat().st_size > 1_500_000:
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                headers = next(reader, [])[:_MAX_TABLE_COLUMNS]
                if any(_identifier_column(value) for value in headers):
                    skipped_sensitive += 1
                    continue
                rows = [
                    row[: len(headers)]
                    for _, row in zip(range(_MAX_TABLE_ROWS), reader)
                ]
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        tables.append(
            {
                "evidence_id": _clean_text(record.get("evidence_id"), 160),
                "label": _clean_text(record.get("description"), 300) or path.stem,
                "name": path.name,
                "headers": [_clean_text(value, 160) for value in headers],
                "rows": [[_clean_text(value, 500) for value in row] for row in rows],
                "preview_truncated": len(rows) >= _MAX_TABLE_ROWS,
            }
        )
        if len(tables) >= 12:
            break
    return {
        "schema_version": "easyicu.web-pipeline-result-tables/1",
        "tables": tables,
        "table_count": len(tables),
        "skipped_identifier_tables": skipped_sensitive,
        "preview_policy": "aggregate_tables_only_no_identifier_columns",
    }


def _readiness_axes(run_dir: Path) -> Dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json", {})
    status = _read_json(run_dir / "run_status.json", {})
    readiness = manifest.get("readiness") if isinstance(manifest, Mapping) else {}
    if not isinstance(readiness, Mapping):
        readiness = {}
    gates = status.get("gates") if isinstance(status, Mapping) else {}
    if not isinstance(gates, Mapping):
        gates = {}
    merged = dict(gates)
    merged.update(readiness)
    return {
        key: merged.get(key)
        for key in (
            "execution_complete",
            "step_scientific_requirements_complete",
            "artifact_valid",
            "scientific_requirement_complete",
            "evidence_complete",
            "numeric_verified",
            "analysis_validated",
            "manuscript_ready",
            "manuscript_generated",
            "publication_ready",
            "paper_authorized",
            "missing_evidence_count",
            "numeric_error_count",
            "evidence_error_count",
            "analysis_error_count",
            "failed_steps",
            "missing_steps",
            "analysis_errors",
        )
        if key in merged
    }


def _gate_from_axes(axes: Mapping[str, Any], *, pending: bool) -> Dict[str, Any]:
    if pending:
        status = "blocked"
        reason = "human_plan_review_required"
    else:
        complete = bool(axes.get("execution_complete"))
        validated = bool(axes.get("analysis_validated"))
        evidence = bool(axes.get("evidence_complete"))
        numeric = bool(axes.get("numeric_verified"))
        status = (
            "analysis_only"
            if complete and validated and evidence and numeric
            else "blocked"
        )
        reason = (
            "research_agent_pipeline_complete_human_interpretation_required"
            if status == "analysis_only"
            else "research_agent_pipeline_failed_closed"
        )
    checks = [
        {
            "id": key,
            "label": key.replace("_", " "),
            "passed": bool(axes.get(key)),
            "reason": None if axes.get(key) else f"{key}_not_satisfied",
        }
        for key in (
            "execution_complete",
            "analysis_validated",
            "evidence_complete",
            "numeric_verified",
            "manuscript_ready",
        )
    ]
    return {
        "status": status,
        "reason": reason,
        "reportable": False,
        "draft_unlocked": False,
        "checks": checks,
    }


def _artifact_record(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    return {
        "name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "kind": "json",
    }


def _projection_privacy_scan(payloads: Mapping[str, Any]) -> Dict[str, Any]:
    """Reject host paths, credentials, and row identifiers in Web artefacts.

    Aggregate result-table rows are intentionally permitted.  The scientific
    pipeline owns their disclosure checks, while this boundary independently
    rejects the three classes that must never cross into the browser.
    """

    hits: List[Dict[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, Path):
            hits.append({"path": path, "reason": "path_object"})
        elif isinstance(value, str):
            for index, pattern in enumerate(_UNSAFE_PROJECTION_PATTERNS):
                if pattern.search(value):
                    hits.append({"path": path, "reason": f"pattern_{index + 1}"})
                    break

    for name, payload in payloads.items():
        visit(payload, str(name))
    return {
        "passed": not hits,
        "scanned_artifacts": len(payloads),
        "unsafe_value_count": len(hits),
        "hits": hits[:40],
    }


def _privacy_blocked_payloads(
    *,
    run_context: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return a fixed-schema package without echoing unsafe source values."""

    gate = {
        "status": "blocked",
        "reason": "research_pipeline_projection_privacy_blocked",
        "reportable": False,
        "draft_unlocked": False,
        "checks": [
            {
                "id": "browser_projection_privacy",
                "label": "Browser projection contains no host path, credential, or row identifier",
                "passed": False,
                "unsafe_value_count": int(scan.get("unsafe_value_count") or 0),
            }
        ],
    }
    return {
        "run_context.json": {
            "run_id": _clean_text(run_context.get("run_id"), 160),
            "study_id": _clean_text(run_context.get("study_id"), 160),
            "mode": "research_agent_pipeline",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "summary": {"projection_withheld": True},
            "local_first": {"uploads": 0},
        },
        "quality_gate.json": {
            "gate": gate,
            "quality": [],
            "privacy": {
                "passed": False,
                "unsafe_value_count": int(scan.get("unsafe_value_count") or 0),
                "payloads_withheld": True,
            },
        },
    }


def _write_projection(
    *,
    wrapper_dir: Path,
    study: Mapping[str, Any],
    provider: Mapping[str, Any],
    acquisition: Any,
    run_dir: Optional[Path],
    pending: Optional[Any] = None,
    blocked_reason: Optional[str] = None,
) -> Dict[str, Any]:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(
        getattr(pending, "run_id", "")
        or (run_dir.name if run_dir else wrapper_dir.name)
    )
    axes = _readiness_axes(run_dir) if run_dir is not None else {}
    gate = _gate_from_axes(axes, pending=pending is not None)
    if blocked_reason:
        gate["status"] = "blocked"
        gate["reason"] = blocked_reason
    selection = getattr(acquisition, "selection", None)
    selected_concepts = list(getattr(selection, "selected_concepts", []) or [])
    coverage = getattr(acquisition, "coverage", None)
    run_context = {
        "run_id": run_id,
        "study_id": _clean_text(study.get("id"), 160),
        "mode": "research_agent_pipeline",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "question": _clean_text(study.get("question"), 1_200),
        "source": {
            "label": _clean_text((study.get("data_source") or {}).get("label"), 160)
            if isinstance(study.get("data_source"), Mapping)
            else "",
            "database": _clean_text(
                (study.get("data_source") or {}).get("database"), 64
            )
            if isinstance(study.get("data_source"), Mapping)
            else "",
        },
        "summary": {
            "selected_concepts": selected_concepts[:64],
            "materialized_concept_count": len(
                list(getattr(acquisition, "materialized_concepts", []) or [])
            ),
            "coverage_sufficient": bool(getattr(coverage, "sufficient", False)),
        },
        "local_first": {"uploads": 0},
    }
    cohort_summary = {
        "summary": run_context["summary"],
        "cohort": {
            "criteria": _inclusion_criteria(study),
            "target_outcome": _target_outcome(study),
            "time_window_hours": _cohort_window(study)[1],
        },
    }
    plan = _read_json(run_dir / "analysis_plan.json", {}) if run_dir else {}
    manuscript_path = run_dir / "manuscript_scaffold_bound.md" if run_dir else None
    manuscript_text = ""
    if manuscript_path is not None:
        try:
            manuscript_text = manuscript_path.read_text(encoding="utf-8")[
                :_MAX_MANUSCRIPT_PREVIEW
            ]
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            manuscript_text = ""
    claims = _safe_claims(run_dir) if run_dir else []
    figure_gallery = (
        _figure_projection(run_dir)
        if run_dir
        else {
            "kind": "figure_gallery",
            "schema_version": "easyicu.web-pipeline-figure-gallery/1",
            "status": "not_available",
            "figures": [],
            "primary_count": 0,
            "supporting_count": 0,
            "embedded_count": 0,
        }
    )
    result_tables = (
        _table_projection(run_dir)
        if run_dir
        else {
            "schema_version": "easyicu.web-pipeline-result-tables/1",
            "tables": [],
            "table_count": 0,
            "skipped_identifier_tables": 0,
            "preview_policy": "aggregate_tables_only_no_identifier_columns",
        }
    )
    evidence_index = (
        _read_json(run_dir / "evidence" / "evidence_index.json", []) if run_dir else []
    )
    evidence_count = len(evidence_index) if isinstance(evidence_index, list) else 0
    pending_requests = []
    if pending is not None:
        pending_requests = [
            {
                "review_id": request.review_id,
                "kind": request.kind,
                "summary": request.summary,
                "authority_sha256": request.authority_sha256,
            }
            for request in pending.requests
        ]
    source_manifest = {
        "schema_version": "easyicu.web-research-pipeline-projection/1",
        "engine": "easyicu.research_agent.pipeline",
        "run_id": run_id,
        "status": "human_review_pending" if pending is not None else gate["status"],
        "resume_scope": getattr(pending, "resume_scope", None),
        "pending_reviews": pending_requests,
        "readiness": axes,
        "evidence_count": evidence_count,
        "result_table_count": result_tables.get("table_count", 0),
        "figure_count": len(figure_gallery.get("figures") or []),
        "provider": {
            key: provider.get(key)
            for key in (
                "provider",
                "model",
                "client",
                "provider_gate",
                "credential_fingerprint",
            )
            if provider.get(key) is not None
        },
        "path_values_returned": False,
    }
    payloads: Dict[str, Dict[str, Any]] = {
        "run_context.json": run_context,
        "cohort_summary.json": cohort_summary,
        "quality_gate.json": {"gate": gate, "quality": []},
        "agent_plan.json": plan if isinstance(plan, dict) else {},
        "manuscript_draft.json": {
            "run_id": run_id,
            "status": "locked_pending_human_review",
            "question": run_context["question"],
            "claims": claims,
            "sentences": [],
            "markdown_preview": manuscript_text,
            "source": "research_agent_manuscript_scaffold_bound",
        },
        "figure_gallery.json": figure_gallery,
        "result_tables.json": result_tables,
        "source_run_manifest.json": source_manifest,
    }
    privacy_scan = _projection_privacy_scan(payloads)
    if not privacy_scan["passed"]:
        payloads = _privacy_blocked_payloads(
            run_context=run_context,
            scan=privacy_scan,
        )
        gate = dict(payloads["quality_gate.json"]["gate"])
        source_manifest = {
            "schema_version": "easyicu.web-research-pipeline-projection/1",
            "engine": "easyicu.research_agent.pipeline",
            "run_id": run_id,
            "status": "blocked",
            "provider": source_manifest["provider"],
            "path_values_returned": False,
            "projection_withheld": True,
        }
        payloads["source_run_manifest.json"] = source_manifest
    for name, payload in payloads.items():
        _write_json(wrapper_dir / name, payload)
    artifact_rows = [_artifact_record(wrapper_dir / name) for name in payloads]
    ledger = {
        "schema_version": "easyicu.web-research-pipeline-ledger/1",
        "run_id": run_id,
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "status": gate["status"],
        "artifacts": artifact_rows,
        "provider": source_manifest["provider"],
        "pipeline_evidence_count": evidence_count,
        "privacy": {
            "patient_rows_in_projection": False,
            "identifier_columns_withheld": (
                result_tables.get("skipped_identifier_tables", 0)
                if privacy_scan["passed"]
                else True
            ),
            "path_values_returned": False,
            "projection_scan_passed": bool(privacy_scan["passed"]),
        },
    }
    _write_json(wrapper_dir / "evidence_ledger.json", ledger)
    artifacts = [*artifact_rows, _artifact_record(wrapper_dir / "evidence_ledger.json")]
    return {
        "run_id": run_id,
        "study_id": run_context["study_id"],
        "mode": "research_agent_pipeline",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "project_dir": str(wrapper_dir),
        "gate": gate,
        "provider": source_manifest["provider"],
        "artifacts": artifacts,
        "human_review_pending": pending is not None,
        "pending_reviews": pending_requests,
        "uploads": 0,
    }


def _register_pending(entry: _PendingRun) -> None:
    with _PENDING_LOCK:
        _PENDING[str(entry.pending.run_id)] = entry
        while len(_PENDING) > _MAX_PENDING:
            oldest = min(_PENDING.items(), key=lambda item: item[1].created_at)[0]
            _PENDING.pop(oldest, None)


def pending_review(run_id: Any) -> Optional[Dict[str, Any]]:
    key = _clean_text(run_id, 160)
    with _PENDING_LOCK:
        entry = _PENDING.get(key)
        if entry is None:
            return None
        pending = entry.pending
        return {
            "run_id": pending.run_id,
            "study_id": _clean_text(entry.study.get("id"), 160),
            "resume_scope": pending.resume_scope,
            "resumable_here": bool(pending.resumable_here),
            "requests": [
                {
                    "review_id": request.review_id,
                    "kind": request.kind,
                    "summary": request.summary,
                    "authority_sha256": request.authority_sha256,
                }
                for request in pending.requests
            ],
        }


def make_research_pipeline_run_runner(
    *,
    export_path: str,
    study_context: Mapping[str, Any],
    project_root: Optional[str],
    provider: Mapping[str, Any],
    provider_environment: Optional[Mapping[str, str]] = None,
) -> Any:
    """Build the JobManager runner for a real, evidence-bound pipeline run."""

    study = dict(study_context)
    question = _clean_text(study.get("question"), 1_200)
    if not question:
        raise ResearchPipelineRunError(
            "research_pipeline_question_required",
            "A scientific question is required before starting the pipeline.",
        )
    source = study.get("data_source")
    database = (
        _clean_text(source.get("database") if isinstance(source, Mapping) else "", 64)
        or "miiv"
    )
    target = _target_outcome(study)
    window = _cohort_window(study)
    research_provider_environment = (
        dict(provider_environment) if provider_environment is not None else None
    )

    def runner(job: Any) -> Dict[str, Any]:
        root = (
            Path(project_root).expanduser()
            if project_root
            else Path.home() / "easyicu" / "projects"
        )
        wrapper_dir = root / _slug(study.get("id")) / f"run_{job.id}"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        _progress(job, step="provider", label="Research Agent provider authorized")
        client, provider_public = provider_adapter.build_research_agent_provider_client(
            dict(provider),
            environ=research_provider_environment,
        )
        try:
            from easyicu.research_agent import ResearchAgentPipeline
            from easyicu.research_agent.acquisition.foundation import (
                acquire_universe_for_question,
            )
            from easyicu.research_agent.orchestration.config import PipelineConfig
            from easyicu.research_agent.orchestration.services import PipelineServices
            from easyicu.research_agent.orchestration.workflow import HumanReviewPending

            _progress(
                job,
                step="data_foundation",
                label="Selecting concepts and materializing a typed analysis universe",
            )
            foundation_profile = _data_foundation_profile(
                export_path=export_path,
                study=study,
                target=target,
            )
            acquisition = acquire_universe_for_question(
                export_dir=Path(export_path).expanduser(),
                question=question,
                llm=client,
                output_dir=wrapper_dir / "pipeline_input",
                stem="web_research_universe",
                target_outcome=target,
                outcome_concepts=foundation_profile["outcome_concepts"],
                required_feature_concepts=foundation_profile[
                    "required_feature_concepts"
                ],
                static_concepts=foundation_profile["static_concepts"],
                allowed_modules=foundation_profile["allowed_modules"],
                cohort_window=window,
                database=database,
                require_outcome=foundation_profile["require_outcome"],
                emit_trajectory=True,
            )
            if acquisition.blocked or acquisition.universe_path is None:
                return _write_projection(
                    wrapper_dir=wrapper_dir,
                    study=study,
                    provider=provider_public,
                    acquisition=acquisition,
                    run_dir=None,
                    blocked_reason="data_foundation_blocked",
                )
            config = PipelineConfig(
                workdir=wrapper_dir / "pipeline",
                enable_reproducibility_envelope=True,
                evidence_enforcement_mode="strict",
            )
            pipeline = ResearchAgentPipeline.from_config(
                config,
                services=PipelineServices(llm=client),
            )
            preferences = _research_user_preferences(study)
            _progress(
                job,
                step="research_pipeline",
                label="Research Agent planning, execution, validation, and writing started",
            )
            outcome = pipeline.run(
                question=question,
                cohort=acquisition.universe_path,
                cohort_authority_path=acquisition.cohort_authority_path,
                cohort_authority_ref=(
                    acquisition.cohort_authority_ref.to_dict()
                    if acquisition.cohort_authority_ref is not None
                    else None
                ),
                trajectory_path=acquisition.trajectory_path,
                trajectory_authority_path=acquisition.trajectory_authority_path,
                trajectory_authority_ref=(
                    acquisition.trajectory_authority_ref.to_dict()
                    if acquisition.trajectory_authority_ref is not None
                    else None
                ),
                cohort_name=f"web_{_slug(study.get('id'))}",
                database=database,
                target_outcome=target,
                inclusion_criteria=_inclusion_criteria(study),
                user_preferences=preferences,
                notes=_clean_text(study.get("analysis_goal"), 1_200) or None,
                progress_callback=lambda event: _pipeline_progress(job, event),
            )
            if isinstance(outcome, HumanReviewPending):
                run_dir = Path(outcome.run_dir)
                entry = _PendingRun(
                    pipeline=pipeline,
                    pending=outcome,
                    wrapper_dir=wrapper_dir,
                    study=study,
                    provider=provider_public,
                    acquisition=acquisition,
                    created_at=time.time(),
                )
                _register_pending(entry)
                return _write_projection(
                    wrapper_dir=wrapper_dir,
                    study=study,
                    provider=provider_public,
                    acquisition=acquisition,
                    run_dir=run_dir,
                    pending=outcome,
                )
            return _write_projection(
                wrapper_dir=wrapper_dir,
                study=study,
                provider=provider_public,
                acquisition=acquisition,
                run_dir=Path(outcome.manifest_path).parent,
            )
        except ResearchPipelineRunError:
            raise
        except Exception as exc:
            raise ResearchPipelineRunError(
                "research_pipeline_execution_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    return runner


def resume_research_pipeline(
    *,
    run_id: str,
    study_context_id: str,
    decision: str,
    reviewer: str,
    note: str,
    job: Any,
) -> Dict[str, Any]:
    """Resume one same-process plan review with digest-bound decisions."""

    key = _clean_text(run_id, 160)
    with _PENDING_LOCK:
        entry = _PENDING.get(key)
    if entry is None:
        raise ResearchPipelineRunError(
            "research_pipeline_review_not_resumable",
            "This plan review is not available in the current server process.",
        )
    if _clean_text(entry.study.get("id"), 160) != _clean_text(study_context_id, 160):
        raise ResearchPipelineRunError(
            "research_pipeline_review_study_mismatch",
            "The pending review belongs to a different research project.",
        )
    resolved = str(decision or "").strip().lower()
    if resolved not in {"approved", "rejected"}:
        raise ResearchPipelineRunError(
            "research_pipeline_review_decision_invalid",
            "Choose approved or rejected for the pending plan review.",
        )
    decisions = [
        {
            "review_id": request.review_id,
            "authority_sha256": request.authority_sha256,
            "decision": resolved,
            "reviewer": _clean_text(reviewer, 200) or "local_web_reviewer",
            "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": _clean_text(note, 1_000),
        }
        for request in entry.pending.requests
    ]
    _progress(
        job,
        step="human_review",
        label=f"Applying {resolved} decision to the digest-bound Research Agent plan",
    )
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRejected,
    )

    try:
        outcome = entry.pipeline.resume_human_review(decisions, run_id=key)
    except HumanReviewRejected:
        with _PENDING_LOCK:
            _PENDING.pop(key, None)
        return _write_projection(
            wrapper_dir=entry.wrapper_dir,
            study=entry.study,
            provider=entry.provider,
            acquisition=entry.acquisition,
            run_dir=Path(entry.pending.run_dir),
            blocked_reason="human_plan_review_rejected",
        )
    except Exception as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_review_resume_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if isinstance(outcome, HumanReviewPending):
        entry.pending = outcome
        return _write_projection(
            wrapper_dir=entry.wrapper_dir,
            study=entry.study,
            provider=entry.provider,
            acquisition=entry.acquisition,
            run_dir=Path(outcome.run_dir),
            pending=outcome,
        )
    with _PENDING_LOCK:
        _PENDING.pop(key, None)
    return _write_projection(
        wrapper_dir=entry.wrapper_dir,
        study=entry.study,
        provider=entry.provider,
        acquisition=entry.acquisition,
        run_dir=Path(outcome.manifest_path).parent,
    )


__all__ = [
    "ResearchPipelineRunError",
    "make_research_pipeline_run_runner",
    "pending_review",
    "resume_research_pipeline",
]
