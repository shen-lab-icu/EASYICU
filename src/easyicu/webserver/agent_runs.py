"""Registry-backed local Research Agent preflight jobs.

This is the native WebApp bridge between the shared export-source registry and
Agent/Copilot run creation. The default path deliberately stops at a
deterministic preflight: summarise the active export, resolve denominators,
audit stay-id coverage, and write an evidence ledger. The optional ``full`` path
can use either the offline mock provider or an external provider after
canonical AI opt-in, per-run opt-in, env credential checks, STRICT-style claim
binding, artifact privacy scanning, and manuscript locking.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from easyicu.webserver import agent_outputs
from easyicu.webserver import dataio
from easyicu.webserver import numeric_evidence_audit
from easyicu.webserver import provider_adapter
from easyicu.webserver import provider_gate
from easyicu.webserver import study_contexts as context_store


class AgentRunConfigError(ValueError):
    """Raised when a requested agent run would violate the local safety gate."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "agent_run_config_error"))
        self.detail = detail


_RUN_ARTIFACT_NAMES = [
    "run_context.json",
    "cohort_summary.json",
    "table1_summary.json",
    "missingness_audit.json",
    "roc_curve.json",
    "calibration_curve.json",
    "quality_gate.json",
    "agent_plan.json",
    "manuscript_draft.json",
    "benchmark_scorecard.json",
    "workflow_graph.json",
    "figure_gallery.json",
    "source_run_manifest.json",
    "evidence_ledger.json",
    "human_signoff.json",
]

_SIGNOFF_CONFIRMATIONS = {
    "evidence_reviewed",
    "claims_remain_locked",
    "no_patient_rows_persisted",
}
_AGENT_PREFLIGHT_FULL_SCAN_ROW_LIMIT = 1_000_000


def make_agent_run_runner(
    export_path: str,
    study_id: str = "study",
    mode: str = "analysis",
    question: Optional[str] = None,
    project_root: Optional[str] = None,
    run_type: str = "preflight",
    llm_provider: str = "mock",
    external_llm_opt_in: bool = False,
    ai_enabled: bool = False,
    study_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a deterministic local runner for ``JobManager``."""
    resolved_run_type = normalize_run_type(run_type)
    context_binding = None
    resolved_question = question
    if study_context is not None:
        context_binding = context_store.build_agent_context_binding(
            study_context,
            export_path=export_path,
            request_question=question,
        )
        resolved_question = context_binding["applied"]["question"]
    provider = resolve_agent_provider_config(
        run_type=resolved_run_type,
        llm_provider=llm_provider,
        external_llm_opt_in=external_llm_opt_in,
        ai_enabled=ai_enabled,
    )

    def runner(job: Any) -> Dict[str, Any]:
        started = time.time()
        run_id = f"run_{job.id}"
        safe_study = _slug(study_id or "study")
        root = (
            Path(project_root).expanduser()
            if project_root
            else Path.home() / "easyicu" / "projects"
        )
        run_dir = root / safe_study / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        total_steps = 5 if resolved_run_type == "full" else 4

        job.emit(
            {
                "type": "start",
                "run_id": run_id,
                "study_id": study_id,
                "mode": mode,
                "run_type": resolved_run_type,
                "provider": dict(provider),
                "study_context_id": (
                    context_binding.get("study_context_id")
                    if context_binding is not None
                    else None
                ),
                "local_only": True,
                "uploads": 0,
                "tokens": 0,
            }
        )
        if getattr(job, "cancel_requested", False):
            return _cancelled_agent_result(
                job,
                run_id=run_id,
                study_id=study_id,
                mode=mode,
                run_type=resolved_run_type,
                run_dir=run_dir,
                provider=provider,
                phase="initializing",
            )

        source = dataio.describe_export_source(export_path)
        if not source.get("ok"):
            raise ValueError(str(source.get("error") or "invalid_export"))
        job.emit(
            {
                "type": "progress",
                "current": 1,
                "total": total_steps,
                "step": "source",
                "label": "Source registry resolved",
                "summary": source.get("summary", {}),
            }
        )
        if getattr(job, "cancel_requested", False):
            return _cancelled_agent_result(
                job,
                run_id=run_id,
                study_id=study_id,
                mode=mode,
                run_type=resolved_run_type,
                run_dir=run_dir,
                provider=provider,
                phase="source",
                source=source,
            )

        if _use_metadata_workspace(source, resolved_run_type):
            workspace = _metadata_workspace(source)
        else:
            workspace = dataio.summarize_export_workspace(export_path)
        if not workspace.get("ok"):
            raise ValueError(str(workspace.get("error") or "workspace_summary_failed"))
        summary = dict(workspace.get("summary") or {})
        cohort = dict(workspace.get("cohort") or {})
        quality = [_quality_public(q) for q in workspace.get("quality", [])]
        job.emit(
            {
                "type": "progress",
                "current": 2,
                "total": total_steps,
                "step": "snapshot",
                "label": (
                    "Export metadata snapshot resolved"
                    if summary.get("snapshot_basis") == "registry_metadata"
                    else "Export snapshot summarised"
                ),
                "stays": summary.get("stays"),
                "modules": summary.get("modules"),
                "snapshot_basis": summary.get(
                    "snapshot_basis", "bounded_row_level_sample"
                ),
            }
        )
        if getattr(job, "cancel_requested", False):
            return _cancelled_agent_result(
                job,
                run_id=run_id,
                study_id=study_id,
                mode=mode,
                run_type=resolved_run_type,
                run_dir=run_dir,
                provider=provider,
                phase="snapshot",
                source=source,
                summary=summary,
            )

        run_context = {
            "run_id": run_id,
            "study_id": study_id,
            "mode": mode,
            "question": resolved_question,
            "source": {
                "path": source.get("path"),
                "label": source.get("label"),
                "database": source.get("database"),
                "generated": source.get("generated"),
                "modules": source.get("modules", []),
            },
            "summary": summary,
            "local_first": {"uploads": 0, "tokens": 0},
        }
        if context_binding is not None:
            run_context["context_binding"] = context_binding
        artifacts = {
            "run_context.json": run_context,
            "cohort_summary.json": {
                "summary": summary,
                "cohort": cohort,
            },
            "quality_gate.json": {
                "quality": quality,
            },
        }
        artifacts.update(
            agent_outputs.build_agent_output_artifacts(
                export_path=export_path,
                source=source,
                summary=summary,
                cohort=cohort,
                quality=quality,
            )
        )
        strict_audit = None
        numeric_audit = None
        if getattr(job, "cancel_requested", False):
            return _cancelled_agent_result(
                job,
                run_id=run_id,
                study_id=study_id,
                mode=mode,
                run_type=resolved_run_type,
                run_dir=run_dir,
                provider=provider,
                phase="planning",
                source=source,
                summary=summary,
            )
        if resolved_run_type == "full":
            if provider.get("external"):
                provider_result = provider_adapter.generate_bound_provider_payload(
                    provider_meta=provider,
                    run_id=run_id,
                    study_id=study_id,
                    question=resolved_question,
                    summary=summary,
                    cohort=cohort,
                    quality=quality,
                    output_artifacts={
                        name: artifacts[name]
                        for name in agent_outputs.OUTPUT_ARTIFACT_NAMES
                        if name in artifacts
                    },
                )
                provider.update(provider_result["provider"])
                full_payload = {
                    "agent_plan": provider_result["agent_plan"],
                    "manuscript_draft": provider_result["manuscript_draft"],
                }
                progress_step = "full_agent_provider"
                progress_label = "External provider scaffold generated"
            else:
                provider["mock_calls"] = 1
                full_payload = _mock_full_agent_payload(
                    run_id=run_id,
                    study_id=study_id,
                    question=resolved_question,
                    summary=summary,
                    cohort=cohort,
                    quality=quality,
                )
                progress_step = "full_agent_mock"
                progress_label = "Mock full-agent scaffold generated"
            artifacts["agent_plan.json"] = full_payload["agent_plan"]
            artifacts["manuscript_draft.json"] = full_payload["manuscript_draft"]
            job.emit(
                {
                    "type": "progress",
                    "current": 3,
                    "total": total_steps,
                    "step": progress_step,
                    "label": progress_label,
                    "provider": dict(provider),
                }
            )
            if getattr(job, "cancel_requested", False):
                return _cancelled_agent_result(
                    job,
                    run_id=run_id,
                    study_id=study_id,
                    mode=mode,
                    run_type=resolved_run_type,
                    run_dir=run_dir,
                    provider=provider,
                    phase=progress_step,
                    source=source,
                    summary=summary,
                )

        gate, privacy_scan, strict_audit, numeric_audit = _evaluate_gate_with_ledger(
            run_id=run_id,
            run_dir=run_dir,
            artifacts=artifacts,
            source=source,
            summary=summary,
            quality=quality,
            run_type=resolved_run_type,
            provider=provider,
            strict_audit=strict_audit,
            numeric_audit=numeric_audit,
        )

        persisted_artifacts, privacy_scan = _privacy_safe_artifacts(
            artifacts, privacy_scan
        )
        if not privacy_scan.get("passed"):
            # Audits are derived from the original artifacts and can copy a
            # patient identifier into an otherwise metadata-only ledger. A
            # privacy failure therefore persists only a newly built minimal
            # gate and drops every derived audit payload.
            gate = persisted_artifacts["quality_gate.json"]["gate"]
            strict_audit = None
            numeric_audit = None

        job.emit(
            {
                "type": "gate",
                "current": total_steps - 1,
                "total": total_steps,
                "step": "gate",
                "label": "Evidence gate evaluated",
                "gate": gate,
            }
        )
        if getattr(job, "cancel_requested", False):
            return _cancelled_agent_result(
                job,
                run_id=run_id,
                study_id=study_id,
                mode=mode,
                run_type=resolved_run_type,
                run_dir=run_dir,
                provider=provider,
                phase="gate",
                source=source,
                summary=summary,
            )

        written = [
            _artifact_payload(
                name,
                run_dir,
                payload,
                payload.get("summary") if isinstance(payload, dict) else None,
            )
            for name, payload in persisted_artifacts.items()
        ]
        ledger = _ledger_payload(
            run_id,
            gate,
            written,
            privacy_scan,
            resolved_run_type,
            provider,
            strict_audit,
            numeric_audit,
        )
        final_scan = _scan_artifact_payloads(
            {**persisted_artifacts, "evidence_ledger.json": ledger}
        )
        if not final_scan.get("passed"):
            persisted_artifacts, privacy_scan = _privacy_safe_artifacts(
                {**artifacts, "evidence_ledger.json": ledger}, final_scan
            )
            gate = persisted_artifacts["quality_gate.json"]["gate"]
            strict_audit = None
            numeric_audit = None
            written = [
                _artifact_payload(name, run_dir, payload)
                for name, payload in persisted_artifacts.items()
            ]
            ledger = _ledger_payload(
                run_id,
                gate,
                written,
                privacy_scan,
                resolved_run_type,
                provider,
                None,
                None,
            )
            if not _scan_artifact_payloads(
                {**persisted_artifacts, "evidence_ledger.json": ledger}
            ).get("passed"):
                raise RuntimeError("privacy failure package did not pass final scan")

        for name, payload in persisted_artifacts.items():
            out = run_dir / name
            _write_json(out, payload)
        ledger_path = run_dir / "evidence_ledger.json"
        _write_json(ledger_path, ledger)
        written.append(_artifact(ledger_path, run_dir))

        job.emit(
            {
                "type": "artifact",
                "current": total_steps,
                "total": total_steps,
                "step": "artifacts",
                "label": (
                    "Safe local artifacts written; row-level payloads withheld"
                    if privacy_scan.get("payloads_withheld")
                    else "Local artifacts written"
                ),
                "artifacts": written,
            }
        )

        return {
            "run_id": run_id,
            "run_label": run_id.replace("_", " "),
            "study_id": study_id,
            "mode": mode,
            "run_type": resolved_run_type,
            "project_dir": str(run_dir),
            "source": {
                "path": source.get("path"),
                "label": source.get("label"),
                "database": source.get("database"),
            },
            "summary": summary,
            "cohort": cohort,
            "quality": quality,
            "gate": gate,
            "provider": provider,
            "study_context_id": (
                context_binding.get("study_context_id")
                if context_binding is not None
                else None
            ),
            "strict_evidence_audit": strict_audit,
            "numeric_evidence_audit": numeric_audit,
            "artifacts": written,
            "duration_sec": round(time.time() - started, 2),
            "uploads": 0,
            "tokens": 0,
        }

    return runner


def project_artifact_governance(review: Mapping[str, Any]) -> Dict[str, Any]:
    """Project one run review into the browser-safe artifact authority contract."""

    readiness = review.get("readiness")
    if not isinstance(readiness, Mapping):
        return {
            "ok": False,
            "error": "run_artifact_governance_readiness_invalid",
        }
    gate = review.get("gate")
    gate = gate if isinstance(gate, Mapping) else {}
    readiness_status = str(readiness.get("status") or "unknown")
    signed = bool(readiness.get("signed"))
    signoff_stale = bool(readiness.get("signoff_stale"))
    reportable = bool(readiness.get("reportable"))
    if signoff_stale:
        human_signoff = "stale"
    elif signed:
        human_signoff = "signed"
    elif readiness_status == "awaiting_human_signoff":
        human_signoff = "required"
    else:
        human_signoff = "not_signable"
    claim_ceiling = "reportable" if reportable else "unsupported"
    if (
        not reportable
        and gate.get("status") == "analysis_only"
        and readiness_status in {"awaiting_human_signoff", "signed_analysis_only"}
    ):
        claim_ceiling = "analysis_only"
    return {
        "ok": True,
        "authority_class": "easyicu_run_artifact",
        "gate_status": gate.get("status"),
        "readiness_status": readiness_status,
        "human_signoff": human_signoff,
        "reportable": reportable,
        "claim_ceiling": claim_ceiling,
    }


def read_run_review(project_dir: str) -> Dict[str, Any]:
    """Read the bounded artifact set for a local agent run review screen."""
    run_dir = _resolve_run_dir(project_dir)
    if run_dir is None:
        return {"ok": False, "error": "project_dir_required"}
    if not run_dir.exists() or not run_dir.is_dir():
        return {"ok": False, "error": "run_dir_not_found", "project_dir": str(run_dir)}

    loaded = _load_run_artifacts(run_dir)
    if not loaded.get("ok"):
        return loaded
    payloads = loaded["payloads"]
    if "quality_gate.json" not in payloads or "evidence_ledger.json" not in payloads:
        return {"ok": False, "error": "not_agent_run_dir", "project_dir": str(run_dir)}

    run_context = payloads.get("run_context.json") or {}
    ledger = payloads.get("evidence_ledger.json") or {}
    gate = (payloads.get("quality_gate.json") or {}).get("gate") or {}
    signoff = payloads.get("human_signoff.json")

    artifacts = _run_artifacts(run_dir)
    signoff_integrity = _signoff_integrity(signoff, artifacts)
    readiness = _readiness_from_gate(
        gate,
        signed=bool(signoff),
        signoff_stale=bool(signoff_integrity.get("signoff_stale")),
    )

    return {
        "ok": True,
        "project_dir": str(run_dir),
        "run_id": run_context.get("run_id") or ledger.get("run_id"),
        "run_type": ledger.get("run_type") or "preflight",
        "study_id": run_context.get("study_id"),
        "mode": run_context.get("mode"),
        "gate": gate,
        "readiness": readiness,
        "signed": bool(signoff),
        "signoff_stale": bool(signoff_integrity.get("signoff_stale")),
        "signoff_integrity": signoff_integrity,
        "signoff": signoff,
        "artifacts": artifacts,
        "artifact_payloads": _public_review_payloads(payloads),
    }


def create_human_signoff(
    project_dir: str,
    reviewer: Optional[str] = None,
    confirmations: Optional[List[str]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a local human signoff artifact without unlocking the draft."""
    review = read_run_review(project_dir)
    if not review.get("ok"):
        return review
    if review.get("signed"):
        return review

    readiness = review.get("readiness") or {}
    if not readiness.get("signable"):
        return {
            "ok": False,
            "error": "readiness_gate_not_signable",
            "project_dir": review.get("project_dir"),
            "readiness": readiness,
        }

    provided = {
        str(item).strip() for item in (confirmations or []) if str(item).strip()
    }
    missing = sorted(_SIGNOFF_CONFIRMATIONS - provided)
    if missing:
        return {
            "ok": False,
            "error": "missing_signoff_confirmations",
            "missing_confirmations": missing,
            "required_confirmations": sorted(_SIGNOFF_CONFIRMATIONS),
        }

    reviewer_text = re.sub(r"\s+", " ", str(reviewer or "local_reviewer")).strip()[:120]
    note_text = re.sub(r"\s+", " ", str(note or "")).strip()[:1000]
    gate = review.get("gate") or {}
    signoff = {
        "run_id": review.get("run_id"),
        "run_type": review.get("run_type"),
        "study_id": review.get("study_id"),
        "signed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reviewer": reviewer_text or "local_reviewer",
        "status": "signed_analysis_only",
        "scope": "local_human_review",
        "confirmations": sorted(provided),
        "note": note_text,
        "reportable": False,
        "draft_unlocked": False,
        "uploads": 0,
        "tokens": 0,
        "external_calls": 0,
        "gate_before_signoff": {
            "status": gate.get("status"),
            "reason": gate.get("reason"),
            "reportable": False,
            "draft_unlocked": False,
            "checks": gate.get("checks", []),
        },
        "artifact_names": [
            item.get("name")
            for item in review.get("artifacts", [])
            if item.get("name") != "human_signoff.json"
        ],
        "signed_artifacts": [
            {
                "name": item.get("name"),
                "sha256": item.get("sha256"),
                "bytes": item.get("bytes"),
            }
            for item in review.get("artifacts", [])
            if item.get("name") != "human_signoff.json"
        ],
    }
    signoff["privacy_scan"] = _scan_artifact_payloads({"human_signoff.json": signoff})
    if not signoff["privacy_scan"].get("passed"):
        return {
            "ok": False,
            "error": "signoff_privacy_scan_failed",
            "privacy_scan": signoff["privacy_scan"],
        }

    _write_json(Path(str(review["project_dir"])) / "human_signoff.json", signoff)
    return read_run_review(str(review["project_dir"]))


def list_run_history(
    study_id: Optional[str] = None,
    project_root: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List local agent run directories by reading whitelisted artifacts only."""
    root = (
        Path(project_root).expanduser()
        if project_root
        else Path.home() / "easyicu" / "projects"
    )
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        return {
            "ok": True,
            "project_root": str(root),
            "study_id": study_id,
            "runs": [],
            "count": 0,
        }
    study_dirs = []
    if study_id:
        study_dirs = [root / _slug(str(study_id))]
    else:
        study_dirs = [p for p in root.iterdir() if p.is_dir()]
    runs = []
    for study_dir in study_dirs:
        if not study_dir.exists() or not study_dir.is_dir():
            continue
        for run_dir in study_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            review = read_run_review(str(run_dir))
            if not review.get("ok"):
                continue
            runs.append(_history_row(review, run_dir))
    runs.sort(key=lambda row: row.get("updated_at_epoch") or 0, reverse=True)
    limit = max(1, min(int(limit or 50), 200))
    return {
        "ok": True,
        "project_root": str(root),
        "study_id": study_id,
        "runs": runs[:limit],
        "count": len(runs),
    }


def read_run_artifact(project_dir: str, artifact_name: str) -> Dict[str, Any]:
    """Return one whitelisted artifact as a bounded JSON viewer payload."""
    run_dir = _resolve_run_dir(project_dir)
    if run_dir is None:
        return {"ok": False, "error": "project_dir_required"}
    artifact_path = _safe_artifact_path(run_dir, artifact_name)
    if artifact_path is None:
        return {"ok": False, "error": "artifact_not_allowed", "artifact": artifact_name}
    if not artifact_path.exists() or not artifact_path.is_file():
        return {
            "ok": False,
            "error": "artifact_not_found",
            "project_dir": str(run_dir),
            "artifact": artifact_name,
        }
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": "artifact_json_invalid",
            "artifact": artifact_name,
            "message": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "artifact_json_not_object",
            "artifact": artifact_name,
        }
    privacy_scan = _scan_artifact_payloads({artifact_path.name: payload})
    return {
        "ok": True,
        "project_dir": str(run_dir),
        "artifact": _artifact(artifact_path, run_dir),
        "payload": _public_single_artifact_payload(artifact_path.name, payload),
        "privacy_scan": privacy_scan,
    }


def read_run_artifact_bytes(project_dir: str, artifact_name: str) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(project_dir)
    if run_dir is None:
        return {"ok": False, "error": "project_dir_required"}
    artifact_path = _safe_artifact_path(run_dir, artifact_name)
    if artifact_path is None:
        return {"ok": False, "error": "artifact_not_allowed", "artifact": artifact_name}
    if not artifact_path.exists() or not artifact_path.is_file():
        return {"ok": False, "error": "artifact_not_found", "artifact": artifact_name}
    return {
        "ok": True,
        "name": artifact_path.name,
        "content": artifact_path.read_bytes(),
        "media_type": "application/json",
    }


def build_run_bundle(project_dir: str) -> Dict[str, Any]:
    """Build a zip containing only whitelisted local run artifacts."""
    run_dir = _resolve_run_dir(project_dir)
    if run_dir is None:
        return {"ok": False, "error": "project_dir_required"}
    if not run_dir.exists() or not run_dir.is_dir():
        return {"ok": False, "error": "run_dir_not_found", "project_dir": str(run_dir)}
    artifacts = _run_artifacts(run_dir)
    if not artifacts:
        return {"ok": False, "error": "no_artifacts", "project_dir": str(run_dir)}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for artifact in artifacts:
            name = str(artifact.get("name") or "")
            path = _safe_artifact_path(run_dir, name)
            if path is not None and path.exists() and path.is_file():
                zf.writestr(name, path.read_bytes())
    filename = f"{run_dir.name}_artifacts.zip"
    return {
        "ok": True,
        "name": filename,
        "content": buffer.getvalue(),
        "media_type": "application/zip",
        "artifact_names": [a.get("name") for a in artifacts],
    }


def normalize_run_type(value: str) -> str:
    text = str(value or "preflight").strip().lower()
    if text in {"preflight", "analysis_only", "local_preflight"}:
        return "preflight"
    if text in {"full", "full_agent", "full_agent_mock"}:
        return "full"
    raise AgentRunConfigError({"error": "unsupported_run_type", "run_type": value})


def validate_agent_run_config(
    *,
    run_type: str,
    llm_provider: str,
    external_llm_opt_in: bool,
    ai_enabled: bool,
) -> None:
    """Fail closed before any external provider can be constructed."""
    resolve_agent_provider_config(
        run_type=run_type,
        llm_provider=llm_provider,
        external_llm_opt_in=external_llm_opt_in,
        ai_enabled=ai_enabled,
    )


def resolve_agent_provider_config(
    *,
    run_type: str,
    llm_provider: str,
    external_llm_opt_in: bool,
    ai_enabled: bool,
) -> Dict[str, Any]:
    """Resolve the provider boundary without loading credentials or clients."""
    resolved_run_type = normalize_run_type(run_type)
    try:
        provider = provider_gate.resolve_provider_gate(
            run_type=resolved_run_type,
            llm_provider=llm_provider,
            external_llm_opt_in=external_llm_opt_in,
            ai_enabled=ai_enabled,
        )
    except provider_gate.ProviderGateError as exc:
        raise AgentRunConfigError(exc.detail) from exc
    if resolved_run_type == "full" and provider.get("external"):
        try:
            provider = provider_adapter.require_external_credentials(provider)
        except provider_adapter.ProviderAdapterError as exc:
            raise AgentRunConfigError(exc.detail) from exc
    return provider


def _cancelled_agent_result(
    job: Any,
    *,
    run_id: str,
    study_id: str,
    mode: str,
    run_type: str,
    run_dir: Path,
    provider: Dict[str, Any],
    phase: str,
    source: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an honest terminal payload for a cooperatively cancelled run.

    A cancelled native web run is not a resumable Python thread after process
    shutdown. The safe continuation is to retry from the active export and use
    local artifact history for completed prior runs.
    """
    return {
        "run_id": run_id,
        "run_label": run_id.replace("_", " "),
        "study_id": study_id,
        "mode": mode,
        "run_type": normalize_run_type(run_type),
        "project_dir": str(run_dir),
        "cancelled": True,
        "cancelled_at": phase,
        "cancel_reason": getattr(job, "cancel_reason", None) or "user_requested",
        "resumable": True,
        "resume_kind": "restart_from_active_export",
        "resume_label": "Retry from the active registered export",
        "source": {
            "path": (source or {}).get("path"),
            "label": (source or {}).get("label"),
            "database": (source or {}).get("database"),
        },
        "summary": summary or {},
        "gate": {
            "status": "cancelled",
            "reportable": False,
            "draft_unlocked": False,
            "reason": "agent_run_cancelled_before_artifacts",
            "checks": [],
        },
        "provider": provider,
        "artifacts": [],
        "uploads": 0,
        "tokens": 0,
        "external_calls": int(provider.get("external_calls") or 0),
    }


def _quality_public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module": row.get("module"),
        "file": row.get("file"),
        "rows": row.get("rows"),
        "columns": row.get("columns"),
        "unique_stays": row.get("unique_stays"),
        "coverage_pct": row.get("coverage_pct"),
        "coverage_basis": row.get("coverage_basis"),
        "denominator": row.get("denominator"),
        "status": row.get("status"),
    }


def _use_metadata_workspace(source: Dict[str, Any], run_type: str) -> bool:
    if normalize_run_type(run_type) != "preflight":
        return False
    summary = source.get("summary") if isinstance(source, dict) else {}
    try:
        total_rows = int((summary or {}).get("total_rows") or 0)
    except (TypeError, ValueError):
        total_rows = 0
    return total_rows > _AGENT_PREFLIGHT_FULL_SCAN_ROW_LIMIT


def _metadata_workspace(source: Dict[str, Any]) -> Dict[str, Any]:
    source_summary = dict(source.get("summary") or {})
    files = [f for f in source.get("files", []) if isinstance(f, dict)]
    modules = sorted(
        {str(f.get("module") or "") for f in files if str(f.get("module") or "")}
    )
    stays = source_summary.get("stays")
    summary = {
        **source_summary,
        "stays": stays,
        "modules": source_summary.get("modules") or len(modules),
        "file_count": source_summary.get("file_count") or len(files),
        "total_rows": source_summary.get("total_rows"),
        "snapshot_basis": "registry_metadata",
        "artifact_scope": "metadata_only_large_export_preflight",
        "row_scan_skipped": True,
        "row_scan_skip_reason": "export_total_rows_exceeds_preflight_limit",
        "preflight_row_limit": _AGENT_PREFLIGHT_FULL_SCAN_ROW_LIMIT,
    }
    cohort = {
        "status": "metadata_only",
        "basis": "registry_manifest",
        "survived": None,
        "deceased": None,
        "characteristics": [
            {
                "label": "Cohort stays",
                "value": stays,
                "unit": "registry denominator",
            },
            {
                "label": "Export modules",
                "value": summary.get("modules"),
                "unit": "manifest modules",
            },
            {
                "label": "Declared rows",
                "value": summary.get("total_rows"),
                "unit": "manifest rows",
            },
        ],
    }
    denominator = _safe_positive_int(stays)
    quality = []
    for file_meta in files:
        rows = _safe_positive_int(file_meta.get("rows"))
        column_count = len(file_meta.get("columns") or [])
        quality.append(
            {
                "module": file_meta.get("module"),
                "file": file_meta.get("file"),
                "rows": rows,
                "columns": column_count,
                "unique_stays": None,
                "coverage_pct": None,
                "coverage_basis": "manifest_file_inventory",
                "denominator": denominator,
                "status": "metadata_only",
            }
        )
    return {
        "ok": True,
        "path": source.get("path"),
        "database": source.get("database"),
        "generated": source.get("generated"),
        "files": files,
        "summary": summary,
        "quality": quality,
        "cohort": cohort,
    }


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _quality_audit_passed(quality: List[Dict[str, Any]], run_type: str) -> bool:
    allowed = {"unique_stay_id_intersection"}
    if normalize_run_type(run_type) == "preflight":
        allowed.add("manifest_file_inventory")
    return all(q.get("coverage_basis") in allowed for q in quality)


def _gate(
    source: Dict[str, Any],
    summary: Dict[str, Any],
    quality: List[Dict[str, Any]],
    privacy_scan: Dict[str, Any],
    run_type: str = "preflight",
    provider: Optional[Dict[str, Any]] = None,
    strict_audit: Optional[Dict[str, Any]] = None,
    numeric_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bad_modules = [
        str(q.get("module") or q.get("file"))
        for q in quality
        if q.get("status") == "bad"
    ]
    checks = [
        {
            "id": "source_valid",
            "label": "Export source valid",
            "passed": bool(source.get("ok") and source.get("path")),
            "evidence": "registry description",
        },
        {
            "id": "denominator_resolved",
            "label": "Cohort denominator resolved",
            "passed": bool(summary.get("stays")),
            "value": summary.get("stays"),
        },
        {
            "id": "quality_audited",
            "label": "Module stay-id coverage audited",
            "passed": _quality_audit_passed(quality, run_type),
            "modules": len(quality),
            "coverage_bases": sorted(
                {str(q.get("coverage_basis") or "") for q in quality}
            ),
        },
        {
            "id": "no_bad_non_event_coverage",
            "label": "No non-event module below coverage threshold",
            "passed": not bad_modules,
            "bad_modules": bad_modules,
        },
        {
            "id": "no_patient_rows_persisted",
            "label": "No patient rows persisted in agent artifacts",
            "passed": bool(privacy_scan.get("passed")),
            "evidence": "artifact_json_scan",
            "scanned_artifacts": privacy_scan.get("scanned_artifacts"),
            "row_level_markers": privacy_scan.get("row_level_markers", []),
        },
    ]
    if normalize_run_type(run_type) == "full":
        provider_ready = bool(
            provider
            and (
                not provider.get("external")
                or provider.get("provider_gate") == "external_provider_ready"
            )
        )
        checks.extend(
            [
                {
                    "id": "provider_opt_in",
                    "label": "LLM provider path resolved before invocation",
                    "passed": provider_ready,
                    "provider": (provider or {}).get("provider"),
                    "external": bool((provider or {}).get("external")),
                    "evidence": (
                        "external_provider_adapter_after_opt_in"
                        if bool((provider or {}).get("external"))
                        else "offline_mock"
                    ),
                    "credentials_loaded": bool(
                        (provider or {}).get("credentials_loaded")
                    ),
                    "client_constructed": bool(
                        (provider or {}).get("client_constructed")
                    ),
                },
                {
                    "id": "strict_evidence_bound_claims",
                    "label": "All manuscript claims bind to known evidence",
                    "passed": bool(strict_audit and strict_audit.get("claims_passed")),
                    "claim_count": (strict_audit or {}).get("claim_count", 0),
                    "unbound_claims": (strict_audit or {}).get("unbound_claims", []),
                    "missing_evidence": (strict_audit or {}).get(
                        "missing_evidence", []
                    ),
                },
                {
                    "id": "strict_evidence_bound_sentences",
                    "label": "All manuscript sentences bind to known evidence",
                    "passed": bool(
                        strict_audit and strict_audit.get("sentences_passed")
                    ),
                    "sentence_count": (strict_audit or {}).get("sentence_count", 0),
                    "unbound_sentences": (strict_audit or {}).get(
                        "unbound_sentences", []
                    ),
                    "missing_evidence": (strict_audit or {}).get(
                        "missing_evidence", []
                    ),
                },
                {
                    "id": "numeric_evidence_value_binding",
                    "label": "All numeric manuscript claims match artifact values",
                    "passed": bool(numeric_audit and numeric_audit.get("passed")),
                    "numeric_claim_count": (numeric_audit or {}).get(
                        "numeric_claim_count", 0
                    ),
                    "numeric_sentence_count": (numeric_audit or {}).get(
                        "numeric_sentence_count", 0
                    ),
                    "numeric_mention_count": (numeric_audit or {}).get(
                        "numeric_mention_count", 0
                    ),
                    "failure_count": (numeric_audit or {}).get("failure_count", 0),
                    "failures": (numeric_audit or {}).get("failures", []),
                    "tolerance_policy": (numeric_audit or {}).get(
                        "tolerance_policy", {}
                    ),
                },
            ]
        )
    checks.append(
        {
            "id": "human_signoff",
            "label": "Human sign-off before manuscript claims",
            "passed": False,
        }
    )
    hard_fail = any(not c["passed"] for c in checks if c["id"] != "human_signoff")
    return {
        "status": "blocked" if hard_fail else "analysis_only",
        "reportable": False,
        "draft_unlocked": False,
        "reason": _gate_reason(checks, hard_fail, run_type),
        "checks": checks,
    }


def _gate_reason(checks: List[Dict[str, Any]], hard_fail: bool, run_type: str) -> str:
    if not hard_fail:
        if normalize_run_type(run_type) == "full":
            return "full_agent_complete_human_signoff_required"
        return "preflight_complete_human_signoff_required"
    failed = {str(c.get("id")) for c in checks if not c.get("passed")}
    if "numeric_evidence_value_binding" in failed:
        return "numeric_evidence_gate_failed"
    if any(item.startswith("strict_evidence") for item in failed):
        return "strict_evidence_gate_failed"
    if "no_patient_rows_persisted" in failed:
        return "privacy_gate_failed"
    return "coverage_gate_failed"


def _resolve_run_dir(project_dir: str) -> Optional[Path]:
    text = str(project_dir or "").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _safe_artifact_path(run_dir: Path, artifact_name: str) -> Optional[Path]:
    name = str(artifact_name or "").strip()
    if name not in _RUN_ARTIFACT_NAMES:
        return None
    if Path(name).name != name:
        return None
    return run_dir / name


def _run_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    artifacts = []
    for name in _RUN_ARTIFACT_NAMES:
        path = run_dir / name
        if path.exists() and path.is_file():
            artifacts.append(_artifact(path, run_dir))
    return artifacts


def _load_run_artifacts(run_dir: Path) -> Dict[str, Any]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for name in _RUN_ARTIFACT_NAMES:
        path = run_dir / name
        if not path.exists():
            continue
        if not path.is_file():
            return {"ok": False, "error": "artifact_path_not_file", "artifact": name}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error": "artifact_json_invalid",
                "artifact": name,
                "message": str(exc),
            }
        if isinstance(payload, dict):
            payloads[name] = payload
        else:
            return {"ok": False, "error": "artifact_json_not_object", "artifact": name}
    return {"ok": True, "payloads": payloads}


def _readiness_from_gate(
    gate: Dict[str, Any],
    signed: bool = False,
    signoff_stale: bool = False,
) -> Dict[str, Any]:
    checks = gate.get("checks") if isinstance(gate, dict) else []
    if not isinstance(checks, list):
        checks = []
    non_human_failures = [
        str(check.get("id") or "check")
        for check in checks
        if isinstance(check, dict)
        and check.get("id") != "human_signoff"
        and not check.get("passed")
    ]
    human_check = next(
        (
            check
            for check in checks
            if isinstance(check, dict) and check.get("id") == "human_signoff"
        ),
        None,
    )
    eligible = bool(checks) and not non_human_failures
    if not eligible:
        status = "blocked"
    elif signoff_stale:
        status = "signoff_stale"
    elif signed:
        status = "signed_analysis_only"
    else:
        status = "awaiting_human_signoff"
    return {
        "status": status,
        "signable": bool(eligible and not signed),
        "signed": bool(signed),
        "signoff_stale": bool(signoff_stale),
        "reportable": False,
        "draft_unlocked": False,
        "gate_status": gate.get("status") if isinstance(gate, dict) else None,
        "gate_reason": gate.get("reason") if isinstance(gate, dict) else None,
        "checks_total": len(checks),
        "checks_passed": sum(
            1 for check in checks if isinstance(check, dict) and check.get("passed")
        ),
        "non_human_failures": non_human_failures,
        "human_signoff_passed_in_gate": bool(human_check and human_check.get("passed")),
        "required_confirmations": sorted(_SIGNOFF_CONFIRMATIONS),
    }


def _signoff_integrity(
    signoff: Optional[Dict[str, Any]],
    artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not signoff:
        return {
            "status": "unsigned",
            "signoff_stale": False,
            "checked_artifacts": 0,
            "tampered_artifacts": [],
            "missing_artifacts": [],
        }
    signed = signoff.get("signed_artifacts")
    if not isinstance(signed, list) or not signed:
        return {
            "status": "unverifiable",
            "signoff_stale": True,
            "reason": "missing_signed_artifact_hashes",
            "checked_artifacts": 0,
            "tampered_artifacts": [],
            "missing_artifacts": [],
        }
    current = {
        str(item.get("name")): item
        for item in artifacts
        if item.get("name") != "human_signoff.json"
    }
    tampered = []
    missing = []
    checked = 0
    for item in signed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        current_item = current.get(name)
        if current_item is None:
            missing.append(name)
            continue
        checked += 1
        if str(item.get("sha256") or "") != str(
            current_item.get("sha256") or ""
        ) or int(item.get("bytes") or -1) != int(current_item.get("bytes") or -2):
            tampered.append(
                {
                    "name": name,
                    "signed_sha256": item.get("sha256"),
                    "current_sha256": current_item.get("sha256"),
                    "signed_bytes": item.get("bytes"),
                    "current_bytes": current_item.get("bytes"),
                }
            )
    stale = bool(tampered or missing)
    return {
        "status": "stale" if stale else "verified",
        "signoff_stale": stale,
        "checked_artifacts": checked,
        "tampered_artifacts": tampered,
        "missing_artifacts": missing,
    }


def _history_row(review: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    readiness = review.get("readiness") or {}
    gate = review.get("gate") or {}
    artifacts = review.get("artifacts") or []
    updated = max((run_dir / str(a.get("name"))).stat().st_mtime for a in artifacts)
    return {
        "run_id": review.get("run_id") or run_dir.name,
        "run_label": str(review.get("run_id") or run_dir.name).replace("_", " "),
        "study_id": review.get("study_id"),
        "mode": review.get("mode"),
        "run_type": review.get("run_type"),
        "project_dir": review.get("project_dir"),
        "gate_status": gate.get("status"),
        "readiness_status": readiness.get("status"),
        "signed": bool(review.get("signed")),
        "signoff_stale": bool(review.get("signoff_stale")),
        "integrity_status": (review.get("signoff_integrity") or {}).get("status"),
        "reportable": False,
        "draft_unlocked": False,
        "artifact_count": len(artifacts),
        "artifact_names": [a.get("name") for a in artifacts],
        "updated_at_epoch": updated,
        "updated_at": datetime.fromtimestamp(updated, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _public_review_payloads(
    payloads: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    public: Dict[str, Dict[str, Any]] = {}
    if "run_context.json" in payloads:
        row = payloads["run_context.json"]
        public["run_context.json"] = {
            "run_id": row.get("run_id"),
            "study_id": row.get("study_id"),
            "mode": row.get("mode"),
            "question": row.get("question"),
            "summary": row.get("summary"),
            "local_first": row.get("local_first"),
            "source": row.get("source"),
            "context_binding": row.get("context_binding"),
        }
    if "cohort_summary.json" in payloads:
        row = payloads["cohort_summary.json"]
        public["cohort_summary.json"] = {
            "summary": row.get("summary"),
            "cohort": row.get("cohort"),
        }
    for name in agent_outputs.OUTPUT_ARTIFACT_NAMES:
        if name in payloads:
            public[name] = payloads[name]
    if "quality_gate.json" in payloads:
        row = payloads["quality_gate.json"]
        public["quality_gate.json"] = {
            "gate": row.get("gate"),
            "quality": row.get("quality"),
        }
    if "agent_plan.json" in payloads:
        public["agent_plan.json"] = payloads["agent_plan.json"]
    if "manuscript_draft.json" in payloads:
        row = payloads["manuscript_draft.json"]
        public["manuscript_draft.json"] = {
            "run_id": row.get("run_id"),
            "status": row.get("status"),
            "question": row.get("question"),
            "claims": row.get("claims", []),
            "sentences": row.get("sentences", []),
        }
    for name in (
        "benchmark_scorecard.json",
        "workflow_graph.json",
        "figure_gallery.json",
        "source_run_manifest.json",
    ):
        if name in payloads:
            public[name] = payloads[name]
    if "evidence_ledger.json" in payloads:
        row = payloads["evidence_ledger.json"]
        public["evidence_ledger.json"] = {
            "run_id": row.get("run_id"),
            "run_type": row.get("run_type"),
            "status": row.get("status"),
            "artifacts": row.get("artifacts", []),
            "provider": row.get("provider", {}),
            "strict_evidence_audit": row.get("strict_evidence_audit"),
            "numeric_evidence_audit": row.get("numeric_evidence_audit"),
            "privacy": row.get("privacy", {}),
        }
    if "human_signoff.json" in payloads:
        public["human_signoff.json"] = payloads["human_signoff.json"]
    return public


def _public_single_artifact_payload(
    name: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    public = _public_review_payloads({name: payload}).get(name)
    if public is not None:
        return public
    return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _ledger_payload(
    run_id: str,
    gate: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    privacy_scan: Dict[str, Any],
    run_type: str = "preflight",
    provider: Optional[Dict[str, Any]] = None,
    strict_audit: Optional[Dict[str, Any]] = None,
    numeric_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detected = not bool(privacy_scan.get("passed"))
    withheld = list(privacy_scan.get("payloads_withheld") or [])
    return {
        "run_id": run_id,
        "run_type": normalize_run_type(run_type),
        "status": gate["status"],
        "artifacts": artifacts,
        "provider": provider or {},
        "strict_evidence_audit": strict_audit,
        "numeric_evidence_audit": numeric_audit,
        "privacy": {
            "patient_rows_detected": detected,
            "patient_rows_persisted": bool(detected and not withheld),
            "payloads_withheld": withheld,
            "ui_preview_payload_excluded": True,
            "uploads": 0,
            "tokens": 0,
            "artifact_scan": privacy_scan,
        },
    }


def _evaluate_gate_with_ledger(
    *,
    run_id: str,
    run_dir: Path,
    artifacts: Dict[str, Dict[str, Any]],
    source: Dict[str, Any],
    summary: Dict[str, Any],
    quality: List[Dict[str, Any]],
    run_type: str,
    provider: Dict[str, Any],
    strict_audit: Optional[Dict[str, Any]],
    numeric_audit: Optional[Dict[str, Any]],
) -> tuple[
    Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]
]:
    if normalize_run_type(run_type) == "full":
        strict_audit = _strict_evidence_audit(artifacts)
        numeric_audit = numeric_evidence_audit.audit_numeric_evidence(artifacts)
    privacy_scan = _scan_artifact_payloads(artifacts)
    gate = _gate(
        source,
        summary,
        quality,
        privacy_scan,
        run_type,
        provider,
        strict_audit,
        numeric_audit,
    )
    artifacts["quality_gate.json"]["gate"] = gate
    # Gate metadata is itself persisted inside quality_gate.json and mirrored
    # inside the ledger, so rescan a bounded fixed point before writing.
    for _ in range(2):
        written_preview = [
            _artifact_payload(name, run_dir, payload, payload.get("summary"))
            for name, payload in artifacts.items()
        ]
        ledger = _ledger_payload(
            run_id,
            gate,
            written_preview,
            privacy_scan,
            run_type,
            provider,
            strict_audit,
            numeric_audit,
        )
        privacy_scan = _scan_artifact_payloads(
            {**artifacts, "evidence_ledger.json": ledger}
        )
        gate = _gate(
            source,
            summary,
            quality,
            privacy_scan,
            run_type,
            provider,
            strict_audit,
            numeric_audit,
        )
        artifacts["quality_gate.json"]["gate"] = gate
    return gate, privacy_scan, strict_audit, numeric_audit


def _mock_full_agent_payload(
    *,
    run_id: str,
    study_id: str,
    question: Optional[str],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    evidence_ids = ["run_context.json", "cohort_summary.json", "quality_gate.json"]
    claim_text = (
        "The active export snapshot contains "
        f"{summary.get('stays', 'unknown')} stays across "
        f"{summary.get('modules', 'unknown')} modules."
    )
    return {
        "agent_plan": {
            "run_id": run_id,
            "study_id": study_id,
            "provider": "MockLLMClient",
            "execution": "full_agent_mock_skeleton",
            "steps": [
                {
                    "id": "snapshot",
                    "title": "Resolve registry export snapshot",
                    "evidence_ids": ["run_context.json", "cohort_summary.json"],
                },
                {
                    "id": "gate",
                    "title": "Bind every draft sentence to artifact evidence",
                    "evidence_ids": evidence_ids,
                },
            ],
        },
        "manuscript_draft": {
            "run_id": run_id,
            "status": "locked_until_human_signoff",
            "question": question,
            "claims": [
                {
                    "claim_id": "claim_001",
                    "text": claim_text,
                    "evidence_ids": ["cohort_summary.json"],
                },
                {
                    "claim_id": "claim_002",
                    "text": "Module stay-id coverage was audited before drafting.",
                    "evidence_ids": ["quality_gate.json"],
                },
            ],
            "sentences": [
                {
                    "sentence_id": "sent_001",
                    "text": claim_text,
                    "evidence_ids": ["cohort_summary.json"],
                },
                {
                    "sentence_id": "sent_002",
                    "text": "This scaffold is not reportable until strict evidence checks and human sign-off pass.",
                    "evidence_ids": ["quality_gate.json"],
                },
            ],
            "bounded_inputs": {
                "summary": summary,
                "cohort_groups": {
                    "survived": cohort.get("survived"),
                    "deceased": cohort.get("deceased"),
                },
                "quality_modules": [q.get("module") for q in quality],
            },
        },
    }


def _strict_evidence_audit(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    evidence_ids = set(artifacts.keys())
    draft = artifacts.get("manuscript_draft.json") or {}
    claims = draft.get("claims") if isinstance(draft, dict) else []
    sentences = draft.get("sentences") if isinstance(draft, dict) else []
    if not isinstance(claims, list):
        claims = []
    if not isinstance(sentences, list):
        sentences = []

    unbound_claims = []
    unbound_sentences = []
    missing_evidence = []
    for row in claims:
        if not isinstance(row, dict):
            continue
        evidence = [str(e) for e in row.get("evidence_ids") or []]
        if not evidence:
            unbound_claims.append(
                str(row.get("claim_id") or row.get("text") or "claim")
            )
        for item in evidence:
            if item not in evidence_ids:
                missing_evidence.append(
                    {
                        "owner": str(row.get("claim_id") or "claim"),
                        "evidence_id": item,
                    }
                )
    for row in sentences:
        if not isinstance(row, dict):
            continue
        evidence = [str(e) for e in row.get("evidence_ids") or []]
        if not evidence:
            unbound_sentences.append(
                str(row.get("sentence_id") or row.get("text") or "sentence")
            )
        for item in evidence:
            if item not in evidence_ids:
                missing_evidence.append(
                    {
                        "owner": str(row.get("sentence_id") or "sentence"),
                        "evidence_id": item,
                    }
                )
    return {
        "mode": "strict",
        "evidence_ids": sorted(evidence_ids),
        "claim_count": len(claims),
        "sentence_count": len(sentences),
        "claims_passed": bool(claims) and not unbound_claims and not missing_evidence,
        "sentences_passed": bool(sentences)
        and not unbound_sentences
        and not missing_evidence,
        "unbound_claims": unbound_claims,
        "unbound_sentences": unbound_sentences,
        "missing_evidence": missing_evidence,
    }


_ROW_LEVEL_KEYS = {
    "tablerows",
    "series",
    "patient",
    "patients",
    "patientid",
    "patientids",
    "stayid",
    "stayids",
    "subjectid",
    "subjectids",
    "hadmid",
    "hadmids",
    "entityid",
    "entityids",
}


def _scan_artifact_payloads(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    hits = []
    for name, payload in artifacts.items():
        try:
            # Scan the same JSON-shaped tree that will be written. Python
            # tuples, non-string dict keys, and other json.dumps coercions must
            # not create a scanner/writer type gap.
            canonical = json.loads(_json_bytes(payload).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError):
            hits.append({"path": name, "marker": "non_json_serializable"})
            continue
        hits.extend(_row_level_markers(canonical, name))
    return {
        "passed": not hits,
        "scanned_artifacts": len(artifacts),
        "row_level_markers": hits[:50],
    }


def _privacy_safe_artifacts(
    artifacts: Dict[str, Dict[str, Any]],
    privacy_scan: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Return the minimal failure package allowed after a privacy violation."""
    if privacy_scan.get("passed"):
        return artifacts, privacy_scan
    # Never reuse quality_gate.json from the unsafe bundle: strict/numeric
    # audit failures and even nested object paths can contain values copied
    # from row-level inputs. Build a fixed-schema failure package instead.
    hits = privacy_scan.get("row_level_markers")
    hits = hits if isinstance(hits, list) else []
    marker_types = sorted(
        {
            normalized
            for hit in hits
            if isinstance(hit, dict)
            for normalized in [
                re.sub(r"[^a-z0-9]+", "", str(hit.get("marker") or "").lower())
            ]
            if normalized in _ROW_LEVEL_KEYS
        }
    )
    safe_scan = {
        "passed": False,
        "scanned_artifacts": int(privacy_scan.get("scanned_artifacts") or 0),
        "row_level_marker_count": len(hits),
        "marker_types": marker_types,
        "payloads_withheld": sorted(str(name) for name in artifacts),
    }
    gate = {
        "status": "blocked",
        "reportable": False,
        "draft_unlocked": False,
        "reason": "privacy_gate_failed",
        "checks": [
            {
                "id": "no_patient_rows_persisted",
                "label": "No patient rows persisted in agent artifacts",
                "passed": False,
                "evidence": "artifact_json_scan",
                "scanned_artifacts": safe_scan["scanned_artifacts"],
                "row_level_marker_count": safe_scan["row_level_marker_count"],
                "marker_types": marker_types,
            },
            {
                "id": "human_signoff",
                "label": "Human sign-off before manuscript claims",
                "passed": False,
            },
        ],
    }
    safe = {
        "quality_gate.json": {
            "gate": gate,
            "privacy_failure": {
                "row_level_marker_count": safe_scan["row_level_marker_count"],
                "marker_types": marker_types,
                "payloads_withheld": safe_scan["payloads_withheld"],
            },
        }
    }
    if not _scan_artifact_payloads(safe).get("passed"):
        raise RuntimeError("minimal privacy failure gate is not metadata-only")
    return safe, safe_scan


def _row_level_markers(value: Any, path: str) -> List[Dict[str, str]]:
    hits: List[Dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized_key = re.sub(r"[^a-z0-9]+", "", str(key).lower())
            if normalized_key in _ROW_LEVEL_KEYS:
                hits.append({"path": child_path, "marker": key})
            hits.extend(_row_level_markers(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_row_level_markers(child, f"{path}[{index}]"))
    return hits


def _artifact_payload(
    name: str,
    root: Path,
    payload: Dict[str, Any],
    summary: Any = None,
) -> Dict[str, Any]:
    raw = _json_bytes(payload)
    out = {
        "name": name,
        "path": str(root / name),
        "relative_path": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": Path(name).suffix.lstrip(".") or "file",
    }
    if summary is not None:
        out["summary"] = summary
    return out


def _artifact(path: Path, root: Path, summary: Any = None) -> Dict[str, Any]:
    raw = path.read_bytes()
    out = {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": path.suffix.lstrip(".") or "file",
    }
    if summary is not None:
        out["summary"] = summary
    return out


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "study")).strip("._")
    return text[:80] or "study"
