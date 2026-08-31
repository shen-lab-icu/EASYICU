"""Browser disclosure authority for Research Agent run artefacts.

The module owns public claim ceilings, signoff integrity, privacy scanning, and
fixed fail-closed projections shared by pipeline writers and browser readers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _clean_text(value: Any, limit: int = 1_200) -> str:
    return re.sub(r"\\s+", " ", str(value or "")).strip()[:limit]


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


def project_artifact_governance(
    review: Mapping[str, Any],
    *,
    artifact: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
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
    artifact_integrity = None
    if artifact is not None:
        artifact_integrity = signed_artifact_integrity(review.get("signoff"), artifact)
        if signed and artifact_integrity != "verified":
            signoff_stale = True
    artifact_name = str((artifact or {}).get("name") or "")
    if artifact_name in {
        "system_validation_report.json",
        "system_validation_report_receipt.json",
        "system_validation_report.html",
        "system_validation_report.pdf",
    }:
        return {
            "ok": True,
            "authority_class": "easyicu_system_validation_report",
            "gate_status": gate.get("status"),
            "readiness_status": readiness_status,
            "human_signoff": "not_signable",
            "reportable": False,
            "claim_ceiling": "engineering_validation_only",
            **(
                {"artifact_integrity": artifact_integrity}
                if artifact_integrity is not None
                else {}
            ),
        }
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
        and not signoff_stale
        and gate.get("status") == "analysis_only"
        and readiness_status in {"awaiting_human_signoff", "signed_analysis_only"}
    ):
        claim_ceiling = "analysis_only"
    projection = {
        "ok": True,
        "authority_class": "easyicu_run_artifact",
        "gate_status": gate.get("status"),
        "readiness_status": readiness_status,
        "human_signoff": human_signoff,
        "reportable": reportable,
        "claim_ceiling": claim_ceiling,
    }
    if artifact_integrity is not None:
        projection["artifact_integrity"] = artifact_integrity
    return projection


def signoff_integrity(
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
            "unexpected_artifacts": [],
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
            "unexpected_artifacts": [],
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
    signed_names = {
        str(item.get("name") or "") for item in signed if isinstance(item, dict)
    }
    unexpected = sorted(name for name in current if name not in signed_names)
    stale = bool(tampered or missing or unexpected)
    return {
        "status": "stale" if stale else "verified",
        "signoff_stale": stale,
        "checked_artifacts": checked,
        "tampered_artifacts": tampered,
        "missing_artifacts": missing,
        "unexpected_artifacts": unexpected,
    }


def signed_artifact_integrity(
    signoff: Any,
    artifact: Mapping[str, Any],
) -> str:
    if not isinstance(signoff, Mapping):
        return "unsigned"
    signed = signoff.get("signed_artifacts")
    if not isinstance(signed, list):
        return "unsigned"
    name = str(artifact.get("name") or "")
    signed_item = next(
        (
            item
            for item in signed
            if isinstance(item, Mapping) and str(item.get("name") or "") == name
        ),
        None,
    )
    if signed_item is None:
        return "unsigned"
    if str(signed_item.get("sha256") or "") == str(
        artifact.get("sha256") or ""
    ) and int(signed_item.get("bytes") or -1) == int(artifact.get("bytes") or -2):
        return "verified"
    return "mismatch"


_ROW_LEVEL_KEYS = {
    "tablerows",
    "patientrows",
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


def scan_artifact_payloads(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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


def safe_artifacts_after_privacy_scan(
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
    if not scan_artifact_payloads(safe).get("passed"):
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


def scan_browser_projection(payloads: Mapping[str, Any]) -> Dict[str, Any]:
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


def privacy_blocked_projection(
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


__all__ = [
    "privacy_blocked_projection",
    "project_artifact_governance",
    "safe_artifacts_after_privacy_scan",
    "scan_artifact_payloads",
    "scan_browser_projection",
    "signed_artifact_integrity",
    "signoff_integrity",
]
