"""Browser projection of registered extraction metadata, without live lookups."""

from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.authority.provenance import RunDataProvenance
from .research_evidence_preview import (
    EvidencePreviewError, build_evidence_preview, project_review_evidence_refs,
)
from . import run_artifact_disclosure


def load_data_provenance_projection(run_dir: Path | None) -> dict:
    if run_dir is not None:
        refs = project_review_evidence_refs(run_dir, ["data_provenance.json"])
        if len(refs) == 1:
            try:
                preview = build_evidence_preview(
                    run_dir, evidence_id=refs[0]["evidence_id"],
                    expected_sha256=refs[0]["sha256"],
                )
                payload = {**preview["value"], "evidence_ref": {
                    "evidence_id": refs[0]["evidence_id"], "sha256": refs[0]["sha256"],
                }}
                scanned = scan_data_provenance_artifact(payload)
                if scanned["ok"]:
                    return scanned["payload"]
            except (ValueError, KeyError, EvidencePreviewError):
                pass
    return RunDataProvenance(
        run_id=run_dir.name if run_dir is not None else "", status="not_recorded",
    ).model_dump(mode="json")


def public_data_provenance(payload: dict) -> dict:
    """An artifact reader accepts only the closed metadata schema."""
    scanned = scan_data_provenance_artifact(payload)
    return scanned["payload"] if scanned["ok"] else RunDataProvenance(
        run_id="", status="not_recorded",
    ).model_dump(mode="json")


def scan_data_provenance_artifact(payload: dict) -> dict:
    """Apply existing disclosure owners to previews and unchanged downloads."""
    artifacts = {"data_provenance.json": payload}
    for scan in (run_artifact_disclosure.scan_artifact_payloads(artifacts),
                 run_artifact_disclosure.scan_browser_projection(artifacts)):
        if not scan["passed"]:
            return {"ok": False, "error": "artifact_privacy_scan_failed",
                    "artifact": "data_provenance.json", "privacy_scan": scan}
    try:
        projected = RunDataProvenance.model_validate(payload).model_dump(mode="json")
    except ValueError:
        return {"ok": False, "error": "artifact_schema_invalid", "artifact": "data_provenance.json"}
    return {"ok": True, "payload": projected, "privacy_scan": scan}
