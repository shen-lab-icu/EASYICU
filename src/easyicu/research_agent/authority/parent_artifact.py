"""Host-owned authority for one figure step's direct-parent artifacts.

These helpers resolve only the checkpoint-selected direct parent and verify its
registered evidence bytes.  They do not select a renderer, infer a scientific
role, scan sibling outputs, or create a second current-evidence authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..authority.evidence_store import sha256_of_file
from ..authority.runtime_artifacts import (
    current_step_records,
    load_run_artifact_authority,
    verified_run_evidence_path,
)

__all__ = [
    "_resolve_upstream_manifest_analysis_request",
    "_resolve_upstream_manifest_step",
    "_verified_direct_parent_artifact_digests",
    "_verified_direct_parent_table_names",
]


def _resolve_upstream_manifest_analysis_request(
    run_dir: Path, current_step_id: str
) -> Optional[Dict[str, Any]]:
    """Return the host-recorded direct-parent planning request.

    This checkpoint object is written by the supervisor before the parent code
    executes.  It is therefore the authority for sealed-renderer method and
    family selection; a coder-authored ``step_summary.json`` may only confirm,
    never replace, this request.
    """

    parent = str(current_step_id or "").removesuffix("_figure")
    if not parent or parent == str(current_step_id):
        return None
    manifest = load_run_artifact_authority(run_dir)
    if not isinstance(manifest, Mapping):
        return None
    records = manifest.get("per_step_records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        return None
    current = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(records)
        if isinstance(record, Mapping)
    }
    record = current.get(parent)
    if not isinstance(record, Mapping):
        return None
    request = record.get("analysis_request")
    return dict(request) if isinstance(request, Mapping) else None


def _resolve_upstream_manifest_step(
    run_dir: Path, current_step_id: str
) -> Optional[Dict[str, Any]]:
    """Return the direct parent's structured AnalysisStep from the checkpoint."""

    request = _resolve_upstream_manifest_analysis_request(run_dir, current_step_id)
    request_step = request.get("step") if isinstance(request, Mapping) else None
    return dict(request_step) if isinstance(request_step, Mapping) else None


def _verified_direct_parent_artifact_digests(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[dict[str, str]]:
    """Return digest-bound direct-parent tables and summary from the checkpoint.

    ``None`` means the figure does not have one exact ``*_figure`` parent or
    the modern outer ledger cannot prove that parent is currently successful.
    Both the immutable evidence copy and the mutable step-output copy must
    match the registered digest before an automatic renderer may read it.
    """

    parent_step_id = str(figure_step_id or "").removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(figure_step_id or ""):
        return None
    authority = load_run_artifact_authority(run_dir)
    if authority is None:
        return None
    raw_records = authority.get("per_step_records")
    if not isinstance(raw_records, list):
        return None
    current = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(raw_records)
    }
    parent_record = current.get(parent_step_id)
    if (
        not isinstance(parent_record, Mapping)
        or str(parent_record.get("status") or "").strip().lower() != "ok"
    ):
        return None
    active_ids = {
        str(evidence_id)
        for evidence_id in (parent_record.get("evidence_ids") or [])
        if str(evidence_id).strip()
    }
    raw_evidence = authority.get("evidence")
    evidence_by_id = {
        str(record.get("evidence_id") or ""): record
        for record in (raw_evidence if isinstance(raw_evidence, list) else [])
        if isinstance(record, Mapping) and str(record.get("evidence_id") or "")
    }
    verified_digests: dict[str, str] = {}
    verified_summary = False
    for evidence_id in active_ids:
        record = evidence_by_id.get(evidence_id)
        if (
            not isinstance(record, Mapping)
            or str(record.get("produced_by_step") or "") != parent_step_id
        ):
            continue
        evidence_path = verified_run_evidence_path(run_dir, record)
        if evidence_path is None:
            continue
        evidence_name = evidence_path.name
        logical_name = (
            evidence_name.split("__", 1)[1] if "__" in evidence_name else evidence_name
        )
        output_path = (
            Path(run_dir) / "steps" / parent_step_id / "outputs" / logical_name
        )
        try:
            if output_path.is_symlink() or not output_path.is_file():
                continue
            output_path.resolve(strict=True).relative_to(Path(run_dir).resolve())
        except (OSError, ValueError):
            continue
        if sha256_of_file(output_path) != str(record.get("sha256") or ""):
            continue
        kind = str(record.get("kind") or "").strip().lower()
        if kind == "table":
            verified_digests[logical_name] = str(record.get("sha256") or "")
        elif (
            kind == "statistic"
            and logical_name == "step_summary.json"
            and evidence_id == str(parent_record.get("step_summary_evidence_id") or "")
        ):
            verified_summary = True
            verified_digests[logical_name] = str(record.get("sha256") or "")
    return verified_digests if verified_summary else None


def _verified_direct_parent_table_names(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[set[str]]:
    """Return the verified table-name projection of the parent digest seal."""

    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    if digests is None:
        return None
    return {name for name in digests if name != "step_summary.json"}
