"""Article-package validation for discovery-to-manuscript runs.

The regular research-agent readiness gate answers a narrow question: did one
analysis run execute and bind its claims? A discovery vignette for a manuscript
has a larger contract: the mined idea, handoff, analysis, figures, and writing
must all survive as one auditable package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .discovery_handoff import (
    ANALYSIS_READY_DECISIONS,
    DiscoveryHandoffPacket,
)
from ..authority.evidence_snapshot import load_current_evidence_snapshot

DISCOVERY_PACKAGE_SCHEMA_VERSION = "easyicu.discovery_manuscript_package/1"

_STORY_ROLES = frozenset(
    {
        "discovery_provenance",
        "cohort_evaluability",
        "primary_result",
        "audit_reproducibility",
    }
)
_DIRECT_PANEL_ROLE_TO_STORY_ROLE = {
    "cohort_accounting": "cohort_evaluability",
    "data_quality": "cohort_evaluability",
    "descriptive_result": "primary_result",
    "primary_estimand": "primary_result",
    "model_performance": "primary_result",
    "temporal_absolute_risk": "primary_result",
    "survival_effect": "primary_result",
    "causal_contrast": "primary_result",
    "audit": "audit_reproducibility",
    "supplementary_provenance": "audit_reproducibility",
}
_FIGURE_SUFFIXES = frozenset({".svg", ".pdf", ".png", ".tif", ".tiff"})
_MAX_SVG_BYTES = 16 * 1024 * 1024
_MAX_SVG_PROLOG_BYTES = 64 * 1024
_SAFE_SVG_DOCTYPE = re.compile(
    rb"<!DOCTYPE\s+svg\s+PUBLIC\s+"
    rb"['\"]-//W3C//DTD\s+SVG\s+1\.1//EN['\"]\s+"
    rb"['\"]https?://www\.w3\.org/Graphics/SVG/1\.1/DTD/svg11\.dtd['\"]\s*>",
    flags=re.IGNORECASE,
)

PackageStatus = Literal[
    "article_ready",
    "manuscript_only",
    "analysis_only",
    "diagnostic_only",
]


class ManuscriptFigureInventoryItem(BaseModel):
    """One manuscript-facing figure stem and its contract metadata."""

    model_config = ConfigDict(extra="forbid")

    stem: str
    contract_path: str
    panel_count: int = 0
    panel_roles: List[str] = Field(default_factory=list)
    story_roles: List[str] = Field(default_factory=list)
    source_data: List[str] = Field(default_factory=list)
    figure_paths: List[str] = Field(default_factory=list)
    invalid_figure_paths: List[str] = Field(default_factory=list)
    unregistered_figure_paths: List[str] = Field(default_factory=list)
    figure_evidence_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    unresolved_source_data: List[str] = Field(default_factory=list)
    unresolved_evidence_ids: List[str] = Field(default_factory=list)
    contract_evidence_id: Optional[str] = None
    contract_registered: bool = False
    required_formats_present: bool = False
    vector_figure_present: bool = False
    primary_result_data_bound: bool = True
    provenance_valid: bool = False


@dataclass(frozen=True)
class _EvidenceRegistry:
    bindings: Dict[str, Dict[str, Any]]
    by_sha256: Dict[str, List[Dict[str, Any]]]


class DiscoveryManuscriptPackageAssessment(BaseModel):
    """Machine-readable article-package gate result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = DISCOVERY_PACKAGE_SCHEMA_VERSION
    status: PackageStatus
    package_ready: bool
    run_dir: str
    checks: Dict[str, bool]
    blocking_reasons: List[str] = Field(default_factory=list)
    handoff_path: Optional[str] = None
    handoff_selection_mode: Optional[str] = None
    handoff_candidate_topic: Optional[str] = None
    manuscript_path: Optional[str] = None
    run_status: Optional[str] = None
    figure_inventory: List[ManuscriptFigureInventoryItem] = Field(default_factory=list)
    figure_stem_count: int = 0
    figure_panel_count: int = 0
    required_story_roles: List[str] = Field(default_factory=list)
    observed_story_roles: List[str] = Field(default_factory=list)
    missing_story_roles: List[str] = Field(default_factory=list)
    blocked_outcome_steps: List[str] = Field(default_factory=list)
    manuscript_outcome_leak_terms: List[str] = Field(default_factory=list)


def validate_discovery_manuscript_package(
    *,
    run_dir: str | Path,
    handoff_path: str | Path | None = None,
    require_handoff: bool = True,
    min_main_figures: int = 3,
    min_total_panels: int = 4,
    required_story_roles: Optional[Sequence[str]] = None,
) -> DiscoveryManuscriptPackageAssessment:
    """Validate a run as a complete discovery-to-article package."""

    root = Path(run_dir).resolve()
    evidence_registry = _evidence_registry(root)
    resolved_handoff_path = _safe_run_path(
        root,
        _resolve_handoff_path(root, handoff_path),
    )
    handoff = _load_handoff(root, handoff_path)
    handoff_binding = _resolve_provenance_token(
        token="discovery_handoff",
        registry=evidence_registry,
    )
    handoff_evidence_registered = handoff_binding is not None
    handoff_evidence_hash_match = bool(
        handoff is not None
        and resolved_handoff_path is not None
        and resolved_handoff_path.is_file()
        and handoff_binding is not None
        and _sha256(resolved_handoff_path) == handoff_binding.get("sha256")
    )
    run_status_payload = _load_json(root / "run_status.json")
    gates = (
        run_status_payload.get("gates", {})
        if isinstance(run_status_payload, Mapping)
        else {}
    )
    manuscript = root / "manuscript_ready.md"
    manuscript_text = (
        manuscript.read_text(encoding="utf-8", errors="replace")
        if manuscript.exists()
        else ""
    )
    inventory = _figure_inventory(root, registry=evidence_registry)
    observed_story_roles = sorted(
        {role for item in inventory for role in item.story_roles}
    )
    required = list(
        required_story_roles
        or (
            handoff.required_manuscript_figure_roles
            if handoff is not None
            else [
                "discovery_provenance",
                "cohort_evaluability",
                "primary_result",
                "audit_reproducibility",
            ]
        )
    )
    missing_story_roles = [
        role for role in required if role not in observed_story_roles
    ]
    panel_count = sum(item.panel_count for item in inventory)
    figure_artifacts_present = bool(inventory) and all(
        item.figure_paths
        and item.required_formats_present
        and item.vector_figure_present
        and not item.invalid_figure_paths
        and not item.unregistered_figure_paths
        and item.contract_registered
        for item in inventory
    )
    figure_source_data_bound = bool(inventory) and all(
        not item.unresolved_source_data and item.source_data for item in inventory
    )
    figure_evidence_bound = bool(inventory) and all(
        not item.unresolved_evidence_ids and item.evidence_ids for item in inventory
    )
    figure_provenance_valid = bool(inventory) and all(
        item.provenance_valid for item in inventory
    )
    figure_story_ready = (
        (len(inventory) >= min_main_figures or panel_count >= min_total_panels)
        and not missing_story_roles
        and figure_provenance_valid
    )
    blocked_steps = _blocked_outcome_steps(root)
    leak_terms = _outcome_leak_terms(manuscript_text) if blocked_steps else []
    source_error = None
    if handoff is not None:
        try:
            handoff.verify_source()
        except ValueError as exc:
            source_error = str(exc)
    checks = {
        "handoff_present": handoff is not None,
        "handoff_source_evidence_intact": handoff is not None and source_error is None,
        "handoff_evidence_registered": handoff_evidence_registered,
        "handoff_evidence_hash_match": handoff_evidence_hash_match,
        "handoff_agent_selected": bool(
            handoff is not None and handoff.selection_mode == "agent_selected"
        ),
        "handoff_recommended": bool(
            handoff is not None
            and str(handoff.go_no_go).strip().lower() in ANALYSIS_READY_DECISIONS
        ),
        "handoff_human_confirmed": bool(
            handoff is not None and handoff.human_confirmed
        ),
        "handoff_endpoint_consistent": bool(
            handoff is not None
            and (
                not handoff.resolved_outcome_concept
                or _normalise_endpoint(handoff.resolved_outcome_concept)
                == _normalise_endpoint(handoff.target_outcome)
            )
        ),
        "run_status_present": bool(run_status_payload),
        "execution_complete": bool(gates.get("execution_complete")),
        "manuscript_ready": bool(gates.get("manuscript_ready")) and manuscript.exists(),
        "publication_ready": bool(gates.get("publication_ready")),
        "figure_artifacts_present": figure_artifacts_present,
        "figure_source_data_bound": figure_source_data_bound,
        "figure_evidence_bound": figure_evidence_bound,
        "figure_provenance_valid": figure_provenance_valid,
        "figure_story_ready": figure_story_ready,
        "blocked_outcome_not_leaked": not leak_terms,
    }
    blocking: List[str] = []
    if require_handoff and not checks["handoff_present"]:
        blocking.append("missing or invalid discovery_handoff.json")
    if source_error is not None:
        blocking.append(f"discovery source evidence failed integrity validation: {source_error}")
    if (require_handoff or handoff is not None) and not handoff_evidence_registered:
        blocking.append(
            "discovery_handoff.json is not explicitly registered in EvidenceStore"
        )
    if (require_handoff or handoff is not None) and not handoff_evidence_hash_match:
        blocking.append(
            "discovery_handoff.json does not exactly match its hashed EvidenceStore record"
        )
    if handoff is not None and handoff.selection_mode != "agent_selected":
        blocking.append(
            f"handoff selection_mode is {handoff.selection_mode}, not agent_selected"
        )
    if handoff is not None and not checks["handoff_recommended"]:
        blocking.append(f"handoff go_no_go is {handoff.go_no_go}, not go/recommend")
    if handoff is not None and not checks["handoff_human_confirmed"]:
        blocking.append("handoff lacks explicit human confirmation")
    if handoff is not None and not checks["handoff_endpoint_consistent"]:
        blocking.append("handoff endpoint is inconsistent or not licensed for analysis")
    if not checks["execution_complete"]:
        blocking.append("research-agent execution is incomplete")
    if not checks["manuscript_ready"]:
        blocking.append("manuscript_ready.md was not emitted by readiness gates")
    if not checks["publication_ready"]:
        blocking.append("research-agent publication_ready gate did not pass")
    if not figure_artifacts_present:
        blocking.append(
            "figure contracts lack non-empty rendered artefacts including a "
            "vector export"
        )
    if not figure_source_data_bound:
        blocking.append("figure source_data entries do not resolve to real provenance")
    if not figure_evidence_bound:
        blocking.append("figure panel evidence ids do not resolve to real provenance")
    if not figure_provenance_valid:
        invalid = [item.stem for item in inventory if not item.provenance_valid]
        blocking.append(
            "figure provenance validation failed for: " + ", ".join(invalid or ["none"])
        )
    if not figure_story_ready:
        blocking.append(
            "manuscript figure set is incomplete: "
            f"{len(inventory)} figure stem(s), {panel_count} panel(s), "
            f"missing story roles {missing_story_roles}"
        )
    if leak_terms:
        blocking.append(
            "blocked outcome analysis appears in manuscript language: "
            + ", ".join(leak_terms)
        )

    package_ready = not blocking
    if package_ready:
        status: PackageStatus = "article_ready"
    elif checks["manuscript_ready"]:
        status = "manuscript_only"
    elif checks["execution_complete"]:
        status = "analysis_only"
    else:
        status = "diagnostic_only"

    return DiscoveryManuscriptPackageAssessment(
        status=status,
        package_ready=package_ready,
        run_dir=str(root),
        checks=checks,
        blocking_reasons=blocking,
        handoff_path=(str(resolved_handoff_path) if handoff is not None else None),
        handoff_selection_mode=handoff.selection_mode if handoff else None,
        handoff_candidate_topic=handoff.candidate_topic if handoff else None,
        manuscript_path=str(manuscript) if manuscript.exists() else None,
        run_status=(
            str(run_status_payload.get("status"))
            if isinstance(run_status_payload, Mapping)
            else None
        ),
        figure_inventory=inventory,
        figure_stem_count=len(inventory),
        figure_panel_count=panel_count,
        required_story_roles=required,
        observed_story_roles=observed_story_roles,
        missing_story_roles=missing_story_roles,
        blocked_outcome_steps=blocked_steps,
        manuscript_outcome_leak_terms=leak_terms,
    )


def write_discovery_package_assessment(
    assessment: DiscoveryManuscriptPackageAssessment,
    path: str | Path,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(assessment.model_dump_json(indent=2), encoding="utf-8")
    return out


def _resolve_handoff_path(root: Path, handoff_path: str | Path | None) -> Path:
    if handoff_path is None:
        return root / "discovery_handoff.json"
    path = Path(handoff_path)
    return path if path.is_absolute() else root / path


def _normalise_endpoint(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _load_handoff(
    root: Path, handoff_path: str | Path | None
) -> Optional[DiscoveryHandoffPacket]:
    path = _safe_run_path(root, _resolve_handoff_path(root, handoff_path))
    if path is None or not path.is_file():
        return None
    try:
        return DiscoveryHandoffPacket.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _figure_inventory(
    root: Path,
    *,
    registry: Optional[_EvidenceRegistry] = None,
) -> List[ManuscriptFigureInventoryItem]:
    root = root.resolve()
    out: List[ManuscriptFigureInventoryItem] = []
    registry = registry or _evidence_registry(root)
    for contract_path in sorted(
        (root / "publication_figures").glob("*.figure_contract.json")
    ):
        stem = contract_path.name.replace(".figure_contract.json", "")
        safe_contract_path = _safe_run_path(root, contract_path)
        if safe_contract_path is None or not safe_contract_path.is_file():
            out.append(
                ManuscriptFigureInventoryItem(
                    stem=stem,
                    contract_path=str(contract_path.relative_to(root)),
                )
            )
            continue
        payload = _load_json(safe_contract_path)
        if not payload:
            out.append(
                ManuscriptFigureInventoryItem(
                    stem=stem,
                    contract_path=str(contract_path.relative_to(root)),
                )
            )
            continue
        panels = [
            panel
            for panel in (payload.get("panels") or [])
            if isinstance(panel, Mapping)
        ]
        panel_roles: List[str] = []
        story_roles: set[str] = set()
        evidence_ids: List[str] = []
        primary_panel_evidence_ids: List[List[str]] = []
        for panel in panels:
            role = str(panel.get("role") or "").strip()
            if role:
                panel_roles.append(role)
            panel_story_roles = _structured_story_roles(panel)
            story_roles.update(panel_story_roles)
            panel_evidence = _string_list(panel.get("evidence_ids"))
            if not panel_evidence:
                panel_evidence = [
                    f"<panel:{panel.get('panel_id') or '?'}:missing-evidence>"
                ]
            evidence_ids.extend(panel_evidence)
            if "primary_result" in panel_story_roles:
                primary_panel_evidence_ids.append(panel_evidence)
        source_data = _string_list(payload.get("source_data"))
        resolved_sources = {
            token: _resolve_provenance_token(
                token=token,
                registry=registry,
            )
            for token in source_data
        }
        resolved_evidence = {
            token: _resolve_provenance_token(
                token=token,
                registry=registry,
            )
            for token in evidence_ids
        }
        unresolved_source_data = [
            token for token, binding in resolved_sources.items() if binding is None
        ]
        unresolved_evidence_ids = [
            token for token, binding in resolved_evidence.items() if binding is None
        ]
        canonical_source_ids = {
            str(binding["evidence_id"])
            for binding in resolved_sources.values()
            if binding is not None
        }
        canonical_panel_ids = {
            str(binding["evidence_id"])
            for binding in resolved_evidence.values()
            if binding is not None
        }
        required_source_ids = canonical_source_ids | canonical_panel_ids
        contract_records = _matching_contract_records(
            contract_path=safe_contract_path,
            figure_id=str(payload.get("figure_id") or stem),
            required_source_ids=required_source_ids,
            registry=registry,
        )
        contract_record = contract_records[0] if contract_records else None
        contract_evidence_id = (
            str(contract_record["evidence_id"]) if contract_record is not None else None
        )
        candidate_figure_paths = [
            path
            for path in sorted(contract_path.parent.glob(f"{stem}.*"))
            if path.suffix.lower() in _FIGURE_SUFFIXES
        ]
        artifact_validity = {
            path: _valid_figure_artifact(path, root=root)
            for path in candidate_figure_paths
        }
        figure_paths = [
            path for path in candidate_figure_paths if artifact_validity[path]
        ]
        invalid_figure_paths = [
            path for path in candidate_figure_paths if not artifact_validity[path]
        ]
        figure_records: List[Dict[str, Any]] = []
        unregistered_figure_paths: List[Path] = []
        if contract_evidence_id is not None:
            for path in figure_paths:
                matches = _matching_figure_records(
                    figure_path=path,
                    figure_id=str(payload.get("figure_id") or stem),
                    contract_evidence_id=contract_evidence_id,
                    required_source_ids=required_source_ids,
                    primary_source_ids={
                        str(resolved_evidence[token]["evidence_id"])
                        for panel_ids in primary_panel_evidence_ids
                        for token in panel_ids
                        if resolved_evidence.get(token) is not None
                    },
                    registry=registry,
                )
                if matches:
                    figure_records.append(matches[0])
                else:
                    unregistered_figure_paths.append(path)
        else:
            unregistered_figure_paths.extend(figure_paths)
        valid_suffixes = {path.suffix.lower() for path in figure_paths}
        required_formats_present = {".svg", ".png"} <= valid_suffixes
        vector_present = any(
            path.suffix.lower() in {".svg", ".pdf"} for path in figure_paths
        )
        primary_result_data_bound = True
        if "primary_result" in story_roles:
            primary_result_data_bound = bool(primary_panel_evidence_ids) and all(
                panel_ids
                and all(resolved_evidence.get(token) is not None for token in panel_ids)
                and any(
                    resolved_evidence[token].get("kind") in {"table", "statistic"}
                    for token in panel_ids
                    if resolved_evidence.get(token) is not None
                )
                for panel_ids in primary_panel_evidence_ids
            )
        provenance_valid = bool(
            figure_paths
            and not invalid_figure_paths
            and not unregistered_figure_paths
            and len(figure_records) == len(figure_paths)
            and required_formats_present
            and vector_present
            and source_data
            and evidence_ids
            and not unresolved_source_data
            and not unresolved_evidence_ids
            and contract_record is not None
            and primary_result_data_bound
        )
        out.append(
            ManuscriptFigureInventoryItem(
                stem=stem,
                contract_path=str(contract_path.relative_to(root)),
                panel_count=len(panels),
                panel_roles=panel_roles,
                story_roles=sorted(story_roles),
                source_data=source_data,
                figure_paths=[str(path.relative_to(root)) for path in figure_paths],
                invalid_figure_paths=[
                    str(path.relative_to(root)) for path in invalid_figure_paths
                ],
                unregistered_figure_paths=[
                    str(path.relative_to(root)) for path in unregistered_figure_paths
                ],
                figure_evidence_ids=[
                    str(record["evidence_id"]) for record in figure_records
                ],
                evidence_ids=evidence_ids,
                unresolved_source_data=unresolved_source_data,
                unresolved_evidence_ids=unresolved_evidence_ids,
                contract_evidence_id=contract_evidence_id,
                contract_registered=contract_record is not None,
                required_formats_present=required_formats_present,
                vector_figure_present=vector_present,
                primary_result_data_bound=primary_result_data_bound,
                provenance_valid=provenance_valid,
            )
        )
    return out


def _evidence_registry(root: Path) -> _EvidenceRegistry:
    """Load a fail-closed, hash-verified view of EvidenceStore."""

    root = root.resolve()
    snapshot = load_current_evidence_snapshot(root)
    records_payload = list(snapshot.records)
    aliases = dict(snapshot.aliases)
    raw_bindings: List[Dict[str, Any]] = []
    id_counts: Dict[str, int] = {}
    for record in records_payload:
        evidence_id = str(record.get("evidence_id") or "").strip()
        relative_path = str(record.get("relative_path") or "").strip()
        expected_sha = str(record.get("sha256") or "").strip().lower()
        if (
            not evidence_id
            or not relative_path
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            continue
        path = _safe_run_path(root, root / relative_path)
        if path is None:
            continue
        valid = bool(
            path.is_file() and path.stat().st_size > 0 and _sha256(path) == expected_sha
        )
        binding = dict(record)
        binding.update(
            {
                "path": path,
                "kind": str(record.get("kind") or "log"),
                "valid": valid,
                "evidence_id": evidence_id,
                "sha256": expected_sha,
            }
        )
        raw_bindings.append(binding)
        id_counts[evidence_id] = id_counts.get(evidence_id, 0) + 1

    bindings: Dict[str, Dict[str, Any]] = {}
    by_sha256: Dict[str, List[Dict[str, Any]]] = {}
    for binding in raw_bindings:
        evidence_id = str(binding["evidence_id"])
        if id_counts[evidence_id] != 1:
            binding["valid"] = False
            continue
        bindings[evidence_id] = binding
        if binding.get("valid"):
            by_sha256.setdefault(str(binding["sha256"]), []).append(binding)
    for alias, evidence_id in aliases.items():
        token = str(alias or "").strip()
        binding = bindings.get(str(evidence_id))
        if token and binding is not None and binding.get("valid"):
            bindings[token] = binding
    return _EvidenceRegistry(bindings=bindings, by_sha256=by_sha256)


def _safe_run_path(root: Path, path: Path) -> Optional[Path]:
    """Resolve ``path`` and reject symlink/path traversal outside ``root``."""

    root = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _canonical_evidence_ids(
    values: Any,
    *,
    registry: _EvidenceRegistry,
) -> Optional[set[str]]:
    tokens = _string_list(values)
    if not tokens:
        return set()
    resolved = [
        _resolve_provenance_token(token=token, registry=registry) for token in tokens
    ]
    if any(binding is None for binding in resolved):
        return None
    return {str(binding["evidence_id"]) for binding in resolved if binding is not None}


def _matching_contract_records(
    *,
    contract_path: Path,
    figure_id: str,
    required_source_ids: set[str],
    registry: _EvidenceRegistry,
) -> List[Dict[str, Any]]:
    digest = _sha256(contract_path)
    matches: List[Dict[str, Any]] = []
    for record in registry.by_sha256.get(digest, []):
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if record.get("kind") != "log":
            continue
        if metadata.get("artifact_role") != "figure_contract":
            continue
        if str(metadata.get("figure_id") or "") != figure_id:
            continue
        metadata_sources = _canonical_evidence_ids(
            metadata.get("source_evidence_ids"),
            registry=registry,
        )
        inputs = _canonical_evidence_ids(record.get("inputs"), registry=registry)
        if metadata_sources is None or inputs is None:
            continue
        if not required_source_ids <= metadata_sources:
            continue
        if not required_source_ids <= inputs:
            continue
        matches.append(record)
    return sorted(matches, key=lambda item: str(item.get("evidence_id") or ""))


def _matching_figure_records(
    *,
    figure_path: Path,
    figure_id: str,
    contract_evidence_id: str,
    required_source_ids: set[str],
    primary_source_ids: set[str],
    registry: _EvidenceRegistry,
) -> List[Dict[str, Any]]:
    digest = _sha256(figure_path)
    matches: List[Dict[str, Any]] = []
    for record in registry.by_sha256.get(digest, []):
        metadata = record.get("metadata")
        if record.get("kind") != "figure" or not isinstance(metadata, Mapping):
            continue
        if metadata.get("artifact_role") != "manuscript_figure":
            continue
        if str(metadata.get("figure_id") or "") != figure_id:
            continue
        linked_contract = _resolve_provenance_token(
            token=str(metadata.get("contract_evidence_id") or ""),
            registry=registry,
        )
        if (
            linked_contract is None
            or linked_contract.get("evidence_id") != contract_evidence_id
        ):
            continue
        script = _resolve_provenance_token(
            token=str(record.get("script_evidence_id") or ""),
            registry=registry,
        )
        if script is None or script.get("kind") != "code":
            continue
        if not str(record.get("producer") or "").strip():
            continue
        if not str(record.get("generation_mode") or "").strip():
            continue
        metadata_sources = _canonical_evidence_ids(
            metadata.get("source_evidence_ids"),
            registry=registry,
        )
        inputs = _canonical_evidence_ids(record.get("inputs"), registry=registry)
        if metadata_sources is None or inputs is None:
            continue
        if not required_source_ids <= metadata_sources:
            continue
        if not required_source_ids | {contract_evidence_id} <= inputs:
            continue
        if (
            not primary_source_ids <= metadata_sources
            or not primary_source_ids <= inputs
        ):
            continue
        matches.append(record)
    return sorted(matches, key=lambda item: str(item.get("evidence_id") or ""))


def _structured_story_roles(panel: Mapping[str, Any]) -> set[str]:
    """Use structured panel semantics; prose can only refine a declared role."""

    roles: set[str] = set()
    metadata = panel.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    for value in (
        panel.get("story_role"),
        panel.get("story_roles"),
        metadata.get("story_role"),
        metadata.get("story_roles"),
    ):
        roles.update(role for role in _string_list(value) if role in _STORY_ROLES)

    panel_role = str(panel.get("role") or "").strip().lower()
    if panel_role in _STORY_ROLES:
        roles.add(panel_role)
    mapped = _DIRECT_PANEL_ROLE_TO_STORY_ROLE.get(panel_role)
    if mapped is not None:
        roles.add(mapped)

    # Generic contract roles are only useful with a compatible structured role;
    # keywords alone never grant all four manuscript-story roles.
    auxiliary_allowed = {
        "overview": {"discovery_provenance"},
        "workflow": {"discovery_provenance"},
        "baseline_context": {"cohort_evaluability"},
        "relationship": {"primary_result"},
        "validation": {"primary_result", "audit_reproducibility"},
        "robustness": {"primary_result"},
        "heterogeneity": {"primary_result"},
        "calibration": {"primary_result"},
        "clinical_utility": {"primary_result"},
        "diagnostics": {"audit_reproducibility"},
        "stability": {"audit_reproducibility"},
    }.get(panel_role, set())
    if auxiliary_allowed:
        text = " ".join(
            [
                str(panel.get("title") or ""),
                str(panel.get("claim") or ""),
                str(panel.get("review_risk") or ""),
            ]
        )
        roles.update(_keyword_story_roles(text) & auxiliary_allowed)
    return roles


def _safe_svg_xml(path: Path) -> bool:
    prolog = bytearray()
    found_root = False
    carry = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            probe = (carry + chunk).upper()
            if b"<!ENTITY" in probe:
                return False
            carry = probe[-16:]
            if found_root:
                continue
            prolog.extend(chunk)
            root_match = re.search(rb"<(?:[A-Za-z_][\w.-]*:)?svg\b", prolog, re.I)
            if root_match is not None:
                del prolog[root_match.start() :]
                found_root = True
            elif len(prolog) > _MAX_SVG_PROLOG_BYTES:
                return False

    upper_prolog = bytes(prolog).upper()
    doctype_count = upper_prolog.count(b"<!DOCTYPE")
    if doctype_count == 0:
        return True
    if doctype_count != 1:
        return False
    start = upper_prolog.index(b"<!DOCTYPE")
    end = upper_prolog.find(b">", start)
    if end < 0:
        return False
    declaration = bytes(prolog[start : end + 1]).strip()
    if b"[" in declaration or b"]" in declaration:
        return False
    return _SAFE_SVG_DOCTYPE.fullmatch(declaration) is not None


def _parse_static_svg(path: Path) -> ET.Element:
    """Parse bounded SVG bytes after removing the one allow-listed external DTD."""

    payload = bytearray()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            payload.extend(chunk)
    safe_payload = _SAFE_SVG_DOCTYPE.sub(b"", bytes(payload), count=1)
    return ET.fromstring(safe_payload)


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_provenance_token(
    *,
    token: str,
    registry: _EvidenceRegistry,
) -> Optional[Dict[str, Any]]:
    text = str(token or "").strip()
    if not text:
        return None
    bound = registry.bindings.get(text)
    return bound if bound is not None and bound.get("valid") else None


def _valid_figure_artifact(
    path: Path,
    *,
    root: Optional[Path] = None,
) -> bool:
    """Verify an in-run figure without loading a whole binary into memory."""

    resolved = (
        _safe_run_path(root.resolve(), path)
        if root is not None
        else path.resolve(strict=False)
    )
    if resolved is None or not resolved.is_file():
        return False
    size = resolved.stat().st_size
    if size <= 0:
        return False
    suffix = resolved.suffix.lower()
    try:
        with resolved.open("rb") as handle:
            header = handle.read(8)
        if suffix == ".svg":
            if size > _MAX_SVG_BYTES or not _safe_svg_xml(resolved):
                return False
            document = _parse_static_svg(resolved)
            if document.tag.rsplit("}", 1)[-1].lower() != "svg":
                return False
            visual_tags = {
                "circle",
                "ellipse",
                "image",
                "line",
                "path",
                "polygon",
                "polyline",
                "rect",
                "text",
                "use",
            }
            return any(
                node.tag.rsplit("}", 1)[-1].lower() in visual_tags
                for node in document.iter()
                if isinstance(node.tag, str)
            )
        if suffix in {".png", ".tif", ".tiff"}:
            from PIL import Image

            expected_format = "PNG" if suffix == ".png" else "TIFF"
            if suffix == ".png" and header != b"\x89PNG\r\n\x1a\n":
                return False
            if suffix in {".tif", ".tiff"} and not header.startswith(
                (b"II*\x00", b"MM\x00*")
            ):
                return False
            with Image.open(resolved) as image:
                width, height = image.size
                image_format = image.format
                image.verify()
            return image_format == expected_format and width > 0 and height > 0
        if suffix == ".pdf":
            if not header.startswith(b"%PDF-") or size < 64:
                return False
            with resolved.open("rb") as handle:
                handle.seek(max(0, size - 1024))
                return b"%%EOF" in handle.read(1024)
    except (ImportError, OSError, ET.ParseError, ValueError):
        return False
    return False


def _keyword_story_roles(text: str) -> set[str]:
    blob = text.lower()
    roles: set[str] = set()
    if re.search(
        r"\b(discovery|idea|literature|source|gap|prior[- ]art|funnel)\b", blob
    ):
        roles.add("discovery_provenance")
    if re.search(
        r"\b(cohort|attrition|evaluable|evaluability|missingness|definition|overlap|discordance)\b",
        blob,
    ):
        roles.add("cohort_evaluability")
    if re.search(
        r"\b(primary|effect|outcome|association|mortality|death|robustness|result)\b",
        blob,
    ):
        roles.add("primary_result")
    if re.search(r"\b(audit|claim|evidence|reproducib|validation|gate)\b", blob):
        roles.add("audit_reproducibility")
    return roles


def _blocked_outcome_steps(root: Path) -> List[str]:
    blocked: List[str] = []
    for path in sorted(root.glob("steps/*/outputs/step_summary.json")):
        payload = _load_json(path)
        if not payload:
            continue
        if (
            payload.get("primary_analysis_authorized") is False
            or payload.get("grouped_death_analysis_executed") is False
            or (
                payload.get("analysis_executed") is False
                and "blocked" in json.dumps(payload, ensure_ascii=False).lower()
            )
        ):
            blocked.append(path.parts[-3])
    for path in sorted(root.glob("steps/*/outputs/*feasibility_gate.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            rows = []
        if any(str(row.get("status") or "").lower() == "blocked" for row in rows):
            step_id = path.parts[-3]
            if step_id not in blocked:
                blocked.append(step_id)
    return blocked


_OUTCOME_LEAK_PATTERNS = [
    r"\bnear[- ]null\b",
    r"\bmortality contrast\b",
    r"\bdeath association\b",
    r"\bdeath-related inference",
    r"\bdeath-related inferences",
    r"\bexploratory association with death\b",
    r"\bpoint estimates? ranging\b",
    r"\brobustness range\b",
]


def _outcome_leak_terms(text: str) -> List[str]:
    found: List[str] = []
    for pattern in _OUTCOME_LEAK_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(pattern.strip(r"\b"))
    return found


__all__ = [
    "DISCOVERY_PACKAGE_SCHEMA_VERSION",
    "DiscoveryManuscriptPackageAssessment",
    "ManuscriptFigureInventoryItem",
    "validate_discovery_manuscript_package",
    "write_discovery_package_assessment",
]
