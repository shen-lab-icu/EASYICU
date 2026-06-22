"""Article-package validation for discovery-to-manuscript runs.

The regular research-agent readiness gate answers a narrow question: did one
analysis run execute and bind its claims? A discovery vignette for a manuscript
has a larger contract: the mined idea, handoff, analysis, figures, and writing
must all survive as one auditable package.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .discovery_handoff import DiscoveryHandoffPacket


DISCOVERY_PACKAGE_SCHEMA_VERSION = "easyicu.discovery_manuscript_package/1"

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

    root = Path(run_dir)
    handoff = _load_handoff(root, handoff_path)
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
    inventory = _figure_inventory(root)
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
    missing_story_roles = [role for role in required if role not in observed_story_roles]
    panel_count = sum(item.panel_count for item in inventory)
    figure_story_ready = (
        (len(inventory) >= min_main_figures or panel_count >= min_total_panels)
        and not missing_story_roles
    )
    blocked_steps = _blocked_outcome_steps(root)
    leak_terms = (
        _outcome_leak_terms(manuscript_text) if blocked_steps else []
    )
    checks = {
        "handoff_present": handoff is not None,
        "handoff_agent_selected": bool(
            handoff is not None and handoff.selection_mode == "agent_selected"
        ),
        "run_status_present": bool(run_status_payload),
        "execution_complete": bool(gates.get("execution_complete")),
        "manuscript_ready": bool(gates.get("manuscript_ready")) and manuscript.exists(),
        "publication_ready": bool(gates.get("publication_ready")),
        "figure_story_ready": figure_story_ready,
        "blocked_outcome_not_leaked": not leak_terms,
    }
    blocking: List[str] = []
    if require_handoff and not checks["handoff_present"]:
        blocking.append("missing discovery_handoff.json")
    if handoff is not None and handoff.selection_mode != "agent_selected":
        blocking.append(
            f"handoff selection_mode is {handoff.selection_mode}, not agent_selected"
        )
    if not checks["execution_complete"]:
        blocking.append("research-agent execution is incomplete")
    if not checks["manuscript_ready"]:
        blocking.append("manuscript_ready.md was not emitted by readiness gates")
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
        handoff_path=str(_resolve_handoff_path(root, handoff_path))
        if handoff is not None
        else None,
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
    if handoff_path is not None:
        return Path(handoff_path)
    return root / "discovery_handoff.json"


def _load_handoff(
    root: Path, handoff_path: str | Path | None
) -> Optional[DiscoveryHandoffPacket]:
    path = _resolve_handoff_path(root, handoff_path)
    if not path.exists():
        return None
    return DiscoveryHandoffPacket.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _figure_inventory(root: Path) -> List[ManuscriptFigureInventoryItem]:
    out: List[ManuscriptFigureInventoryItem] = []
    for contract_path in sorted((root / "publication_figures").glob("*.figure_contract.json")):
        payload = _load_json(contract_path)
        if not payload:
            continue
        panels = payload.get("panels") or []
        panel_roles: List[str] = []
        story_roles: set[str] = set()
        text_parts = [
            str(payload.get("figure_id") or ""),
            str(payload.get("core_claim") or ""),
        ]
        for panel in panels:
            if not isinstance(panel, Mapping):
                continue
            role = str(panel.get("role") or "").strip()
            if role:
                panel_roles.append(role)
            text_parts.extend(
                [
                    role,
                    str(panel.get("title") or ""),
                    str(panel.get("claim") or ""),
                ]
            )
            metadata = panel.get("metadata")
            if isinstance(metadata, Mapping):
                text_parts.extend(str(v) for v in metadata.values())
        source_data = [str(x) for x in payload.get("source_data") or []]
        text_parts.extend(source_data)
        for role in _infer_story_roles(" ".join(text_parts)):
            story_roles.add(role)
        out.append(
            ManuscriptFigureInventoryItem(
                stem=contract_path.name.replace(".figure_contract.json", ""),
                contract_path=str(contract_path.relative_to(root)),
                panel_count=len(panels),
                panel_roles=panel_roles,
                story_roles=sorted(story_roles),
                source_data=source_data,
            )
        )
    return out


def _infer_story_roles(text: str) -> List[str]:
    blob = text.lower()
    roles: List[str] = []
    if re.search(r"\b(discovery|idea|literature|source|gap|prior[- ]art|funnel)\b", blob):
        roles.append("discovery_provenance")
    if re.search(
        r"\b(cohort|attrition|evaluable|evaluability|missingness|definition|overlap|discordance)\b",
        blob,
    ):
        roles.append("cohort_evaluability")
    if re.search(
        r"\b(primary|effect|outcome|association|mortality|death|robustness|result)\b",
        blob,
    ):
        roles.append("primary_result")
    if re.search(r"\b(audit|claim|evidence|reproducib|validation|gate)\b", blob):
        roles.append("audit_reproducibility")
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
