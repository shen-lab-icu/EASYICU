"""Runtime boundary types for the EasyICU research-agent pipeline.

This module re-exports canonical runtime types so phase modules and future
consumers can import the public API from one place without creating shadow
schemas. Numeric claims come from :mod:`evidence`; evidence artifacts and
validation findings come from :mod:`schema`.

The execution result and phase-result dataclasses are defined here so runtime,
authority, and orchestration modules share one dependency-neutral contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..cohort.schema import CohortDefinition, ConceptPredicate, TimeWindow
from ..authority.evidence_store import NumericClaim
from ..authority.scientific_claims import ScientificClaim
from ..robustness.panel import RobustnessPanel, RobustnessPanelRow, RobustnessSpec
from ..schema import EvidenceRecord as EvidenceArtifact
from ..schema import ValidationFinding
from .execution_result import RunnerFailureCode, RunResult

DerivedClaim = NumericClaim


if TYPE_CHECKING:
    from ..providers.cost import CostMeter
    from ..authority.evidence_store import EvidenceStore
    from ..literature import LiteratureBundle
    from ..replication.envelope import ReproEnvelope
    from ..schema import (
        AgentRuntimeState,
        AnalysisPlan,
        CritiqueReport,
        ManuscriptDraftPacket,
        PipelineResult,
        ResearchContext,
    )


@dataclass
class _PlanPhaseResult:
    """Runtime handoff from plan phase to execute/write/package phases."""

    context: ResearchContext
    agent_context: ResearchContext
    context_path: Path
    evidence: EvidenceStore
    findings: List[ValidationFinding]
    plan: AnalysisPlan
    plan_path: Path
    llm_signature: str
    used_mock_llm: bool
    prompt_version: str
    prompt_files: Dict[str, str]
    role_resolver: Callable[[str], Any]
    cost_meter: Optional[CostMeter]
    repro_envelope: Optional[ReproEnvelope]
    started_at: datetime
    resume_state: Optional[Dict[str, Any]]
    aborted_result: Optional[PipelineResult] = None


@dataclass
class _ExecutePhaseResult:
    """Runtime handoff from execute phase to write/package phases."""

    plan: AnalysisPlan
    per_step_records: List[Dict[str, Any]]
    step_attempt_history: List[Dict[str, Any]]
    probe_summary: Dict[str, Any]
    runtime_state: AgentRuntimeState
    flush_partial_manifest: Callable[[Optional[Dict[str, Any]]], None]


@dataclass
class _WritePhaseResult:
    """Runtime handoff from write phase to package phase."""

    literature: Optional[LiteratureBundle]
    bound_path: Path
    manuscript_packet: Optional[ManuscriptDraftPacket] = None
    manuscript_critique: Optional[CritiqueReport] = None
    writer_probe_mode: bool = False
    writer_probe_failed_steps: Tuple[str, ...] = ()


__all__ = [
    "EvidenceArtifact",
    "NumericClaim",
    "ScientificClaim",
    "DerivedClaim",
    "RunResult",
    "RunnerFailureCode",
    "TimeWindow",
    "ConceptPredicate",
    "CohortDefinition",
    "RobustnessSpec",
    "RobustnessPanelRow",
    "RobustnessPanel",
    "ValidationFinding",
    "_PlanPhaseResult",
    "_ExecutePhaseResult",
    "_WritePhaseResult",
]
