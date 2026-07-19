"""Public boundary types for the EasyICU research-agent pipeline.

This module re-exports canonical runtime types so phase modules and future
consumers can import the public API from one place without creating shadow
schemas. Numeric claims come from :mod:`evidence`; evidence artifacts and
validation findings come from :mod:`schema`.

The execution result and phase-result dataclasses are defined here so runtime,
authority, and orchestration modules share one dependency-neutral contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .cohort.schema import CohortDefinition, ConceptPredicate, TimeWindow
from .authority.evidence_store import NumericClaim
from .robustness_panel import RobustnessPanel, RobustnessPanelRow, RobustnessSpec
from .schema import EvidenceRecord as EvidenceArtifact
from .schema import ValidationFinding
from .side_findings import SideFinding

DerivedClaim = NumericClaim


@dataclass
class RunResult:
    """Everything captured from one generated-code execution."""

    step_id: str
    script_path: Path
    cwd: Path
    out_dir: Path
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float
    artefacts: List[Path] = field(default_factory=list)
    timed_out: bool = False
    requested_network_policy: str = "none"
    effective_isolation: str = "unknown"
    isolation_degraded: bool = False
    isolation_degradation_reason: Optional[str] = None
    runtime_provenance: Dict[str, object] = field(default_factory=dict)
    # False means callers must not scan or hash anything under ``out_dir``.
    outputs_safe_to_collect: bool = True
    runner_log_path: Optional[Path] = None

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0 and not self.timed_out and self.outputs_safe_to_collect
        )


if TYPE_CHECKING:
    from .providers.cost import CostMeter
    from .authority.evidence_store import EvidenceStore
    from .literature import LiteratureBundle
    from .replication.envelope import ReproEnvelope
    from .schema import (
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
    "DerivedClaim",
    "RunResult",
    "TimeWindow",
    "ConceptPredicate",
    "CohortDefinition",
    "RobustnessSpec",
    "RobustnessPanelRow",
    "RobustnessPanel",
    "SideFinding",
    "ValidationFinding",
    "_PlanPhaseResult",
    "_ExecutePhaseResult",
    "_WritePhaseResult",
]
