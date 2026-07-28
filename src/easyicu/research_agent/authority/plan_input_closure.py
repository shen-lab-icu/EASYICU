"""Host-owned measurement-provenance closure for analysis-plan inputs.

The Planner chooses the scientific variables.  This module owns the narrower
structural operation that adds exact registered ``*_measured``/``*_n``
companions and persists each resulting plan as immutable EvidenceStore
authority.  Execution and resume use the same contract instead of rebuilding
or naming closure evidence independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..plan_utils import _augment_measurement_companion_inputs
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .evidence_store import EvidenceStore, sha256_of_bytes
from .plan_scope import measurement_companion_input_closure_evidence_id
from .runtime_artifacts import verified_run_evidence_path

__all__ = [
    "RegisteredPlanInputClosure",
    "close_measurement_companion_inputs",
    "register_measurement_companion_input_closure",
]


@dataclass(frozen=True)
class RegisteredPlanInputClosure:
    """Mutable convenience path plus its immutable evidence authority."""

    plan_path: Path
    evidence_path: Path
    evidence_id: str


def close_measurement_companion_inputs(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Apply the single canonical structural measurement-input closure."""

    return _augment_measurement_companion_inputs(plan=plan, context=context)


def register_measurement_companion_input_closure(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    plan: AnalysisPlan,
    prompt_pack_version: Optional[str],
) -> RegisteredPlanInputClosure:
    """Persist one live closure plan and its digest-verified authority."""

    payload = plan.model_dump_json(indent=2).encode("utf-8")
    digest = sha256_of_bytes(payload)
    evidence_id = measurement_companion_input_closure_evidence_id(
        revision=int(plan.revision),
        sha256=digest,
    )
    source_path = Path(run_dir) / "analysis_plan_input_closure.json"
    if source_path.is_symlink() or (
        source_path.exists() and not source_path.is_file()
    ):
        raise ValueError("analysis plan input closure path is not a regular file")
    source_path.write_bytes(payload)

    record = evidence.register_file(
        kind="log",
        description=(
            "Analysis plan with structural measurement-provenance input closure."
        ),
        source_path=source_path,
        evidence_id=evidence_id,
        producer="runtime_supervisor",
        generation_mode="system",
        prompt_pack_version=prompt_pack_version,
        metadata={
            "reason": "measurement_companion_input_closure",
            "source_plan_revision": int(plan.revision),
            "closure_sha256": digest,
        },
        publish_aliases=False,
    )
    verified_path = verified_run_evidence_path(run_dir, record)
    if verified_path is None:
        raise ValueError(
            "registered analysis plan input closure failed digest verification"
        )
    return RegisteredPlanInputClosure(
        plan_path=source_path,
        evidence_path=verified_path,
        evidence_id=evidence_id,
    )
