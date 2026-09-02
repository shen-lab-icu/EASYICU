"""Current manuscript gate state and stale-finding supersession policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..authority.manuscript_claim_policy import missing_scientific_claims_in_results


GATE_STATE_SUPERSESSION_PATTERNS = (
    ("manuscript_gate", "execution gate did not pass", "execution_complete"),
    ("manuscript_gate", "manuscript generation skipped", "execution_complete"),
    (
        "robustness_panel",
        "locked robustness specifications that no step estimated",
        "robustness_panel_complete",
    ),
    (
        "evidence_bound_writer",
        "strict evidence enforcement blocked manuscript generation",
        "manuscript_bound_clean",
    ),
    (
        "evidence_bound_writer",
        "bound manuscript is empty or non-substantive",
        "manuscript_bound_clean",
    ),
    (
        "writer_agent",
        "failed before producing a manuscript scaffold",
        "manuscript_bound_clean",
    ),
    (
        "manuscript_literature",
        "manuscript literature authority is incomplete",
        "manuscript_literature_complete",
    ),
    (
        "manuscript_numeric_auditor",
        "strict evidence enforcement blocked manuscript generation",
        "manuscript_numeric_bound_clean",
    ),
    ("critic_agent", "criticagent marked manuscript", "manuscript_critique_passed"),
    (
        "manuscript_quality",
        "deterministic manuscript quality audit requires changes",
        "manuscript_quality_complete",
    ),
    (
        "manuscript_result_sufficiency",
        "manuscript has no results section",
        "manuscript_result_claims_complete",
    ),
    (
        "manuscript_result_sufficiency",
        "final evidence/numeric filtering removed or failed to bind",
        "manuscript_result_claims_complete",
    ),
    (
        "evidence_bound_writer",
        "unresolved manifest caveats",
        "manuscript_manifest_caveats_clean",
    ),
)


def current_manuscript_completion_state(
    *,
    run_dir: Path,
    manuscript_text: str,
    evidence: Any,
    per_step_records: Sequence[Mapping[str, Any]],
    stop_after_analysis: bool,
    writer_probe_mode: bool,
) -> dict[str, bool]:
    """Project quality and scientific-claim completion from current artifacts."""

    quality_complete = False
    quality_audit_path = run_dir / "manuscript_quality_audit.json"
    if quality_audit_path.exists():
        try:
            quality_audit_payload = json.loads(
                quality_audit_path.read_text(encoding="utf-8")
            )
        except Exception:
            quality_audit_payload = {}
        quality_complete = bool(
            isinstance(quality_audit_payload, dict)
            and quality_audit_payload.get("status") == "pass"
        )

    authoritative_claims = evidence.authoritative_scientific_claims(per_step_records)
    claims_complete = bool(
        manuscript_text
        and authoritative_claims
        and not missing_scientific_claims_in_results(
            manuscript_text,
            claims=authoritative_claims,
        )
        and not stop_after_analysis
        and not writer_probe_mode
    )
    return {
        "manuscript_quality_complete": quality_complete,
        "manuscript_result_claims_complete": claims_complete,
    }


__all__ = [
    "GATE_STATE_SUPERSESSION_PATTERNS",
    "current_manuscript_completion_state",
]
