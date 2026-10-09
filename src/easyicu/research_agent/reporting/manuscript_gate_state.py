"""Current manuscript gate state and stale-finding supersession policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..authority.manuscript_claim_policy import missing_scientific_claims_in_results
from ..authority.runtime_artifacts import current_evidence_records
from .manuscript_figures import manuscript_figure_receipt_is_current
from .manuscript_result_facts import recorded_result_fact_carriage


GATE_STATE_SUPERSESSION_PATTERNS = (
    ("manuscript_figure_projection", "a reader figure has no source-bound",
     "manuscript_figures_complete"),
    ("manuscript_figure_projection", "source-bound figure projection failed",
     "manuscript_figures_complete"),
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
    reader_labels: Mapping[str, str] | None,
) -> dict[str, bool]:
    """Project quality and scientific-claim completion from current artifacts.

    ``reader_labels`` are the claim labels the manuscript was bound with.  A
    claim that a host fact recorded for these exact bytes carries is reported
    by that fact (:func:`.manuscript_result_facts.recorded_result_fact_carriage`).
    """

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
    carried = recorded_result_fact_carriage(
        run_dir=run_dir,
        manuscript_text=manuscript_text,
        evidence=evidence,
        claims=authoritative_claims,
    ).carried
    claims_complete = bool(
        manuscript_text
        and authoritative_claims
        and not missing_scientific_claims_in_results(
            manuscript_text,
            claims=[
                claim
                for claim in authoritative_claims
                if claim.claim_ref not in carried
            ],
            reader_labels=reader_labels,
        )
        and not stop_after_analysis
        and not writer_probe_mode
    )
    return {
        "manuscript_figures_complete": (
            manuscript_figure_receipt_is_current(
                run_dir=run_dir, evidence_records=current_evidence_records(evidence.records(), per_step_records),
            ) if callable(getattr(evidence, "records", None)) else False
        ),
        "manuscript_quality_complete": quality_complete,
        "manuscript_result_claims_complete": claims_complete,
    }


def manuscript_result_fact_trace(
    *,
    run_dir: Path,
    manuscript_text: str,
    evidence: Any,
    per_step_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Which claims the completion state let recorded facts carry, and why
    every other fact naming a claim carried none."""

    carriage = recorded_result_fact_carriage(
        run_dir=run_dir,
        manuscript_text=manuscript_text,
        evidence=evidence,
        claims=evidence.authoritative_scientific_claims(per_step_records),
    )
    return {"carried": dict(carriage.carried), **carriage.trace}


__all__ = [
    "GATE_STATE_SUPERSESSION_PATTERNS",
    "current_manuscript_completion_state",
    "manuscript_result_fact_trace",
]
