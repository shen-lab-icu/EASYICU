"""The exact-digest plan a development run reuses instead of a Planner call.

Owner
-----
A development run may name a locked AnalysisPlan file and its SHA-256.  The
host reads that plan only from a regular file with exactly that digest,
parses it with the run's sealed cohort roster, and requires it to answer the
run's research question.  :mod:`easyicu.research_agent.pipeline` then shapes
and validates it like any other plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from ..authority.evidence_store import sha256_of_file
from ..planning.cohort_contract import (
    cohort_concept_id_scope,
    sealed_cohort_concept_ids,
)
from ..schema import AnalysisPlan, ResearchContext
from ..contracts.runtime import ValidationFinding


def _read_locked_plan(path: Path, context: ResearchContext) -> AnalysisPlan:
    """Parse a development locked plan with its run's sealed cohort roster.

    Its cohort may filter on a column the run materialized, which validation
    knows only with that roster.
    """

    text = path.read_text(encoding="utf-8")
    with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
        return AnalysisPlan.model_validate_json(text)


def read_development_locked_plan(
    path: Union[str, Path],
    expected_sha256: Optional[str],
    context: ResearchContext,
) -> Tuple[AnalysisPlan, str]:
    """The locked plan at ``path`` and its verified SHA-256.

    Raises ``ValueError`` when the file is not a regular file, its digest is
    not ``expected_sha256``, it is not a valid plan, or it answers another
    research question than ``context``.
    """

    locked_plan_path = Path(path).expanduser()
    expected_digest = str(expected_sha256 or "")
    if not locked_plan_path.is_file():
        raise ValueError(
            "development locked analysis plan is not a regular file: "
            f"{locked_plan_path}"
        )
    observed_digest = sha256_of_file(locked_plan_path)
    if observed_digest != expected_digest:
        raise ValueError(
            "development locked analysis plan SHA-256 mismatch: "
            f"expected={expected_digest} observed={observed_digest}"
        )
    try:
        plan = _read_locked_plan(locked_plan_path, context)
    except Exception as exc:
        raise ValueError("development locked analysis plan is invalid") from exc
    if plan.research_question != context.research_question:
        raise ValueError("development locked analysis plan research question mismatch")
    return plan, observed_digest


def bind_development_plan_capability(plan, findings):
    """Bind a locked plan to its already-declared single primary capability."""
    primary_steps = [
        step for step in plan.steps if step.planned_analysis_role == "primary"
    ]
    if len(primary_steps) == 1 and primary_steps[0].scientific_capability:
        from ..planning.analysis_types import (
            get_analysis_type,
            optional_analysis_type_for_capability,
        )
        from ..planning.capability_registry import get_capability_by_id

        current_type = get_analysis_type(plan.analysis_type)
        declared_capability = get_capability_by_id(
            primary_steps[0].scientific_capability
        )
        expected_capability = get_capability_by_id(current_type.capability_id)
        rebound_type = None
        if (
            declared_capability is not None
            and expected_capability is not None
            and declared_capability.family == expected_capability.family
            and declared_capability.capability_id
            != expected_capability.capability_id
        ):
            rebound_type = optional_analysis_type_for_capability(
                declared_capability.capability_id
            )
        if rebound_type is not None:
            findings.append(
                ValidationFinding(
                    validator="scientific_capability",
                    severity="warning",
                    message=(
                        "Rebound a locked development plan's analysis "
                        "subtype to the capability declared by its sole "
                        "primary owner; no estimator or result changed."
                    ),
                    detail={
                        "reason_code": (
                            "development_locked_plan_primary_capability_rebound"
                        ),
                        "source_analysis_type": plan.analysis_type,
                        "rebound_analysis_type": rebound_type.key,
                        "primary_step_id": primary_steps[0].step_id,
                        "primary_capability_id": (
                            declared_capability.capability_id
                        ),
                    },
                )
            )
            plan = plan.model_copy(update={"analysis_type": rebound_type.key})
    return plan


__all__ = ["read_development_locked_plan"]
