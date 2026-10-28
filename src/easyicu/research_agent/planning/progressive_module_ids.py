"""Progressive module-id vocabulary shared by runtime readiness and planning.

This leaf module exists so the execution kernel can read which modules a
selected analysis family can execute without importing the progressive
planning contract (``test_execution_kernel_identity``).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, get_args

from ..contracts.cohort_summary import DESCRIPTIVE_SUMMARY_ACTION as DESCRIPTIVE_SUMMARY_ACTION

ProgressiveModuleId = Literal[
    "cohort_definition",
    "table_one",
    "exposure_outcome_distribution",
    "continuous_outcome_summary",
    "measurement_audit",
    "adjusted_association",
    "absolute_risk_context",
    "robustness_replay",
    "custom_analysis",
    "visualization",
    "report",
]

def descriptive_custom_module_allowed(role: str, action: str | None) -> bool:
    """The registered primary summary or the existing provisional baseline."""

    return (
        role == "primary" and action == DESCRIPTIVE_SUMMARY_ACTION
        or role == "auxiliary" and action is None
    )


def _family_cohort_flow_figure(step: Any) -> bool:
    """Recognize the family-owned count renderer without opening other modules."""
    if (getattr(step, "step_id", None) != "family_cohort_flow_figure"
        or getattr(step, "module_id", None) != "visualization"
        or getattr(step, "planned_analysis_role", None) != "auxiliary"
        or getattr(step, "scientific_action_id", None) is not None
        or len(getattr(step, "depends_on", ())) != 1):
        return False
    if not hasattr(step, "product_inputs"):
        return True  # The outline does not carry products; compilation checks them.
    return (
        not step.raw_inputs
        and [(ref.producer_step_id, ref.product_id) for ref in step.product_inputs]
        == [(step.depends_on[0], "table:cohort_flow")]
        and [(output.product_id, output.semantic_role) for output in step.outputs]
        == [("figure:cohort_flow", "figure")]
    )


def progressive_module_ids_for_analysis_types(
    analysis_types: Sequence[str], *, step: Any = None,
) -> tuple[str, ...]:
    """Return the union of modules the selected analysis families can execute.

    Descriptive epidemiology has no fitted primary effect or uncertainty
    interval. Its outline must therefore not advertise the adjusted-model or
    locked-effect replay owners that require those quantities. Publication
    figures and manuscript text are also host-owned for this family: the host
    binds deterministic renderers to the exact typed result/audit products and
    the evidence-bound Writer runs after execution. The registered measurement
    audit already owns observation-process and timing diagnostics, so an
    arbitrary custom-analysis fallback would duplicate that owner and bypass
    its typed outputs. The custom module is reserved for the registered primary
    descriptive summary and the provisional ungrouped baseline; the outline
    and compiler check those roles and actions. Asking for other parallel steps
    would widen source-lineage obligations without adding a scientific decision.
    """

    normalized = {
        str(value or "").strip().casefold()
        for value in analysis_types
        if str(value or "").strip()
    }
    modules = list(get_args(ProgressiveModuleId))
    if normalized and normalized <= {"descriptive_epidemiology"}:
        modules = [
            module
            for module in modules
            if module
            not in {
                "adjusted_association",
                "absolute_risk_context",
                "robustness_replay",
                "visualization",
                "report",
            }
        ]
        if step is not None and _family_cohort_flow_figure(step):
            modules.append("visualization")
    return tuple(modules)
