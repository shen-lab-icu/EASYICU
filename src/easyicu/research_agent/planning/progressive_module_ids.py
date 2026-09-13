"""Progressive module-id vocabulary shared by runtime readiness and planning.

This leaf module exists so the execution kernel can read which modules a
selected analysis family can execute without importing the progressive
planning contract (``test_execution_kernel_identity``).
"""

from __future__ import annotations

from typing import Literal, Sequence, get_args

ProgressiveModuleId = Literal[
    "cohort_definition",
    "table_one",
    "exposure_outcome_distribution",
    "measurement_audit",
    "adjusted_association",
    "absolute_risk_context",
    "robustness_replay",
    "custom_analysis",
    "visualization",
    "report",
]


def progressive_module_ids_for_analysis_types(
    analysis_types: Sequence[str],
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
    its typed outputs. Asking the Planner to add any of those parallel steps
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
                "custom_analysis",
                "visualization",
                "report",
            }
        ]
    return tuple(modules)
