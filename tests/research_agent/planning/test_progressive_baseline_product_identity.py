"""Regression for the typed ungrouped-baseline compiler boundary."""

from easyicu.research_agent.planning.progressive_compiler import (
    _canonical_outputs,
    _is_ungrouped_baseline_summary,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveSkeletonStep,
)


def test_baseline_product_identity_does_not_depend_on_free_text_method_alias() -> None:
    step = ProgressiveSkeletonStep(
        step_id="baseline_context",
        planned_analysis_role="auxiliary",
        module_id="custom_analysis",
        objective="Provide an ungrouped descriptive baseline context for review.",
        depends_on=["cohort_accounting"],
        raw_inputs=["age", "exposure", "outcome"],
        product_inputs=[
            {
                "producer_step_id": "cohort_accounting",
                "product_id": "artifact:analysis_cohort",
            }
        ],
        outputs=[
            {
                "product_id": "artifact:baseline_context",
                "semantic_role": "custom",
            }
        ],
        custom_method="descriptive_baseline_context",
    )

    assert _is_ungrouped_baseline_summary(step)
    assert _canonical_outputs(step) == [("table:cohort_summary", "custom")]
