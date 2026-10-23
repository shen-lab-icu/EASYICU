"""Regression for the typed ungrouped-baseline compiler boundary."""

import pytest

from easyicu.research_agent.planning.progressive_compiler import (
    _canonical_outputs,
    _is_ungrouped_baseline_summary,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveSkeletonStep,
)


def _custom_baseline(product_id: str) -> ProgressiveSkeletonStep:
    return ProgressiveSkeletonStep(
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
                "product_id": product_id,
                "semantic_role": "custom",
            }
        ],
        custom_method="descriptive_baseline_context",
    )


def test_baseline_product_identity_does_not_depend_on_free_text_method_alias() -> None:
    step = _custom_baseline("artifact:baseline_context")

    assert _is_ungrouped_baseline_summary(step)
    assert _canonical_outputs(step) == [("table:cohort_summary", "custom")]


@pytest.mark.parametrize("product_id", ["table:table_one", "table:cohort_summary"])
def test_a_custom_step_naming_table_one_is_the_ungrouped_summary(product_id) -> None:
    """A real plan stopped three times on this spelling: the outline fixed the
    product, and step repairs could not change it.  A custom step owns no
    Table 1 spec, so the name can only mean the ungrouped summary."""

    assert _canonical_outputs(_custom_baseline(product_id)) == [
        ("table:cohort_summary", "custom")
    ]
