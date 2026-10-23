"""Step-local product identity for the existing closed auxiliary summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cohort_summary import is_descriptive_cohort_summary_step
from .product_identity import is_canonical_typed_product_token

if TYPE_CHECKING:
    from ..schema import AnalysisStep


def auxiliary_summary_product(step_id: str) -> str:
    """Give an ungrouped baseline alias a default owned by its own step."""

    return f"table:cohort_summary_{step_id}"


def is_descriptive_summary_product_step(step: AnalysisStep) -> bool:
    """Reuse the closed summary contract with its exact step-local product."""

    product = auxiliary_summary_product(step.step_id)
    if is_canonical_typed_product_token(product) and step.expected_outputs == [product]:
        step = step.model_copy(update={"expected_outputs": ["table:cohort_summary"]})
    return is_descriptive_cohort_summary_step(step)
