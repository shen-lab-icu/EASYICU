"""Atomic host augmentation of declared robustness replay products."""

from __future__ import annotations

from typing import Sequence

from ..schema import AnalysisStep, RobustnessReplayProduct, RobustnessReplaySpec

ROBUSTNESS_ARTICLE_OUTPUT_BINDINGS: tuple[tuple[str, str], ...] = (
    ("statistic:primary_or", "primary_effect"),
    ("statistic:complete_case_n", "complete_case_n"),
    ("table:robustness_summary", "robustness_summary"),
    ("log:missingness_strategy_notes", "missingness_strategy_notes"),
)
_OUTPUT_BY_PRODUCT = dict(ROBUSTNESS_ARTICLE_OUTPUT_BINDINGS)


def with_family_contract_outputs(
    step: AnalysisStep,
    *,
    family: str,
    expected_outputs: Sequence[str],
) -> AnalysisStep:
    """Add host-required outputs without desynchronising a typed owner spec."""

    output_list = list(expected_outputs)
    replay_spec = step.robustness_replay_spec
    if family != "robustness" or replay_spec is None:
        return step.model_copy(update={"expected_outputs": output_list})

    products = list(replay_spec.products)
    declared_product_ids = {item.product_id for item in products}
    for product in output_list:
        replay_output = _OUTPUT_BY_PRODUCT.get(product)
        if replay_output is None:
            continue
        product_id = product.split(":", 1)[1]
        if product_id in declared_product_ids:
            continue
        products.append(
            RobustnessReplayProduct(product_id=product_id, output=replay_output)
        )
        declared_product_ids.add(product_id)

    updated_spec = RobustnessReplaySpec.model_validate(
        {
            **replay_spec.model_dump(mode="python"),
            "products": [item.model_dump(mode="python") for item in products],
        }
    )
    return AnalysisStep.model_validate(
        {
            **step.model_dump(mode="python"),
            "expected_outputs": output_list,
            "robustness_replay_spec": updated_spec.model_dump(mode="python"),
        }
    )


__all__ = ["ROBUSTNESS_ARTICLE_OUTPUT_BINDINGS", "with_family_contract_outputs"]
