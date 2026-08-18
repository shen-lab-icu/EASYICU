"""Structural figure-step identity and replan preservation authority."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from ..contracts.declared_product import typed_product
from ..schema import AnalysisPlan, AnalysisStep, ValidationFinding
from .robustness_plan_mutation import with_family_contract_outputs

_FIGURE_OUTPUT_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_FIGURE_FILE_SUFFIXES = (".png", ".svg", ".pdf", ".tif", ".tiff")


def _output_declares_figure(output: str) -> bool:
    token = str(output or "").strip().lower()
    if not token:
        return False
    kind, separator, name = token.partition(":")
    if separator:
        return kind.strip() in _FIGURE_OUTPUT_KINDS and bool(name.strip())
    if token.endswith(_FIGURE_FILE_SUFFIXES):
        return True
    words = set(filter(None, re.split(r"[^a-z0-9]+", token)))
    return bool(words & {"figure", "plot", "chart", "heatmap"})


def _parent_step_id_for_figure_step(step: AnalysisStep) -> Optional[str]:
    step_id = str(step.step_id or "")
    if step_id.endswith("_figure") and len(step_id) > len("_figure"):
        return step_id[: -len("_figure")]
    match = re.search(
        r"declared by step ['`]([^'`]+)['`]",
        str(step.intent or ""),
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _step_produces_figure(step: AnalysisStep) -> bool:
    """Return whether expected outputs declare a figure-like artifact."""

    return any(
        _output_declares_figure(output) for output in step.expected_outputs or []
    )


def _preserve_figure_steps_after_replan(
    *,
    current: AnalysisPlan,
    revised: AnalysisPlan,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Restore dropped render steps and only their exact prior parent edges."""

    revised_ids = {step.step_id for step in revised.steps}
    dropped_figure_steps = [
        step
        for step in current.steps
        if step.step_id not in revised_ids and _step_produces_figure(step)
    ]
    new_steps = list(revised.steps) + list(dropped_figure_steps)

    current_output_owners: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for step in current.steps:
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is not None:
                current_output_owners.setdefault(product, []).append(
                    (str(step.step_id), str(raw_output))
                )

    resulting_producers: Dict[Tuple[str, str], Set[str]] = {}
    for step in new_steps:
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is not None:
                resulting_producers.setdefault(product, set()).add(str(step.step_id))

    result_ids = {str(step.step_id) for step in new_steps}
    current_figure_ids = {
        str(step.step_id) for step in current.steps if _step_produces_figure(step)
    }
    restored_by_parent: Dict[str, List[str]] = {}
    for figure_step in new_steps:
        if str(figure_step.step_id) not in current_figure_ids:
            continue
        parent_id = _parent_step_id_for_figure_step(figure_step)
        if not parent_id or parent_id not in result_ids:
            continue
        for raw_input in figure_step.inputs or []:
            product = typed_product(raw_input)
            if product is None or resulting_producers.get(product):
                continue
            prior_owners = current_output_owners.get(product, [])
            if len(prior_owners) != 1 or prior_owners[0][0] != parent_id:
                continue
            restored_output = prior_owners[0][1]
            restored_by_parent.setdefault(parent_id, []).append(restored_output)
            resulting_producers.setdefault(product, set()).add(parent_id)

    if restored_by_parent:
        repaired_steps: List[AnalysisStep] = []
        for step in new_steps:
            additions = restored_by_parent.get(str(step.step_id), [])
            if not additions:
                repaired_steps.append(step)
                continue
            repaired_steps.append(
                with_family_contract_outputs(
                    step,
                    family=(
                        "robustness"
                        if step.robustness_replay_spec is not None
                        else ""
                    ),
                    expected_outputs=list(
                        dict.fromkeys([*(step.expected_outputs or []), *additions])
                    ),
                )
            )
        new_steps = repaired_steps

    if not dropped_figure_steps and not restored_by_parent:
        return revised, []

    preserved = revised.model_copy(update={"steps": new_steps})
    findings: List[ValidationFinding] = []
    if dropped_figure_steps:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="warning",
                message=(
                    "Replanner attempted to drop "
                    f"{len(dropped_figure_steps)} figure-producing step(s); "
                    "they were re-attached to preserve task contract."
                ),
                detail={
                    "preserved_step_ids": [s.step_id for s in dropped_figure_steps],
                },
            )
        )
    if restored_by_parent:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="warning",
                message=(
                    "Restored exact typed outputs on existing direct parent "
                    "steps so preserved figure children retain a valid product DAG."
                ),
                detail={
                    "reason": "preserved_figure_parent_output_contract",
                    "restored_outputs_by_parent": restored_by_parent,
                },
            )
        )
    return preserved, findings


__all__ = [
    "_output_declares_figure",
    "_parent_step_id_for_figure_step",
    "_preserve_figure_steps_after_replan",
    "_step_produces_figure",
]
