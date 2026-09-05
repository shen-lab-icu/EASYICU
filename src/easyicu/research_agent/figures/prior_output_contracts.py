"""Resolve the reviewed parent-method and fitted-model figure contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ..authority.parent_artifact import _resolve_upstream_manifest_step


def _resolve_upstream_analysis_method(
    run_dir: Path, current_step_id: str
) -> Optional[str]:
    """Return the controlled ``method`` recorded by a figure step's parent."""

    parent = str(current_step_id or "").removesuffix("_figure")
    if not parent or parent == str(current_step_id):
        return None
    summ = Path(run_dir) / "steps" / parent / "outputs" / "step_summary.json"
    try:
        method = json.loads(summ.read_text("utf-8")).get("method")
    except Exception:
        method = None
    if method:
        return str(method).strip().lower()

    # Free-model summaries need not repeat planning metadata.  The partial
    # manifest retains the exact structured AnalysisStep that produced the
    # parent, so use that method as a closed fallback instead of inferring a
    # renderer from the stochastic step id or intent prose.
    request_step = _resolve_upstream_manifest_step(run_dir, current_step_id)
    method = request_step.get("method") if request_step else None
    if method:
        return str(method).strip().lower()
    return None



def _planned_primary_association_contract(
    run_dir: Path,
    figure_step_id: str,
    summary: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Resolve one Planner-required primary model to its validated contract."""

    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return None
    requirements = request_step.get("model_requirements")
    primary_requirements = [
        item
        for item in requirements or []
        if isinstance(item, Mapping)
        and str(item.get("analysis_role") or "").strip().lower() == "primary"
        and item.get("required_for_step_success") is not False
    ]
    if len(primary_requirements) != 1:
        return None
    requirement_id = str(primary_requirements[0].get("requirement_id") or "")
    contracts = summary.get("model_contracts")
    matching_contracts = [
        item
        for item in contracts or []
        if isinstance(item, Mapping)
        and str(item.get("requirement_id") or "") == requirement_id
        and str(item.get("analysis_role") or "").strip().lower() == "primary"
        and str(item.get("fit_status") or "").strip().lower() == "fitted"
    ]
    if len(matching_contracts) != 1:
        return None
    contract = dict(matching_contracts[0])
    if not str(contract.get("model_id") or "").strip():
        return None
    if not str(contract.get("exposure_source") or "").strip():
        return None
    return contract

