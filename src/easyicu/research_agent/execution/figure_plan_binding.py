"""Bind Planner-owned figure panels to the contracts produced at runtime.

``AnalysisStep.figure_panels`` is scientific plan authority: it names the
reader-facing role, exact chart grammar, and exact typed source products for
each planned panel.  A valid JSON figure contract is not enough on its own;
without this join a renderer can write a different chart while retaining the
same output filename.  This module performs that read-only, end-of-execute
join and emits fail-closed findings without changing either artifact.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..authority.runtime_artifacts import current_successful_step_records
from ..schema import AnalysisPlan, AnalysisStep, ValidationFinding

PLANNED_FIGURE_CONTRACT_BINDING_VALIDATOR = "planned_figure_contract_binding"

_TYPED_PRODUCT = re.compile(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*")


def _finding(
    *,
    step_id: str,
    figure_output: str,
    reason: str,
    message: str,
    detail: Mapping[str, Any] | None = None,
) -> ValidationFinding:
    payload = {
        "reason": reason,
        "step_id": step_id,
        "figure_output": figure_output,
    }
    if detail:
        payload.update(detail)
    return ValidationFinding(
        validator=PLANNED_FIGURE_CONTRACT_BINDING_VALIDATOR,
        severity="error",
        message=message,
        detail=payload,
    )


def _flat_output_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if path.is_absolute() or path.name != value.strip() or path.name in {".", ".."}:
        return None
    return path.name


def _runtime_panel_signature(
    panel: Mapping[str, Any],
) -> tuple[str, str, str, tuple[str, ...]] | None:
    """Return only explicit scientific coordinates from one runtime panel.

    Chart inference and prose search are intentionally forbidden here.  They
    are useful for display summaries, but cannot prove that a planned
    ``coverage_heatmap`` was not implemented as a horizontal bar chart.
    """

    metadata = panel.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    panel_id = str(panel.get("panel_id") or "").strip()
    role = str(
        panel.get("article_role")
        or metadata.get("article_role")
        or panel.get("role")
        or ""
    ).strip()
    chart_type = str(
        panel.get("chart_type")
        or metadata.get("chart_type")
        or panel.get("visual_form")
        or metadata.get("visual_form")
        or ""
    ).strip()
    raw_sources = panel.get("source_products")
    if raw_sources is None:
        raw_sources = metadata.get("source_products")
    if not panel_id or not role or not chart_type or not isinstance(raw_sources, list):
        return None
    sources = [str(value or "").strip() for value in raw_sources]
    if (
        not sources
        or len(sources) != len(set(sources))
        or any(_TYPED_PRODUCT.fullmatch(value) is None for value in sources)
    ):
        return None
    return panel_id, role, chart_type, tuple(sorted(sources))


def _planned_panel_signature(panel: Any) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        str(panel.panel_id),
        str(panel.article_role),
        str(panel.chart_type),
        tuple(sorted(str(value) for value in panel.source_products)),
    )


def _contract_panels_for_output(
    *,
    contract: Mapping[str, Any],
    step_summary: Mapping[str, Any],
    figure_output: str,
    declared_output_count: int,
) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    raw_panels = contract.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        return None, "runtime_contract_has_no_panels"
    panels = [panel for panel in raw_panels if isinstance(panel, Mapping)]
    if len(panels) != len(raw_panels):
        return None, "runtime_contract_has_malformed_panels"

    bindings = step_summary.get("planner_product_slot_bindings")
    binding = bindings.get(figure_output) if isinstance(bindings, Mapping) else None
    panel_ids = binding.get("panel_ids") if isinstance(binding, Mapping) else None
    if panel_ids is not None:
        if (
            not isinstance(panel_ids, list)
            or not panel_ids
            or len(panel_ids) != len(set(panel_ids))
            or any(
                not isinstance(value, str) or not value.strip() for value in panel_ids
            )
        ):
            return None, "runtime_panel_slot_binding_is_malformed"
        selected_ids = set(panel_ids)
        selected = [
            panel
            for panel in panels
            if str(panel.get("panel_id") or "").strip() in selected_ids
        ]
        if len(selected) != len(selected_ids):
            return None, "runtime_panel_slot_binding_is_incomplete"
        return selected, None

    if declared_output_count != 1:
        return None, "runtime_panel_output_binding_is_ambiguous"
    return panels, None


def _contract_for_output(
    *,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    figure_output: str,
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    output_files = step_summary.get("output_files")
    figure_name = (
        _flat_output_name(output_files.get(figure_output))
        if isinstance(output_files, Mapping)
        else None
    )
    if figure_name is None:
        return None, None, "runtime_figure_output_is_unbound"
    contract_name = f"{Path(figure_name).stem}.figure_contract.json"
    declared_contracts = step_summary.get("contract_files")
    if declared_contracts is not None and (
        not isinstance(declared_contracts, list)
        or contract_name
        not in {
            name
            for value in declared_contracts
            if (name := _flat_output_name(value)) is not None
        }
    ):
        return None, contract_name, "runtime_contract_is_not_currently_declared"

    out_dir = Path(out_dir).resolve()
    contract_path = out_dir / contract_name
    try:
        if (
            contract_path.is_symlink()
            or not contract_path.is_file()
            or contract_path.resolve(strict=True).parent != out_dir
        ):
            return None, contract_name, "runtime_contract_is_missing_or_unsafe"
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, contract_name, "runtime_contract_is_unreadable"
    if not isinstance(raw, Mapping):
        return None, contract_name, "runtime_contract_is_not_an_object"
    return raw, contract_name, None


def validate_step_planned_figure_contract_binding(
    *,
    step: AnalysisStep,
    out_dir: Path,
    step_summary: Mapping[str, Any],
) -> list[ValidationFinding]:
    """Validate one completed figure step inside the candidate repair loop."""

    if not step.figure_panels:
        return []
    panels_by_output: dict[str, list[Any]] = {}
    for panel in step.figure_panels:
        # This validator binds the current main-figure artifact.  A panel that
        # the final article strategy moved to the supplement is satisfied by
        # the dedicated supporting display step and must not be required to
        # remain embedded in the main composite.
        if str(getattr(panel, "placement", "main")) == "supplementary":
            continue
        panels_by_output.setdefault(str(panel.figure_output), []).append(panel)

    findings: list[ValidationFinding] = []
    for figure_output, planned_panels in panels_by_output.items():
        contract, contract_name, contract_error = _contract_for_output(
            out_dir=out_dir,
            step_summary=step_summary,
            figure_output=figure_output,
        )
        if contract_error is not None or contract is None:
            findings.append(
                _finding(
                    step_id=str(step.step_id),
                    figure_output=figure_output,
                    reason=contract_error or "runtime_contract_unavailable",
                    message=(
                        f"Runtime figure contract for {step.step_id!r} does not bind "
                        f"the planned output {figure_output!r}."
                    ),
                    detail={"contract_file": contract_name},
                )
            )
            continue
        runtime_panels, panel_error = _contract_panels_for_output(
            contract=contract,
            step_summary=step_summary,
            figure_output=figure_output,
            declared_output_count=len(panels_by_output),
        )
        if panel_error is not None or runtime_panels is None:
            findings.append(
                _finding(
                    step_id=str(step.step_id),
                    figure_output=figure_output,
                    reason=panel_error or "runtime_panels_unavailable",
                    message=(
                        f"Runtime panels for {step.step_id!r} cannot be bound to "
                        f"the planned output {figure_output!r}."
                    ),
                    detail={"contract_file": contract_name},
                )
            )
            continue

        signatures = [_runtime_panel_signature(panel) for panel in runtime_panels]
        if any(signature is None for signature in signatures):
            findings.append(
                _finding(
                    step_id=str(step.step_id),
                    figure_output=figure_output,
                    reason="runtime_panel_scientific_coordinates_missing",
                    message=(
                        f"Runtime contract {contract_name!r} omits an explicit "
                        "panel id, article role, chart type, or typed "
                        "source-products list."
                    ),
                    detail={"contract_file": contract_name},
                )
            )
            continue

        planned = Counter(_planned_panel_signature(panel) for panel in planned_panels)
        observed = Counter(signature for signature in signatures if signature)
        if observed != planned:
            findings.append(
                _finding(
                    step_id=str(step.step_id),
                    figure_output=figure_output,
                    reason="runtime_panel_contract_mismatch",
                    message=(
                        f"Runtime contract {contract_name!r} does not implement "
                        "the exact planned panel id, article role, chart type, "
                        "and source products."
                    ),
                    detail={
                        "contract_file": contract_name,
                        "planned_panel_signatures": [
                            {
                                "panel_id": panel_id,
                                "article_role": role,
                                "chart_type": chart,
                                "source_products": list(sources),
                                "count": count,
                            }
                            for (panel_id, role, chart, sources), count in sorted(
                                planned.items()
                            )
                        ],
                        "runtime_panel_signatures": [
                            {
                                "panel_id": panel_id,
                                "article_role": role,
                                "chart_type": chart,
                                "source_products": list(sources),
                                "count": count,
                            }
                            for (panel_id, role, chart, sources), count in sorted(
                                observed.items()
                            )
                        ],
                    },
                )
            )
    return findings


def validate_planned_figure_contract_bindings(
    *,
    plan: AnalysisPlan | None,
    run_dir: Path,
    per_step_records: Sequence[Mapping[str, Any]],
) -> list[ValidationFinding]:
    """Require every typed planned panel to match the current runtime contract."""

    if plan is None or getattr(plan, "steps", None) is None:
        return []
    current = {
        str(record.get("step_id") or "").strip(): record
        for record in current_successful_step_records(per_step_records)
    }
    findings: list[ValidationFinding] = []
    for step in plan.steps:
        if not step.figure_panels:
            continue
        step_id = str(step.step_id)
        record = current.get(step_id)
        summary = record.get("step_summary") if isinstance(record, Mapping) else None
        if not isinstance(summary, Mapping):
            for figure_output in {
                str(panel.figure_output) for panel in step.figure_panels
            }:
                findings.append(
                    _finding(
                        step_id=step_id,
                        figure_output=figure_output,
                        reason="figure_step_has_no_current_successful_summary",
                        message=(
                            f"Planned figure panels for {step_id!r} have no current "
                            "successful runtime summary to bind."
                        ),
                    )
                )
            continue
        findings.extend(
            validate_step_planned_figure_contract_binding(
                step=step,
                out_dir=Path(run_dir) / "steps" / step_id / "outputs",
                step_summary=summary,
            )
        )
    return findings


__all__ = [
    "PLANNED_FIGURE_CONTRACT_BINDING_VALIDATOR",
    "validate_step_planned_figure_contract_binding",
    "validate_planned_figure_contract_bindings",
]
