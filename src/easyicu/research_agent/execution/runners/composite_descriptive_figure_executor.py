"""Deterministic renderer for a closed four-table descriptive figure.

The owner is deliberately narrow: it consumes the canonical cohort-flow,
exposure/outcome-distribution, measurement-missingness, and
measurement-process tables under all-row contracts.  It may derive display
coordinates from values already present in those tables, but it never scans a
run, filters rows, changes a denominator, or fits a model.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...contracts.figure_plan import COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS
from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .figure_input_capability import TypedInputCapability
from .typed_input_binding import BoundTypedInput, load_typed_input, sha256_file

COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS = (
    "table:cohort_flow",
    "table:exposure_outcome_distribution",
    "table:missingness_measurement_audit",
    "table:measurement_process_audit",
)
COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS = (
    "table:cohort_flow",
    "table:exposure_outcome_distribution",
    "table:missingness_measurement_audit",
    "table:robustness_summary",
)
COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS = (
    "table:exposure_outcome_distribution",
    "table:adjusted_association_estimates",
    "table:robustness_matrix",
    "table:measurement_missingness",
)
COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS = (
    "table:exposure_outcome_distribution",
    "table:adjusted_association_estimates",
    "table:robustness_summary",
    "table:measurement_missingness",
)
COMPOSITE_ASSOCIATION_MEASUREMENT_PUBLICATION_FIGURE_INPUTS = (
    "table:exposure_outcome_distribution",
    "table:adjusted_association_estimates",
    "table:missingness_measurement_audit",
    "table:exposure_component_completeness_audit",
)
COMPOSITE_ASSOCIATION_ROBUSTNESS_PUBLICATION_FIGURE_INPUTS = (
    "table:exposure_outcome_distribution",
    "table:adjusted_association_estimates",
    "table:robustness_matrix",
    "table:robustness_summary",
)
COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS = (
    "table:adjusted_association_estimates",
    "table:absolute_risk_context",
    "table:robustness_summary",
    "table:measurement_process_audit",
)
_ASSOCIATION_SENSITIVITY_FIXED_INPUTS = frozenset(
    {
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
        "table:exposure_component_completeness_audit",
    }
)
_SCIENTIFIC_SENSITIVITY_REQUIRED_COLUMNS = frozenset(
    {
        "analysis_id",
        "is_reference",
        "n_stays",
        "n_events",
        "estimate",
        "ci_low",
        "ci_high",
        "effect_measure",
        "converged",
    }
)

_REQUIRED_COLUMNS = {
    "table:cohort_flow": frozenset({"n_remaining"}),
    "table:table_one": frozenset(
        {
            "variable",
            "group",
            "absolute_standardized_mean_difference",
            "standardized_difference_status",
        }
    ),
    "table:exposure_outcome_distribution": frozenset(
        {
            "row_role",
            "exposure_level",
            "n_rows",
            "exposure_denominator",
            "exposure_pct",
            "outcome_events",
            "outcome_denominator",
            "outcome_rate_pct",
        }
    ),
    "table:missingness_measurement_audit": frozenset(
        {"variable", "n_total", "missing_n", "missing_pct"}
    ),
    "table:measurement_process_audit": frozenset(
        {"concept", "n_total", "measured_one_n", "eligible_n"}
    ),
    "table:robustness_summary": frozenset(
        {
            "axis",
            "total_specs",
            "converged_specs",
            "non_independent_specs",
            "range_low",
            "range_high",
        }
    ),
    "table:adjusted_association_estimates": frozenset(
        {"fit_status", "estimate", "ci_low", "ci_high", "effect_scale", "model_id"}
    ),
    "table:absolute_risk_context": frozenset(
        {
            "estimate_type",
            "label",
            "n",
            "event_n",
            "estimate",
            "ci_low",
            "ci_high",
        }
    ),
    "table:robustness_matrix": frozenset(
        {"spec_id", "point_estimate", "ci_low", "ci_high", "effect_scale", "converged"}
    ),
    "table:measurement_missingness": frozenset(
        {"variable", "n_total", "missing_n", "missing_pct"}
    ),
    "table:exposure_component_completeness_audit": frozenset(
        {
            "concept",
            "exposure_category",
            "row_role",
            "n_stratum",
            "measured_n",
            "measured_pct",
        }
    ),
}

_COMPOSITE_DESCRIPTIVE_FIGURE_PROFILES = (
    COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS,
    COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS,
    COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS,
    COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS,
    COMPOSITE_ASSOCIATION_MEASUREMENT_PUBLICATION_FIGURE_INPUTS,
    COMPOSITE_ASSOCIATION_ROBUSTNESS_PUBLICATION_FIGURE_INPUTS,
    COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS,
    COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS,
)
_COMPOSITE_DESCRIPTIVE_FIGURE_CAPABILITIES = tuple(
    TypedInputCapability(required=frozenset(profile))
    for profile in _COMPOSITE_DESCRIPTIVE_FIGURE_PROFILES
)


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", product)
    ):
        return None
    return product


def _binding_carries_required_columns(binding: Any, input_key: str) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    return bool(
        isinstance(columns, list)
        and all(isinstance(column, str) for column in columns)
        and _REQUIRED_COLUMNS[input_key] <= set(columns)
    )


def _association_sensitivity_input(inputs: tuple[str, ...]) -> str | None:
    values = tuple(str(value) for value in inputs)
    extra = [value for value in values if value not in _ASSOCIATION_SENSITIVITY_FIXED_INPUTS]
    if (
        len(values) == 4
        and len(values) == len(set(values))
        and _ASSOCIATION_SENSITIVITY_FIXED_INPUTS <= set(values)
        and len(extra) == 1
        and extra[0].startswith("table:")
        and extra[0] not in _REQUIRED_COLUMNS
    ):
        return extra[0]
    return None


def composite_descriptive_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Own only the exact typed four-table, one-figure contract."""

    products = [_figure_product(value) for value in step.expected_outputs]
    profile = next(
        (
            candidate
            for candidate, capability in zip(
                _COMPOSITE_DESCRIPTIVE_FIGURE_PROFILES,
                _COMPOSITE_DESCRIPTIVE_FIGURE_CAPABILITIES,
            )
            if capability.admits_step(step)
        ),
        None,
    )
    dynamic_sensitivity_input = _association_sensitivity_input(tuple(step.inputs))
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and (profile is not None or dynamic_sensitivity_input is not None)
        and len(products) == 1
        and products[0] is not None
        and step.trajectory_stability_spec is None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == set(profile or tuple(step.inputs))
    ):
        return False
    if dynamic_sensitivity_input is not None:
        for key in _ASSOCIATION_SENSITIVITY_FIXED_INPUTS:
            if not _binding_carries_required_columns(resolved_bindings.get(key), key):
                return False
        binding = resolved_bindings.get(dynamic_sensitivity_input)
        contract = binding.get("product_contract") if isinstance(binding, Mapping) else None
        columns = contract.get("columns") if isinstance(contract, Mapping) else None
        return bool(
            isinstance(columns, list)
            and _SCIENTIFIC_SENSITIVITY_REQUIRED_COLUMNS <= set(columns)
        )
    assert profile is not None
    return all(
        _binding_carries_required_columns(resolved_bindings.get(key), key)
        for key in profile
    )


def composite_descriptive_figure_consumed_input_keys(
    step: AnalysisStep,
) -> tuple[str, ...]:
    for profile, capability in zip(
        _COMPOSITE_DESCRIPTIVE_FIGURE_PROFILES,
        _COMPOSITE_DESCRIPTIVE_FIGURE_CAPABILITIES,
    ):
        if capability.admits_step(step):
            return profile
    if _association_sensitivity_input(tuple(step.inputs)) is not None:
        return tuple(str(value) for value in step.inputs)
    return ()


def composite_descriptive_figure_executor_code(
    step: AnalysisStep,
    *,
    display_labels: Mapping[str, str] | None = None,
) -> str:
    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("composite descriptive figure has no safe figure product")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.composite_descriptive_figure_executor import (
            run_composite_descriptive_figure,
        )

        run_composite_descriptive_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            input_keys={composite_descriptive_figure_consumed_input_keys(step)!r},
            display_labels={dict(display_labels or {})!r},
        )
        """
    ).strip()


def _load_inputs(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    input_keys: tuple[str, ...],
) -> dict[str, BoundTypedInput]:
    return {
        key: load_typed_input(
            input_key=key,
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            expected_declared_kind="table",
            expected_evidence_kind="table",
            require_consumption_contract=True,
            minimum_row_count=1,
        )
        for key in input_keys
    }


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column!r} must contain only finite numeric values")
    return values.astype(float)


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _finite_series(frame, column)
    if not np.isclose(values, np.rint(values), rtol=0.0, atol=1e-9).all():
        raise ValueError(f"{column!r} must contain only integer-like values")
    return values.astype("int64")


def _assert_percentage(
    *,
    reported: pd.Series,
    numerator: pd.Series,
    denominator: pd.Series,
    label: str,
) -> None:
    if (
        (denominator <= 0).any()
        or (numerator < 0).any()
        or (numerator > denominator).any()
    ):
        raise ValueError(f"{label} counts do not nest within positive denominators")
    expected = 100.0 * numerator.astype(float) / denominator.astype(float)
    # Source tables may persist percentages rounded to six decimal places.
    if not np.isclose(reported, expected, rtol=0.0, atol=5e-6).all():
        raise ValueError(f"{label} does not reconcile to its counts and denominators")


def _reader_label(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "Variable"
    return re.sub(r"[_\s]+", " ", str(value).strip()) or "Variable"


def _write_exact_source(bound: BoundTypedInput, *, out_dir: Path) -> str:
    source = bound.frame.copy()
    source.insert(0, "source_row_index", source.index.astype(int))
    source.insert(1, "source_table", bound.path.name)
    name = f"{bound.product}_source_data.csv"
    source.to_csv(out_dir / name, index=False)
    return name


def run_composite_descriptive_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...] = COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS,
    display_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Render four source-bound descriptive panels without model-authored code."""

    if _figure_product(f"figure:{figure_product}") is None:
        raise ValueError("unsafe or malformed figure product id")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bound = _load_inputs(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        input_keys=input_keys,
    )
    sensitivity_key = _association_sensitivity_input(tuple(input_keys))
    for key, item in bound.items():
        required = (
            _SCIENTIFIC_SENSITIVITY_REQUIRED_COLUMNS
            if key == sensitivity_key
            else _REQUIRED_COLUMNS[key]
        )
        missing = required - set(item.frame.columns)
        if missing:
            raise ValueError(f"{key} is missing required columns: {sorted(missing)!r}")

    if tuple(input_keys) in {
        COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS,
        COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS,
        COMPOSITE_ASSOCIATION_MEASUREMENT_PUBLICATION_FIGURE_INPUTS,
        COMPOSITE_ASSOCIATION_ROBUSTNESS_PUBLICATION_FIGURE_INPUTS,
        COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS,
        COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS,
    } or sensitivity_key is not None:
        from .association_publication_figure_renderer import (
            render_association_publication_figure,
        )

        return render_association_publication_figure(
            bound=bound,
            out_dir=out_dir,
            step_id=step_id,
            figure_product=figure_product,
            input_keys=input_keys,
            display_labels=display_labels,
        )

    flow = bound["table:cohort_flow"].frame.copy()
    distribution = bound["table:exposure_outcome_distribution"].frame.copy()
    missingness = bound["table:missingness_measurement_audit"].frame.copy()
    process = (
        bound["table:measurement_process_audit"].frame.copy()
        if "table:measurement_process_audit" in bound
        else None
    )
    robustness = (
        bound["table:robustness_summary"].frame.copy()
        if "table:robustness_summary" in bound
        else None
    )

    levels = distribution.loc[
        distribution["row_role"].astype(str).eq("exposure_level")
    ].copy()
    if levels.empty:
        raise ValueError("exposure/outcome distribution has no exposure-level rows")
    levels["n_rows"] = _integer_series(levels, "n_rows")
    levels["exposure_denominator"] = _integer_series(levels, "exposure_denominator")
    levels["outcome_events"] = _integer_series(levels, "outcome_events")
    levels["outcome_denominator"] = _integer_series(levels, "outcome_denominator")
    levels["exposure_pct"] = _finite_series(levels, "exposure_pct")
    levels["outcome_rate_pct"] = _finite_series(levels, "outcome_rate_pct")
    _assert_percentage(
        reported=levels["exposure_pct"],
        numerator=levels["n_rows"],
        denominator=levels["exposure_denominator"],
        label="exposure percentage",
    )
    _assert_percentage(
        reported=levels["outcome_rate_pct"],
        numerator=levels["outcome_events"],
        denominator=levels["outcome_denominator"],
        label="outcome percentage",
    )
    missingness["n_total"] = _integer_series(missingness, "n_total")
    missingness["missing_n"] = _integer_series(missingness, "missing_n")
    missingness["missing_pct"] = _finite_series(missingness, "missing_pct")
    _assert_percentage(
        reported=missingness["missing_pct"],
        numerator=missingness["missing_n"],
        denominator=missingness["n_total"],
        label="missingness percentage",
    )
    process_display_pct = None
    if process is not None:
        process["n_total"] = _integer_series(process, "n_total")
        process["measured_one_n"] = _integer_series(process, "measured_one_n")
        if (process["n_total"] <= 0).any() or (
            process["measured_one_n"] > process["n_total"]
        ).any():
            raise ValueError(
                "measurement-process counts do not nest within denominators"
            )
        process_display_pct = 100.0 * process["measured_one_n"] / process["n_total"]
    if robustness is not None:
        for column in ("total_specs", "converged_specs", "non_independent_specs"):
            robustness[column] = _integer_series(robustness, column)
        robustness["range_low"] = _finite_series(robustness, "range_low")
        robustness["range_high"] = _finite_series(robustness, "range_high")
        if (
            (robustness["total_specs"] <= 0).any()
            or (robustness["converged_specs"] < 0).any()
            or (robustness["converged_specs"] > robustness["total_specs"]).any()
            or (robustness["non_independent_specs"] < 0).any()
            or (robustness["non_independent_specs"] > robustness["total_specs"]).any()
            or (robustness["range_low"] > robustness["range_high"]).any()
        ):
            raise ValueError("robustness summary counts or ranges are inconsistent")

    source_files = [
        _write_exact_source(bound[key], out_dir=out_dir) for key in input_keys
    ]
    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}

    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    remaining = _integer_series(flow, "n_remaining")
    if (remaining < 0).any():
        raise ValueError("cohort-flow counts must be non-negative")
    positions = np.arange(len(flow))
    ax.barh(positions, remaining, color=palette["blue"])
    flow_labels = []
    for index, row in flow.iterrows():
        concept = row.get("concept_id")
        predicate = row.get("predicate_kind")
        if concept is not None and not pd.isna(concept):
            flow_labels.append(_reader_label(concept))
        elif predicate is not None and not pd.isna(predicate):
            flow_labels.append(_reader_label(predicate))
        else:
            flow_labels.append(f"Cohort step {index + 1}")
    ax.set_yticks(positions, flow_labels)
    ax.invert_yaxis()
    ax.set_xlabel("ICU stays remaining")
    ax.set_title("Cohort accounting", loc="left", pad=12)
    add_panel_label(ax, "A", x=-0.12, y=1.04)

    ax = axes[0, 1]
    display_labels = dict(display_labels or {})
    exposure_column = str(levels.iloc[0].get("exposure_column") or "exposure")
    labels = []
    for value in levels["exposure_level"]:
        numeric = float(value)
        level = str(int(numeric)) if numeric.is_integer() else str(numeric)
        labels.append(
            display_labels.get(f"{exposure_column}={level}", _reader_label(value))
        )
    positions = np.arange(len(levels))
    width = 0.36
    ax.bar(
        positions - width / 2,
        levels["exposure_pct"],
        width,
        label="Cohort share",
        color=palette["blue"],
    )
    ax.bar(
        positions + width / 2,
        levels["outcome_rate_pct"],
        width,
        label="Outcome rate",
        color=palette["orange"],
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Percent")
    ax.set_title("Exposure and observed outcome", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=6.2)
    add_panel_label(ax, "B", x=-0.12, y=1.04)

    missing_order = missingness.sort_values("missing_pct", ascending=True)
    missing_label_column = "label" if "label" in missing_order.columns else "variable"
    ax = axes[1, 0]
    positions = np.arange(len(missing_order))
    ax.barh(positions, missing_order["missing_pct"], color=palette["orange"])
    ax.set_yticks(
        positions,
        [_reader_label(value) for value in missing_order[missing_label_column]],
        fontsize=5.8,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Measurement missingness", loc="left", pad=12)
    add_panel_label(ax, "C", x=-0.12, y=1.04)

    ax = axes[1, 1]
    if process is not None and process_display_pct is not None:
        process_order = process.assign(_display_pct=process_display_pct).sort_values(
            "_display_pct", ascending=True
        )
        positions = np.arange(len(process_order))
        ax.barh(positions, process_order["_display_pct"], color=palette["blue_soft"])
        ax.set_yticks(
            positions,
            [_reader_label(value) for value in process_order["concept"]],
            fontsize=5.8,
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Measured at least once (%)")
        panel_d_title = "Measurement process"
        panel_d_role = "data_quality"
        panel_d_input = "table:measurement_process_audit"
    elif robustness is not None:
        positions = np.arange(len(robustness))
        centres = (robustness["range_low"] + robustness["range_high"]) / 2.0
        errors = np.vstack(
            [centres - robustness["range_low"], robustness["range_high"] - centres]
        )
        ax.errorbar(
            centres,
            positions,
            xerr=errors,
            fmt="o",
            color=palette["blue"],
            capsize=2.5,
        )
        ax.set_yticks(
            positions,
            [
                f"{_reader_label(row.axis)} ({int(row.converged_specs)}/{int(row.total_specs)})"
                for row in robustness.itertuples(index=False)
            ],
            fontsize=5.8,
        )
        ax.set_xlabel("Estimate range")
        panel_d_title = "Robustness range"
        panel_d_role = "robustness"
        panel_d_input = "table:robustness_summary"
    else:  # pragma: no cover - guarded by the typed profile
        raise ValueError("unsupported composite descriptive input profile")
    ax.set_title(panel_d_title, loc="left", pad=12)
    add_panel_label(ax, "D", x=-0.12, y=1.04)

    panel_specs = [
        (
            "A",
            "Cohort accounting",
            "cohort_accounting",
            [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[0]],
        ),
        (
            "B",
            "Exposure and observed outcome",
            "descriptive_result",
            [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[1]],
        ),
        (
            "C",
            "Measurement missingness",
            "data_quality",
            [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[2]],
        ),
        ("D", panel_d_title, panel_d_role, [panel_d_input]),
    ]
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The bound cohort accounting, descriptive exposure/outcome, and "
            "measurement-quality tables provide a traceable overview of the "
            "analysed population and its observability."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=178.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "claim": f"This panel renders every row of {sources[0]}.",
                "evidence_ids": [evidence[sources[0]]],
                "metadata": {
                    "source_products": sources,
                    "source_data": [f"{sources[0].partition(':')[2]}_source_data.csv"],
                },
            }
            for panel_id, title, role, sources in panel_specs
        ],
        source_data=source_files,
        statistics_note=(
            "All source rows and original value columns are preserved. "
            "Displayed percentages are either bound estimates or arithmetic "
            "ratios of the shown bound numerator and denominator; no model is fit."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)

    for item in bound.values():
        if sha256_file(item.path) != item.sha256:
            raise ValueError(f"typed input changed while rendering: {item.input_key}")
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_composite_descriptive_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "composite_descriptive_figure",
        "rendering_only": True,
        "source_inputs": list(input_keys),
        "input_bindings": [
            {
                "input_key": key,
                "evidence_id": item.evidence_id,
                "sha256": item.sha256,
                "loaded": True,
                "row_count": item.row_count,
            }
            for key, item in bound.items()
        ],
        "source_data_files": source_files,
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "COMPOSITE_ASSOCIATION_MEASUREMENT_PUBLICATION_FIGURE_INPUTS",
    "COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS",
    "COMPOSITE_ASSOCIATION_ROBUSTNESS_PUBLICATION_FIGURE_INPUTS",
    "COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS",
    "COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS",
    "COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS",
    "composite_descriptive_figure_consumed_input_keys",
    "composite_descriptive_figure_executor_code",
    "composite_descriptive_figure_executor_owns_step",
    "run_composite_descriptive_figure",
]
