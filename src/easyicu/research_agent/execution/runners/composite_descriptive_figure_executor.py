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

_REQUIRED_COLUMNS = {
    "table:cohort_flow": frozenset({"n_remaining"}),
    "table:exposure_outcome_distribution": frozenset(
        {
            "row_role",
            "exposure_level",
            "n_rows",
            "exposure_pct",
            "outcome_events",
            "outcome_denominator",
            "outcome_rate_pct",
        }
    ),
    "table:missingness_measurement_audit": frozenset(
        {"variable", "label", "n_total", "missing_n", "missing_pct"}
    ),
    "table:measurement_process_audit": frozenset(
        {"concept", "n_total", "measured_one_n"}
    ),
}

COMPOSITE_DESCRIPTIVE_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset(COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS),
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


def composite_descriptive_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Own only the exact typed four-table, one-figure contract."""

    products = [_figure_product(value) for value in step.expected_outputs]
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and COMPOSITE_DESCRIPTIVE_FIGURE_CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] is not None
        and step.trajectory_stability_spec is None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == set(COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS)
    ):
        return False
    return all(
        _binding_carries_required_columns(resolved_bindings.get(key), key)
        for key in COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS
    )


def composite_descriptive_figure_executor_code(step: AnalysisStep) -> str:
    product = _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
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
        )
        """
    ).strip()


def _load_inputs(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
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
        for key in COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS
    }


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column!r} must contain only finite numeric values")
    return values.astype(float)


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
    )
    for key, item in bound.items():
        missing = _REQUIRED_COLUMNS[key] - set(item.frame.columns)
        if missing:
            raise ValueError(f"{key} is missing required columns: {sorted(missing)!r}")

    flow = bound["table:cohort_flow"].frame.copy()
    distribution = bound["table:exposure_outcome_distribution"].frame.copy()
    missingness = bound["table:missingness_measurement_audit"].frame.copy()
    process = bound["table:measurement_process_audit"].frame.copy()

    levels = distribution.loc[
        distribution["row_role"].astype(str).eq("exposure_level")
    ].copy()
    if levels.empty:
        raise ValueError("exposure/outcome distribution has no exposure-level rows")
    for column in ("exposure_pct", "outcome_rate_pct", "n_rows"):
        levels[column] = _finite_series(levels, column)
    missingness["missing_pct"] = _finite_series(missingness, "missing_pct")
    process["n_total"] = _finite_series(process, "n_total")
    process["measured_one_n"] = _finite_series(process, "measured_one_n")
    if (process["n_total"] <= 0).any() or (
        process["measured_one_n"] > process["n_total"]
    ).any():
        raise ValueError("measurement-process counts do not nest within denominators")
    process_display_pct = 100.0 * process["measured_one_n"] / process["n_total"]

    source_files = [
        _write_exact_source(bound[key], out_dir=out_dir)
        for key in COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS
    ]
    evidence = {
        key: str(item.evidence_id or "") for key, item in bound.items()
    }

    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    remaining = _finite_series(flow, "n_remaining")
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
    labels = [_reader_label(value) for value in levels["exposure_level"]]
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
    ax = axes[1, 0]
    positions = np.arange(len(missing_order))
    ax.barh(positions, missing_order["missing_pct"], color=palette["orange"])
    ax.set_yticks(
        positions,
        [_reader_label(value) for value in missing_order["label"]],
        fontsize=5.8,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Measurement missingness", loc="left", pad=12)
    add_panel_label(ax, "C", x=-0.12, y=1.04)

    process_order = process.assign(_display_pct=process_display_pct).sort_values(
        "_display_pct", ascending=True
    )
    ax = axes[1, 1]
    positions = np.arange(len(process_order))
    ax.barh(positions, process_order["_display_pct"], color=palette["blue_soft"])
    ax.set_yticks(
        positions,
        [_reader_label(value) for value in process_order["concept"]],
        fontsize=5.8,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Measured at least once (%)")
    ax.set_title("Measurement process", loc="left", pad=12)
    add_panel_label(ax, "D", x=-0.12, y=1.04)

    panel_specs = [
        ("A", "Cohort accounting", "cohort_accounting", [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[0]]),
        ("B", "Exposure and observed outcome", "descriptive_result", [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[1]]),
        ("C", "Measurement missingness", "data_quality", [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[2]]),
        ("D", "Measurement process", "data_quality", [COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS[3]]),
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
                    "source_data": [
                        f"{sources[0].partition(':')[2]}_source_data.csv"
                    ],
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
        "source_inputs": list(COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS),
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
    "COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS",
    "composite_descriptive_figure_executor_code",
    "composite_descriptive_figure_executor_owns_step",
    "run_composite_descriptive_figure",
]
