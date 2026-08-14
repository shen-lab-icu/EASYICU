"""Deterministic rendering for already-computed descriptive results.

This owner makes no scientific choice.  It accepts either one host-typed
distribution table with the locked descriptive-summary schema, or one
host-typed scalar statistic, verifies the exact evidence bytes, and renders
only those values.  Cohort, variable, grouping, estimator, and missing-data
decisions remain with the producing Planner step.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...contracts.figure_plan import (
    GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS,
)
from ...figures.publication import (
    PALETTE_CLINICAL,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep

__all__ = [
    "DESCRIPTIVE_DISTRIBUTION_COLUMNS",
    "descriptive_result_figure_executor_code",
    "descriptive_result_figure_executor_owns_step",
    "run_descriptive_result_figure",
]


DESCRIPTIVE_DISTRIBUTION_COLUMNS = (
    "variable",
    "group",
    "row_role",
    "n_total",
    "n_nonmissing",
    "missing_n",
    "missing_pct",
    "median",
    "q25",
    "q75",
    "mean",
    "sd",
)

_DESCRIPTIVE_DISTRIBUTION_REQUIRED_COLUMNS = tuple(
    column
    for column in DESCRIPTIVE_DISTRIBUTION_COLUMNS
    if column not in {"row_role", "n_total"}
)

# The producing step may name the exact group denominator either ``group_n``
# (the canonical grouped-distribution contract) or ``n_total`` (the generic
# descriptive-summary contract).  Both carry the same checkable invariant;
# accepting any other count spelling would make the renderer guess.
DESCRIPTIVE_DISTRIBUTION_COUNT_COLUMNS = frozenset({"group_n", "n_total"})

_WIDE_DISTRIBUTION_SUFFIXES = (
    "n_nonmissing",
    "missing_n",
    "missing_pct",
    "median",
    "q25",
    "q75",
    "mean",
    "sd",
)


def _wide_distribution_shape(columns: list[str]) -> tuple[str, str] | None:
    """Resolve one strict ``group + metric-prefixed summaries`` table.

    This is the shape emitted by older generated grouped summaries.  It is not
    inferred from values: every summary suffix must identify the same one
    metric stem, and exactly one remaining column must identify the groups.
    Ambiguous or widened tables remain outside this deterministic owner.
    """

    column_set = set(columns)
    stems: set[str] = set()
    summary_columns: set[str] = set()
    for column in columns:
        for suffix in _WIDE_DISTRIBUTION_SUFFIXES:
            marker = f"_{suffix}"
            if column.endswith(marker) and len(column) > len(marker):
                stems.add(column[: -len(marker)])
                summary_columns.add(column)
                break
    if len(stems) != 1:
        return None
    stem = next(iter(stems))
    required_summaries = {f"{stem}_{suffix}" for suffix in _WIDE_DISTRIBUTION_SUFFIXES}
    if summary_columns != required_summaries:
        return None
    fixed = {"n", "percentage", "denominator"}
    group_columns = column_set - required_summaries - fixed
    if not fixed.issubset(column_set) or len(group_columns) != 1:
        return None
    if column_set != required_summaries | fixed | group_columns:
        return None
    return next(iter(group_columns)), stem


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _typed_product(value: Any, expected_kind: str) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != expected_kind
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]*", product)
    ):
        return None
    return product


def _binding_mode(binding: Any) -> str | None:
    if not isinstance(binding, Mapping):
        return None
    contract = binding.get("product_contract")
    if not isinstance(contract, Mapping):
        return None
    declared_kind = str(binding.get("declared_kind") or "")
    evidence_kind = str(binding.get("evidence_kind") or "")
    if declared_kind == evidence_kind == "table":
        columns = contract.get("columns")
        consumption = binding.get("consumption_contract")
        column_names = set(columns) if isinstance(columns, list) else set()
        count_columns = column_names & DESCRIPTIVE_DISTRIBUTION_COUNT_COLUMNS
        if (
            isinstance(columns, list)
            and set(_DESCRIPTIVE_DISTRIBUTION_REQUIRED_COLUMNS) <= column_names
            and len(count_columns) == 1
            and isinstance(consumption, Mapping)
            and consumption.get("mode") == "all_rows"
        ):
            return "distribution_table"
        if (
            isinstance(columns, list)
            and all(isinstance(column, str) for column in columns)
            and _wide_distribution_shape(columns) is not None
            and isinstance(consumption, Mapping)
            and consumption.get("mode") == "all_rows"
        ):
            return "distribution_table_wide"
    if declared_kind == evidence_kind == "statistic":
        structure = contract.get("json_structure")
        paths = structure.get("paths") if isinstance(structure, Mapping) else None
        root = paths.get("") if isinstance(paths, Mapping) else None
        keys = root.get("keys") if isinstance(root, Mapping) else None
        if (
            isinstance(structure, Mapping)
            and structure.get("root_type") == "object"
            and isinstance(keys, list)
            and {"name", "value"} <= set(keys)
        ):
            return "scalar_statistic"
    return None


def descriptive_result_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether one exact typed parent fully fixes the rendering."""

    figures = [_typed_product(value, "figure") for value in step.expected_outputs]
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and len(step.inputs) == 1
        and len(figures) == 1
        and figures[0] is not None
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == {step.inputs[0]}
    ):
        return False
    input_key = step.inputs[0]
    mode = _binding_mode(resolved_bindings.get(input_key))
    if mode in {"distribution_table", "distribution_table_wide"}:
        contracts = list(step.input_consumption_contracts or [])
        return bool(
            _typed_product(input_key, "table")
            and len(contracts) == 1
            and contracts[0].input_key == input_key
            and contracts[0].mode == "all_rows"
        )
    return bool(mode == "scalar_statistic" and _typed_product(input_key, "statistic"))


def descriptive_result_figure_executor_code(step: AnalysisStep) -> str:
    if len(step.inputs) != 1 or len(step.expected_outputs) != 1:
        raise ValueError("descriptive figure requires one input and one output")
    input_key = step.inputs[0]
    product = _typed_product(step.expected_outputs[0], "figure")
    if product is None:
        raise ValueError("descriptive figure output is not a typed figure")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.descriptive_result_figure_executor import (
            run_descriptive_result_figure,
        )

        run_descriptive_result_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            input_key={input_key!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_binding(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    input_key: str,
) -> tuple[Path, Mapping[str, Any], str]:
    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else None
    if payload.get("step_id") != step_id or not isinstance(inputs, Mapping):
        raise ValueError("resolved-input manifest does not belong to this step")
    if set(inputs) != {input_key} or not isinstance(inputs.get(input_key), Mapping):
        raise ValueError("descriptive figure input binding is absent or widened")
    binding = inputs[input_key]
    mode = _binding_mode(binding)
    if mode is None:
        raise ValueError("descriptive figure input has no supported host contract")

    expected_sha = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    identity = binding.get("identity_row")
    kind, _, product = input_key.partition(":")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(identity, Mapping)
        or binding.get("declared_kind") != kind
        or binding.get("evidence_kind") != kind
        or binding.get("product") != product
        or identity.get("input_key") != input_key
        or identity.get("product") != product
        or identity.get("sha256") != expected_sha
    ):
        raise ValueError("descriptive figure authority binding is incomplete")

    base = Path(run_dir).resolve()
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError("descriptive figure input escapes the run directory") from exc
    expected_suffix = ".csv" if mode.startswith("distribution_table") else ".json"
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != expected_suffix:
        raise ValueError("descriptive figure input is not a safe regular file")
    if _sha256(path) != expected_sha:
        raise ValueError("descriptive figure input digest verification failed")
    return path, binding, mode


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_contract_and_summary(
    *,
    fig: Any,
    out_dir: Path,
    figure_product: str,
    input_key: str,
    binding: Mapping[str, Any],
    source_path: Path,
    core_claim: str,
    panel_title: str,
    panel_id: str,
    panel_role: str,
    chart_type: str,
    statistics_note: str,
    source_rows: int,
) -> dict[str, Any]:
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=core_claim,
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=92.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": panel_title,
                "role": panel_role,
                "claim": core_claim,
                "evidence_ids": [str(binding.get("evidence_id") or "")],
                "metadata": {
                    "article_role": panel_role,
                    "chart_type": chart_type,
                    "source_products": [input_key],
                    "source_data": [source_path.name],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=statistics_note,
    )
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": "",
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_descriptive_result_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "descriptive_result_figure",
        "rendering_only": True,
        "source_input": input_key,
        "source_step_id": binding.get("produced_by_step"),
        "source_evidence_id": binding.get("evidence_id"),
        "source_sha256": binding.get("sha256"),
        "source_rows_consumed": source_rows,
        "input_bindings": [
            {
                "input_key": input_key,
                "evidence_id": binding.get("evidence_id"),
                "sha256": binding.get("sha256"),
                "loaded": True,
                "row_count": source_rows,
            }
        ],
        "source_data_files": [source_path.name],
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    return summary


def run_descriptive_result_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    input_key: str,
    figure_product: str,
) -> dict[str, Any]:
    """Verify one typed result and render it without changing its science."""

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path, binding, mode = _load_binding(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        input_key=input_key,
    )
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    if mode.startswith("distribution_table"):
        raw_frame = pd.read_csv(path)
        contract = binding["product_contract"]
        expected_rows = contract.get("row_count")
        if (
            list(raw_frame.columns) != list(contract.get("columns") or [])
            or isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or expected_rows < 1
            or len(raw_frame) != expected_rows
        ):
            raise ValueError("distribution table bytes disagree with their contract")
        if mode == "distribution_table_wide":
            shape = _wide_distribution_shape(list(raw_frame.columns))
            if shape is None:  # pragma: no cover - ownership and binding guard it
                raise ValueError("wide distribution contract is ambiguous")
            group_column, stem = shape
            group_n = pd.to_numeric(raw_frame["n"], errors="coerce")
            denominator = pd.to_numeric(raw_frame["denominator"], errors="coerce")
            percentage = pd.to_numeric(raw_frame["percentage"], errors="coerce")
            if (
                group_n.isna().any()
                or denominator.isna().any()
                or percentage.isna().any()
                or (group_n <= 0).any()
                or (denominator <= 0).any()
                or denominator.nunique() != 1
                or int(group_n.sum()) != int(denominator.iloc[0])
                or not (
                    percentage - 100.0 * group_n / denominator
                ).abs().le(1e-6).all()
            ):
                raise ValueError("wide distribution group denominator invariants failed")
            frame = pd.DataFrame(
                {
                    "variable": stem,
                    "group": raw_frame[group_column],
                    "row_role": "exposure_level",
                    "group_n": group_n,
                    "n_nonmissing": raw_frame[f"{stem}_n_nonmissing"],
                    "missing_n": raw_frame[f"{stem}_missing_n"],
                    "missing_pct": raw_frame[f"{stem}_missing_pct"],
                    "median": raw_frame[f"{stem}_median"],
                    "q25": raw_frame[f"{stem}_q25"],
                    "q75": raw_frame[f"{stem}_q75"],
                    "mean": raw_frame[f"{stem}_mean"],
                    "sd": raw_frame[f"{stem}_sd"],
                    "unit": "",
                }
            )
        else:
            frame = raw_frame
        variables = frame["variable"].dropna().astype(str).str.strip().unique()
        groups = frame["group"].dropna().astype(str).str.strip()
        if len(variables) != 1 or groups.eq("").any() or groups.duplicated().any():
            raise ValueError("distribution table must describe one variable and unique groups")
        count_columns = set(frame.columns) & DESCRIPTIVE_DISTRIBUTION_COUNT_COLUMNS
        if len(count_columns) != 1:
            raise ValueError("distribution table must declare one exact group count")
        count_column = next(iter(count_columns))
        for column in (
            count_column,
            "n_nonmissing",
            "missing_n",
            "missing_pct",
            "median",
            "q25",
            "q75",
            "mean",
            "sd",
        ):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                raise ValueError(f"distribution table has invalid {column} values")
        total = pd.to_numeric(frame[count_column])
        observed = pd.to_numeric(frame["n_nonmissing"])
        missing = pd.to_numeric(frame["missing_n"])
        missing_pct = pd.to_numeric(frame["missing_pct"])
        q25 = pd.to_numeric(frame["q25"])
        median = pd.to_numeric(frame["median"])
        q75 = pd.to_numeric(frame["q75"])
        if (
            (total <= 0).any()
            or not (observed + missing).eq(total).all()
            or not (missing_pct - 100.0 * missing / total).abs().le(1e-6).all()
            or not ((q25 <= median) & (median <= q75)).all()
        ):
            raise ValueError("distribution table count or quantile invariants failed")
        source = raw_frame.copy()
        source_name = path.name.split("__", 1)[1] if "__" in path.name else path.name
        source.insert(0, "source_step_id", binding.get("produced_by_step"))
        source.insert(0, "source_table", source_name)
        source.insert(0, "source_row_index", range(len(source)))
        source_path = out_dir / f"{figure_product}_source_data.csv"
        source.to_csv(source_path, index=False)

        positions = list(range(len(frame)))
        ax.errorbar(
            median,
            positions,
            xerr=[median - q25, q75 - median],
            fmt="o",
            color=PALETTE_CLINICAL["blue"],
            ecolor=PALETTE_CLINICAL["neutral"],
            capsize=3,
            markersize=5,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(groups.tolist())
        ax.invert_yaxis()
        units = (
            frame["unit"].dropna().astype(str).str.strip().unique()
            if "unit" in frame.columns
            else []
        )
        axis_label = variables[0]
        if len(units) == 1 and units[0]:
            axis_label = f"{axis_label} ({units[0]})"
        ax.set_xlabel(axis_label)
        ax.set_title("Median and interquartile range", loc="left")
        ax.grid(axis="x", color=PALETTE_CLINICAL["neutral_light"], linewidth=0.6)
        for position, value, n_value in zip(positions, q75, observed):
            ax.annotate(
                f"n={int(n_value):,}",
                (float(value), position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )
        fig.tight_layout()
        summary = _write_contract_and_summary(
            fig=fig,
            out_dir=out_dir,
            figure_product=figure_product,
            input_key=input_key,
            binding=binding,
            source_path=source_path,
            core_claim=(
                "The figure reproduces the median and interquartile range for "
                "every group in one digest-verified descriptive table."
            ),
            panel_title="Descriptive distribution",
            panel_id=(
                GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS[0].panel_id
            ),
            panel_role="distribution",
            chart_type=(
                GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS[0].chart_type
            ),
            statistics_note=(
                "All parent rows are preserved in source data. The renderer "
                "introduces no grouping, filtering, imputation, or estimator choice."
            ),
            source_rows=len(frame),
        )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("scalar statistic payload must be an object")
        product = input_key.partition(":")[2]
        name = str(payload.get("name") or "").strip()
        value = _finite(payload.get("value"))
        if name != product or value is None:
            raise ValueError("scalar statistic name/value disagrees with its binding")
        source = pd.DataFrame(
            [
                {
                    "name": name,
                    "value": value,
                    "effect_scale": str(payload.get("effect_scale") or "").strip(),
                    "unit": str(payload.get("unit") or "").strip(),
                    "source_table": path.name,
                    "source_step_id": binding.get("produced_by_step"),
                }
            ]
        )
        source_path = out_dir / f"{figure_product}_source_data.csv"
        source.to_csv(source_path, index=False)
        label = str(payload.get("effect_scale") or name).strip()
        unit = str(payload.get("unit") or "").strip()
        value_text = f"{value:.4g}" + (f" {unit}" if unit and unit != "unitless" else "")
        ax.axis("off")
        ax.text(0.5, 0.61, value_text, ha="center", va="center", fontsize=24)
        ax.text(0.5, 0.39, label, ha="center", va="center", fontsize=10)
        ax.text(
            0.5,
            0.20,
            "Digest-verified descriptive statistic",
            ha="center",
            va="center",
            fontsize=7,
            color=PALETTE_CLINICAL["neutral"],
        )
        fig.tight_layout()
        summary = _write_contract_and_summary(
            fig=fig,
            out_dir=out_dir,
            figure_product=figure_product,
            input_key=input_key,
            binding=binding,
            source_path=source_path,
            core_claim=(
                "The figure displays one finite scalar from an exact "
                "digest-verified statistic artifact."
            ),
            panel_title="Descriptive statistic",
            panel_id="descriptive_statistic",
            panel_role="descriptive_result",
            chart_type="scalar_display",
            statistics_note=(
                "Only the bound statistic value is plotted. Additional numeric "
                "payload fields are not promoted into unverified figure results."
            ),
            source_rows=1,
        )
    plt.close(fig)
    if _sha256(path) != binding.get("sha256"):
        raise ValueError("descriptive figure input changed while it was being rendered")
    summary["step_id"] = step_id
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
