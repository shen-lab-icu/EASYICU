"""Deterministic renderer for a closed prevalence-and-outcome-risk table.

The executor is intentionally narrow.  It consumes one digest-bound
``table:absolute_risk_context`` input under an ``all_rows`` contract and
renders the two binary partitions already chosen by the Planner/producer.  It
does not read the cohort, choose an exposure or outcome, define categories, or
fit a model.
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

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .figure_input_capability import TypedInputCapability

__all__ = [
    "PREVALENCE_OUTCOME_FIGURE_INPUT",
    "prevalence_outcome_figure_executor_code",
    "prevalence_outcome_figure_executor_owns_step",
    "run_prevalence_outcome_figure",
]


PREVALENCE_OUTCOME_FIGURE_INPUT = "table:absolute_risk_context"
_SUPPORTED_FIGURE_PRODUCTS = frozenset(
    {
        "absolute_risk_context",
        "prevalence_mortality",
        "prevalence_outcome",
    }
)
_REQUIRED_COLUMNS = (
    "row_type",
    "variable",
    "category",
    "n",
    "denominator",
    "percentage",
    "estimate",
    "ci_lower",
    "ci_upper",
    "events",
    "missing_n",
    "missing_pct",
)


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]*", product)
    ):
        return None
    return product


#: One table, read whole; there is nothing this renderer could do without it.
PREVALENCE_OUTCOME_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset({PREVALENCE_OUTCOME_FIGURE_INPUT}),
)


def prevalence_outcome_figure_executor_owns_step(step: AnalysisStep) -> bool:
    """Return whether every scientific choice is fixed by the typed contract."""

    products = [_figure_product(value) for value in step.expected_outputs]
    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and PREVALENCE_OUTCOME_FIGURE_CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] in _SUPPORTED_FIGURE_PRODUCTS
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


def prevalence_outcome_figure_executor_code(step: AnalysisStep) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    if not prevalence_outcome_figure_executor_owns_step(step):
        raise ValueError("The step is not owned by the prevalence/outcome renderer")
    product = _figure_product(step.expected_outputs[0])
    assert product is not None
    return textwrap.dedent(f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.prevalence_outcome_figure_executor import (
            run_prevalence_outcome_figure,
        )

        run_prevalence_outcome_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """).strip()


def _canonical_sha256(path: Path) -> str:
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
) -> tuple[pd.DataFrame, Mapping[str, Any], str]:
    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        resolved_path = Path(resolved_inputs)
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {PREVALENCE_OUTCOME_FIGURE_INPUT}
        or not isinstance(inputs.get(PREVALENCE_OUTCOME_FIGURE_INPUT), dict)
    ):
        raise ValueError("exact absolute-risk table binding is absent or widened")
    binding = inputs[PREVALENCE_OUTCOME_FIGURE_INPUT]
    expected_sha256 = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    product_contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    identity = binding.get("identity_row")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(product_contract, dict)
        or not isinstance(consumption, dict)
        or not isinstance(identity, dict)
        or binding.get("declared_kind") != "table"
        or binding.get("evidence_kind") != "table"
        or binding.get("product") != "absolute_risk_context"
        or identity.get("input_key") != PREVALENCE_OUTCOME_FIGURE_INPUT
        or identity.get("product") != "absolute_risk_context"
        or identity.get("sha256") != expected_sha256
        or consumption.get("input_key") != PREVALENCE_OUTCOME_FIGURE_INPUT
        or consumption.get("mode") != "all_rows"
        or consumption.get("artifact_sha256") != expected_sha256
    ):
        raise ValueError("absolute-risk table authority binding is incomplete")

    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError(
            "absolute-risk table binding escapes the run directory"
        ) from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("absolute-risk table must be a regular bound CSV")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("absolute-risk table digest verification failed")

    columns = product_contract.get("columns")
    row_count = product_contract.get("row_count")
    if (
        columns != list(_REQUIRED_COLUMNS)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 4
        or consumption.get("verified_row_count") != row_count
    ):
        raise ValueError("absolute-risk table product contract is unsupported")
    frame = pd.read_csv(path)
    if list(frame.columns) != list(_REQUIRED_COLUMNS) or len(frame) != row_count:
        raise ValueError("absolute-risk table bytes disagree with its product contract")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("absolute-risk table changed while it was being read")
    return frame, binding, "absolute_risk_context.csv"


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    parsed = _finite(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _fraction_matches(value: Any, expected: float) -> bool:
    parsed = _finite(value)
    return parsed is not None and math.isclose(
        parsed, expected, rel_tol=1e-9, abs_tol=1e-10
    )


def _binary_categories(rows: pd.DataFrame) -> list[int] | None:
    values = pd.to_numeric(rows["category"], errors="coerce")
    if (
        values.isna().any()
        or not values.mod(1).eq(0).all()
        or set(values.astype(int)) != {0, 1}
        or len(values) != 2
    ):
        return None
    return values.astype(int).tolist()


def _validated_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, str, str]:
    row_types = frame["row_type"].fillna("").astype(str).str.strip()
    prevalence = frame.loc[row_types.eq("prevalence")].copy()
    outcome = frame.loc[~row_types.isin({"cohort_context", "prevalence"})].copy()
    if (
        len(prevalence) != 2
        or len(outcome) != 2
        or row_types.eq("").any()
        or set(row_types)
        != {
            "cohort_context",
            "prevalence",
            str(outcome["row_type"].iloc[0]),
        }
        or outcome["row_type"].astype(str).nunique(dropna=False) != 1
        or prevalence["variable"].astype(str).nunique(dropna=False) != 1
        or outcome["variable"].astype(str).nunique(dropna=False) != 1
    ):
        raise ValueError("absolute-risk table does not contain one closed binary pair")
    prevalence_categories = _binary_categories(prevalence)
    outcome_categories = _binary_categories(outcome)
    if prevalence_categories is None or outcome_categories is None:
        raise ValueError("prevalence and outcome rows must contain categories 0 and 1")

    prevalence["__category"] = prevalence_categories
    outcome["__category"] = outcome_categories
    prevalence = prevalence.sort_values("__category")
    outcome = outcome.sort_values("__category")
    locked_values = {_integer(value) for value in prevalence["denominator"]}
    if None in locked_values or len(locked_values) != 1:
        raise ValueError("prevalence rows do not share a finite denominator")
    locked_denominator = next(iter(locked_values))
    assert locked_denominator is not None
    if locked_denominator <= 0:
        raise ValueError("prevalence denominator must be positive")

    prevalence_counts: dict[int, int] = {}
    for _, row in prevalence.iterrows():
        category = int(row["__category"])
        count = _integer(row["n"])
        if count is None or count > locked_denominator:
            raise ValueError("prevalence count is invalid")
        expected = count / locked_denominator
        if not _fraction_matches(row["estimate"], expected) or not _fraction_matches(
            row["percentage"], expected
        ):
            raise ValueError("prevalence fraction does not reconcile to its count")
        if pd.notna(row["events"]):
            raise ValueError(
                "prevalence rows must not relabel counts as outcome events"
            )
        prevalence_counts[category] = count
    if sum(prevalence_counts.values()) != locked_denominator:
        raise ValueError("binary prevalence rows do not partition the denominator")

    for _, row in outcome.iterrows():
        category = int(row["__category"])
        denominator = _integer(row["denominator"])
        count = _integer(row["n"])
        events = _integer(row["events"])
        if (
            denominator is None
            or count is None
            or events is None
            or denominator != count
            or count != prevalence_counts[category]
            or events > denominator
        ):
            raise ValueError("outcome counts do not reconcile to the prevalence group")
        expected = events / denominator
        ci_lower = _finite(row["ci_lower"])
        ci_upper = _finite(row["ci_upper"])
        if (
            not _fraction_matches(row["estimate"], expected)
            or not _fraction_matches(row["percentage"], expected)
            or ci_lower is None
            or ci_upper is None
            or not 0 <= ci_lower <= expected <= ci_upper <= 1
        ):
            raise ValueError("outcome risk or confidence interval is inconsistent")
        missing_n = _finite(row["missing_n"])
        missing_pct = _finite(row["missing_pct"])
        if (
            missing_n is not None and not math.isclose(missing_n, 0.0, abs_tol=1e-12)
        ) or (
            missing_pct is not None
            and not math.isclose(missing_pct, 0.0, abs_tol=1e-12)
        ):
            raise ValueError("outcome risk rows include unresolved missing outcomes")

    exposure_label = str(prevalence["variable"].iloc[0]).strip()
    outcome_label = str(outcome["row_type"].iloc[0]).strip()
    if not exposure_label or not outcome_label:
        raise ValueError("prevalence/outcome labels are empty")
    return prevalence, outcome, locked_denominator, exposure_label, outcome_label


def _reader_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).strip()
    return cleaned.title() if cleaned else "Outcome"


def _write_source_projection(
    rows: pd.DataFrame,
    *,
    path: Path,
    source_table: str,
) -> None:
    export = rows.drop(columns=["__category"], errors="ignore").copy()
    export.insert(0, "source_row_index", export.index.astype(int))
    export.insert(1, "source_table", source_table)
    export.to_csv(path, index=False)


def run_prevalence_outcome_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> Mapping[str, Any]:
    """Render one verified binary prevalence/outcome table and write its contract."""

    if figure_product not in _SUPPORTED_FIGURE_PRODUCTS:
        raise ValueError("unsupported prevalence/outcome figure product")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, binding, source_table = _load_binding(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
    )
    prevalence, outcome, locked_n, exposure_name, outcome_name = _validated_rows(frame)

    full_source = out_dir / f"{figure_product}_input_source_data.csv"
    prevalence_source = out_dir / f"{figure_product}_prevalence_source_data.csv"
    outcome_source = out_dir / f"{figure_product}_outcome_source_data.csv"
    _write_source_projection(frame, path=full_source, source_table=source_table)
    _write_source_projection(
        prevalence,
        path=prevalence_source,
        source_table=source_table,
    )
    _write_source_projection(outcome, path=outcome_source, source_table=source_table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 86 / 25.4),
        gridspec_kw={"width_ratios": [0.95, 1.15]},
    )
    categories = prevalence["__category"].astype(int).tolist()
    category_labels = [f"Category {value}" for value in categories]
    prevalence_pct = pd.to_numeric(prevalence["estimate"]).to_numpy() * 100.0
    prevalence_counts = pd.to_numeric(prevalence["n"]).astype(int).to_numpy()
    positions = list(range(len(categories)))
    bars = ax_a.barh(
        positions,
        prevalence_pct,
        color=palette["blue_soft"],
        height=0.58,
    )
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(category_labels)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Analysis cohort (%)")
    ax_a.set_title(f"{_reader_label(exposure_name)} prevalence", loc="left", pad=4)
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for bar, percentage, count in zip(bars, prevalence_pct, prevalence_counts):
        ax_a.text(
            min(float(percentage) + 1.0, 97.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.1f}%  n={int(count):,}",
            va="center",
            ha="left" if percentage < 94 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.14, y=1.04)

    outcome_pct = pd.to_numeric(outcome["estimate"]).to_numpy() * 100.0
    ci_low = pd.to_numeric(outcome["ci_lower"]).to_numpy() * 100.0
    ci_high = pd.to_numeric(outcome["ci_upper"]).to_numpy() * 100.0
    outcome_events = pd.to_numeric(outcome["events"]).astype(int).to_numpy()
    outcome_denominators = pd.to_numeric(outcome["denominator"]).astype(int).to_numpy()
    ax_b.errorbar(
        outcome_pct,
        positions,
        xerr=[outcome_pct - ci_low, ci_high - outcome_pct],
        fmt="o",
        color=palette["blue"],
        ecolor=palette["neutral"],
        elinewidth=1.0,
        capsize=2.0,
        markersize=4.2,
    )
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(category_labels)
    ax_b.invert_yaxis()
    upper = min(100.0, max(5.0, float(ci_high.max()) * 1.35))
    ax_b.set_xlim(0, upper)
    ax_b.set_xlabel("Absolute outcome risk (%)")
    ax_b.set_title(_reader_label(outcome_name), loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, estimate, events, denominator in zip(
        positions, outcome_pct, outcome_events, outcome_denominators
    ):
        ax_b.text(
            min(float(estimate) + upper * 0.025, upper * 0.98),
            position,
            f"{float(estimate):.1f}%  {int(events):,}/{int(denominator):,}",
            va="center",
            ha="left" if estimate < upper * 0.86 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_b, "B", x=-0.14, y=1.04)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.84, wspace=0.48)

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The complete binary exposure partition and absolute outcome risks "
            "are rendered from one digest-verified parent table."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=86.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Exposure prevalence",
                "role": "baseline_context",
                "claim": (
                    "Both parent-defined exposure categories partition the locked "
                    "analysis denominator without dropping rows."
                ),
                "evidence_ids": [prevalence_source.name],
                "metadata": {
                    "chart_type": "bar_prevalence",
                    "source_data": [prevalence_source.name],
                },
            },
            {
                "panel_id": "B",
                "title": "Absolute outcome risk",
                "role": "descriptive_result",
                "claim": (
                    "Outcome events, denominators, point estimates, and confidence "
                    "intervals are shown for both parent-defined categories."
                ),
                "evidence_ids": [outcome_source.name],
                "metadata": {
                    "chart_type": "dot_interval_absolute_risk",
                    "source_data": [outcome_source.name],
                },
            },
        ],
        source_data=[
            full_source.name,
            prevalence_source.name,
            outcome_source.name,
        ],
        statistics_note=(
            "Percentages are 100 times the source-table fractions. The executor "
            "validates all source rows and introduces no cohort, exposure, outcome, "
            "grouping, missing-data, or modeling decision."
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
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_prevalence_outcome_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "prevalence_outcome_figure",
        "rendering_only": True,
        "source_input": PREVALENCE_OUTCOME_FIGURE_INPUT,
        "source_evidence_id": binding.get("evidence_id"),
        "source_sha256": binding.get("sha256"),
        "source_rows_consumed": int(len(frame)),
        "source_table": source_table,
        "locked_denominator": locked_n,
        "source_data_files": [
            full_source.name,
            prevalence_source.name,
            outcome_source.name,
        ],
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
