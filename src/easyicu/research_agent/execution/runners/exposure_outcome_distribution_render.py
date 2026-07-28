"""Deterministic renderer for the closed exposure-outcome distribution product.

This renderer consumes **one** table and nothing else. That is the point of it:
the product it reads carries its own denominators, missing counts, event counts
and intervals, so there is no second lookup into a cohort summary to make the
percentages meaningful. A renderer that needed two tables could not have its
input contract closed before its parent ran, which is what left the figure
steps unresolvable in a preflight.

It draws what the parent already measured and decides nothing: no cohort, no
exposure, no outcome, no category, no denominator, no interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT,
)
from .figure_input_capability import TypedInputCapability
from .planner_display_labels import planner_binary_level_labels

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT",
    "exposure_outcome_distribution_figure_code",
    "exposure_outcome_distribution_figure_owns_step",
    "run_exposure_outcome_distribution_figure",
]

EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT = EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT

#: Same rule as the missingness renderer: the figure product id is a
#: Planner-owned label that becomes a filename, never a capability claim.
_FIGURE_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")

_CAPABILITY = TypedInputCapability(
    required=frozenset({EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT}),
)

_OVERALL_ROLE = "overall"
_LEVEL_ROLE = "exposure_level"


def _is_safe_figure_product_id(value: Any) -> bool:
    return bool(_FIGURE_PRODUCT_ID.fullmatch(str(value or "")))


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not _is_safe_figure_product_id(product):
        return None
    return product


def exposure_outcome_distribution_figure_owns_step(step: AnalysisStep) -> bool:
    """Own a rendering-only step whose single typed input is this product."""

    products = [_figure_product(value) for value in step.expected_outputs]
    if not _CAPABILITY.admits_step(step):
        return False
    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and len(products) == 1
        and products[0] is not None
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
    )


def exposure_outcome_distribution_figure_code(
    step: AnalysisStep,
    *,
    display_labels: Mapping[str, str] | None = None,
) -> str:
    if not exposure_outcome_distribution_figure_owns_step(step):
        raise ValueError(
            "The step is not owned by the exposure-outcome distribution renderer"
        )
    product = _figure_product(step.expected_outputs[0])
    resolved = planner_binary_level_labels(display_labels)
    labels = (resolved[1], resolved[2]) if resolved is not None else None
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
            run_exposure_outcome_distribution_figure,
        )

        run_exposure_outcome_distribution_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            level_labels={labels!r},
        )
        """
    ).strip()


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
    """Read exactly the one bound table, verifying digest and schema."""

    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT}
        or not isinstance(inputs.get(EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT), dict)
    ):
        raise ValueError("exact distribution-table binding is absent or widened")
    binding = inputs[EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT]
    expected_sha256 = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    product_contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(product_contract, dict)
        or not isinstance(consumption, dict)
        or binding.get("declared_kind") != "table"
        or consumption.get("input_key") != EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT
        or consumption.get("mode") != "all_rows"
        or consumption.get("artifact_sha256") != expected_sha256
    ):
        raise ValueError("distribution-table authority binding is incomplete")

    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError(
            "distribution-table binding escapes the run directory"
        ) from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("distribution table must be a regular bound CSV")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("distribution-table digest verification failed")

    columns = product_contract.get("columns")
    row_count = product_contract.get("row_count")
    if (
        columns != list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 3
    ):
        raise ValueError("distribution-table product contract is unsupported")
    frame = pd.read_csv(path)
    if (
        list(frame.columns) != list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS)
        or len(frame) != row_count
    ):
        raise ValueError("distribution-table bytes disagree with its product contract")
    return frame, binding, path.name


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Re-check the arithmetic the producer asserted, before drawing it."""

    levels = frame[frame["row_role"] == _LEVEL_ROLE]
    overall = frame[frame["row_role"] == _OVERALL_ROLE]
    if len(overall) != 1:
        raise ValueError("distribution table needs exactly one overall row")
    if len(levels) < 2:
        raise ValueError("distribution table needs at least two exposure levels")
    total = overall.iloc[0]
    if int(levels["n_rows"].sum()) != int(total["n_rows"]):
        raise ValueError("exposure levels do not partition the reported cohort")
    if int(levels["outcome_events"].sum()) != int(total["outcome_events"]):
        raise ValueError("level events do not sum to the overall events")
    for _, row in frame.iterrows():
        if int(row["outcome_observed_n"]) + int(row["outcome_missing_n"]) != int(
            row["n_rows"]
        ):
            raise ValueError("observed plus missing does not equal the row count")
        if int(row["outcome_events"]) > int(row["outcome_denominator"]):
            raise ValueError("more events than the denominator they are taken over")
        rate = _finite(row["outcome_rate_pct"])
        low = _finite(row["ci_low_pct"])
        high = _finite(row["ci_high_pct"])
        if rate is not None and low is not None and high is not None:
            if not (low - 1e-6 <= rate <= high + 1e-6):
                raise ValueError("a reported rate falls outside its own interval")
    return levels, total


def _labels(levels: pd.DataFrame, level_labels: tuple[str, str] | None) -> list[str]:
    """Label rows from the Planner's display labels when they are binary.

    Falls back to the level value itself: an unlabelled category is still an
    honest category, whereas inventing a clinical name would not be.
    """

    values = list(levels["exposure_level"])
    if level_labels is not None and len(values) == 2:
        return [str(level_labels[0]), str(level_labels[1])]
    return [str(value) for value in values]


def run_exposure_outcome_distribution_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    level_labels: tuple[str, str] | None = None,
) -> Mapping[str, Any]:
    """Render the two-panel distribution figure from its one bound table."""

    if not _is_safe_figure_product_id(figure_product):
        raise ValueError("unsafe or malformed figure product id")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, binding, source_name = _load_binding(
        run_dir=Path(run_dir), resolved_inputs=resolved_inputs, step_id=step_id
    )
    levels, total = _validate(frame)

    full_source = out_dir / f"{figure_product}_input_source_data.csv"
    prevalence_source = out_dir / f"{figure_product}_prevalence_source_data.csv"
    outcome_source = out_dir / f"{figure_product}_outcome_source_data.csv"
    frame.to_csv(full_source, index=False)
    levels[["exposure_level", "n_rows", "exposure_denominator", "exposure_pct"]].to_csv(
        prevalence_source, index=False
    )
    levels[
        [
            "exposure_level",
            "outcome_events",
            "outcome_denominator",
            "outcome_observed_n",
            "outcome_missing_n",
            "outcome_rate_pct",
            "ci_low_pct",
            "ci_high_pct",
        ]
    ].to_csv(outcome_source, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style()
    labels = _labels(levels, level_labels)
    positions = list(range(len(levels)))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    ax_a.barh(
        positions,
        [float(value) for value in levels["exposure_pct"]],
        color=palette["blue"],
        height=0.55,
    )
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Share of the analysed cohort (%)")
    ax_a.set_title("Exposure distribution", loc="left", pad=4)
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, pct, n_rows, denominator in zip(
        positions,
        levels["exposure_pct"],
        levels["n_rows"],
        levels["exposure_denominator"],
    ):
        ax_a.text(
            float(pct) + 1.0,
            position,
            f"{float(pct):.1f}%  {int(n_rows):,}/{int(denominator):,}",
            va="center",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.14, y=1.04)

    rate = levels["outcome_rate_pct"].astype(float)
    low = levels["ci_low_pct"].astype(float)
    high = levels["ci_high_pct"].astype(float)
    ax_b.errorbar(
        rate,
        positions,
        xerr=[rate - low, high - rate],
        fmt="o",
        color=palette["blue"],
        ecolor=palette["neutral"],
        elinewidth=1.0,
        capsize=2.0,
        markersize=4.2,
    )
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(labels)
    ax_b.invert_yaxis()
    upper = min(100.0, max(5.0, float(high.max()) * 1.35))
    ax_b.set_xlim(0, upper)
    ax_b.set_xlabel("Outcome rate (%)")
    ax_b.set_title("Outcome rate by exposure", loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, estimate, events, denominator, missing in zip(
        positions,
        rate,
        levels["outcome_events"],
        levels["outcome_denominator"],
        levels["outcome_missing_n"],
    ):
        suffix = f"  ({int(missing):,} unobserved)" if int(missing) else ""
        ax_b.text(
            min(float(estimate) + upper * 0.025, upper * 0.98),
            position,
            f"{float(estimate):.1f}%  {int(events):,}/{int(denominator):,}{suffix}",
            va="center",
            ha="left" if estimate < upper * 0.86 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_b, "B", x=-0.14, y=1.04)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.84, wspace=0.48)

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The declared exposure levels and their outcome rates are rendered "
            "from one digest-verified, self-contained parent table."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=86.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Exposure distribution",
                "role": "baseline_context",
                "claim": (
                    "The parent's declared exposure levels partition the analysed "
                    "denominator, which each row carries with it."
                ),
                "evidence_ids": [prevalence_source.name],
                "metadata": {
                    "chart_type": "bar_prevalence",
                    "source_data": [prevalence_source.name],
                },
            },
            {
                "panel_id": "B",
                "title": "Outcome rate by exposure",
                "role": "descriptive_result",
                "claim": (
                    "Events, the denominator they are taken over, the unobserved "
                    "count and the interval are shown for every declared level."
                ),
                "evidence_ids": [outcome_source.name],
                "metadata": {
                    "chart_type": "dot_interval_absolute_risk",
                    "source_data": [outcome_source.name],
                },
            },
        ],
        source_data=[full_source.name, prevalence_source.name, outcome_source.name],
        statistics_note=(
            "Percentages and intervals are reproduced from the bound parent table; "
            "the renderer recomputes none of them and introduces no cohort, "
            "exposure, outcome, denominator, or missing-data decision."
        ),
    )
    # The contract is written by the exporter, not here: it decides the export
    # formats from the contract itself, so serialising a second copy alongside
    # would be a second source of truth for what was exported.
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    contract_path = outputs["contract"]
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]

    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "descriptive",
        "rendering_only": True,
        "deterministic_standard_analysis": "exposure_outcome_distribution_figure",
        "interpretation_class": "exposure_outcome_distribution_figure",
        "source_input": EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
        "source_table": source_name,
        "source_sha256": binding.get("sha256"),
        "source_evidence_id": binding.get("evidence_id"),
        "source_rows_consumed": int(len(frame)),
        "cohort_n": int(total["n_rows"]),
        "figure_path": f"{figure_product}.png",
        "figure_contract": contract_path.name,
        "contract_files": [contract_path.name],
        "figure_files": figure_files,
        "source_data_files": [
            full_source.name,
            prevalence_source.name,
            outcome_source.name,
        ],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
        "adjusted_effect": None,
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary
