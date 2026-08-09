"""Deterministic renderer for the closed exposure-outcome distribution product.

This renderer consumes **one** table and nothing else. That is the point of it:
the product it reads carries its own denominators, missing counts, event counts
and intervals, *and* the design that produced them, so there is no second
lookup into a cohort summary to make the percentages meaningful. A renderer
that needed two tables could not have its input contract closed before its
parent ran, which is what left the figure steps unresolvable in a preflight.

It draws what the parent already measured and decides nothing: no cohort, no
exposure, no outcome, no category, no denominator, no interval. What it does do
is **re-derive** every published quantity from the counts beside it, using the
method and confidence level the table itself declares, and refuse to draw when
one disagrees. Recomputing with the producer's own kernel cannot catch a bug in
that kernel -- the producer verifies itself for that -- but it does catch a
table that was edited, truncated or rebuilt between the two steps, which is the
failure this boundary exists to stop.
"""

from __future__ import annotations

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
    EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS,
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT,
    percentage,
    wilson_interval,
)
from .figure_input_capability import TypedInputCapability
from .planner_display_labels import planner_binary_level_labels
from .typed_input_binding import BoundTypedInput, load_typed_input

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY",
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT",
    "exposure_outcome_distribution_figure_code",
    "exposure_outcome_distribution_figure_owns_step",
    "run_exposure_outcome_distribution_figure",
]

EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT = EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT

#: Same rule as the missingness renderer: the figure product id is a
#: Planner-owned label that becomes a filename, never a capability claim.
_FIGURE_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")

EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY = TypedInputCapability(
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
    if not EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY.admits_step(step):
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


def _load_binding(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> BoundTypedInput:
    """Read exactly the one bound table through the shared binding owner.

    There is no separate loader here on purpose. Every check this renderer
    needs -- one manifest for this step, one input and no other, a capsule that
    agrees with its own identity record, a contained path, a digest verified
    before and after the read, and the exact product schema -- is the same
    question every other typed consumer asks, and a second implementation of it
    would only guarantee that the two drift.
    """

    return load_typed_input(
        input_key=EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
        exclusive=True,
        require_consumption_contract=True,
        minimum_row_count=3,
    )


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _close(left: Any, right: Any) -> bool:
    """Whether two published quantities agree, both possibly absent."""

    first, second = _finite(left), _finite(right)
    if first is None or second is None:
        return first is None and second is None
    return abs(first - second) <= 1e-6


def _declared_design(frame: pd.DataFrame) -> dict[str, Any]:
    """The one design every row must agree on, or refuse.

    The design columns are constant by construction, so a table whose rows
    disagree about which outcome value was the event, or at what confidence its
    intervals were built, is not one table -- and picking either answer would
    be this renderer deciding something it does not own.
    """

    design: dict[str, Any] = {}
    for column in EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS:
        values = frame[column].astype("object")
        distinct = {repr(value) for value in values}
        if len(distinct) != 1:
            raise ValueError(
                f"distribution table rows disagree on {column!r}; the design "
                "that produced the numbers must be one declaration"
            )
        design[column] = values.iloc[0]
    return design


def _validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Re-derive every published quantity before drawing it.

    Not a plausibility check: each percentage is recomputed from the counts on
    its own row, each interval is rebuilt by the method and confidence level
    the table declares, and the strata are required to sum to the totals. A
    rate that merely lands inside its interval is not evidence that it is the
    right rate.
    """

    design = _declared_design(frame)
    interval_method = str(design["interval_method"])
    if interval_method != "wilson":
        raise ValueError(
            f"distribution table declares interval_method={interval_method!r}, "
            "which this renderer cannot re-derive"
        )
    confidence_level = _finite(design["confidence_level"])
    if confidence_level is None or not (0.5 < confidence_level < 1.0):
        raise ValueError("distribution table declares an unusable confidence level")
    denominator_policy = str(design["denominator_policy"])
    if denominator_policy not in {"all_declared_rows", "observed_outcome_rows"}:
        raise ValueError("distribution table declares an unknown denominator policy")

    # Selecting the two known roles would silently drop a third: a row nobody
    # recognises would then be excluded from every sum below and still be drawn.
    unknown_roles = set(frame["row_role"].astype(str)) - {_LEVEL_ROLE, _OVERALL_ROLE}
    if unknown_roles:
        raise ValueError(
            f"distribution table carries unknown row roles: {sorted(unknown_roles)}"
        )
    levels = frame[frame["row_role"] == _LEVEL_ROLE]
    overall = frame[frame["row_role"] == _OVERALL_ROLE]
    if len(overall) != 1:
        raise ValueError("distribution table needs exactly one overall row")
    if len(levels) < 2:
        raise ValueError("distribution table needs at least two exposure levels")
    if levels["exposure_level"].astype(str).duplicated().any():
        raise ValueError("an exposure level appears more than once")
    try:
        declared_levels = json.loads(str(design["exposure_levels_declared"]))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "distribution table declares unreadable exposure levels"
        ) from exc
    if not isinstance(declared_levels, list) or len(declared_levels) != len(levels):
        raise ValueError(
            "distribution table reports a different number of exposure levels "
            "from the number its own declaration closes over"
        )
    total = overall.iloc[0]
    if int(levels["n_rows"].sum()) != int(total["n_rows"]):
        raise ValueError("exposure levels do not partition the reported cohort")
    for column in ("outcome_events", "outcome_observed_n", "outcome_missing_n"):
        if int(levels[column].sum()) != int(total[column]):
            raise ValueError(f"level {column} does not sum to the overall {column}")

    for _, row in frame.iterrows():
        if int(row["outcome_observed_n"]) + int(row["outcome_missing_n"]) != int(
            row["n_rows"]
        ):
            raise ValueError("observed plus missing does not equal the row count")
        if int(row["outcome_events"]) > int(row["outcome_denominator"]):
            raise ValueError("more events than the denominator they are taken over")
        if int(row["exposure_denominator"]) != int(total["n_rows"]):
            raise ValueError(
                "an exposure denominator is not the cohort the table reports"
            )
        expected_denominator = (
            int(row["n_rows"])
            if denominator_policy == "all_declared_rows"
            else int(row["outcome_observed_n"])
        )
        if int(row["outcome_denominator"]) != expected_denominator:
            raise ValueError(
                "an outcome denominator does not follow the declared "
                f"denominator_policy={denominator_policy!r}"
            )
        if not _close(
            row["exposure_pct"],
            percentage(int(row["n_rows"]), int(row["exposure_denominator"])),
        ):
            raise ValueError("an exposure percentage is not its own counts")
        if not _close(
            row["outcome_rate_pct"],
            percentage(int(row["outcome_events"]), int(row["outcome_denominator"])),
        ):
            raise ValueError("an outcome rate is not its own events over denominator")
        expected_low, expected_high = wilson_interval(
            int(row["outcome_events"]),
            int(row["outcome_denominator"]),
            confidence_level=confidence_level,
        )
        if not _close(row["ci_low_pct"], expected_low) or not _close(
            row["ci_high_pct"], expected_high
        ):
            raise ValueError(
                "a reported interval is not the declared method at the declared "
                "confidence level"
            )
        # Deliberately NOT checked here: that the rate and interval are finite,
        # that ci_low <= ci_high, and that both lie in 0-100. Each was written,
        # probed, and removed as unreachable -- the exact re-derivations above
        # already pin all three quantities to values recomputed from the counts,
        # so a non-finite or out-of-range endpoint fails the equality check
        # several lines earlier with a message that names the real disagreement.
        # A range check downstream of an equality check cannot fire; adding one
        # back would only look like more safety. If the equality checks are ever
        # relaxed, these become live again and must return with them.
        rate = _finite(row["outcome_rate_pct"])
        low = _finite(row["ci_low_pct"])
        high = _finite(row["ci_high_pct"])
        if rate is not None and low is not None and high is not None:
            if not (low - 1e-6 <= rate <= high + 1e-6):
                raise ValueError("a reported rate falls outside its own interval")
    return levels, total, design


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
    bound = _load_binding(
        run_dir=Path(run_dir), resolved_inputs=resolved_inputs, step_id=step_id
    )
    frame, binding, source_name = bound.frame, bound.binding, bound.path.name
    levels, total, design = _validate(frame)

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
            "Percentages and intervals are reproduced from the bound parent "
            "table, which declares them: outcome rates are taken over "
            f"{design['denominator_policy']} with missing outcomes handled as "
            f"{design['missing_outcome_policy']}, and intervals are "
            f"{design['interval_method']} at "
            f"{float(design['confidence_level']):.3g} coverage. The renderer "
            "re-derives each published quantity from the counts beside it and "
            "introduces no cohort, exposure, outcome, denominator, or "
            "missing-data decision of its own."
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
        # Echoed from the bound table, not re-decided: a reader of the summary
        # can see which design the drawing was made under without opening the
        # plan, and a mismatch against the parent is detectable.
        "declared_design": {
            key: (float(value) if key == "confidence_level" else str(value))
            for key, value in design.items()
        },
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
