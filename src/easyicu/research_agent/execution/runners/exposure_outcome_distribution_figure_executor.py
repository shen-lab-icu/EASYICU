"""Deterministic renderer for a role-tagged exposure/outcome joint distribution.

The producing step already owns the cohort, the exposure, the outcome, the
missing-data policy and every count.  It publishes those counts as one table
whose ``row_type`` column carries an explicit **row role**: the cells of the
joint exposure-by-outcome grid are one role, and the marginal missingness
accounting is another.

This executor renders that table and nothing else.  It selects rows by their
declared role rather than by "which cells happen to have a non-empty label",
which is precisely the confusion that made a generated script treat two
zero-count missingness rows as plotted joint cells.  It reads no cohort, fits
no model, chooses no exposure/outcome, and never decides which outcome level is
"the event" — every level of the declared grid is rendered.

Ownership is decided from the Planner contract **and** from the host-emitted
product contract of the bound parent tables.  When either is not the exact
shape below the step is simply not claimed and the ordinary coder path runs, so
a widened or unfamiliar schema can never be mis-rendered as if it were this one.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .planner_display_labels import planner_binary_level_labels

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS",
    "exposure_outcome_distribution_figure_executor_code",
    "exposure_outcome_distribution_figure_executor_owns_step",
    "run_exposure_outcome_distribution_figure",
]


_COHORT_SUMMARY_INPUT = "table:cohort_summary"
_DISTRIBUTION_INPUT = "table:exposure_outcome_distribution"
EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS = (
    _COHORT_SUMMARY_INPUT,
    _DISTRIBUTION_INPUT,
)

# The exact host-recorded product contracts this renderer can read.  Both are
# compared as ordered column lists against the binding the host itself emitted,
# never against prose, a filename or a benchmark identity.
_DISTRIBUTION_COLUMNS = (
    "row_type",
    "exposure_variable",
    "exposure_category",
    "outcome_variable",
    "outcome_category",
    "count",
    "percentage_of_locked_cohort",
    "denominator_n",
)
_COHORT_SUMMARY_COLUMNS = ("variable", "metric", "value", "denominator_n")

# The row-role vocabulary.  ``joint`` rows are the rendered grid; every other
# known role is accounting that must never enter a plotted cell.
_JOINT_ROLE = "joint_distribution"
_ACCOUNTING_ROLES = frozenset({"missingness"})
_KNOWN_ROLES = frozenset({_JOINT_ROLE}) | _ACCOUNTING_ROLES

_PERCENTAGE_TOLERANCE = 1e-6


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


def _bound_columns(
    resolved_bindings: Mapping[str, Any] | None,
    input_key: str,
    product: str,
) -> tuple[str, ...] | None:
    """Return the host-recorded column list for one digest-bound table."""

    if not isinstance(resolved_bindings, Mapping):
        return None
    binding = resolved_bindings.get(input_key)
    if not isinstance(binding, Mapping):
        return None
    if (
        binding.get("declared_kind") != "table"
        or binding.get("evidence_kind") != "table"
        or binding.get("product") != product
    ):
        return None
    contract = binding.get("product_contract")
    if not isinstance(contract, Mapping):
        return None
    columns = contract.get("columns")
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        return None
    return tuple(str(column) for column in columns)


def exposure_outcome_distribution_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
    display_labels: Mapping[str, str] | None = None,
) -> bool:
    """Own only the exact typed contract whose bound schema is also known.

    Without ``resolved_bindings`` the step is never claimed: the Planner names a
    product, but only the host-emitted product contract proves the bytes have
    the schema this renderer can read.
    """

    contracts = list(step.input_consumption_contracts)
    products = [_figure_product(value) for value in step.expected_outputs]
    plan_contract_closed = bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and tuple(step.inputs) == EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS
        and len(products) == 1
        and products[0] is not None
        and len(contracts) == len(EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS)
        and tuple(contract.input_key for contract in contracts)
        == EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS
        and all(
            contract.mode == "all_rows"
            and contract.role_column is None
            and not contract.expected_roles
            for contract in contracts
        )
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )
    if not plan_contract_closed:
        return False
    if planner_binary_level_labels(display_labels) is None:
        # Without one unambiguous Planner label pair this renderer would have to
        # invent a placeholder level name.  Leave the step unclaimed instead.
        return False
    return (
        _bound_columns(
            resolved_bindings,
            _DISTRIBUTION_INPUT,
            "exposure_outcome_distribution",
        )
        == _DISTRIBUTION_COLUMNS
        and _bound_columns(resolved_bindings, _COHORT_SUMMARY_INPUT, "cohort_summary")
        == _COHORT_SUMMARY_COLUMNS
    )


def exposure_outcome_distribution_figure_executor_code(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
    display_labels: Mapping[str, str] | None = None,
) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    if not exposure_outcome_distribution_figure_executor_owns_step(
        step,
        resolved_bindings=resolved_bindings,
        display_labels=display_labels,
    ):
        raise ValueError(
            "The step is not owned by the exposure/outcome distribution renderer"
        )
    product = _figure_product(step.expected_outputs[0])
    labels = dict(sorted((display_labels or {}).items()))
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.exposure_outcome_distribution_figure_executor import (
            run_exposure_outcome_distribution_figure,
        )

        run_exposure_outcome_distribution_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            display_labels={labels!r},
        )
        """
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_table(
    *,
    run_dir: Path,
    inputs: Mapping[str, Any],
    input_key: str,
    product: str,
    expected_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Read one digest-verified parent table under a closed authority binding."""

    binding = inputs.get(input_key)
    if not isinstance(binding, dict):
        raise ValueError(f"exact {input_key} binding is absent")
    digest = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    product_contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    identity = binding.get("identity_row")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", digest)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(product_contract, dict)
        or not isinstance(consumption, dict)
        or not isinstance(identity, dict)
        or binding.get("declared_kind") != "table"
        or binding.get("evidence_kind") != "table"
        or binding.get("product") != product
        or identity.get("input_key") != input_key
        or identity.get("product") != product
        or identity.get("sha256") != digest
        or consumption.get("input_key") != input_key
        or consumption.get("mode") != "all_rows"
        or consumption.get("artifact_sha256") != digest
    ):
        raise ValueError(f"{input_key} authority binding is incomplete")

    columns = product_contract.get("columns")
    row_count = product_contract.get("row_count")
    if (
        columns != list(expected_columns)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or consumption.get("verified_row_count") != row_count
    ):
        raise ValueError(f"{input_key} product contract is unsupported")

    resolved_run_dir = Path(run_dir).resolve()
    path = (resolved_run_dir / relative_path).resolve()
    try:
        path.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise ValueError(f"{input_key} binding escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"{input_key} must be a regular bound CSV")
    if _sha256(path) != digest:
        raise ValueError(f"{input_key} digest verification failed")
    frame = pd.read_csv(path)
    if list(frame.columns) != list(expected_columns) or len(frame) != row_count:
        raise ValueError(f"{input_key} bytes disagree with its product contract")
    if _sha256(path) != digest:
        raise ValueError(f"{input_key} changed while it was being read")
    return frame, binding


def _nonnegative_integers(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column]
    converted = pd.to_numeric(source, errors="coerce")
    if bool((source.notna() & converted.isna()).any()) or not bool(
        converted.notna().all()
    ):
        raise ValueError(f"{column} must contain non-negative whole counts")
    numeric = converted.astype(float)
    if (
        not bool(numeric.map(math.isfinite).all())
        or not bool(numeric.map(float.is_integer).all())
        or bool((numeric < 0).any())
    ):
        raise ValueError(f"{column} must contain non-negative whole counts")
    return numeric.astype(int)


def _labelled(values: pd.Series, column: str) -> pd.Series:
    labels = values.fillna("").astype(str).str.strip()
    if bool(labels.eq("").any()):
        raise ValueError(f"{column} must be labelled on every row")
    return labels


def _validated_distribution(
    distribution: pd.DataFrame,
    cohort_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, str, str]:
    """Partition the parent table by row role and reconcile every count.

    Returns the joint grid, the accounting rows, the locked denominator and the
    declared exposure/outcome variable names.
    """

    roles = _labelled(distribution["row_type"], "row_type")
    unknown = sorted(set(roles) - _KNOWN_ROLES)
    if unknown:
        raise ValueError(f"unknown distribution row roles: {unknown}")

    denominators = _nonnegative_integers(distribution, "denominator_n")
    if denominators.nunique() != 1:
        raise ValueError("the distribution table must share one locked denominator")
    locked_n = int(denominators.iloc[0])
    if locked_n <= 0:
        raise ValueError("the locked denominator must be positive")

    counts = _nonnegative_integers(distribution, "count")
    percentages = pd.to_numeric(
        distribution["percentage_of_locked_cohort"],
        errors="coerce",
    )
    if not bool(percentages.notna().all()) or not np.allclose(
        percentages.astype(float),
        100.0 * counts / locked_n,
        rtol=0.0,
        atol=_PERCENTAGE_TOLERANCE,
    ):
        raise ValueError("percentages do not reconcile to counts over the denominator")

    joint = distribution.loc[roles.eq(_JOINT_ROLE)].copy()
    accounting = distribution.loc[~roles.eq(_JOINT_ROLE)].copy()
    if joint.empty:
        raise ValueError("the distribution table declares no joint grid rows")

    accounting_counts = counts.loc[accounting.index]
    if bool((accounting_counts > 0).any()):
        # A positive missingness row means the joint grid does not cover the
        # locked cohort, so a prevalence panel drawn from it would misstate its
        # own denominator.  Refuse rather than silently drop the remainder.
        raise ValueError(
            "non-joint accounting rows carry a positive count; the joint grid "
            "does not cover the locked denominator"
        )

    exposure_variable = _labelled(joint["exposure_variable"], "exposure_variable")
    outcome_variable = _labelled(joint["outcome_variable"], "outcome_variable")
    if exposure_variable.nunique() != 1 or outcome_variable.nunique() != 1:
        raise ValueError("the joint grid must describe one exposure and one outcome")

    exposure_category = _labelled(joint["exposure_category"], "exposure_category")
    outcome_category = _labelled(joint["outcome_category"], "outcome_category")
    exposure_levels = list(dict.fromkeys(exposure_category))
    outcome_levels = list(dict.fromkeys(outcome_category))
    if len(exposure_levels) != 2 or len(outcome_levels) < 2:
        raise ValueError(
            "the joint grid must contain a binary exposure and at least two "
            "outcome levels"
        )
    if len(joint) != len(exposure_levels) * len(outcome_levels) or (
        len(set(zip(exposure_category, outcome_category))) != len(joint)
    ):
        raise ValueError("the joint grid is incomplete or contains duplicate cells")
    if int(counts.loc[joint.index].sum()) != locked_n:
        raise ValueError("joint grid counts do not sum to the locked denominator")

    summary_denominators = _nonnegative_integers(cohort_summary, "denominator_n")
    if locked_n not in set(summary_denominators.astype(int)):
        raise ValueError(
            "the cohort summary does not record the distribution's locked "
            "denominator"
        )

    joint = joint.assign(
        __count=counts.loc[joint.index].to_numpy(),
        __exposure=exposure_category.to_numpy(),
        __outcome=outcome_category.to_numpy(),
    )
    return (
        joint,
        accounting,
        locked_n,
        str(exposure_variable.iloc[0]),
        str(outcome_variable.iloc[0]),
    )


def _binary_level_order(joint: pd.DataFrame) -> list[str]:
    """Order the two exposure categories as declared levels 0 then 1."""

    levels = list(dict.fromkeys(joint["__exposure"]))
    numeric = pd.to_numeric(pd.Series(levels), errors="coerce")
    if (
        not bool(numeric.notna().all())
        or not bool(numeric.map(float.is_integer).all())
        or set(numeric.astype(int)) != {0, 1}
    ):
        raise ValueError(
            "Planner binary labels describe levels 0 and 1, but the joint grid "
            "does not declare that exposure coding"
        )
    ordered = dict(zip(numeric.astype(int), levels))
    return [ordered[0], ordered[1]]


def _reader_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Outcome"


def _outcome_title(
    display_labels: Mapping[str, str] | None,
    outcome_variable: str,
) -> str:
    """Prefer the Planner's own name for the outcome variable, verbatim."""

    label = " ".join(
        str((display_labels or {}).get(outcome_variable, "") or "").split()
    )
    return label or _reader_label(outcome_variable)


def _upstream_table_name(binding: Mapping[str, Any]) -> str:
    """Return the producing step's plain output filename for one binding.

    Evidence files are stored as ``<evidence id>__<logical name>``; the logical
    half is the file the producing step actually wrote, and it is what a
    source-data claim must name.  A binding without that form contributes its
    own basename rather than a path.
    """

    name = Path(str(binding.get("relative_path") or "")).name
    return name.split("__", 1)[1] if "__" in name else name


def _write_source_projection(
    rows: pd.DataFrame,
    *,
    path: Path,
    source_table: str,
) -> None:
    """Write upstream rows verbatim, keyed by their real source row index."""

    export = rows.drop(
        columns=[column for column in rows.columns if str(column).startswith("__")],
        errors="ignore",
    ).copy()
    export.insert(0, "source_row_index", export.index.astype(int))
    export.insert(1, "source_table", source_table)
    export.to_csv(path, index=False)


def run_exposure_outcome_distribution_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    display_labels: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Render the role-partitioned joint distribution and its source data."""

    if _figure_product(f"figure:{figure_product}") is None:
        raise ValueError("unsupported joint-distribution figure product")
    resolved_labels = planner_binary_level_labels(display_labels)
    if resolved_labels is None:
        raise ValueError("the Planner declared no unambiguous binary label pair")
    label_column, label_absent, label_present = resolved_labels

    payload = (
        dict(resolved_inputs)
        if isinstance(resolved_inputs, Mapping)
        else json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(
        EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS
    ):
        raise ValueError("exact distribution input pair is absent or widened")

    cohort_summary, cohort_binding = _load_bound_table(
        run_dir=Path(run_dir),
        inputs=inputs,
        input_key=_COHORT_SUMMARY_INPUT,
        product="cohort_summary",
        expected_columns=_COHORT_SUMMARY_COLUMNS,
    )
    distribution, distribution_binding = _load_bound_table(
        run_dir=Path(run_dir),
        inputs=inputs,
        input_key=_DISTRIBUTION_INPUT,
        product="exposure_outcome_distribution",
        expected_columns=_DISTRIBUTION_COLUMNS,
    )
    joint, accounting, locked_n, exposure_variable, outcome_variable = (
        _validated_distribution(distribution, cohort_summary)
    )
    if exposure_variable != label_column:
        raise ValueError(
            "the Planner binary labels describe a different column than the "
            "joint grid's exposure variable"
        )

    exposure_order = _binary_level_order(joint)
    category_labels = [label_absent, label_present]
    outcome_order = list(dict.fromkeys(joint["__outcome"]))
    cell_counts = {
        (str(row["__exposure"]), str(row["__outcome"])): int(row["__count"])
        for _, row in joint.iterrows()
    }
    exposure_totals = [
        sum(cell_counts[(level, outcome)] for outcome in outcome_order)
        for level in exposure_order
    ]
    if sum(exposure_totals) != locked_n:
        raise ValueError("exposure marginals do not reconstruct the locked denominator")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_source = out_dir / f"{figure_product}_distribution_source_data.csv"
    cohort_source = out_dir / f"{figure_product}_cohort_summary_source_data.csv"
    _write_source_projection(
        distribution,
        path=distribution_source,
        source_table=_upstream_table_name(distribution_binding),
    )
    _write_source_projection(
        cohort_summary,
        path=cohort_source,
        source_table=_upstream_table_name(cohort_binding),
    )

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
    positions = np.arange(len(exposure_order))
    prevalence = [100.0 * total / locked_n for total in exposure_totals]
    bars = ax_a.barh(
        positions,
        prevalence,
        color=[palette["neutral_light"], palette["blue_soft"]],
        height=0.58,
    )
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(category_labels)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Analysis cohort (%)")
    ax_a.set_title("Exposure prevalence", loc="left", pad=4)
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for bar, percentage, total in zip(bars, prevalence, exposure_totals):
        ax_a.text(
            min(float(percentage) + 1.0, 97.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.1f}%  n={int(total):,}",
            va="center",
            ha="left" if percentage < 94 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.14, y=1.04)

    group_height = 0.72 / len(outcome_order)
    shades = [palette["blue"], palette["neutral"], palette["blue_soft"]]
    within_shares = [
        [
            100.0 * cell_counts[(level, outcome)] / total if total else 0.0
            for level, total in zip(exposure_order, exposure_totals)
        ]
        for outcome in outcome_order
    ]
    upper = min(100.0, max(12.0, max(max(row) for row in within_shares) * 1.18))
    ax_b.set_xlim(0, upper)
    for index, outcome in enumerate(outcome_order):
        offsets = positions + (index - (len(outcome_order) - 1) / 2.0) * group_height
        within = within_shares[index]
        ax_b.barh(
            offsets,
            within,
            color=shades[index % len(shades)],
            height=group_height * 0.9,
            label=str(outcome),
        )
        for offset, percentage, level, total in zip(
            offsets,
            within,
            exposure_order,
            exposure_totals,
        ):
            # Long bars would push their label past the axis, so annotate
            # inside the bar instead of clipping the counts a reader needs.
            outside = float(percentage) < upper * 0.58
            ax_b.text(
                (
                    float(percentage) + upper * 0.02
                    if outside
                    else max(float(percentage) - upper * 0.02, upper * 0.02)
                ),
                offset,
                f"{float(percentage):.1f}%  "
                f"{cell_counts[(level, outcome)]:,}/{int(total):,}",
                va="center",
                ha="left" if outside else "right",
                color="black" if outside else "white",
                fontsize=5.9,
            )
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(category_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Within-category share (%)")
    ax_b.set_title(
        _outcome_title(display_labels, outcome_variable),
        loc="left",
        pad=4,
    )
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    ax_b.legend(loc="lower right", frameon=False, fontsize=5.9)
    add_panel_label(ax_b, "B", x=-0.14, y=1.04)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.84, wspace=0.48)

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The declared exposure categories partition the locked analysis "
            "cohort, and every outcome level of the parent joint grid is shown "
            "within each category."
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
                    "Both parent-declared exposure categories partition the "
                    "locked denominator without dropping rows."
                ),
                "evidence_ids": [str(distribution_binding.get("evidence_id") or "")],
                "metadata": {
                    "chart_type": "bar_prevalence",
                    "source_data": [distribution_source.name],
                    "locked_denominator": locked_n,
                },
            },
            {
                "panel_id": "B",
                "title": "Outcome composition within exposure category",
                "role": "descriptive_result",
                "claim": (
                    "Each outcome level is shown as its share of the parent "
                    "exposure category, with the exact cell and category counts."
                ),
                "evidence_ids": [str(distribution_binding.get("evidence_id") or "")],
                "metadata": {
                    "chart_type": "grouped_bar_composition",
                    "source_data": [distribution_source.name],
                    "outcome_levels": [str(level) for level in outcome_order],
                },
            },
        ],
        source_data=[distribution_source.name, cohort_source.name],
        statistics_note=(
            "Rows are selected by the role their parent table declares: only "
            "cells of the joint exposure-by-outcome grid are plotted, and the "
            "marginal accounting rows are carried into the source data "
            "unplotted. Percentages in panel A are 100 x category count / "
            "locked denominator; panel B shows 100 x cell count / category "
            "count. The executor validates the role vocabulary, the "
            "completeness of the grid, the shared denominator and every "
            "count and percentage identity. It introduces no cohort, "
            "exposure, outcome, event, missing-data or modeling decision."
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
        "method": "deterministic_exposure_outcome_distribution_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "exposure_outcome_distribution_figure",
        "rendering_only": True,
        "source_inputs": list(EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS),
        "source_evidence_ids": [
            cohort_binding.get("evidence_id"),
            distribution_binding.get("evidence_id"),
        ],
        "source_sha256": [
            cohort_binding.get("sha256"),
            distribution_binding.get("sha256"),
        ],
        "source_rows_consumed": int(len(cohort_summary) + len(distribution)),
        "joint_cell_rows": int(len(joint)),
        "accounting_rows_excluded": int(len(accounting)),
        "row_role_column": "row_type",
        "plotted_row_role": _JOINT_ROLE,
        "exposure_variable": exposure_variable,
        "outcome_variable": outcome_variable,
        "category_labels": list(category_labels),
        "outcome_levels": [str(level) for level in outcome_order],
        "exposure_category_counts": [int(total) for total in exposure_totals],
        "locked_denominator": locked_n,
        "source_data_files": [distribution_source.name, cohort_source.name],
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
