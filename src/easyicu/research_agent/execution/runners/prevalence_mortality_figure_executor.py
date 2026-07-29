"""Deterministic renderer for a sealed prevalence/mortality table pair.

The Planner and the producing analysis step already own the cohort,
binary-exposure definition, mortality outcome, denominators, estimates, and
confidence intervals.  This executor only validates the two digest-bound table
products and renders them.  It never reads the cohort or chooses a scientific
coordinate.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from .planner_display_labels import planner_binary_level_labels
from .figure_input_capability import TypedInputCapability

__all__ = [
    "PREVALENCE_MORTALITY_FIGURE_INPUTS",
    "binary_level_labels",
    "prevalence_mortality_figure_executor_code",
    "prevalence_mortality_figure_executor_owns_step",
    "run_prevalence_mortality_figure",
]


PREVALENCE_MORTALITY_FIGURE_INPUTS = (
    "table:cohort_summary",
    "table:outcome_incidence",
)
_FIGURE_PRODUCT = "prevalence_mortality"
_COHORT_COLUMNS = (
    "summary",
    "exposure_level",
    "numerator",
    "denominator",
    "percentage",
    "ci_low",
    "ci_high",
)
_OUTCOME_COLUMNS = (
    "exposure_level",
    "exposure_count",
    "outcome_observed_n",
    "deaths",
    "mortality_pct",
    "ci_low_pct",
    "ci_high_pct",
)


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


#: Both panels are indexed by key while rendering, so neither is optional.
PREVALENCE_MORTALITY_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset(PREVALENCE_MORTALITY_FIGURE_INPUTS),
)


def prevalence_mortality_figure_executor_owns_step(step: AnalysisStep) -> bool:
    """Return whether the exact two-table rendering contract is closed."""

    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and PREVALENCE_MORTALITY_FIGURE_CAPABILITY.admits_step(step)
        and list(step.expected_outputs) == [f"figure:{_FIGURE_PRODUCT}"]
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


def binary_level_labels(
    display_labels: Mapping[str, str] | None,
) -> tuple[str, str]:
    """Resolve one unambiguous Planner-owned ``column=0/1`` label pair."""

    resolved = planner_binary_level_labels(display_labels)
    if resolved is None:
        return ("Level 0", "Level 1")
    return (resolved[1], resolved[2])


def prevalence_mortality_figure_executor_code(
    step: AnalysisStep,
    *,
    display_labels: Mapping[str, str] | None = None,
) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    if not prevalence_mortality_figure_executor_owns_step(step):
        raise ValueError("The step is not owned by the prevalence/mortality renderer")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.prevalence_mortality_figure_executor import (
            run_prevalence_mortality_figure,
        )

        run_prevalence_mortality_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            category_labels={binary_level_labels(display_labels)!r},
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


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column]
    converted = pd.to_numeric(source, errors="coerce")
    invalid = source.notna() & converted.isna()
    finite = converted.dropna().map(math.isfinite)
    if bool(invalid.any()) or not bool(finite.all()):
        raise ValueError(f"{column} contains a non-numeric or non-finite value")
    return converted.astype(float)


def _integer(value: Any, *, name: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        raise ValueError(f"{name} is not a non-negative integer")
    return int(parsed)


def _validated_tables(
    cohort: pd.DataFrame,
    outcome: pd.DataFrame,
) -> tuple[pd.DataFrame, int, float, float, float]:
    labels = cohort["summary"].fillna("").astype(str).str.strip()
    if labels.eq("").any() or labels.duplicated().any():
        raise ValueError("cohort summary labels must be unique and non-empty")
    numerators = _numeric(cohort, "numerator")
    denominators = _numeric(cohort, "denominator")
    percentages = _numeric(cohort, "percentage")
    if (
        not numerators.map(float.is_integer).all()
        or not denominators.map(float.is_integer).all()
        or (numerators < 0).any()
        or (denominators <= 0).any()
        or (numerators > denominators).any()
        or denominators.nunique() != 1
        or not np.allclose(
            percentages,
            100.0 * numerators / denominators,
            rtol=0.0,
            atol=1e-8,
        )
    ):
        raise ValueError("cohort summary counts and percentages do not reconcile")
    locked_n = int(denominators.iloc[0])
    exposure_levels = _numeric(cohort, "exposure_level")
    prevalence_rows = cohort.loc[exposure_levels.notna()].copy()
    if len(prevalence_rows) != 1:
        raise ValueError("cohort summary must contain one prevalence result row")
    prevalence_index = prevalence_rows.index[0]
    level = exposure_levels.loc[prevalence_index]
    if level != 1.0:
        raise ValueError("prevalence result must identify the binary positive level")
    prevalence_n = _integer(
        numerators.loc[prevalence_index],
        name="prevalence numerator",
    )
    prevalence_pct = float(percentages.loc[prevalence_index])
    ci_low = _numeric(prevalence_rows, "ci_low").iloc[0]
    ci_high = _numeric(prevalence_rows, "ci_high").iloc[0]
    if not 0.0 <= ci_low <= prevalence_pct <= ci_high <= 100.0:
        raise ValueError("prevalence confidence interval is inconsistent")

    outcome_labels = outcome["exposure_level"].fillna("").astype(str).str.strip()
    overall_mask = outcome_labels.str.casefold().eq("overall")
    if int(overall_mask.sum()) != 1 or len(outcome) != 3:
        raise ValueError("outcome table must contain overall plus binary groups")
    grouped = outcome.loc[~overall_mask].copy()
    grouped_levels = pd.to_numeric(
        grouped["exposure_level"],
        errors="coerce",
    )
    if (
        grouped_levels.isna().any()
        or not grouped_levels.map(float.is_integer).all()
        or set(grouped_levels.astype(int)) != {0, 1}
    ):
        raise ValueError("outcome groups must be the complete binary partition")
    grouped["__category"] = grouped_levels.astype(int)
    grouped = grouped.sort_values("__category").reset_index(drop=True)

    exposure_count = _numeric(outcome, "exposure_count")
    observed_n = _numeric(outcome, "outcome_observed_n")
    deaths = _numeric(outcome, "deaths")
    mortality = _numeric(outcome, "mortality_pct")
    ci_low_all = _numeric(outcome, "ci_low_pct")
    ci_high_all = _numeric(outcome, "ci_high_pct")
    if (
        not exposure_count.map(float.is_integer).all()
        or not observed_n.map(float.is_integer).all()
        or not deaths.map(float.is_integer).all()
        or (exposure_count < 0).any()
        or (observed_n < 0).any()
        or (deaths < 0).any()
        or (observed_n > exposure_count).any()
        or (deaths > observed_n).any()
        or not np.allclose(
            mortality,
            100.0 * deaths / observed_n,
            rtol=0.0,
            atol=1e-8,
        )
        or not (
            (0.0 <= ci_low_all)
            & (ci_low_all <= mortality)
            & (mortality <= ci_high_all)
            & (ci_high_all <= 100.0)
        ).all()
    ):
        raise ValueError("outcome counts, risks, or intervals do not reconcile")

    overall_index = outcome.index[overall_mask][0]
    group_indices = outcome.index[~overall_mask]
    group_counts = exposure_count.loc[group_indices].astype(int)
    group_observed = observed_n.loc[group_indices].astype(int)
    group_deaths = deaths.loc[group_indices].astype(int)
    category_to_count = dict(
        zip(
            grouped["__category"].astype(int),
            pd.to_numeric(grouped["exposure_count"]).astype(int),
        )
    )
    if (
        int(exposure_count.loc[overall_index]) != locked_n
        or int(group_counts.sum()) != locked_n
        or int(group_observed.sum()) != int(observed_n.loc[overall_index])
        or int(group_deaths.sum()) != int(deaths.loc[overall_index])
        or category_to_count[1] != prevalence_n
        or category_to_count[0] != locked_n - prevalence_n
    ):
        raise ValueError("prevalence and mortality partitions do not share authority")
    return grouped, locked_n, prevalence_pct, float(ci_low), float(ci_high)


def run_prevalence_mortality_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    category_labels: tuple[str, str] = ("Level 0", "Level 1"),
) -> Mapping[str, Any]:
    """Render the verified table pair and write a source-backed figure bundle."""

    if (
        len(category_labels) != 2
        or any(not str(label or "").strip() for label in category_labels)
        or str(category_labels[0]).strip() == str(category_labels[1]).strip()
    ):
        raise ValueError("category_labels must contain two distinct non-empty labels")
    category_labels = tuple(str(label).strip() for label in category_labels)

    payload = (
        dict(resolved_inputs)
        if isinstance(resolved_inputs, Mapping)
        else json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(
        PREVALENCE_MORTALITY_FIGURE_INPUTS
    ):
        raise ValueError("exact prevalence/mortality input pair is absent or widened")
    cohort, cohort_binding = _load_bound_table(
        run_dir=Path(run_dir),
        inputs=inputs,
        input_key=PREVALENCE_MORTALITY_FIGURE_INPUTS[0],
        product="cohort_summary",
        expected_columns=_COHORT_COLUMNS,
    )
    outcome, outcome_binding = _load_bound_table(
        run_dir=Path(run_dir),
        inputs=inputs,
        input_key=PREVALENCE_MORTALITY_FIGURE_INPUTS[1],
        product="outcome_incidence",
        expected_columns=_OUTCOME_COLUMNS,
    )
    grouped, locked_n, prevalence_pct, prevalence_low, prevalence_high = (
        _validated_tables(cohort, outcome)
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_source = out_dir / f"{_FIGURE_PRODUCT}_cohort_summary_source_data.csv"
    outcome_source = out_dir / f"{_FIGURE_PRODUCT}_outcome_incidence_source_data.csv"
    cohort_source_frame = cohort.copy()
    cohort_source_frame["display_label"] = cohort_source_frame["exposure_level"].map(
        {0.0: category_labels[0], 1.0: category_labels[1]}
    )
    outcome_source_frame = outcome.copy()
    outcome_source_frame["display_label"] = (
        outcome_source_frame["exposure_level"]
        .astype(str)
        .map(
            {
                "0": category_labels[0],
                "0.0": category_labels[0],
                "1": category_labels[1],
                "1.0": category_labels[1],
                "overall": "Overall",
            }
        )
    )
    cohort_source_frame.to_csv(cohort_source, index=False)
    outcome_source_frame.to_csv(outcome_source, index=False)

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
    prevalence_counts = pd.to_numeric(grouped["exposure_count"]).astype(int).to_numpy()
    prevalence_values = 100.0 * prevalence_counts / locked_n
    positions = np.arange(2)
    bars = ax_a.barh(
        positions,
        prevalence_values,
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
    for bar, percentage, count in zip(
        bars,
        prevalence_values,
        prevalence_counts,
    ):
        ax_a.text(
            min(float(percentage) + 1.0, 97.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.1f}%  n={int(count):,}",
            va="center",
            ha="left" if percentage < 94 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.14, y=1.04)

    mortality = pd.to_numeric(grouped["mortality_pct"]).to_numpy(dtype=float)
    mortality_low = pd.to_numeric(grouped["ci_low_pct"]).to_numpy(dtype=float)
    mortality_high = pd.to_numeric(grouped["ci_high_pct"]).to_numpy(dtype=float)
    deaths = pd.to_numeric(grouped["deaths"]).astype(int).to_numpy()
    observed = pd.to_numeric(grouped["outcome_observed_n"]).astype(int).to_numpy()
    ax_b.errorbar(
        mortality,
        positions,
        xerr=[mortality - mortality_low, mortality_high - mortality],
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
    upper = min(100.0, max(5.0, float(mortality_high.max()) * 1.35))
    ax_b.set_xlim(0, upper)
    ax_b.set_xlabel("In-hospital mortality (%)")
    ax_b.set_title("Absolute mortality risk", loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, estimate, event_n, denominator_n in zip(
        positions,
        mortality,
        deaths,
        observed,
    ):
        label_on_right = estimate < upper * 0.68
        label_x = (
            float(estimate) + upper * 0.025
            if label_on_right
            else max(float(estimate) - upper * 0.06, upper * 0.02)
        )
        ax_b.text(
            label_x,
            position,
            f"{float(estimate):.1f}%  {int(event_n):,}/{int(denominator_n):,}",
            va="center",
            ha="left" if label_on_right else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_b, "B", x=-0.14, y=1.04)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.84, wspace=0.48)

    contract = make_figure_contract(
        figure_id=f"figure:{_FIGURE_PRODUCT}",
        core_claim=(
            "The binary exposure partition and in-hospital mortality risks are "
            "rendered from two reconciled, digest-verified parent tables."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=86.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Exposure prevalence",
                "role": "baseline_context",
                "claim": "The two categories partition the locked cohort.",
                "evidence_ids": [str(cohort_binding.get("evidence_id") or "")],
                "metadata": {
                    "chart_type": "bar_prevalence",
                    "source_data": [cohort_source.name],
                    "positive_prevalence_percent": prevalence_pct,
                    "positive_ci_percent": [
                        prevalence_low,
                        prevalence_high,
                    ],
                },
            },
            {
                "panel_id": "B",
                "title": "Absolute mortality risk",
                "role": "descriptive_result",
                "claim": (
                    "Deaths, observed outcomes, risks, and confidence intervals "
                    "are shown for both parent-defined categories."
                ),
                "evidence_ids": [str(outcome_binding.get("evidence_id") or "")],
                "metadata": {
                    "chart_type": "dot_interval_absolute_risk",
                    "source_data": [outcome_source.name],
                },
            },
        ],
        source_data=[cohort_source.name, outcome_source.name],
        statistics_note=(
            "The executor validates count/denominator identities, confidence "
            "interval ordering, and the shared binary partition. It introduces "
            "no cohort, exposure, outcome, missing-data, or modeling decision."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / _FIGURE_PRODUCT,
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
        "method": "deterministic_prevalence_mortality_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "prevalence_mortality_figure",
        "rendering_only": True,
        "source_inputs": list(PREVALENCE_MORTALITY_FIGURE_INPUTS),
        "source_evidence_ids": [
            cohort_binding.get("evidence_id"),
            outcome_binding.get("evidence_id"),
        ],
        "source_sha256": [
            cohort_binding.get("sha256"),
            outcome_binding.get("sha256"),
        ],
        "source_rows_consumed": int(len(cohort) + len(outcome)),
        "category_labels": list(category_labels),
        "locked_denominator": locked_n,
        "source_data_files": [cohort_source.name, outcome_source.name],
        "figure_files": figure_files,
        "figure_path": f"{_FIGURE_PRODUCT}.png",
        "figure_contract": f"{_FIGURE_PRODUCT}.figure_contract.json",
        "contract_files": [f"{_FIGURE_PRODUCT}.figure_contract.json"],
        "output_files": {f"figure:{_FIGURE_PRODUCT}": f"{_FIGURE_PRODUCT}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
