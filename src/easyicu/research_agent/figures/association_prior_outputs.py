"""Association publication rendering from registered parent products."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

import math

from .prior_output_support import (
    figure_parent_candidate_step_dirs as _figure_parent_candidate_step_dirs,
    publication_label as _publication_label,
    short_figure_label as _short_figure_label,
)
from .prior_output_contracts import _planned_primary_association_contract
from ..reporting.publication_bundles import _association_descriptive_context, _context_axis_label


def _render_association_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
    authorized_repair_id: Optional[str] = None,
) -> Optional[str]:
    """Deterministically build a multi-panel figure from a prior association step.

    Mirror of the prediction repair for adjusted-association analyses. Small
    models sometimes write a coefficient table (``odds_ratio`` + ``or_ci_low`` /
    ``or_ci_high`` columns) in the regression step but fail the follow-up
    figure-only step (e.g. hard-coding a wrong results filename). Rather than
    accepting a one-panel placeholder, render a source-data-backed association
    figure with uncertainty context from the registered parent table.
    """
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    # Resolve the odds-ratio + CI columns by common name variants, so the
    # rescue works whether the parent step wrote ``or_ci_low/or_ci_high`` (our
    # deterministic fallback) or ``ci_lower/ci_upper`` etc. (free-model code).
    # Without this the rescue silently skips a perfectly good coefficient table
    # and the figure-only step fails the whole run.
    _OR_ALIASES = ("odds_ratio", "oddsratio", "adjusted_or", "aor", "or")
    _CI_LOW_ALIASES = (
        "or_ci_low",
        "or_ci_lower",
        "ci_lower",
        "ci_low",
        "or_lower",
        "conf_low",
        "ci95_low",
        "ci_low_95",
        "lower",
    )
    _CI_HIGH_ALIASES = (
        "or_ci_high",
        "or_ci_upper",
        "ci_upper",
        "ci_high",
        "or_upper",
        "conf_high",
        "ci95_high",
        "ci_high_95",
        "upper",
    )

    def _resolve_or_ci_columns(frame: pd.DataFrame):
        lower_to_orig = {str(c).lower(): c for c in frame.columns}
        or_c = next((lower_to_orig[a] for a in _OR_ALIASES if a in lower_to_orig), None)
        if or_c is None and any(
            key in lower_to_orig
            for key in ("estimate", "point_estimate", "effect_estimate")
        ):
            scale_col = next(
                (
                    lower_to_orig[a]
                    for a in ("effect_scale", "scale", "measure")
                    if a in lower_to_orig
                ),
                None,
            )
            if scale_col is not None:
                scale_text = (
                    frame[scale_col]
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[_-]+", " ", regex=True)
                )
                if scale_text.str.contains(
                    r"\b(?:odds ratio|or)\b",
                    regex=True,
                    na=False,
                ).any():
                    estimate_key = next(
                        (
                            key
                            for key in (
                                "estimate",
                                "point_estimate",
                                "effect_estimate",
                            )
                            if key in lower_to_orig
                        ),
                        None,
                    )
                    if estimate_key is not None:
                        or_c = lower_to_orig[estimate_key]
        lo_c = next(
            (lower_to_orig[a] for a in _CI_LOW_ALIASES if a in lower_to_orig), None
        )
        hi_c = next(
            (lower_to_orig[a] for a in _CI_HIGH_ALIASES if a in lower_to_orig), None
        )
        if or_c and lo_c and hi_c:
            return or_c, lo_c, hi_c
        return None

    sealed_repair_id = "association_publication_bundle_from_planned_model_contract_v1"
    parent: Optional[tuple[Path, pd.DataFrame, tuple[str, str, str]]] = None
    if preverified_parent_artifacts is not None:
        if authorized_repair_id != sealed_repair_id:
            return None
        try:
            candidate_frame = pd.read_csv(
                io.BytesIO(
                    preverified_parent_artifacts["adjusted_association_estimates.csv"]
                )
            )
        except (KeyError, OSError, ValueError):
            return None
        resolved = _resolve_or_ci_columns(candidate_frame)
        if resolved is None:
            return None
        parent_step_id = str(current_step_id or "").removesuffix("_figure")
        table_path = (
            Path(run_dir)
            / "steps"
            / parent_step_id
            / "outputs"
            / "adjusted_association_estimates.csv"
        )
        parent = (table_path, candidate_frame, resolved)
    else:
        candidate_step_dirs, _direct_parent_only = _figure_parent_candidate_step_dirs(
            steps_dir=steps_dir, current_step_id=current_step_id
        )
        for step_dir in candidate_step_dirs:
            outputs_dir = step_dir / "outputs"
            if not outputs_dir.exists():
                continue
            candidates: List[
                tuple[tuple[int, int], Path, pd.DataFrame, tuple[str, str, str]]
            ] = []
            for csv_path in sorted(outputs_dir.glob("*.csv")):
                try:
                    candidate_frame = pd.read_csv(csv_path)
                except Exception:
                    continue
                resolved = _resolve_or_ci_columns(candidate_frame)
                if resolved is None:
                    continue
                columns = {str(column).lower() for column in candidate_frame.columns}
                structured_coefficients = {
                    "model_id",
                    "term",
                    "term_role",
                    "source_variable",
                }.issubset(columns)
                score = (
                    int(structured_coefficients),
                    int(
                        structured_coefficients
                        and "coefficient" in csv_path.stem.lower()
                    ),
                )
                candidates.append((score, csv_path, candidate_frame, resolved))
            if candidates:
                _, csv_path, candidate_frame, resolved = max(
                    candidates,
                    key=lambda item: item[0],
                )
                parent = (csv_path, candidate_frame, resolved)
                break
    if parent is None:
        return None

    table_path, frame, (or_col, lo_col, hi_col) = parent
    lower_to_orig = {str(c).lower(): c for c in frame.columns}

    parent_summary: Dict[str, Any] = {}
    summary_path = table_path.parent / "step_summary.json"
    if preverified_parent_artifacts is not None:
        try:
            loaded = json.loads(
                preverified_parent_artifacts["step_summary.json"].decode("utf-8")
            )
            if isinstance(loaded, dict):
                parent_summary = loaded
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    elif summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                parent_summary = loaded
        except Exception:
            parent_summary = {}
    primary_model_id = str(parent_summary.get("primary_model_id") or "").strip()
    model_contracts = parent_summary.get("model_contracts") or []
    if not isinstance(model_contracts, list):
        model_contracts = []
    primary_contract: Optional[Mapping[str, Any]] = None
    if authorized_repair_id == sealed_repair_id:
        primary_contract = _planned_primary_association_contract(
            run_dir,
            current_step_id,
            parent_summary,
        )
        if primary_contract is None:
            return None
        primary_model_id = str(primary_contract.get("model_id") or "").strip()
    else:
        primary_contract = next(
            (
                contract
                for contract in model_contracts
                if isinstance(contract, dict)
                and primary_model_id
                and str(contract.get("model_id") or "") == primary_model_id
            ),
            None,
        )
        if primary_contract is None:
            primary_contract = next(
                (
                    contract
                    for contract in model_contracts
                    if isinstance(contract, dict)
                    and str(contract.get("analysis_role") or "").lower() == "primary"
                    and str(contract.get("exposure_role") or "primary").lower()
                    == "primary"
                ),
                None,
            )
    if not primary_model_id and isinstance(primary_contract, dict):
        primary_model_id = str(primary_contract.get("model_id") or "").strip()
    primary_exposure = (
        str(primary_contract.get("exposure_source") or "").strip()
        if isinstance(primary_contract, dict)
        else ""
    )
    matching_model_ids = (
        {primary_model_id}
        if authorized_repair_id == sealed_repair_id
        else {
            str(contract.get("model_id") or "").strip()
            for contract in model_contracts
            if isinstance(contract, dict)
            and primary_exposure
            and str(contract.get("exposure_source") or "").strip() == primary_exposure
            and str(contract.get("exposure_role") or "primary").lower() == "primary"
            and str(contract.get("analysis_role") or "").lower()
            in {"primary", "sensitivity"}
        }
    )
    matching_model_ids.discard("")

    plot_df = frame.copy()
    term_role_col = lower_to_orig.get("term_role")
    if term_role_col is not None:
        exposure_rows = plot_df[term_role_col].astype(str).str.lower().eq("exposure")
        if exposure_rows.any():
            plot_df = plot_df.loc[exposure_rows].copy()
    model_id_col = lower_to_orig.get("model_id")
    if model_id_col is not None:
        if matching_model_ids:
            selected_models = plot_df[model_id_col].astype(str).isin(matching_model_ids)
            if selected_models.any():
                plot_df = plot_df.loc[selected_models].copy()
        elif primary_model_id:
            selected_primary = plot_df[model_id_col].astype(str).eq(primary_model_id)
            if selected_primary.any():
                plot_df = plot_df.loc[selected_primary].copy()
    source_variable_col = lower_to_orig.get("source_variable")
    if source_variable_col is not None and primary_exposure:
        selected_exposure = (
            plot_df[source_variable_col].astype(str).eq(primary_exposure)
        )
        if selected_exposure.any():
            plot_df = plot_df.loc[selected_exposure].copy()

    def _n_distinct(col: str) -> int:
        try:
            return int(plot_df[col].astype(str).nunique(dropna=True))
        except Exception:
            return 0

    # Pick the column that LABELS / keys each forest row. Prefer a known
    # variable/exposure-descriptor column, but only if it actually VARIES across
    # rows: an association table for a single graded exposure keeps the exposure
    # name constant (e.g. exposure_variable='sofa2_liver_cat' on every row) and
    # distinguishes rows by an ordinal level/band column. Keying on the constant
    # column collapses every forest row to one label and drops the per-row trace
    # key. Skip constant candidates and fall back to
    # the first varying column rather than blindly to columns[0].
    _LABEL_CANDIDATES = (
        "term",
        "variable",
        "exposure",
        "predictor",
        "feature",
        "covariate",
        "exposure_variable",
        "level",
        "band",
        "category",
        "stage",
        "group",
        "bin",
        "quantile",
        "tertile",
        "quartile",
        "decile",
    )
    _present = [
        str(lower_to_orig[key]) for key in _LABEL_CANDIDATES if key in lower_to_orig
    ]
    var_col = (
        # a named candidate that VARIES across rows (avoids the collapse) ...
        next((c for c in _present if _n_distinct(c) > 1), None)
        # ... else the first named candidate (single-row / genuinely all-constant
        # forests have no collapse risk; keep the original semantic label) ...
        or next(iter(_present), None)
        # ... else the first varying column, else the first column.
        or next((str(c) for c in plot_df.columns if _n_distinct(str(c)) > 1), None)
        or str(plot_df.columns[0])
    )
    # Drop the intercept term; it is not an interpretable effect estimate.
    intercept_col = lower_to_orig.get("term", var_col)
    plot_df = plot_df[
        ~plot_df[intercept_col].astype(str).str.lower().isin({"const", "intercept"})
    ]
    for _c in (or_col, lo_col, hi_col):
        plot_df = plot_df.assign(**{_c: pd.to_numeric(plot_df[_c], errors="coerce")})
    plot_df = plot_df.dropna(subset=[or_col, lo_col, hi_col])
    if plot_df.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = plot_df[var_col].astype(str).tolist()
    full_display_labels = [_publication_label(label) for label in labels]
    analysis_set_col = lower_to_orig.get("analysis_set")
    if len(plot_df) > 1 and model_id_col is not None:
        qualified_labels: List[str] = []
        for row_idx, base_label in enumerate(full_display_labels):
            row = plot_df.iloc[row_idx]
            qualifier = (
                row.get(analysis_set_col)
                if analysis_set_col is not None
                else row.get(model_id_col)
            )
            qualified_labels.append(f"{base_label} ({_publication_label(qualifier)})")
        full_display_labels = qualified_labels
    display_labels = [
        _short_figure_label(label.replace("Maximum ", "Max "), limit=32)
        for label in full_display_labels
    ]
    or_vals = plot_df[or_col].astype(float).to_numpy()
    lo = plot_df[lo_col].astype(float).to_numpy()
    hi = plot_df[hi_col].astype(float).to_numpy()
    ci_width = hi - lo
    y = list(range(len(labels)))
    source_row_indices = plot_df.index.to_list()
    source_data = plot_df.copy().reset_index(drop=True)
    source_data = source_data.assign(
        source_row_index=source_row_indices,
        display_label=full_display_labels,
        plot_label=display_labels,
        point_estimate=or_vals,
        odds_ratio=or_vals,
        ci_low=lo,
        ci_high=hi,
        ci_width=ci_width,
        source_table=table_path.name,
    )
    source_data.to_csv(out_dir / "publication_figure_source_data.csv", index=False)
    descriptive_context = (
        {
            "plot_rows": [],
            "source_files": [],
            "has_prevalence": False,
            "has_outcome_risk": False,
            "title": "",
            "claim": "",
        }
        if preverified_parent_artifacts is not None
        else _association_descriptive_context(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            primary_exposure=primary_exposure or None,
        )
    )
    descriptive_rows = list(descriptive_context.get("plot_rows") or [])
    association_panel_title = (
        "Primary adjusted association" if len(labels) <= 3 else "Adjusted association"
    )
    association_panel_claim = (
        "The primary adjusted odds ratio and 95% CI are read from the parent association table."
        if len(labels) <= 3
        else (
            "Per-covariate adjusted odds ratios and 95% CIs are read "
            "from the parent association table."
        )
    )
    association_chart_type = "dot_interval" if len(labels) <= 3 else "forest"

    palette = apply_publication_style()
    if descriptive_rows:
        fig_height_mm = max(82, 18 * len(labels) + 22, 16 * len(descriptive_rows) + 28)
        fig = plt.figure(
            figsize=(183 / 25.4, fig_height_mm / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.02, 1.28],
            left=0.18,
            right=0.98,
            top=0.90,
            bottom=0.18,
            wspace=0.82,
        )
        ax_context = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1])

        context_df = pd.DataFrame(descriptive_rows)
        context_labels = []
        for _, row in context_df.iterrows():
            metric = str(row.get("plot_metric") or "").strip()
            group = str(row.get("plot_group_label") or "").strip()
            context_labels.append(_context_axis_label(metric, group))
        context_x = pd.to_numeric(
            context_df["plot_estimate_pct"], errors="coerce"
        ).to_numpy()
        context_lo = (
            pd.to_numeric(
                context_df.get("plot_ci_low_pct", context_df["plot_estimate_pct"]),
                errors="coerce",
            )
            .fillna(pd.Series(context_x))
            .to_numpy()
        )
        context_hi = (
            pd.to_numeric(
                context_df.get("plot_ci_high_pct", context_df["plot_estimate_pct"]),
                errors="coerce",
            )
            .fillna(pd.Series(context_x))
            .to_numpy()
        )
        y_context = list(range(len(context_labels)))
        ax_context.errorbar(
            context_x,
            y_context,
            xerr=[
                [
                    max(0.0, center - lower)
                    for center, lower in zip(context_x, context_lo)
                ],
                [
                    max(0.0, upper - center)
                    for center, upper in zip(context_x, context_hi)
                ],
            ],
            fmt="o",
            color=palette.get("teal", "#42949E"),
            ecolor=palette.get("teal", "#42949E"),
            elinewidth=1.0,
            capsize=2.3,
            markersize=4.0,
        )
        max_context = max(
            [float(x) for x in context_hi if math.isfinite(float(x))] or [1.0]
        )
        ax_context.set_xlim(0, max(5.0, max_context + 8.0, max_context * 1.35))
        ax_context.set_yticks(y_context)
        ax_context.set_yticklabels(context_labels, fontsize=6.8)
        ax_context.set_xlabel("Percent (95% CI)")
        ax_context.set_title(str(descriptive_context["title"]), loc="left", pad=4)
        ax_context.set_ylim(max(len(context_labels) + 1.8, 4.2), -0.5)
        ax_context.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        for row_idx, row in context_df.iterrows():
            event_n = pd.to_numeric(
                pd.Series([row.get("plot_event_n")]), errors="coerce"
            ).iloc[0]
            denom = pd.to_numeric(
                pd.Series([row.get("plot_denominator")]), errors="coerce"
            ).iloc[0]
            if pd.notna(event_n) and pd.notna(denom):
                label = f"{float(context_x[row_idx]):.1f}% ({int(event_n):,}/{int(denom):,})"
            else:
                label = f"{float(context_x[row_idx]):.1f}%"
            ax_context.text(
                max(float(context_hi[row_idx]), float(context_x[row_idx])) + 0.6,
                row_idx,
                label,
                va="center",
                fontsize=6.3,
                color=palette.get("baseline", "#272727"),
            )
        add_panel_label(ax_context, "A", x=0.0, y=1.08)
    else:
        fig = plt.figure(
            figsize=(183 / 25.4, max(72, 18 * len(labels) + 18) / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.45, 0.85],
            left=0.22,
            right=0.98,
            top=0.90,
            bottom=0.18,
            wspace=0.42,
        )
        ax = fig.add_subplot(grid[0, 0])
        ax_width = fig.add_subplot(grid[0, 1], sharey=ax)
    ax.errorbar(
        or_vals,
        y,
        xerr=[
            [max(0.0, center - lower) for center, lower in zip(or_vals, lo)],
            [max(0.0, upper - center) for center, upper in zip(or_vals, hi)],
        ],
        fmt="o",
        color=palette.get("blue", "#0F4D92"),
        ecolor=palette.get("blue", "#0F4D92"),
        elinewidth=1.0,
        capsize=2.3,
        markersize=4.0,
    )
    ax.axvline(
        1.0,
        color=palette.get("neutral", "#8F8F8F"),
        linewidth=0.9,
        linestyle="--",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=6.6)
    ax.set_xlabel("Adjusted odds ratio (95% CI)")
    ax.set_title(association_panel_title, loc="left", pad=4)
    if len(labels) <= 3:
        max_hi = max(float(value) for value in hi if math.isfinite(float(value)))
        ax.set_xlim(
            left=max(0.01, min(float(value) for value in lo) * 0.96),
            right=max_hi * 1.28,
        )
        for row_idx, (center, lower, upper) in enumerate(zip(or_vals, lo, hi)):
            ax.text(
                float(upper) * 1.025,
                row_idx,
                f"OR {float(center):.2f} ({float(lower):.2f}-{float(upper):.2f})",
                va="center",
                fontsize=6.2,
                color=palette.get("baseline", "#272727"),
            )
    ax.invert_yaxis()
    ax.grid(
        axis="x",
        color=palette.get("neutral_light", "#D8D8D8"),
        linewidth=0.55,
        alpha=0.8,
    )
    add_panel_label(ax, "B" if descriptive_rows else "A", x=0.0, y=1.08)

    if not descriptive_rows:
        ax_width.barh(
            y,
            ci_width,
            color=palette.get("orange", "#E69F00"),
            height=0.5,
        )
        ax_width.set_xlabel("95% CI width")
        ax_width.set_title("Estimate precision", loc="left", pad=4)
        ax_width.tick_params(axis="y", labelleft=False)
        ax_width.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        add_panel_label(ax_width, "B", x=0.0, y=1.08)

    source_data_files = [
        "publication_figure_source_data.csv",
        *[str(item) for item in descriptive_context.get("source_files", [])],
    ]
    if descriptive_rows:
        panels = [
            {
                "panel_id": "A",
                "title": str(descriptive_context["title"]),
                "role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": str(descriptive_context["claim"]),
                "evidence_ids": [
                    item for item in descriptive_context.get("source_files", [])
                ],
            },
            {
                "panel_id": "B",
                "title": association_panel_title,
                "role": "primary_estimand",
                "chart_type": association_chart_type,
                "claim": association_panel_claim,
                "evidence_ids": ["publication_figure_source_data.csv"],
                "metadata": {"planner_product_slots": ["primary_estimand"]},
            },
        ]
        core_claim = (
            "The figure pairs reader-facing prevalence or absolute-risk context "
            "with the adjusted association estimate and uncertainty."
        )
    else:
        panels = [
            {
                "panel_id": "A",
                "title": association_panel_title,
                "role": "primary_estimand",
                "chart_type": association_chart_type,
                "claim": association_panel_claim,
                "evidence_ids": ["publication_figure_source_data.csv"],
                "metadata": {"planner_product_slots": ["primary_estimand"]},
            },
            {
                "panel_id": "B",
                "title": "Interval-width audit",
                "role": "robustness",
                "chart_type": "bar",
                "claim": (
                    "The width of each 95% CI is shown to expose estimate "
                    "precision rather than hiding uncertainty in the forest plot."
                ),
                "evidence_ids": ["publication_figure_source_data.csv"],
                "metadata": {"planner_product_slots": ["precision_audit"]},
            },
        ]
        core_claim = (
            "Adjusted associations and their uncertainty are summarised from "
            "the registered association coefficient table."
        )
    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=core_claim,
        panels=panels,
        source_data=source_data_files,
        statistics_note=(
            "Generated deterministically from registered parent-step tables; "
            "the association panel uses the coefficient table and any context "
            "panel uses prevalence or outcome-risk source tables when present."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "publication_figure", contract=contract, dpi=300
    )
    plt.close(fig)

    existing_summary: Dict[str, Any] = {}
    step_summary_path = out_dir / "step_summary.json"
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    observed_repair_id = (
        sealed_repair_id
        if authorized_repair_id == sealed_repair_id
        else (
            "association_publication_bundle_from_parent_outputs_v3"
            if descriptive_rows
            else "association_publication_bundle_from_parent_outputs_v2"
        )
    )
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_association_publication_figure_repair",
            "rendering_only": True,
            "deterministic_publication_figure_rescue": observed_repair_id,
            "source_step_id": current_step_id.removesuffix("_figure"),
            "figure_contract": "publication_figure.figure_contract.json",
        }
    )
    existing_summary.setdefault("publication_figure_repair", {})
    existing_summary["publication_figure_repair"].update(
        {
            "mode": "association_forest_from_parent_outputs",
            "source_association_table": str(table_path),
            "source_data": "publication_figure_source_data.csv",
            "descriptive_source_data": descriptive_context.get("source_files", []),
            "primary_model_id": primary_model_id or None,
            "primary_exposure_source": primary_exposure or None,
            "selected_model_ids": sorted(matching_model_ids),
            "n_association_rows": int(len(plot_df)),
        }
    )
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    existing_summary["figure_files"] = figure_files
    if figure_files:
        existing_summary["figure_path"] = figure_files[0]
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return observed_repair_id
