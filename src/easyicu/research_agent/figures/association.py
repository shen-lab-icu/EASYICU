"""Generic continuous-exposure association publication figure.

The renderer consumes only registered, digest-verified result tables.  It
recognises a continuous exposure grid with a single reference value and a
ratio-scale estimate, and optionally adds a model-standardised absolute-risk
panel when an aligned registered grid is available.  It never reads patient
rows or refits a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..authority.evidence_store import EvidenceStore
from ..schema import AnalysisPlan, EvidenceRecord, ResearchContext
from .base import RenderedFigure, find_table_records, read_table, verified_record_path


_CURVE_NAMES = (
    "continuous_association_curve",
    "exposure_response_curve",
    "dose_response_curve",
    "rcs_curve",
    "restricted_cubic_spline_curve",
)
_RISK_NAMES = (
    "adjusted_absolute_risk",
    "standardized_absolute_risk",
    "standardised_absolute_risk",
    "absolute_risk_curve",
)
_X_COLUMNS = ("exposure_value", "dose", "exposure_level_numeric")
_REFERENCE_COLUMNS = (
    "reference_exposure_value",
    "reference_dose",
    "reference_value",
)
_RATIO_COLUMNS = (
    "adjusted_odds_ratio",
    "odds_ratio",
    "adjusted_hazard_ratio",
    "hazard_ratio",
    "adjusted_risk_ratio",
    "risk_ratio",
    "relative_risk",
)
_RISK_COLUMNS = (
    "adjusted_absolute_risk",
    "standardized_absolute_risk",
    "standardised_absolute_risk",
    "absolute_risk",
    "predicted_risk",
)
_LOWER_COLUMNS = ("ci_low", "ci_lower", "lower_ci", "lower")
_UPPER_COLUMNS = ("ci_high", "ci_upper", "upper_ci", "upper")


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    return next((lookup[name] for name in names if name in lookup), None)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _concept_label(context: ResearchContext, name: Optional[str]) -> str:
    raw_name = str(name or "").strip()
    variable = context.variable(raw_name) if raw_name else None
    description = str(getattr(variable, "description", "") or "").strip()
    label = description or raw_name.replace("_", " ") or "Variable"
    unit = str(getattr(variable, "unit", "") or "").strip()
    return f"{label} ({unit})" if unit else label


def _normalise_curve(
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
    value_name: str,
    min_rows: int = 7,
) -> Optional[pd.DataFrame]:
    x_col = _column(frame, _X_COLUMNS)
    reference_col = _column(frame, _REFERENCE_COLUMNS)
    value_col = _column(frame, value_columns)
    lower_col = _column(frame, _LOWER_COLUMNS)
    upper_col = _column(frame, _UPPER_COLUMNS)
    if None in {x_col, reference_col, value_col, lower_col, upper_col}:
        return None
    assert x_col and reference_col and value_col and lower_col and upper_col
    out = pd.DataFrame(
        {
            "exposure_value": _numeric(frame, x_col),
            "reference_exposure_value": _numeric(frame, reference_col),
            value_name: _numeric(frame, value_col),
            "ci_low": _numeric(frame, lower_col),
            "ci_high": _numeric(frame, upper_col),
        }
    ).dropna()
    out = out.sort_values("exposure_value").drop_duplicates("exposure_value")
    if len(out) < min_rows or out["reference_exposure_value"].nunique() != 1:
        return None
    if not out["exposure_value"].is_monotonic_increasing:
        return None
    if not (
        (out["ci_low"] <= out[value_name])
        & (out[value_name] <= out["ci_high"])
    ).all():
        return None
    return out.reset_index(drop=True)


def _load_best_curve(
    evidence: EvidenceStore,
    run_dir: Path,
    *,
    names: tuple[str, ...],
    value_columns: tuple[str, ...],
    value_name: str,
) -> tuple[Optional[EvidenceRecord], Optional[pd.DataFrame]]:
    candidates = find_table_records(evidence, names)
    # A primary-lineage evidence view has already narrowed the scientific
    # ownership. Inspecting the remaining registered tables by schema lets the
    # renderer support neutral filenames without searching the filesystem.
    seen = {record.evidence_id for record in candidates}
    candidates.extend(
        record
        for record in evidence.records()
        if record.kind == "table" and record.evidence_id not in seen
    )
    accepted: list[tuple[int, EvidenceRecord, pd.DataFrame]] = []
    for record in candidates:
        try:
            frame = read_table(verified_record_path(run_dir, record))
            normalised = _normalise_curve(
                frame,
                value_columns=value_columns,
                value_name=value_name,
            )
        except Exception:
            continue
        if normalised is None:
            continue
        stem = Path(record.relative_path).stem.lower()
        name_score = sum(token in stem for token in names)
        accepted.append((100 * name_score + len(normalised), record, normalised))
    if not accepted:
        return None, None
    accepted.sort(key=lambda row: row[0], reverse=True)
    _, record, normalised = accepted[0]
    return record, normalised


def render_continuous_association_figure(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Optional[RenderedFigure]:
    curve_record, curve = _load_best_curve(
        evidence,
        run_dir,
        names=_CURVE_NAMES,
        value_columns=_RATIO_COLUMNS,
        value_name="ratio_estimate",
    )
    if curve_record is None or curve is None:
        return None
    if (curve[["ratio_estimate", "ci_low", "ci_high"]] <= 0).any().any():
        return None

    risk_record, risk = _load_best_curve(
        evidence,
        run_dir,
        names=_RISK_NAMES,
        value_columns=_RISK_COLUMNS,
        value_name="absolute_risk",
    )
    if risk is not None:
        aligned = (
            len(risk) == len(curve)
            and risk["exposure_value"].round(10).equals(
                curve["exposure_value"].round(10)
            )
            and 0 <= float(risk["ci_low"].min())
            and float(risk["ci_high"].max()) <= 1
        )
        if not aligned:
            risk_record, risk = None, None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .publication import add_panel_label, apply_publication_style

    palette = apply_publication_style()
    panel_count = 2 if risk is not None else 1
    fig, raw_axes = plt.subplots(
        1,
        panel_count,
        figsize=((183 if panel_count == 2 else 126) / 25.4, 91 / 25.4),
        squeeze=False,
    )
    axes = raw_axes[0]
    fig.subplots_adjust(
        left=0.10 if panel_count == 2 else 0.16,
        right=0.985,
        bottom=0.18,
        top=0.88,
        wspace=0.34,
    )

    exposure_label = _concept_label(context, context.primary_exposure)
    outcome_label = _concept_label(context, context.target_outcome)
    reference = float(curve["reference_exposure_value"].iloc[0])
    x = curve["exposure_value"].to_numpy(float)
    estimate = curve["ratio_estimate"].to_numpy(float)
    lower = curve["ci_low"].to_numpy(float)
    upper = curve["ci_high"].to_numpy(float)
    ax = axes[0]
    ax.fill_between(x, lower, upper, color="#DDE8F5", linewidth=0, zorder=1)
    ax.plot(x, estimate, color=palette.get("blue", "#0F4D92"), linewidth=1.8)
    ax.axhline(1.0, color="#727A87", linestyle=(0, (3, 3)), linewidth=0.8)
    ax.axvline(reference, color="#727A87", linestyle=(0, (3, 3)), linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(lower.min()) * 0.94, float(upper.max()) * 1.04)
    ax.set_xlabel(exposure_label)
    ax.set_ylabel(f"{outcome_label} ratio (95% CI)")
    ax.set_title("Exposure–response association", loc="left", pad=6)
    ax.grid(axis="y", color="#E3E6EA", linewidth=0.55, zorder=0)
    add_panel_label(ax, "A", x=-0.14, y=1.05, fontsize=10.0)

    panels = [
        {
            "panel_id": "A",
            "title": "Continuous exposure–response association",
            "role": "primary_estimand",
            "metadata": {"chart_type": "marginal_effect_panel"},
            "claim": (
                "The registered ratio-scale association and 95% confidence "
                "interval are shown across the prespecified exposure grid."
            ),
            "evidence_ids": [curve_record.evidence_id],
            "review_risk": (
                "The curve is an observational association and depends on the "
                "registered model, reference value, and adjustment set."
            ),
        }
    ]
    source_ids = [curve_record.evidence_id]
    source_frames = {"continuous_association": curve}
    if risk is not None and risk_record is not None:
        ax = axes[1]
        risk_x = risk["exposure_value"].to_numpy(float)
        risk_estimate = 100 * risk["absolute_risk"].to_numpy(float)
        risk_lower = 100 * risk["ci_low"].to_numpy(float)
        risk_upper = 100 * risk["ci_high"].to_numpy(float)
        ax.fill_between(
            risk_x, risk_lower, risk_upper, color="#D9EFEE", linewidth=0, zorder=1
        )
        ax.plot(risk_x, risk_estimate, color="#248A8D", linewidth=1.8)
        ax.axvline(reference, color="#727A87", linestyle=(0, (3, 3)), linewidth=0.8)
        ax.set_xlim(float(risk_x.min()), float(risk_x.max()))
        span = max(float(risk_upper.max() - risk_lower.min()), 0.5)
        ax.set_ylim(
            max(0.0, float(risk_lower.min()) - 0.08 * span),
            float(risk_upper.max()) + 0.08 * span,
        )
        ax.set_xlabel(exposure_label)
        ax.set_ylabel(f"Model-standardised {outcome_label} risk (%)")
        ax.set_title("Model-standardised absolute risk", loc="left", pad=6)
        ax.grid(axis="y", color="#E3E6EA", linewidth=0.55, zorder=0)
        add_panel_label(ax, "B", x=-0.14, y=1.05, fontsize=10.0)
        panels.append(
            {
                "panel_id": "B",
                "title": "Model-standardised absolute risk",
                "role": "descriptive_result",
                "metadata": {"chart_type": "absolute_risk_curve"},
                "claim": (
                    "The registered model-standardised absolute outcome risk and "
                    "95% confidence interval are shown on the same exposure grid."
                ),
                "evidence_ids": [risk_record.evidence_id],
                "review_risk": (
                    "Model-standardised risk is conditional on the registered "
                    "analysis population and covariate distribution."
                ),
            }
        )
        source_ids.append(risk_record.evidence_id)
        source_frames["absolute_risk"] = risk

    return RenderedFigure(
        fig=fig,
        figure_id="easyicu_publication_figure",
        core_claim=(
            f"The registered continuous {exposure_label} association with "
            f"{outcome_label} is displayed across the prespecified exposure grid."
        ),
        generation_mode="continuous_association_publication_figure",
        panels=panels,
        source_evidence_ids=source_ids,
        source_frames=source_frames,
        statistics_note=(
            "Rendered deterministically from digest-verified continuous-effect "
            "and aligned absolute-risk tables. No patient rows are read and no "
            "model is refitted by the figure renderer."
        ),
    )


__all__ = ["render_continuous_association_figure"]
