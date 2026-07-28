"""Render a sealed missingness and measurement-availability audit.

The renderer consumes exactly the two tables declared by one direct parent.
It does not scan sibling outputs, choose variables, redefine missingness, or
read the cohort.  Percentages are recomputed from the sealed integer counts.
"""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from ..authority.parent_artifact import (
    _resolve_upstream_manifest_step,
    _verified_direct_parent_artifact_digests,
)
from ..contracts.declared_product import read_digest_bound_artifact_snapshot
from ..planning.method_vocabulary import MISSINGNESS_SOURCE_AVAILABILITY_AUDIT
from .publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)

REPAIR_ID = "missingness_publication_bundle_from_parent_outputs_v1"
CONTROLLED_METHOD = MISSINGNESS_SOURCE_AVAILABILITY_AUDIT
_PRODUCT_FILES = {
    "table:missingness_audit": "missingness_audit.csv",
    "table:measurement_source_audit": "measurement_source_audit.csv",
}
_MISSINGNESS_COLUMNS = {
    "concept",
    "variable",
    "value_column",
    "n_total",
    "n_nonmissing",
    "missing_n",
    "missing_pct",
}
_SOURCE_COLUMNS = {
    "concept",
    "variable",
    "value_column",
    "n_total",
    "measured_one_n",
    "value_missing_n",
    "indicator_semantics",
    "missingness_kind",
}


def _safe_output_files(summary: Mapping[str, object]) -> Optional[dict[str, str]]:
    raw = summary.get("output_files")
    if not isinstance(raw, Mapping):
        return None
    selected: dict[str, str] = {}
    for product, expected_name in _PRODUCT_FILES.items():
        name = str(raw.get(product) or "").strip()
        if name != expected_name or Path(name).name != name:
            return None
        selected[product] = name
    return selected


def missingness_source_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[dict[str, str]]:
    """Return the exact three-file parent seal for the controlled renderer."""

    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return None
    if str(request_step.get("method") or "").strip().lower() != CONTROLLED_METHOD:
        return None
    expected_outputs = {
        str(item).strip() for item in request_step.get("expected_outputs") or []
    }
    if set(_PRODUCT_FILES) != expected_outputs:
        return None

    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    required = {"step_summary.json", *_PRODUCT_FILES.values()}
    if not digests or not required <= set(digests):
        return None
    parent_step_id = str(figure_step_id).removesuffix("_figure")
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    try:
        snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests={name: digests[name] for name in required},
        )
        summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(summary, Mapping) or _safe_output_files(summary) is None:
        return None
    if _validated_source_frame(snapshot) is None:
        return None
    return {name: digests[name] for name in sorted(required)}


def _read_csv(payload: bytes) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(io.BytesIO(payload))
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _validated_source_frame(
    snapshot: Mapping[str, bytes],
) -> Optional[pd.DataFrame]:
    missing = _read_csv(snapshot.get("missingness_audit.csv", b""))
    source = _read_csv(snapshot.get("measurement_source_audit.csv", b""))
    if missing is None or source is None or missing.empty or source.empty:
        return None
    if not _MISSINGNESS_COLUMNS <= set(missing) or not _SOURCE_COLUMNS <= set(source):
        return None
    if missing["concept"].duplicated().any() or source["concept"].duplicated().any():
        return None
    if set(missing["concept"].astype(str)) != set(source["concept"].astype(str)):
        return None

    for frame in (missing, source):
        frame["eligible_n"] = frame.get("eligible_n", frame["n_total"])
        frame["not_applicable_n"] = frame.get("not_applicable_n", 0)
    source["event_present_n"] = source.get("event_present_n", 0)
    source["event_absent_n"] = source.get("event_absent_n", 0)
    source["before_origin_n"] = source.get("before_origin_n", 0)

    number_columns = (
        "n_total",
        "n_nonmissing",
        "missing_n",
        "eligible_n",
        "not_applicable_n",
    )
    source_number_columns = (
        "n_total",
        "measured_one_n",
        "value_missing_n",
        "eligible_n",
        "not_applicable_n",
        "event_present_n",
        "event_absent_n",
        "before_origin_n",
    )
    for column in number_columns:
        missing[column] = pd.to_numeric(missing[column], errors="coerce")
    for column in source_number_columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if missing[list(number_columns)].isna().any().any():
        return None
    if source[list(source_number_columns)].isna().any().any():
        return None
    if (missing[list(number_columns)] < 0).any().any():
        return None
    if (source[list(source_number_columns)] < 0).any().any():
        return None
    if not (
        missing["n_nonmissing"]
        + missing["missing_n"]
        + missing["not_applicable_n"]
    ).equals(missing["n_total"]):
        return None

    merged = missing.merge(
        source,
        on="concept",
        suffixes=("_missing", "_source"),
        validate="one_to_one",
    )
    if not merged["n_total_missing"].equals(merged["n_total_source"]):
        return None
    if not merged["n_nonmissing"].equals(merged["measured_one_n"]):
        return None
    if not merged["missing_n"].equals(merged["value_missing_n"]):
        return None
    if not merged["eligible_n_missing"].equals(merged["eligible_n_source"]):
        return None
    if not merged["not_applicable_n_missing"].equals(
        merged["not_applicable_n_source"]
    ):
        return None

    semantics_column = (
        "indicator_semantics_source"
        if "indicator_semantics_source" in merged
        else "indicator_semantics"
    )
    kind_column = (
        "missingness_kind_source"
        if "missingness_kind_source" in merged
        else "missingness_kind"
    )
    merged["indicator_semantics"] = merged[semantics_column].astype(str)
    merged["missingness_kind"] = merged[kind_column].astype(str)
    if "indicator_semantics_missing" in merged and not merged[
        "indicator_semantics_missing"
    ].astype(str).equals(merged["indicator_semantics"]):
        return None
    if "missingness_kind_missing" in merged and not merged[
        "missingness_kind_missing"
    ].astype(str).equals(merged["missingness_kind"]):
        return None

    measurement = merged["indicator_semantics"].eq("measurement_availability")
    binary_event = merged["indicator_semantics"].eq("binary_event_presence")
    conditional_time = merged["indicator_semantics"].eq("conditional_event_time")
    if not (measurement | binary_event | conditional_time).all():
        return None
    kinds = merged["missingness_kind"]
    if not kinds[measurement].isin(
        ["measurement_missing", "structural_no_source"]
    ).all():
        return None
    if not kinds[binary_event].eq("binary_event_status_complete").all():
        return None
    if not kinds[conditional_time].eq("conditional_event_time").all():
        return None

    source_closed = (
        merged["measured_one_n"]
        + merged["value_missing_n"]
        + merged["not_applicable_n_source"]
    ).eq(merged["n_total_source"])
    if not source_closed[measurement | conditional_time].all():
        return None
    if not (
        merged.loc[binary_event, "event_present_n"]
        + merged.loc[binary_event, "event_absent_n"]
    ).eq(merged.loc[binary_event, "n_total_source"]).all():
        return None
    if not merged.loc[binary_event, "measured_one_n"].eq(
        merged.loc[binary_event, "n_total_source"]
    ).all():
        return None
    if not merged.loc[binary_event, "value_missing_n"].eq(0).all():
        return None
    if not merged.loc[conditional_time, "eligible_n_source"].eq(
        merged.loc[conditional_time, "event_present_n"]
    ).all():
        return None
    if not merged.loc[conditional_time, "not_applicable_n_source"].eq(
        merged.loc[conditional_time, "event_absent_n"]
    ).all():
        return None
    if not merged.loc[conditional_time, "before_origin_n"].eq(0).all():
        return None

    structural = kinds == "structural_no_source"
    if bool((merged.loc[structural, "measured_one_n"] != 0).any()):
        return None

    denominator = merged["n_total_missing"].astype(float)
    if (denominator <= 0).any():
        return None
    merged["observed_or_present_n"] = merged["n_nonmissing"]
    merged.loc[binary_event, "observed_or_present_n"] = merged.loc[
        binary_event, "event_present_n"
    ]
    merged["absent_or_not_applicable_n"] = merged["not_applicable_n_missing"]
    merged.loc[binary_event, "absent_or_not_applicable_n"] = merged.loc[
        binary_event, "event_absent_n"
    ]
    merged["missing_pct"] = merged["missing_n"].astype(float) * 100.0 / denominator
    merged["available_pct"] = (
        merged["observed_or_present_n"].astype(float) * 100.0 / denominator
    )
    merged["not_applicable_pct"] = (
        merged["absent_or_not_applicable_n"].astype(float) * 100.0 / denominator
    )
    return merged.sort_values(
        ["missing_pct", "concept"], ascending=[False, True]
    ).reset_index(drop=True)


def _label(value: object) -> str:
    text = str(value).replace("_", " ").strip()
    return "\n".join(
        textwrap.wrap(text, width=24, break_long_words=False, break_on_hyphens=False)
    )


def render_missingness_source_bundle(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Mapping[str, bytes],
) -> Optional[str]:
    """Render percentages from an already digest-verified parent snapshot."""

    required = {"step_summary.json", *_PRODUCT_FILES.values()}
    if set(preverified_parent_artifacts) != required:
        return None
    try:
        summary = json.loads(preverified_parent_artifacts["step_summary.json"].decode())
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(summary, Mapping) or _safe_output_files(summary) is None:
        return None
    source = _validated_source_frame(preverified_parent_artifacts)
    if source is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = out_dir / "missingness_measurement_panel_source_data.csv"
    source.loc[
        :,
        [
            "concept",
            "variable_missing",
            "value_column_missing",
            "n_total_missing",
            "n_nonmissing",
            "missing_n",
            "eligible_n_missing",
            "not_applicable_n_missing",
            "event_present_n",
            "event_absent_n",
            "before_origin_n",
            "available_pct",
            "missing_pct",
            "not_applicable_pct",
            "indicator_semantics",
            "missingness_kind",
        ],
    ].rename(
        columns={
            "variable_missing": "variable",
            "value_column_missing": "value_column",
            "n_total_missing": "n_total",
            "eligible_n_missing": "eligible_n",
            "not_applicable_n_missing": "not_applicable_n",
        }
    ).to_csv(
        source_path, index=False
    )

    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    height_mm = max(88.0, 11.0 * len(source) + 30.0)
    fig, ax = plt.subplots(figsize=(183 / 25.4, height_mm / 25.4))
    positions = list(range(len(source)))
    available = source["available_pct"].astype(float)
    missing = source["missing_pct"].astype(float)
    not_applicable = source["not_applicable_pct"].astype(float)
    structural_no_source = (
        source["missingness_kind"].astype(str).eq("structural_no_source")
    )
    plotted_available = available.where(~structural_no_source, 0.0)
    plotted_missing = missing.where(~structural_no_source, 0.0)
    plotted_no_source = structural_no_source.astype(float) * 100.0
    ax.barh(
        positions,
        plotted_available,
        color=palette["blue_soft"],
        height=0.62,
        label="Observed / event present",
    )
    ax.barh(
        positions,
        plotted_missing,
        left=plotted_available,
        color=palette["neutral_light"],
        height=0.62,
        label="Missing among eligible",
    )
    ax.barh(
        positions,
        not_applicable,
        left=plotted_available + plotted_missing,
        color=palette["neutral"],
        alpha=0.45,
        height=0.62,
        label="Event absent / not applicable",
    )
    ax.barh(
        positions,
        plotted_no_source,
        color=palette["band"],
        edgecolor=palette["neutral"],
        hatch="////",
        linewidth=0.6,
        height=0.62,
        label="No source",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([_label(value) for value in source["concept"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Analysis cohort (%)")
    ax.set_title("Availability of declared audit inputs", loc="left")
    ax.grid(axis="x", color=palette["neutral_light"], linewidth=0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2)
    for y, (
        missing_pct,
        missing_n,
        total_n,
        kind,
        semantics,
        event_present_n,
        event_absent_n,
        eligible_n,
    ) in enumerate(
        zip(
            missing,
            source["missing_n"],
            source["n_total_missing"],
            source["missingness_kind"].astype(str),
            source["indicator_semantics"].astype(str),
            source["event_present_n"],
            source["event_absent_n"],
            source["eligible_n_missing"],
        )
    ):
        if kind == "structural_no_source":
            label = f"No source in cohort (0/{int(total_n):,})"
        elif semantics == "binary_event_presence":
            label = (
                f"Event present {int(event_present_n):,}; "
                f"absent {int(event_absent_n):,}"
            )
        elif semantics == "conditional_event_time":
            label = (
                f"Time missing {int(missing_n):,}/{int(eligible_n):,} "
                "event-positive"
            )
        else:
            label = f"Missing {missing_pct:.1f}% ({int(missing_n):,}/{int(total_n):,})"
        ax.text(101.0, y, label, va="center", ha="left", fontsize=6.2, clip_on=False)
    add_panel_label(ax, "A", x=-0.14, y=1.02)
    fig.subplots_adjust(left=0.20, right=0.76, bottom=0.14, top=0.88)

    stem = "missingness_measurement_panel"
    contract = make_figure_contract(
        figure_id="figure:missingness_measurement",
        core_claim=(
            "Availability, event absence, conditional non-applicability, and true "
            "missingness are reported for every Planner-declared audit input."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=height_mm,
        panels=[
            {
                "panel_id": "A",
                "title": "Measurement availability",
                "role": "data_quality",
                "claim": (
                    "Observed or event-present, truly missing, event-absent or "
                    "not-applicable, and structural no-source counts form a "
                    "typed closed partition."
                ),
                "evidence_ids": [source_path.name],
                # Anchor the sealed renderer's authorized figure product slot to
                # this panel.  ``bind_declared_figure_products`` fails closed with
                # "authorized product slot is not anchored to a contract panel"
                # unless a panel claims the slot via ``planner_product_slots``.
                # The slot name is the one the repair registry authorizes for
                # this renderer (``missingness_measurement``); mirror the pattern
                # already used by absolute_risk / distribution_availability.
                "metadata": {
                    "planner_product_slots": ["missingness_measurement"],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "Percentages are recomputed from digest-bound integer counts; this "
            "renderer performs no cohort selection, imputation, or modelling. "
            "Concepts with no source in this cohort (structural no-source) are "
            "labelled distinctly from measurement missingness. Positive-only "
            "events and conditional event times use their typed semantics."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / stem,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    parent_step_id = str(current_step_id).removesuffix("_figure")
    rendered_summary = {
        "step_id": current_step_id,
        "method": "deterministic_missingness_measurement_figure",
        "analysis_family": "data_quality",
        "rendering_only": True,
        "status": "completed",
        "source_step_id": parent_step_id,
        "source_tables": list(_PRODUCT_FILES.values()),
        "source_data_files": [source_path.name],
        "figure_files": figure_files,
        "figure_path": f"{stem}.png",
        "figure_contract": f"{stem}.figure_contract.json",
        "output_files": {"fig:missingness_measurement": f"{stem}.png"},
        "warnings": [],
        "skipped": [],
        "errors": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(rendered_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return REPAIR_ID


__all__ = [
    "CONTROLLED_METHOD",
    "REPAIR_ID",
    "missingness_source_parent_digest_seal",
    "render_missingness_source_bundle",
]
