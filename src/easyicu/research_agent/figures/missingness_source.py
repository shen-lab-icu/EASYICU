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
from .publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)

REPAIR_ID = "missingness_publication_bundle_from_parent_outputs_v1"
CONTROLLED_METHOD = "missingness_and_source_availability_audit"
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

    number_columns = ("n_total", "n_nonmissing", "missing_n")
    source_number_columns = ("n_total", "measured_one_n", "value_missing_n")
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
    if not (missing["n_nonmissing"] + missing["missing_n"]).equals(missing["n_total"]):
        return None
    if not (source["measured_one_n"] + source["value_missing_n"]).equals(
        source["n_total"]
    ):
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
    if not (
        merged["indicator_semantics"].astype(str) == "measurement_availability"
    ).all():
        return None
    if not (merged["missingness_kind"].astype(str) == "measurement_missing").all():
        return None

    denominator = merged["n_total_missing"].astype(float)
    if (denominator <= 0).any():
        return None
    merged["missing_pct"] = merged["missing_n"].astype(float) * 100.0 / denominator
    merged["available_pct"] = merged["n_nonmissing"].astype(float) * 100.0 / denominator
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
            "available_pct",
            "missing_pct",
            "indicator_semantics",
            "missingness_kind",
        ],
    ].rename(
        columns={
            "variable_missing": "variable",
            "value_column_missing": "value_column",
            "n_total_missing": "n_total",
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
    ax.barh(
        positions,
        available,
        color=palette["blue_soft"],
        height=0.62,
        label="Available",
    )
    ax.barh(
        positions,
        missing,
        left=available,
        color=palette["neutral_light"],
        height=0.62,
        label="Missing",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([_label(value) for value in source["concept"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Analysis cohort (%)")
    ax.set_title("Availability of declared audit inputs", loc="left")
    ax.grid(axis="x", color=palette["neutral_light"], linewidth=0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2)
    for y, (missing_pct, missing_n, total_n) in enumerate(
        zip(missing, source["missing_n"], source["n_total_missing"])
    ):
        label = f"Missing {missing_pct:.1f}% ({int(missing_n):,}/{int(total_n):,})"
        ax.text(101.0, y, label, va="center", ha="left", fontsize=6.2, clip_on=False)
    add_panel_label(ax, "A", x=-0.14, y=1.02)
    fig.subplots_adjust(left=0.20, right=0.76, bottom=0.14, top=0.88)

    stem = "missingness_measurement_panel"
    contract = make_figure_contract(
        figure_id="figure:missingness_measurement",
        core_claim=(
            "Availability and missingness are reported for every Planner-declared "
            "audit input using the locked analysis cohort as denominator."
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
                    "Available and missing counts partition each input's locked "
                    "analysis-cohort denominator."
                ),
                "evidence_ids": [source_path.name],
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "Percentages are recomputed from digest-bound integer counts; this "
            "renderer performs no cohort selection, imputation, or modelling."
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
