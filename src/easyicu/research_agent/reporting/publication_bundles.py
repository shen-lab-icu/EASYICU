"""Publication-bundle renderers projected from prior step outputs.

Extracted from ``pipeline.py`` (byte-preserved bodies). Renderer selection
stays with the pipeline module: this module owns the pure rendering of one
bundle from registered prior outputs, not the decision of which renderer a
step gets.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from ..reporting.readiness import (
    execution_gate_status,
    render_report,
    write_readiness_artifacts,
)

# Back-compat aliases. Tests (and any downstream code) that imported the
# leading-underscore names from this module before the readiness/report
# helpers were moved to ``reporting.readiness`` keep working unchanged.
from ..reporting.readiness import (
    _publication_figure_bundle_ready,  # noqa: F401
    _compute_readiness_gates,  # noqa: F401
    _count_missing_evidence_markers,  # noqa: F401
    _extract_claim_ledger_rows,  # noqa: F401
    _render_author_review_note,  # noqa: F401
)

_execution_gate_status = execution_gate_status  # noqa: F841 (legacy alias)
_write_readiness_artifacts = write_readiness_artifacts  # noqa: F841 (legacy alias)
_render_report = render_report  # noqa: F841 (legacy alias)
from ..audits.manuscript_claims import audit_manuscript_numeric_claims  # noqa: E402
from ..audits.manuscript_claims import (  # noqa: E402,F401
    _first_summary_scalar,
    _extract_metric_claims,
    _extract_percent_claims_near,
)

_audit_manuscript_numeric_claims = audit_manuscript_numeric_claims  # noqa: F841 (legacy alias)

from ..figures.prior_output_support import (
    figure_parent_candidate_step_dirs as _figure_parent_candidate_step_dirs,
    publication_label as _publication_label,
    short_figure_label as _short_figure_label,
)
from ..schema import (
    ResearchContext,
    VariableRole,
)


from ..orchestration.finalize import (
    _concept_dictionary_manifest_fields,  # noqa: F401
    _render_cost_summary,  # noqa: F401
)


def _build_probe_summary(
    *,
    context: ResearchContext,
    cohort_path: Path,
    out_dir: Path,
) -> tuple[Dict[str, Any], List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cohort_path)
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_outcome": context.target_outcome,
        "top_missing_columns": [],
        "score_completeness": [],
    }
    missing_rows = []
    for col in df.columns:
        frac = float(df[col].isna().mean()) if len(df) else 0.0
        missing_rows.append(
            {
                "variable": col,
                "fraction_missing": frac,
                "n_missing": int(df[col].isna().sum()),
                "n_unique_non_missing": int(df[col].dropna().nunique()),
            }
        )
    missing_df = pd.DataFrame(missing_rows).sort_values(
        ["fraction_missing", "variable"], ascending=[False, True]
    )
    summary["top_missing_columns"] = missing_df.head(10).to_dict(orient="records")
    files: List[Path] = []
    missing_path = out_dir / "probe_variable_profile.csv"
    missing_df.to_csv(missing_path, index=False)
    files.append(missing_path)

    from easyicu.io.data_quality import composite_score_completeness

    for variable in context.variables:
        if variable.role not in {
            VariableRole.ORDINAL_SCORE,
            VariableRole.COMPOSITE_SCORE,
        }:
            continue
        if variable.name not in df.columns:
            continue
        observed = df[variable.name].dropna()
        if observed.empty:
            continue
        stats: Dict[str, Any] = {
            "variable": variable.name,
            "min": float(observed.min()),
            "max": float(observed.max()),
            "n_zero": (
                int((observed == 0).sum())
                if pd.api.types.is_numeric_dtype(observed)
                else None
            ),
        }
        n_components_col = f"{variable.name}_n_components"
        if n_components_col in df.columns:
            stats["completeness"] = composite_score_completeness(
                df,
                variable.name,
                n_components_col=n_components_col,
            )
            summary["score_completeness"].append(stats)
    summary_path = out_dir / "probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    files.append(summary_path)
    return summary, files
def _promote_sibling_figure_exports(*, out_dir: Path) -> Optional[str]:
    """Promote figure files written beside ``outputs/`` into ``outputs/``.

    Some generated scripts treat ``STEP_OUT_DIR`` as a filename stem and
    write ``outputs.svg`` / ``outputs.png`` beside the canonical
    ``outputs/`` directory. The execution contract only registers files inside
    ``outputs/``, so normalize that common layout before declaring the
    publication figure missing.
    """
    parent = out_dir.parent
    source_stem = out_dir.name
    figure_suffixes = (".pdf", ".png", ".svg", ".tiff", ".tif", ".pptx")
    figure_sources = [
        parent / f"{source_stem}{suffix}"
        for suffix in figure_suffixes
        if (parent / f"{source_stem}{suffix}").is_file()
    ]
    if not figure_sources:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    target_stem = "publication_figure"
    exported_figure_files: List[str] = []
    for source in figure_sources:
        target = out_dir / f"{target_stem}{source.suffix.lower()}"
        shutil.copy2(source, target)
        exported_figure_files.append(target.name)

    contract_source = parent / f"{source_stem}.figure_contract.json"
    if contract_source.is_file():
        shutil.copy2(contract_source, out_dir / f"{target_stem}.figure_contract.json")

    step_summary_path = out_dir / "step_summary.json"
    summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}
    summary.setdefault("publication_figure_rescue", {})
    summary["publication_figure_rescue"].update(
        {
            "mode": "sibling_outputs_stem",
            "source_stem": source_stem,
            "source_dir": str(parent),
        }
    )
    summary["figure_files"] = sorted(exported_figure_files)
    if exported_figure_files:
        summary["figure_path"] = sorted(exported_figure_files)[0]
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "sibling_figure_exports_promote_v1"
def _promote_prior_publication_bundle(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    required_roles: Optional[Sequence[str]] = None,
    require_declared_sources: bool = False,
) -> Optional[str]:
    """Promote the strongest earlier figure bundle into a publication step."""
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    contract_suffix = ".figure_contract.json"
    best: Optional[tuple[tuple[int, int, int], str, Dict[str, Path]]] = None
    role_filter = {
        str(role).strip().lower()
        for role in (required_roles or [])
        if str(role).strip()
    }

    # A split ``<parent>_figure`` step may only promote exports produced by its
    # direct parent.  If that parent has no figure bundle, copying an unrelated
    # earlier figure is scientifically worse than failing closed (for example,
    # a cohort-flow figure must never satisfy an absolute-risk figure step).
    # Generic terminal publication steps that do not have a sibling parent keep
    # the historical run-wide strongest-bundle behaviour.
    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    direct_parent = steps_dir / parent_step_id
    if parent_step_id != str(current_step_id or "") and direct_parent.is_dir():
        candidate_step_dirs = [direct_parent]
    else:
        candidate_step_dirs = sorted(steps_dir.iterdir())

    for step_dir in candidate_step_dirs:
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        bundles: Dict[str, Dict[str, Path]] = {}
        for path in outputs_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith(contract_suffix):
                stem = path.name[: -len(contract_suffix)]
                bundles.setdefault(stem, {})["contract"] = path
                continue
            if path.suffix.lower() in figure_suffixes:
                bundles.setdefault(path.stem, {})[path.suffix.lower()] = path
        for stem, files in bundles.items():
            figure_count = sum(1 for key in files if key.startswith("."))
            if figure_count == 0:
                continue
            if role_filter and not _publication_bundle_has_any_role(files, role_filter):
                continue
            if (
                require_declared_sources
                and not _publication_bundle_has_resolvable_sources(files)
            ):
                continue
            score = (
                1 if "publication_figure" in stem else 0,
                1 if "primary_association" in stem else 0,
                figure_count,
            )
            if best is None or score > best[0]:
                best = (score, stem, files)

    if best is None:
        return None

    _, source_stem, files = best
    target_stem = "publication_figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, source in files.items():
        if key == "contract":
            target = out_dir / f"{target_stem}.figure_contract.json"
        else:
            target = out_dir / f"{target_stem}{key}"
        shutil.copy2(source, target)

    # A figure contract is not self-contained when its source-data and panel
    # evidence files remain behind in the analysis step.  Promotion previously
    # copied only the rendered exports + JSON contract, leaving a formally
    # untraceable bundle in the split figure step.  Copy every file-like local
    # reference while preserving safe relative names; logical evidence IDs
    # without a file suffix are intentionally left alone.
    copied_trace_files: List[str] = []
    contract_path = files.get("contract")
    if contract_path is not None and contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            contract = {}

        artifact_refs = _publication_contract_file_references(contract)

        source_outputs = contract_path.parent.resolve()
        for ref in dict.fromkeys(artifact_refs):
            relative_ref = Path(ref)
            if relative_ref.is_absolute() or ".." in relative_ref.parts:
                relative_ref = Path(relative_ref.name)
            source = (source_outputs / relative_ref).resolve()
            if not source.is_relative_to(source_outputs) or not source.is_file():
                continue
            target = out_dir / relative_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_trace_files.append(str(relative_ref))

    step_summary_path = out_dir / "step_summary.json"
    summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            summary = json.loads(step_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.setdefault("publication_figure_rescue", {})
    source_outputs_dir = files[next(iter(files))].parent
    source_step_id = source_outputs_dir.parent.name
    summary["publication_figure_rescue"].update(
        {
            "mode": "promotion",
            "source_step_stem": source_stem,
            "source_outputs_dir": str(source_outputs_dir),
            "copied_trace_files": sorted(copied_trace_files),
        }
    )
    exported_figure_files = [
        str((out_dir / f"{target_stem}{key}").name)
        for key in sorted(files)
        if key != "contract"
    ]
    summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_publication_bundle_promotion",
            "rendering_only": True,
            "source_step_id": source_step_id,
            "source_data_files": sorted(copied_trace_files),
        }
    )
    summary["figure_files"] = exported_figure_files
    if exported_figure_files:
        summary["figure_path"] = exported_figure_files[0]
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "publication_bundle_promote_v1"
def _publication_contract_file_references(contract: Any) -> List[str]:
    """Return local file-like source/evidence references from a contract."""

    artifact_refs: List[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            token = value.strip()
            if token and Path(token).suffix:
                artifact_refs.append(token)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _collect(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect(item)

    if isinstance(contract, dict):
        _collect(contract.get("source_data"))
        panels = contract.get("panels") or []
        if isinstance(panels, list):
            for panel in panels:
                if isinstance(panel, dict):
                    _collect(panel.get("evidence_ids"))
    return list(dict.fromkeys(artifact_refs))
def _publication_bundle_has_resolvable_sources(files: Mapping[str, Path]) -> bool:
    """Require every declared file reference to exist beside the parent bundle."""

    contract_path = files.get("contract")
    if contract_path is None or not contract_path.is_file():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    refs = _publication_contract_file_references(contract)
    if not refs:
        return False
    source_outputs = contract_path.parent.resolve()
    for ref in refs:
        relative_ref = Path(ref)
        if relative_ref.is_absolute() or ".." in relative_ref.parts:
            relative_ref = Path(relative_ref.name)
        source = (source_outputs / relative_ref).resolve()
        if not source.is_relative_to(source_outputs) or not source.is_file():
            return False
    return True
def _publication_bundle_has_any_role(
    files: Mapping[str, Path], required_roles: set[str]
) -> bool:
    contract_path = files.get("contract")
    if contract_path is None or not contract_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(contract, dict):
        return False
    roles: set[str] = set()
    panels = contract.get("panels") or []
    if isinstance(panels, list):
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            role = str(panel.get("role") or "").strip().lower()
            if role:
                roles.add(role)
    top_role = str(contract.get("role") or "").strip().lower()
    if top_role:
        roles.add(top_role)
    return bool(roles & required_roles)
def _render_prediction_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically build a validation figure from prior prediction outputs.

    Some small models successfully write ``model_performance.csv`` and
    ``step_summary.json`` in the parent model-training step but fail to
    render the follow-up figure step. When that happens, we can still
    construct a publication-style validation bundle from the structured
    parent artefacts instead of failing the entire run.
    """
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    best_parent: Optional[tuple[Path, Path, Dict[str, Any]]] = None
    candidate_step_dirs, _direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        outputs_dir = step_dir / "outputs"
        perf_path = outputs_dir / "model_performance.csv"
        summary_path = outputs_dir / "step_summary.json"
        if not perf_path.exists() or not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        best_parent = (perf_path, summary_path, summary)
        break
    if best_parent is None:
        return None

    perf_path, summary_path, summary = best_parent
    try:
        frame = pd.read_csv(perf_path)
    except Exception:
        return None
    metric_cols = [col for col in ("auroc", "brier_score") if col in frame.columns]
    calib_cols = [
        col
        for col in ("calibration_slope", "calibration_intercept")
        if col in frame.columns
    ]
    if not metric_cols and not calib_cols:
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
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True
    )
    apply_publication_style(fig)
    if not isinstance(axes, (list, tuple)):
        axes = axes.ravel()
    folds = frame.get("fold")
    if folds is None:
        folds = pd.Series([f"Fold {idx + 1}" for idx in range(len(frame))])
    folds = folds.astype(str)

    ax1, ax2 = axes[0], axes[1]
    if "auroc" in frame.columns:
        ax1.plot(
            folds,
            frame["auroc"].astype(float),
            marker="o",
            linewidth=1.4,
            label="AUROC",
        )
    if "brier_score" in frame.columns:
        ax1.plot(
            folds,
            frame["brier_score"].astype(float),
            marker="s",
            linewidth=1.2,
            label="Brier",
        )
    ax1.set_title("Cross-validation discrimination", loc="left", pad=4)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Metric value")
    ax1.tick_params(axis="x", rotation=35)
    ax1.legend(frameon=False, fontsize=7)
    add_panel_label(ax1, "A", x=-0.1)

    if "calibration_slope" in frame.columns:
        ax2.plot(
            folds,
            frame["calibration_slope"].astype(float),
            marker="o",
            linewidth=1.4,
            label="Slope",
        )
        ax2.axhline(1.0, linestyle="--", linewidth=0.8, color="#8F8F8F")
    if "calibration_intercept" in frame.columns:
        ax2.plot(
            folds,
            frame["calibration_intercept"].astype(float),
            marker="s",
            linewidth=1.2,
            label="Intercept",
        )
        ax2.axhline(0.0, linestyle=":", linewidth=0.8, color="#B64342")
    ax2.set_title("Cross-validation calibration", loc="left", pad=4)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Calibration statistic")
    ax2.tick_params(axis="x", rotation=35)
    if calib_cols:
        ax2.legend(frameon=False, fontsize=7)
    add_panel_label(ax2, "B", x=-0.1)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "Prediction-model validation metrics are summarised from the "
            "registered cross-validation performance table."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Discrimination",
                "role": "validation",
                "claim": "Fold-level AUROC and Brier score are derived from the model-performance table.",
                "evidence_ids": ["model_performance", "01_model_training"],
            },
            {
                "panel_id": "B",
                "title": "Calibration",
                "role": "validation",
                "claim": "Fold-level calibration slope and intercept are derived from the registered step summary and performance table.",
                "evidence_ids": ["model_performance", "01_model_training"],
            },
        ],
        source_data=["model_performance", "01_model_training"],
        statistics_note=(
            "Deterministic rescue figure generated from parent-step outputs "
            "when the figure-only child step did not emit exports."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
        contract=contract,
        dpi=300,
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
    existing_summary.setdefault("publication_figure_rescue", {})
    existing_summary["publication_figure_rescue"].update(
        {
            "mode": "prediction_validation_from_parent_outputs",
            "source_model_performance": str(perf_path),
            "source_step_summary": str(summary_path),
        }
    )
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    existing_summary["figure_files"] = figure_files
    if figure_files:
        existing_summary["figure_path"] = figure_files[0]
    existing_summary.setdefault("cv_auroc_mean", summary.get("statistic:auroc"))
    existing_summary.setdefault("cv_brier_mean", summary.get("statistic:brier_score"))
    existing_summary.setdefault(
        "calibration_slope", summary.get("statistic:calibration_slope")
    )
    existing_summary.setdefault(
        "calibration_intercept", summary.get("statistic:calibration_intercept")
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "prediction_publication_bundle_from_parent_outputs_v1"
def _render_cohort_overlap_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically build a cohort-definition overlap figure.

    Cohort eligibility and overlap steps do not emit OR/CI tables, so they
    should never fall through to the generic association forest rescue. This
    renderer consumes the immediate parent step's attrition and overlap tables
    and writes traceable source-data copies keyed by cohort-definition ids.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    parent_step_id = current_step_id.removesuffix("_figure")
    parent_outputs = steps_dir / parent_step_id / "outputs"
    attrition_path = parent_outputs / "alternative_cohort_attrition.csv"
    overlap_path = parent_outputs / "cohort_overlap_matrix.csv"
    audit_path = parent_outputs / "cohort_definition_empirical_equivalence_audit.csv"
    if not attrition_path.exists() or not overlap_path.exists():
        return None

    try:
        attrition = pd.read_csv(attrition_path)
        overlap = pd.read_csv(overlap_path)
    except Exception:
        return None

    attrition_required = {
        "definition_id",
        "definition_label",
        "definition_type",
        "n_included",
        "included_pct_of_rows",
        "overlap_with_primary_pct_of_primary",
        "moved_in_vs_primary_n",
        "moved_out_vs_primary_n",
    }
    overlap_required = {"definition_a", "definition_b", "jaccard"}
    if not attrition_required <= set(attrition.columns):
        return None
    if not overlap_required <= set(overlap.columns):
        return None
    if attrition.empty or overlap.empty:
        return None

    source_attrition = attrition.copy()
    for col in (
        "n_included",
        "n_excluded",
        "included_pct_of_rows",
        "overlap_with_primary_n",
        "overlap_with_primary_pct_of_primary",
        "overlap_with_primary_pct_of_definition",
        "moved_in_vs_primary_n",
        "moved_out_vs_primary_n",
    ):
        if col in source_attrition.columns:
            source_attrition[col] = pd.to_numeric(
                source_attrition[col], errors="coerce"
            )

    def _cohort_definition_display_label(row: Mapping[str, Any]) -> str:
        definition_id = str(row.get("definition_id") or "").strip()
        known = {
            "primary_adult_los1_all_vitals_sepsis3_derivable": "Primary",
            "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
            "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
            "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
            "alt_adult_los2_all_vitals_sepsis3_derivable": "LOS >=2 d",
            "primary_adult_los1_all_vitals_sep3_measured": "Primary",
            "alt_adult_no_los_all_vitals_sep3_measured": "No LOS threshold",
            "alt_adult_los1_three_of_four_vitals_sep3_measured": ">=3 of 4 vitals",
            "alt_adult_los1_no_temp_requirement_sep3_measured": "No temperature",
            "alt_adult_los2_all_vitals_sep3_measured": "LOS >=2 d",
        }
        if definition_id in known:
            return known[definition_id]
        label = str(row.get("definition_label") or definition_id or "").strip()
        return label or "Definition"

    source_attrition["display_label"] = [
        _cohort_definition_display_label(row)
        for row in source_attrition.to_dict(orient="records")
    ]

    source_overlap = overlap.copy()
    for col in (
        "n_a",
        "n_b",
        "intersection_n",
        "union_n",
        "jaccard",
        "a_in_b_pct",
        "b_in_a_pct",
    ):
        if col in source_overlap.columns:
            source_overlap[col] = pd.to_numeric(source_overlap[col], errors="coerce")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_attrition_path = out_dir / "publication_figure_definition_source_data.csv"
    source_overlap_path = out_dir / "publication_figure_overlap_source_data.csv"
    source_attrition.to_csv(source_attrition_path, index=False)
    source_overlap.to_csv(source_overlap_path, index=False)

    plot_df = source_attrition.reset_index(drop=True).copy()
    labels = plot_df["display_label"].astype(str).tolist()
    y = list(range(len(plot_df)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    fig = plt.figure(figsize=(183 / 25.4, 132 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.0, 0.88],
        left=0.18,
        right=0.98,
        top=0.92,
        bottom=0.16,
        wspace=0.45,
        hspace=0.58,
    )
    ax_n = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[:, 1])

    colors = [
        (
            palette.get("blue", "#0F4D92")
            if str(row.get("definition_type", "")).lower() == "primary"
            else palette.get("teal", "#42949E")
        )
        for row in plot_df.to_dict(orient="records")
    ]
    ax_n.barh(y, plot_df["n_included"].astype(float), color=colors, height=0.58)
    ax_n.set_yticks(y)
    ax_n.set_yticklabels(labels)
    ax_n.invert_yaxis()
    ax_n.set_xlabel("Included ICU stays")
    ax_n.set_title("Eligibility definitions", loc="left", pad=4)
    ax_n.grid(axis="x", color=palette.get("neutral_light", "#D8D8D8"), linewidth=0.55)
    add_panel_label(ax_n, "A", x=-0.20)

    moved_in = plot_df["moved_in_vs_primary_n"].astype(float)
    moved_out = -plot_df["moved_out_vs_primary_n"].astype(float)
    ax_delta.barh(
        y,
        moved_in,
        color=palette.get("green", "#008B5E"),
        height=0.36,
        label="Added vs primary",
    )
    ax_delta.barh(
        y,
        moved_out,
        color=palette.get("orange", "#E28E2C"),
        height=0.36,
        label="Removed vs primary",
    )
    ax_delta.axvline(0, color=palette.get("neutral", "#8F8F8F"), linewidth=0.8)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(labels)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("ICU-stay count change")
    ax_delta.set_title("Movement relative to primary", loc="left", pad=4)
    ax_delta.grid(
        axis="x", color=palette.get("neutral_light", "#D8D8D8"), linewidth=0.55
    )
    ax_delta.legend(
        frameon=False,
        fontsize=6.2,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.36),
        ncol=2,
    )
    add_panel_label(ax_delta, "B", x=-0.20)

    definition_order = plot_df["definition_id"].astype(str).tolist()
    label_map = dict(zip(definition_order, labels))
    heat = (
        source_overlap.pivot_table(
            index="definition_a",
            columns="definition_b",
            values="jaccard",
            aggfunc="first",
        )
        .reindex(index=definition_order, columns=definition_order)
        .astype(float)
    )
    image = ax_heat.imshow(
        heat.to_numpy() * 100.0,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    ax_heat.set_xticks(range(len(definition_order)))
    ax_heat.set_xticklabels(
        [
            _short_figure_label(label_map.get(item, item), limit=18)
            for item in definition_order
        ],
        rotation=45,
        ha="right",
    )
    ax_heat.set_yticks(range(len(definition_order)))
    ax_heat.set_yticklabels(
        [
            _short_figure_label(label_map.get(item, item), limit=18)
            for item in definition_order
        ]
    )
    ax_heat.set_title("Pairwise cohort overlap", loc="left", pad=4)
    for row_idx in range(len(definition_order)):
        for col_idx in range(len(definition_order)):
            value = heat.iat[row_idx, col_idx]
            if pd.isna(value):
                continue
            ax_heat.text(
                col_idx,
                row_idx,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="#1F1F1F" if value < 0.72 else "white",
            )
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label("Jaccard overlap (%)")
    add_panel_label(ax_heat, "C", x=-0.12)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "Alternative eligibility definitions change the cohort denominator "
            "and overlap structure, which must be visible before interpreting "
            "model sensitivity."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Eligibility denominators",
                "role": "overview",
                "claim": "Included ICU-stay counts are read from the parent attrition table.",
                "evidence_ids": ["alternative_cohort_attrition"],
            },
            {
                "panel_id": "B",
                "title": "Movement relative to primary",
                "role": "audit",
                "claim": "Each alternative definition's added and removed stays are explicit.",
                "evidence_ids": ["alternative_cohort_attrition"],
            },
            {
                "panel_id": "C",
                "title": "Pairwise overlap",
                "role": "robustness",
                "claim": "Jaccard overlap is computed from the parent overlap matrix.",
                "evidence_ids": ["cohort_overlap_matrix"],
            },
        ],
        source_data=[
            "alternative_cohort_attrition",
            "cohort_overlap_matrix",
            "publication_figure_definition_source_data.csv",
            "publication_figure_overlap_source_data.csv",
        ],
        statistics_note=(
            "Generated deterministically from the parent cohort-definition "
            "attrition and overlap tables; no values are inferred from the image."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
        contract=contract,
        dpi=300,
    )
    plt.close(fig)

    step_summary_path = out_dir / "step_summary.json"
    existing_summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_cohort_overlap_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_attrition_table": str(attrition_path),
            "source_overlap_table": str(overlap_path),
            "source_equivalence_audit": (
                str(audit_path) if audit_path.exists() else None
            ),
            "source_data_files": [
                source_attrition_path.name,
                source_overlap_path.name,
            ],
            "n_definitions": int(len(plot_df)),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "publication_figure.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "cohort_overlap_publication_bundle_from_parent_outputs_v1"
def _render_cohort_flow_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
) -> Optional[str]:
    """Deterministically render a simple sequential cohort-flow contract.

    This is deliberately narrower than the cohort-definition overlap renderer:
    it accepts only a parent ``cohort_flow.csv`` plus ``attrition.csv`` with
    explicit stages, denominators, removals, and exclusion categories.  The
    overlap renderer remains the first choice for multi-definition analyses.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    parent_step_id = current_step_id.removesuffix("_figure")
    parent_outputs = steps_dir / parent_step_id / "outputs"
    flow_path = parent_outputs / "cohort_flow.csv"
    attrition_path = parent_outputs / "attrition.csv"
    flow_payload = (
        preverified_parent_artifacts.get("cohort_flow.csv")
        if preverified_parent_artifacts is not None
        else None
    )
    attrition_payload = (
        preverified_parent_artifacts.get("attrition.csv")
        if preverified_parent_artifacts is not None
        else None
    )
    if preverified_parent_artifacts is None and (
        not flow_path.exists() or not attrition_path.exists()
    ):
        return None
    if preverified_parent_artifacts is not None and (
        flow_payload is None
        or attrition_payload is None
        or "step_summary.json" not in preverified_parent_artifacts
    ):
        return None

    try:
        flow = pd.read_csv(
            io.BytesIO(flow_payload) if flow_payload is not None else flow_path
        )
        attrition = pd.read_csv(
            io.BytesIO(attrition_payload)
            if attrition_payload is not None
            else attrition_path
        )
    except Exception:
        return None

    flow_required = {
        "stage",
        "n",
        "percent_of_universe",
        "n_removed_from_prior_stage",
        "criterion",
    }
    attrition_required = {
        "attrition_category",
        "n",
        "percent_of_universe",
        "status",
        "reason",
        "partition_role",
    }
    if not flow_required <= set(flow.columns):
        return None
    if not attrition_required <= set(attrition.columns):
        return None
    if flow.empty or attrition.empty:
        return None

    flow_plot = flow.copy()
    attrition_plot = attrition.copy()
    for frame, numeric_columns in (
        (
            flow_plot,
            ("n", "percent_of_universe", "n_removed_from_prior_stage"),
        ),
        (attrition_plot, ("n", "percent_of_universe")),
    ):
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any() or (~frame[column].map(math.isfinite)).any():
                return None
            if (frame[column] < 0).any():
                return None

    if flow_plot["stage"].fillna("").astype(str).str.strip().eq("").any():
        return None
    if (
        attrition_plot["attrition_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    source_flow = flow.copy()
    source_flow["source_table"] = flow_path.name
    source_attrition = attrition.copy()
    source_attrition["source_table"] = attrition_path.name
    source_flow_path = out_dir / "publication_figure_source_data.csv"
    source_attrition_path = out_dir / "publication_figure_attrition_source_data.csv"
    source_flow.to_csv(source_flow_path, index=False)
    source_attrition.to_csv(source_attrition_path, index=False)

    excluded = attrition_plot[
        attrition_plot["status"].fillna("").astype(str).str.lower().eq("excluded")
    ].copy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    height_mm = max(112.0, min(170.0, 70.0 + 13.0 * len(flow_plot)))
    fig, (ax_flow, ax_attrition) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, height_mm / 25.4),
        gridspec_kw={"width_ratios": [1.12, 0.88]},
        constrained_layout=True,
    )

    n_stages = len(flow_plot)
    if n_stages == 1:
        y_positions = [0.50]
    else:
        y_positions = [
            0.90 - index * 0.78 / (n_stages - 1) for index in range(n_stages)
        ]
    box_height = min(0.13, 0.52 / max(n_stages, 1))
    flow_rows = flow_plot.reset_index(drop=True).to_dict(orient="records")
    for index, (row, y_pos) in enumerate(zip(flow_rows, y_positions)):
        stage_label = str(row["stage"]).replace("_", " ").strip().title()
        stage_label = _short_figure_label(stage_label, limit=34)
        count = int(round(float(row["n"])))
        percent = float(row["percent_of_universe"])
        facecolor = (
            palette.get("blue", "#0F4D92")
            if index in (0, n_stages - 1)
            else palette.get("teal", "#42949E")
        )
        ax_flow.text(
            0.46,
            y_pos,
            f"{stage_label}\n{count:,} ({percent:.1f}% of universe)",
            ha="center",
            va="center",
            fontsize=7.0,
            color="white",
            transform=ax_flow.transAxes,
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": facecolor,
                "edgecolor": "none",
            },
        )
        if index + 1 >= n_stages:
            continue
        next_y = y_positions[index + 1]
        ax_flow.annotate(
            "",
            xy=(0.46, next_y + box_height / 2),
            xytext=(0.46, y_pos - box_height / 2),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": palette.get("neutral", "#8F8F8F"),
                "linewidth": 0.9,
            },
        )
        removed = int(round(float(flow_rows[index + 1]["n_removed_from_prior_stage"])))
        ax_flow.text(
            0.62,
            (y_pos + next_y) / 2,
            f"Removed: {removed:,}",
            ha="left",
            va="center",
            fontsize=6.3,
            color=palette.get("neutral_dark", "#4A4A4A"),
            transform=ax_flow.transAxes,
        )
    ax_flow.set_title("Registered eligibility sequence", loc="left", pad=4)
    ax_flow.set_axis_off()
    add_panel_label(ax_flow, "A", x=-0.04)

    if excluded.empty:
        ax_attrition.text(
            0.5,
            0.5,
            "No excluded categories were registered",
            ha="center",
            va="center",
            transform=ax_attrition.transAxes,
            fontsize=7.0,
        )
        ax_attrition.set_xticks([])
        ax_attrition.set_yticks([])
    else:
        excluded = excluded.reset_index(drop=True)
        y = list(range(len(excluded)))
        values = excluded["n"].astype(float)
        labels = [
            _short_figure_label(
                str(value).replace("_", " ").strip().title(),
                limit=28,
            )
            for value in excluded["attrition_category"]
        ]
        ax_attrition.barh(
            y,
            values,
            color=palette.get("orange", "#E28E2C"),
            height=0.58,
        )
        ax_attrition.set_yticks(y)
        ax_attrition.set_yticklabels(labels)
        ax_attrition.invert_yaxis()
        max_count = float(values.max())
        ax_attrition.set_xlim(0, max(1.0, max_count * 1.28))
        for index, row in excluded.iterrows():
            count = int(round(float(row["n"])))
            percent = float(row["percent_of_universe"])
            x_pos = max(float(row["n"]), max(max_count, 1.0) * 0.015)
            ax_attrition.text(
                x_pos,
                index,
                f" {count:,} ({percent:.1f}%)",
                ha="left",
                va="center",
                fontsize=6.2,
            )
        ax_attrition.set_xlabel("Excluded records")
        ax_attrition.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
        )
    ax_attrition.set_title("Recorded attrition", loc="left", pad=4)
    add_panel_label(ax_attrition, "B", x=-0.16)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "The registered eligibility sequence defines the analysis cohort "
            "and explicitly accounts for exclusions from the supplied study universe."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Eligibility sequence",
                "role": "overview",
                "claim": (
                    "Stage-specific denominators and removals are read from the "
                    "registered cohort-flow table."
                ),
                "evidence_ids": ["cohort_flow"],
                "metadata": {"planner_product_slots": ["cohort_flow"]},
            },
            {
                "panel_id": "B",
                "title": "Attrition accounting",
                "role": "audit",
                "claim": (
                    "Explicit exclusion categories and percentages are read from "
                    "the registered attrition table."
                ),
                "evidence_ids": ["attrition"],
                "metadata": {"planner_product_slots": ["attrition_audit"]},
            },
        ],
        height_mm=height_mm,
        source_data=[
            "cohort_flow",
            "attrition",
            source_flow_path.name,
            source_attrition_path.name,
        ],
        statistics_note=(
            "Counts, percentages, and stage removals are rendered directly from "
            "the parent cohort-flow and attrition tables; no values are inferred "
            "from the image."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
        contract=contract,
        dpi=300,
    )
    plt.close(fig)

    step_summary_path = out_dir / "step_summary.json"
    existing_summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    if (
        existing_summary.get("deterministic_publication_figure_rescue")
        == "no_parent_outputs"
    ):
        existing_summary.pop("warning", None)
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_cohort_flow_publication_figure_repair",
            "rendering_only": True,
            "deterministic_publication_figure_rescue": (
                "cohort_flow_publication_bundle_from_parent_outputs_v1"
            ),
            "source_step_id": parent_step_id,
            "source_cohort_flow_table": str(flow_path),
            "source_attrition_table": str(attrition_path),
            "source_data_files": [
                source_flow_path.name,
                source_attrition_path.name,
            ],
            "n_flow_stages": int(len(flow_plot)),
            "n_exclusion_categories": int(len(excluded)),
            "figure_files": figure_files,
            "figure_path": "publication_figure.png",
            "figure_contract": "publication_figure.figure_contract.json",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "cohort_flow_publication_bundle_from_parent_outputs_v1"
def _render_phenotype_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically rebuild a phenotyping figure from agent-produced,
    source-backed standardized clustering products.

    Reads ``outcome_by_cluster.csv`` (the descriptive outcome contrast, which
    carries ``ci_low``/``ci_high`` -- validator-checked value columns) as the
    primary traceable source, falling back to ``cluster_sizes.csv``. Renders a
    two-panel figure (cluster sizes + descriptive outcome-by-cluster with CIs) and
    emits a validator-conformant ``*_source_data.csv`` traced positionally via
    ``source_row_index`` (like the prediction/association renderers). Returns
    ``None`` when no cluster table with >= 2 clusters is found, so a run that never
    produced a partition falls through cleanly rather than emitting an empty
    figure. Descriptive by construction -- no OR/HR is drawn or claimed.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_paths: List[Path] = []
    candidate_step_dirs, direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        text = step_dir.name.lower()
        if not direct_parent_only and not any(
            token in text
            for token in ("cluster", "phenotype", "subphenotype", "trajectory")
        ):
            continue
        outputs_dir = step_dir / "outputs"
        if outputs_dir.exists():
            candidate_paths.extend(sorted(outputs_dir.glob("*.csv")))

    def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None

    # Prefer the outcome-by-cluster table (traceable ci_low/ci_high); else sizes.
    outcome: Optional[tuple[Path, pd.DataFrame]] = None
    sizes: Optional[tuple[Path, pd.DataFrame]] = None
    for csv_path in candidate_paths:
        name = csv_path.name.lower()
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        cluster_col = _first_col(frame, ("cluster", "cluster_id", "phenotype", "label"))
        if cluster_col is None:
            continue
        if outcome is None and "outcome_by_cluster" in name:
            outcome = (csv_path, frame)
        if sizes is None and (
            "cluster_sizes" in name or "size" in frame.columns or "n" in frame.columns
        ):
            sizes = (csv_path, frame)
    primary = outcome or sizes
    if primary is None:
        return None
    table_path, frame = primary
    cluster_col = _first_col(frame, ("cluster", "cluster_id", "phenotype", "label"))
    if cluster_col is None or frame[cluster_col].nunique() < 2:
        return None

    # Positional trace: keep the ORIGINAL row order so source_row_index maps 1:1
    # into the upstream table the validator re-reads.
    plot_df = frame.reset_index(drop=True)
    plot_df.insert(0, "source_row_index", plot_df.index.astype(int))
    n_col = _first_col(plot_df, ("n", "n_stays", "cluster_size", "size", "count"))
    rate_col = _first_col(
        plot_df, ("mortality_rate", "outcome_rate", "event_rate", "rate")
    )
    ci_low_col = _first_col(plot_df, ("ci_low", "ci_lower", "lower"))
    ci_high_col = _first_col(plot_df, ("ci_high", "ci_upper", "upper"))

    clusters = plot_df[cluster_col].astype(str)
    source_payload: Dict[str, Any] = {
        "source_row_index": plot_df["source_row_index"].astype(int),
        # The cluster label is carried under a NON-key column name so the validator
        # traces POSITIONALLY (source_row_index) and value-checks ci_low/ci_high,
        # rather than joining on a key column that is either unshared or non-unique.
        "cluster_label": clusters,
        "source_table": table_path.name,
        "source_transform": "phenotype_cluster_outcome_summary_v1",
    }
    if n_col is not None:
        source_payload["n"] = pd.to_numeric(plot_df[n_col], errors="coerce")
    if rate_col is not None:
        source_payload["mortality_rate"] = pd.to_numeric(
            plot_df[rate_col], errors="coerce"
        )
    if ci_low_col is not None:
        source_payload["ci_low"] = pd.to_numeric(plot_df[ci_low_col], errors="coerce")
    if ci_high_col is not None:
        source_payload["ci_high"] = pd.to_numeric(plot_df[ci_high_col], errors="coerce")
    source_data = pd.DataFrame(source_payload)
    if source_data.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(out_dir / "phenotype_cluster_panel_source_data.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    labels = [f"C{c}" for c in clusters.tolist()]
    x = list(range(len(labels)))
    has_outcome = rate_col is not None
    ncols = 2 if (n_col is not None and has_outcome) else 1
    fig = plt.figure(figsize=(183 / 25.4, 92 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1, ncols, left=0.12, right=0.97, top=0.88, bottom=0.18, wspace=0.42
    )
    col = 0
    if n_col is not None:
        ax_size = fig.add_subplot(grid[0, col])
        sizes_v = pd.to_numeric(plot_df[n_col], errors="coerce").fillna(0).to_numpy()
        ax_size.bar(x, sizes_v, color=palette.get("blue", "#0F4D92"), width=0.62)
        ax_size.set_xticks(x, labels, fontsize=6.2)
        ax_size.set_ylabel("Cluster size (n)")
        ax_size.set_title("Cluster sizes", loc="left", pad=4)
        add_panel_label(ax_size, "A", x=-0.12)
        col += 1
    if has_outcome:
        ax_out = fig.add_subplot(grid[0, col])
        rate = pd.to_numeric(plot_df[rate_col], errors="coerce").fillna(0).to_numpy()
        # Scale a 0-1 proportion to percent for display; leave an already-percent
        # column untouched.
        scale = 100.0 if float(np.nanmax(rate)) <= 1.0 else 1.0
        rate_pct = rate * scale
        yerr = None
        if ci_low_col is not None and ci_high_col is not None:
            lo = (
                pd.to_numeric(plot_df[ci_low_col], errors="coerce").fillna(0).to_numpy()
                * scale
            )
            hi = (
                pd.to_numeric(plot_df[ci_high_col], errors="coerce")
                .fillna(0)
                .to_numpy()
                * scale
            )
            yerr = np.vstack(
                [np.clip(rate_pct - lo, 0, None), np.clip(hi - rate_pct, 0, None)]
            )
        ax_out.bar(
            x,
            rate_pct,
            yerr=yerr,
            color=palette.get("red", "#B2182B"),
            width=0.62,
            capsize=3,
        )
        ax_out.set_xticks(x, labels, fontsize=6.2)
        ax_out.set_ylabel("Outcome rate (%)")
        ax_out.set_title("Outcome by cluster (descriptive)", loc="left", pad=4)
        add_panel_label(ax_out, "B" if n_col is not None else "A", x=-0.12)

    contract = make_figure_contract(
        figure_id="phenotype_cluster_panel",
        core_claim=(
            "Discovered phenotype clusters are shown by size and a DESCRIPTIVE "
            "outcome-by-cluster comparison, rendered from the agent-produced, "
            "source-backed clustering products (no causal claim)."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Cluster sizes",
                "role": "phenotype_structure",
                "claim": "Cluster sizes come directly from the declared parent table.",
                "evidence_ids": ["phenotype_cluster_panel_source_data.csv"],
            },
            {
                "panel_id": "B",
                "title": "Outcome by cluster",
                "role": "phenotype_outcome",
                "claim": (
                    "Outcome rates and confidence intervals are copied from the "
                    "agent's descriptive outcome-by-cluster product; comparison is "
                    "descriptive, explicitly not causal."
                ),
                "evidence_ids": ["phenotype_cluster_panel_source_data.csv"],
            },
        ],
        source_data=["phenotype_cluster_panel_source_data.csv"],
        statistics_note=(
            "Rendered deterministically from agent-produced standardized "
            "clustering products; outcome-by-cluster is descriptive (no adjusted "
            "effect)."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "phenotype_cluster_panel", contract=contract, dpi=300
    )
    plt.close(fig)

    step_summary_path = out_dir / "step_summary.json"
    existing_summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_phenotype_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_cluster_table": str(table_path),
            "source_data_csv": str(out_dir / "phenotype_cluster_panel_source_data.csv"),
            "n_clusters_plotted": int(frame[cluster_col].nunique()),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "phenotype_cluster_panel.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "phenotype_publication_bundle_from_parent_outputs_v1"
_TABLE_ONE_ROWTYPE_COLS = ("row_type", "summary_type", "variable_class")
_TABLE_ONE_VALUE_TOKENS = ("median", "mean", "percentage", "count", "q25", "q75")
_RESULT_TABLE_COLS = (
    "odds_ratio",
    "hazard_ratio",
    "risk_ratio",
    "estimate",
    "point_estimate",
    "coef",
    "auroc",
)
def _render_descriptive_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically rebuild a descriptive baseline / table-one figure.

    Fires ONLY for a genuine table-one summary (a ``variable`` key column, a
    row-type column, and per-group median/percentage cells) and returns ``None``
    for anything else -- an association/result table (has an odds_ratio/estimate
    column) is left to its own renderer, and a run without a descriptive table
    falls through cleanly. Renders continuous (median) and categorical (percent)
    baseline summaries and emits a validator-conformant ``*_source_data.csv``
    traced positionally via ``source_row_index`` into the parent table.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_paths: List[Path] = []
    candidate_step_dirs, direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        text = step_dir.name.lower()
        if not direct_parent_only and not any(
            token in text
            for token in (
                "baseline",
                "table_one",
                "table1",
                "descriptive",
                "characteristic",
            )
        ):
            continue
        outputs_dir = step_dir / "outputs"
        if outputs_dir.exists():
            candidate_paths.extend(sorted(outputs_dir.glob("*.csv")))

    def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None

    def _is_table_one(frame: pd.DataFrame) -> bool:
        cols = {str(c).lower() for c in frame.columns}
        if "variable" not in cols:
            return False
        # A result/effect table is NOT a table one.
        if any(c in cols for c in _RESULT_TABLE_COLS):
            return False
        if not any(c in cols for c in _TABLE_ONE_ROWTYPE_COLS):
            return False
        return any(any(tok in c for c in cols) for tok in _TABLE_ONE_VALUE_TOKENS)

    parent: Optional[tuple[Path, pd.DataFrame]] = None
    for csv_path in candidate_paths:
        name = csv_path.name.lower()
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if not _is_table_one(frame):
            continue
        parent = (csv_path, frame)
        if "table_one" in name or "baseline" in name or "characteristic" in name:
            break
    if parent is None:
        return None

    table_path, frame = parent
    frame = frame.reset_index(drop=True)
    row_type_col = _first_col(frame, _TABLE_ONE_ROWTYPE_COLS)
    label_col = _first_col(frame, ("label", "variable"))
    category_col = _first_col(frame, ("category",))
    median_col = _first_col(frame, ("overall_median", "median"))
    pct_col = _first_col(frame, ("overall_percentage", "percentage"))
    if label_col is None or (median_col is None and pct_col is None):
        return None

    # Positional trace: keep original row order; source_row_index maps 1:1 into the
    # upstream table the validator re-reads.
    rows: List[Dict[str, Any]] = []
    for idx, row in frame.iterrows():
        rtype = str(row.get(row_type_col, "") if row_type_col else "").lower()
        label = str(row.get(label_col, "")).strip()
        cat = str(row.get(category_col, "")).strip() if category_col else ""
        median_v = (
            pd.to_numeric(pd.Series([row.get(median_col)]), errors="coerce").iloc[0]
            if median_col
            else float("nan")
        )
        pct_v = (
            pd.to_numeric(pd.Series([row.get(pct_col)]), errors="coerce").iloc[0]
            if pct_col
            else float("nan")
        )
        is_cont = ("continuous" in rtype) or (pd.notna(median_v) and pd.isna(pct_v))
        display = label if not cat or cat.lower() == "nan" else f"{label} ({cat})"
        rows.append(
            {
                # ``variable``/``category`` are _KEY_COLUMNS and non-unique in a
                # table one (e.g. sex -> Female + Male); carrying them under NON-key
                # names forces the validator to trace POSITIONALLY via
                # source_row_index rather than an ambiguous named-key join that
                # false-flags disagreement.
                "source_row_index": int(idx),
                "variable_name": str(row.get("variable", label)),
                "row_category": cat,
                "display_label": display,
                "is_continuous": bool(is_cont),
                "overall_median": (
                    float(median_v) if pd.notna(median_v) else float("nan")
                ),
                "overall_percentage": float(pct_v) if pd.notna(pct_v) else float("nan"),
                "source_table": table_path.name,
                "source_transform": "table_one_baseline_summary_v1",
            }
        )
    source_data = pd.DataFrame(rows)
    cont_rows = source_data[
        source_data["is_continuous"] & source_data["overall_median"].notna()
    ].head(12)
    cat_rows = source_data[
        (~source_data["is_continuous"]) & source_data["overall_percentage"].notna()
    ].head(12)
    if cont_rows.empty and cat_rows.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = pd.concat([cont_rows, cat_rows]).sort_values("source_row_index")
    keep.to_csv(out_dir / "baseline_table_one_source_data.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    ncols = int(not cont_rows.empty) + int(not cat_rows.empty)
    fig = plt.figure(figsize=(183 / 25.4, 104 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1, max(1, ncols), left=0.30, right=0.97, top=0.88, bottom=0.14, wspace=0.55
    )
    col = 0
    if not cont_rows.empty:
        ax = fig.add_subplot(grid[0, col])
        y = list(range(len(cont_rows)))
        ax.barh(
            y,
            cont_rows["overall_median"].to_numpy(),
            color=palette.get("blue", "#0F4D92"),
            height=0.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_figure_label(v, limit=28) for v in cont_rows["display_label"]]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Median (overall)")
        ax.set_title("Continuous characteristics", loc="left", pad=4)
        add_panel_label(ax, "A", x=0.0, y=1.06)
        col += 1
    if not cat_rows.empty:
        ax = fig.add_subplot(grid[0, col])
        y = list(range(len(cat_rows)))
        ax.barh(
            y,
            cat_rows["overall_percentage"].clip(0, 100).to_numpy(),
            color=palette.get("green", "#2E7D32"),
            height=0.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_figure_label(v, limit=28) for v in cat_rows["display_label"]]
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage (overall)")
        ax.set_title("Categorical characteristics", loc="left", pad=4)
        add_panel_label(ax, "B" if not cont_rows.empty else "A", x=0.0, y=1.06)

    contract = make_figure_contract(
        figure_id="baseline_table_one",
        core_claim=(
            "Baseline cohort characteristics are shown directly from the "
            "registered descriptive table-one summary."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Baseline characteristics",
                "role": "descriptive",
                "claim": (
                    "Median and percentage summaries are copied from the parent "
                    "table-one; no effect is estimated."
                ),
                "evidence_ids": ["table_one"],
            }
        ],
        source_data=["table_one"],
        statistics_note=(
            "Generated deterministically from the registered parent-step "
            "table-one; descriptive only (no adjusted effect)."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "baseline_table_one", contract=contract, dpi=300
    )
    plt.close(fig)

    step_summary_path = out_dir / "step_summary.json"
    existing_summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_descriptive_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_table_one": str(table_path),
            "source_data_csv": str(out_dir / "baseline_table_one_source_data.csv"),
            "n_rows_plotted": int(len(keep)),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "baseline_table_one.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "descriptive_publication_bundle_from_parent_outputs_v1"
def _iter_prior_output_tables(
    *,
    run_dir: Path,
    current_step_id: str,
) -> Sequence[Tuple[Path, pd.DataFrame]]:
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return []
    tables: List[Tuple[Path, pd.DataFrame]] = []
    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        for csv_path in sorted(outputs_dir.glob("*.csv")):
            try:
                tables.append((csv_path, pd.read_csv(csv_path)))
            except Exception:
                continue
    return tables
def _find_column(
    frame: pd.DataFrame,
    *,
    exact: Sequence[str] = (),
    suffixes: Sequence[str] = (),
    contains: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Optional[str]:
    excluded = {item.lower() for item in exclude}
    lower_to_orig = {str(c).lower(): c for c in frame.columns}
    for candidate in exact:
        key = candidate.lower()
        if key in lower_to_orig and key not in excluded:
            return str(lower_to_orig[key])
    for column in frame.columns:
        key = str(column).lower()
        if key in excluded:
            continue
        if suffixes and any(key.endswith(suffix.lower()) for suffix in suffixes):
            return str(column)
        if contains and any(token.lower() in key for token in contains):
            return str(column)
    return None
def _as_percent(row: pd.Series, column: Optional[str]) -> Optional[float]:
    if not column:
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value
def _event_count_column(
    frame: pd.DataFrame, denominator_col: Optional[str]
) -> Optional[str]:
    excluded = {
        "n",
        "n_total",
        "total_n",
        "denominator",
        "n_denominator",
        str(denominator_col or "").lower(),
    }
    column = _find_column(
        frame,
        exact=("event_n", "events", "outcome_n", "n_positive"),
        suffixes=("_event_n", "_events", "_n"),
        exclude=tuple(excluded),
    )
    return column
def _label_column(frame: pd.DataFrame) -> Optional[str]:
    return _find_column(
        frame,
        exact=(
            "label",
            "group_label",
            "exposure_label",
            "stratum_label",
            "category_label",
        ),
        suffixes=("_label",),
    )
_BINARY_GROUP_EXCLUDED_TOKENS = (
    "n",
    "count",
    "event",
    "events",
    "death",
    "mort",
    "risk",
    "rate",
    "prevalence",
    "incidence",
    "ci",
    "lower",
    "upper",
    "pct",
    "percent",
    "source",
    "row",
)
def _binary_group_column(frame: pd.DataFrame) -> Optional[str]:
    binary_tokens = {"0", "1", "0.0", "1.0", "false", "true", "no", "yes"}
    for column in frame.columns:
        key = str(column).lower()
        if any(token in key for token in _BINARY_GROUP_EXCLUDED_TOKENS):
            continue
        values = [
            str(value).strip().lower()
            for value in frame[column].dropna().tolist()
            if str(value).strip()
        ]
        if not values:
            continue
        binary_values = [value for value in values if value in binary_tokens]
        allowed_extra_values = [
            value
            for value in values
            if value not in binary_tokens and "risk_difference" not in value
        ]
        if len(set(binary_values)) >= 2 and not allowed_extra_values:
            return str(column)
    return None
def _binary_group_label(column: str, value: Any) -> str:
    normalized = str(value).strip().lower()
    base = _publication_label(column)
    if normalized in {"1", "1.0", "true", "yes"}:
        return f"{base} positive"
    if normalized in {"0", "0.0", "false", "no"}:
        return f"{base} negative"
    return _publication_label(value)
def _is_risk_difference_row(row: pd.Series, *values: Any) -> bool:
    haystack = " ".join([str(value or "") for value in values])
    haystack = (
        f"{haystack} {' '.join(str(value or '') for value in row.to_dict().values())}"
    )
    return (
        "risk_difference" in haystack.lower() or "risk difference" in haystack.lower()
    )
def _context_axis_label(metric: Any, group: Any) -> str:
    metric_text = str(metric or "").strip()
    group_text = str(group or "").strip()
    if metric_text.lower() == "exposure prevalence":
        suffix = " prevalence"
        if group_text.lower().endswith(suffix) and len(group_text) > len(suffix):
            return (
                f"{_short_figure_label(group_text[: -len(suffix)].strip(), limit=24)}\n"
                "prevalence"
            )
        return _short_figure_label(group_text or metric_text, limit=28)
    if metric_text and group_text and metric_text.lower() not in group_text.lower():
        return (
            f"{_short_figure_label(group_text, limit=24)}\n"
            f"{_short_figure_label(metric_text, limit=24)}"
        )
    return _short_figure_label(group_text or metric_text or "Context", limit=28)
def _association_descriptive_context(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    primary_exposure: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect source-backed prevalence or absolute-risk rows for association figures.

    The helper is deliberately keyed to generic column semantics
    (prevalence/risk/event-rate), not to a benchmark variable name.
    """

    plot_rows: List[Dict[str, Any]] = []
    source_files: List[str] = []
    has_prevalence = False
    has_outcome_risk = False

    def _canonical_exposure_token(value: Any) -> str:
        token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        suffixes = (
            "_log1p_active",
            "_log1p",
            "_active",
            "_maximum",
            "_minimum",
            "_median",
            "_mean",
            "_first",
            "_last",
            "_value",
            "_max",
            "_min",
        )
        changed = True
        while changed and token:
            changed = False
            for suffix in suffixes:
                if token.endswith(suffix) and len(token) > len(suffix):
                    token = token[: -len(suffix)]
                    changed = True
                    break
        return token

    primary_token = _canonical_exposure_token(primary_exposure)

    for table_path, frame in _iter_prior_output_tables(
        run_dir=run_dir,
        current_step_id=current_step_id,
    ):
        if frame.empty:
            continue
        frame = frame.copy()
        if primary_token:
            exposure_col = _find_column(
                frame,
                exact=(
                    "exposure",
                    "exposure_source",
                    "source_variable",
                    "concept",
                    "variable",
                ),
            )
            if exposure_col:
                matches = (
                    frame[exposure_col].map(_canonical_exposure_token).eq(primary_token)
                )
                if matches.any():
                    frame = frame.loc[matches].copy()
                elif frame[exposure_col].nunique(dropna=True) > 1:
                    # This is explicitly a multi-exposure context table but it
                    # contains no row for the locked primary exposure.
                    continue
        prevalence_pct_col = _find_column(
            frame,
            exact=("prevalence_pct", "incidence_pct"),
        )
        prevalence_prop_col = _find_column(frame, exact=("prevalence", "incidence"))
        if not has_prevalence and (prevalence_pct_col or prevalence_prop_col):
            denominator_col = _find_column(
                frame,
                exact=("n_denominator", "denominator", "n_total", "total_n", "n"),
            )
            event_col = _find_column(frame, exact=("n_positive", "event_n", "events"))
            label_col = _find_column(
                frame,
                exact=("label", "exposure", "variable", "concept"),
            )
            source_rows: List[Dict[str, Any]] = []
            for idx, row in frame.iterrows():
                estimate = _as_percent(row, prevalence_pct_col or prevalence_prop_col)
                if estimate is None:
                    continue
                base_label = row.get(label_col) if label_col else "Exposure"
                display_label = f"{_publication_label(base_label)} prevalence"
                record = row.to_dict()
                record.update(
                    {
                        "plot_metric": "Exposure prevalence",
                        "plot_group_label": display_label,
                        "plot_estimate_pct": estimate,
                        "plot_ci_low_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_low_pct", "lower_pct"))
                            or _find_column(frame, exact=("ci_low", "lower")),
                        ),
                        "plot_ci_high_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_high_pct", "upper_pct"))
                            or _find_column(frame, exact=("ci_high", "upper")),
                        ),
                        "plot_denominator": (
                            row.get(denominator_col) if denominator_col else None
                        ),
                        "plot_event_n": row.get(event_col) if event_col else None,
                        "source_table": table_path.name,
                        "source_row_index": int(idx),
                    }
                )
                source_rows.append(record)
                plot_rows.append(record)
            if source_rows:
                source_path = out_dir / "publication_figure_prevalence_source_data.csv"
                pd.DataFrame(source_rows).to_csv(source_path, index=False)
                source_files.append(source_path.name)
                has_prevalence = True

        risk_pct_col = _find_column(
            frame,
            exact=("outcome_risk_pct", "risk_pct", "event_rate_pct"),
            suffixes=("_risk_pct", "_rate_pct"),
            exclude=("prevalence_pct", "incidence_pct"),
        )
        risk_prop_col = _find_column(
            frame,
            exact=("outcome_risk", "risk", "event_rate"),
            suffixes=("_risk", "_rate"),
            exclude=("prevalence", "incidence"),
        )
        if not has_outcome_risk and (risk_pct_col or risk_prop_col):
            denominator_col = _find_column(
                frame,
                exact=("n", "n_total", "total_n", "denominator", "n_denominator"),
            )
            event_col = _event_count_column(frame, denominator_col)
            label_col = _label_column(frame) or _find_column(
                frame,
                exact=("group", "category", "stratum", "exposure"),
            )
            binary_group_col = None if label_col else _binary_group_column(frame)
            metric_source = str(risk_pct_col or risk_prop_col or "outcome risk")
            metric_label = _publication_label(
                metric_source.replace("_pct", "").replace("_risk", " risk")
            )
            source_rows = []
            for idx, row in frame.iterrows():
                estimate = _as_percent(row, risk_pct_col or risk_prop_col)
                if estimate is None:
                    continue
                if _is_risk_difference_row(
                    row,
                    row.get(label_col) if label_col else None,
                    row.get(binary_group_col) if binary_group_col else None,
                ):
                    continue
                if label_col:
                    group_label = row.get(label_col)
                elif binary_group_col:
                    group_label = _binary_group_label(
                        binary_group_col,
                        row.get(binary_group_col),
                    )
                else:
                    group_label = f"Group {idx + 1}"
                record = row.to_dict()
                record.update(
                    {
                        "plot_metric": metric_label,
                        "plot_group_label": _publication_label(group_label),
                        "plot_estimate_pct": estimate,
                        "plot_ci_low_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_low_pct", "lower_pct"))
                            or _find_column(
                                frame,
                                exact=("ci_low", "lower"),
                                suffixes=(
                                    "_ci_low_pct",
                                    "_ci_low",
                                    "_lower_pct",
                                    "_lower",
                                ),
                            ),
                        ),
                        "plot_ci_high_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_high_pct", "upper_pct"))
                            or _find_column(
                                frame,
                                exact=("ci_high", "upper"),
                                suffixes=(
                                    "_ci_high_pct",
                                    "_ci_high",
                                    "_upper_pct",
                                    "_upper",
                                ),
                            ),
                        ),
                        "plot_denominator": (
                            row.get(denominator_col) if denominator_col else None
                        ),
                        "plot_event_n": row.get(event_col) if event_col else None,
                        "source_table": table_path.name,
                        "source_row_index": int(idx),
                    }
                )
                source_rows.append(record)
                plot_rows.append(record)
            if source_rows:
                source_path = (
                    out_dir / "publication_figure_absolute_risk_source_data.csv"
                )
                pd.DataFrame(source_rows).to_csv(source_path, index=False)
                source_files.append(source_path.name)
                has_outcome_risk = True

        if has_prevalence and has_outcome_risk:
            break

    if has_prevalence and has_outcome_risk:
        title = "Prevalence and absolute outcome risk"
        claim = "Exposure prevalence and absolute outcome risk are shown before adjusted relative estimates."
    elif has_prevalence:
        title = "Exposure prevalence"
        claim = "Exposure prevalence is shown before adjusted relative estimates."
    elif has_outcome_risk:
        title = "Absolute outcome risk"
        claim = "Absolute outcome risk is shown before adjusted relative estimates."
    else:
        title = ""
        claim = ""
    return {
        "plot_rows": plot_rows,
        "source_files": source_files,
        "has_prevalence": has_prevalence,
        "has_outcome_risk": has_outcome_risk,
        "title": title,
        "claim": claim,
    }
def _render_absolute_risk_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Render measurement availability and unadjusted outcome risk.

    The renderer accepts only the direct parent's tidy absolute-risk contract
    (``exposure``, ``group_type``, ``estimate_type``).  It never re-reads the
    cohort and every plotted row carries a positional trace back to the parent
    CSV.  This is intentionally separate from the association renderer: an
    absolute-risk context step has no adjusted estimand to invent or borrow.
    """

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(current_step_id or ""):
        return None
    parent_outputs = Path(run_dir) / "steps" / parent_step_id / "outputs"
    if not parent_outputs.exists():
        return None

    candidates = [parent_outputs / "exposure_outcome_summary.csv"]
    candidates.extend(
        path for path in sorted(parent_outputs.glob("*.csv")) if path not in candidates
    )
    table_path: Optional[Path] = None
    frame: Optional[pd.DataFrame] = None
    required = {"exposure", "group_type", "estimate_type"}
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loaded = pd.read_csv(candidate).reset_index(drop=True)
        except Exception:
            continue
        if loaded.empty or not required.issubset(set(loaded.columns)):
            continue
        if not any(
            column in loaded.columns
            for column in ("outcome_risk_pct", "outcome_risk", "estimate")
        ):
            continue
        table_path, frame = candidate, loaded
        break
    if table_path is None or frame is None:
        return None

    estimate_type = frame["estimate_type"].astype(str).str.lower()
    group_type = frame["group_type"].astype(str).str.lower()
    group_value = (
        frame.get("group_value", pd.Series("", index=frame.index, dtype="object"))
        .astype(str)
        .str.lower()
    )

    availability_mask = (
        estimate_type.eq("prevalence")
        & group_type.eq("source_state")
        & group_value.eq("observed")
    )
    availability = frame.loc[availability_mask].copy()

    risk_mask = estimate_type.eq("outcome_risk")
    level_risk = frame.loc[risk_mask & group_type.eq("exposure_level")].copy()
    if len(level_risk) >= 2:
        counts = level_risk["exposure"].astype(str).value_counts()
        primary_exposure = str(counts.index[0])
        risk = level_risk[
            level_risk["exposure"].astype(str).eq(primary_exposure)
        ].copy()
        no_source = frame.loc[
            risk_mask
            & group_type.eq("source_state")
            & group_value.eq("no_source")
            & frame["exposure"].astype(str).eq(primary_exposure)
        ].copy()
        risk = pd.concat([risk, no_source], axis=0)
    else:
        risk = frame.loc[risk_mask & group_type.eq("source_state")].copy()

    distribution = frame.loc[
        estimate_type.eq("continuous_distribution")
        & frame.get("median", pd.Series(float("nan"), index=frame.index)).notna()
    ].copy()
    if availability.empty or risk.empty:
        return None

    def traced(rows: pd.DataFrame, transform: str) -> pd.DataFrame:
        traced_rows = rows.copy()
        traced_rows.insert(0, "source_row_index", traced_rows.index.astype(int))
        # The parent table repeats exposure/label across prevalence and risk
        # rows.  Those names are generic validator key columns, so preserving
        # them would trigger an ambiguous many-to-many join.  Rename only the
        # display identifiers and let ``source_row_index`` provide the exact
        # positional trace; numeric source columns remain byte-for-byte intact.
        traced_rows = traced_rows.rename(
            columns={
                column: f"source_{column}"
                for column in (
                    "label",
                    "variable",
                    "term",
                    "exposure",
                    "contrast",
                    "stage",
                    "level",
                    "band",
                    "category",
                )
                if column in traced_rows.columns
            }
        )
        traced_rows["source_table"] = table_path.name
        traced_rows["source_transform"] = transform
        return traced_rows.reset_index(drop=True)

    availability_source = traced(availability, "observed_source_prevalence_rows_v1")
    risk_source = traced(risk, "absolute_outcome_risk_rows_v1")
    distribution_source = traced(distribution, "continuous_distribution_rows_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    availability_name = "absolute_risk_availability_source_data.csv"
    risk_name = "absolute_risk_outcome_source_data.csv"
    availability_source.to_csv(out_dir / availability_name, index=False)
    risk_source.to_csv(out_dir / risk_name, index=False)
    source_files = [availability_name, risk_name]
    distribution_name = "absolute_risk_distribution_source_data.csv"
    if not distribution_source.empty:
        distribution_source.to_csv(out_dir / distribution_name, index=False)
        source_files.append(distribution_name)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    has_distribution = not distribution.empty
    fig = plt.figure(
        figsize=(183 / 25.4, (112 if has_distribution else 88) / 25.4),
        constrained_layout=False,
    )
    if has_distribution:
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[0.82, 1.35],
            height_ratios=[1.0, 0.72],
            left=0.16,
            right=0.98,
            top=0.84,
            bottom=0.14,
            wspace=0.78,
            hspace=0.70,
        )
        ax_availability = fig.add_subplot(grid[0, 0])
        ax_distribution = fig.add_subplot(grid[1, 0])
        ax_risk = fig.add_subplot(grid[:, 1])
    else:
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[0.88, 1.32],
            left=0.16,
            right=0.98,
            top=0.84,
            bottom=0.18,
            wspace=0.78,
        )
        ax_availability = fig.add_subplot(grid[0, 0])
        ax_distribution = None
        ax_risk = fig.add_subplot(grid[0, 1])

    availability_x = [
        _as_percent(
            row, "prevalence_pct" if "prevalence_pct" in frame.columns else "prevalence"
        )
        for _, row in availability.iterrows()
    ]
    availability_lo = [_as_percent(row, "ci_low") for _, row in availability.iterrows()]
    availability_hi = [
        _as_percent(row, "ci_high") for _, row in availability.iterrows()
    ]
    availability_labels = [
        _short_figure_label(_publication_label(value), limit=25)
        for value in availability["exposure"].astype(str)
    ]
    y_availability = list(range(len(availability)))
    ax_availability.errorbar(
        availability_x,
        y_availability,
        xerr=[
            [
                max(0.0, x - (lo if lo is not None else x))
                for x, lo in zip(availability_x, availability_lo)
            ],
            [
                max(0.0, (hi if hi is not None else x) - x)
                for x, hi in zip(availability_x, availability_hi)
            ],
        ],
        fmt="o",
        color=palette.get("teal", "#42949E"),
        ecolor=palette.get("teal", "#42949E"),
        elinewidth=1.0,
        capsize=2.3,
        markersize=4.2,
    )
    ax_availability.set_yticks(y_availability)
    ax_availability.set_yticklabels(availability_labels, fontsize=6.7)
    ax_availability.set_xlim(0, 100)
    ax_availability.set_xlabel("Observed stays, % (95% CI)")
    ax_availability.set_title("Measurement availability", loc="left", pad=4)
    ax_availability.invert_yaxis()
    ax_availability.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    add_panel_label(ax_availability, "A", x=0.0, y=1.08)

    risk_x = [
        _as_percent(
            row,
            (
                "outcome_risk_pct"
                if "outcome_risk_pct" in frame.columns
                else "outcome_risk"
            ),
        )
        for _, row in risk.iterrows()
    ]
    risk_lo = [_as_percent(row, "ci_low") for _, row in risk.iterrows()]
    risk_hi = [_as_percent(row, "ci_high") for _, row in risk.iterrows()]
    risk_labels = []
    for _, row in risk.iterrows():
        exposure_label = _publication_label(row.get("exposure"))
        if str(row.get("group_type") or "").lower() == "exposure_level":
            label = f"{exposure_label} = {row.get('group_value')}"
        elif str(row.get("group_value") or "").lower() == "no_source":
            label = "No recorded source"
        else:
            label = _publication_label(row.get("label") or row.get("group_value"))
        risk_labels.append(_short_figure_label(label, limit=30))
    y_risk = list(range(len(risk)))
    ax_risk.errorbar(
        risk_x,
        y_risk,
        xerr=[
            [
                max(0.0, x - (lo if lo is not None else x))
                for x, lo in zip(risk_x, risk_lo)
            ],
            [
                max(0.0, (hi if hi is not None else x) - x)
                for x, hi in zip(risk_x, risk_hi)
            ],
        ],
        fmt="o",
        color=palette.get("blue", "#0F4D92"),
        ecolor=palette.get("blue", "#0F4D92"),
        elinewidth=1.0,
        capsize=2.3,
        markersize=4.2,
    )
    max_risk = max([value for value in risk_hi if value is not None] or risk_x)
    ax_risk.set_xlim(0, max(10.0, max_risk * 1.28))
    ax_risk.set_yticks(y_risk)
    ax_risk.set_yticklabels(risk_labels, fontsize=6.8)
    ax_risk.set_xlabel("Outcome risk, % (95% CI)")
    ax_risk.set_title("Absolute outcome risk", loc="left", pad=4)
    ax_risk.invert_yaxis()
    ax_risk.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    for row_index, (value, upper) in enumerate(zip(risk_x, risk_hi)):
        ax_risk.text(
            max(value, upper if upper is not None else value) + 0.45,
            row_index,
            f"{value:.1f}%",
            va="center",
            fontsize=6.2,
            color=palette.get("baseline", "#272727"),
        )
    add_panel_label(ax_risk, "B", x=0.0, y=1.06)

    panels: List[Dict[str, Any]] = [
        {
            "panel_id": "A",
            "title": "Measurement availability",
            "role": "descriptive_result",
            "chart_type": "dot_interval_prevalence",
            "claim": "Source-consistent observed prevalence is shown for each requested exposure.",
            "evidence_ids": [availability_name],
        },
        {
            "panel_id": "B",
            "title": "Absolute outcome risk",
            "role": "descriptive_result",
            "chart_type": "dot_interval_absolute_risk",
            "claim": "Unadjusted outcome risks and Wilson 95% confidence intervals are shown for prespecified exposure levels, retaining the no-source group.",
            "evidence_ids": [risk_name],
        },
    ]
    if ax_distribution is not None:
        medians = pd.to_numeric(distribution["median"], errors="coerce").to_numpy()
        q25 = pd.to_numeric(distribution["q25"], errors="coerce").to_numpy()
        q75 = pd.to_numeric(distribution["q75"], errors="coerce").to_numpy()
        y_distribution = list(range(len(distribution)))
        ax_distribution.errorbar(
            medians,
            y_distribution,
            xerr=[medians - q25, q75 - medians],
            fmt="o",
            color=palette.get("orange", "#E69F00"),
            ecolor=palette.get("orange", "#E69F00"),
            elinewidth=1.0,
            capsize=2.3,
            markersize=4.2,
        )
        ax_distribution.set_yticks(y_distribution)
        ax_distribution.set_yticklabels(
            [
                _short_figure_label(_publication_label(value), limit=25)
                for value in distribution["exposure"].astype(str)
            ],
            fontsize=6.7,
        )
        ax_distribution.set_xlabel("Median (IQR)")
        ax_distribution.set_title("Observed distribution", loc="left", pad=4)
        ax_distribution.invert_yaxis()
        ax_distribution.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        add_panel_label(ax_distribution, "C", x=0.0, y=1.10)
        panels.append(
            {
                "panel_id": "C",
                "title": "Observed distribution",
                "role": "descriptive_result",
                "chart_type": "median_iqr",
                "claim": "Continuous observed exposures are summarised by median and interquartile range without post-hoc binning.",
                "evidence_ids": [distribution_name],
            }
        )

    fig.suptitle(
        "Measurement availability and unadjusted outcome context",
        x=0.16,
        ha="left",
        y=0.98,
        fontsize=9.2,
        fontweight="bold",
    )
    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "The figure shows measurement availability, absolute outcome risk, "
            "and continuous-distribution context before adjusted modelling."
        ),
        panels=panels,
        source_data=source_files,
        statistics_note=(
            "Rendered deterministically from the direct parent's tidy summary. "
            "Risk and prevalence intervals are Wilson 95% confidence intervals; "
            "continuous values are median (IQR)."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "publication_figure", contract=contract, dpi=300
    )
    plt.close(fig)

    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": current_step_id,
        "method": "deterministic_absolute_risk_publication_figure",
        "rendering_only": True,
        "source_step_id": parent_step_id,
        "source_table": str(table_path),
        "source_data_files": source_files,
        "n_availability_rows": int(len(availability)),
        "n_risk_rows": int(len(risk)),
        "n_distribution_rows": int(len(distribution)),
        "figure_files": figure_files,
        "figure_path": "publication_figure.png",
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "absolute_risk_publication_bundle_from_parent_outputs_v1"
_UPSTREAM_FAMILY_TO_RENDERER_KEY: dict[str, str] = {
    "association": "association",
    "dose_response": "association",
    "prediction": "prediction",
    "prediction_model": "prediction",
    "survival": "survival",
    "survival_analysis": "survival",
    "cohort_definition": "cohort",
    "cohort_definition_sensitivity": "sensitivity",
    "sensitivity_analysis": "sensitivity",
    "missingness": "missingness",
    "measurement": "missingness",
    "data_quality": "missingness",
    "absolute_risk_context": "absolute_risk",
    "phenotyping": "phenotype",
    "clustering": "phenotype",
    "descriptive": "descriptive",
    "table_one": "descriptive",
    "baseline": "descriptive",
}
_UPSTREAM_METHOD_TO_RENDERER_KEY: dict[str, str] = {
    "ordinal_exposure_derivation_and_quality_control": "ordered_distribution",
    "exposure_distribution_and_missingness_audit": "distribution_availability",
    "cohort_definition_sensitivity": "sensitivity",
    "missingness": "missingness",
    "missingness_audit": "missingness",
    "missingness_measurement_audit": "missingness",
}
_UPSTREAM_FIGURE_DATA_FAMILY_TO_RENDERER_KEY: dict[str, str] = {
    "ordered_category_distribution": "ordered_distribution",
}
_AMBIGUOUS_FIGURE_DATA_FAMILY = "__ambiguous_figure_data_family__"
_INCOMPATIBLE_FIGURE_DATA_FAMILY = "__incompatible_figure_data_family__"
def _resolve_upstream_analysis_family(
    run_dir: Path, current_step_id: str
) -> Optional[str]:
    """Return the ``analysis_family`` recorded by a figure step's parent step."""

    parent = str(current_step_id or "").removesuffix("_figure")
    if not parent or parent == str(current_step_id):
        return None
    summ = Path(run_dir) / "steps" / parent / "outputs" / "step_summary.json"
    try:
        fam = json.loads(summ.read_text("utf-8")).get("analysis_family")
    except Exception:
        return None
    return str(fam).strip().lower() if fam else None
def _renderer_for_upstream_figure_data_family(family: Optional[str]):
    """Map an explicit step-level figure-data contract to its renderer."""

    key = _UPSTREAM_FIGURE_DATA_FAMILY_TO_RENDERER_KEY.get(
        str(family or "").strip().lower()
    )
    if key == "ordered_distribution":
        from ..figures.ordered_distribution import (
            render_ordered_distribution_bundle_from_prior_outputs,
        )

        return render_ordered_distribution_bundle_from_prior_outputs
    return None
def deterministic_figure_family_supported(step_id: str) -> bool:
    """Deprecated name-only compatibility probe; names never establish ownership."""

    del step_id
    return False
def _truthy_figure_value(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}
def _explicit_false_figure_value(value: Any) -> bool:
    if value is False:
        return True
    if value is True or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"false", "0", "no"}
def _sensitivity_plot_label(row: Mapping[str, Any]) -> str:
    spec_id = str(row.get("spec_id") or "").strip().lower()
    if spec_id.endswith("_crude_rd"):
        spec_id = spec_id.removesuffix("_crude_rd")
    mapping = {
        "full_export_step03_scope": "Full export",
        "primary_los_ge_1d": "Primary cohort",
        "primary": "Primary cohort",
        "cohort_no_los_restriction": "No ICU LOS restriction",
        "cohort_los_ge_2d": "ICU LOS >=2 d",
        "cohort_core_physiology_present": "Core physiology present",
        "primary_adult_los1_all_vitals_sepsis3_derivable": "Primary cohort",
        "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
        "alt_adult_los2_all_vitals_sepsis3_derivable": "ICU LOS >=2 d",
        "primary_lactate_complete_case": "Lactate obs.",
        "primary_without_lactate_adjustment": "No lactate adj.",
        "missing_raw_complete_case": "Complete-case",
        "missing_drop_lactate": "Drop lactate",
        "effect_robust_poisson_rr": "Risk ratio",
        "effect_marginal_standardized_rd": "Risk difference",
    }
    if spec_id in mapping:
        return mapping[spec_id]
    los_match = re.fullmatch(r"alt_cohort_los_ge_(\d+(?:p\d+)?)([hd])", spec_id)
    if los_match:
        value = los_match.group(1).replace("p", ".")
        unit = "h" if los_match.group(2) == "h" else "d"
        return f"ICU LOS >= {value} {unit}"
    if "complete_case" in spec_id:
        return "Complete-case"
    if "source_aware" in spec_id:
        return "Source-aware"
    label = str(row.get("display_label") or row.get("label") or spec_id).strip()
    return _short_figure_label(label.replace("LOS ≥", "LOS >="), limit=30)
