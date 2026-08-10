"""Deterministic renderer for the framework-added supporting audit panel.

The panel reports only whether prior successful step summaries contain
structured audit fields.  It never labels an absent field as a failed check and
never re-runs or reinterprets a scientific analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...figures.publication import (
    PALETTE_CLINICAL,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep

__all__ = [
    "audit_panel_executor_code",
    "audit_panel_executor_owns_step",
    "run_audit_panel",
]


_DOMAIN_TOKENS = {
    "Data quality / missingness": (
        "available",
        "completeness",
        "measurement",
        "missing",
        "plausibility",
    ),
    "Sensitivity / robustness": (
        "bootstrap",
        "robustness",
        "sensitivity",
        "specification",
    ),
    "Leakage / validation": (
        "calibration",
        "leakage",
        "temporal_validity",
        "validation",
    ),
}


def audit_panel_executor_owns_step(step: AnalysisStep) -> bool:
    return bool(
        step.planned_analysis_role == "auxiliary"
        and str(step.method or "").strip().lower().split(" with ", 1)[0]
        == "visualization"
        and not step.inputs
        and list(step.expected_outputs) == ["figure:audit_panel"]
        and list(step.icu_rule_refs) == ["visualization_rule"]
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
    )


def audit_panel_executor_code(step: AnalysisStep) -> str:
    if not audit_panel_executor_owns_step(step):
        raise ValueError("step is not the framework audit-panel contract")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.audit_panel_executor import (
            run_audit_panel,
        )

        run_audit_panel(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            step_id={step.step_id!r},
        )
        """
    ).strip()


def _key_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(_key_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_key_paths(child, f"{prefix}[{index}]"))
    return paths


def run_audit_panel(
    *,
    out_dir: Path,
    run_dir: Path,
    step_id: str,
) -> dict[str, Any]:
    """Render structured-field coverage across prior step summaries."""

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[tuple[str, list[str]]] = []
    for summary_path in sorted(Path(run_dir).glob("steps/*/outputs/step_summary.json")):
        if summary_path.parent.parent.parent.name == step_id:
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        status = str(payload.get("status") or payload.get("analysis_status") or "").lower()
        if status not in {"completed", "ok", "success", "succeeded"}:
            continue
        summaries.append((summary_path.parent.parent.parent.name, _key_paths(payload)))

    rows: list[dict[str, Any]] = []
    denominator = len(summaries)
    for domain, tokens in _DOMAIN_TOKENS.items():
        matching_steps: list[str] = []
        matched_paths: list[str] = []
        for source_step, paths in summaries:
            matched = sorted(
                path for path in paths if any(token in path.lower() for token in tokens)
            )
            if matched:
                matching_steps.append(source_step)
                matched_paths.extend(f"{source_step}:{path}" for path in matched)
        count = len(matching_steps)
        rows.append(
            {
                "audit_domain": domain,
                "matching_summary_n": count,
                "eligible_summary_n": denominator,
                "coverage_pct": 100.0 * count / denominator if denominator else 0.0,
                "matched_path_n": len(matched_paths),
                "matching_step_ids": "; ".join(matching_steps),
            }
        )
    source = pd.DataFrame(rows)
    source_path = out_dir / "audit_panel_source_data.csv"
    source.to_csv(source_path, index=False)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    positions = list(range(len(source)))
    ax.barh(
        positions,
        source["coverage_pct"],
        color=PALETTE_CLINICAL["teal"],
        height=0.56,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(source["audit_domain"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Prior successful summaries with structured fields (%)")
    ax.set_title("Audit evidence availability", loc="left")
    ax.grid(axis="x", color=PALETTE_CLINICAL["neutral_light"], linewidth=0.6)
    for position, row in source.iterrows():
        ax.text(
            min(float(row["coverage_pct"]) + 2.0, 98.0),
            position,
            f"{int(row['matching_summary_n'])}/{int(row['eligible_summary_n'])}",
            va="center",
            ha="right" if float(row["coverage_pct"]) > 90 else "left",
            fontsize=7,
        )
    fig.tight_layout()
    contract = make_figure_contract(
        figure_id="figure:audit_panel",
        core_claim=(
            "The panel reports the presence of structured audit fields in prior "
            "successful steps; absence is not interpreted as a failed check."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=96.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Structured audit-field availability",
                "role": "audit",
                "claim": (
                    "Counts show which prior summaries expose domain-matching "
                    "structured audit fields, without asserting that checks passed."
                ),
                "evidence_ids": [source_path.name],
                "metadata": {
                    "chart_type": "audit_field_coverage",
                    "source_data": [source_path.name],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "The denominator is the number of prior successful step summaries. "
            "Zero means no matching structured field was found, not audit failure."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "audit_panel",
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_audit_panel",
        "analysis_family": "audit",
        "deterministic_standard_analysis": "audit_panel",
        "rendering_only": True,
        "prior_successful_summary_count": denominator,
        "source_data_files": [source_path.name],
        "figure_files": [path.name for key, path in outputs.items() if key != "contract"],
        "figure_path": "audit_panel.png",
        "figure_contract": "audit_panel.figure_contract.json",
        "contract_files": ["audit_panel.figure_contract.json"],
        "output_files": {"figure:audit_panel": "audit_panel.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
