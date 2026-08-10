"""Deterministic rendering of one digest-bound cohort-flow table.

The cohort-definition owner has already fixed every eligibility predicate and
count.  This renderer verifies those exact bytes and draws the remaining
denominator after each recorded step; it never reloads the cohort or invents
another inclusion rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
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
    "COHORT_FLOW_INPUT",
    "cohort_flow_figure_executor_code",
    "cohort_flow_figure_executor_owns_step",
    "run_cohort_flow_figure",
]


COHORT_FLOW_INPUT = "table:cohort_flow"
_REQUIRED_COLUMNS = (
    "step_order",
    "predicate_kind",
    "n_before",
    "n_excluded",
    "n_remaining",
)
_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")


def _method_head(value: Any) -> str:
    return str(value or "").strip().casefold().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not _PRODUCT_ID.fullmatch(product):
        return None
    return product


def _binding_is_cohort_flow(binding: Any) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    return bool(
        binding.get("declared_kind") == "table"
        and binding.get("evidence_kind") == "table"
        and binding.get("product") == "cohort_flow"
        and isinstance(columns, list)
        and set(_REQUIRED_COLUMNS).issubset(set(columns))
        and isinstance(consumption, Mapping)
        and consumption.get("mode") == "all_rows"
    )


def cohort_flow_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the exact typed parent fully fixes the figure."""

    products = [_figure_product(value) for value in step.expected_outputs]
    contracts = list(step.input_consumption_contracts or [])
    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and list(step.inputs) == [COHORT_FLOW_INPUT]
        and len(products) == 1
        and products[0] is not None
        and len(contracts) == 1
        and contracts[0].input_key == COHORT_FLOW_INPUT
        and contracts[0].mode == "all_rows"
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == {COHORT_FLOW_INPUT}
        and _binding_is_cohort_flow(resolved_bindings.get(COHORT_FLOW_INPUT))
    )


def cohort_flow_figure_executor_code(step: AnalysisStep) -> str:
    if list(step.inputs) != [COHORT_FLOW_INPUT] or len(step.expected_outputs) != 1:
        raise ValueError("cohort-flow figure requires its one exact input and output")
    product = _figure_product(step.expected_outputs[0])
    if product is None:
        raise ValueError("cohort-flow figure output is not a typed figure")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
            run_cohort_flow_figure,
        )

        run_cohort_flow_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _sha256(path: Path) -> str:
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
) -> tuple[Path, Mapping[str, Any]]:
    payload = (
        dict(resolved_inputs)
        if isinstance(resolved_inputs, Mapping)
        else json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    )
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else None
    if payload.get("step_id") != step_id or not isinstance(inputs, Mapping):
        raise ValueError("resolved-input manifest does not belong to this step")
    if set(inputs) != {COHORT_FLOW_INPUT}:
        raise ValueError("cohort-flow input binding is absent or widened")
    binding = inputs[COHORT_FLOW_INPUT]
    if not _binding_is_cohort_flow(binding):
        raise ValueError("cohort-flow input has no supported host contract")
    expected_sha = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    identity = binding.get("identity_row")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(identity, Mapping)
        or identity.get("input_key") != COHORT_FLOW_INPUT
        or identity.get("product") != "cohort_flow"
        or identity.get("sha256") != expected_sha
    ):
        raise ValueError("cohort-flow authority binding is incomplete")
    base = Path(run_dir).resolve()
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError("cohort-flow input escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("cohort-flow input is not a safe CSV file")
    if _sha256(path) != expected_sha:
        raise ValueError("cohort-flow input digest verification failed")
    return path, binding


def _verified_flow(path: Path, binding: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    contract = binding["product_contract"]
    expected_rows = contract.get("row_count")
    if (
        list(frame.columns) != list(contract.get("columns") or [])
        or isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows < 1
        or len(frame) != expected_rows
    ):
        raise ValueError("cohort-flow bytes disagree with their contract")
    numeric: dict[str, pd.Series] = {}
    for column in ("step_order", "n_before", "n_excluded", "n_remaining"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or not (values % 1 == 0).all():
            raise ValueError(f"cohort-flow has invalid {column} values")
        numeric[column] = values.astype("int64")
    if numeric["step_order"].duplicated().any():
        raise ValueError("cohort-flow step_order values are not unique")
    frame = frame.assign(**numeric).sort_values("step_order", kind="stable")
    labels = frame["predicate_kind"].fillna("").astype(str).str.strip()
    if labels.eq("").any():
        raise ValueError("cohort-flow has an empty predicate label")
    if not (frame["n_before"] - frame["n_excluded"]).eq(
        frame["n_remaining"]
    ).all():
        raise ValueError("cohort-flow denominator arithmetic failed")
    if len(frame) > 1 and not frame["n_before"].iloc[1:].reset_index(drop=True).eq(
        frame["n_remaining"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise ValueError("cohort-flow denominator sequence is discontinuous")
    return frame.reset_index(drop=True)


def run_cohort_flow_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> dict[str, Any]:
    """Verify and render every row in the canonical cohort-flow artifact."""

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path, binding = _load_binding(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
    )
    frame = _verified_flow(path, binding)
    source = frame.copy()
    source.insert(0, "source_step_id", binding.get("produced_by_step"))
    source.insert(0, "source_table", path.name)
    source.insert(0, "source_row_index", range(len(source)))
    source_path = out_dir / f"{figure_product}_source_data.csv"
    source.to_csv(source_path, index=False)

    labels = frame["predicate_kind"].astype(str).tolist()
    positions = list(range(len(frame)))
    apply_publication_style()
    height = max(3.2, 0.42 * len(frame) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, height))
    bars = ax.barh(
        positions,
        frame["n_remaining"],
        color=PALETTE_CLINICAL["blue"],
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ICU stays remaining")
    ax.set_title("Cohort accounting", loc="left")
    ax.grid(axis="x", color=PALETTE_CLINICAL["neutral_light"], linewidth=0.6)
    for bar, remaining, excluded in zip(
        bars, frame["n_remaining"], frame["n_excluded"]
    ):
        suffix = f"  (-{int(excluded):,})" if int(excluded) else ""
        ax.annotate(
            f"{int(remaining):,}{suffix}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    fig.tight_layout()
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The figure reproduces the remaining denominator at every row of "
            "the digest-verified cohort-flow table."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=92.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Cohort accounting",
                "role": "cohort_flow",
                "claim": "Every displayed count comes from the bound cohort flow.",
                "evidence_ids": [str(binding.get("evidence_id") or "")],
                "metadata": {
                    "chart_type": "cohort_flow",
                    "source_data": [source_path.name],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "All parent rows are preserved. The renderer introduces no cohort "
            "filter, imputation, or denominator change."
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
    if _sha256(path) != binding.get("sha256"):
        raise ValueError("cohort-flow input changed while it was rendered")
    figure_files = [item.name for key, item in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_cohort_flow_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "cohort_flow_figure",
        "rendering_only": True,
        "source_input": COHORT_FLOW_INPUT,
        "source_step_id": binding.get("produced_by_step"),
        "source_evidence_id": binding.get("evidence_id"),
        "source_sha256": binding.get("sha256"),
        "source_rows_consumed": len(frame),
        "input_bindings": [
            {
                "input_key": COHORT_FLOW_INPUT,
                "evidence_id": binding.get("evidence_id"),
                "sha256": binding.get("sha256"),
                "loaded": True,
                "row_count": len(frame),
            }
        ],
        "source_data_files": [source_path.name],
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
