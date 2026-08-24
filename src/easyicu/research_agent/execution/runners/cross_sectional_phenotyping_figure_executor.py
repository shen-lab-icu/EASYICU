"""Source-bound publication renderer for cross-sectional phenotyping."""

from __future__ import annotations

import json
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
from .cross_sectional_phenotyping_executor import (
    CLUSTER_STABILITY_PRODUCT,
    PHENOTYPE_ASSIGNMENTS_PRODUCT,
    PHENOTYPE_PROFILES_PRODUCT,
)
from .figure_input_capability import TypedInputCapability
from .typed_input_binding import BoundTypedInput, load_typed_input, sha256_file

PHENOTYPING_FIGURE_INPUTS = (
    PHENOTYPE_PROFILES_PRODUCT,
    PHENOTYPE_ASSIGNMENTS_PRODUCT,
    CLUSTER_STABILITY_PRODUCT,
)
PHENOTYPING_FIGURE_ANALYSIS_KIND = "cross_sectional_phenotyping_figure"
_CAPABILITY = TypedInputCapability(required=frozenset(PHENOTYPING_FIGURE_INPUTS))
_REQUIRED_COLUMNS = {
    PHENOTYPE_PROFILES_PRODUCT: frozenset(
        {"cluster", "variable", "standardised_centroid", "n"}
    ),
    PHENOTYPE_ASSIGNMENTS_PRODUCT: frozenset({"unit_id", "cluster"}),
    CLUSTER_STABILITY_PRODUCT: frozenset(
        {
            "replicate",
            "adjusted_rand_index",
            "mean_adjusted_rand_index",
            "algorithm_agreement_ari",
        }
    ),
}


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", product)
    ):
        return None
    return product


def _binding_has_columns(binding: Any, key: str) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    return bool(isinstance(columns, list) and _REQUIRED_COLUMNS[key] <= set(columns))


def cross_sectional_phenotyping_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    products = [_figure_product(value) for value in step.expected_outputs]
    return bool(
        step.planned_analysis_role == "auxiliary"
        and str(step.method or "").strip().casefold().split(" with ", 1)[0]
        == "visualization"
        and len(step.inputs) == len(PHENOTYPING_FIGURE_INPUTS)
        and set(step.inputs) == set(PHENOTYPING_FIGURE_INPUTS)
        and _CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] is not None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == set(PHENOTYPING_FIGURE_INPUTS)
        and all(
            _binding_has_columns(resolved_bindings.get(key), key)
            for key in PHENOTYPING_FIGURE_INPUTS
        )
    )


def cross_sectional_phenotyping_figure_executor_code(step: AnalysisStep) -> str:
    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("phenotyping figure has no safe figure product")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_figure_executor import run_cross_sectional_phenotyping_figure

        run_cross_sectional_phenotyping_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _load_inputs(
    *, run_dir: Path, resolved_inputs: Path | Mapping[str, Any], step_id: str
) -> dict[str, BoundTypedInput]:
    return {
        key: load_typed_input(
            input_key=key,
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            expected_declared_kind="table",
            expected_evidence_kind="table",
            require_consumption_contract=True,
            minimum_row_count=1,
        )
        for key in PHENOTYPING_FIGURE_INPUTS
    }


def run_cross_sectional_phenotyping_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> dict[str, Any]:
    if _figure_product(f"figure:{figure_product}") is None:
        raise ValueError("unsafe phenotyping figure product")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bound = _load_inputs(
        run_dir=Path(run_dir), resolved_inputs=resolved_inputs, step_id=step_id
    )
    for key, item in bound.items():
        missing = _REQUIRED_COLUMNS[key] - set(item.frame.columns)
        if missing:
            raise RuntimeError(f"{key} is missing columns {sorted(missing)!r}")
    profiles = bound[PHENOTYPE_PROFILES_PRODUCT].frame.copy()
    assignments = bound[PHENOTYPE_ASSIGNMENTS_PRODUCT].frame.copy()
    stability = bound[CLUSTER_STABILITY_PRODUCT].frame.copy()
    profiles["standardised_centroid"] = pd.to_numeric(
        profiles["standardised_centroid"], errors="coerce"
    )
    wide = profiles.pivot_table(
        index="cluster",
        columns="variable",
        values="standardised_centroid",
        aggfunc="mean",
    ).sort_index()
    if wide.shape[0] < 2 or wide.shape[1] < 2 or not np.isfinite(wide.to_numpy()).all():
        raise RuntimeError("phenotyping profile table is not a finite cluster matrix")
    sizes = assignments.groupby("cluster", sort=True).size()
    stability_values = pd.to_numeric(stability["adjusted_rand_index"], errors="coerce")
    if stability_values.isna().any() or not np.isfinite(stability_values).all():
        raise RuntimeError("phenotyping stability table is not finite")
    algorithm_values = pd.to_numeric(
        stability.get("algorithm_agreement_ari"), errors="coerce"
    )
    if (
        algorithm_values.isna().any()
        or not np.isfinite(algorithm_values).all()
        or algorithm_values.nunique() != 1
    ):
        raise RuntimeError("phenotyping algorithm-agreement value is not sealed")
    algorithm_agreement = float(algorithm_values.iloc[0])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = []
    for key, item in bound.items():
        filename = f"{key.partition(':')[2]}_source_data.csv"
        source = item.frame.copy()
        parent_name = item.path.name.split("__", 1)[-1]
        source.insert(0, "source_step_id", item.binding.get("produced_by_step"))
        source.insert(0, "source_table", parent_name)
        source.insert(0, "source_row_index", range(len(source)))
        source.to_csv(out_dir / filename, index=False)
        source_files.append(filename)

    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(183 / 25.4, 92 / 25.4), constrained_layout=True
    )
    image = axes[0].imshow(
        wide.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5
    )
    axes[0].set_yticks(range(len(wide.index)), [f"C{x}" for x in wide.index])
    axes[0].set_xticks(
        range(len(wide.columns)),
        [str(value)[:12] for value in wide.columns],
        rotation=45,
        ha="right",
    )
    axes[0].set_title("Standardised candidate-cluster profiles", loc="left", pad=10)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    add_panel_label(axes[0], "A", x=-0.16, y=1.05)

    axes[1].bar(range(len(sizes)), sizes.to_numpy(), color=palette["blue"], width=0.65)
    axes[1].set_xticks(range(len(sizes)), [f"C{x}" for x in sizes.index])
    axes[1].set_ylabel("Stays, n")
    axes[1].set_title("Cluster size", loc="left", pad=10)
    add_panel_label(axes[1], "B", x=-0.16, y=1.05)

    axes[2].bar(
        stability["replicate"].astype(str), stability_values, color=palette["orange"]
    )
    axes[2].axhline(
        float(stability_values.mean()), color="#333333", linestyle="--", linewidth=0.9
    )
    axes[2].axhline(
        algorithm_agreement,
        color=palette["blue"],
        linestyle=":",
        linewidth=1.1,
        label="GMM agreement",
    )
    axes[2].set_ylim(-0.05, 1.0)
    axes[2].set_xlabel("Resample")
    axes[2].set_ylabel("Adjusted Rand index")
    axes[2].set_title("Stability and algorithm agreement", loc="left", pad=10)
    axes[2].legend(frameon=False, fontsize=6, loc="upper right")
    add_panel_label(axes[2], "C", x=-0.16, y=1.05)

    evidence = {key: item.evidence_id for key, item in bound.items()}
    panel_specs = (
        (
            "A",
            "Candidate-cluster profiles",
            "phenotype_structure",
            (PHENOTYPE_PROFILES_PRODUCT,),
        ),
        ("B", "Cluster size", "phenotype_profile", (PHENOTYPE_ASSIGNMENTS_PRODUCT,)),
        (
            "C",
            "Stability and algorithm agreement",
            "stability",
            (CLUSTER_STABILITY_PRODUCT,),
        ),
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The analysis-only candidate clustering solution is displayed with "
            "exact standardised profiles, cluster sizes and resampling agreement; "
            "no phenotype name or biological entity is authorized."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=92.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "claim": "This panel is descriptive and does not establish a biological ground truth or clinical utility.",
                "evidence_ids": [evidence[source] for source in sources],
                "metadata": {
                    "source_products": list(sources),
                    "source_data": [
                        f"{source.partition(':')[2]}_source_data.csv"
                        for source in sources
                    ],
                },
            }
            for panel_id, title, role, sources in panel_specs
        ],
        source_data=source_files,
        statistics_note=(
            "Profiles and assignments are produced by the deterministic adapter; "
            "adjusted Rand indices compare fixed-seed subsample refits, and the "
            "dotted line compares the primary MiniBatchKMeans assignments with "
            "a deterministic diagonal-GMM alternative at the same K. Results "
            "remain analysis_only and do not establish external reproducibility."
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
    for item in bound.values():
        if sha256_file(item.path) != item.sha256:
            raise RuntimeError(f"typed figure input changed: {item.input_key}")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_cross_sectional_phenotyping_figure",
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": PHENOTYPING_FIGURE_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "phenotype_naming_authorized": False,
        "outcome_claim_authorized": False,
        "solution_label": "candidate_clusters_only",
        "rendering_only": True,
        "source_inputs": list(PHENOTYPING_FIGURE_INPUTS),
        "input_bindings": [
            {
                "input_key": key,
                "evidence_id": item.evidence_id,
                "sha256": item.sha256,
                "loaded": True,
                "row_count": item.row_count,
            }
            for key, item in bound.items()
        ],
        "source_data_files": source_files,
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "PHENOTYPING_FIGURE_ANALYSIS_KIND",
    "PHENOTYPING_FIGURE_INPUTS",
    "cross_sectional_phenotyping_figure_executor_code",
    "cross_sectional_phenotyping_figure_executor_owns_step",
    "run_cross_sectional_phenotyping_figure",
]
