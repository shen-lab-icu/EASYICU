"""Deterministic rescue renderers for the phenotyping (clustering) figure and the
descriptive baseline / table-one figure.

Both mirror the existing System-B rescue renderers (`(run_dir, current_step_id,
out_dir) -> Optional[str]`): they read a parent step's certified CSV, render a
figure, and emit a validator-conformant `*_source_data.csv` traced positionally
via `source_row_index`. These tests lock: (a) the renderer fires on its real
runner tables and returns a repair-id; (b) the emitted source data passes the REAL
FigureSourceDataValidator (no fabrication); (c) the strict guards return None for a
non-matching table (a run without a partition / a result table); (d) the step-level
router claims the right step and does NOT steal an association/survival figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.pipeline import (
    _render_descriptive_publication_bundle_from_prior_outputs as descriptive_rescue,
    _render_phenotype_publication_bundle_from_prior_outputs as phenotype_rescue,
    _render_publication_bundle_from_prior_outputs_for_step as routed_rescue,
    deterministic_figure_family_supported,
)
from easyicu.research_agent.schema import AnalysisStep


def _fig_step(step_id: str) -> AnalysisStep:
    return AnalysisStep(step_id=step_id, intent="Render a figure", method="figure")


def _no_error(findings) -> bool:
    return [f for f in findings if f.severity == "error"] == []


# --------------------------------------------------------------------------- #
# Phenotype rescue                                                            #
# --------------------------------------------------------------------------- #


def _write_phenotype_parent(run_dir: Path, parent_id: str = "05_trajectory_clustering"):
    out = run_dir / "steps" / parent_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cluster": [0, 1, 2],
            "n": [412, 233, 155],
            "n_deaths": [25, 35, 62],
            "mortality_rate": [0.0607, 0.1502, 0.4000],
            "ci_low": [0.0413, 0.1094, 0.3236],
            "ci_high": [0.0884, 0.2028, 0.4813],
        }
    ).to_csv(out / "outcome_by_cluster.csv", index=False)
    pd.DataFrame({"cluster": [0, 1, 2], "n": [412, 233, 155]}).to_csv(
        out / "cluster_sizes.csv", index=False
    )
    pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1, 2, 2],
            "feature": ["sofa2_mean", "sofa2_slope"] * 3,
            "mean": [2.1, 0.0, 3.4, 0.6, 9.1, -0.5],
            "median": [2.0, 0.0, 3.2, 0.5, 9.0, -0.4],
        }
    ).to_csv(out / "cluster_characteristics.csv", index=False)
    return out


def test_phenotype_rescue_emits_traceable_source_data(tmp_path: Path):
    _write_phenotype_parent(tmp_path)
    fig_out = tmp_path / "steps" / "05_trajectory_clustering_figure" / "outputs"
    rid = phenotype_rescue(
        run_dir=tmp_path,
        current_step_id="05_trajectory_clustering_figure",
        out_dir=fig_out,
    )
    assert rid == "phenotype_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(fig_out / "phenotype_cluster_panel_source_data.csv")
    # positional trace + verbatim CI copy (validator value-checks ci_low/ci_high)
    assert "source_row_index" in source.columns
    assert list(source["source_row_index"]) == [0, 1, 2]
    assert source["ci_low"].tolist() == [0.0413, 0.1094, 0.3236]
    assert source["source_table"].iloc[0] == "outcome_by_cluster.csv"

    findings = FigureSourceDataValidator().audit(
        step=_fig_step("05_trajectory_clustering_figure"),
        out_dir=fig_out,
        run_dir=tmp_path,
        step_summary={},
    )
    assert _no_error(findings), [f.message for f in findings]


def test_phenotype_source_is_value_traceable_to_declared_parent(tmp_path: Path):
    # The emitted source_data must be a genuine VALUE-traceable subset of its
    # declared parent (outcome_by_cluster.csv): the low-level comparator value-
    # checks ci_low/ci_high positionally, so a fabricated CI disagrees. (This is
    # the meaningful per-renderer contract; the full audit's multi-candidate
    # leniency is separate pre-existing gate behaviour.)
    parent_out = _write_phenotype_parent(tmp_path)
    up = parent_out / "outcome_by_cluster.csv"
    fig_out = tmp_path / "steps" / "05_trajectory_clustering_figure" / "outputs"
    phenotype_rescue(
        run_dir=tmp_path,
        current_step_id="05_trajectory_clustering_figure",
        out_dir=fig_out,
    )
    sp = fig_out / "phenotype_cluster_panel_source_data.csv"
    clean = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=pd.read_csv(sp), source_path=sp, upstream_path=up
    )
    assert clean["ok"] is True, clean

    src = pd.read_csv(sp)
    src.loc[0, "ci_low"] = 0.999  # not in the parent
    src.to_csv(sp, index=False)
    bad = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=pd.read_csv(sp), source_path=sp, upstream_path=up
    )
    assert bad["ok"] is False and bad["reason"] == "source_values_disagree", bad


def test_phenotype_rescue_blocks_without_two_clusters(tmp_path: Path):
    out = tmp_path / "steps" / "05_trajectory_clustering" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"cluster": [0], "n": [800], "mortality_rate": [0.2], "ci_low": [0.17], "ci_high": [0.23]}
    ).to_csv(out / "outcome_by_cluster.csv", index=False)
    fig_out = tmp_path / "steps" / "05_trajectory_clustering_figure" / "outputs"
    rid = phenotype_rescue(
        run_dir=tmp_path,
        current_step_id="05_trajectory_clustering_figure",
        out_dir=fig_out,
    )
    assert rid is None


# --------------------------------------------------------------------------- #
# Descriptive / table-one rescue                                             #
# --------------------------------------------------------------------------- #


def _write_table_one_parent(run_dir: Path, parent_id: str = "03_baseline_table_and_absolute_risk"):
    out = run_dir / "steps" / parent_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    # Mirrors the real E1 table_one.csv columns (continuous + categorical rows).
    pd.DataFrame(
        {
            "variable": ["age", "los_icu", "sex", "sex", "hr_measured_label"],
            "label": ["Age, years", "ICU LOS, days", "Sex", "Sex", "HR measured"],
            "row_type": ["continuous", "continuous", "categorical", "categorical", "categorical"],
            "category": ["", "", "Female", "Male", "Measured"],
            "overall_median": [65.0, 2.5, float("nan"), float("nan"), float("nan")],
            "overall_percentage": [
                float("nan"),
                float("nan"),
                43.25,
                56.75,
                99.90,
            ],
        }
    ).to_csv(out / "table_one.csv", index=False)
    return out


def test_descriptive_rescue_emits_traceable_source_data(tmp_path: Path):
    _write_table_one_parent(tmp_path)
    fig_out = (
        tmp_path / "steps" / "03_baseline_table_and_absolute_risk_figure" / "outputs"
    )
    rid = descriptive_rescue(
        run_dir=tmp_path,
        current_step_id="03_baseline_table_and_absolute_risk_figure",
        out_dir=fig_out,
    )
    assert rid == "descriptive_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(fig_out / "baseline_table_one_source_data.csv")
    assert "source_row_index" in source.columns
    assert source["source_table"].iloc[0] == "table_one.csv"
    # both a continuous (median) and categorical (percentage) row are kept
    assert (source["overall_median"].notna()).any()
    assert (source["overall_percentage"].notna()).any()

    findings = FigureSourceDataValidator().audit(
        step=_fig_step("03_baseline_table_and_absolute_risk_figure"),
        out_dir=fig_out,
        run_dir=tmp_path,
        step_summary={},
    )
    assert _no_error(findings), [f.message for f in findings]


def test_descriptive_rescue_returns_none_for_a_result_table(tmp_path: Path):
    # A result/effect table (has an odds_ratio column) is NOT a table one; the
    # strict guard must decline it so the association renderer keeps the step.
    out = tmp_path / "steps" / "06_primary_association_model" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variable": ["lactate", "age"],
            "row_type": ["continuous", "continuous"],
            "odds_ratio": [2.1, 1.3],
            "ci_low": [1.8, 1.1],
            "ci_high": [2.5, 1.5],
            "overall_median": [3.0, 65.0],
        }
    ).to_csv(out / "association_results.csv", index=False)
    fig_out = tmp_path / "steps" / "06_primary_association_model_figure" / "outputs"
    rid = descriptive_rescue(
        run_dir=tmp_path,
        current_step_id="06_primary_association_model_figure",
        out_dir=fig_out,
    )
    assert rid is None


# --------------------------------------------------------------------------- #
# Step-level routing                                                          #
# --------------------------------------------------------------------------- #


def test_router_claims_phenotype_and_table_one_figure_steps(tmp_path: Path):
    assert deterministic_figure_family_supported("05_trajectory_clustering_figure")
    assert deterministic_figure_family_supported("03_baseline_table_and_absolute_risk_figure")

    _write_phenotype_parent(tmp_path)
    pheno_out = tmp_path / "steps" / "05_trajectory_clustering_figure" / "outputs"
    assert (
        routed_rescue(
            run_dir=tmp_path,
            current_step_id="05_trajectory_clustering_figure",
            out_dir=pheno_out,
        )
        == "phenotype_publication_bundle_from_parent_outputs_v1"
    )

    _write_table_one_parent(tmp_path)
    desc_out = (
        tmp_path / "steps" / "03_baseline_table_and_absolute_risk_figure" / "outputs"
    )
    assert (
        routed_rescue(
            run_dir=tmp_path,
            current_step_id="03_baseline_table_and_absolute_risk_figure",
            out_dir=desc_out,
        )
        == "descriptive_publication_bundle_from_parent_outputs_v1"
    )


def test_router_does_not_steal_association_or_survival_figure(tmp_path: Path):
    # An association forest / survival figure step must keep routing to its own
    # renderer even though the new phenotype/descriptive branches now exist. These
    # token checks run BEFORE phenotype/descriptive in the chain.
    parent = tmp_path / "steps" / "06_primary_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "level": ["<2", ">=4"],
            "odds_ratio": [0.76, 4.38],
            "or_ci_low": [0.71, 4.11],
            "or_ci_high": [0.81, 4.67],
        }
    ).to_csv(parent / "dose_response.csv", index=False)
    out = tmp_path / "steps" / "06_primary_association_model_figure" / "outputs"
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="06_primary_association_model_figure",
        out_dir=out,
    )
    # association renderer id, NOT a phenotype/descriptive id
    assert rid is not None
    assert "phenotype" not in str(rid) and "descriptive" not in str(rid)
