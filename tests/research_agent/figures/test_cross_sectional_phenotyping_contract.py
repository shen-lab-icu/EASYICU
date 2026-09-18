"""Clinical profiles, structural displays and negative agreement stay distinct."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.runners import cross_sectional_phenotyping_figure_executor as renderer
from easyicu.research_agent.execution.figure_plan_binding import validate_step_planned_figure_contract_binding
from easyicu.research_agent.planning.figure_plan_shaping import (
    bind_deterministic_figure_panels,
    select_deterministic_result_renderers,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.research_agent.core.test_cross_sectional_phenotyping_executor import _binding, _context


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Display the existing candidate cluster structure and profiles.",
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="primary", planned_analysis_role="primary",
                intent="Fit the declared clinical feature roster.", method="cross_sectional_phenotyping",
                scientific_action_id="phenotyping.cluster_solution",
                inputs=["marker_a", "marker_b", "artifact:analysis_cohort"],
                expected_outputs=["table:phenotype_profiles", "table:phenotype_assignments"],
            ),
            AnalysisStep(
                step_id="stability", planned_analysis_role="secondary",
                intent="Evaluate conditional subsample agreement.", method="conditional_stability",
                scientific_action_id="phenotyping.cluster_stability",
                inputs=["table:phenotype_assignments"], expected_outputs=["table:cluster_stability"],
            ),
        ],
    )


def test_native_phenotyping_figure_is_selected_before_generic_overview() -> None:
    from easyicu.research_agent.planning.figure_plan_mutation import _ensure_publication_figure_step_in_plan

    original = _plan()
    shaped, findings = select_deterministic_result_renderers(plan=original)
    assert len(shaped.steps) == 3
    assert shaped.steps[:2] == original.steps
    figure = shaped.steps[-1]
    assert tuple(figure.inputs) == renderer.PHENOTYPING_FIGURE_INPUTS
    assert {(panel.article_role, panel.chart_type) for panel in figure.figure_panels} == {
        ("phenotype_structure", "embedding_plot"),
        ("phenotype_profile", "profile_heatmap"),
        ("stability", "subsampling_ari"),
    }
    assert findings
    again, duplicates = select_deterministic_result_renderers(plan=shaped)
    assert again == shaped and duplicates == []
    after_fallback, _ = _ensure_publication_figure_step_in_plan(plan=shaped, context=_context(60), force=True)
    assert after_fallback == shaped


@pytest.mark.parametrize("mutation", ("missing_stability", "duplicate_product", "unregistered_primary"))
def test_native_selector_does_not_borrow_ambiguous_or_unregistered_sources(mutation) -> None:
    plan = _plan()
    if mutation == "missing_stability":
        plan.steps.pop()
    elif mutation == "duplicate_product":
        plan.steps.append(plan.steps[-1].model_copy(update={"step_id": "other_stability"}))
    else:
        plan.steps[0].scientific_action_id = None
    shaped, findings = select_deterministic_result_renderers(plan=plan)
    assert shaped == plan and findings == []


def _source_tables(
    names: tuple[str, ...] = ("marker_a", "marker_b", "marker_c"), *, clusters: int = 3,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(clusters), 20)
    features = rng.normal(size=(len(labels), len(names))) + labels[:, None] * rng.normal(size=len(names))
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    assignments = pd.DataFrame({"unit_id": np.arange(len(labels)), "cluster": labels})
    for index, name in enumerate(names):
        assignments[f"feature__{name}"] = features[:, index]
    profiles = pd.DataFrame([
        {"cluster": int(cluster), "variable": name,
         "standardised_centroid": float(features[labels == cluster, index].mean()), "n": 20}
        for cluster in range(clusters) for index, name in enumerate(names)
    ])
    stability = pd.DataFrame({
        "replicate": range(1, 6), "adjusted_rand_index": [-0.5, -0.2, 0.0, 0.2, 0.4],
        "mean_adjusted_rand_index": [-0.02] * 5, "algorithm_agreement_ari": [-0.7] * 5,
    })
    return {"table:phenotype_profiles": profiles, "table:phenotype_assignments": assignments, "table:cluster_stability": stability}


def _render(tmp_path: Path, tables: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    bindings = {}
    for key, frame in tables.items():
        path = source_dir / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path, "figure")
    summary = renderer.run_cross_sectional_phenotyping_figure(
        out_dir=tmp_path / "figure", run_dir=tmp_path,
        resolved_inputs={"step_id": "figure", "inputs": bindings},
        step_id="figure", figure_product="phenotypes",
    )
    contract = json.loads((tmp_path / "figure" / "phenotypes.figure_contract.json").read_text())
    return summary, contract


def test_negative_ari_is_visible_and_projection_is_reconstructible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = {}
    original_save = renderer.save_publication_figure

    def capture(fig, *args, **kwargs):
        saved["figure"] = fig
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(renderer, "save_publication_figure", capture)
    tables = _source_tables()
    summary, contract = _render(tmp_path, tables)
    ari_axis = next(ax for ax in saved["figure"].axes if ax.get_ylabel() == "Adjusted Rand index")
    assert ari_axis.get_ylim()[0] < -0.7
    assert ari_axis.get_ylim()[1] > 0.4
    projection_axis = next(ax for ax in saved["figure"].axes if ax.get_xlabel().startswith("PC1"))
    assert projection_axis.get_aspect() == 1.0
    projection = pd.read_csv(tmp_path / "figure" / "phenotype_projection_source_data.csv")
    transform = json.loads((tmp_path / "figure" / "phenotype_projection_transform.json").read_text())
    source = tables["table:phenotype_assignments"]
    matrix = source[transform["feature_columns"]].to_numpy()
    reconstructed = (matrix - np.array(transform["mean"])) @ np.array(transform["components"]).T
    np.testing.assert_allclose(projection[["pc1", "pc2"]], reconstructed, rtol=1e-12, atol=1e-12)
    assert projection["source_row_index"].tolist() == list(range(len(source)))
    assert projection["cluster"].tolist() == source["cluster"].tolist()
    assert transform["svd_solver"] == "full" and transform["whiten"] is False
    assert transform["refit_clustering"] is False
    assert len(projection) == len(source)
    by_role = {panel["role"]: panel for panel in contract["panels"]}
    assert by_role["phenotype_structure"]["metadata"]["chart_type"] == "embedding_plot"
    assert by_role["phenotype_profile"]["metadata"]["source_products"] == ["table:phenotype_profiles"]
    assert by_role["stability"]["metadata"]["chart_type"] == "subsampling_ari"
    step = AnalysisStep(
        step_id="figure", planned_analysis_role="auxiliary", intent="Render the exact source-bound figure.",
        method="visualization", inputs=list(renderer.PHENOTYPING_FIGURE_INPUTS), expected_outputs=["figure:phenotypes"],
        input_consumption_contracts=[{"input_key": key, "mode": "all_rows"} for key in renderer.PHENOTYPING_FIGURE_INPUTS],
    )
    shaped, _ = bind_deterministic_figure_panels(plan=AnalysisPlan(research_question="Display profiles.", steps=[step]))
    assert len(shaped.steps[0].figure_panels) == 3
    assert validate_step_planned_figure_contract_binding(step=shaped.steps[0], out_dir=tmp_path / "figure", step_summary=summary) == []


@pytest.mark.parametrize("mutation", (
    "invalid_ari", "nonfinite_feature", "duplicate_profile", "wrong_centroid", "wrong_count", "wrong_mean_ari",
))
def test_figure_rejects_invalid_sources_without_sanitizing_them(tmp_path: Path, mutation) -> None:
    tables = _source_tables()
    if mutation == "invalid_ari":
        tables["table:cluster_stability"].loc[0, "adjusted_rand_index"] = -1.1
    elif mutation == "nonfinite_feature":
        tables["table:phenotype_assignments"].loc[0, "feature__marker_a"] = np.nan
    elif mutation == "wrong_centroid":
        tables["table:phenotype_profiles"].loc[0, "standardised_centroid"] += 0.1
    elif mutation == "wrong_count":
        tables["table:phenotype_profiles"].loc[0, "n"] = 25
    elif mutation == "wrong_mean_ari":
        tables["table:cluster_stability"]["mean_adjusted_rand_index"] = 0.9
    else:
        profiles = tables["table:phenotype_profiles"]
        tables["table:phenotype_profiles"] = pd.concat([profiles, profiles.iloc[:1]], ignore_index=True)
    with pytest.raises((RuntimeError, ValueError)):
        _render(tmp_path, tables)


def test_dense_clinical_profiles_preserve_readable_labels_and_editable_text(tmp_path, monkeypatch) -> None:
    names = tuple(f"{name}_first" for name in (
        "hr", "sbp", "dbp", "map", "temp", "resp", "spo2", "lact", "ph", "pco2", "po2", "fio2",
        "bicar", "na", "k", "cl", "glu", "crea", "plt", "bun", "arterial_oxygen_partial_pressure",
    ))
    saved = {}
    original_save = renderer.save_publication_figure

    def capture(fig, *args, **kwargs):
        saved["figure"] = fig
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(renderer, "save_publication_figure", capture)
    summary, contract = _render(tmp_path, _source_tables(names, clusters=6))
    figure = saved["figure"]
    assert contract["height_mm"] > 118
    assert summary["display_projection"]["n_rows"] == 120
    axis = next(ax for ax in figure.axes if ax.get_title(loc="left") == "Clinical feature profiles")
    labels = axis.get_yticklabels()
    assert len(labels) == len(names)
    assert all("First" in label.get_text() for label in labels)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    canvas_renderer = canvas.get_renderer()
    boxes = [label.get_window_extent(canvas_renderer) for label in labels]
    assert not any(left.overlaps(right) for left, right in zip(boxes, boxes[1:]))
    import xml.etree.ElementTree as ET

    tree = ET.parse(tmp_path / "figure" / "phenotypes.svg")
    texts = [element.text or "" for element in tree.iter("{http://www.w3.org/2000/svg}text")]
    assert any("Clinical feature profiles" in text for text in texts)
    assert any("First" in text for text in texts)
