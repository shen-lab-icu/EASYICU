"""Typed distribution figures retain interval and counts-only authority."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.figures.publication import make_figure_contract


def _e1_typed_distribution_frame(*, include_risk_difference: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_role": ["exposure_level", "exposure_level", "overall"],
            "exposure_level_index": [0, 1, None],
            "exposure_level": [0, 1, None],
            "n_rows": [62_900, 31_600, 94_500],
            # Deliberately differ from n_rows: Panel B must use the analysed
            # outcome denominator, not whichever structural count comes first.
            "outcome_denominator": [62_862, 31_596, 94_458],
            "outcome_rate_pct": [8.210047, 13.625142, 10.021385],
            "ci_low_pct": [7.995860, 13.245177, 9.830581],
            "ci_high_pct": [8.424235, 14.005108, 10.212189],
            "exposure_column": ["sep3_sofa1_max"] * 3,
            "confidence_level": [0.95] * 3,
        }
    )
    if include_risk_difference:
        frame = frame.assign(
            risk_difference_pct=5.415095,
            risk_difference_ci_low_pct=4.977273,
            risk_difference_ci_high_pct=5.852917,
            risk_difference_reference_index=0,
            risk_difference_comparison_index=1,
            risk_difference_effect_measure="risk_difference",
        )
    return frame


def test_e1_typed_distribution_uses_authorized_risk_difference_not_level_codes(ra):
    from easyicu.research_agent.figures.skill import (
        _association_axis_metadata,
        _normalise_association_frame,
    )

    normalized = _normalise_association_frame(
        _e1_typed_distribution_frame(include_risk_difference=True),
        primary_exposure="sep3_sofa1_max",
        display_labels={"sep3_sofa1_max": "Sepsis-3"},
    )

    assert normalized["estimate"].tolist() == pytest.approx([5.415095])
    assert normalized["lower"].tolist() == pytest.approx([4.977273])
    assert normalized["upper"].tolist() == pytest.approx([5.852917])
    assert normalized["label"].tolist() == ["Sepsis-3: Exposed vs Unexposed"]
    assert not set(normalized["estimate"]) & {0.0, 1.0}
    axis = _association_axis_metadata(normalized)
    assert axis["null_value"] == 0.0
    assert axis["ratio_scale"] is False
    assert axis["xlabel"] == "Risk difference (percentage points)"


def test_binary_level_display_labels_reach_adjusted_and_distribution_panels(ra):
    from easyicu.research_agent.figures.skill import (
        _normalise_association_frame,
        _normalise_strata_frame,
        _strata_score_label,
    )

    labels = {
        "marker_a=0": "Marker A absent",
        "marker_a=1": "Marker A present",
    }
    adjusted = _normalise_association_frame(
        pd.DataFrame(
            {
                "exposure": ["marker_a"],
                "estimate": [1.4],
                "ci_low": [1.2],
                "ci_high": [1.6],
            }
        ),
        primary_exposure="marker_a",
        display_labels=labels,
    )
    distribution = _e1_typed_distribution_frame(
        include_risk_difference=True
    ).assign(exposure_column="marker_a")
    strata = _normalise_strata_frame(distribution, display_labels=labels)

    assert adjusted["label"].tolist() == [
        "Marker A present vs Marker A absent"
    ]
    assert strata["score"].tolist() == ["Marker A absent", "Marker A present"]
    assert _strata_score_label(strata) == "Marker A status"


def test_e1_typed_distribution_without_authorized_contrast_fails_closed(ra):
    from easyicu.research_agent.figures.skill import _normalise_association_frame

    incomplete_typed = _normalise_association_frame(
        _e1_typed_distribution_frame(include_risk_difference=False),
        primary_exposure="sep3_sofa1_max",
    )
    arbitrary_numeric = _normalise_association_frame(
        pd.DataFrame(
            {
                "exposure_level": [0, 1],
                "n_rows": [700, 300],
                "outcome_rate_pct": [8.0, 14.0],
            }
        )
    )

    assert incomplete_typed.empty
    assert arbitrary_numeric.empty


def test_counts_only_descriptive_bundle_is_promoted_without_invented_contrast(
    ra,
    tmp_path: Path,
):
    from PIL import Image

    from easyicu.research_agent.contracts.figure_plan import (
        EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS,
    )
    from easyicu.research_agent.planning.study_design import (
        infer_study_design_family,
    )

    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    primary_step_id = "primary_exposure_outcome_distribution"
    figure_step_id = "05_primary_result_figure"

    distribution = tmp_path / "exposure_outcome_distribution.csv"
    counts_only = _e1_typed_distribution_frame(
        include_risk_difference=False
    ).assign(
        ci_low_pct=pd.NA,
        ci_high_pct=pd.NA,
        confidence_level=pd.NA,
    )
    counts_only.to_csv(distribution, index=False)
    evidence.register_file(
        kind="table",
        description="Counts-only E1 exposure/outcome distribution.",
        source_path=distribution,
        evidence_id="table_step_artifact_distribution",
        produced_by_step=primary_step_id,
        producer="runner",
        generation_mode="deterministic_standard",
    )

    outputs = run_dir / "steps" / figure_step_id / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    svg = outputs / "primary_result.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="340">'
        '<rect width="720" height="340" fill="white"/>'
        '<text x="20" y="30">Counts-only descriptive result</text>'
        "</svg>",
        encoding="utf-8",
    )
    png = outputs / "primary_result.png"
    Image.new("RGB", (720, 340), "white").save(png)
    source = outputs / "primary_result_input_source_data.csv"
    counts_only.to_csv(source, index=False)
    templates = EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS
    contract = make_figure_contract(
        figure_id="figure:primary_result",
        core_claim=(
            "Exposure prevalence and observed outcome rates are reproduced "
            "from one registered counts-only distribution table."
        ),
        panels=[
            {
                "panel_id": template.panel_id,
                "title": (
                    "Exposure distribution"
                    if index == 0
                    else "Outcome rate by exposure"
                ),
                "role": template.article_role,
                "chart_type": template.chart_type,
                "claim": "Observed counts and denominators are displayed.",
                "evidence_ids": [source.name],
            }
            for index, template in enumerate(templates)
        ],
        source_data=[source.name],
        statistics_note=(
            "Counts and observed percentages only; no uncertainty or contrast "
            "is computed."
        ),
    )
    contract_path = outputs / "primary_result.figure_contract.json"
    contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")
    metadata = {
        "figure_role": "publication_figure",
        "step_id": figure_step_id,
    }
    for path, kind, evidence_id in (
        (svg, "figure", "figure_primary_result_svg"),
        (png, "figure", "figure_primary_result_png"),
        (contract_path, "log", "log_primary_result_contract"),
        (source, "table", "table_primary_result_source"),
    ):
        evidence.register_file(
            kind=kind,
            description="Registered counts-only E1 result bundle.",
            source_path=path,
            evidence_id=evidence_id,
            produced_by_step=figure_step_id,
            producer="runner",
            generation_mode="deterministic_standard",
            metadata=metadata if kind != "table" else {"step_id": figure_step_id},
        )

    context = ra.ResearchContext(
        research_question=(
            "Describe exposure prevalence and observed outcome rates across "
            "the declared exposure levels."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="MIMIC-IV ICU stays",
            database="mimiciv",
            n_patients=65_366,
            n_stays=94_458,
        ),
        variables=[],
        primary_exposure="sep3_sofa2_max",
        target_outcome="death",
    )
    assert infer_study_design_family(context) == "descriptive"
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id=primary_step_id,
                intent="Report the declared counts-only distribution.",
                expected_outputs=["table:exposure_outcome_distribution"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id=figure_step_id,
                intent="Render the primary descriptive result.",
                inputs=["table:exposure_outcome_distribution"],
                expected_outputs=["figure:primary_result"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    from easyicu.research_agent.figures.skill import (
        _PrimaryLineageEvidenceView,
        _bundle_primary_strategy_ready,
        _declared_primary_lineage_step_ids,
        _select_existing_step_publication_figure_bundle,
    )

    lineage = _PrimaryLineageEvidenceView(
        evidence,
        _declared_primary_lineage_step_ids(plan),
    )
    bundle = _select_existing_step_publication_figure_bundle(lineage)
    assert bundle is not None
    assert _bundle_primary_strategy_ready(context, bundle), bundle[
        "contract_payload"
    ]

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    assert not (
        run_dir
        / "publication_figures"
        / "publication_figure_source_primary_association.csv"
    ).exists()
    summary = json.loads(
        (
            run_dir
            / "evidence"
            / "publication_figure_skill_summary__publication_figure_skill_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["generation_mode"] == "promoted_step_publication_figure"
    promoted = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in promoted["panels"]] == [
        "distribution",
        "descriptive_result",
    ]
    assert "no uncertainty or contrast is computed" in promoted["statistics_note"]


def test_e1_typed_distribution_panel_b_uses_rate_ci_and_outcome_denominator(ra):
    from easyicu.research_agent.figures.skill import _draw_strata_panel
    from easyicu.research_agent.figures.strata import normalise_strata_frame

    normalized = normalise_strata_frame(
        _e1_typed_distribution_frame(include_risk_difference=True)
    )

    assert normalized["score"].tolist() == ["Unexposed", "Exposed"]
    assert normalized["rate"].tolist() == pytest.approx([0.08210047, 0.13625142])
    assert normalized["lower"].tolist() == pytest.approx([0.07995860, 0.13245177])
    assert normalized["upper"].tolist() == pytest.approx([0.08424235, 0.14005108])
    assert normalized["n"].tolist() == [62_862, 31_596]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_strata_panel(
        ax,
        normalized,
        palette={"blue": "#0F4D92", "neutral_light": "#D8D8D8"},
        outcome_label="In-hospital mortality",
    )
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["Unexposed (n=62,862)", "Exposed (n=31,596)"]
    # One collection comes from the lollipop stems and another from the
    # horizontal confidence intervals.  A point-only panel has only the former.
    assert len(ax.collections) >= 2
    plt.close(fig)


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_publication_figure_skill_e1_typed_distribution_projects_both_panels(
    ra,
    tmp_path: Path,
    confidence: float,
):
    run_dir = tmp_path / "run"
    evidence = ra.EvidenceStore(run_dir)
    distribution = tmp_path / "exposure_outcome_distribution.csv"
    _e1_typed_distribution_frame(include_risk_difference=True).assign(
        confidence_level=confidence,
    ).to_csv(
        distribution,
        index=False,
    )
    evidence.register_file(
        kind="table",
        description="E1 typed exposure/outcome distribution.",
        source_path=distribution,
        evidence_id="exposure_outcome_distribution",
        produced_by_step="03_distribution",
    )
    missingness = tmp_path / "measurement_missingness.csv"
    missingness.write_text(
        "variable,missing_fraction\nage,0.0\ndeath,0.0\n",
        encoding="utf-8",
    )
    evidence.register_file(
        kind="table",
        description="E1 measurement missingness.",
        source_path=missingness,
        evidence_id="measurement_missingness",
        produced_by_step="02_table_one",
    )
    context = ra.ResearchContext(
        research_question="Describe Sepsis-3 and in-hospital mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="MIMIC-IV ICU stays",
            database="mimiciv",
            n_patients=65_366,
            n_stays=94_458,
        ),
        variables=[],
        primary_exposure="sep3_sofa1_max",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        display_labels={
            "sep3_sofa1_max=0": "Sepsis-3 absent",
            "sep3_sofa1_max=1": "Sepsis-3 present",
            "age": "ICU入科时年龄（岁）",
            "death": "院内死亡",
        },
        steps=[
            ra.AnalysisStep(
                step_id="03_distribution",
                intent="Estimate the prespecified descriptive contrast.",
                expected_outputs=["table:exposure_outcome_distribution"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="04_figure",
                intent="Render the publication figure.",
                inputs=["table:exposure_outcome_distribution"],
                expected_outputs=["figure:publication_figure"],
                planned_analysis_role="auxiliary",
            ),
        ],
    )

    result = ra.PublicationFigureSkill().run(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )

    assert result.generated is True
    assert result.findings == []
    primary_source = pd.read_csv(
        run_dir
        / "publication_figures"
        / "publication_figure_source_primary_association.csv"
    )
    strata_source = pd.read_csv(
        run_dir
        / "publication_figures"
        / "publication_figure_source_stratified_outcome.csv"
    )
    assert primary_source["estimate"].tolist() == pytest.approx([5.415095])
    assert primary_source["label"].tolist() == [
        "Sepsis-3 present vs Sepsis-3 absent"
    ]
    assert not set(primary_source["estimate"]) & {0.0, 1.0}
    assert strata_source["rate"].tolist() == pytest.approx([0.08210047, 0.13625142])
    assert strata_source["n"].tolist() == [62_862, 31_596]
    assert strata_source["score"].tolist() == [
        "Sepsis-3 absent",
        "Sepsis-3 present",
    ]
    contract = json.loads(
        (
            run_dir
            / "publication_figures"
            / "easyicu_publication_figure.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["title"] == "Unadjusted risk difference"
    assert "percentage points" in contract["panels"][0]["claim"]
    assert f"{100 * confidence:g}% confidence interval" in contract["panels"][0]["claim"]
    assert primary_source["confidence_level"].tolist() == [confidence]
    assert strata_source["confidence_level"].tolist() == [confidence, confidence]
    assert "analysed denominators" in contract["panels"][1]["claim"]
    assert "relative estimate" not in json.dumps(contract)
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
        "data_quality",
    ]
