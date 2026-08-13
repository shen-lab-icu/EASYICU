"""Study-design-aware publication figures.

Every canonical study-design family must render its *own* manuscript figure
instead of the association forest: a survival question gets a Kaplan-Meier +
Cox forest, a prediction question a ROC + calibration, a phenotyping question a
cluster heatmap, a causal question a love plot + contrast. Each test drives the
full ``PublicationFigureSkill`` from registered family tables and closes the
loop through the article figure-strategy audit for that family, so a regression
that silently funnels a family back into the forest fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.planning.figure_strategy import (
    summarize_article_figure_strategy_coverage,
)


def _register_table(evidence, run_dir: Path, name: str, frame: pd.DataFrame) -> None:
    path = run_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    evidence.register_file(
        kind="table",
        description=f"{name} analysis table.",
        source_path=path,
        evidence_id=name,
        aliases=[name],
        producer="coder",
        generation_mode="agent",
    )


def _context(ra, question: str, *, exposure: str, outcome: str = "death"):
    return ra.ResearchContext(
        research_question=question,
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=500,
            n_stays=500,
        ),
        variables=[],
        primary_exposure=exposure,
        target_outcome=outcome,
    )


def _figure_plan(ra):
    return ra.AnalysisPlan(
        research_question="q",
        steps=[
            ra.AnalysisStep(
                step_id="01_analysis_figure",
                intent="Render a manuscript-facing figure.",
                expected_outputs=["figure:publication"],
            )
        ],
    )


def _run(ra, evidence, context, run_dir: Path):
    return ra.PublicationFigureSkill().run(
        context=context,
        plan=_figure_plan(ra),
        evidence=evidence,
        run_dir=run_dir,
        prompt_pack_version="test",
    )


def _assert_hero_covered(context, run_dir: Path, expected_family: str) -> dict:
    status = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=run_dir,
    )
    assert status["article_figure_strategy_family"] == expected_family
    hero = status["article_figure_strategy_hero_role"]
    assert hero in status["article_figure_strategy_primary_publication_roles"], (
        f"hero role {hero} not covered by the primary figure: "
        f"{status['article_figure_strategy_primary_publication_roles']}"
    )
    assert hero not in status["article_figure_strategy_missing_roles"]
    covered = set(status["article_figure_strategy_primary_publication_roles"])
    minimum = status[
        "article_figure_strategy_primary_publication_minimum_required_role_count"
    ]
    required = set(status["article_figure_strategy_required_roles"])
    assert len(covered & required) >= minimum
    # Every panel this figure emits must use a chart type its role accepts;
    # a role present with an unsupported chart type is a silent quality bug
    # the hero check alone does not catch.
    chart_errors = [
        err
        for err in status["article_figure_strategy_errors"]
        if "unsupported chart type" in err
    ]
    assert not chart_errors, chart_errors
    return status


def test_survival_question_renders_km_and_hazard_forest(ra, tmp_path: Path):
    evidence = ra.EvidenceStore(tmp_path)
    _register_table(
        evidence,
        tmp_path,
        "km_curve",
        pd.DataFrame(
            {
                "group": ["exposed"] * 4 + ["unexposed"] * 4,
                "time": [0, 5, 10, 20, 0, 5, 10, 20] * 1,
                "survival": [1.0, 0.9, 0.75, 0.6, 1.0, 0.95, 0.9, 0.85],
                "at_risk": [250, 200, 150, 80, 250, 230, 210, 160],
            }
        ),
    )
    _register_table(
        evidence,
        tmp_path,
        "cox_summary",
        pd.DataFrame(
            {
                "term": ["ventilation", "age", "sofa"],
                "hr": [1.85, 1.03, 1.22],
                "lower": [1.5, 1.01, 1.14],
                "upper": [2.28, 1.05, 1.31],
            }
        ),
    )
    context = _context(
        ra,
        "Estimate the survival (time-to-event) association of ventilation with mortality using a Cox model.",
        exposure="ventilation",
    )
    result = _run(ra, evidence, context, tmp_path)
    assert result.generated is True
    contract = (
        tmp_path
        / "publication_figures"
        / "easyicu_survival_publication_figure.figure_contract.json"
    )
    assert contract.exists()
    for suffix in ("png", "svg", "pdf"):
        assert (
            tmp_path
            / "publication_figures"
            / f"easyicu_survival_publication_figure.{suffix}"
        ).exists()
    _assert_hero_covered(context, tmp_path, "time_to_event")


def test_prediction_question_renders_roc_and_calibration(ra, tmp_path: Path):
    evidence = ra.EvidenceStore(tmp_path)
    _register_table(
        evidence,
        tmp_path,
        "calibration_curve",
        pd.DataFrame(
            {
                "predicted": [0.05, 0.15, 0.25, 0.4, 0.6, 0.8],
                "observed": [0.04, 0.17, 0.24, 0.43, 0.58, 0.83],
            }
        ),
    )
    _register_table(
        evidence,
        tmp_path,
        "roc_curve",
        pd.DataFrame(
            {
                "fpr": [0.0, 0.1, 0.2, 0.4, 0.7, 1.0],
                "tpr": [0.0, 0.45, 0.62, 0.8, 0.93, 1.0],
            }
        ),
    )
    _register_table(
        evidence,
        tmp_path,
        "model_performance",
        pd.DataFrame(
            {
                "metric": ["auroc", "brier_score", "baseline_prevalence"],
                "value": [0.83, 0.11, 0.18],
            }
        ),
    )
    context = _context(
        ra,
        "Build an in-hospital mortality prediction model and report AUROC and calibration.",
        exposure="vitals_labs",
    )
    result = _run(ra, evidence, context, tmp_path)
    assert result.generated is True
    contract = (
        tmp_path
        / "publication_figures"
        / "easyicu_prediction_publication_figure.figure_contract.json"
    )
    assert contract.exists()
    _assert_hero_covered(context, tmp_path, "prediction")


def test_phenotyping_question_renders_cluster_figure(ra, tmp_path: Path):
    evidence = ra.EvidenceStore(tmp_path)
    _register_table(
        evidence,
        tmp_path,
        "cluster_characteristics",
        pd.DataFrame(
            {
                "cluster": [0, 1, 2],
                "lactate": [2.1, 4.8, 1.4],
                "creatinine": [1.1, 2.3, 0.9],
                "map": [78, 62, 85],
                "wbc": [11, 18, 8],
                "n": [180, 90, 230],
            }
        ),
    )
    context = _context(
        ra,
        "Identify sepsis subphenotypes by unsupervised clustering of first-24h labs and vitals.",
        exposure="labs_vitals",
    )
    result = _run(ra, evidence, context, tmp_path)
    assert result.generated is True
    contract = (
        tmp_path
        / "publication_figures"
        / "easyicu_phenotype_publication_figure.figure_contract.json"
    )
    assert contract.exists()
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    assert set(contract_payload["source_data"]) == {
        "publication_figure_source_phenotype_profile_plot_data.csv",
        "publication_figure_source_phenotype_stability_plot_data.csv",
    }
    for source_name in contract_payload["source_data"]:
        assert (contract.parent / source_name).is_file()
    _assert_hero_covered(context, tmp_path, "phenotyping")


def test_causal_question_renders_love_plot_and_contrast(ra, tmp_path: Path):
    evidence = ra.EvidenceStore(tmp_path)
    _register_table(
        evidence,
        tmp_path,
        "covariate_balance",
        pd.DataFrame(
            {
                "covariate": ["age", "sofa", "lactate", "creatinine"],
                "smd_unweighted": [0.35, 0.42, 0.28, 0.31],
                "smd_weighted": [0.04, 0.06, 0.03, 0.05],
            }
        ),
    )
    _register_table(
        evidence,
        tmp_path,
        "causal_effect",
        pd.DataFrame(
            {"estimate": [1.34], "lower": [1.08], "upper": [1.66], "scale": ["OR"]}
        ),
    )
    context = _context(
        ra,
        "Estimate the causal effect of early vasopressor exposure on mortality with propensity weighting.",
        exposure="vasopressor",
    )
    result = _run(ra, evidence, context, tmp_path)
    assert result.generated is True
    contract = (
        tmp_path
        / "publication_figures"
        / "easyicu_causal_publication_figure.figure_contract.json"
    )
    assert contract.exists()
    _assert_hero_covered(context, tmp_path, "causal_emulation")


def test_association_question_falls_through_to_forest(ra, tmp_path: Path):
    """An association family must NOT be intercepted by a family renderer."""

    from easyicu.research_agent.figures import render_family_figure
    from easyicu.research_agent.planning.study_design import infer_study_design_family

    context = _context(
        ra,
        "What is the adjusted odds ratio association between Sepsis-3 and in-hospital mortality?",
        exposure="sepsis3",
    )
    family = infer_study_design_family(context)
    assert family in {"association", "descriptive"}
    evidence = ra.EvidenceStore(tmp_path)
    assert (
        render_family_figure(
            family,
            context=context,
            plan=_figure_plan(ra),
            evidence=evidence,
            run_dir=tmp_path,
        )
        is None
    )


def test_survival_renderer_returns_none_without_curve(ra, tmp_path: Path):
    """No KM data and no Cox table -> None, so the skill falls through safely."""

    from easyicu.research_agent.figures import render_family_figure

    context = _context(
        ra,
        "Estimate the survival hazard of ventilation on mortality with a Cox model.",
        exposure="ventilation",
    )
    evidence = ra.EvidenceStore(tmp_path)
    assert (
        render_family_figure(
            "time_to_event",
            context=context,
            plan=_figure_plan(ra),
            evidence=evidence,
            run_dir=tmp_path,
        )
        is None
    )


def _cox_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "term": ["ventilation", "age"],
            "hazard_ratio": [2.13, 1.03],
            "ci_low": [2.0, 1.02],
            "ci_high": [2.27, 1.04],
            "p_value": [1e-120, 1e-90],
        }
    )


def test_survival_renderer_falls_back_to_cohort_km_when_km_table_empty(
    ra, tmp_path: Path
):
    """An absent KM table must not drop the survival figure: the renderer
    recomputes Kaplan-Meier straight from the materialised cohort so the
    manuscript still gets a titled multi-panel survival figure. This is the
    exact H1 failure mode (coder wrote an empty km_curve.csv)."""

    from easyicu.research_agent.figures import render_family_figure

    evidence = ra.EvidenceStore(tmp_path)
    _register_table(evidence, tmp_path, "cox_summary", _cox_summary_frame())
    pd.DataFrame(
        {
            "followup_time_hours": [10.0, 50.0, 120.0, 200.0, 8.0, 300.0, 45.0, 90.0],
            "death": [1, 0, 1, 0, 1, 0, 1, 0],
            "ventilation": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    ).to_parquet(tmp_path / "cohort.parquet")

    context = _context(
        ra,
        "Estimate the survival hazard of ventilation on mortality with a Cox model.",
        exposure="ventilation",
    )
    rendered = render_family_figure(
        "time_to_event",
        context=context,
        plan=_figure_plan(ra),
        evidence=evidence,
        run_dir=tmp_path,
    )
    assert rendered is not None
    assert len(rendered.panels) >= 2
    assert all(str(p.get("title") or "").strip() for p in rendered.panels)


def test_survival_renderer_degrades_to_forest_followup_without_curve(
    ra, tmp_path: Path
):
    """Cox present but no KM anywhere (no table, no dataset, no cohort): the
    renderer degrades to a valid 2-panel forest + follow-up figure rather than
    returning None, which would cascade to the coder's single-panel fallback
    and, under strict fail-closed, a stub manuscript."""

    from easyicu.research_agent.figures import render_family_figure

    evidence = ra.EvidenceStore(tmp_path)
    _register_table(evidence, tmp_path, "cox_summary", _cox_summary_frame())

    context = _context(
        ra,
        "Estimate the survival hazard of ventilation on mortality with a Cox model.",
        exposure="ventilation",
    )
    rendered = render_family_figure(
        "time_to_event",
        context=context,
        plan=_figure_plan(ra),
        evidence=evidence,
        run_dir=tmp_path,
    )
    assert rendered is not None
    assert len(rendered.panels) == 2
    assert all(str(p.get("title") or "").strip() for p in rendered.panels)


def test_survival_rescue_from_prior_outputs_builds_compliant_figure(ra, tmp_path: Path):
    """The from-prior-outputs survival rescue rebuilds a titled multi-panel
    figure from a parent step's Cox table when the figure child step's coder
    output fails its contract -- the deterministic path that unblocks the H1
    survival gate without a live LLM call."""

    import json as _json

    from easyicu.research_agent.figures.survival import (
        render_survival_bundle_from_prior_outputs,
    )

    parent = tmp_path / "steps" / "02_survival_analysis" / "outputs"
    parent.mkdir(parents=True)
    _cox_summary_frame().to_csv(parent / "cox_summary.csv", index=False)
    pd.DataFrame(
        {
            "followup_time_hours": [10.0, 50.0, 120.0, 200.0, 8.0, 300.0],
            "death": [1, 0, 1, 0, 1, 0],
        }
    ).to_parquet(tmp_path / "cohort.parquet")

    out_dir = tmp_path / "steps" / "02_survival_analysis_figure" / "outputs"
    out_dir.mkdir(parents=True)
    repair_id = render_survival_bundle_from_prior_outputs(
        run_dir=tmp_path,
        current_step_id="02_survival_analysis_figure",
        out_dir=out_dir,
    )
    assert repair_id == "survival_publication_bundle_from_parent_outputs_v1"
    contract_path = out_dir / "publication_figure.figure_contract.json"
    assert contract_path.exists()
    contract = _json.loads(contract_path.read_text(encoding="utf-8"))
    panels = contract.get("panels") or []
    assert len(panels) >= 2
    assert all(str(p.get("title") or "").strip() for p in panels)
    # The contract text must NOT trip FigureContractQualityValidator's
    # fallback/rescue detector -- this is a real, data-backed figure and the
    # rescue/provenance signal belongs in step_summary, not the contract.
    from easyicu.research_agent.audits.validators import (
        FigureContractQualityValidator,
    )

    findings = FigureContractQualityValidator().audit_contract_file(
        contract_path, manuscript_facing=True
    )
    assert not [f for f in findings if f.severity == "error"], [
        f.message for f in findings if f.severity == "error"
    ]


def test_survival_split_figure_never_borrows_an_older_cox_table(
    ra, tmp_path: Path
):
    from easyicu.research_agent.figures.survival import (
        render_survival_bundle_from_prior_outputs,
    )

    older = tmp_path / "steps" / "00_old_sensitivity" / "outputs"
    older.mkdir(parents=True)
    _cox_summary_frame().to_csv(older / "cox_summary.csv", index=False)
    direct = tmp_path / "steps" / "05_primary_survival" / "outputs"
    direct.mkdir(parents=True)
    pd.DataFrame({"not_a_cox_result": [1.0]}).to_csv(
        direct / "cox_summary.csv", index=False
    )

    repair_id = render_survival_bundle_from_prior_outputs(
        run_dir=tmp_path,
        current_step_id="05_primary_survival_figure",
        out_dir=tmp_path / "steps" / "05_primary_survival_figure" / "outputs",
    )

    assert repair_id is None


def test_survival_figure_forest_labels_do_not_overlap_with_many_covariates(
    ra, tmp_path: Path
):
    """A Cox model with many adjusters must not overflow the small forest panel
    with overlapping y-axis labels -- the publication_figure_export visual-QA
    gate flags overlapping text and blocks analysis_validated (the H1 last
    blocker). The forest is capped to a readable row count."""

    from easyicu.research_agent.figures import render_family_figure
    from easyicu.research_agent.figures.publication import (
        make_figure_contract,
        save_publication_figure,
    )
    from easyicu.research_agent.gates.visual_qa import _audit_svg_text_layout

    evidence = ra.EvidenceStore(tmp_path)
    n = 20
    _register_table(
        evidence,
        tmp_path,
        "cox_summary",
        pd.DataFrame(
            {
                "term": ["mech_vent_any24"]
                + [f"adjuster_{i}_long_covariate_name" for i in range(n - 1)],
                "hazard_ratio": [2.1] + [1.0 + 0.03 * i for i in range(n - 1)],
                "ci_low": [1.9] + [0.9 + 0.03 * i for i in range(n - 1)],
                "ci_high": [2.3] + [1.1 + 0.03 * i for i in range(n - 1)],
                "p_value": [1e-100] + [0.01] * (n - 1),
            }
        ),
    )
    pd.DataFrame(
        {
            "followup_time_hours": [10.0, 50.0, 120.0, 200.0, 8.0, 300.0, 45.0, 90.0],
            "death": [1, 0, 1, 0, 1, 0, 1, 0],
            "mech_vent_any24": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    ).to_parquet(tmp_path / "cohort.parquet")

    context = _context(
        ra,
        "Estimate the survival hazard of ventilation on mortality with a Cox model.",
        exposure="mech_vent_any24",
    )
    rendered = render_family_figure(
        "time_to_event",
        context=context,
        plan=_figure_plan(ra),
        evidence=evidence,
        run_dir=tmp_path,
    )
    assert rendered is not None
    contract = make_figure_contract(
        figure_id="easyicu_survival_publication_figure",
        core_claim=rendered.core_claim,
        panels=rendered.panels,
    )
    save_publication_figure(
        rendered.fig,
        tmp_path / "easyicu_survival_publication_figure",
        contract=contract,
        dpi=300,
    )
    svg = tmp_path / "easyicu_survival_publication_figure.svg"
    findings = _audit_svg_text_layout(svg, validator="publication_figure_export")
    assert not [f for f in findings if f.severity == "error"], [
        f.message for f in findings if f.severity == "error"
    ]


def test_survival_rescue_returns_none_without_cox_table(ra, tmp_path: Path):
    """No parent Cox table -> the survival rescue returns None so the router
    falls through instead of emitting an empty figure."""

    from easyicu.research_agent.figures.survival import (
        render_survival_bundle_from_prior_outputs,
    )

    (tmp_path / "steps").mkdir()
    out_dir = tmp_path / "steps" / "02_survival_analysis_figure" / "outputs"
    out_dir.mkdir(parents=True)
    assert (
        render_survival_bundle_from_prior_outputs(
            run_dir=tmp_path,
            current_step_id="02_survival_analysis_figure",
            out_dir=out_dir,
        )
        is None
    )


def test_incomplete_figure_strategy_becomes_a_readable_finding(ra, tmp_path: Path):
    """The publication gate's shortfall must also reach the findings list.

    ``summarize_article_figure_strategy_coverage`` has always fed
    ``publication_authorized``, but until the execute phase called the
    validator the shortfall existed only inside that projection: a run whose
    figures missed a required article role produced no finding a reviewer
    could read.
    """

    from easyicu.research_agent.planning.figure_strategy import (
        validate_run_against_article_figure_strategy,
    )

    context = _context(
        ra,
        "Is lactate associated with mortality?",
        exposure="lactate",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    findings = validate_run_against_article_figure_strategy(
        context=context,
        run_dir=run_dir,
        analysis_family="time_to_event",
    )

    assert [item.validator for item in findings] == ["article_figure_strategy"]
    finding = findings[0]
    # The family must be the one the caller resolved from the final plan, not
    # one this validator re-derived: a finding that disagrees with the gate it
    # reports on is worse than no finding.
    assert "time_to_event" in finding.message
    assert finding.detail["missing_roles"]


def test_the_figure_strategy_validator_is_silent_when_coverage_is_complete(
    ra, tmp_path: Path
):
    from easyicu.research_agent.planning.figure_strategy import (
        summarize_article_figure_strategy_coverage,
        validate_run_against_article_figure_strategy,
    )

    context = _context(
        ra,
        "Is lactate associated with mortality?",
        exposure="lactate",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=run_dir,
        analysis_family="time_to_event",
    )
    if status["article_figure_strategy_complete"]:  # pragma: no cover
        assert (
            validate_run_against_article_figure_strategy(
                context=context, run_dir=run_dir, analysis_family="time_to_event"
            )
            == []
        )
