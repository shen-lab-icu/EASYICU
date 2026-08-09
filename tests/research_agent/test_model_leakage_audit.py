"""Model-leakage in-run audit: a fitted model that conditions on its own
declared outcome is target leakage (error, routed through the same re-fit loop
as overadjustment); a treatment covariate or a *different* endpoint used as a
predictor is a non-gating caution (the DAG/timing is unknown).

Twin of test_overadjustment_audit.py — same harness, the outcome/treatment side.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.plan_utils import _primary_model_leakage_findings
from easyicu.research_agent.schema import AnalysisStep


def _step(step_id="06_primary_association"):
    return AnalysisStep(
        step_id=step_id,
        intent="Estimate the adjusted association.",
        planned_analysis_role="primary",
        method="logistic_regression",
        expected_outputs=["table:adjusted_association_estimates"],
    )


def _ctx(*, outcome="death_icu", exposure="sepsis3"):
    return SimpleNamespace(target_outcome=outcome, primary_exposure=exposure)


def _write_coef_table(out_dir: Path, variables, *, name="primary_association.csv"):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "coef", "odds_ratio"])
        w.writeheader()
        for v in variables:
            w.writerow({"variable": v, "coef": "0.1", "odds_ratio": "1.1"})


def test_outcome_as_predictor_is_a_gating_error(tmp_path: Path):
    # death_icu (the declared outcome) appears among predictors -> self-leakage.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "death_icu"])
    findings = _primary_model_leakage_findings(
        step=_step(), context=_ctx(), out_dir=tmp_path
    )
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    f = errors[0]
    assert f.detail["kind"] == "outcome_leakage"
    assert f.detail["offending_predictors"] == ["death_icu"]
    assert "death_icu" in f.message


def test_other_endpoint_predictor_is_a_caution_not_error(tmp_path: Path):
    # los_icu is a different endpoint used as a covariate -> timing-dependent
    # caution (warning), never a gating error.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "los_icu"])
    findings = _primary_model_leakage_findings(
        step=_step(), context=_ctx(), out_dir=tmp_path
    )
    assert all(f.severity != "error" for f in findings)
    cautions = [
        f for f in findings if f.detail.get("kind") == "outcome_leakage_caution"
    ]
    assert len(cautions) == 1
    assert cautions[0].severity == "warning"


def test_treatment_covariate_is_a_mediator_caution(tmp_path: Path):
    # A treatment/intervention covariate may be a mediator -> caution, no gate.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "furosemide"])
    findings = _primary_model_leakage_findings(
        step=_step(), context=_ctx(), out_dir=tmp_path
    )
    assert all(f.severity != "error" for f in findings)
    cautions = [
        f for f in findings if f.detail.get("kind") == "treatment_mediator_caution"
    ]
    assert len(cautions) == 1
    assert cautions[0].severity == "warning"
    assert cautions[0].detail["exposure"] == "sepsis3"


def test_clean_model_emits_nothing(tmp_path: Path):
    # Outcome only on the LHS, no endpoint/treatment covariate -> silent.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "sex", "lactate"])
    assert (
        _primary_model_leakage_findings(step=_step(), context=_ctx(), out_dir=tmp_path)
        == []
    )


def test_no_outcome_declared_no_self_leakage_error(tmp_path: Path):
    # Without a declared outcome the firm self-leakage error never fires
    # (never inferred); the endpoint caution may still scan, but death_icu here
    # is the would-be outcome so we assert no gating error specifically.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "death_icu"])
    findings = _primary_model_leakage_findings(
        step=_step(), context=_ctx(outcome=None), out_dir=tmp_path
    )
    assert all(f.detail.get("kind") != "outcome_leakage" for f in findings)


def test_no_coefficient_table_is_silent(tmp_path: Path):
    assert (
        _primary_model_leakage_findings(step=_step(), context=_ctx(), out_dir=tmp_path)
        == []
    )


def test_descriptive_outcome_table_is_not_a_model_leakage_target(
    tmp_path: Path,
):
    """Regression for E1 r25: a descriptive outcome row is not a predictor."""

    _write_coef_table(tmp_path, ["sep3_sofa2_max", "death"])
    step = AnalysisStep(
        step_id="04_prevalence_mortality",
        intent="Report prevalence and absolute mortality.",
        planned_analysis_role="auxiliary",
        method="descriptive",
        expected_outputs=["table:absolute_risk_context"],
    )

    assert (
        _primary_model_leakage_findings(
            step=step,
            context=_ctx(outcome="death", exposure="sep3_sofa2_max"),
            out_dir=tmp_path,
        )
        == []
    )
