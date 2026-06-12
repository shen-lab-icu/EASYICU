"""Overadjustment hard-block: a primary model that conditioned on a constituent
of a composite/derived exposure is an objective design error, routed through the
same in-run repair loop the exposure-contract auditor uses (re-fit, no restart).

Twin of test_exposure_contract_audit.py: that one enforces the model MUST use
the exposure; this one enforces it must NOT adjust for the exposure's own
constituents.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.plan_utils import (
    _primary_exposure_overadjustment_findings,
    read_model_covariate_names,
)


def _step(step_id="06_primary_association"):
    return SimpleNamespace(step_id=step_id)


def _ctx(required="sepsis3"):
    return SimpleNamespace(primary_exposure=required)


def _write_coef_table(out_dir: Path, variables, *, name="primary_association.csv"):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "coef", "odds_ratio"])
        w.writeheader()
        for v in variables:
            w.writerow({"variable": v, "coef": "0.1", "odds_ratio": "1.1"})


# ---------------------------------------------------------------------------
# read_model_covariate_names — content-based coefficient-table detection
# ---------------------------------------------------------------------------


def test_reader_detects_coef_table_regardless_of_filename(tmp_path: Path):
    # Real runs name this primary_association.csv / model_coefficients.csv, not
    # regression_results.csv: detection must ride the column contract.
    _write_coef_table(
        tmp_path, ["const", "age", "sofa_max"], name="model_coefficients.csv"
    )
    assert read_model_covariate_names(tmp_path) == ["age", "sofa_max"]  # const dropped


def test_reader_ignores_non_model_variable_tables(tmp_path: Path):
    # missingness.csv has a `variable` column but no coefficient column: it must
    # NOT inject phantom covariates into the overadjustment check.
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "missingness.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "missing_frac"])
        w.writeheader()
        w.writerow({"variable": "sofa_max", "missing_frac": "0.66"})
    assert read_model_covariate_names(tmp_path) == []


def test_reader_missing_dir_degrades_to_empty(tmp_path: Path):
    assert read_model_covariate_names(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# _primary_exposure_overadjustment_findings — the hard block
# ---------------------------------------------------------------------------


def test_flags_overadjustment_for_exposure_constituent(tmp_path: Path):
    # Sepsis-3 is defined via SOFA, so adjusting for SOFA is overadjustment.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "sofa_max"])
    findings = _primary_exposure_overadjustment_findings(
        step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "error"
    assert f.detail["kind"] == "overadjustment"
    assert f.detail["exposure"] == "sepsis3"
    assert f.detail["offending_covariates"] == ["sofa_max"]
    assert "sofa_max" in f.message


def test_exposure_row_itself_is_not_flagged(tmp_path: Path):
    # The exposure appears in its own coefficient table; that is correct, not
    # overadjustment, and must not be flagged.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "sex"])
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
        )
        == []
    )


def test_no_flag_without_required_exposure(tmp_path: Path):
    # Question names no exposure -> nothing to enforce (never inferred).
    _write_coef_table(tmp_path, ["const", "age", "sofa_max"])
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx(None), out_dir=tmp_path
        )
        == []
    )


def test_no_flag_when_no_coefficient_table(tmp_path: Path):
    # No model output yet -> silent, not a guess.
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
        )
        == []
    )


def test_unresolvable_derived_exposure_emits_caution_not_error(tmp_path: Path):
    # NEWS is a callback score with an empty dependency closure: the
    # deterministic check is blind, so instead of silently passing, a non-gating
    # caution (warning) is emitted to prompt manual verification.
    _write_coef_table(tmp_path, ["const", "news", "age", "sex", "heart_rate"])
    findings = _primary_exposure_overadjustment_findings(
        step=_step(), context=_ctx("news"), out_dir=tmp_path
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "warning"  # caution, never the gating "error"
    assert f.detail["kind"] == "overadjustment_caution"
    assert f.detail["exposure"] == "news"


def test_non_derived_exposure_emits_nothing(tmp_path: Path):
    # A raw lab exposure is not derived -> no caution, no error (silent).
    _write_coef_table(tmp_path, ["const", "lact", "age", "sex"])
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("lact"), out_dir=tmp_path
        )
        == []
    )


# ---------------------------------------------------------------------------
# Prevention layer: the replanner must also see the methodological principles
# (the planner already did via 94cf8db; the replanner revises the model spec,
# so dropping the guard there is exactly where overadjustment can re-enter).
# ---------------------------------------------------------------------------


def test_replanner_injects_methodological_principles(monkeypatch):
    from easyicu.research_agent import agents as A
    from easyicu.research_agent import structured_retry as SR
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    # The overadjustment principle must actually be in the shared guide. The
    # guide renders each principle's `principle` text (not its rationale), so
    # match that wording rather than the word "overadjustment".
    assert "neither constituents nor downstream" in A._PRINCIPLES_GUIDE.lower()

    plan = AnalysisPlan(
        research_question="q", steps=[AnalysisStep(step_id="01", intent="x")]
    )
    captured = {}

    def _fake_retry(llm, messages, parser, **kwargs):
        captured["system"] = messages[0].content
        return plan

    # Sidestep the heavy ResearchContext rendering; we only assert the system
    # message the replanner builds, not the user prompt.
    monkeypatch.setattr(A, "_format_context", lambda ctx: "CTX")
    monkeypatch.setattr(SR, "call_llm_with_structured_retry", _fake_retry)

    out = A.ReplannerAgent(llm=object()).run(
        context=object(), current_plan=plan, probe_summary={}, completed_step_records=[]
    )
    # The replanner returns a revised copy (revision bumped), so assert on
    # content, not identity. The point of the test is the captured system msg.
    assert out.research_question == "q"
    assert "neither constituents nor downstream" in captured["system"].lower()
