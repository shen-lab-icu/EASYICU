"""Exposure-contract auditor: catch a primary model that estimated the wrong
exposure, so the in-run repair loop re-fits it (no full pipeline restart).

Regression for E1 20260611: the agent misread sep3_sofa2's NA (= not septic)
as 66% missingness, dropped Sepsis-3, and modelled SOFA instead.
"""
from types import SimpleNamespace

from easyicu.research_agent.plan_utils import (
    _exposure_names_match,
    _primary_exposure_contract_findings,
)


def _step(step_id="06_primary_association"):
    return SimpleNamespace(step_id=step_id)


def _ctx(required="sepsis3"):
    return SimpleNamespace(primary_exposure=required)


def test_flags_wrong_exposure_on_primary_model():
    summary = {"primary_predictor": "sofa_max_int", "primary_or": 1.29}
    findings = _primary_exposure_contract_findings(
        step=_step(), step_summary=summary, context=_ctx("sepsis3")
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "error"
    assert f.detail["kind"] == "exposure_contract"
    assert f.detail["required_exposure"] == "sepsis3"
    assert f.detail["actual_predictor"] == "sofa_max_int"
    assert "sepsis3" in f.message


def test_no_flag_when_exposure_matches():
    summary = {"primary_predictor": "sepsis3", "primary_or": 0.8}
    assert (
        _primary_exposure_contract_findings(
            step=_step(), step_summary=summary, context=_ctx("sepsis3")
        )
        == []
    )


def test_no_flag_for_related_name():
    summary = {"primary_predictor": "sep3_sofa2_max", "odds_ratio": 0.8}
    assert (
        _primary_exposure_contract_findings(
            step=_step(), step_summary=summary, context=_ctx("sepsis3")
        )
        == []
    )


def test_no_flag_without_required_exposure():
    # descriptive/QC task: question names no exposure -> nothing to enforce
    summary = {"primary_predictor": "sofa_max_int", "primary_or": 1.29}
    assert (
        _primary_exposure_contract_findings(
            step=_step(), step_summary=summary, context=_ctx(None)
        )
        == []
    )


def test_no_flag_when_step_is_not_an_association_model():
    # a predictor name but no association-effect estimate -> not the primary
    # model step; don't double-flag the omitted-predictor case
    summary = {"primary_predictor": "sofa_max_int"}
    assert (
        _primary_exposure_contract_findings(
            step=_step("02_missingness"), step_summary=summary, context=_ctx("sepsis3")
        )
        == []
    )


def test_exposure_names_match_table():
    assert _exposure_names_match("sepsis3", "sofa_max_int") is False
    assert _exposure_names_match("sepsis3", "sepsis3") is True
    assert _exposure_names_match("sepsis3", "sepsis3_binary") is True
    assert _exposure_names_match("vasopressor", "norepinephrine") is False
