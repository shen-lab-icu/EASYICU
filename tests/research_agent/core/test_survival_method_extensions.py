"""Synthetic reference checks for the sealed survival owner's estimands."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from easyicu.research_agent.execution.runners import survival_primary_executor as owner
from easyicu.research_agent.methods.competing_risks import estimate_cif
from easyicu.research_agent.methods.gray_test import gray_test
from tests.support.survival_primary import (
    _context,
    _requirement,
    _runtime_kwargs,
    _step,
)


def competing_step():
    step = _step()
    payload = step.family_primary_result_requirement.model_dump()
    payload.update(
        competing_risk_strategy="cause_specific_and_subdistribution",
        competing_event_value=2, competing_event_definition="terminal competing outcome",
    )
    step.family_primary_result_requirement = type(_requirement()).model_validate(payload)
    return step


def cohort():
    rng = np.random.default_rng(418)
    n = 180
    return pd.DataFrame({
        "treatment": rng.integers(0, 2, n),
        "age": rng.normal(0, 1, n),
        "follow_up_days": rng.uniform(0.1, 36, n),
        "death": rng.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3]),
    })


def test_survival_owner_import_does_not_require_optional_fit_libraries():
    code = '''
import importlib.abc
import sys
class NoLifelines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lifelines" or fullname.startswith("lifelines."):
            raise ModuleNotFoundError("lifelines deliberately absent")
sys.meta_path.insert(0, NoLifelines())
from easyicu.research_agent.execution.runners.survival_primary_executor import survival_primary_executor_scaffold
assert callable(survival_primary_executor_scaffold)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def run(tmp_path, monkeypatch, frame, *, competing=False):
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path))
    kwargs = _runtime_kwargs(
        _requirement(), input_path=path,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    if competing:
        kwargs.update(
            competing_risk_strategy="cause_specific_and_subdistribution",
            competing_event_value=2,
            competing_event_definition="terminal competing outcome",
        )
    return owner.run_survival_primary(**kwargs, emit_step_summary=False)


def test_logrank_is_computed_on_the_cox_complete_cases_and_horizon(tmp_path, monkeypatch):
    scaffold = owner.survival_primary_executor_scaffold(_step())
    assert scaffold.body == ""
    frame = cohort().replace({"death": {2: 0}})
    frame.loc[0, "age"] = np.nan
    summary = run(tmp_path, monkeypatch, frame)
    table = pd.read_csv(tmp_path / summary["output_files"]["table:survival_logrank"])
    used = frame.dropna().copy()
    used["death"] *= (used.follow_up_days <= 28).astype(int)
    used["follow_up_days"] = used.follow_up_days.clip(upper=28)
    a, b = (used.loc[used.treatment == g] for g in (0, 1))
    expected = logrank_test(a.follow_up_days, b.follow_up_days, a.death, b.death)
    assert table.statistic.to_numpy() == pytest.approx(expected.test_statistic)
    assert table.p_value.to_numpy() == pytest.approx(expected.p_value)
    assert table.df.tolist() == [1, 1]
    assert table.n.sum() == len(used) == summary["n_analysis"]
    assert table.observed_events.sum() == used.death.sum()
    assert table.expected_events.sum() == pytest.approx(used.death.sum())
    receipt = json.loads((tmp_path / "survival_analysis_receipt.json").read_text())
    product = "table:survival_logrank"
    assert receipt["supplementary_sha256"][product] == hashlib.sha256(
        (tmp_path / summary["output_files"][product]).read_bytes()
    ).hexdigest()


def test_logrank_matches_hand_computed_tied_two_by_two_example():
    from easyicu.research_agent.execution.runners.survival_primary_executor import logrank_by_group

    table = logrank_by_group([1, 1, 2, 2], [1, 1, 0, 0], ["a", "a", "b", "b"])
    assert table.statistic.tolist() == pytest.approx([3, 3])
    assert table.observed_events.tolist() == [2, 0]
    assert table.expected_events.tolist() == pytest.approx([1, 1])


def test_logrank_matches_lifelines_for_three_groups():
    from easyicu.research_agent.execution.runners.survival_primary_executor import logrank_by_group

    times = [1, 2, 4, 5, 1, 3, 4, 6, 2, 3, 6, 7]
    events = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    result = logrank_by_group(times, events, groups)
    expected = multivariate_logrank_test(times, groups, events)
    assert result.statistic.to_numpy() == pytest.approx(expected.test_statistic)
    assert result.p_value.to_numpy() == pytest.approx(expected.p_value)
    assert result.df.tolist() == [2, 2, 2]
    assert result.expected_events.sum() == pytest.approx(sum(events))


@pytest.mark.parametrize("times,events,groups", [
    ([1, 2], [1], ["a", "b"]),
    ([1, np.inf], [1, 0], ["a", "b"]),
    ([1, 2], [1, 2], ["a", "b"]),
    ([1, 2], [1, 0], ["a", "a"]),
    ([1, 2], [1, 0], ["a", None]),
])
def test_logrank_refuses_an_unclosed_group_comparison(times, events, groups):
    with pytest.raises(ValueError):
        owner.logrank_by_group(times, events, groups)


def test_declared_competition_is_owned_and_matches_existing_kernels(tmp_path, monkeypatch):
    step = competing_step()
    assert owner.survival_primary_executor_owns_step(step)
    frame = cohort()
    frame.loc[0, "age"] = np.nan
    summary = run(tmp_path, monkeypatch, frame, competing=True)
    used = frame.dropna().copy()
    used.loc[used.follow_up_days > 28, "death"] = 0
    used["follow_up_days"] = used.follow_up_days.clip(upper=28)
    products = summary["output_files"]
    cif = pd.read_csv(tmp_path / products["table:survival_cumulative_incidence"])
    for group, subset in used.groupby("treatment"):
        expected = estimate_cif(subset.follow_up_days, subset.death, 1)
        actual = cif.loc[cif.group == group]
        assert actual.time.to_numpy() == pytest.approx(expected.times)
        assert actual.cumulative_incidence.to_numpy() == pytest.approx(expected.cif)
    gray = pd.read_csv(tmp_path / products["table:survival_gray_test"]).iloc[0]
    reference = gray_test(used.follow_up_days, used.death, used.treatment)
    assert gray.statistic == pytest.approx(reference.statistic)
    assert gray.p_value == pytest.approx(reference.p_value)
    assert gray.df == 1
    cs = used.rename(columns={"treatment": "treatment__is_1"})
    cs["death"] = cs.death.eq(1).astype(int)
    fit = CoxPHFitter().fit(cs, duration_col="follow_up_days", event_col="death")
    assert summary["hazard_ratio"] == pytest.approx(fit.hazard_ratios_["treatment__is_1"])
    primary = pd.read_csv(tmp_path / products["table:survival_effect_estimates"]).iloc[0]
    assert primary.estimand == "cause_specific_hazard_ratio"
    assert "event-free" in primary.estimand_question
    fg = pd.read_csv(tmp_path / products["table:survival_fine_gray"])
    assert set(fg.estimand) == {"subdistribution_hazard_ratio"}
    assert "cumulative incidence" in fg.estimand_question.iloc[0]
    exposure_fg = fg.loc[fg.term == "treatment__is_1"].iloc[0]
    assert primary.subdistribution_hazard_ratio == pytest.approx(exposure_fg.hazard_ratio)
    assert primary.subdistribution_ci_low == pytest.approx(exposure_fg.ci_low)
    assert "cumulative incidence" in primary.subdistribution_question
    assert "table:survival_logrank" not in products


@pytest.mark.parametrize("only_after_horizon", [False, True])
def test_declared_competing_event_absent_from_analysis_fails_closed(tmp_path, monkeypatch, only_after_horizon):
    frame = cohort().replace({"death": {2: 0}})
    if only_after_horizon:
        frame.loc[0, ["death", "follow_up_days"]] = [2, 40]
    with pytest.raises(owner.SurvivalPrimaryExecutionError) as exc:
        run(tmp_path, monkeypatch, frame, competing=True)
    assert exc.value.reason_code == "survival_competing_event_absent"
    assert not (tmp_path / "survival_effect_estimates.csv").exists()


def test_competing_declaration_requires_distinct_code_and_definition():
    payload = _requirement().model_dump()
    payload["competing_risk_strategy"] = "cause_specific_and_subdistribution"
    for update in (
        {}, {"competing_event_value": 1, "competing_event_definition": "competing outcome"},
        {"competing_event_value": 2},
        {"competing_event_value": 2, "competing_event_definition": " "},
    ):
        with pytest.raises(ValueError):
            type(_requirement()).model_validate({**payload, **update})
    with pytest.raises(ValueError, match="requires cause_specific_and_subdistribution"):
        type(_requirement()).model_validate({**_requirement().model_dump(),
                                            "competing_event_value": 2,
                                            "competing_event_definition": "competing outcome"})


def test_declared_competitor_lost_to_complete_cases_is_refused(tmp_path, monkeypatch):
    frame = cohort().replace({"death": {2: 0}})
    frame.loc[0, ["death", "age"]] = [2, np.nan]
    with pytest.raises(owner.SurvivalPrimaryExecutionError) as exc:
        run(tmp_path, monkeypatch, frame, competing=True)
    assert exc.value.reason_code == "survival_competing_event_absent"


def test_competing_plan_is_validated_selected_and_receipted(tmp_path, monkeypatch):
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.research_agent.gates.family_primary_result import family_primary_result_reconciliation_findings
    from easyicu.research_agent.planning.capability_registry import assess_scientific_capability
    from easyicu.research_agent.planning.primary_result_contract import validate_required_primary_result
    from easyicu.research_agent.schema import AnalysisPlan

    step = competing_step()
    context = _context()
    context = context.model_copy(update={
        "endpoint": context.endpoint.model_copy(update={"levels": [0, 1, 2]}),
    })
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="survival", steps=[step])
    validate_required_primary_result(plan=plan, context=context)
    selected = select_standard_executor(step, plan=plan)
    assert selected is not None
    assert "competing_event_value" in selected.code
    assessment = assess_scientific_capability(analysis_type="survival", context=context, plan=plan)
    assert assessment.issue_code is None
    summary = run(tmp_path, monkeypatch, cohort(), competing=True)
    kwargs = dict(step=step, plan=plan, context=context, step_summary=summary, out_dir=tmp_path)
    expected_issues = [] if summary["paper_authorization_allowed"] else ["survival_ph_policy_blocks_paper_authorization"]
    assert [f.detail["issue"] for f in family_primary_result_reconciliation_findings(**kwargs)] == expected_issues
    for product in ("table:survival_fine_gray", "table:survival_gray_test", "table:survival_cumulative_incidence"):
        path = tmp_path / summary["output_files"][product]
        saved = path.read_bytes()
        path.write_bytes(saved + b"\n")
        findings = family_primary_result_reconciliation_findings(**kwargs)
        assert [f.detail["issue"] for f in findings] == ["survival_host_receipt_binding_mismatch"]
        path.write_bytes(saved)


def test_ordinal_survival_emits_a_k_group_logrank(tmp_path, monkeypatch):
    frame = cohort().replace({"death": {2: 0}})
    frame["treatment"] = np.arange(len(frame)) % 3
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    requirement = _requirement().model_dump()
    requirement["model_terms"][0] = dict(name="treatment", role="exposure", coding="ordinal_linear",
                                         levels=["0", "1", "2"], transform="declared_level_index")
    requirement = type(_requirement()).model_validate(requirement)
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path))
    kwargs = _runtime_kwargs(requirement, input_path=path,
                             input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary = owner.run_survival_primary(**kwargs, emit_step_summary=False)
    result = pd.read_csv(tmp_path / summary["output_files"]["table:survival_logrank"])
    expected = multivariate_logrank_test(frame.follow_up_days.clip(upper=28), frame.treatment,
                                         frame.death * (frame.follow_up_days <= 28))
    assert result.df.tolist() == [2, 2, 2]
    assert result.statistic.to_numpy() == pytest.approx(expected.test_statistic)
