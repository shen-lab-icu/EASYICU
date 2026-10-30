"""A researched uncertainty request must survive compilation into a plan."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.orchestration.scientific_plan_review_gate import (
    ScientificPlanReviewArtifactError,
    persist_or_validate_scientific_plan_review,
    prepare_scientific_plan_review_gate,
)
from easyicu.research_agent.orchestration.workflow import HumanReviewPending, human_review_requests_for_plan
from easyicu.research_agent.planning.descriptive_question_uncertainty import (
    amendment_user_messages,
    requested_uncertainty,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from tests.support.descriptive_plan import _context, _plan
from easyicu.webserver.agent_pipeline_runs import _load_pending_scientific_review


def _contract(message: str, reference: str = "") -> str:
    return "Host-bound request to revise the complete candidate plan. Changes follow.\n" + json.dumps({
        "schema_version": "easyicu.plan-change-request/1",
        "source_run_id": "run_one",
        "user_message": message,
        "reference_plans": [{"plan": {"research_question": reference}}],
    }, ensure_ascii=False)


def _review(question: str, *, amendment: str = "", plan=None):
    context = _context().model_copy(update={"research_question": question})
    plan = plan or _plan()
    plan = plan.model_copy(update={"research_question": question})
    return build_plan_scientific_review(
        context=context, plan=plan,
        bound_plan_revision_contract=_contract(amendment) if amendment else "",
    )


@pytest.mark.parametrize("question", [
    "按性别给出死亡比例及 95% 置信区间",
    "Report grouped mortality with a 95% confidence interval",
    "Provide proportions and 95% CI",
])
def test_explicit_uncertainty_cannot_be_replaced_by_counts_only(question):
    review = _review(question)
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)
    assert review.facts["requested_descriptive_uncertainty"]["unmet_distribution_step_ids"] == [
        "exposure_outcome_distribution"
    ]


@pytest.mark.parametrize("question", [
    "按性别描述死亡人数和比例",
    "报告死亡比例，不计算置信区间",
    "Report mortality without confidence intervals",
    "Report mortality, no 95% CI",
    "Confidence intervals are not needed",
    "置信区间不需要",
    "不再报告 95% 置信区间",
    "Report the cardiac index (CI) by sex",
    "请提供不确定性",
    "不确定性用标准误",
    "无需置信区间但提供标准误",
])
def test_silent_or_negated_interval_is_not_a_positive_request(question):
    review = _review(question)
    assert not any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)
    assert "requested_descriptive_uncertainty" not in review.facts


def test_revision_request_is_separately_bound_to_the_review():
    review = _review("按性别描述死亡比例", amendment="请报告每组 95% 置信区间。")
    fact = review.facts["requested_descriptive_uncertainty"]
    assert fact["requested_confidence_levels"] == [0.95]
    assert fact["revision_contract_sha256"]
    assert not review.approval_allowed


def test_later_amendment_can_change_or_withdraw_the_interval_request():
    changed = _review("请报告 95% 置信区间", amendment="置信区间改为 90%")
    assert changed.facts["requested_descriptive_uncertainty"]["requested_confidence_levels"] == [0.90]
    withdrawn = _review("请报告 95% 置信区间", amendment="无需置信区间但提供标准误")
    assert "requested_descriptive_uncertainty" not in withdrawn.facts


def test_later_explicit_positive_mention_overrides_an_earlier_negation():
    value = requested_uncertainty("不要 90% 置信区间；请报告 95% 置信区间")
    assert value.requested and value.confidence_levels == (0.95,)


def test_later_explicit_negative_mention_overrides_a_positive_mention():
    value = requested_uncertainty("请给出 95% 置信区间；不需要置信区间")
    assert not value.requested


def test_two_levels_in_one_question_are_not_silently_replaced():
    question = "请给出 90% 置信区间和 95% 置信区间"
    assert requested_uncertainty(question).confidence_levels == (0.9, 0.95)
    plan = _plan()
    steps = [
        step.model_copy(update={"exposure_outcome_distribution_spec":
            step.exposure_outcome_distribution_spec.model_copy(update={
                "interval_method": "wilson", "confidence_level": 0.95,
            })}) if step.exposure_outcome_distribution_spec is not None else step
        for step in plan.steps
    ]
    review = _review(question, plan=plan.model_copy(update={"steps": steps}))
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)
    assert review.facts["requested_descriptive_uncertainty"]["requested_confidence_levels"] == [0.9, 0.95]


def test_later_message_can_replace_multiple_levels():
    value = requested_uncertainty(
        "报告 90% 置信区间和 95% 置信区间", "改为 99% 置信区间",
    )
    assert value.confidence_levels == (0.99,)


@pytest.mark.parametrize("amendment", [
    "另外请报告 90% 置信区间",
    "Also report 90% confidence intervals",
])
def test_explicit_addition_keeps_the_previous_level(amendment):
    value = requested_uncertainty("请报告 95% 置信区间", amendment)
    assert value.confidence_levels == (0.9, 0.95)


def test_withdrawing_one_level_does_not_remove_another():
    value = requested_uncertainty(
        "报告 90% 置信区间和 95% 置信区间", "不再报告 90% 置信区间",
    )
    assert value.requested
    assert value.confidence_levels == (0.95,)


def test_replacement_after_a_withdrawal_does_not_restore_old_levels():
    value = requested_uncertainty(
        "报告 90% 置信区间和 95% 置信区间",
        "不再报告 90% 置信区间，改为 99% 置信区间",
    )
    assert value.confidence_levels == (0.99,)


@pytest.mark.parametrize("question", [
    "请报告 90% 和 95% 置信区间",
    "Report 90% and 95% confidence intervals",
    "Report 90% and 95% CI",
])
def test_joined_levels_are_both_retained(question):
    assert requested_uncertainty(question).confidence_levels == (0.9, 0.95)


@pytest.mark.parametrize("question", [
    "不需要 90% 和 95% 置信区间",
    "Do not report 90% and 95% confidence intervals",
    "Do not report 90% and 95% CI",
])
def test_negated_joined_levels_are_not_requested(question):
    assert not requested_uncertainty(question).requested


def test_only_the_actual_amendment_not_a_historical_plan_is_a_user_request():
    contract = _contract("请只修正文献引用", reference="请给出 95% 置信区间")
    assert amendment_user_messages(contract) == ("请只修正文献引用",)
    assert not requested_uncertainty(*amendment_user_messages(contract)).requested


@pytest.mark.parametrize("contract", [
    'Host-bound request to revise the complete candidate plan.\n{"schema_version":"easyicu.plan-change-request/1",',
    'Host-bound request to revise the complete candidate plan.\n{"schema_version":"unknown","user_message":"95% CI"}',
    'Host-bound request to revise the complete candidate plan.\n{"schema_version":"easyicu.plan-change-request/1"}',
])
def test_a_malformed_host_revision_is_not_silently_ignored(contract):
    with pytest.raises(ValueError, match="host_bound_plan_revision_contract_unreadable"):
        amendment_user_messages(contract)


def test_interval_with_requested_level_satisfies_this_requirement():
    plan = _plan()
    steps = []
    for step in plan.steps:
        spec = step.exposure_outcome_distribution_spec
        if spec is not None:
            spec = spec.model_copy(update={"interval_method": "wilson", "confidence_level": 0.95})
            step = step.model_copy(update={"exposure_outcome_distribution_spec": spec})
        steps.append(step)
    review = _review("按性别报告 95% 置信区间", plan=plan.model_copy(update={"steps": steps}))
    assert not any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)
    # Other source, interval-authority, and scientific gates still decide whether it can run.


def test_interval_with_a_different_level_does_not_satisfy_request():
    plan = _plan()
    steps = []
    for step in plan.steps:
        spec = step.exposure_outcome_distribution_spec
        if spec is not None:
            spec = spec.model_copy(update={"interval_method": "wilson", "confidence_level": 0.90})
            step = step.model_copy(update={"exposure_outcome_distribution_spec": spec})
        steps.append(step)
    review = _review("按性别报告 95% 置信区间", plan=plan.model_copy(update={"steps": steps}))
    assert any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)


def test_an_old_review_cannot_be_reused_to_skip_a_new_interval_requirement(tmp_path):
    question = "按性别描述死亡比例"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    old = build_plan_scientific_review(context=context, plan=plan)
    new = build_plan_scientific_review(
        context=context, plan=plan, bound_plan_revision_contract=_contract("请给出 95% 置信区间"),
    )
    assert old.context_sha256 == new.context_sha256
    assert old.plan_sha256 == new.plan_sha256
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=old)
    with pytest.raises(ScientificPlanReviewArtifactError, match="binding_drift"):
        persist_or_validate_scientific_plan_review(
            run_dir=tmp_path, evidence=evidence, current_review=new, reuse_existing_review=True,
        )


def test_review_gate_persists_and_reports_the_user_amendment(tmp_path):
    question = "按性别描述死亡比例"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    result = prepare_scientific_plan_review_gate(
        context=context, plan=plan, literature=None, figure_strategy=None,
        run_dir=tmp_path, evidence=EvidenceStore(tmp_path), model_retention=None,
        bound_plan_revision_contract=_contract("请给出各组 95% 置信区间"),
    )
    assert result.review.approval_allowed is False
    assert "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" in result.finding.detail["finding_codes"]
    assert json.loads(result.artifact_path.read_text())["facts"]["requested_descriptive_uncertainty"]


@pytest.mark.parametrize("question,review_allowed", [
    ("按性别报告死亡率和 95% 置信区间", False),
    ("按性别报告死亡率，无需置信区间", True),
    ("按性别报告死亡率", True),
])
def test_historical_pending_reviews_keep_their_data_but_cannot_waive_requested_ci(
    tmp_path, question, review_allowed,
):
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    current = build_plan_scientific_review(context=context, plan=plan)
    old = current.model_copy(update={
        "approval_allowed": True,
        "facts": {k: v for k, v in current.facts.items() if k != "requested_descriptive_uncertainty"},
        "findings": [f for f in current.findings if f.code != "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET"],
    })
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=old)
    before = (tmp_path / "scientific_plan_review.json").read_bytes()
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(tmp_path), requests=requests)
    assert bool(_load_pending_scientific_review(tmp_path, pending)) is review_allowed
    assert (tmp_path / "scientific_plan_review.json").read_bytes() == before


def test_pending_review_refuses_an_unreadable_bound_revision(tmp_path, monkeypatch):
    import easyicu.webserver.agent_pipeline_runs as owner
    from easyicu.research_agent.orchestration.config import PipelineConfig
    import easyicu.research_agent.orchestration.human_review_checkpoint as checkpoints

    run = tmp_path / "pipeline" / "run_one"
    run.mkdir(parents=True)
    question = "按性别报告死亡率"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    review = build_plan_scientific_review(context=context, plan=plan)
    evidence = EvidenceStore(run)
    persist_or_validate_scientific_plan_review(run_dir=run, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(run), requests=requests)
    monkeypatch.setattr(owner, "load_recovery_seed", lambda _: SimpleNamespace(
        wrapper_dir=tmp_path, pipeline_config={}, pipeline_config_sha256="a" * 64,
    ))
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda *args, **kwargs: SimpleNamespace(
        run_id="run_one", pipeline_config_sha256="a" * 64,
    ))
    monkeypatch.setattr(PipelineConfig, "from_recovery_payload", lambda *args, **kwargs: SimpleNamespace(
        workdir=run.parent, bound_plan_revision_contract="Host-bound request to revise the complete candidate plan.\n{corrupt",
    ))
    before = (run / "scientific_plan_review.json").read_bytes()
    assert _load_pending_scientific_review(run, pending) == {}
    assert (run / "scientific_plan_review.json").read_bytes() == before


def test_missing_seed_uses_only_a_bound_recovery_record_or_refuses(tmp_path, monkeypatch):
    import easyicu.webserver.agent_pipeline_runs as owner
    from easyicu.research_agent.orchestration.config import PipelineConfig
    import easyicu.research_agent.orchestration.human_review_checkpoint as checkpoints

    run = tmp_path / "pipeline" / "run_one"
    run.mkdir(parents=True)
    question = "按性别报告死亡比例及其不确定性"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    review = build_plan_scientific_review(context=context, plan=plan)
    evidence = EvidenceStore(run)
    persist_or_validate_scientific_plan_review(run_dir=run, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(run), requests=requests)
    monkeypatch.setattr(owner, "load_recovery_seed", lambda _: None)
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda *args, **kwargs: SimpleNamespace(
        run_id="run_one", pipeline_config_sha256="a" * 64,
    ))
    monkeypatch.setattr(PipelineConfig, "from_recovery_payload", lambda payload, **kwargs: SimpleNamespace(
        workdir=run.parent, bound_plan_revision_contract=payload["contract"],
    ))
    record = SimpleNamespace(
        run_id="run_one", wrapper_dir=tmp_path, pipeline_config_sha256="a" * 64,
        pipeline_config={"contract": _contract("请提供 95% 置信区间")},
    )
    monkeypatch.setattr(owner, "get_review_recovery_record", lambda _: record)
    before = (run / "scientific_plan_review.json").read_bytes()
    assert _load_pending_scientific_review(run, pending) == {}
    # A proven non-revision run remains readable, without rewriting its review.
    record.pipeline_config = {"contract": ""}
    assert _load_pending_scientific_review(run, pending)
    # An unbound digest, or no configuration authority, refuses rather than
    # silently interpreting the original question as the whole request.
    record.pipeline_config_sha256 = "b" * 64
    assert _load_pending_scientific_review(run, pending) == {}
    monkeypatch.setattr(owner, "get_review_recovery_record", lambda _: None)
    assert _load_pending_scientific_review(run, pending) == {}
    assert (run / "scientific_plan_review.json").read_bytes() == before
