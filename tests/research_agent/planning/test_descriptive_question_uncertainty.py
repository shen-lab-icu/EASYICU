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
from easyicu.research_agent.planning.question_interpretation import (
    amendment_user_messages,
    requested_uncertainty,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from tests.support.descriptive_plan import _context, _plan
from tests.support.descriptive_review import _contract, _review
from easyicu.webserver.agent_pipeline_runs import _load_pending_scientific_review


def _register_review_context(directory, evidence, context):
    path = directory / "research_context.json"
    path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(kind="log", description="Synthetic reviewed context", source_path=path,
                           evidence_id="research_context", producer="pipeline", generation_mode="system")






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


@pytest.mark.parametrize("question", [
    "按性别报告死亡率和95%的置信区间",
    "Report grouped mortality with 95 percent confidence intervals",
    "无论何种方法都报告95%的置信区间",
    "不要画图，报告95%的置信区间",
    "无需画图，提供95%的置信区间",
])
@pytest.mark.parametrize("level,blocked", [(0.90, True), (0.95, False)])
@pytest.mark.parametrize("amendment", [False, True])
def test_explicit_full_interval_level_survives_question_and_revision(question, level, blocked, amendment):
    plan = _plan()
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "confidence_level": level,
        })}) if step.exposure_outcome_distribution_spec is not None else step
        for step in plan.steps]
    review = _review("按性别描述死亡率" if amendment else question,
        amendment=question if amendment else "", plan=plan.model_copy(update={"steps": steps}))
    assert review.facts["requested_descriptive_uncertainty"]["requested_confidence_levels"] == [0.95]
    assert any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings) is blocked
    if blocked:
        assert not review.approval_allowed


@pytest.mark.parametrize("question", [
    "不需要95%的置信区间", "不需要提供95%的置信区间",
    "Do not report 95 percent confidence intervals", "I no longer need 95 percent confidence intervals",
])
def test_new_explicit_level_syntax_does_not_change_negated_interpretation(question):
    requirement = requested_uncertainty(question)
    # These existing unsupported negation phrasings remain unchanged and must
    # be addressed separately; do not claim that they were interpreted safely.
    assert requirement.requested and requirement.confidence_levels == ()


def test_negating_another_interval_does_not_hide_the_positive_level():
    value = requested_uncertainty("不需要95%的置信区间，但请报告90%的置信区间")
    assert value.confidence_levels == (0.90,)


@pytest.mark.parametrize("question", [
    "报告95%置信区间，不要改为90%的置信区间",
    "报告95%置信区间，不能改为90%的置信区间",
    "报告95%置信区间，别改为90%的置信区间",
    "报告95%置信区间，不选择90%的置信区间",
    "Report 95% confidence intervals; do not change to 90 percent confidence intervals",
    "Report 95% confidence intervals; never change to 90 percent confidence intervals",
])
def test_new_positive_syntax_does_not_add_a_negated_replacement(question):
    assert requested_uncertainty(question).confidence_levels == (0.95,)


@pytest.mark.parametrize("question", [
    "95%的置信区间不要",
    "95%的置信区间不要，报告90%置信区间",
    "Report 90 and 95 percent confidence intervals",
    "报告90%和95%的置信区间",
    "90%的置信区间不变，改成95%的置信区间",
    "Please report CI at 95%",
    "Report CI at 95 percent",
    "请报告95%的 CI",
    "Please report 95 percent CI",
    "Report cardiac index; report CI at 95% for mortality",
    "Cardiac index and mortality CI at 95%",
    "Cardiac index (CI) and mortality CI at 95%",
    "置信区间水平 95%",
    "95% CI and 90% too",
    "The 95 percent confidence intervals should never be presented",
    "95%的置信区间没有必要",
    "95%的置信区间暂时不用",
    "The 95 percent confidence intervals are unnecessary",
    "95%的置信区间禁止报告",
    "95%的置信区间请勿报告",
    "95%的置信区间暂不报告",
    "报告95%置信区间，不要从95%改为90%置信区间",
])
def test_ambiguous_new_level_expression_blocks_approval(question):
    review = _review(question)
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" and f.severity == "blocker"
               for f in review.findings)


def test_explicit_next_message_can_clarify_an_unresolved_interval():
    review = _review("95%的置信区间不要", amendment="改为只报告90%的置信区间")
    assert not any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)
    assert review.facts["requested_descriptive_uncertainty"]["requested_confidence_levels"] == [0.90]


def test_explicit_full_withdrawal_can_clarify_a_prior_ambiguous_interval():
    review = _review("报告90%和95%的置信区间", amendment="不需要置信区间")
    assert not any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)
    assert "requested_descriptive_uncertainty" not in review.facts


def test_repeating_one_of_several_levels_does_not_clear_ambiguity():
    review = _review("Report 90 and 95 percent confidence intervals",
                     amendment="Report 95 percent confidence intervals")
    assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)
    assert not review.approval_allowed


@pytest.mark.parametrize("amendment", ["报告95%置信区间，另外90%", "Report 95% confidence interval; 90% too"])
def test_partially_bound_multiple_percentages_remain_unresolved(amendment):
    review = _review("报告95%置信区间", amendment=amendment)
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)


def test_withdrawing_interval_does_not_hide_an_unbound_new_percentage():
    review = _review("报告95%置信区间", amendment="不需要置信区间，但请给出90%")
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)


def test_unbound_clinical_percentage_is_not_guessed_as_interval_level():
    review = _review("25% of patients died; report a Wilson confidence interval")
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)


@pytest.mark.parametrize("question", [
    "报告0%置信区间", "置信区间为0%", "报告100%置信区间",
    "报告-95%置信区间", "报告−95%置信区间", "Report -95% CI",
    "Report 95-percent confidence intervals", "Report 95 per cent confidence intervals",
])
def test_invalid_or_unparsed_interval_level_cannot_satisfy_scientific_review(question):
    plan = _plan()
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "confidence_level": 0.95,
        })}) if step.exposure_outcome_distribution_spec is not None else step
        for step in plan.steps]
    review = _review(question, plan=plan.model_copy(update={"steps": steps}))
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" and f.severity == "blocker"
               for f in review.findings)
    if "-95" in question or "−95" in question:
        assert requested_uncertainty(question).confidence_levels != (0.95,)


def test_level_only_revision_is_not_silently_ignored():
    for amendment in ("改为90%", "不，要90%", "不要报告置信区间", "请用90%", "改成90 percent", "Make it 90%"):
        review = _review("请报告95%置信区间", amendment=amendment)
        assert any(f.code == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in review.findings)
        assert not review.approval_allowed


def test_cardio_index_abbreviation_is_not_forced_to_be_a_confidence_interval():
    value = requested_uncertainty("CI at 95% of baseline represents cardiac index")
    assert not value.unresolved


def test_unresolved_interval_invalidates_an_old_approvable_review(tmp_path):
    question = "报告90%和95%的置信区间"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    current = build_plan_scientific_review(context=context, plan=plan)
    old = current.model_copy(update={
        "approval_allowed": True,
        "findings": [f for f in current.findings if f.code not in {
            "QUESTION_UNCERTAINTY_UNRESOLVED", "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET"}],
    })
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=old)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(tmp_path), requests=requests)
    assert _load_pending_scientific_review(tmp_path, pending) == {}


def test_current_unresolved_review_remains_visible_without_approval(tmp_path):
    question = "报告90%和95%的置信区间"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    review = build_plan_scientific_review(context=context, plan=plan)
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(tmp_path), requests=requests)
    projected = _load_pending_scientific_review(tmp_path, pending)
    assert projected["approval_allowed"] is False
    assert any(f["code"] == "QUESTION_UNCERTAINTY_UNRESOLVED" for f in projected["findings"])


def test_pending_review_uses_the_original_question_even_if_plan_wording_differs(tmp_path):
    context = _context().model_copy(update={"research_question": "报告90%和95%的置信区间"})
    plan = _plan().model_copy(update={"research_question": "报告死亡率"})
    review = build_plan_scientific_review(context=context, plan=plan)
    old = review.model_copy(update={"approval_allowed": True, "findings": []})
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=old)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(tmp_path), requests=requests)
    assert _load_pending_scientific_review(tmp_path, pending) == {}


def test_pending_review_allows_plan_paraphrase_when_original_requirements_are_met(tmp_path):
    context = _context().model_copy(update={"research_question": "按性别描述死亡率"})
    plan = _plan().model_copy(update={"research_question": "Describe mortality by sex"})
    review = build_plan_scientific_review(context=context, plan=plan)
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(tmp_path), requests=requests)
    assert _load_pending_scientific_review(tmp_path, pending)["plan_sha256"] == review.plan_sha256


@pytest.mark.parametrize("drift", ["missing", "changed"])
def test_pending_review_refuses_a_missing_or_modified_bound_context(tmp_path, drift):
    question = "按性别描述死亡率"
    context = _context().model_copy(update={"research_question": question})
    plan = _plan().model_copy(update={"research_question": question})
    review = build_plan_scientific_review(context=context, plan=plan)
    evidence = EvidenceStore(tmp_path)
    _register_review_context(tmp_path, evidence, context)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    requests = human_review_requests_for_plan(findings=(), plan=plan, evidence=evidence, require_plan_review=True)
    pending = HumanReviewPending(run_id="run_one", thread_id="run_one", run_dir=str(tmp_path), requests=requests)
    path = tmp_path / "research_context.json"
    if drift == "missing":
        path.unlink()
    else:
        path.write_text(context.model_copy(update={"research_question": "改变问题"}).model_dump_json(), encoding="utf-8")
    assert _load_pending_scientific_review(tmp_path, pending) == {}


def test_unresolved_interval_finding_has_reader_copy_in_both_languages():
    import shutil
    from pathlib import Path
    from tests.support.node import run_node

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    owner = Path(__file__).parents[3] / "src/easyicu/webserver/static/js/screens-agent-render.js"
    program = r'''
global.window = global;
window.EU_HTML = {esc: String, escAttr: String};
window.AGENT_REPORT_READER = {};
window.icon = () => '';
let lang = 'en'; window.t = (en, zh) => lang === 'en' ? en : zh;
require(process.argv[1]);
const row = {code: 'QUESTION_UNCERTAINTY_UNRESOLVED', remediation_route: 'study_authority_change'};
const result = {en: window.AGENT_RENDER.scientificFindingCopy(row)};
lang = 'zh'; result.zh = window.AGENT_RENDER.scientificFindingCopy(row);
process.stdout.write(JSON.stringify(result));
'''
    output = json.loads(run_node(node, program, str(owner), check=True).stdout)
    assert "clarification" in output["en"]["title"]
    assert "确认" in output["zh"]["title"]
    assert "置信水平" in output["zh"]["detail"]


def test_unresolved_interval_user_question_renders_in_the_actual_review_artifact():
    import shutil
    from pathlib import Path
    from tests.support.node import run_node

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    owner = Path(__file__).parents[3] / "src/easyicu/webserver/static/js/screens-agent-render.js"
    program = r'''
global.window = global;
window.EU_HTML = {esc: String, escAttr: String}; window.AGENT_REPORT_READER = {};
window.icon = () => ''; window.t = (en, zh) => zh;
require(process.argv[1]);
const payload = JSON.parse(process.argv[2]);
process.stdout.write(window.AGENT_RENDER.artifactStructuredView('scientific_plan_review.json', payload));
'''
    review = _review("Report CI at 95 percent")
    html = run_node(node, program, str(owner), review.model_dump_json(), check=True).stdout
    assert "心脏指数" in html and "置信区间" in html


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
    _register_review_context(tmp_path, evidence, context)
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
    _register_review_context(run, evidence, context)
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


@pytest.mark.parametrize("question", [
    "请给出 95% Wilson 置信区间",
    "请给出 95% bootstrap 置信区间",
    "报告 95% Clopper-Pearson 置信区间",
    "Report a 95% Wilson confidence interval",
    "Report a 95% Wilson CI",
    "Report a 95% Clopper-Pearson CI",
    "Report a 95% bootstrap confidence interval",
    "Report a 95% Clopper-Pearson confidence interval",
    "Report a 95% Katz confidence interval",
    "Report a 95% Example method confidence interval",
])
def test_explicit_level_survives_an_intervening_interval_method(question):
    assert requested_uncertainty(question).confidence_levels == (0.95,)
    plan = _plan()
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "confidence_level": 0.9,
        })}) if step.exposure_outcome_distribution_spec is not None else step for step in plan.steps]
    review = _review(question, plan=plan.model_copy(update={"steps": steps}))
    assert not review.approval_allowed
    assert any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)


@pytest.mark.parametrize("question", [
    "25% mortality and a Wilson confidence interval",
    "25% Male confidence interval",
    "25% of patients died; report a bootstrap confidence interval",
    "死亡率 25%，并报告 Wilson 置信区间",
])
def test_outcome_percentage_is_not_inferred_as_confidence_level(question):
    requirement = requested_uncertainty(question)
    assert requirement.requested
    assert requirement.confidence_levels == ()


@pytest.mark.parametrize("variant", ["bound_blocker", "missing_finding", "wrong_facts", "missing_method_facts", "approval_true", "old_schema", "wrong_plan", "wrong_digest"])
def test_current_bound_review_is_visible_only_when_ci_gap_is_already_blocked(tmp_path, monkeypatch, variant):
    import hashlib
    import easyicu.webserver.agent_pipeline_runs as owner
    from easyicu.research_agent.canonical_json import canonical_sha256

    question = "按性别报告死亡率和 95% Wilson 置信区间"
    plan = _plan().model_copy(update={"research_question": question})
    context = _context().model_copy(update={"research_question": question})
    current = build_plan_scientific_review(context=context, plan=plan)
    payload = current.model_dump(mode="json")
    assert payload["facts"]["requested_descriptive_uncertainty"]["unmet_distribution_step_ids"]
    if variant == "missing_finding":
        payload["findings"] = [f for f in payload["findings"] if f["code"] != "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET"]
    elif variant == "wrong_facts":
        payload["facts"]["requested_descriptive_uncertainty"]["unmet_distribution_step_ids"] = []
    elif variant == "missing_method_facts":
        payload["facts"]["requested_descriptive_uncertainty"].pop("requested_interval_methods")
    elif variant == "approval_true":
        payload["approval_allowed"] = True
    elif variant == "old_schema":
        payload["schema_version"] = "easyicu.plan_scientific_review/20"
    elif variant == "wrong_plan":
        payload["plan_sha256"] = "a" * 64
    raw = json.dumps(payload, ensure_ascii=False).encode()
    (tmp_path / "scientific_plan_review.json").write_bytes(raw)
    context_raw = context.model_dump_json(indent=2).encode()
    (tmp_path / "research_context.json").write_bytes(context_raw)
    monkeypatch.setattr(owner, "_pending_bound_evidence_sha256", lambda _pending, kind:
        hashlib.sha256(context_raw).hexdigest() if kind == "research_context" else
        "0" * 64 if variant == "wrong_digest" else hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(owner, "_pending_plan_authority", lambda *_: plan.model_dump(mode="json"))
    request = SimpleNamespace(payload={"reason": "operator_plan_approval_required"})
    pending = SimpleNamespace(requests=(request,))
    result = owner._load_pending_scientific_review(tmp_path, pending)
    if variant == "bound_blocker":
        assert result["plan_sha256"] == canonical_sha256(plan.model_dump(mode="json"))
        assert result["approval_allowed"] is False
        assert owner._pending_review_reason_code(request=request, plan_recommendation_complete=True,
            scientific_plan_review=result) == "plan_scientific_changes_required"
    else:
        assert result == {}
    assert owner._pending_plan_approval_allowed(run_dir=tmp_path, pending=pending,
        plan_recommendation_complete=True) is False
    assert (tmp_path / "scientific_plan_review.json").read_bytes() == raw


@pytest.mark.parametrize("question", [
    "无需 95% Wilson 置信区间",
    "Do not report 95% bootstrap confidence intervals",
    "Do not report 95% Wilson CI",
])
def test_negated_interval_with_method_is_not_requested(question):
    assert not requested_uncertainty(question).requested


def test_multiple_levels_with_intervening_method_are_all_preserved():
    assert requested_uncertainty("请给出 90% 和 95% Wilson 置信区间").confidence_levels == (0.9, 0.95)


@pytest.mark.parametrize("question,methods", [
    ("请报告 95% Wilson 置信区间", ("wilson",)),
    ("Report 95% Wilson CI", ("wilson",)),
    ("Report Wilson CI", ("wilson",)),
    ("Report 95% CI using Wilson method", ("wilson",)),
    ("Report confidence intervals using Wilson method", ("wilson",)),
    ("Report 95% patient-cluster-robust Wald confidence intervals", ("patient_cluster_robust_wald",)),
    ("报告 95% patient-weighted Hoeffding 置信区间", ("patient_weighted_hoeffding",)),
    ("Report 95% Example method confidence interval", ("example",)),
    ("Report 95% Foobar confidence interval", ("foobar",)),
    ("Report 95% Foobar CI", ("foobar",)),
    ("Report 95% confidence intervals", ()),
])
def test_explicit_interval_method_is_preserved(question, methods):
    requirement = requested_uncertainty(question)
    assert requirement.requested
    assert requirement.interval_methods == methods


def test_interval_method_revision_replacement_and_withdrawal():
    original = "Report 95% Wilson confidence interval"
    changed = requested_uncertainty(original, "Change to 95% Wald confidence interval")
    assert changed.requested and changed.confidence_levels == (0.95,)
    assert changed.interval_methods == ("wald",)
    withdrawn = requested_uncertainty(original, "Do not report 95% Wilson confidence intervals")
    assert not withdrawn.requested and not withdrawn.interval_methods
    added = requested_uncertainty(original, "Also report 95% bootstrap confidence intervals")
    assert added.interval_methods == ("bootstrap", "wilson")


@pytest.mark.parametrize("method,dependence,repeated,should_match", [
    ("wilson", None, "patient_cluster_robust_wald", True),
    ("wilson", {"cluster_unit": "patient"}, "patient_cluster_robust_wald", False),
    ("wald", {"cluster_unit": "patient"}, "patient_cluster_robust_wald", True),
    ("wald", None, "patient_cluster_robust_wald", False),
    ("patient_cluster_robust_wald", {"cluster_unit": "patient"}, "patient_cluster_robust_wald", True),
    ("patient_weighted_hoeffding", {"cluster_unit": "patient"}, "patient_weighted_hoeffding", True),
    ("bootstrap", {"cluster_unit": "patient"}, "patient_cluster_robust_wald", False),
    ("example", None, "patient_cluster_robust_wald", False),
    (None, {"cluster_unit": "patient"}, "patient_cluster_robust_wald", True),
])
def test_requested_method_is_compared_to_the_effective_distribution_method(method, dependence, repeated, should_match):
    from easyicu.research_agent.planning.question_interpretation import UncertaintyRequirement, primary_distribution_gaps

    spec = {"interval_method": "wilson", "repeated_unit_interval_method": repeated,
            "dependence": dependence, "confidence_level": 0.95}
    plan = {"analysis_type": "descriptive_epidemiology", "steps": [{"step_id": "primary",
            "planned_analysis_role": "primary", "exposure_outcome_distribution_spec": spec}]}
    requirement = UncertaintyRequirement(requested=True, confidence_levels=(0.95,),
                                         interval_methods=(method,) if method else ())
    _, missing = primary_distribution_gaps(plan, requirement,
        independent_patient_rows_verified=(method == "wilson" and dependence is None))
    assert (not missing) is should_match


def test_scientific_review_blocks_wilson_when_executor_would_use_patient_cluster_wald():
    from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement

    plan = _plan()
    dependence = PlannedDependenceRequirement(group_source="patient_id", group_derivation="identity")
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "repeated_unit_interval_method": "patient_cluster_robust_wald",
            "confidence_level": 0.95, "dependence": dependence,
        })}) if step.exposure_outcome_distribution_spec is not None else step for step in plan.steps]
    review = _review("按性别报告 95% Wilson 置信区间", plan=plan.model_copy(update={"steps": steps}))
    finding = next(f for f in review.findings if f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET")
    assert finding.severity == "blocker" and not review.approval_allowed
    assert "Wald" in finding.message and "Wilson" in finding.message
    assert review.facts["requested_descriptive_uncertainty"]["requested_interval_methods"] == ["wilson"]


def test_scientific_review_accepts_wilson_only_for_the_independent_runtime_branch(monkeypatch):
    from easyicu.research_agent.planning import question_interpretation

    monkeypatch.setattr(question_interpretation, "verified_independent_patient_rows", lambda _: True)
    plan = _plan()
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "repeated_unit_interval_method": "patient_cluster_robust_wald",
            "confidence_level": 0.95, "dependence": None,
        })}) if step.exposure_outcome_distribution_spec is not None else step for step in plan.steps]
    review = _review("按性别报告 95% Wilson 置信区间", plan=plan.model_copy(update={"steps": steps}))
    assert not any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)
    # This finding alone does not confer source or independent-patient authority.


def test_unknown_counts_and_analysis_unit_do_not_prove_patient_independence():
    plan = _plan()
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "repeated_unit_interval_method": "patient_cluster_robust_wald",
            "confidence_level": 0.95, "dependence": None,
        })}) if step.exposure_outcome_distribution_spec is not None else step for step in plan.steps]
    review = _review("按性别报告 95% Wilson 置信区间", plan=plan.model_copy(update={"steps": steps}))
    finding = next(f for f in review.findings if f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET")
    assert finding.severity == "blocker" and "One row per patient" in finding.message
    assert not review.approval_allowed


@pytest.mark.parametrize("values,expected", [(["p1", "p2"], True), (["p1", "p1"], False),
                                             (["p1", None], False)])
def test_independence_proof_checks_the_bound_cohort_patient_rows(tmp_path, monkeypatch, values, expected):
    import pyarrow as pa
    from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
    from easyicu.research_agent.intake import materialized_metadata
    from easyicu.research_agent.planning import dependence_authority
    from easyicu.research_agent.planning.question_interpretation import verified_independent_patient_rows

    grouping = PlannedDependenceRequirement(group_source="patient_id", group_derivation="identity")
    verified = SimpleNamespace(reference=SimpleNamespace(sha256="a" * 64),
                               authority=SimpleNamespace(cohort_sha256="b" * 64))
    context = SimpleNamespace(cohort_parquet=str(tmp_path / "cohort.parquet"), cohort=SimpleNamespace(
        provenance={"data_provenance_materialization_sha256": "a" * 64,
                    "data_provenance_cohort_file_sha256": "b" * 64}))
    monkeypatch.setattr(dependence_authority, "context_patient_group_authority", lambda _: grouping)
    monkeypatch.setattr(materialized_metadata, "load_verified_materialized_cohort_authority", lambda _: verified)
    monkeypatch.setattr(materialized_metadata, "read_verified_materialized_cohort_table",
                        lambda *_, **__: pa.table({"patient_id": values}))
    assert verified_independent_patient_rows(context) is expected
    context.cohort.provenance["data_provenance_materialization_sha256"] = "c" * 64
    assert not verified_independent_patient_rows(context)
