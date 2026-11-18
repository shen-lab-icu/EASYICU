"""Scripted intake through the requirement owner, with independent typed facts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import pytest

from easyicu.clinical_contracts import load_clinical_contracts
from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request,
    parse_family_plan_spec,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.model_terms import DataPreparationPlan, ModelTermSpec, WindowAggregation
from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
from easyicu.research_agent.planning.cohort_contract import CohortDefinition, ConceptPredicate
from easyicu.research_agent.planning.cohort_contract import TimeWindow as PredicateWindow
from easyicu.research_agent.planning.dependence_authority import context_dependence_authority
from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    judge_question_requirements,
    question_requirement_host_facts,
)
from easyicu.research_agent.planning.requirement_typing.contract import (
    AtomicPart, CohortPredicate, Delivery, Grading, Method, Parameter, Point, RequirementKey,
    SourceSpan, TimeZero, Unit, Window,
)
from easyicu.research_agent.planning.requirement_typing.facts import ProjectedFacts, project_facts
from easyicu.research_agent.planning.requirement_typing.intake import PROPOSALS_FIELD, TypingIntake
from easyicu.research_agent.planning.requirement_typing.registry import contract_registry_sha256, registered_dictionary
from easyicu.research_agent.planning.requirement_typing.shadow import FLAG, LOGGER
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.research_context.clinical_definition import ClinicalDefinitionReference
from easyicu.research_agent.schema import (
    AnalysisPlan, AnalysisStep, ConceptDescriptor, ObservationSemantics, ResearchContext, UserPreferences,
)

from .family_spec_fixtures import _context as _family_context
from .family_spec_fixtures import _request, _run, _spec_payload
from .test_typed_question_requirement_coverage import _context, _model, _model_plan
from .test_typed_question_requirement_regressions import (
    AUTHORITY, CASES, CLUSTER, DIALYSIS16, DICTIONARY, FIRST_ONLY, ICU, REPORT_STAYS, STUDY_POPULATION,
    bind, fact, interval, key,
)
from .test_typed_question_requirement_shadow import _shadow_event, proposal


def _clock() -> ConceptDescriptor:
    return ConceptDescriptor(name="clock", dtype="float64", observation_semantics=ObservationSemantics(
        kind="conditional_event_time", event_status_column="death", time_origin=ICU, time_unit="hours",
    ))


def _with_exposure(concept: str, *, unit=None, definition=None, preparation=None):
    context = _context(variables=[
        ConceptDescriptor(name=concept, dtype="float64", unit=unit, clinical_definition=definition),
        ConceptDescriptor(name="death", dtype="bool"), _clock(),
    ]).model_copy(update={"primary_exposure": concept})
    plan = _model_plan(preparation)
    plan.steps[0].inputs[0] = concept
    plan.steps[0].model_requirements = [_model(exposure=concept)]
    return context, plan


def _landmark(hours: int, *, applied=True):
    context, plan = _with_exposure("k")
    spec = PrespecifiedSensitivitySpec(
        spec_id="timing", axis="timing", strategy="landmark", landmark_hours=hours,
        event_time_variable="clock", execution_variables=("clock",),
    )
    context = context.model_copy(update={
        "endpoint": EndpointSpec(
            name="death", kind="time_to_event", absence_semantics="no_absent_rows", levels=[False, True],
            event_column="death", time_column="clock", time_origin=ICU, censoring_rule="specified",
        ),
        "user_preferences": UserPreferences(sensitivity_specs=[spec]),
    })
    if applied:
        plan.steps.append(AnalysisStep(
            step_id="sensitivity", intent="prespecified sensitivity", method="landmark_analysis",
            inputs=["clock", "death"], planned_analysis_role="sensitivity", sensitivity_spec_ids=[spec.spec_id],
        ))
    return context, plan


def _window(concept: str, hours: int, op):
    return _with_exposure(concept, preparation=DataPreparationPlan(windows=(WindowAggregation(
        variable=concept, group_column="stay_id", time_column="clock",
        start_hours=0, end_hours=hours, function=op,
    ),)))


def _grading(concept: str, contract_id: str):
    owner = load_clinical_contracts()[contract_id]
    return _with_exposure(concept, definition=ClinicalDefinitionReference(
        contract_id=owner.contract_id, version=owner.version, source_id=owner.source_id,
        status=owner.status, definition=owner.definition, validation_status=owner.validation_status,
        canonical_definition=owner.canonical_definition,
    ))


def _cohort(concept, hours, op, operator, value, polarity, missing):
    plan = _model_plan()
    predicate = ConceptPredicate(
        concept_id=concept, time_window=PredicateWindow("icu_admit", 0, hours),
        aggregation=op, op=operator, value=value, missing_policy=missing,
    )
    plan.cohort = CohortDefinition(
        "selected", inclusion=(predicate,) if polarity == "include" else (),
        exclusion=(predicate,) if polarity == "exclude" else (),
    )
    return _context(), plan


def _stays(first: bool):
    if first:
        return _context(first=True), _model_plan()
    context = _context(preferences=UserPreferences(data_constraints=json.dumps({"analysis_design": {
        "analysis_unit": "icu_stay", "cluster_unit": "patient", "variance_estimator": "cluster_robust",
    }})))
    plan = _model_plan()
    plan.steps[0].model_requirements = [_model(dependence=context_dependence_authority(context))]
    return context, plan


def _model_terms(concept):
    context, plan = _with_exposure(concept)
    plan.steps[0].model_requirements[0].model_terms = [ModelTermSpec(
        name=concept, role="exposure", coding="continuous", transform="identity",
    )]
    return context, plan


@dataclass(frozen=True)
class Pair:
    id: str
    en: str
    zh: str
    key: RequirementKey
    build: Callable[[], tuple[ResearchContext, AnalysisPlan]]
    state: str = "covered"
    origin: str | None = "plan_contract"
    scripted: bool = False
    kind: str = "analysis"
    legacy_concepts: tuple[str, ...] = ()
    bucket: str | None = None


def _pairs():
    for hours in (4, 8, 12, 16):
        applied = hours != 12
        yield Pair(
            f"landmark-{hours}", f"Use the mortality sensitivity landmark at {hours} hours after ICU entry",
            f"入ICU后{hours}小时设死亡敏感性分析界标",
            key(TimeZero(purpose="sensitivity_landmark"), bind("death", "outcome"),
                time=Point(anchor=ICU, unit="h", offset=hours)),
            partial(_landmark, hours, applied=applied), state="covered" if applied else "not_covered",
            origin="study_choice",
        )
    for concept, en, zh, hours, op, operation_en, operation_zh in (
        ("k", "potassium", "血钾", 6, "max", "peak", "最高"),
        ("lact", "lactate", "乳酸", 8, "min", "minimum", "最低"),
        ("mg", "magnesium", "血镁", 12, "mean", "mean", "平均"),
        ("crea", "creatinine", "肌酐", 24, "max", "peak", "最高"),
    ):
        yield Pair(
            f"window-{concept}", f"Use {operation_en} {en} in the first {hours} hours after ICU entry",
            f"入ICU后前{hours}小时取{operation_zh}{zh}",
            key(Window(purpose="aggregation", operation=op), bind(concept), time=interval(0, hours)),
            partial(_window, concept, hours, op), scripted=concept in {"k", "lact"},
        )
    owner = load_clinical_contracts()["kdigo_aki_2012"]
    for concept, en, zh in (
        ("kdigo_creat", "creatinine", "肌酐"), ("kdigo_aki", "kdigo_aki", "kdigo_aki"),
        ("kdigo_uo", "kdigo_uo", "kdigo_uo"),
    ):
        if concept in owner.concepts:
            yield Pair(
                f"grading-{concept}", f"Specify {en} grading {owner.contract_id} version {owner.version}",
                f"明确{zh}按{owner.contract_id}版本{owner.version}分级",
                key(Grading(contract_id=owner.contract_id, version=owner.version), bind(concept)),
                partial(_grading, concept, owner.contract_id),
            )
    yield Pair(
        "grading-paraphrase",
        "Specify the creatinine grading according to KDIGO 2012",
        "明确肌酐的KDIGO 2012分级",
        key(Grading(contract_id=owner.contract_id, version=owner.version), bind("kdigo_creat")),
        partial(_grading, "kdigo_creat", owner.contract_id),
        legacy_concepts=("kdigo_creat",), bucket="v2_under_typing",
    )
    for concept, english, chinese, unit in (
        ("k", "potassium", "血钾", "mmol/L"),
        ("lact", "lactate", "乳酸", "mmol/L"),
        ("mg", "magnesium", "血镁", "mg/dL"),
        ("crea", "creatinine", "肌酐", "mg/dL"),
    ):
        yield Pair(
            f"unit-{concept}", f"Specify {english} units in {unit}", f"明确{chinese}单位为{unit}",
            key(Unit(code=unit), bind(concept)), partial(_with_exposure, concept, unit=unit), scripted=concept == "mg",
        )
    for concept, english, chinese in (
        ("k", "potassium", "血钾"), ("lact", "lactate", "乳酸"), ("mg", "magnesium", "血镁"),
    ):
        model = _model(exposure=concept).method_family
        yield Pair(
            f"method-model-{concept}", f"Model {english} as continuous for mortality with {model}",
            f"用{model}分析死亡时将{chinese}作为连续变量",
            key(Method(action_id="model_term", parameters=(
                Parameter(name="model_method", value=model), Parameter(name="coding", value="continuous"),
                Parameter(name="transform", value="identity"), Parameter(name="levels", value=()),
                Parameter(name="reference_level", value=None),
            )), bind(concept), bind("death", "outcome")), partial(_model_terms, concept), scripted=concept == "k",
        )
    yield Pair(
        "method-odds", "Report the odds ratio for potassium and mortality", "报告血钾与死亡的比值比",
        key(Method(action_id="binary_effect_measure", parameters=(Parameter(name="measure", value="odds_ratio"),)),
            bind("k"), bind("death", "outcome")),
        partial(_with_exposure, "k"), scripted=True,
    )
    for concept, en, zh, hours, operand, value, polarity in (
        ("k", "potassium", "血钾", 24, ">=", 5.5, "include"),
        ("lact", "lactate", "乳酸", 12, ">=", 2.5, "include"),
        ("crea", "creatinine", "肌酐", 24, ">", 2.0, "exclude"),
        ("mg", "magnesium", "血镁", 6, ">=", 3.5, "exclude"),
    ):
        verb = "纳入" if polarity == "include" else "排除"
        yield Pair(
            f"cohort-{concept}",
            f"{polarity.title()} stays with peak {en} {operand} {value:g} in the first {hours} hours after ICU entry, exclude missing",
            f"{verb}入ICU后前{hours}小时最高{zh}{operand}{value:g}的住院，缺失时排除",
            key(CohortPredicate(
                operation="max", operator=operand, value=value, polarity=polarity, missing_policy="exclude",
            ), bind(concept, "population"), time=interval(0, hours)),
            partial(_cohort, concept, hours, "max", operand, value, polarity, "exclude"),
            scripted=concept in {"crea", "mg"},
        )
    for index, (english, chinese) in enumerate((
        ("Report handling of repeated ICU stays", "说明重复入住ICU的处理"),
        ("Specify policy for repeated ICU stays", "明确重复入住ICU的政策"),
        ("Document repeat-stay handling for included ICU admissions", "写明重复入住ICU的处理方式"),
        ("Clarify handling for repeated ICU stays", "报告重复入住ICU的处理"),
    )):
        yield Pair(f"report-stays-{index}", english, chinese, REPORT_STAYS, partial(_stays, index % 2 == 0))
    for index, (english, chinese, select) in enumerate((
        ("Include only the first ICU stay per patient", "仅纳入每位患者的首次ICU住院", FIRST_ONLY),
        ("Keep only first ICU stay per patient", "每位患者仅保留首次入住ICU", FIRST_ONLY),
        ("Retain repeated ICU stays and cluster by patient", "保留重复入住ICU并按患者聚类", CLUSTER),
        ("Specify that repeated ICU stays are retained with clustering by patient", "说明保留重复入住ICU并按患者聚类", CLUSTER),
    )):
        yield Pair(
            f"policy-stays-{index}", english, chinese, select, partial(_stays, select == FIRST_ONLY),
            origin="study_choice",
        )
    for output_id, english, chinese in (
        ("article_preview", "Provide a browser preview", "提供网页预览"),
        ("manuscript_pdf", "Deliver the manuscript PDF", "交付PDF"),
        ("figures", "Include figures", "包含图件"),
        ("references", "Provide a bibliography", "提供参考文献清单"),
    ):
        yield Pair(
            f"delivery-{output_id}", english, chinese, key(Delivery(output_id=output_id), STUDY_POPULATION),
            partial(_with_exposure, "k"), kind="delivery", scripted=output_id == "figures",
        )


LARGER_CASES = [
    pytest.param(pair, lang, quote, id=f"{pair.id}-{lang}")
    for pair in _pairs() for lang, quote in (("en", pair.en), ("zh", pair.zh))
]


def _judge(quote, context, plan, *, kind="analysis", concepts=(), outputs=(), proposals=(), clauses=()):
    context = context.model_copy(update={"research_question": quote})
    plan = plan.model_copy(update={"research_question": quote})
    item = QuestionRequirement(
        id="r1", kind=kind, quote=quote, concepts=list(concepts), clauses=list(clauses),
        deliverable_kinds=list(outputs), coverage="plan",
    )
    return judge_question_requirements(
        [item], plan=plan, family_template=True, relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"), question=quote, shadow_context=context,
        host_facts=question_requirement_host_facts(context=context, plan=plan, requirements=[item]),
        typing_proposals=proposals,
    )


@pytest.mark.parametrize(("pair", "language", "quote"), LARGER_CASES)
def test_larger_bilingual_fixture(pair, language, quote, monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = pair.build()
    scripted = [proposal("r1", quote, pair.key)] if pair.scripted else ()
    outputs = (pair.key.facet.output_id,) if pair.kind == "delivery" else ()
    _judge(quote, context, plan, kind=pair.kind, concepts=pair.legacy_concepts, outputs=outputs, proposals=scripted)
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2"] == pair.state
    assert row["result"]["part"]["origin"] == ("model_proposal_host_validated" if pair.scripted else pair.origin)
    if row["v2"] == "covered":
        assert RequirementKey.model_validate(row["result"]["part"]["key"]) == pair.key
    if pair.bucket == "v2_under_typing":
        dictionary = registered_dictionary(context.cohort.database)
        assert pair.key in {proof.key for proof in project_facts(plan, context=context, dictionary=dictionary).facts}
        assert row["v1"] == "covered"
    print("RC1C_ROW", json.dumps({
        "dataset": 72,
        "id": pair.id, "language": language, "facet": pair.key.facet.kind, "v1": row["v1"],
        "v2": row["v2"], "reason": row["v1_reason"],
        "bucket": pair.bucket if row["v1"] == "covered" and row["v2"] != "covered" else None,
    }))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_wrong_facts_remain_uncovered_through_intake(case) -> None:
    authority = AUTHORITY.model_copy(update={"registry_sha256": contract_registry_sha256()})
    part = AtomicPart(part_id=case.id, source_span=SourceSpan(start=0, end=len(case.quote)),
                      key=None, facet_hint=case.facet)
    scope = case.requested.bindings if case.requested else case.rejected[0].bindings
    roles = {binding.role: binding.concept_id for binding in scope}
    context = _context(preferences=UserPreferences(
        covariates=[binding.concept_id for binding in scope if binding.role == "covariate"],
        covariate_selection="exact", covariate_authority="user",
    )).model_copy(update={
        "primary_exposure": roles.get("exposure", "k"),
        "target_outcome": roles.get("outcome", "death"),
    })

    def decide(evidence, proposals):
        return TypingIntake(
            question=case.quote, dictionary=DICTIONARY,
            projected=ProjectedFacts(authority, tuple(fact(item, authority) for item in evidence)),
            context=context, proposals=proposals, part_ids=frozenset({case.id}),
        ).judge(part)[0].decision.state

    for rejected in (case.rejected, *((item,) for item in case.rejected)):
        for batch in (
            (), [proposal(case.id, case.quote, case.requested, facet=case.facet)],
            *([proposal(case.id, case.quote, item)] for item in rejected),
        ):
            assert decide(rejected, batch) != "covered"

    if case.id.startswith("new/peak-potassium"):
        assert decide((case.requested,), ()) == "covered"


def test_matching_but_misbound_proposal_is_unknown(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _window("lact", 6, "max")
    quote = "Use peak potassium in the first six hours after ICU entry"
    wrong = key(Window(purpose="aggregation", operation="max"), bind("lact"), time=interval(0, 6))
    _judge(quote, context, plan, proposals=[proposal("r1", quote, wrong)])
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2"] == "unknown"
    assert row["v2_reason"] == "unbound_source_coordinates"
    assert row["repair"] == [{"path": f"{PROPOSALS_FIELD}[0].key", "code": "coordinates_need_source_binding"}]


def test_even_a_perfect_fact_match_cannot_rebind_vasopressor_to_dialysis() -> None:
    context = _context().model_copy(update={"primary_exposure": "rrt"})
    quote = "Clarify vasopressor initiation time zero at 16 hours"
    authority = AUTHORITY.model_copy(update={"registry_sha256": contract_registry_sha256()})
    part = AtomicPart(part_id="r1", source_span=SourceSpan(start=0, end=len(quote)), key=None)
    intake = TypingIntake(
        question=quote, dictionary=DICTIONARY, projected=ProjectedFacts(authority, (fact(DIALYSIS16, authority),)),
        context=context, proposals=[proposal("r1", quote, DIALYSIS16)], part_ids=frozenset({"r1"}),
    )
    result, issues = intake.judge(part)
    assert result.decision.state == "unknown"
    assert issues[0].path == f"{PROPOSALS_FIELD}[0].key"


@pytest.mark.parametrize(("change", "path"), [
    ({"source_span": {"start": -1, "end": 3}}, "source_span.start"),
    ({"source_span": {"start": 1, "end": 3}}, "source_span"),
    ({"facet_hint": "clinical"}, "facet_hint"),
    ({"key": {"bindings": [{"concept_id": "k", "role": "source"}], "facet": {"kind": "window"}}}, "key.bindings[0].role"),
    ({"key": {"bindings": [{"concept_id": "study_population", "role": "population"}],
              "time": {"kind": "not_applicable"}, "facet": {"kind": "delivery", "output_id": "private_export"}}},
     "key.facet.delivery.output_id"),
])
def test_malformed_proposal_has_bounded_repair_path(change, path, monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k", unit="mmol/L")
    quote = "Specify potassium units in mmol/L"
    typing = proposal("r1", quote, key(Unit(code="mmol/L"), bind("k")))
    _judge(quote, context, plan, proposals=[{**typing, **change}])
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2"] == "unknown"
    assert row["result"]["part"]["key"] is None
    assert f"{PROPOSALS_FIELD}[0].{path}" in {issue["path"] for issue in row["repair"]}
    assert len(row["repair"]) <= 8


def test_unresolved_alternatives_remain_open_even_with_one_matching_fact(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _stays(True)
    quote = "Choose the first ICU stay per patient or retain repeated ICU stays and cluster by patient"
    _judge(quote, context, plan, proposals=[proposal(
        "r1", quote, None, facet="repeat_stay_policy", alternatives=(FIRST_ONLY, CLUSTER),
    )])
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2"] == "unknown"
    assert row["v2_reason"] == "unresolved_alternatives"
    assert len(row["result"]["part"]["alternatives"]) == 2


def test_current_provenance_and_facts_are_required_at_intake() -> None:
    context, plan = _window("k", 6, "max")
    dictionary = registered_dictionary("miiv")
    projected = project_facts(plan, context=context, dictionary=dictionary)
    asked = key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 6))
    quote = "Use peak potassium in the first six hours after ICU entry"
    part = AtomicPart(part_id="r1", source_span=SourceSpan(start=0, end=len(quote)), key=None)
    for evidence, state in (
        (projected, "covered"),
        (replace(projected, facts=()), "not_covered"),
        (replace(projected, authority=projected.authority.model_copy(update={"plan_sha256": "e" * 64})), "not_covered"),
        (replace(projected, authority=projected.authority.model_copy(update={"registry_sha256": "f" * 64})), "unknown"),
    ):
        assert TypingIntake(
            question=quote, dictionary=dictionary, projected=evidence, context=context,
            proposals=[proposal("r1", quote, asked)], part_ids=frozenset({"r1"}),
        ).judge(part)[0].decision.state == state


def test_proposal_may_express_the_same_window_in_an_equivalent_time_unit(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _window("k", 6, "max")
    quote = "Use peak potassium in the first six hours after ICU entry"
    asked = key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 360, unit="min"))
    _judge(quote, context, plan, proposals=[proposal("r1", quote, asked)])
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2"] == "covered"
    assert row["result"]["part"]["origin"] == "model_proposal_host_validated"


def test_delivery_alternatives_and_uncontracted_exports_do_not_get_credit(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k")
    result_tables = key(Delivery(output_id="result_tables"), STUDY_POPULATION)
    for quote in ("Deliver result tables or figures", "Deliver a patient-level spreadsheet"):
        caplog.clear()
        _judge(quote, context, plan, kind="delivery", outputs=("result_tables",),
               proposals=[proposal("r1", quote, result_tables)])
        [row] = _shadow_event(caplog)["parts"]
        assert row["v1"] == "covered"
        assert row["v2"] == "unknown"


def test_multiple_outputs_keep_distinct_parts_even_with_the_same_span(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k")
    quote = "Deliver result tables and figures"
    wrong = proposal("r1/delivery/figures", quote, key(Delivery(output_id="result_tables"), STUDY_POPULATION))
    _judge(quote, context, plan, kind="delivery", outputs=("result_tables", "figures"), proposals=[wrong])
    table, figures = _shadow_event(caplog)["parts"]
    assert table["source_span"] == figures["source_span"]
    assert table["v2"] == "covered"
    assert figures["v2"] == "unknown"
    assert figures["repair"][0]["path"] == f"{PROPOSALS_FIELD}[0].key.facet.output_id"


def test_a_plan_declared_note_stays_prose_only(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k")
    question = "external validation, calibration"
    items = [
        QuestionRequirement(id="r1", kind="analysis", quote=question, coverage="plan",
                            clauses=["external validation", "calibration"]),
        QuestionRequirement(id="r2", kind="analysis", quote="external validation", coverage="plan",
                            note="An external demonstration will be discussed."),
    ]
    judge_question_requirements(
        items, plan=plan, family_template=True, relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"), question=question, shadow_context=context,
    )
    first = _shadow_event(caplog)["parts"][0]
    assert (first["v1"], first["v1_reason"], first["v2"]) == ("covered", "plan_declared", "unknown")


def test_family_response_typings_stay_out_of_v1_and_flow_to_shadow(monkeypatch, caplog, tmp_path) -> None:
    quote = "Deliver result tables"
    context = _family_context().model_copy(update={"research_question": _family_context().research_question + " " + quote})
    request = _request(context)
    payload = _spec_payload(request, question_requirements=[{
        "id": "r1", "quote": quote, "kind": "delivery", "coverage": "plan",
        "concepts": [], "deliverable_kinds": ["result_tables"], "note": None, "gap": None,
    }])
    monkeypatch.delenv(FLAG, raising=False)
    baseline = parse_family_plan_spec(json.dumps(payload), request)
    assert PROPOSALS_FIELD not in family_spec_response_shape(request)
    assert PROPOSALS_FIELD not in json.loads(family_spec_structured_output_request(request).schema_json)["properties"]
    start = context.research_question.index(quote)
    typing = proposal("r1", quote, key(Delivery(output_id="result_tables"), STUDY_POPULATION), start=start)
    payload[PROPOSALS_FIELD] = [{**typing, "source_span": {"start": -1, "end": 3}}]

    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    assert PROPOSALS_FIELD in family_spec_response_shape(request)
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert PROPOSALS_FIELD in schema["required"]
    assert parse_family_plan_spec(json.dumps(payload), request) == baseline
    llm, planned = _run(context, [json.dumps(payload)])
    assert len(llm.calls) == 1
    assert planned.facts.question_requirement_typings == payload[PROPOSALS_FIELD]
    assert question_requirement_outcome(context=context, plan=planned.output, facts=planned.facts, run_dir=tmp_path) == []
    [row] = _shadow_event(caplog)["parts"]
    assert row["v1"] == "covered"
    assert row["v2"] == "unknown"
    assert row["repair"][0]["path"] == f"{PROPOSALS_FIELD}[0].source_span.start"


@pytest.mark.parametrize("endpoint_case", ["absent", "different_status", "different_name", "different_time"])
def test_mortality_landmark_requires_the_endpoints_own_clock(endpoint_case) -> None:
    context, plan = _landmark(16)
    if endpoint_case == "absent":
        context = context.model_copy(update={"endpoint": None})
    elif endpoint_case == "different_status":
        clock = context.variable("clock")
        clock.observation_semantics = clock.observation_semantics.model_copy(update={"event_status_column": "rrt"})
    else:
        change = {"name": "rrt"} if endpoint_case == "different_name" else {"time_column": "other_clock"}
        context = context.model_copy(update={"endpoint": context.endpoint.model_copy(update=change)})
    dictionary = registered_dictionary(context.cohort.database)
    projected = project_facts(plan, context=context, dictionary=dictionary)
    assert not any(isinstance(item.key.facet, TimeZero) for item in (*projected.facts, *projected.study_choices))


@pytest.mark.parametrize("quote", [
    "Deliver result tables and a patient-level spreadsheet",
    "Do not include figures", "不要包含图件", "请勿交付结果表",
])
def test_delivery_requires_a_complete_affirmative_request(quote, monkeypatch, caplog) -> None:
    from easyicu.research_agent.planning.requirement_typing.source import source_supports

    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k")
    output = "figures" if ("figures" in quote or "图件" in quote) else "result_tables"
    asked = key(Delivery(output_id=output), STUDY_POPULATION)
    dictionary = registered_dictionary(context.cohort.database)
    assert not source_supports(quote, asked, context=context, dictionary=dictionary)
    for proposals in ((), [proposal("r1", quote, asked)]):
        caplog.clear()
        _judge(quote, context, plan, kind="delivery", outputs=(output,), proposals=proposals)
        [row] = _shadow_event(caplog)["parts"]
        assert row["v2"] == "unknown"


def test_unregistered_unit_cannot_be_registered_by_an_applied_variable(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k", unit="mg/24h")
    quote = "Specify potassium units in mg/24h"
    asked = key(Unit(code="mg/24h"), bind("k"))
    for proposals in ((), [proposal("r1", quote, asked)]):
        caplog.clear()
        _judge(quote, context, plan, proposals=proposals)
        [row] = _shadow_event(caplog)["parts"]
        assert row["v2"] == "unknown"
        if proposals:
            assert row["v2_reason"] == "unregistered_coordinate"


@pytest.mark.parametrize("submits_wrong_key", [False, True], ids=["abstention", "unmatched-key"])
def test_proposal_cannot_override_host_bound_window(submits_wrong_key, monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _window("k", 6, "max")
    quote = "Use peak potassium in the first 6 hours after ICU entry"
    wrong = key(Window(purpose="aggregation", operation="max"), bind("lact"), time=interval(0, 6))
    _judge(quote, context, plan, proposals=[proposal("r1", quote, wrong if submits_wrong_key else None)])
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2"] == "covered"
    assert row["result"]["part"]["origin"] == "plan_contract"
    assert row["result"]["part"]["key"]["bindings"][0]["concept_id"] == "k"
    if submits_wrong_key:
        assert row["repair"] == [{"path": f"{PROPOSALS_FIELD}[0].key", "code": "coordinates_need_source_binding"}]


@pytest.mark.parametrize("choices", [
    (key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 6)),
     key(Unit(code="mmol/L"), bind("k"))),
    (key(Window(purpose="aggregation", operation="max"), bind("k"), time=interval(0, 6)),),
    (key(Delivery(output_id="figures"), STUDY_POPULATION),),
], ids=["mixed-facets", "wrong-facet", "wrong-output"])
def test_delivery_checks_every_alternative(choices, monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _with_exposure("k")
    quote = "Deliver result tables"
    _judge(quote, context, plan, kind="delivery", outputs=("result_tables",),
           proposals=[proposal("r1", quote, None, alternatives=choices)])
    [row] = _shadow_event(caplog)["parts"]
    assert row["v2_reason"] == "invalid_typing_proposal"
    assert not row["result"]["part"]["alternatives"]


def test_atomic_alternatives_must_share_one_facet() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="facet"):
        AtomicPart(
            part_id="r1", source_span=SourceSpan(start=0, end=4), key=None, alternatives=(
                key(Unit(code="mmol/L"), bind("k")), key(Delivery(output_id="figures"), STUDY_POPULATION),
            ),
        )


@pytest.mark.parametrize("refusal", ["schema", "gap"])
def test_rejected_family_response_does_not_publish_typings(refusal, monkeypatch) -> None:
    from easyicu.research_agent.agents import family_spec_planner
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
    from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure

    monkeypatch.setenv(FLAG, "shadow")
    context = _family_context()
    request = _request(context)
    payload = _spec_payload(request)
    payload[PROPOSALS_FIELD] = [proposal("r1", "Deliver figures", key(Delivery(output_id="figures"), STUDY_POPULATION))]
    if refusal == "schema":
        payload["unexpected_field"] = True
    else:
        def reject_gap(*args, **kwargs):
            raise ValueError("scripted invalid requirement gap")

        monkeypatch.setattr(family_spec_planner, "check_question_requirement_gaps", reject_gap)
    with pytest.raises(StructuredResponseFailure) as error:
        _run(context, [json.dumps(payload), "[]", "[]"])
    assert progressive_planner_failure_facts(error.value).question_requirement_typings == ()


@pytest.mark.parametrize(("quote", "is_requested"), [
    ("Specify the creatinine grading according to KDIGO 2012", True),
    ("明确肌酐的KDIGO 2012分级", True),
    ("Specify the urine output grading according to KDIGO 2012", False),
    ("明确肌酐的KDIGO 2024分级", False),
], ids=["en", "zh", "wrong-criterion", "wrong-version"])
def test_kdigo_paraphrase_binds_the_exact_registered_grading(quote, is_requested, monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    context, plan = _grading("kdigo_creat", "kdigo_aki_2012")
    expected = key(Grading(contract_id="kdigo_aki_2012", version="2012"), bind("kdigo_creat"))
    dictionary = registered_dictionary(context.cohort.database)
    projected = project_facts(plan, context=context, dictionary=dictionary)
    [receipt] = [item for item in projected.facts if item.key == expected]
    assert receipt.provenance.source_field == "context.variables[0].clinical_definition"
    for proposals in ((), [proposal("r1", quote, expected)]):
        caplog.clear()
        _judge(quote, context, plan, concepts=("kdigo_creat",), proposals=proposals)
        [row] = _shadow_event(caplog)["parts"]
        assert (row["v2"] == "covered") == is_requested
