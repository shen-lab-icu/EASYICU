"""A population the Planner states decides the plan's cohort.

Step 2b of the population spec design.  The Planner states whom the study
includes twice: as cohort predicates and as a typed spec.  The predicates
were the ones applied, and a Planner that wrote a stay length in hours on a
column recorded in days, or a status by its first value instead of its
presence in the window, analysed a population other than the one stated with
nothing to say so.  Now a stated spec decides the cohort: the host compiles
it at the plan's own time zero, and the Planner's predicates no longer
select rows.  Where they differ, a typed finding records it.

A spec its owner refuses, or one whose quote of the study is not the study's
own words (a paraphrase, a translation), goes back to the Planner.  An
inclusion the cohort does not apply no longer stops planning: the plan is
made and lists it as unapplied, but its approval is refused with a typed
reason, one per remedy (an extraction of the study's own population, or
nothing that can apply it as stated), because analysing everyone else would
include stays the study excludes.  The cohort step documents only the
criteria the cohort applies, and criteria the study did not state are
recorded as the plan's proposals.  Without a spec, the Planner's predicates
select the rows as before.  Contexts, questions and specs are synthetic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.orchestration.progressive_planning import (
    population_approval_findings,
    population_proposals_finding,
    run_progressive_planner,
)
from easyicu.research_agent.orchestration.workflow import (
    human_review_requests_for_plan,
)
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_shadow import (
    population_cohort_audit,
    superseded_predicates_finding,
)
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
    stated_population,
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
    _payload,
)

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
_QUESTION = (
    "Among adults aged 18 years or older who stayed in the ICU for at least "
    "12 hours and met the marker criteria within the first day, is the "
    "exposure associated with the outcome?"
)
_ADULTS = {
    "kind": "age_years",
    "min_years": 18,
    "quote": "adults aged 18 years or older",
}
_STAYED = {
    "kind": "icu_stay_hours",
    "min_hours": 12,
    "quote": "stayed in the ICU for at least 12 hours",
}
_MARKED = {
    "kind": "condition_present",
    "concepts_all_of": ["marker_flag"],
    "window": {"start_hours": 0, "end_hours": 24},
    "quote": "met the marker criteria within the first day",
}


def _context(
    *, question: str = _QUESTION, window: float | None = 24.0, wording=()
) -> ResearchContext:
    base = _planner_context()
    constraints = (
        json.dumps(
            {
                "materialization_window": {
                    "role": "outer_observation_window",
                    "anchor": "ICU admission",
                    "hours": window,
                }
            }
        )
        if window is not None
        else None
    )
    return base.model_copy(
        update={
            "research_question": question,
            "cohort": base.cohort.model_copy(
                update={"inclusion_criteria": list(wording)}
            ),
            "variables": [
                *base.variables,
                ConceptDescriptor(
                    name="age",
                    role=VariableRole.DEMOGRAPHIC,
                    dtype="float64",
                    unit="years",
                ),
                ConceptDescriptor(
                    name="los_icu",
                    role=VariableRole.OUTCOME,
                    dtype="float64",
                    unit="days",
                ),
                ConceptDescriptor(
                    name="marker_flag",
                    role=VariableRole.OTHER,
                    dtype="int64",
                    analysis_window="icu_admission[0,24]h",
                    observed_domain=_BINARY,
                ),
            ],
            "user_preferences": UserPreferences(data_constraints=constraints),
        }
    )


def _spec(*criteria: dict[str, Any], source: str = "question") -> dict[str, Any]:
    return {
        "criteria": [
            {"id": f"c{index}", "source": source, "role": "include", **item}
            for index, item in enumerate(criteria, start=1)
        ]
    }


def _predicate(concept, op, value, *, end=24.0, aggregation="first") -> dict[str, Any]:
    return {
        "concept_id": concept,
        "anchor": "icu_admission",
        "start_offset_hours": 0.0,
        "end_offset_hours": end,
        "aggregation": aggregation,
        "op": op,
        "value": {"mode": "number", "number_value": value},
    }


#: The Planner's own predicates for the same population, as they were
#: written: the stay length in hours on a column recorded in days, and the
#: status by its first value.
_PLANNERS_OWN = (
    _predicate("age", ">=", 18),
    _predicate("los_icu", ">=", 12),
    _predicate("marker_flag", "==", 1),
)


def _cohort(spec: Any, *inclusion: dict[str, Any], criteria=()) -> dict[str, Any]:
    return {
        "name": "primary",
        "selection_mode": "predicate_filtered" if inclusion else "all_input_rows",
        "inclusion": list(inclusion),
        "exclusion": [],
        "population_criteria": list(criteria),
        **({"population_spec": spec} if spec is not None else {}),
    }


def _skeleton(cohort: dict[str, Any]) -> ProgressivePlanSkeleton:
    return ProgressivePlanSkeleton.model_validate({**_payload(), "cohort": cohort})


def _compile(cohort: dict[str, Any], context: ResearchContext | None = None):
    return compile_progressive_plan(
        skeleton=_skeleton(cohort), context=context or _context()
    )


def _foundation(cohort: dict[str, Any], context: ResearchContext | None = None) -> None:
    skeleton = _skeleton(cohort)
    validate_progressive_foundation(
        ProgressivePlanFoundation(
            cohort=skeleton.cohort,
            display_labels=skeleton.display_labels,
            robustness_intents=skeleton.robustness_intents,
            know_how_decisions=skeleton.know_how_decisions,
        ),
        context=context or _context(),
        analysis_type=skeleton.analysis_type,
    )


def _selection(cohort: CohortDefinition | dict[str, Any]) -> tuple:
    data = cohort if isinstance(cohort, dict) else cohort.plan_dict()
    return (
        # A cohort omits its default mode.
        data.get("selection_mode", "predicate_filtered"),
        data["inclusion"],
        data["exclusion"],
        list(data.get("unapplied_population_criteria") or []),
    )


def _compiled(spec: dict[str, Any], context: ResearchContext, time_zero: float | None):
    return compile_population(
        PopulationSpec.model_validate(spec), context, time_zero_hours=time_zero
    ).cohort_definition()


# -- the cohort the plan applies -----------------------------------------------


def test_the_plan_applies_the_cohort_its_spec_compiles_to() -> None:
    spec = _spec(_ADULTS, _STAYED, _MARKED)
    context = _context()

    plan, _receipt = _compile(_cohort(spec, *_PLANNERS_OWN), context)

    assert _selection(plan.cohort) == _selection(_compiled(spec, context, 24.0))
    rows = {
        (item.concept_id, item.aggregation, item.op, item.value)
        for item in plan.cohort.inclusion
    }
    # 12 hours on a column recorded in days, and the status present in the window.
    assert ("los_icu", "first", ">=", 0.5) in rows
    assert ("marker_flag", "max", "==", 1) in rows
    assert ("los_icu", "first", ">=", 12.0) not in rows
    assert ("marker_flag", "first", "==", 1.0) not in rows


def test_without_a_spec_the_planners_predicates_select_the_rows() -> None:
    plan, _receipt = _compile(_cohort(None, *_PLANNERS_OWN))

    rows = [(item.concept_id, item.op, item.value) for item in plan.cohort.inclusion]
    assert rows == [
        ("age", ">=", 18.0),
        ("los_icu", ">=", 12.0),
        ("marker_flag", "==", 1.0),
    ]


def test_an_empty_spec_keeps_every_input_row() -> None:
    plan, _receipt = _compile(_cohort({"criteria": []}, *_PLANNERS_OWN))

    assert _selection(plan.cohort) == ("all_input_rows", [], [], [])


def _eligibility(plan) -> list[str]:
    step = next(step for step in plan.steps if step.cohort_definition_spec is not None)
    return [
        item.description for item in step.cohort_definition_spec.eligibility_criteria
    ]


def test_the_cohort_step_documents_only_the_criteria_its_cohort_applies() -> None:
    context = _context(
        question=(
            "Among adults aged 18 years or older who stayed in the ICU, excluding "
            "those who met the marker criteria within the first day or had a "
            "prior referral, is the exposure associated with the outcome?"
        )
    )
    marked = {**_MARKED, "role": "exclude"}
    referral = {
        "kind": "not_typed",
        "why": "no kind states a prior referral",
        "role": "exclude",
        "quote": "had a prior referral",
    }
    long_stay = {**_STAYED, "min_hours": 48, "quote": "stayed in the ICU"}
    spec = _spec(_ADULTS, long_stay, marked, referral)

    plan, _receipt = _compile(_cohort(spec, *_PLANNERS_OWN), context)

    # The eligibility a flow diagram reports: never a criterion left unapplied.
    assert _eligibility(plan) == [
        _ADULTS["quote"],
        f"Exclude: {_MARKED['quote']}",
    ]
    assert plan.cohort.unapplied_population_criteria == (
        "stayed in the ICU",
        "had a prior referral",
    )


def test_the_studys_own_eligibility_is_documented_as_it_states_it() -> None:
    context = _context(wording=["Adults aged 18 years or older"])

    plan, _receipt = _compile(_cohort(_spec(_ADULTS, _STAYED), *_PLANNERS_OWN), context)

    assert _eligibility(plan) == ["Adults aged 18 years or older"]


# -- an inclusion the plan cannot apply ----------------------------------------


_LONG_STAY = {**_STAYED, "min_hours": 48, "quote": "stayed in the ICU"}
_FIRST_STAY = {"kind": "first_icu_stay", "quote": "in their first ICU stay"}
_FIRST_STAY_QUESTION = (
    "Among adults aged 18 years or older in their first ICU stay who stayed in "
    "the ICU for at least 12 hours, is the exposure associated with the outcome?"
)


def _stops(cohort: dict[str, Any], context: ResearchContext | None = None):
    context = context or _context(question=_FIRST_STAY_QUESTION)
    skeleton = _skeleton(cohort)
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=context)
    population = stated_population(skeleton.cohort, context=context, plan=plan)
    return plan, population_approval_findings(population)


def _rows(finding) -> list[tuple]:
    return [
        (row["id"], row["disposition"], row["reason"], row["stated_by_study"])
        for row in finding.detail["criteria"]
    ]


def test_an_inclusion_decided_after_time_zero_leaves_a_plan_that_cannot_be_approved() -> (
    None
):
    plan, stops = _stops(_cohort(_spec(_ADULTS, _LONG_STAY), *_PLANNERS_OWN[:2]))

    # The plan is made, and its cohort lists the inclusion it does not apply.
    assert plan.cohort.unapplied_population_criteria == ("stayed in the ICU",)
    assert [item.concept_id for item in plan.cohort.inclusion] == ["age"]
    (stop,) = stops
    assert (stop.validator, stop.severity) == ("population_compile", "error")
    assert stop.detail["reason"] == "population_inclusion_not_applied"
    assert stop.detail["approval_allowed"] is False
    assert _rows(stop) == [
        ("c2", "not_applied", "population_determined_after_time_zero", True)
    ]
    assert stop.detail["time_zero_hours"] == 24.0
    # Its review cannot be approved, and no approval is requested beside it.
    requests = human_review_requests_for_plan(
        findings=[stop], plan=plan, require_plan_review=True
    )
    assert [
        (item.kind, item.payload["reason"], item.payload["approval_allowed"])
        for item in requests
    ] == [("scientific_stop", "population_inclusion_not_applied", False)]
    assert requests[0].summary.startswith("This plan cannot be approved")
    assert "'stayed in the ICU'" in requests[0].summary


def test_an_inclusion_only_an_extraction_applies_names_that_extraction() -> None:
    plan, stops = _stops(_cohort(_spec(_ADULTS, _FIRST_STAY), _PLANNERS_OWN[0]))

    assert plan.cohort.unapplied_population_criteria == ("in their first ICU stay",)
    (stop,) = stops
    assert stop.detail["reason"] == "population_inclusion_requires_extraction"
    assert stop.detail["approval_allowed"] is False
    assert stop.detail["human_review_required"] is True
    assert _rows(stop) == [
        (
            "c2",
            "requires_extraction",
            "population_first_icu_stay_not_restricted",
            True,
        )
    ]
    assert "extraction of the study's own population" in stop.message


def test_each_remedy_refuses_approval_with_its_own_reason() -> None:
    _plan, stops = _stops(
        _cohort(_spec(_ADULTS, _LONG_STAY, _FIRST_STAY), _PLANNERS_OWN[0])
    )

    assert [
        (item.detail["reason"], [row[0] for row in _rows(item)]) for item in stops
    ] == [
        ("population_inclusion_requires_extraction", ["c3"]),
        ("population_inclusion_not_applied", ["c2"]),
    ]


def test_a_cohort_that_applies_every_inclusion_refuses_nothing() -> None:
    _plan, stops = _stops(
        _cohort(_spec(_ADULTS, _STAYED, _MARKED), *_PLANNERS_OWN), _context()
    )

    assert stops == []


def test_the_same_inclusion_is_applied_when_the_time_zero_follows_it() -> None:
    long_stay = {**_STAYED, "min_hours": 48, "quote": "stayed in the ICU"}
    context = _context(window=72.0)

    plan, _receipt = _compile(_cohort(_spec(long_stay), _PLANNERS_OWN[1]), context)

    assert [(item.concept_id, item.value) for item in plan.cohort.inclusion] == [
        ("los_icu", 2.0)
    ]


def test_an_exclusion_the_plan_cannot_apply_is_listed_not_stopped() -> None:
    excluded = {
        "kind": "not_typed",
        "why": "no kind states a prior referral",
        "role": "exclude",
        "quote": "met the marker criteria",
    }

    plan, stops = _stops(
        _cohort(_spec(_ADULTS, excluded), _PLANNERS_OWN[0]), _context()
    )

    assert plan.cohort.unapplied_population_criteria == ("met the marker criteria",)
    assert [item.concept_id for item in plan.cohort.inclusion] == ["age"]
    # An exclusion left unapplied keeps more stays; it refuses no approval.
    assert stops == []


# -- what goes back to the Planner ---------------------------------------------


def test_a_spec_its_owner_refuses_goes_back_to_the_planner() -> None:
    nested = {
        "criteria": [
            {
                "id": "c1",
                "quote": _ADULTS["quote"],
                "source": "question",
                "role": "include",
                "kind": "age_years",
                "age_years": {"min_years": 18},
            }
        ]
    }

    with pytest.raises(ProgressivePlanCompileError) as refused:
        _foundation(_cohort(nested))

    assert refused.value.reason_code == "progressive_population_spec_invalid"
    assert refused.value.path == "cohort.population_spec"
    assert "beside kind" in str(refused.value)


@pytest.mark.parametrize(
    "quote",
    ["adults at least 18 years old", "成年患者（18 岁及以上）", "adults"],
    ids=["paraphrase", "translation", "fragment-not-in-wording"],
)
def test_a_quote_not_written_in_the_study_goes_back_to_the_planner(quote) -> None:
    context = _context(
        question="Among adults aged 18 years or older, is X associated with Y?"
    )
    if quote == "adults":
        context = _context(
            question="In those aged 18 years or older, is X associated with Y?"
        )
    spec = _spec({**_ADULTS, "quote": quote})

    with pytest.raises(ProgressivePlanCompileError) as refused:
        _foundation(_cohort(spec), context)

    assert refused.value.reason_code == "progressive_population_quote_not_verbatim"
    assert refused.value.path == "cohort.population_spec.criteria[0].quote"


@pytest.mark.parametrize(
    "question, quote",
    [
        ("纳入年龄 ≥ 18 岁的成人", "年龄≥18岁"),
        (
            "Among Adults Aged 18 Years Or Older, is X associated with Y?",
            "adults aged 18 years or older",
        ),
        ("纳入年龄＞18岁的成人", "年龄>18岁"),
    ],
    ids=["spacing", "letter-case", "full-width-form"],
)
def test_spacing_case_and_width_are_not_words(question, quote) -> None:
    _foundation(
        _cohort(_spec({**_ADULTS, "quote": quote})), _context(question=question)
    )


def test_the_studys_own_wording_holds_its_quote() -> None:
    context = _context(
        question="Is X associated with Y?", wording=["Adults aged 18 years or older"]
    )

    _foundation(_cohort(_spec(_ADULTS, source="study_wording")), context)
    _foundation(_cohort(_spec(_ADULTS, source="question")), context)


@pytest.mark.parametrize("source", ["outline", "preset"])
def test_a_quote_from_the_outline_or_a_preset_is_not_held_to_the_study(source) -> None:
    spec = _spec(
        {**_ADULTS, "quote": "adult population of the template"}, source=source
    )

    _foundation(_cohort(spec), _context(question="Is X associated with Y?"))


def test_with_a_spec_the_planners_predicates_are_not_its_population() -> None:
    # A stated criterion that no predicate applies fails without a spec...
    criteria = [{"criterion": "adults only", "concept_ids": ["age"]}]
    unapplied = _cohort(None, _PLANNERS_OWN[2], criteria=criteria)
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _foundation(unapplied)
    assert refused.value.reason_code == "progressive_population_criterion_unapplied"

    # ...and is not the Planner's to apply once the spec decides the cohort.
    _foundation({**unapplied, "population_spec": _spec(_ADULTS)})


# -- criteria the study did not state -------------------------------------------


def test_criteria_the_study_did_not_state_are_recorded_as_the_plans_proposals() -> None:
    context = _context()
    spec = {
        "criteria": [
            {"id": "c1", "source": "question", "role": "include", **_ADULTS},
            {
                "id": "c2",
                "source": "outline",
                "role": "include",
                **_STAYED,
                "quote": "stays long enough to observe the first day",
            },
        ]
    }
    skeleton = _skeleton(_cohort(spec))
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=context)

    finding = population_proposals_finding(
        stated_population(skeleton.cohort, context=context, plan=plan)
    )

    assert finding is not None
    assert (finding.validator, finding.severity) == ("population_compile", "warning")
    assert finding.detail["reason_code"] == "population_criteria_proposed_by_system"
    assert [
        (row["id"], row["source"], row["stated_by_study"], row["disposition"])
        for row in finding.detail["criteria"]
    ] == [("c2", "outline", False, "applied_by_plan")]
    assert "'stays long enough to observe the first day'" in finding.message


def test_a_population_the_study_states_in_full_proposes_nothing() -> None:
    context = _context()
    skeleton = _skeleton(_cohort(_spec(_ADULTS, _STAYED, _MARKED)))
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=context)

    population = stated_population(skeleton.cohort, context=context, plan=plan)

    assert population_proposals_finding(population) is None
    assert (
        stated_population(
            _skeleton(_cohort(None, *_PLANNERS_OWN)).cohort, context=context, plan=plan
        )
        is None
    )


# -- the record of a difference ------------------------------------------------


def _audit(cohort: dict[str, Any], context: ResearchContext | None = None):
    context = context or _context()
    plan, _receipt = _compile(cohort, context)
    return plan, population_cohort_audit(
        context=context, plan=plan, cohort=_skeleton(cohort).cohort
    )


def test_predicates_that_differ_from_the_spec_are_a_typed_finding() -> None:
    plan, audit = _audit(_cohort(_spec(_ADULTS, _STAYED, _MARKED), *_PLANNERS_OWN))

    assert audit["cohort_source"] == "population_spec"
    assert audit["plan_applies_compiled"] is True
    finding = superseded_predicates_finding(audit)
    assert finding is not None
    assert finding.validator == "population_compile"
    assert finding.detail["reason_code"] == "population_planner_predicates_superseded"
    planner_only = {
        (item["concept_id"], item["aggregation"], item["op"], item["value"])
        for item in finding.detail["inclusion"]["plan_only"]
    }
    assert planner_only == {
        ("los_icu", "first", ">=", 12.0),
        ("marker_flag", "first", "==", 1.0),
    }
    assert finding.detail["time_zero_hours"] == 24.0


def test_predicates_that_agree_with_the_spec_leave_no_finding() -> None:
    # One value per stay: a window on age or the stay length selects the same rows.
    agreeing = (
        _predicate("age", ">=", 18),
        _predicate("los_icu", ">=", 0.5),
        _predicate("marker_flag", "==", 1, aggregation="max"),
    )

    _plan, audit = _audit(_cohort(_spec(_ADULTS, _STAYED, _MARKED), *agreeing))

    assert audit["differs"] is False
    assert superseded_predicates_finding(audit) is None


def test_without_a_spec_the_audit_reads_the_planners_cohort() -> None:
    _plan, audit = _audit(_cohort(None, *_PLANNERS_OWN))

    assert audit["status"] == "no_spec"
    assert audit["cohort_source"] == "planner_predicates"
    assert superseded_predicates_finding(audit) is None


def test_planner_predicates_the_roster_cannot_read_are_still_recorded() -> None:
    context = _context()
    spec = _spec(_ADULTS)
    unknown = _predicate("not_a_concept", ">=", 1)
    skeleton = _skeleton(_cohort(spec, unknown))
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=context)

    audit = population_cohort_audit(context=context, plan=plan, cohort=skeleton.cohort)

    assert audit["planner_cohort_unreadable"]
    finding = superseded_predicates_finding(audit)
    assert finding is not None
    assert (
        finding.detail["planner_cohort_unreadable"]
        == audit["planner_cohort_unreadable"]
    )


# -- a planning run ------------------------------------------------------------


class _Evidence:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def get(self, evidence_id_or_alias: str) -> object | None:
        return self.records.get(evidence_id_or_alias)

    def register_file(self, **kwargs: object) -> object:
        source = Path(str(kwargs["source_path"]))
        self.records[str(kwargs["evidence_id"])] = {
            **dict(kwargs),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        return self.records[str(kwargs["evidence_id"])]


def _planning_run(tmp_path: Path, cohort: dict[str, Any]):
    foundation = _foundation_payload()
    foundation["foundation"]["cohort"] = cohort
    llm = ScriptedMockLLMClient(
        [
            json.dumps(_outline_payload()),
            json.dumps(foundation),
            *[json.dumps(item) for item in _materialization_payloads()],
        ]
    )
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic cohort")
    findings: list[Any] = []

    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(llm),
        context=_context(),
        run_dir=tmp_path,
        evidence=_Evidence(),
        prompt_pack_version="test-v1",
        resume_checkpoint_path=None,
        resume_checkpoint_sha256=None,
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=findings.append,
    )
    return result, findings


def test_a_planning_run_applies_the_spec_and_records_what_it_did_not(
    tmp_path: Path,
) -> None:
    result, findings = _planning_run(
        tmp_path, _cohort(_spec(_ADULTS, _STAYED, _MARKED), *_PLANNERS_OWN)
    )

    rows = {
        (item.concept_id, item.aggregation, item.op, item.value)
        for item in result.plan.cohort.inclusion
    }
    assert ("los_icu", "first", ">=", 0.5) in rows
    assert ("los_icu", "first", ">=", 12.0) not in rows
    (finding,) = [item for item in findings if item.validator == "population_compile"]
    assert finding.detail["reason_code"] == "population_planner_predicates_superseded"
    audit = json.loads((tmp_path / "population_shadow_audit.json").read_text("utf-8"))
    assert audit["cohort_source"] == "population_spec"
    assert audit["plan_applies_compiled"] is True


def test_a_planning_run_makes_the_plan_and_refuses_its_approval(
    tmp_path: Path,
) -> None:
    spec = _spec(_ADULTS, _LONG_STAY)
    spec["criteria"].append(
        {
            "id": "c3",
            "source": "outline",
            "role": "include",
            **_MARKED,
            "quote": "the marker on the first day",
        }
    )

    result, findings = _planning_run(tmp_path, _cohort(spec, *_PLANNERS_OWN[:2]))

    assert result.plan.cohort.unapplied_population_criteria == ("stayed in the ICU",)
    stops = [item for item in findings if item.severity == "error"]
    assert [item.detail["reason"] for item in stops] == [
        "population_inclusion_not_applied"
    ]
    (request,) = human_review_requests_for_plan(
        findings=findings, plan=result.plan, require_plan_review=True
    )
    assert request.payload["approval_allowed"] is False
    (proposals,) = [
        item
        for item in findings
        if (item.detail or {}).get("reason_code")
        == "population_criteria_proposed_by_system"
    ]
    assert [row["id"] for row in proposals.detail["criteria"]] == ["c3"]
