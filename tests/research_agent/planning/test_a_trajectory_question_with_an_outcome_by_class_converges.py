"""A trajectory-class question that asks an outcome by class plans whole on a source's metadata.

A question such as "into which classes do a vital sign's trajectories over the
first 48 h divide, and how does hospital mortality differ between them?",
planned on the metadata of a source whose ICU-stay key is not named
``stay_id`` (eICU's ``patientunitstayid``), planned on the signed fixed-window
suite but never converged:

- ``bind_plan`` refused the template's description of the frozen classes: the
  description was keyed by the source's stay key, while the signed contract
  admitted only the name the executing run materializes (``stay_id``).  The
  bound plan lost its outcome comparison, and every revision met
  ``PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE`` again;
- the question's requirements that name the coordinate were judged
  unanswered, because the signed candidate owner reads its coordinates
  through the sealed representation, not through its inputs.

The bind is given the run's source, so it admits that source's own stay key
only: one source's stay key can be another's patient key.

Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import easyicu.research_agent.orchestration.progressive_planning as _progressive_planning
import easyicu.research_agent.pipeline as _pipeline
from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    question_requirement_outcome,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    FROZEN_CLASS_DESCRIPTION_STEP_ID,
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.family_spec import (
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_FILENAME,
    QuestionRequirement,
    QuestionRequirementsError,
    judge_recorded_requirements,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.trajectory.runtime_validation import (
    signed_trajectory_plan_contract_errors,
    signed_trajectory_stay_identities,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

_SOURCE_STAY = "patientunitstayid"
_ZH = ("入 ICU 后 48 小时的心率轨迹可分为哪几类？", "各类患者的院内死亡率有何不同？")
_EN = (
    "Into how many classes can heart-rate trajectories over the first 48 hours after "
    "ICU admission be divided",
    "how does hospital mortality differ between the classes",
)
_QUESTIONS = {
    "zh": _ZH[0] + _ZH[1],
    "en": f"{_EN[0]}, and {_EN[1]}?",
}
_QUOTES = {"zh": _ZH, "en": _EN}
_CITATIONS = ("strobe_2007", "record_2015")
_COMPARISON = "phenotyping.outcome_by_cluster"
_CANDIDATES = "observed_data_diagonal_gaussian_mixture_candidate_selection"


def _context(
    question: str = _QUESTIONS["en"],
    *,
    outcome: bool = True,
    database: str = "eicu_demo",
    identities: tuple[str, ...] = (_SOURCE_STAY,),
) -> ResearchContext:
    stay = identities[0]
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="trajectory_outcome_synthetic",
            database=database,
            n_stays=0,
            id_columns=list(identities),
            outcome_columns=["death"] if outcome else [],
            requested_outcome_columns=["death"] if outcome else [],
            provenance={
                "analysis_unit": "icu_stay",
                "patient_identity_available": False,
                "stay_id_columns": [stay],
                "patient_id_columns": [],
                "evidence_stage": "metadata_only_planning",
                "patient_rows_read": False,
            },
        ),
        variables=[
            *(
                ConceptDescriptor(name=name, role=VariableRole.ID, dtype="int64")
                for name in identities
            ),
            ConceptDescriptor(
                name="hr", description="heart rate measurement", role=VariableRole.VITAL,
                dtype="float64", source_concept="hr", analysis_window="icu_admission[0,24]h",
            ),
            *(
                [
                    ConceptDescriptor(
                        name="death", description="hospital mortality",
                        role=VariableRole.OUTCOME, dtype="float64",
                        observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
                    )
                ]
                if outcome
                else []
            ),
        ],
        target_outcome="death" if outcome else None,
        endpoint=(
            EndpointSpec(
                name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]
            )
            if outcome
            else None
        ),
        user_preferences=UserPreferences(
            inferred_analysis_family="trajectory_clustering",
            covariate_selection="planner_selectable",
        ),
    )


def _authorities() -> ScientificRuntimeAuthorities:
    design = load_trajectory_design(
        {
            "coordinate_concepts": ["hr"],
            "descriptive_only_concepts": [],
            "window_start_hours": 0,
            "window_end_hours": 48,
            "grid_width_hours": 8,
        }
    )
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="e" * 64)
    )
    return ScientificRuntimeAuthorities(trajectory=authority, current_case=None)


def _requirements(language: str, *, outcome: bool = True) -> list[dict]:
    classes, by_class = _QUOTES[language]
    return [
        {"id": "r1", "kind": "analysis", "quote": classes, "concepts": ["hr"], "coverage": "plan"},
        *(
            [
                {
                    "id": "r2", "kind": "estimand", "quote": by_class,
                    "concepts": ["hr", "death"], "coverage": "plan",
                }
            ]
            if outcome
            else []
        ),
    ]


def _plan_on_the_suite(context: ResearchContext, requirements: list[dict], tmp_path):
    """Draft on the sealed suite's template, record its requirements, shape and bind it."""

    authorities = _authorities()
    contract = authorities.planning_contract_context()
    types = candidate_analysis_types(context)
    assert (
        family_template_id_for_context(context, analysis_types=types, planning_contract_context=contract)
        == FIXED_WINDOW_TRAJECTORY_FAMILY_ID
    )
    request = build_family_spec_request(
        context, analysis_types=types, variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=_CITATIONS,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=contract,
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": f"{key} (label)"}
            for key in dict.fromkeys(
                [*request.required_reader_label_keys, *request.level_label_keys]
            )
        ],
        "comparator_applications": [],
        "roster_decision_note": "The sealed suite owns every coordinate.",
        "question_requirements": requirements,
    }
    llm = ScriptedMockLLMClient([json.dumps(payload, ensure_ascii=False)])
    attempt = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY, allowed_literature_citation_keys=_CITATIONS,
        direct_comparator_literature_keys=(), comparison_literature_keys=(),
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=contract, required_primary_cohort_selection_mode="all_input_rows",
    )
    assert len(llm.calls) == 1
    draft = attempt.output
    stops = question_requirement_outcome(
        context=context, plan=draft, facts=attempt.facts, run_dir=tmp_path,
        planning_contract_context=contract,
    )
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text(encoding="utf-8"))
    findings: list = []
    plan = _pipeline._shape_fresh_plan(
        pipeline=SimpleNamespace(
            _scientific_runtime_authorities=authorities,
            _enable_publication_figure_skill=True,
            _max_total_steps=24,
        ),
        plan=draft, context=context, agent_context=context,
        long_trajectory_bound=False, findings=findings,
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, bind_findings = authorities.bind_plan(plan, context=context)
    bound = _figure_plan.apply_runtime_bound_figure_contracts(bound, findings)
    authorities.validate_plan(bound)
    _final_plan.validate_final_plan_shape(bound)
    return SimpleNamespace(
        draft=draft, bound=bound, record=record, stops=stops,
        description=bind_findings[0].detail["frozen_class_description"],
    )


def _review(context: ResearchContext, plan: AnalysisPlan):
    return build_plan_scientific_review(
        context=context, plan=plan, literature=None, runtime_authority=None
    )


def _errors(findings) -> list:
    return [finding for finding in findings if finding.severity == "error"]


def test_the_stay_has_its_sources_name_while_planning_and_stay_id_when_run() -> None:
    assert signed_trajectory_stay_identities("eicu_demo") == {"stay_id", _SOURCE_STAY}
    assert signed_trajectory_stay_identities("eicu") == {"stay_id", _SOURCE_STAY}
    assert signed_trajectory_stay_identities("miiv") == {"stay_id"}
    assert signed_trajectory_stay_identities("aumc") == {"stay_id", "admissionid"}
    assert signed_trajectory_stay_identities("hirid") == {"stay_id", "patientid"}
    # A source the host does not register keys its stays only as the run does.
    assert signed_trajectory_stay_identities("unregistered_source") == {"stay_id"}
    every = signed_trajectory_stay_identities()
    assert {"stay_id", _SOURCE_STAY, "admissionid", "icustay_id"} <= every
    # An admission, a hospital stay or a patient group is not the ICU stay.
    assert every.isdisjoint({"hadm_id", "patienthealthsystemstayid", "patient_stay_id", "subject_id"})
    # The union is structural only: HiRID's stay key names the patient in
    # AmsterdamUMCdb, so only a check given the source is exact.
    assert "patientid" in every
    assert "patientid" not in signed_trajectory_stay_identities("aumc")


@pytest.mark.parametrize("language", ["zh", "en"])
def test_a_trajectory_question_with_an_outcome_by_class_plans_whole(language, tmp_path) -> None:
    context = _context(_QUESTIONS[language])
    planned = _plan_on_the_suite(context, _requirements(language), tmp_path)

    # The template keys the description by the source's stay key, and the host
    # wires it to the run cohort and the frozen labels.
    assert planned.description["carried"] is True, planned.description
    description = next(
        step for step in planned.bound.steps if step.step_id == FROZEN_CLASS_DESCRIPTION_STEP_ID
    )
    assert description.phenotype_comparison_spec.identity_column == _SOURCE_STAY
    assert description.phenotype_comparison_spec.outcome_columns == ["death"]
    assert [variable.summary for variable in description.phenotype_comparison_spec.variables] == [
        "count_percent"
    ]
    # Descriptive only: no model is fitted to the outcome.
    assert not any(step.model_requirements for step in planned.bound.steps)
    assert signed_trajectory_plan_contract_errors(planned.bound) == []

    review = _review(context, planned.bound)
    codes = {finding.code for finding in review.findings}
    assert review.approval_allowed, sorted(codes)
    assert not codes & {
        "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE",
        "PHENOTYPING_COMPARISON_CONTRACT_INVALID",
        "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
    }

    # The signed candidate owner reads the sealed coordinate, so the stated
    # requirements rest on the plan's steps, as compiled and as bound.
    assert planned.record["sealed_owner_concepts"] == {_CANDIDATES: ["hr"]}
    assert _errors(planned.stops) == []
    judged = judge_recorded_requirements(planned.record, plan=planned.bound)
    assert _errors(judged.findings()) == []
    rows = {row["id"]: row for row in judged.record["judged"]}
    assert rows["r1"]["disposition"] == "attested"
    assert rows["r1"]["reading_step_ids"] == ["01_authority_compiled_trajectory_candidates"]
    assert rows["r2"]["disposition"] == "attested"
    assert rows["r2"]["reading_step_ids"] == [
        "01_authority_compiled_trajectory_candidates",
        FROZEN_CLASS_DESCRIPTION_STEP_ID,
    ]


def test_the_run_that_executes_it_keys_the_same_description_by_stay_id(tmp_path) -> None:
    context = _context(identities=("stay_id",))
    planned = _plan_on_the_suite(context, _requirements("en"), tmp_path)

    assert planned.description["carried"] is True
    description = next(
        step for step in planned.bound.steps if step.step_id == FROZEN_CLASS_DESCRIPTION_STEP_ID
    )
    assert description.phenotype_comparison_spec.identity_column == "stay_id"
    assert _review(context, planned.bound).approval_allowed


def test_a_trajectory_question_without_an_outcome_describes_no_class(tmp_path) -> None:
    context = _context(_ZH[0], outcome=False)
    planned = _plan_on_the_suite(context, _requirements("zh", outcome=False), tmp_path)

    assert planned.description == {"carried": False, "reason": "not_requested"}
    assert not any(step.scientific_action_id == _COMPARISON for step in planned.bound.steps)
    review = _review(context, planned.bound)
    assert review.approval_allowed, [finding.code for finding in review.findings]
    rows = {row["id"]: row for row in judge_recorded_requirements(planned.record, plan=planned.bound).record["judged"]}
    assert rows["r1"]["disposition"] == "attested"


def _owners_with(
    context: ResearchContext, identity: str, *, on_its_source: bool
) -> tuple[AnalysisPlan, dict]:
    authorities = _authorities()
    owners = authorities.trajectory.development_execution_only_plan(
        research_question=context.research_question
    ).model_dump(mode="json")
    draft = AnalysisPlan.model_validate(
        {
            **owners,
            "steps": [
                *owners["steps"],
                {
                    "step_id": "outcome_by_class",
                    "planned_analysis_role": "secondary",
                    "intent": "Describe hospital mortality by frozen class.",
                    "method": "descriptive_outcome_by_cluster",
                    "scientific_action_id": _COMPARISON,
                    "inputs": [
                        identity, "death", "artifact:analysis_cohort",
                        "table:cluster_assignments", "artifact:stability_freeze",
                    ],
                    "expected_outputs": ["table:outcome_by_cluster"],
                    "phenotype_comparison_spec": {
                        "identity_column": identity,
                        "outcome_columns": ["death"],
                        "variables": [
                            {
                                "name": "death", "variable_kind": "categorical",
                                "summary": "count_percent",
                                "test": "none_descriptive_smd_only", "levels": [0, 1],
                            }
                        ],
                    },
                },
            ],
        }
    )
    bound, findings = authorities.bind_plan(draft, context=context if on_its_source else None)
    return bound, findings[0].detail["frozen_class_description"]


def _comparisons(plan: AnalysisPlan) -> list:
    return [step for step in plan.steps if step.scientific_action_id == _COMPARISON]


def test_a_description_keyed_by_another_identity_covers_nothing(tmp_path) -> None:
    # A hospital stay is not the ICU stay the frozen labels are keyed by, on
    # this source or any other.
    context = _context(identities=(_SOURCE_STAY, "patienthealthsystemstayid"))
    for on_its_source in (True, False):
        bound, description = _owners_with(
            context, "patienthealthsystemstayid", on_its_source=on_its_source
        )
        assert description["reason"] == "signed_contract_refused"
        assert not _comparisons(bound)
    codes = {finding.code for finding in _review(context, bound).findings}
    assert "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" in codes

    # Another source's stay key is not this source's stay: the bind, given the
    # run's source, does not carry it, and no outcome is described.
    context = _context(identities=(_SOURCE_STAY, "admissionid"))
    bound, description = _owners_with(context, "admissionid", on_its_source=True)
    assert description["reason"] == "signed_contract_refused"
    assert not _comparisons(bound)
    review = _review(context, bound)
    assert "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" in {finding.code for finding in review.findings}
    assert not review.approval_allowed

    # A context-free bind admits any source's stay key; the review, which
    # knows the source, still admits no comparison on it.
    bound, description = _owners_with(context, "admissionid", on_its_source=False)
    assert description["carried"] is True
    assert signed_trajectory_plan_contract_errors(bound) == []
    assert signed_trajectory_plan_contract_errors(bound, database=context.cohort.database)
    review = _review(context, bound)
    messages = {finding.code: finding.message for finding in review.findings}
    assert "phenotype_comparison_identity_invalid" in messages["PHENOTYPING_COMPARISON_CONTRACT_INVALID"]
    assert "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" in messages
    assert not review.approval_allowed


def test_a_patient_key_named_as_another_sources_stay_is_refused_at_bind(tmp_path) -> None:
    # AmsterdamUMCdb keys its ICU stays by admissionid and its patients by
    # patientid, the name HiRID gives its stays.
    context = _context(database="aumc", identities=("admissionid", "patientid"))
    bound, description = _owners_with(context, "patientid", on_its_source=True)
    assert description["carried"] is False
    assert description["reason"] == "signed_contract_refused"
    assert not _comparisons(bound)
    review = _review(context, bound)
    assert "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" in {finding.code for finding in review.findings}
    assert not review.approval_allowed

    # Bound without the source, it passes the structural contract only; the
    # contract given the source and the review both refuse it.
    loose, description = _owners_with(context, "patientid", on_its_source=False)
    assert description["carried"] is True
    assert signed_trajectory_plan_contract_errors(loose, database="aumc")
    review = _review(context, loose)
    messages = {finding.code: finding.message for finding in review.findings}
    assert "phenotype_comparison_identity_invalid" in messages["PHENOTYPING_COMPARISON_CONTRACT_INVALID"]
    assert not review.approval_allowed

    # The same source's own stay key is carried.
    bound, description = _owners_with(context, "admissionid", on_its_source=True)
    assert description["carried"] is True
    assert signed_trajectory_plan_contract_errors(bound, database="aumc") == []


def test_a_source_that_keys_its_stays_by_patientid_describes_its_classes(tmp_path) -> None:
    # HiRID keys its ICU stays by patientid.
    context = _context(database="hirid", identities=("patientid",))
    bound, description = _owners_with(context, "patientid", on_its_source=True)
    assert description["carried"] is True, description
    assert signed_trajectory_plan_contract_errors(bound, database="hirid") == []
    (comparison,) = _comparisons(bound)
    assert comparison.phenotype_comparison_spec.identity_column == "patientid"
    codes = {finding.code for finding in _review(context, bound).findings}
    assert not codes & {
        "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE",
        "PHENOTYPING_COMPARISON_CONTRACT_INVALID",
    }


def test_an_outcome_read_by_no_valid_step_stays_unanswered(tmp_path) -> None:
    context = _context()
    planned = _plan_on_the_suite(context, _requirements("en"), tmp_path)
    without = planned.bound.model_copy(
        update={
            "steps": [
                step for step in planned.bound.steps if step.scientific_action_id != _COMPARISON
            ]
        }
    )
    rows = {
        row["id"]: row
        for row in judge_recorded_requirements(planned.record, plan=without).record["judged"]
    }
    # The coordinate is still read by the signed owner; the outcome is read by none.
    assert rows["r1"]["disposition"] == "attested"
    assert rows["r2"]["disposition"] == "not_covered"


def test_a_record_credits_a_sealed_owner_only_as_it_states(tmp_path) -> None:
    context = _context()
    planned = _plan_on_the_suite(context, _requirements("en"), tmp_path)

    # A record written before it named the sealed owners credits none.
    older = {key: value for key, value in planned.record.items() if key != "sealed_owner_concepts"}
    rows = {
        row["id"]: row for row in judge_recorded_requirements(older, plan=planned.bound).record["judged"]
    }
    assert rows["r1"]["disposition"] == "not_covered"

    with pytest.raises(QuestionRequirementsError) as caught:
        judge_recorded_requirements(
            {**planned.record, "sealed_owner_concepts": {_CANDIDATES: "hr"}}, plan=planned.bound
        )
    assert caught.value.reason_code == "question_requirements_unreadable"


def test_a_disclosure_without_a_candidate_owner_credits_no_owner(tmp_path, monkeypatch) -> None:
    disclosed = _progressive_planning.sealed_trajectory_suite_coordinates
    monkeypatch.setattr(
        _progressive_planning,
        "sealed_trajectory_suite_coordinates",
        lambda contract: disclosed(contract).model_copy(update={"candidate_owner": ""}),
    )
    planned = _plan_on_the_suite(_context(), _requirements("en"), tmp_path)

    # No owner is named, so none is credited: the record carries no field and
    # the coordinate is read by no step.
    assert "sealed_owner_concepts" not in planned.record
    rows = {
        row["id"]: row
        for row in judge_recorded_requirements(planned.record, plan=planned.bound).record["judged"]
    }
    assert rows["r1"]["disposition"] == "not_covered"


def test_requirements_parse_as_stated() -> None:
    # The fixtures state requirements the owner's own model accepts.
    for language in ("zh", "en"):
        for item in _requirements(language):
            QuestionRequirement.model_validate(item)
