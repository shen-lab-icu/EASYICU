"""A question that asks how often its exposure occurs gets a step that says so.

A landmark association keeps only the stays alive and observed at the
landmark, so its level counts describe a narrower population than the one the
study selected.  When the question also asks what proportion of stays have the
exposure, nothing in the plan answered that: the template reported the
exposure only on the landmark cohort, and the review had no check that each
part of the question has a step.  Now the host seals the request, the
landmark template republishes the study cohort through the host root and
reports the exposure's level distribution on it, and the review blocks a plan
in which no step covers that part of the question.  Every context here is
synthetic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent import pipeline as research_pipeline
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.primary_cohort import step_cohort_population
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import figure_plan_shaping, final_plan_shape
from easyicu.research_agent.planning.analysis_types import (
    requested_exposure_occurrence_cues,
)
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecRequest,
    StudyPopulationOccurrence,
    spec_from_mapping,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    requested_exposure_occurrence,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    MissingnessProfile,
    VariableRole,
)

from .family_spec_fixtures import (
    LABELS,
    PLANNER_ROSTER,
    _context,
    _descriptive_context,
    _request,
    _run,
    _spec_payload,
)

ASKS = " What proportion of stays reach each injury stage?"
OCCURRENCE_CODE = "REQUESTED_OCCURRENCE_COVERAGE_INCOMPLETE"


def _question(context, text: str):
    return context.model_copy(update={"research_question": text})


# ---------------------------------------------------------------------------
# Which words ask for an occurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "cues"),
    [
        ("What proportion of stays are ventilated, and is it associated with death?", ("proportion",)),
        ("Report the prevalence of early shock and its mortality.", ("prevalence",)),
        ("How common is early vasopressor use among adult ICU stays?", ("how common",)),
        ("早期机械通气的比例是多少？与死亡相关吗？", ("比例",)),
        ("统计各分期的占比和发生率。", ("占比", "发生率")),
    ],
)
def test_a_question_asking_how_often_the_exposure_occurs_is_read(question, cues) -> None:
    assert requested_exposure_occurrence_cues(_question(_context(), question)) == cues


@pytest.mark.parametrize(
    "question",
    [
        "Does the effect violate proportional hazards?",
        "Is the proportional odds assumption met for the ordinal outcome?",
        "Estimate the proportion of variance explained by the centre.",
        "What proportion of the effect is mediated by shock?",
        "Report the proportion missing for each covariate.",
        "What is the cumulative incidence of death by day 28?",
        "Estimate the incidence rate ratio per 1000 patient-days.",
        "Report the risk difference in percentage points.",
        "What is the mortality rate and how many stays died?",
        "检验比例风险假设，并报告缺失比例。",
        "报告28天累积发生率。",
    ],
)
def test_a_phrase_that_asks_something_else_is_not_an_occurrence(question) -> None:
    assert requested_exposure_occurrence_cues(_question(_context(), question)) == ()


def test_only_the_question_and_its_requested_outputs_are_read() -> None:
    context = _question(_context(), "How is the stage associated with death?")
    assert requested_exposure_occurrence_cues(context) == ()
    asked = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"must_have_outputs": "The prevalence of each stage."}
            )
        }
    )
    assert requested_exposure_occurrence_cues(asked) == ("prevalence",)
    # Design prose (data constraints) never asks the question.
    described = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps({"note": "report the proportion"})}
            )
        }
    )
    assert requested_exposure_occurrence_cues(described) == ()


# ---------------------------------------------------------------------------
# The sealed request
# ---------------------------------------------------------------------------


def _post_baseline(context):
    """The exposure is ascertained in the first 24 h after ICU admission."""

    return context.model_copy(
        update={
            "variables": [
                item.model_copy(
                    update={
                        "analysis_window": "icu_admission[0,24]h",
                        "analysis_window_role": "exposure_definition",
                    }
                )
                if item.name == context.primary_exposure
                else item
                for item in context.variables
            ]
        }
    )


def _asking(context=None, suffix: str = ASKS):
    base = _post_baseline(context or _context())
    return base.model_copy(update={"research_question": base.research_question + suffix})


def test_the_landmark_request_seals_the_occurrence_the_question_asks_for() -> None:
    request = _request(_asking())

    occurrence = request.study_population_occurrence
    assert occurrence is not None
    assert occurrence.product_id == "cohort:study_population"
    assert occurrence.requested_cues == ["proportion"]
    # No known missing values: a missing row would stop the step.
    assert (
        occurrence.denominator_policy,
        occurrence.missing_exposure_policy,
        occurrence.missing_outcome_policy,
    ) == ("all_declared_rows", "fail_closed", "fail_closed")


def test_without_the_question_or_a_post_baseline_exposure_nothing_is_sealed() -> None:
    plain = _request(_post_baseline(_context()))
    baseline_exposure = _request(
        _question(_context(), _context().research_question + ASKS)
    )

    for request in (plain, baseline_exposure):
        assert request.study_population_occurrence is None
        assert request.level_label_keys == []
        # The request keeps the identity it had before the field existed.
        assert "study_population_occurrence" not in request.model_dump(mode="json")


def test_known_missing_values_leave_the_denominator_and_are_counted() -> None:
    context = _asking()
    context = context.model_copy(
        update={
            "variables": [
                item.model_copy(
                    update={
                        "missingness": MissingnessProfile(
                            fraction_missing=0.035, n_missing=7, n_total=200
                        )
                    }
                )
                if item.name in {"injury_stage", "death"}
                else item
                for item in context.variables
            ]
        }
    )

    occurrence = _request(context).study_population_occurrence

    assert occurrence is not None
    assert (
        occurrence.denominator_policy,
        occurrence.missing_exposure_policy,
        occurrence.missing_outcome_policy,
    ) == ("observed_outcome_rows", "exclude_from_denominator", "exclude_from_denominator")


def test_a_cohort_named_after_the_study_population_gets_the_other_spelling() -> None:
    context = _asking()
    context = context.model_copy(
        update={"cohort": context.cohort.model_copy(update={"cohort_name": "study_population"})}
    )

    request = _request(context)

    assert request.study_population_occurrence.product_id == "cohort:eligible_study_population"


def test_the_occurrence_belongs_only_to_the_categorical_landmark_family() -> None:
    request = _request(_asking())
    payload = request.model_dump(mode="json")

    with pytest.raises(ValueError, match="categorical landmark family"):
        FamilySpecRequest.model_validate(
            {
                **payload,
                "family_id": "landmark_spline_association",
                "exposure_kind": "continuous",
                "exposure_levels": [],
                "exposure_is_ordered": False,
                "reference_level_index": 0,
                "primary_contrast_level_index": 0,
                "alternate_exposures": [],
            }
        )
    with pytest.raises(ValueError, match="cannot claim"):
        FamilySpecRequest.model_validate(
            {
                **payload,
                "study_population_occurrence": {
                    **payload["study_population_occurrence"],
                    "product_id": "cohort:eligible_study_population",
                },
            }
        )
    with pytest.raises(ValueError, match="observed-outcome rows"):
        StudyPopulationOccurrence(
            product_id="cohort:study_population",
            requested_cues=["proportion"],
            denominator_policy="all_declared_rows",
            missing_exposure_policy="fail_closed",
            missing_outcome_policy="exclude_from_denominator",
        )


# ---------------------------------------------------------------------------
# A binary exposure names its two levels
# ---------------------------------------------------------------------------

BINARY = "early_injury_flag"
BINARY_LABELS = {**LABELS, BINARY: "Early injury in the first 24 h"}


def _binary_context():
    base = _context(exact=False)
    flag = ConceptDescriptor(
        name=BINARY,
        description="early injury status",
        role=VariableRole.OTHER,
        dtype="int64",
        source_concept="early_injury",
        analysis_window="icu_admission[0,24]h",
        analysis_window_role="exposure_definition",
        observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
    )
    specs = [
        spec
        for spec in base.user_preferences.sensitivity_specs
        if str(spec.axis) != "exposure_definition"
    ]
    return base.model_copy(
        update={
            "research_question": (
                "Among adult ICU stays, what proportion have an early injury in the first 24 h, "
                "and how is it associated with in-hospital death after a 24 h landmark?"
            ),
            "primary_exposure": BINARY,
            "variables": [*base.variables, flag],
            "user_preferences": base.user_preferences.model_copy(
                update={"sensitivity_specs": specs}
            ),
        }
    )


def _binary_payload(request, *, level_labels):
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": PLANNER_ROSTER[:2],
        "reader_display_labels": [
            {"key": key, "value": BINARY_LABELS[key]}
            for key in request.required_reader_label_keys
        ]
        + [{"key": key, "value": value} for key, value in level_labels.items()],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    "Compare population, exposure definition, time zero, and estimand "
                    f"with {key} without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Roster follows the host candidates and their timing authority.",
    }


def test_a_binary_exposure_needs_two_distinct_level_labels() -> None:
    request = _request(_binary_context())
    assert request.level_label_keys == [f"{BINARY}=0", f"{BINARY}=1"]

    def validate(level_labels):
        validate_family_plan_spec(
            spec_from_mapping(_binary_payload(request, level_labels=level_labels)), request
        )

    validate({f"{BINARY}=0": "No early injury", f"{BINARY}=1": "Early injury"})
    with pytest.raises(FamilySpecError) as missing:
        validate({f"{BINARY}=0": "No early injury"})
    assert missing.value.reason_code == "family_spec_reader_label_missing"
    with pytest.raises(FamilySpecError) as same:
        validate({f"{BINARY}=0": "Early injury", f"{BINARY}=1": "early  injury"})
    assert same.value.reason_code == "family_spec_level_labels_not_distinct"


def test_the_binary_plan_labels_both_levels_of_its_study_population_distribution() -> None:
    context = _binary_context()
    request = _request(context)
    _llm, result = _run(
        context,
        [
            json.dumps(
                _binary_payload(
                    request,
                    level_labels={
                        f"{BINARY}=0": "No early injury",
                        f"{BINARY}=1": "Early injury",
                    },
                )
            )
        ],
    )
    plan = result.output

    assert [step.step_id for step in plan.steps][:3] == [
        "study_population",
        "exposure_occurrence",
        "cohort_definition",
    ]
    assert plan.display_labels[f"{BINARY}=1"] == "Early injury"


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------


def _plan(context, *, adjustment_set=None):
    request = _request(context)
    _llm, result = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=adjustment_set))]
    )
    return result.output


def test_the_template_reports_the_occurrence_on_the_study_cohort() -> None:
    plan = _plan(_asking())
    steps = {step.step_id: step for step in plan.steps}

    # The pair runs first, so the landmark cohort stays the latest cohort count.
    assert [step.step_id for step in plan.steps][:3] == [
        "study_population",
        "exposure_occurrence",
        "cohort_definition",
    ]
    root = steps["study_population"]
    assert root.method == "host_materialized_locked_cohort"
    assert (root.planned_analysis_role, root.inputs, root.expected_outputs) == (
        "auxiliary",
        [],
        ["cohort:study_population"],
    )
    assert root.cohort_definition_spec is None
    occurrence = steps["exposure_occurrence"]
    assert occurrence.planned_analysis_role == "secondary"
    assert occurrence.method == "descriptive"
    assert [value for value in occurrence.inputs if ":" in value] == ["cohort:study_population"]
    spec = occurrence.exposure_outcome_distribution_spec
    assert (spec.exposure, spec.outcome) == ("injury_stage", "death")
    # A crude contrast beside the plan's own primary estimate is not reported.
    assert spec.risk_difference_contrast is None
    assert occurrence.descriptive_claim is not None
    assert occurrence.descriptive_claim.unresolved_limitations == (
        "post_baseline_exposure_opportunity_unresolved",
    )
    assert step_cohort_population(step=occurrence, plan=plan) == "study_cohort"
    assert "table:exposure_outcome_distribution" in steps["report"].inputs
    selected = plan.design_selection.selected
    assert "how often each" in selected.supports
    assert "among all stays of the study cohort" in selected.reviewable_plan[1]


def test_a_chinese_question_gets_the_clause_in_chinese() -> None:
    context = _asking(suffix="")
    context = context.model_copy(
        update={"research_question": "成年 ICU 入住中，损伤分期的比例是多少？与 24 h landmark 后院内死亡的关联如何？"}
    )

    plan = _plan(context)

    assert "发生比例" in plan.design_selection.selected.reviewable_plan[1]


def test_without_the_question_the_template_is_unchanged() -> None:
    plan = _plan(_post_baseline(_context()))

    assert "study_population" not in {step.step_id for step in plan.steps}
    assert all(step.method != "host_materialized_locked_cohort" for step in plan.steps)
    assert "how often" not in plan.design_selection.selected.supports


# ---------------------------------------------------------------------------
# The signed landmark route and the review
# ---------------------------------------------------------------------------

_AUTHORITY = {
    "schema_version": "easyicu.landmark_categorical_association_runtime_authority/3",
    "authority_kind": "landmark_categorical_association",
    "protocol_content_sha256": "c" * 64,
    "cohort_method": "signed_landmark_analysis_cohort",
    "primary_method": "signed_landmark_categorical_association",
    "plan_intent": "Estimate the adjusted categorical association at the 24-hour landmark.",
    "landmark_spec_id": "landmark_24h_primary",
    "cohort_product": "artifact:analysis_cohort",
    "cohort_flow_product": "table:cohort_flow",
    "primary_product": "table:adjusted_association_estimates",
    "exposure_column": "injury_stage",
    "exposure_kind": "ordinal",
    "exposure_levels": ["0", "1", "2", "3"],
    "exposure_reference_level": "0",
    "primary_contrast_level": "3",
    "outcome_column": "death",
    "event_time_column": "death_time_hours",
    "observation_duration_column": "followup_time_hours",
    "observation_duration_unit": "hours",
    "landmark_hours": 24,
    "exclude_negative_event_times": True,
    "require_alive_at_landmark": True,
    "required_adjustment_columns": [],
    "categorical_adjustment_columns": [],
    "dependence": None,
    "interpretation": "descriptive_prognostic_association_not_causal",
    "association_model_grid": None,
    "plan_bound_adjustment_roster": {
        "authority": "plan_primary_model",
        "admissible_columns": ["age", "sex", "comorbidity_index", "severity_score_24h"],
        "admissible_categorical_columns": [],
        "sealed": False,
    },
}


def _independent_rows(context):
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["analysis_design"].pop("cluster_unit", None)
    constraints["analysis_design"]["variance_estimator"] = "model_based"
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _signed(context):
    """Plan, shape and bind through the host exactly as a fresh run does."""

    context = _independent_rows(context)
    plan = _plan(context, adjustment_set=PLANNER_ROSTER)
    authorities = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=build_current_case_scientific_runtime_authority(_AUTHORITY),
    )
    host = SimpleNamespace(
        _scientific_runtime_authorities=authorities,
        _enable_publication_figure_skill=True,
        _max_total_steps=PipelineConfig(workdir="./unused").max_total_steps,
    )
    findings: list = []
    plan = research_pipeline._shape_fresh_plan(
        pipeline=host, plan=plan, context=context, agent_context=context,
        long_trajectory_bound=False, findings=findings,
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, _ = authorities.bind_plan(plan)
    bound = figure_plan_shaping.apply_runtime_bound_figure_contracts(bound, findings)
    sealed = authorities.seal_for_plan(bound)
    sealed.validate_plan(bound)
    final_plan_shape.validate_final_plan_shape(bound)
    review = build_plan_scientific_review(
        context=context, plan=bound, literature=None,
        figure_strategy=build_article_figure_strategy(context),
        runtime_authority=sealed.current_case,
    )
    return bound, findings, review


def _codes(review) -> set[tuple[str, str]]:
    return {(finding.code, finding.severity) for finding in review.findings}


def test_the_signed_plan_keeps_every_step_and_adds_no_finding() -> None:
    plain, _plain_findings, plain_review = _signed(_post_baseline(_context(exact=False)))
    asked, findings, review = _signed(_asking(_context(exact=False)))

    added = {step.step_id for step in asked.steps} - {step.step_id for step in plain.steps}
    assert {"study_population", "exposure_occurrence"} <= added
    assert len(asked.steps) == len(plain.steps) + 3
    # The heaviest fixture still fits under the default cap: nothing is dropped.
    assert not [item for item in findings if (item.detail or {}).get("plan_truncated")]
    assert _codes(review) == _codes(plain_review)
    coverage = review.facts["requested_estimate_coverage"]
    assert coverage == {
        "schema_version": "easyicu.requested_estimate_coverage/1",
        "exposure_occurrence": {
            "requested_cues": ["proportion"],
            "covering_step_ids": ["exposure_occurrence"],
        },
    }


def test_the_review_blocks_a_plan_that_leaves_the_occurrence_unanswered() -> None:
    context = _asking()
    plan = _plan(context)
    unanswered = plan.model_copy(
        update={
            "steps": [
                step
                for step in plan.steps
                if step.step_id not in {"study_population", "exposure_occurrence"}
            ]
        }
    )

    def review(candidate):
        return build_plan_scientific_review(
            context=context, plan=candidate, literature=None,
            figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
        )

    blocked = review(unanswered)
    finding = next(item for item in blocked.findings if item.code == OCCURRENCE_CODE)
    assert (finding.severity, finding.remediation_route) == ("blocker", "agent_plan_revision")
    assert OCCURRENCE_CODE not in {item.code for item in review(plan).findings}

    # Level counts on the landmark cohort answer a different question.
    occurrence = next(step for step in plan.steps if step.step_id == "exposure_occurrence")
    on_landmark = occurrence.model_copy(
        update={
            "inputs": [
                "artifact:analysis_cohort" if value == "cohort:study_population" else value
                for value in occurrence.inputs
            ]
        }
    )
    landmark_cohort = next(step for step in plan.steps if step.step_id == "cohort_definition")
    signed_cohort = landmark_cohort.model_copy(
        update={"icu_rule_refs": ["scientific_runtime_contract:" + "d" * 64]}
    )
    narrowed = unanswered.model_copy(
        update={
            "steps": [
                signed_cohort if step.step_id == "cohort_definition" else step
                for step in unanswered.steps
            ]
            + [on_landmark]
        }
    )
    assert step_cohort_population(step=on_landmark, plan=narrowed) == "restricted"
    assert OCCURRENCE_CODE in {item.code for item in review(narrowed).findings}


def test_a_descriptive_plan_answers_the_occurrence_with_its_own_distribution() -> None:
    context = _descriptive_context()
    request = _request(context, cohort_mode="all_input_rows")
    labels = {
        "phenotype_flag": "Phenotype in the first 24 h",
        "death": "In-hospital death",
        "age": "Age (years)",
        "sex": "Patient sex",
        "score_first": "Chronic disease score",
        "phenotype_flag=0": "Phenotype absent",
        "phenotype_flag=1": "Phenotype present",
    }
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "baseline_variables": ["age"],
        "reader_display_labels": [
            {"key": key, "value": labels[key]}
            for key in [*request.required_reader_label_keys, *request.level_label_keys]
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this description with {key} on population, exposure "
                    "definition, time zero, and estimand without claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Descriptive family: baseline variables from host-timed candidates.",
    }
    _llm, result = _run(
        context, [json.dumps(payload)], required_primary_cohort_selection_mode="all_input_rows"
    )

    review = build_plan_scientific_review(
        context=context, plan=result.output, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )

    assert review.facts["requested_estimate_coverage"]["exposure_occurrence"] == {
        "requested_cues": ["proportion"],
        "covering_step_ids": ["exposure_outcome_distribution"],
    }
    assert OCCURRENCE_CODE not in {item.code for item in review.findings}


def test_an_exposure_without_closed_levels_has_no_occurrence_to_report() -> None:
    context = _asking()
    plan = _plan(context)
    continuous = context.model_copy(update={"primary_exposure": "severity_score_24h"})

    assert requested_exposure_occurrence_cues(continuous) == ("proportion",)
    assert requested_exposure_occurrence(continuous, plan) == ()
    assert requested_exposure_occurrence(context, plan) == ("proportion",)


def test_a_question_that_does_not_ask_is_not_reviewed_for_it() -> None:
    context = _post_baseline(_context())
    review = build_plan_scientific_review(
        context=context, plan=_plan(context), literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )

    assert review.facts["requested_estimate_coverage"]["exposure_occurrence"] == {
        "requested_cues": [],
        "covering_step_ids": [],
    }
    assert OCCURRENCE_CODE not in {item.code for item in review.findings}
