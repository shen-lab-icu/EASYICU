"""A static prediction predicts from what is known by its prediction time.

The prediction template predicts at the end of the host-bound feature window,
keeps the stays still in the ICU then, and offers as predictors only values
the host can place before then.  A plan from another route had none of this:
a predictor summarized over a later window, or a value of the whole stay,
entered the model and inflated its discrimination without any error.  The
review now checks each static prediction against the same facts
(``planning.prediction_timing``), and an outline the Progressive Planner
composes, which states no prediction time, stops at a static prediction
primary instead of compiling one, also when it is continued from a
checkpoint.  A continued template plan never reaches that route: the resume
checks refuse it before any Provider call.  Synthetic contexts only.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.outline_action_rules import (
    STATIC_PREDICTION_TEMPLATE_REQUIRED,
    static_prediction_outline_stop,
)
from easyicu.research_agent.planning.prediction_timing import static_prediction_timing_facts
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    prediction_timing_findings,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)
from .progressive_planner_fixtures import _context, _cox_outline_payload

CODE = "PREDICTION_PREDICTOR_TIMING_UNPROVEN"
RISK_SET = "PREDICTION_RISK_SET_NOT_KEPT"
PRIMARY = "prediction.discrimination_calibration"
FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]
_DEPENDENCIES = {
    "cohort_file_sha256": "b" * 64,
    "llm_signature": "codex:gpt-test",
    "prompt_version": "test-v1",
}


def _template_plan(context: ResearchContext):
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=None,
    )
    return result.output


def _with_inclusion(plan, inclusion):
    return plan.model_copy(
        update={"cohort": dataclasses.replace(plan.cohort, inclusion=tuple(inclusion))}
    )


def _with_predictors(plan, *names: str):
    """The plan's static prediction primary with ``names`` added to its model columns."""

    steps = [
        step.model_copy(update={"inputs": [*names, *step.inputs]})
        if step.scientific_action_id == PRIMARY
        else step
        for step in plan.steps
    ]
    return plan.model_copy(update={"steps": steps})


def _with_variables(context: ResearchContext, *variables: ConceptDescriptor) -> ResearchContext:
    return context.model_copy(update={"variables": [*context.variables, *variables]})


def test_a_template_prediction_predicts_from_what_is_known_by_its_prediction_time() -> None:
    context = _prediction_context()
    plan = _template_plan(context)

    [facts] = static_prediction_timing_facts(context, plan)

    assert facts.step_id == "primary_performance"
    assert (facts.prediction_time_hours, facts.stays_kept_after_hours) == (24.0, 24.0)
    assert facts.unproven_predictors == ()
    assert facts.proven
    assert prediction_timing_findings(context, plan) == []


def test_a_cohort_that_keeps_stays_gone_before_the_prediction_is_refused() -> None:
    context = _prediction_context()
    plan = _template_plan(context)
    [risk_set] = plan.cohort.inclusion

    for inclusion in ([], [dataclasses.replace(risk_set, value=0.25)]):
        [finding] = prediction_timing_findings(context, _with_inclusion(plan, inclusion))

        assert (finding.code, finding.severity) == (RISK_SET, "blocker")
        assert finding.remediation_route == "agent_plan_revision"
        assert "does not keep only the stays still in the ICU after 24 h" in finding.message
        assert "these predictors" not in finding.message


def test_the_plan_review_states_the_refusal() -> None:
    context = _prediction_context()
    plan = _template_plan(context)

    def codes(reviewed) -> dict[str, str]:
        return {
            finding.code: finding.severity
            for finding in build_plan_scientific_review(context=context, plan=reviewed).findings
        }

    assert not {CODE, RISK_SET} & set(codes(plan))
    assert codes(_with_inclusion(plan, []))[RISK_SET] == "blocker"


def test_a_predictor_the_host_cannot_place_before_the_prediction_is_refused() -> None:
    context = _with_variables(
        _prediction_context(),
        # Summarized over the first 48 h: ends after the 24 h prediction.
        ConceptDescriptor(
            name="creatinine_max", role=VariableRole.LAB, dtype="float64",
            source_concept="crea", analysis_window="icu_admission[0,48]h",
        ),
        # One value for the whole stay: no window ends it, so the outer
        # window it would once have inherited does not place it either.
        ConceptDescriptor(
            name="apache_iv", role=VariableRole.COMPOSITE_SCORE, dtype="float64",
            source_concept="apache_iv",
        ),
    )
    plan = _with_predictors(_template_plan(_prediction_context()), "creatinine_max", "apache_iv")

    [facts] = static_prediction_timing_facts(context, plan)
    [finding] = prediction_timing_findings(context, plan)

    assert facts.unproven_predictors == ("creatinine_max", "apache_iv")
    assert facts.risk_set_kept
    assert finding.code == CODE
    assert "predicts at 24 h after ICU admission" in finding.message
    assert "creatinine_max, apache_iv" in finding.message
    assert "does not keep only" not in finding.message


def test_each_cause_has_its_own_code() -> None:
    context = _with_variables(
        _prediction_context(),
        ConceptDescriptor(
            name="apache_iv", role=VariableRole.COMPOSITE_SCORE, dtype="float64",
            source_concept="apache_iv",
        ),
    )
    plan = _with_inclusion(_with_predictors(_template_plan(_prediction_context()), "apache_iv"), [])

    # A reader's sentence for one cause is never shown for the other.
    assert [finding.code for finding in prediction_timing_findings(context, plan)] == [
        RISK_SET, CODE,
    ]


def test_without_a_bound_window_only_baseline_demographics_are_known() -> None:
    windowed = _prediction_context()
    plan = _template_plan(windowed)
    # The same rows with no host-bound feature window: the model then
    # predicts at ICU admission, before any window-summarized measurement.
    unbound = windowed.model_copy(
        update={
            "time_windows": [],
            "user_preferences": windowed.user_preferences.model_copy(
                update={"data_constraints": "{}"}
            ),
        }
    )

    [facts] = static_prediction_timing_facts(unbound, plan)
    [finding] = prediction_timing_findings(unbound, plan)

    assert facts.prediction_time_hours is None and facts.risk_set_kept
    assert facts.unproven_predictors == ("hr_max", "lactate_max", "map_min")
    assert "predicts at ICU admission" in finding.message


def _outline_step(step_id: str, action_id: str, role: str = "primary") -> ProgressiveOutlineStep:
    return ProgressiveOutlineStep(
        step_id=step_id,
        module_id="custom_analysis",
        planned_analysis_role=role,
        objective="Run the selected scientific action.",
        depends_on=[],
        variable_names=["outcome_flag"],
        scientific_action_id=action_id,
    )


def test_a_composed_outline_stops_at_a_static_prediction_primary() -> None:
    selects = ProgressivePlanOutline.model_construct(
        analysis_type="prediction_model",
        steps=[
            _outline_step("calibration", "prediction.calibration_metrics", role="secondary"),
            _outline_step("model", PRIMARY),
        ],
    )
    stop = static_prediction_outline_stop(selects)

    assert stop.reason_code == STATIC_PREDICTION_TEMPLATE_REQUIRED
    assert stop.step_id == "model" and stop.step_index == 1
    assert "only the prediction family template" in str(stop)
    other = ProgressivePlanOutline.model_construct(
        analysis_type="prediction_model",
        steps=[_outline_step("model", "prediction.dynamic_prediction")],
    )
    assert static_prediction_outline_stop(other) is None


def _prediction_outline() -> dict:
    outline = _cox_outline_payload()
    outline["analysis_type"] = "prediction_model"
    for candidate in outline["design_selection"]["candidates"]:
        candidate["analysis_type"] = "prediction_model"
    primary = next(step for step in outline["steps"] if step["step_id"] == "05_primary")
    primary["scientific_action_id"] = PRIMARY
    return outline


def _composed(**kwargs):
    context = _context().model_copy(
        update={"research_question": "Predict outcome_flag from the first-day measurements."}
    )
    llm = ScriptedMockLLMClient([json.dumps(_prediction_outline())] * 3)
    try:
        result = ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy="progressive_v2", **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - the test reads which stop it was
        return llm, exc
    return llm, result


def test_the_progressive_planner_stops_at_its_first_such_outline() -> None:
    llm, stopped = _composed()

    assert isinstance(stopped, ProgressivePlanCompileError)
    assert stopped.reason_code == STATIC_PREDICTION_TEMPLATE_REQUIRED
    # The first outline was the only request: no retry, foundation or step call.
    assert len(llm.calls) == 1


def test_a_design_canary_is_not_stopped_for_it() -> None:
    _llm, result = _composed(stop_after_outline=True)

    assert getattr(result, "reason_code", None) != STATIC_PREDICTION_TEMPLATE_REQUIRED


def _continued(context: ResearchContext, checkpoint, **kwargs) -> ProgressivePlanCompileError:
    """The stop a continuation from ``checkpoint`` meets; it calls no Provider."""

    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            resume_checkpoint=checkpoint,
            resume_dependency_context=_DEPENDENCIES,
            **kwargs,
        )
    assert llm.calls == []
    return stopped.value


def test_a_continued_template_prediction_never_reaches_the_composed_route() -> None:
    context = _prediction_context()
    checkpoints = []
    _run(
        context,
        [json.dumps(_prediction_payload(_request(context, cohort_mode=None), features=FEATURES))],
        required_primary_cohort_selection_mode=None,
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=_DEPENDENCIES,
    )
    assert checkpoints[-1].prompt_metrics["foundation_cohort_owner"] == "family_template"

    for checkpoint in (checkpoints[0], checkpoints[-1]):
        stopped = _continued(
            context,
            checkpoint,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            direct_comparator_literature_keys=DIRECT_COMPARATORS,
            comparison_literature_keys=DIRECT_COMPARATORS,
            enforce_article_contract=True,
            article_contract_context=context,
            planning_contract_context="",
            required_primary_cohort_selection_mode=None,
        )

        # The resume checks refuse it: no capability gap, no composed outline.
        assert stopped.reason_code.startswith("progressive_resume_")


def test_a_composed_prediction_outline_is_not_continued_into_a_plan() -> None:
    context = _context().model_copy(
        update={"research_question": "Predict outcome_flag from the first-day measurements."}
    )
    checkpoints = []
    # A design canary keeps its composed outline as a checkpoint, as every
    # outline did before the stop; continuing it would compile that outline.
    ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(_prediction_outline())])).run_attempt(
        context,
        planner_strategy="progressive_v2",
        checkpoint_callback=checkpoints.append,
        resume_dependency_context=_DEPENDENCIES,
        stop_after_outline=True,
    )
    [outline_checkpoint] = checkpoints

    stopped = _continued(context, outline_checkpoint, planner_strategy="progressive_v2")

    assert stopped.reason_code == STATIC_PREDICTION_TEMPLATE_REQUIRED
