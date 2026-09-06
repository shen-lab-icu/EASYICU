"""Natural-question routing and first-pass planner contract regressions."""

import json
import re

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    step_materialization_shape_contract,
)
from easyicu.research_agent.planning.analysis_types import infer_analysis_type
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveModelTermIntent,
    ProgressiveOutlineStep,
)
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext


@pytest.mark.parametrize(
    ("question", "family"),
    [
        ("Describe mortality by infection status.", "descriptive_epidemiology"),
        ("Summarize outcome prevalence by exposure group.", "descriptive_epidemiology"),
        ("描述器官功能异常的患病率及两组患者的院内死亡情况。", "descriptive_epidemiology"),
        ("描述不同暴露组的结局分布。", "descriptive_epidemiology"),
        ("Estimate prevalence and its association with mortality.", "association_study"),
        ("描述患病率并估计暴露与院内死亡的关联。", "association_study"),
        ("Predict mortality and report calibration and AUROC.", "prediction_model"),
        ("Estimate the causal treatment effect using IPTW.", "causal_inference"),
    ],
)
def test_question_intent_precedes_exposure_outcome_presence(question, family):
    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="test", database="synthetic", n_stays=0),
        variables=[],
        primary_exposure="exposure",
        target_outcome="outcome",
    )

    assert infer_analysis_type(context).key == family
    assert candidate_analysis_types(context)[0] == family


def test_first_pass_model_term_example_matches_typed_contract():
    prompt = step_materialization_shape_contract(
        outline_step=ProgressiveOutlineStep(
            step_id="primary_model",
            module_id="adjusted_association",
            planned_analysis_role="primary",
            objective="Estimate the requested adjusted association.",
            depends_on=[],
            variable_names=["exposure", "outcome"],
            scientific_action_id=None,
        ),
        outline_step_sha256="a" * 64,
    )
    match = re.search(r"model_terms items are exactly (\{[^{}]+\})", prompt)
    assert match is not None
    example = json.loads(match.group(1))
    assert set(example) == set(ProgressiveModelTermIntent.model_fields)
    example.update(name="age", role="covariate", coding="continuous")
    example["clinical_rationale"] = (
        "Age precedes exposure and is a plausible common cause of exposure and outcome."
    )
    assert ProgressiveModelTermIntent.model_validate(example).name == "age"
    example.update(role="exposure", clinical_rationale=None)
    assert ProgressiveModelTermIntent.model_validate(example).role == "exposure"


def test_outline_data_card_retains_physical_window_and_its_role():
    variable = ConceptDescriptor(
        name="exposure_max", role="other", dtype="int64",
        analysis_window="icu_admission[0,24]h",
        analysis_window_role="outer_observation_window",
    )
    context = ResearchContext(
        research_question="Describe the observed exposure distribution.",
        cohort=CohortDescriptor(cohort_name="test", database="synthetic", n_stays=0),
        variables=[variable],
        primary_exposure=variable.name,
    )
    card, = ProgressivePlannerAgent._retrieved_data_cards(context, [variable.name])
    for field in ("analysis_window", "analysis_window_role"):
        assert card[field] == getattr(variable, field)
