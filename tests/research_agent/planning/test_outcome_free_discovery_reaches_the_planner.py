"""A phenotype-discovery question is planned without a target outcome.

Discovery asks how patients group, not how a predictor relates to an outcome.
Its ordinary wordings must reach the clustering family, and the pre-plan
hypothesis blueprint must not stop a question for lacking an outcome that its
family never has.  An association question without its outcome still stops.
"""

from __future__ import annotations

from typing import Optional

import pytest

from easyicu.research_agent.literature import (
    HypothesisBlueprintAgent,
    LiteratureBundle,
)
from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type,
    strong_trajectory_clustering_framing,
)
from easyicu.research_agent.planning.study_design import infer_study_design_family
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    HypothesisBlueprint,
    ResearchContext,
)


def _context(question: str, *, outcome: Optional[str] = None) -> ResearchContext:
    variables = [
        ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
        ConceptDescriptor(name="age", role="demographic", dtype="float64"),
        ConceptDescriptor(name="lact", role="lab", dtype="float64"),
        ConceptDescriptor(name="map", role="vital", dtype="float64"),
        ConceptDescriptor(name="sofa2_renal", role="ordinal_score", dtype="int64"),
    ]
    if outcome:
        variables.append(ConceptDescriptor(name=outcome, role="outcome", dtype="int64"))
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic_icu",
            database="miiv",
            n_patients=400,
            n_stays=400,
        ),
        variables=variables,
        target_outcome=outcome,
    )


def _blueprint(context: ResearchContext) -> HypothesisBlueprint:
    return HypothesisBlueprintAgent().run(
        context=context,
        literature=LiteratureBundle(
            research_question=context.research_question, citations=[]
        ),
    )


@pytest.mark.parametrize(
    "question",
    [
        "In adults with ARDS, do early oxygenation trajectories cluster into "
        "distinct subgroups?",
        "Do hemodynamic trajectories in the first 48 hours after ICU admission "
        "form reproducible sub-phenotypes?",
        "Can we identify latent classes of organ dysfunction trajectories after "
        "cardiac surgery?",
        "Can distinct sub-phenotypes of acute kidney injury be derived from routine "
        "laboratory profiles?",
        "Do temperature curves in the first 72 hours define distinct endotypes?",
        "成人脓毒症患者入科后前48小时的器官功能轨迹能否聚成可重复的亚型？",
        "心脏术后患者的乳酸曲线是否聚为不同的类别？",
    ],
)
def test_ordinary_discovery_wording_chooses_the_clustering_family(question):
    context = _context(question)

    assert strong_trajectory_clustering_framing(question)
    assert infer_analysis_type(context).key == "trajectory_clustering"
    assert infer_study_design_family(context) == "phenotyping"


@pytest.mark.parametrize(
    "question",
    [
        "Do patients in the previously published hyperinflammatory sub-phenotype "
        "have higher mortality?",
        "Compare mortality across established endotypes of sepsis.",
        "Identify the previously published sub-phenotypes in this cohort and "
        "compare their mortality.",
        "Can the established endotypes of sepsis be identified in this cohort?",
        "Identify patients with the hyperinflammatory phenotype and compare "
        "28-day mortality.",
        "Does the lactate trajectory define the exposure group for a mortality model?",
        "Use GEE to account for clustering among patients within hospitals when "
        "estimating the mortality association.",
        "患者分为高乳酸组和低乳酸组，比较死亡率。",
        "不进行轨迹聚成亚型的分析，只描述各指标分布。",
    ],
)
def test_groups_the_question_does_not_ask_to_discover_stay_out(question):
    assert not strong_trajectory_clustering_framing(question)
    assert infer_analysis_type(_context(question)).key != "trajectory_clustering"


def test_an_outcome_free_discovery_question_is_ready_to_plan():
    blueprint = _blueprint(
        _context(
            "Do first-day lactate and blood-pressure trajectories cluster into "
            "distinct subgroups of ICU patients?"
        )
    )

    assert blueprint.feasibility_status == "ready"
    assert blueprint.missing_variables == []
    assert "phenotype discovery" in blueprint.hypothesis
    assert "associated with" not in blueprint.hypothesis
    assert "target outcome must be specified" not in blueprint.hypothesis
    assert any(
        "clustering and stability checks" in step for step in blueprint.stepwise_plan
    )
    assert not any("association" in step for step in blueprint.stepwise_plan)
    # The audit anchor and the database check are as before: a materialised
    # analysis variable, available in the selected database.
    assert blueprint.concept_dependencies
    assert blueprint.cross_database_feasibility["miiv"] != "blocked"


def test_an_outcome_free_descriptive_question_is_ready_to_plan():
    blueprint = _blueprint(
        _context(
            "Describe the baseline characteristics and first-day lactate "
            "distribution of adults admitted with sepsis."
        )
    )

    assert blueprint.feasibility_status == "ready"
    assert blueprint.missing_variables == []
    assert "descriptive epidemiology" in blueprint.hypothesis


def test_an_association_question_without_its_outcome_still_stops():
    blueprint = _blueprint(
        _context("Is peak lactate associated with in-hospital mortality?")
    )

    assert blueprint.feasibility_status == "blocked"
    assert blueprint.missing_variables == ["target_outcome"]


def test_a_discovery_outcome_is_described_without_an_association():
    blueprint = _blueprint(
        _context(
            "Do first-day lactate and blood-pressure trajectories cluster into "
            "distinct subgroups, and how does mortality differ between them?",
            outcome="hospital_death",
        )
    )

    assert blueprint.feasibility_status == "ready"
    assert "hospital_death is summarised descriptively" in blueprint.hypothesis
    assert "associated with" not in blueprint.hypothesis
    assert not any("association" in step for step in blueprint.stepwise_plan)
