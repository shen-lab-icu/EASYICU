"""A cohort the study states in words reaches the Planner as a population to apply.

The Planner reads the context's inclusion and exclusion criteria as contracts
already applied to its input rows. The Web caller used to file the study's own
wording there -- ``label``, ``review`` and ``exclusion_statement`` -- although
nothing executes prose: Data Extraction executes the typed fields, and the host
the first-ICU-stay restriction. A study whose cohort read "Adults with septic
shock" therefore planned on an all-ICU export with every input row kept, under
a cohort named for the population it never selected. The criteria now declare
only what something applies; the wording still reaches the Planner, verbatim,
in ``data_constraints.cohort``.
"""

from __future__ import annotations

import json

import pandas as pd

from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
)
from easyicu.webserver.agent_pipeline_runs import (
    _exclusion_criteria,
    _inclusion_criteria,
    _research_user_preferences,
)

_WORDING = {
    "label": "Adults with condition-x",
    "review": "adult stays with condition-x; eligibility proposed by the plan",
    "exclusion_statement": "stays that ended before the 24 h landmark",
}


def test_the_wording_of_a_cohort_is_not_declared_as_applied() -> None:
    for wording in (
        {"label": _WORDING["label"]},
        {"review": _WORDING["review"]},
        {"exclusion_statement": _WORDING["exclusion_statement"]},
        dict(_WORDING),
    ):
        study = {"cohort": wording}
        assert _inclusion_criteria(study) == [], wording
        assert _exclusion_criteria(study) == [], wording


def test_the_typed_fields_are_declared_beside_the_wording() -> None:
    study = {
        "cohort": {
            **_WORDING,
            "age_min": 18,
            "age_max": 80,
            "min_icu_los_hours": 24,
            "exclude_readmissions": True,
            "icd_enabled": True,
            "icd_include": "A41",
            "icd_exclude": "T20-T32",
        }
    }

    inclusion = _inclusion_criteria(study)
    exclusion = _exclusion_criteria(study)

    assert inclusion == [
        "age range: 18 to 80",
        "minimum ICU length of stay: 24 hours",
        "include diagnoses: A41",
    ]
    assert len(exclusion) == 2
    assert "keeps only the first ICU stay per patient" in exclusion[0]
    assert exclusion[1] == "exclude diagnoses: T20-T32"
    declared = " | ".join([*inclusion, *exclusion])
    for text in _WORDING.values():
        assert text not in declared


def test_the_wording_reaches_the_planner_as_the_study_states_it() -> None:
    study = {"cohort": {**_WORDING, "age_min": 18}}

    stated = json.loads(_research_user_preferences(study)["data_constraints"])

    for field, text in _WORDING.items():
        assert stated["cohort"][field] == text
    assert stated["cohort"]["age_min"] == 18


def test_the_planner_sees_no_contract_for_a_population_stated_only_in_words() -> None:
    """What the foundation prompt shows: no applied contract, the wording intact."""

    study = {"cohort": dict(_WORDING)}
    context = build_research_context(
        research_question="Among adults with condition-x, do early trajectories form classes?",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "age": [34.0, 61.0, 15.0]}),
        cohort_name="synthetic",
        database="synthetic",
        inclusion_criteria=_inclusion_criteria(study),
        exclusion_criteria=_exclusion_criteria(study),
        user_preferences=_research_user_preferences(study),
    )

    payload = outbound_safe_context_payload(context)

    assert "inclusion_contract" not in payload["cohort"]
    assert "exclusion_contract" not in payload["cohort"]
    stated = json.loads(payload["study_preferences"]["data_constraints"])
    assert stated["cohort"] == _WORDING


def test_typed_fields_still_reach_the_planner_as_applied_contracts() -> None:
    study = {"cohort": {**_WORDING, "age_min": 18}}
    context = build_research_context(
        research_question="Among adults with condition-x, do early trajectories form classes?",
        cohort=pd.DataFrame({"stay_id": [1, 2], "age": [34.0, 61.0]}),
        cohort_name="synthetic",
        database="synthetic",
        inclusion_criteria=_inclusion_criteria(study),
        exclusion_criteria=_exclusion_criteria(study),
        user_preferences=_research_user_preferences(study),
    )

    cohort = outbound_safe_context_payload(context)["cohort"]

    assert cohort["inclusion_contract"] == ["age range: 18 to *"]
    assert "exclusion_contract" not in cohort


def test_the_foundation_names_the_study_wording_as_a_population_to_apply() -> None:
    free = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)

    assert "only the inclusion and exclusion contracts shown there are already applied" in free
    assert (
        "the study's own cohort wording (the cohort in "
        "study_preferences.data_constraints: its label, review and "
        "exclusion_statement) names"
    ) in free
    assert "in population_criteria, in the words that state it" in free
    assert "keep population_criteria empty when the study includes every input row" in free
