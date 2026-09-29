"""Agents that choose concepts or plan steps learn what the host models longitudinally.

The signed fixed-window trajectory owner counts each stay's eligible windows on
SOFA-2 components, which carry per-window observed/available receipts.  Concept
selection and the Planner are told that rule and which offered concepts meet
it; choosing question-faithful concepts stays theirs.  Synthetic catalogs only.
"""

from __future__ import annotations

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.acquisition.foundation import DataFoundationAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    outline_shape_contract,
)
from easyicu.research_agent.contracts.trajectory_design import (
    ELIGIBILITY_COORDINATE_PREFIX,
    TRAJECTORY_OUTCOME_DESCRIPTION_RULE,
    TRAJECTORY_OWNER_PLANNER_RULE,
    executable_trajectory_coordinates,
    longitudinal_capability_note,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.scores.sofa2_aggregate import SOFA2_COMPONENT_NAMES

#: Concepts that share the SOFA-2 prefix but carry no per-window receipt.
_PREFIXED_WITHOUT_RECEIPTS = (
    "sofa2",
    "sofa2_cns_ascertainment",
    "sofa2_cns_delirium_tx_ascertainment",
    "sofa2_cns_proxy_sensitivity",
)
_OTHER_OFFERED = (
    "sofa_resp",
    "sofa_coag",
    "sofa_liver",
    "sofa_cardio",
    "sofa_cns",
    "sofa_renal",
    "lact",
    "map",
    "age",
    "death",
)


def _selection_prompt(*concept_ids: str) -> str:
    llm = ScriptedMockLLMClient(
        ['{"selected_concepts": ["map"], "inclusion_exclusion": [], "rationale": "r"}']
    )
    DataFoundationAgent(llm).select_concepts(
        question="How do vasopressor-dependence trajectories differ between classes?",
        catalog=AvailableCatalog(
            source="mem",
            concepts=[CatalogConcept(concept_id=concept) for concept in concept_ids],
        ),
    )
    ((messages, _kwargs),) = llm.calls
    return messages[-1].content


def _listed(prompt: str) -> list[str]:
    start = prompt.index("LONGITUDINAL CAPABILITY")
    block = prompt[start : prompt.index("\n\nReturn JSON", start)]
    return block.split("at least one of: ", 1)[1].split(". ", 1)[0].split(", ")


def test_selection_is_told_which_offered_concepts_the_owner_counts():
    prompt = _selection_prompt(
        *_OTHER_OFFERED, *_PREFIXED_WITHOUT_RECEIPTS, *SOFA2_COMPONENT_NAMES
    )

    assert _listed(prompt) == sorted(SOFA2_COMPONENT_NAMES)


def test_a_source_without_a_domain_lists_only_what_it_offers():
    offered = [name for name in SOFA2_COMPONENT_NAMES if name != "sofa2_cns"]

    assert _listed(_selection_prompt(*_OTHER_OFFERED, *offered)) == sorted(offered)


def test_no_receipt_bearing_component_means_no_capability_note():
    prompt = _selection_prompt(*_OTHER_OFFERED, *_PREFIXED_WITHOUT_RECEIPTS)

    assert "LONGITUDINAL CAPABILITY" not in prompt
    assert longitudinal_capability_note(_OTHER_OFFERED) == ""


def test_every_listed_concept_meets_the_owner_rule_and_nothing_else_offered_does():
    listed = _listed(_selection_prompt(*_OTHER_OFFERED, *SOFA2_COMPONENT_NAMES))

    assert all(executable_trajectory_coordinates([name, "lact"]) for name in listed)
    assert not executable_trajectory_coordinates(
        [name for name in _OTHER_OFFERED if name not in {"age", "death"}]
    )


def test_the_note_leaves_the_version_to_the_question():
    prompt = _selection_prompt(*_OTHER_OFFERED, *SOFA2_COMPONENT_NAMES)

    assert "a host fact, not a recommendation" in prompt
    assert "for a question about trajectories over ICU time" in prompt
    assert "If such a question names a score without its version" in prompt
    assert "select one version and say which, and why, in the rationale" in prompt
    assert "Keep a version the question names, even when the owner cannot model it" in prompt


def test_the_planner_reads_the_same_rule():
    text = outline_shape_contract(
        analysis_types=["trajectory_clustering"],
        module_ids_by_analysis_type={"trajectory_clustering": ["custom_analysis"]},
    )

    assert TRAJECTORY_OWNER_PLANNER_RULE in text
    assert repr(ELIGIBILITY_COORDINATE_PREFIX) in TRAJECTORY_OWNER_PLANNER_RULE
    assert "no outcome or one-value-per-stay variable" in TRAJECTORY_OWNER_PLANNER_RULE
    # Where the outcomes go: the suite describes them, so they stay out of a
    # primary the host can replace; only a primary it cannot replace
    # describes them in its own products.
    assert TRAJECTORY_OUTCOME_DESCRIPTION_RULE in TRAJECTORY_OWNER_PLANNER_RULE
    assert "describes the requested outcomes on its frozen" in TRAJECTORY_OUTCOME_DESCRIPTION_RULE
    assert (
        "Only a primary whose coordinates cannot meet the rule describes requested "
        "outcomes in its own characterization products" in TRAJECTORY_OUTCOME_DESCRIPTION_RULE
    )
