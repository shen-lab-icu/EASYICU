"""The vocabulary was in the context and the contract threw it away.

``trajectory_role_code_contract`` opened with ``del context  # Schema is
selected from typed products, never task prose.`` The principle is right about
SCHEMA. It is not about VOCABULARY: which concept ids the bound trajectory
actually materialized is typed input metadata, not task prose.

verify42 is the cost. The bound table held sofa2, sofa2_resp, sofa2_coag,
sofa2_liver, sofa2_cardio, sofa2_cns, sofa2_renal and lact -- all eight present,
``unavailable_concepts`` empty -- and the generated script queried ``sofa_resp``,
``sofa_coag``, ``sofa_liver``: every SOFA name one character off, assembled from
the research question's phrase "SOFA components and lactate". It then raised
"Required lactate concept family is absent from the trajectory".

Telling the Coder to read the ``concept`` column (a6cdab9, d0f887a) helped --
verify43's script began using ``sofa2`` -- but left it assembling names. Naming
them in the contract removes the guess.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import AnalysisStep, CohortDescriptor
from easyicu.research_agent.trajectory.plan_contract import (
    trajectory_role_code_contract,
)


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="03_build_representation",
        intent="Build the fixed-window representation.",
        method="fixed_anchor_hourly_trajectory_representation",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=[
            "artifact:trajectory_representation",
            "table:trajectory_membership",
            "manifest:trajectory_representation_schema",
            "manifest:trajectory_window_manifest",
        ],
    )


class _Trajectory:
    def __init__(self, concepts):
        self.materialized_concepts = list(concepts)


class _Inputs:
    def __init__(self, trajectory):
        self.trajectory = trajectory


class _Context:
    """A stand-in for ResearchContextV2, which IS a ResearchContext subclass."""

    def __init__(self, concepts=None):
        self.materialized_inputs = (
            _Inputs(_Trajectory(concepts)) if concepts is not None else None
        )


_REAL = (
    "sofa2",
    "sofa2_resp",
    "sofa2_coag",
    "sofa2_liver",
    "sofa2_cardio",
    "sofa2_cns",
    "sofa2_renal",
    "lact",
)


def test_the_contract_lists_the_exact_concept_ids():
    text = trajectory_role_code_contract(context=_Context(_REAL), step=_step())

    assert "BOUND TRAJECTORY VOCABULARY" in text
    for concept in _REAL:
        assert concept in text, concept
    # The two the agent got wrong must be answerable from this text alone.
    assert "sofa2_resp" in text
    assert "lact" in text


def test_it_forbids_deriving_a_concept_id_from_the_question():
    text = trajectory_role_code_contract(context=_Context(_REAL), step=_step())

    assert "by exact string" in text
    assert "do not\n" in text or "do not " in text
    assert "research question's wording" in text
    # And it forbids the exact wrong conclusion verify42 reached.
    assert "without checking this list first" in text


def test_a_context_without_a_bound_trajectory_changes_nothing():
    """Wide-column and non-trajectory runs must be byte-identical."""

    without = trajectory_role_code_contract(context=_Context(None), step=_step())
    assert "BOUND TRAJECTORY VOCABULARY" not in without

    # A context object that has no such attribute at all must not raise.
    class _Bare:
        pass

    bare = trajectory_role_code_contract(context=_Bare(), step=_step())
    assert "BOUND TRAJECTORY VOCABULARY" not in bare
    assert bare == without


def test_an_empty_concept_list_says_nothing():
    text = trajectory_role_code_contract(context=_Context(()), step=_step())
    assert "BOUND TRAJECTORY VOCABULARY" not in text


def test_the_vocabulary_is_not_offered_to_a_step_that_has_no_trajectory_role():
    """It attaches to the roles that read the table, not to every step."""

    unrelated = AnalysisStep(
        step_id="09_outcome_figure",
        intent="Render the outcome figure.",
        method="outcome_by_cluster_figure",
        inputs=["table:outcome_by_cluster"],
        expected_outputs=["figure:outcome_by_cluster"],
    )
    text = trajectory_role_code_contract(context=_Context(_REAL), step=unrelated)
    assert "BOUND TRAJECTORY VOCABULARY" not in text
