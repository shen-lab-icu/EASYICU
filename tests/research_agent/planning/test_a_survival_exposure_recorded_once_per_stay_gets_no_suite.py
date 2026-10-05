"""A survival exposure recorded once per stay gets no landmark suite.

The landmark suite classifies each stay's exposure as prevalent or incident by
the exposure's first record as present.  A concept its owner records once per stay
(sex, age, admission type) has no recorded time, so materialization emits no
timing companion for it.  The host still proposed the suite for any two-level
exposure: planning, review and the sealed replan passed, and the run failed
only at launch, with its onset column missing.  The proposal now refuses such
an exposure, so the survival family stops before the Provider, and the web's
sealing compile refuses it with its own reason.

Synthetic study only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.concept_availability import concept_records_one_value_per_stay
from easyicu.research_agent.planning.family_spec.contract import LANDMARK_SURVIVAL_FAMILY_ID
from easyicu.research_agent.planning.family_spec.request import (
    family_template_id_for_context,
    proposed_survival_suite_coordinates,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.support.survival_proposal import BINARY, survival_context


@pytest.mark.parametrize(
    ("concept", "stay_level"),
    [
        ("sex", True), ("age", True), ("adm", True), ("weight", True),
        ("rrt", False), ("mech_vent", False), ("vent_ind", False), ("lact", False),
        # Outside the dictionary's authority: not declared, so not refused here.
        ("a_local_flag_no_owner_declares", False),
    ],
)
def test_the_concept_owner_says_which_concepts_have_no_record_time(concept, stay_level):
    assert concept_records_one_value_per_stay(concept) is stay_level


def _template(context):
    return family_template_id_for_context(context, analysis_types=("survival",))


def test_a_time_stamped_exposure_keeps_its_proposed_suite():
    context = survival_context()

    proposed = proposed_survival_suite_coordinates(context)

    assert proposed is not None and proposed.exposure_onset_column == "rrt_onset_time"
    assert _template(context) == LANDMARK_SURVIVAL_FAMILY_ID


@pytest.mark.parametrize("derived", [False, True])
def test_a_stay_level_exposure_gets_no_suite(derived):
    context = survival_context(primary_exposure="sex")
    if derived:
        older = ConceptDescriptor(
            name="age_65_or_older", description="age 65 years or older", role=VariableRole.DEMOGRAPHIC,
            dtype="float64", observed_domain=BINARY, source_concept="age",
        )
        context = context.model_copy(update={
            "variables": [*context.variables, older], "primary_exposure": "age_65_or_older",
        })

    assert proposed_survival_suite_coordinates(context) is None
    assert _template(context) is None
