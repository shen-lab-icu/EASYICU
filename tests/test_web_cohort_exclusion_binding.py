"""An exclusion the researcher stated has to reach the plan as an exclusion.

``CohortSpec`` has carried ``exclusion_criteria`` all along: the CLI fills it
from ``--exclusion``, the pipeline threads it through, and the outbound Planner
context publishes it as ``exclusion_contract``. The Web caller was the one
consumer that had no exclusion channel -- it folded every filter, exclusions
included, into ``inclusion_criteria``. Across 52 recorded plans not one carried
a single exclusion predicate, and a cohort-flow figure cannot draw an exclusion
stage that was never declared as one.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
)
from easyicu.webserver.agent_pipeline_runs import (
    _exclusion_criteria,
    _inclusion_criteria,
)
from easyicu.webserver.pi_copilot import cohort_eligibility


def _study(**cohort) -> dict:
    return {"cohort": cohort}


def test_an_exclusion_is_compiled_as_an_exclusion() -> None:
    study = _study(
        label="ICU stays",
        age_min=18,
        min_icu_los_hours=24,
        exclude_readmissions=True,
        include_diagnoses=["condition-a"],
        exclude_diagnoses=["condition-b", "condition-c"],
    )

    inclusion = _inclusion_criteria(study)
    exclusion = _exclusion_criteria(study)

    # who enters
    assert "ICU stays" in inclusion
    assert "age range: 18 to *" in inclusion
    assert "minimum ICU length of stay: 24 hours" in inclusion
    assert "include diagnoses: condition-a" in inclusion

    # who is removed -- and none of it is still filed as inclusion
    assert "exclude diagnoses: condition-b, condition-c" in exclusion
    assert any("readmissions after the first" in row for row in exclusion)
    joined_inclusion = " | ".join(inclusion)
    assert "exclude" not in joined_inclusion
    assert "readmission" not in joined_inclusion


def test_a_study_with_no_filters_declares_neither_side() -> None:
    for study in ({}, _study(), _study(label=""), {"cohort": None}):
        assert _inclusion_criteria(study) == []
        assert _exclusion_criteria(study) == []


def test_a_readmission_exclusion_alone_still_produces_an_exclusion() -> None:
    """The commonest single exclusion must not need a diagnosis list."""

    study = _study(exclude_readmissions=True)
    assert _exclusion_criteria(study)
    assert _inclusion_criteria(study) == []
    # and a false flag declares nothing
    assert _exclusion_criteria(_study(exclude_readmissions=False)) == []


def test_the_pipeline_call_passes_the_exclusion_channel() -> None:
    """Compiling exclusions and not sending them changes nothing."""

    source = Path("src/easyicu/webserver/agent_pipeline_runs.py").read_text(
        encoding="utf-8"
    )
    assert "inclusion_criteria=_inclusion_criteria(study)," in source
    assert "exclusion_criteria=_exclusion_criteria(study)," in source
    # the reviewer-facing cohort summary reports both sides too
    assert '"exclusion_criteria": _exclusion_criteria(study),' in source


def test_the_planner_is_shown_the_exclusion_contract() -> None:
    """A field only the host reads is a field the Planner cannot honour."""

    source = inspect.getsource(outbound_safe_context_payload)
    assert '"inclusion_contract": context.cohort.inclusion_criteria' in source
    assert '"exclusion_contract": context.cohort.exclusion_criteria' in source


def test_the_offered_eligibility_options_reach_the_exclusion_channel() -> None:
    """The two halves of this repair have to meet.

    Copilot offers first-admission options; if those never compiled into an
    exclusion, choosing one would still produce a plan with no exclusion.
    """

    by_id = {option["id"]: option for option in cohort_eligibility.ELIGIBILITY_OPTIONS}

    first_admission = _study(**by_id["adults_first_admission"]["cohort"])
    assert _exclusion_criteria(first_admission), "first-admission must exclude"
    assert _inclusion_criteria(first_admission), "and must still state its age floor"

    all_admissions = _study(**by_id["adults_all_admissions"]["cohort"])
    assert _exclusion_criteria(all_admissions) == [], "keeping repeats excludes nobody"

    declined = _study(**by_id["no_eligibility_filter"]["cohort"])
    assert _exclusion_criteria(declined) == []
    assert _inclusion_criteria(declined) == []


def test_an_idea_handoff_claims_no_eligibility_it_was_not_given() -> None:
    """Same rule on the Idea Mining path: do not assert an unchosen criterion.

    The fallback used to send "Adult ICU cohort" as a bound inclusion contract,
    so an idea that never mentioned age reached the Planner claiming an adult
    restriction the export may not apply.
    """

    from easyicu.webserver.ideas.handoff import (
        _inclusion_criteria as idea_inclusion_criteria,
    )

    assert idea_inclusion_criteria({}) == []
    assert idea_inclusion_criteria({"cohort": {}}) == []
    assert idea_inclusion_criteria({"cohort": {"default": "   "}}) == []
    assert idea_inclusion_criteria({"cohort": {"default": "adults on ventilation"}}) == [
        "adults on ventilation"
    ]
    source = Path("src/easyicu/webserver/ideas/handoff.py").read_text(encoding="utf-8")
    assert "Adult ICU cohort from the active prepared" not in source


def test_both_sides_stay_bounded() -> None:
    study = _study(
        exclude_diagnoses=[f"condition-{index}" for index in range(40)],
        include_diagnoses=[f"other-{index}" for index in range(40)],
    )
    assert len(_inclusion_criteria(study)) <= 32
    assert len(_exclusion_criteria(study)) <= 32
    # the roster itself is capped before it is joined
    joined = _exclusion_criteria(study)[0]
    assert "condition-19" in joined
    assert "condition-20" not in joined
