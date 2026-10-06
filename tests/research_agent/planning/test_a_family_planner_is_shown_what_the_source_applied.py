"""A family Planner is shown what the source export applied.

A family that applies its own cohort is offered the run's cohort concepts and
the study's wording, and its population authority lists what is already
applied: the typed bounds and the source's concept population.  The export's
inclusion and exclusion contracts were missing from that list.  A Planner could
restate, through another definition, a restriction the export had already
applied, narrowing the rows to an intersection nobody chose, and was never told
when the export's recorded selection was everything that selected them.  The
request now carries the criteria the source is known to have applied: its whole
recorded selection, or only the host's own criteria when that selection is not
recorded.  It also says whether the selection is recorded; the authority lists
those criteria, and states the record only when there is one.  Fixtures are
generic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_user_prompt,
)
from easyicu.research_agent.agents.progressive_planner import (
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import FamilySpecRequest
from easyicu.research_agent.planning.progressive_compiler import progressive_cohort_concept_ids
from easyicu.research_agent.schema import ResearchContext

from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
)

_FIRST_STAY = (
    "each patient's later ICU stays: the host keeps only the first ICU stay "
    "per patient across the bound source, before planning"
)
_INCLUSION = ["age range: 18 to *", "include diagnoses: condition-a"]
_FIELDS = ("source_applied_inclusion", "source_applied_exclusion", "source_selection_recorded")


def _with(
    context: ResearchContext,
    *,
    inclusion: list[str],
    exclusion: list[str],
    source_selection: dict[str, Any] | None,
    **cohort: Any,
) -> ResearchContext:
    """The criteria a context declares, and the record of its source selection."""

    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    if source_selection is not None:
        constraints["source_selection"] = source_selection
    constraints["cohort"] = {**constraints.get("cohort", {}), **cohort}
    return context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={"inclusion_criteria": inclusion, "exclusion_criteria": exclusion}
            ),
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            ),
        }
    )


def _request(context: ResearchContext, *, cohort_mode: str | None = None) -> FamilySpecRequest:
    variables = select_progressive_variables(context)
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=cohort_mode,
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
    )


def _authority(request: FamilySpecRequest) -> dict[str, Any]:
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    return json.loads(prompt.split("population you write):\n", 1)[1].split("\n\n", 1)[0])


def test_a_recorded_selection_is_everything_that_selected_the_rows() -> None:
    context = _with(
        _context(),
        inclusion=_INCLUSION,
        exclusion=[_FIRST_STAY],
        source_selection={"recorded": True, "host_applied": [_FIRST_STAY]},
    )

    request = _request(context)
    authority = _authority(request)

    assert request.source_applied_inclusion == _INCLUSION
    assert request.source_applied_exclusion == [_FIRST_STAY]
    assert request.source_selection_recorded is True
    assert authority["already_applied"]["source_inclusion_contracts"] == _INCLUSION
    assert authority["already_applied"]["source_exclusion_contracts"] == [_FIRST_STAY]
    assert authority["source_selection_recorded"] is True
    # The guide says what both mean.
    assert "a source inclusion or exclusion contract) is applied already: do not state it again" in FAMILY_SPEC_GUIDE
    assert (
        "When source_selection_recorded is true, already_applied is everything that "
        "selected the input rows"
    ) in FAMILY_SPEC_GUIDE


def test_an_unrecorded_selection_shows_only_what_the_host_applied() -> None:
    context = _with(
        _context(),
        inclusion=_INCLUSION,
        exclusion=[_FIRST_STAY],
        source_selection={"recorded": False, "host_applied": [_FIRST_STAY]},
    )

    request = _request(context)
    authority = _authority(request)

    # The export's own criteria are declared, not known to be applied.
    assert request.source_applied_inclusion == []
    assert request.source_applied_exclusion == [_FIRST_STAY]
    assert request.source_selection_recorded is False
    assert "source_inclusion_contracts" not in authority["already_applied"]
    assert authority["already_applied"]["source_exclusion_contracts"] == [_FIRST_STAY]
    # Nothing is claimed about an unrecorded selection.
    assert "source_selection_recorded" not in authority


def test_a_context_without_a_record_adds_nothing_to_the_authority() -> None:
    plain = _context()
    request = _request(plain)

    assert all(field not in request.model_dump(mode="json") for field in _FIELDS)
    assert _authority(request)["already_applied"] == {"age_min": 18.0}


def test_the_study_wording_copied_into_the_criteria_is_not_a_source_contract() -> None:
    label = "Adults with condition-a"
    context = _with(
        _context(),
        inclusion=[label, "age range: 18 to *"],
        exclusion=[],
        source_selection={"recorded": True, "host_applied": []},
        label=label,
    )

    request = _request(context)

    assert request.source_applied_inclusion == ["age range: 18 to *"]
    assert request.source_selection_recorded is True


def test_a_request_offered_no_population_carries_no_source_fields() -> None:
    context = _with(
        _context(),
        inclusion=_INCLUSION,
        exclusion=[_FIRST_STAY],
        source_selection={"recorded": True, "host_applied": [_FIRST_STAY]},
    )

    bound = _request(context, cohort_mode="all_input_rows")
    offered = _request(context)

    # The caller binds every input row: nothing is offered, and the digest
    # keeps the identity it had before these fields existed.
    assert all(field not in bound.model_dump(mode="json") for field in _FIELDS)
    assert "Population authority" not in family_spec_user_prompt(bound, variable_descriptions={})
    # Offered, the source's record joins the request digest.
    cleared = offered.model_copy(
        update={
            "source_applied_inclusion": [],
            "source_applied_exclusion": [],
            "source_selection_recorded": False,
        }
    )
    assert all(field in offered.model_dump(mode="json") for field in _FIELDS)
    assert cleared.request_sha256 != offered.request_sha256


@pytest.mark.parametrize("field", ["source_applied_inclusion", "source_applied_exclusion"])
def test_a_source_contract_list_holds_unique_criteria(field: str) -> None:
    offered = _request(
        _with(
            _context(),
            inclusion=_INCLUSION,
            exclusion=[_FIRST_STAY],
            source_selection={"recorded": True, "host_applied": [_FIRST_STAY]},
        )
    )
    payload = offered.model_dump(mode="json")

    for values in (["criterion-a", "criterion-a"], ["criterion-a", " "]):
        with pytest.raises(ValidationError):
            FamilySpecRequest.model_validate({**payload, field: values})
