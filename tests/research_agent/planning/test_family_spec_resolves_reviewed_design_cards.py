"""A family-spec plan resolves the reviewed comparator design cards.

Plan acceptance requires, whenever reviewed design cards are supplied, that
the selected design state one adopt/adapt/diverge/not-applicable decision per
design dimension, citing a card that states that dimension.  The family-spec
path passed the cards only to that final check: the request, the prompt and
the spec carried none of their facts, every template wrote an empty decision
list, and the plan was refused on every attempt.  The request now seals the
included sources' cards, the Planner answers each dimension, and the template
carries the answer into the selected design; the final check is unchanged.
Synthetic contexts, cards and scripted responses only.
"""

from __future__ import annotations

import json
import re

import pytest

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
)
from easyicu.research_agent.planning.progressive_compiler import progressive_cohort_concept_ids
from easyicu.research_agent.agents.progressive_planner import (
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS,
    LiteratureDesignEvidenceCard,
)
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    PLANNER_ROSTER,
    _context,
    _phenotyping_context,
    _phenotyping_payload,
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
    _spec_payload,
)

COMPARATOR = DIRECT_COMPARATORS[0]


def _card(key: str = COMPARATOR, dimensions=LITERATURE_DESIGN_DIMENSIONS):
    return LiteratureDesignEvidenceCard(
        citation_key=key,
        evidence_role="direct_comparator",
        access_mode="user_supplied_fulltext",
        full_text_locator="synthetic-reviewed-fulltext",
        full_text_sha256="c" * 64,
        supplement_status="not_published",
        reviewed_at="2026-09-30T00:00:00Z",
        evidence=[
            {
                "dimension": dimension,
                "source_backed_summary": (
                    f"CARD_FACT_{dimension}: the comparator's reviewed choice for this "
                    "dimension, paraphrased from its full text."
                ),
            }
            for dimension in dimensions
        ],
    )


def _decisions(*, skip=(), cite=None) -> list[dict]:
    return [
        {
            "dimension": dimension,
            "citation_keys": [(cite or {}).get(dimension, COMPARATOR)],
            "disposition": "adapt" if dimension == "time_zero_and_windows" else "adopt",
            "rationale": (
                f"This study follows the comparator on {dimension.replace('_', ' ')} "
                "within its own sealed cohort and time grid."
            ),
        }
        for dimension in LITERATURE_DESIGN_DIMENSIONS
        if dimension not in skip
    ]


FAMILIES = [
    pytest.param(
        lambda: _context(exact=False), "predicate_filtered",
        lambda request: _spec_payload(request, adjustment_set=PLANNER_ROSTER),
        id="landmark_association",
    ),
    pytest.param(
        _phenotyping_context, None,
        lambda request: _phenotyping_payload(
            request, features=["hr_max", "lactate_max", "map_min"], baseline=["age"],
            membership=None,
        ),
        id="phenotyping",
    ),
    pytest.param(
        _prediction_context, None,
        lambda request: _prediction_payload(
            request, features=["hr_max", "lactate_max", "map_min", "age"],
        ),
        id="prediction",
    ),
]


@pytest.mark.parametrize(("make_context", "cohort_mode", "payload"), FAMILIES)
def test_the_selected_design_resolves_every_reviewed_dimension(
    make_context, cohort_mode, payload,
) -> None:
    context = make_context()
    card = _card()
    request = _sealed_request(context, cohort_mode, cards=[card])
    assert [item.citation_key for item in request.literature_design_cards] == [COMPARATOR]

    llm, outcome = _run(
        context,
        [json.dumps({**payload(request), "literature_design_decisions": _decisions()})],
        required_primary_cohort_selection_mode=cohort_mode,
        literature_design_evidence_cards=[card],
    )

    assert len(llm.calls) == 1
    prompt = "\n\n".join(message.content for message in llm.calls[0][0])
    for dimension in LITERATURE_DESIGN_DIMENSIONS:
        assert f"CARD_FACT_{dimension}" in prompt
    assert '"literature_design_decisions"' in prompt
    selected = outcome.output.design_selection.selected
    assert [
        (item.dimension, item.disposition, item.citation_keys)
        for item in selected.literature_design_decisions
    ] == [
        (item["dimension"], item["disposition"], item["citation_keys"])
        for item in _decisions()
    ]


def test_an_unresolved_dimension_is_refused_and_the_retry_resolves_it() -> None:
    context = _context(exact=False)
    card = _card()
    request = _sealed_request(context, "predicate_filtered", cards=[card])
    base = _spec_payload(request, adjustment_set=PLANNER_ROSTER)

    llm, outcome = _run(
        context,
        [
            json.dumps({**base, "literature_design_decisions": _decisions(
                skip=("conclusion_boundaries",)
            )}),
            json.dumps({**base, "literature_design_decisions": _decisions()}),
        ],
        literature_design_evidence_cards=[card],
    )

    assert len(llm.calls) == 2
    assert "family_spec_literature_decision_missing" in llm.calls[1][0][-1].content
    assert len(outcome.output.design_selection.selected.literature_design_decisions) == 7


def test_a_decision_cites_only_a_card_that_states_its_dimension() -> None:
    second = "comparator_beta_2023_2"
    context = _context(exact=False)
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=(*ALLOWED_CITATIONS, second),
        direct_comparator_literature_keys=(COMPARATOR, second),
        comparison_literature_keys=(COMPARATOR, second),
        required_primary_cohort_selection_mode="predicate_filtered",
        literature_design_cards=[_card(), _card(second, dimensions=("study_population",))],
        cohort_concept_ids=progressive_cohort_concept_ids(context, select_progressive_variables(context)),
    )
    base = _spec_payload(request, adjustment_set=PLANNER_ROSTER)

    for dimension, accepted in (("study_population", True), ("time_zero_and_windows", False)):
        spec = spec_from_mapping(
            {**base, "literature_design_decisions": _decisions(cite={dimension: second})}
        )
        if accepted:
            validate_family_plan_spec(spec, request)
            continue
        with pytest.raises(FamilySpecError) as refused:
            validate_family_plan_spec(spec, request)
        assert refused.value.reason_code == "family_spec_literature_decision_source_unsupported"


@pytest.mark.parametrize(
    "cards",
    [
        pytest.param([_card(dimensions=LITERATURE_DESIGN_DIMENSIONS[:-1])], id="a_dimension_unstated"),
        pytest.param([_card("strobe_2007")], id="a_card_for_a_source_not_compared"),
    ],
)
def test_cards_that_cannot_support_every_dimension_fail_before_the_provider(cards) -> None:
    context = _context(exact=False)

    with pytest.raises(FamilySpecError) as refused:
        _run(context, [], literature_design_evidence_cards=cards)

    assert refused.value.reason_code == "family_spec_literature_design_dimensions_unsupported"


def test_a_request_without_cards_keeps_its_digest_schema_and_shape() -> None:
    context = _context(exact=False)
    plain = _request(context)
    carded = _sealed_request(context, "predicate_filtered", cards=[_card()])

    assert "literature_design_cards" not in plain.model_dump(mode="json")
    assert plain.request_sha256 != carded.request_sha256
    assert plain.request_sha256 == _sealed_request(
        context, "predicate_filtered", cards=[]
    ).request_sha256
    for request, expected in ((plain, False), (carded, True)):
        schema = json.loads(family_spec_structured_output_request(request).schema_json)
        shape = family_spec_response_shape(request)
        written = re.findall(r'^- "([a-z_0-9]+)"', shape, flags=re.MULTILINE)
        assert set(written) == set(schema["properties"])
        assert ("literature_design_decisions" in schema["properties"]) is expected
        assert ("CARD_FACT_" in family_spec_user_prompt(
            request, variable_descriptions={}
        )) is expected

    spec = spec_from_mapping({
        **_spec_payload(plain, adjustment_set=PLANNER_ROSTER),
        "literature_design_decisions": _decisions(),
    })
    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec, plain)
    assert refused.value.reason_code == "family_spec_literature_decision_unrequested"


def _sealed_request(context, cohort_mode, *, cards):
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=cohort_mode,
        literature_design_cards=cards,
        cohort_concept_ids=progressive_cohort_concept_ids(context, select_progressive_variables(context)),
    )
