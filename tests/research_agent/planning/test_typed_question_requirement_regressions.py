"""The approved review table, expressed in independent typed coordinates."""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.requirement_typing.contract import AtomicPart, SourceSpan
from easyicu.research_agent.planning.requirement_typing.matcher import match
from tests.support.requirement_typing import (
    AUTHORITY, CASES, DIALYSIS12, DICTIONARY, LAC12, REPORT_STAYS, Case, bind, fact, point,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_regression_table(case: Case) -> None:
    part = case.atomic_part()
    expected = "not_covered" if part.key is not None else "unknown"
    assert match(part.key, authority=AUTHORITY, dictionary=DICTIONARY, facts=()).state == expected
    # The negative must fail even if every rejected key is in this same plan.
    assert match(
        part.key, authority=AUTHORITY, dictionary=DICTIONARY,
        facts=tuple(fact(rejected) for rejected in case.rejected),
    ).state == expected
    if part.key is not None:
        matching_fact = case.accepted or part.key
        assert match(
            part.key, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(matching_fact),),
        ).state == "covered"
        assert match(
            part.key, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(
                matching_fact, AUTHORITY.model_copy(update={"plan_sha256": "3" * 64}),
            ),),
        ).state == "not_covered"


def test_duplicate_quotes_siblings_and_order_cannot_rebind_a_part() -> None:
    question = "定义透析时间零点为12小时。定义透析时间零点为12小时。交代重复入住。"
    quote = "定义透析时间零点为12小时"
    first = AtomicPart(
        part_id="dialysis", source_span=SourceSpan(start=0, end=len(quote)), key=DIALYSIS12,
    )
    second_start = question.index(quote, len(quote))
    second = AtomicPart(
        part_id="unresolved", source_span=SourceSpan(start=second_start, end=second_start + len(quote)),
        key=None, facet_hint="time_zero",
    )
    siblings = (
        AtomicPart(part_id="repeat-stays", source_span=SourceSpan(
            start=question.index("交代"), end=len(question),
        ), key=REPORT_STAYS),
        AtomicPart(part_id="lactate", source_span=SourceSpan(start=4, end=6), key=LAC12),
        AtomicPart(part_id="sofa", source_span=SourceSpan(start=5, end=7), key=point(8, bind("sofa"))),
    )
    facts = tuple(fact(item) for item in (DIALYSIS12, LAC12, REPORT_STAYS))
    for parts in ((first, second), (second, siblings[0], first), (*siblings, second, first)):
        results = {
            part.part_id: match(part.key, authority=AUTHORITY, dictionary=DICTIONARY, facts=facts).state
            for part in parts
        }
        assert results["dialysis"] == "covered"
        assert results["unresolved"] == "unknown"
