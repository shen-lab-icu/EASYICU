"""A cohort predicate has to separate rows by the values its column takes.

A predicate filters one column, and when that column's values form a closed
set each value can be judged with the comparison the cohort builder applies.
An inclusion no value meets, or an exclusion every value meets, leaves no row
that has a value: the review refuses the plan.  An inclusion every value
meets, or an exclusion none meets, cannot separate rows by value; it may be a
bound the study states for every value, so the review reports it as a major
finding.  A planning schema has no rows, so the set a column's owner declares
is what catches a threshold copied from a concept's description: a Planner
once required a 0/1 diagnosis flag to be at least 2.  Levels observed in one
cohort are not every value a column can take, so they judge only an empty
cohort, and they are never named.  The review sends the plan back to rewrite
the comparison and never to drop the predicate.  Fixtures are synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.planning import scientific_review
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    cohort_concept_id_scope,
)
from easyicu.research_agent.planning.cohort_predicate_domain import (
    cohort_predicates_outside_column_domain,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    cohort_predicate_domain_findings,
    plan_revision_blocker_codes,
)
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole

from .scientific_review_fixtures import _context, _literature, _plan

_CODE = "COHORT_PREDICATE_OUTSIDE_COLUMN_DOMAIN"
_VARIABLES = (
    # A logical event status the concept dictionary declares 0/1.
    ConceptDescriptor(
        name="dictionary_flag", role=VariableRole.OTHER, dtype="float64", source_concept="vent_ind"
    ),
    # A declared ordinal scale, and the same scale as a window summary column.
    ConceptDescriptor(
        name="organ_score",
        role=VariableRole.ORDINAL_SCORE,
        dtype="int64",
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3, 4],
    ),
    ConceptDescriptor(
        name="organ_max",
        role=VariableRole.ORDINAL_SCORE,
        dtype="int64",
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3, 4],
    ),
    # Sets observed in the cohort, with no declaration.
    ConceptDescriptor(
        name="observed_flag",
        role=VariableRole.OTHER,
        dtype="int64",
        observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
    ),
    ConceptDescriptor(
        name="category_code",
        role=VariableRole.OTHER,
        dtype="object",
        observed_domain={"n_unique": 3, "is_binary": False, "levels": ["A", "B", "C"]},
    ),
    # A column observed constant in this cohort.
    ConceptDescriptor(
        name="constant_flag",
        role=VariableRole.OTHER,
        dtype="int64",
        observed_domain={"n_unique": 1, "is_constant": True, "levels": [1]},
    ),
    # A measurement without a closed set.
    ConceptDescriptor(
        name="lactate_value",
        role=VariableRole.LAB,
        dtype="float64",
        observed_domain={"n_unique": 80, "min": 0.5, "max": 12.0},
    ),
)
_IDS = (
    "dictionary_flag",
    "organ_score",
    "organ",
    "observed_flag",
    "category_code",
    "constant_flag",
    "lactate_value",
    "unbound_concept",
)


@pytest.fixture(autouse=True)
def _known_concepts():
    with cohort_concept_id_scope(_IDS):
        yield


def _study() -> ResearchContext:
    base = _context()
    return base.model_copy(update={"variables": [*base.variables, *_VARIABLES]})


def _predicate(concept: str, op: str = ">=", value: object = 2.0, *, aggregation: str = "max") -> dict:
    return {
        "concept_id": concept,
        "time_window": {"anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": 24.0},
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


def _cohort(*, inclusion: list[dict] = (), exclusion: list[dict] = ()) -> CohortDefinition:
    return CohortDefinition.from_dict(
        {"name": "primary", "inclusion": list(inclusion), "exclusion": list(exclusion)}
    )


def _judge(**sides: list[dict]):
    cohort = _cohort(**sides)
    return cohort_predicates_outside_column_domain(
        _study(), inclusion=cohort.inclusion, exclusion=cohort.exclusion
    )


def test_a_threshold_no_declared_value_meets_keeps_no_row() -> None:
    (item,) = _judge(inclusion=[_predicate("dictionary_flag", ">=", 2.0)])

    assert (item.side, item.column, item.effect, item.declared_levels) == (
        "inclusion",
        "dictionary_flag",
        "keeps_no_row",
        (0, 1),
    )
    assert item.message() == (
        "The cohort inclusion dictionary_flag (max) >= 2.0 filters column "
        "'dictionary_flag': its owner declares the values [0, 1], and none of "
        "them meets it, so it keeps no row that has a value"
    )
    assert item.empties_cohort


@pytest.mark.parametrize(
    ("side", "predicate", "effect"),
    [
        ("inclusion", _predicate("organ_score", ">", 4), "keeps_no_row"),
        ("inclusion", _predicate("organ_score", ">=", 0), "keeps_every_row"),
        ("inclusion", _predicate("dictionary_flag", "in", [0, 1]), "keeps_every_row"),
        ("exclusion", _predicate("organ_score", ">=", 0), "removes_every_row"),
        ("exclusion", _predicate("dictionary_flag", "==", 2), "removes_no_row"),
        ("inclusion", _predicate("dictionary_flag", "==", 1), None),
        ("inclusion", _predicate("organ_score", "in", [2, 3]), None),
        ("exclusion", _predicate("organ_score", ">=", 3), None),
    ],
)
def test_a_declared_set_judges_an_empty_cohort_and_a_restriction_that_applies_nothing(
    side: str, predicate: dict, effect: str | None
) -> None:
    found = _judge(**{side: [predicate]})

    assert [item.effect for item in found] == ([effect] if effect else [])
    assert [item.empties_cohort for item in found] == (
        [effect in {"keeps_no_row", "removes_every_row"}] if effect else []
    )


def test_an_observed_set_judges_only_an_empty_cohort_and_is_never_named() -> None:
    (empty,) = _judge(inclusion=[_predicate("observed_flag", ">=", 2.0)])
    assert (empty.effect, empty.declared_levels, empty.level_count) == ("keeps_no_row", None, 2)
    assert "it holds 2 observed values" in empty.message()
    assert "[0, 1]" not in empty.message()

    (every,) = _judge(exclusion=[_predicate("observed_flag", "in", [0, 1])])
    assert every.effect == "removes_every_row"

    # Matching no one in this cohort, or everyone in it, is not an error.
    assert _judge(exclusion=[_predicate("observed_flag", "==", 2)]) == ()
    assert _judge(inclusion=[_predicate("observed_flag", "in", [0, 1])]) == ()


def test_a_summary_column_is_judged_by_its_own_values() -> None:
    (item,) = _judge(inclusion=[_predicate("organ", ">=", 5, aggregation="max")])

    assert (item.column, item.effect) == ("organ_max", "keeps_no_row")


def test_any_and_all_are_judged_on_an_event_status_only() -> None:
    # They summarize an event status: a 0/1 set keeps its values under them.
    (item,) = _judge(inclusion=[_predicate("dictionary_flag", "==", 2, aggregation="any")])
    assert item.effect == "keeps_no_row"
    (item,) = _judge(exclusion=[_predicate("dictionary_flag", "!=", 2, aggregation="all")])
    assert item.effect == "removes_every_row"
    assert _judge(inclusion=[_predicate("dictionary_flag", "==", 1, aggregation="any")]) == ()

    # A scale is not an event status; its levels do not describe them.
    assert _judge(inclusion=[_predicate("organ_score", "==", 5, aggregation="any")]) == ()
    assert _judge(exclusion=[_predicate("organ_score", "!=", 7, aggregation="all")]) == ()


@pytest.mark.parametrize(
    "predicate",
    [
        _predicate("dictionary_flag", "missing", None),
        _predicate("dictionary_flag", "not_missing", None),
        _predicate("lactate_value", ">=", 2.0),
        _predicate("category_code", ">=", 2.0),
        _predicate("constant_flag", "==", 0),
        _predicate("unbound_concept", ">=", 2.0),
        # A mean of a 0/1 flag lies between its values.
        _predicate("dictionary_flag", "==", 0.5, aggregation="mean"),
        _predicate("organ_score", ">", 4, aggregation="sum"),
    ],
    ids=[
        "missing",
        "not_missing",
        "no_closed_set",
        "not_comparable",
        "one_level",
        "no_column",
        "mean",
        "sum",
    ],
)
def test_a_predicate_without_a_value_or_a_closed_set_is_not_judged(predicate: dict) -> None:
    assert _judge(inclusion=[predicate]) == ()
    assert _judge(exclusion=[predicate]) == ()


def test_the_review_sends_the_plan_back_to_rewrite_the_comparison() -> None:
    plan = _plan().model_copy(
        update={"cohort": _cohort(inclusion=[_predicate("dictionary_flag", ">=", 2.0)])}
    )

    (finding,) = cohort_predicate_domain_findings(_study(), plan)

    assert (finding.code, finding.severity, finding.remediation_route) == (
        _CODE,
        "blocker",
        "agent_plan_revision",
    )
    assert "its owner declares the values [0, 1]" in finding.message
    assert "Keep the predicate" in finding.remediation
    assert "missing or not_missing" in finding.remediation
    # The Planner repairs it; nothing else has to act first.
    assert plan_revision_blocker_codes([finding]) == ()
    assert cohort_predicate_domain_findings(_study(), plan.model_copy(update={"cohort": None})) == []


@pytest.mark.parametrize(
    ("sides", "severity"),
    [
        ({"inclusion": [_predicate("organ_score", ">", 4)]}, "blocker"),
        ({"exclusion": [_predicate("organ_score", ">=", 0)]}, "blocker"),
        # A bound every value meets may be one the study states: a doubt.
        ({"inclusion": [_predicate("organ_score", ">=", 0)]}, "major"),
        ({"exclusion": [_predicate("dictionary_flag", "==", 2)]}, "major"),
    ],
    ids=["keeps_no_row", "removes_every_row", "keeps_every_row", "removes_no_row"],
)
def test_an_empty_cohort_is_refused_and_a_criterion_applying_nothing_is_doubted(
    sides: dict, severity: str
) -> None:
    plan = _plan().model_copy(update={"cohort": _cohort(**sides)})

    (finding,) = cohort_predicate_domain_findings(_study(), plan)

    assert (finding.severity, finding.remediation_route) == (severity, "agent_plan_revision")


def test_a_plan_review_states_the_finding() -> None:
    context = _study()

    def codes(predicate: dict) -> set[str]:
        plan = _plan().model_copy(update={"cohort": _cohort(inclusion=[predicate])})
        review = build_plan_scientific_review(
            context=context,
            plan=plan,
            literature=_literature(),
            figure_strategy=build_article_figure_strategy(context),
        )
        return {item.code for item in review.findings}

    assert _CODE in codes(_predicate("dictionary_flag", ">=", 2.0))
    assert _CODE not in codes(_predicate("dictionary_flag", "==", 1))


def test_the_finding_reads_in_chinese() -> None:
    vocab = (
        Path(scientific_review.__file__).resolve().parents[2]
        / "webserver/static/js/screens-agent-reader-vocab.js"
    ).read_text(encoding="utf-8")

    assert f"{_CODE}: ['入组谓词与所筛选列的取值对不上'" in vocab
