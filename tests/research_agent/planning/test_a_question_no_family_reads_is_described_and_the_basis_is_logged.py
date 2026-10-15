"""A question is planned on the family its words state, and the run log says how.

The planner's keyword routing (``analysis_types.infer_analysis_type``) reads a
relationship question in the orders researchers write it ("X 是否与 Y 相关",
"X 与 Y 是否有关"), a proportion question as descriptive, and a question about
the data themselves as a data-quality audit, whichever fields the context
holds.  A question no family's words read is described; only an empty
question is a data-quality audit.  How the family was chosen -- the study's
typed design, the question's words, or that fallback -- is written to the run
log (``analysis_family_trace``).  Synthetic questions only.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.planning.analysis_family_trace import (
    traced_analysis_type_key,
)
from easyicu.research_agent.planning.analysis_types import (
    FAMILY_BASIS_FALLBACK,
    FAMILY_BASIS_KEYWORD,
    FAMILY_BASIS_STUDY_DESIGN,
    infer_analysis_type_with_basis,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ResearchContext,
    UserPreferences,
)


def _context(question: str, **fields: Any) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="study", database="miiv", n_stays=0),
        variables=[],
        **fields,
    )


_FIELDS = (
    {},
    {"target_outcome": "death"},
    {"target_outcome": "death", "primary_exposure": "lactate_max"},
)


@pytest.mark.parametrize("fields", _FIELDS)
@pytest.mark.parametrize(
    ("question", "family"),
    [
        (
            "在成人 ICU 住院中，入 ICU 时的乳酸是否与 28 天死亡相关？",
            "association_study",
        ),
        (
            "在成人 ICU 住院中，入 ICU 时的乳酸与 28 天死亡是否有关？",
            "association_study",
        ),
        ("在成人 ICU 住院中，乳酸和院内死亡是否存在关联？", "association_study"),
        ("在成人 ICU 住院中，乳酸跟院内死亡是否有关？", "association_study"),
        (
            "Among adult ICU stays, what proportion had a sodium below 130 mmol/L?",
            "descriptive_epidemiology",
        ),
        (
            "在成人 ICU 住院中，前 24 小时低钠血症的比例是多少？",
            "descriptive_epidemiology",
        ),
        (
            "在成人 ICU 住院中，前 24 小时有多大比例发生低血糖？",
            "descriptive_epidemiology",
        ),
        (
            "在成人 ICU 住院中，前 24 小时乳酸的缺失率和数据完整性如何？",
            "data_quality_audit",
        ),
    ],
)
def test_the_question_states_its_family_whatever_the_context_holds(
    question, family, fields
) -> None:
    spec, basis = infer_analysis_type_with_basis(_context(question, **fields))

    assert (spec.key, basis) == (family, FAMILY_BASIS_KEYWORD)


@pytest.mark.parametrize(
    "question",
    [
        "列出与本研究相关的变量，并报告它们的缺失率。",
        "Among adult ICU stays, what proportion died after adjustment for age?",
    ],
)
def test_an_attributive_or_inferential_wording_is_not_read_as_those_families(
    question,
) -> None:
    spec, basis = infer_analysis_type_with_basis(_context(question))

    if "相关的变量" in question:
        # A noun qualified as relevant asks no relationship; the data are asked about.
        assert spec.key == "data_quality_audit"
    else:
        # An adjusted proportion is not a description by its words.
        assert basis == FAMILY_BASIS_FALLBACK


def test_a_question_no_family_reads_is_described_and_an_empty_one_audited() -> None:
    question = "成人 ICU 住院患者前 24 小时的血糖情况。"

    spec, basis = infer_analysis_type_with_basis(_context(question))
    assert (spec.key, basis) == ("descriptive_epidemiology", FAMILY_BASIS_FALLBACK)
    paired = infer_analysis_type_with_basis(
        _context(question, target_outcome="death", primary_exposure="glu_min")
    )
    assert (paired[0].key, paired[1]) == ("association_study", FAMILY_BASIS_FALLBACK)
    empty = infer_analysis_type_with_basis(_context(""))
    assert (empty[0].key, empty[1]) == ("data_quality_audit", FAMILY_BASIS_FALLBACK)


def test_a_stated_design_decides_and_the_run_log_says_how() -> None:
    context = _context(
        "成人 ICU 住院患者前 24 小时的血糖情况。",
        user_preferences=UserPreferences(inferred_analysis_family="association_study"),
    )
    events: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    key = traced_analysis_type_key(
        context, lambda *args, **kwargs: events.append((args, kwargs)), "run-1"
    )

    assert key == "association_study"
    ((args, kwargs),) = events
    assert args[0] == "plan"
    assert kwargs == {
        "run_id": "run-1",
        "analysis_family": "association_study",
        "family_basis": FAMILY_BASIS_STUDY_DESIGN,
    }


@pytest.mark.parametrize(
    ("question", "family", "basis"),
    [
        (
            "前 24 小时低钠血症的比例是多少？",
            "descriptive_epidemiology",
            FAMILY_BASIS_KEYWORD,
        ),
        (
            "成人 ICU 入住患者的血糖情况。",
            "descriptive_epidemiology",
            FAMILY_BASIS_FALLBACK,
        ),
        ("", "data_quality_audit", FAMILY_BASIS_FALLBACK),
    ],
)
def test_the_live_progress_channel_keeps_the_family_basis(
    question, family, basis
) -> None:
    from types import SimpleNamespace
    from easyicu.research_agent.orchestration.progress import ResumableProgressChannel

    events = []
    audit = []
    channel = ResumableProgressChannel(events.append)
    channel.bind_audit_logger(SimpleNamespace(emit=lambda **row: audit.append(row)))
    assert traced_analysis_type_key(_context(question), channel.emit, "run-1") == family
    assert len(events) == 1
    assert events[0]["stage"] == "plan"
    assert events[0]["analysis_family"] == family
    assert events[0]["family_basis"] == basis
    assert "research_question" not in events[0]
    assert audit[0]["phase"] == "plan"
    assert audit[0]["detail"]["analysis_family"] == family
    assert audit[0]["detail"]["family_basis"] == basis


def test_the_current_design_wins_over_conflicting_question_words() -> None:
    context = _context(
        "前 24 小时低钠血症的比例是多少？",
        user_preferences=UserPreferences(inferred_analysis_family="association_study"),
    )
    assert infer_analysis_type_with_basis(context)[0].key == "association_study"
    assert infer_analysis_type_with_basis(context)[1] == FAMILY_BASIS_STUDY_DESIGN
