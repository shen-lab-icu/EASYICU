"""An adult scope is read through a negation and a minor condition.

Literature screening requires adult design analogues when the study declares
an adult population.  The statement rules took the severity adjective
"minor" for a group of minors ("Adults with minor head injury"), did not see
a child group a negation excludes ("Adult patients but not children",
"成人患者，不纳入儿童"), and took "no older than 18" and "未年满18周岁" (not yet 18)
for an adult age floor: only "not older than" was guarded, and "未年满18周岁"
contains "年满18周岁".  An adult study with a minor condition was then
screened against analogues of any age, and a study of minors against adult
analogues only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.literature import _adult_population_required, _states_adult_scope
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext


def _context(question: str) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="local cohort", database="miiv", n_stays=0),
        variables=[
            ConceptDescriptor(name="condition_x", dtype="int64"),
            ConceptDescriptor(name="death", dtype="int64"),
        ],
        primary_exposure="condition_x",
        target_outcome="death",
    )


@pytest.mark.parametrize(
    "question",
    [
        # A minor condition is no group of minors.
        "Among adults with minor head injury, is condition-x associated with hospital mortality?",
        "Among adult patients undergoing minor surgery, is condition-x associated with hospital mortality?",
        # A child group a negation excludes.
        "Among adult ICU patients but not children, is condition-x associated with hospital mortality?",
        "Among adults (not including children), is condition-x associated with hospital mortality?",
        "在成人 ICU 患者中，不纳入儿童，condition-x 与住院死亡有何关联？",
        "在成年 ICU 患者中，未纳入儿童，condition-x 与住院死亡有何关联？",
    ],
)
def test_an_adult_study_with_a_minor_condition_or_an_excluded_child_declares_the_scope(question):
    assert _adult_population_required(_context(question))


@pytest.mark.parametrize(
    "question",
    [
        # A ceiling of 18 years names minors.
        "Among ICU patients no older than 18, is condition-x associated with hospital mortality?",
        "Among ICU patients not over 18, is condition-x associated with hospital mortality?",
        "在未年满18周岁的 ICU 患者中，condition-x 与住院死亡有何关联？",
        # Minors named as people, beside adults.
        "Among adult and minor patients, is condition-x associated with hospital mortality?",
        "Among adults and minors in the ICU, is condition-x associated with hospital mortality?",
        # A negation of something else leaves the children included.
        "Among adults with no sepsis and children, is condition-x associated with hospital mortality?",
    ],
)
def test_a_ceiling_of_18_or_minors_beside_adults_declares_no_adult_only_scope(question):
    assert not _adult_population_required(_context(question))


@pytest.mark.parametrize(
    ("statement", "adult"),
    [
        ("Adults with minor injuries", True),
        ("Adult patients after minor-surgery procedures", True),
        ("Adults, not children", True),
        ("Adults with minor trauma and no children", True),
        ("成人患者（不包括儿童）", True),
        ("年满18周岁的患者", True),
        ("Patients older than 18", True),
        ("Adults; minors were excluded", True),
        ("Adults excluding minors", True),
        ("Patients not older than 18", False),
        ("Minors and adults", False),
        ("Not only adults but also children", False),
        ("Adults, or a patient who is a minor", False),
    ],
)
def test_each_statement_reads_its_own_negations(statement, adult):
    assert _states_adult_scope(statement) is adult
