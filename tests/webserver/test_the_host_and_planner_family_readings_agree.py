"""The host's family reading and the planner's routing agree on a question's words.

Two layers read a question's analysis family: the host's study setup records
a design from it (``webserver.study_family_reading``), and the planner's
keyword routing plans a study that records none
(``research_agent.planning.analysis_types``).  On every question the host
reads, the planner, given no design and no fields, reads the same family; on
a question about the data themselves, which the host leaves unread, the
planner audits the data.  Synthetic questions only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type_with_basis,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
from easyicu.webserver.study_family_reading import study_family_reading

_READ = [
    "Is admission lactate associated with 28-day mortality among adult ICU stays?",
    "Examine the association between body mass index and in-hospital mortality.",
    "What is the relationship between early fluid balance and ICU mortality?",
    "What are the risk factors for delirium in adult ICU stays?",
    "在成人 ICU 住院中，入 ICU 时的乳酸是否与 28 天死亡相关？",
    "在成人 ICU 住院中，入 ICU 时的乳酸与 28 天死亡是否有关？",
    "在成人 ICU 住院中，乳酸跟院内死亡是否有关？",
    "成人 ICU 住院中谵妄的危险因素有哪些？",
    "Among adult ICU stays, what proportion had a sodium below 130 mmol/L in the first 24 hours?",
    "How common is hypoglycemia in the first 24 hours of an ICU stay?",
    "在成人 ICU 住院中，前 24 小时低钠血症的比例是多少？",
    "在成人 ICU 住院中，前 24 小时有多大比例发生低血糖？",
    "估计成人 ICU 住院中低钠血症的患病率，并描述其与院内死亡的关系。",
]
_DATA = [
    "What is the missingness of lactate in the first 24 hours of adult ICU stays?",
    "在成人 ICU 住院中，前 24 小时乳酸的缺失率和数据完整性如何？",
]


def _planner(question: str) -> tuple[str, str]:
    spec, basis = infer_analysis_type_with_basis(
        ResearchContext(
            research_question=question,
            cohort=CohortDescriptor(cohort_name="study", database="miiv", n_stays=0),
            variables=[],
        )
    )
    return spec.key, basis


@pytest.mark.parametrize("question", _READ)
def test_the_planner_reads_the_family_the_host_reads(question: str) -> None:
    reading = study_family_reading(question)

    assert reading is not None
    assert _planner(question) == (reading.family, "keyword")


@pytest.mark.parametrize("question", _DATA)
def test_a_question_about_the_data_is_unread_by_the_host_and_audited_by_the_planner(
    question: str,
) -> None:
    assert study_family_reading(question) is None
    assert _planner(question) == ("data_quality_audit", "keyword")
