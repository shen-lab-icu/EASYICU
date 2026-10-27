"""A prediction made at several times, each over the horizon after it, is dynamic.

pilot1 10-27: 「在入 ICU 第 24、48、72 小时，用到当时为止的生命体征和化验，预测之后
7 天内 ICU 死亡的效果如何？」 was routed by its 「预测」 to one static prediction
model, and the plan declared it could not do the three prediction times.  The
dynamic-prediction family's trigger words were English method names only; the
question's structure -- several prediction times on one clock and a horizon
that follows each -- now routes it in any wording.  One prediction time, one
look-back window, or listed horizons of one endpoint stay a static model.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.analysis_types import (
    FAMILY_BASIS_KEYWORD,
    infer_analysis_type_with_basis,
    landmark_prediction_framing,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


def _family(question: str) -> tuple[str, str]:
    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="study", database="eicu_demo", n_stays=0),
        variables=[],
    )
    spec, basis = infer_analysis_type_with_basis(context)
    return spec.key, basis


@pytest.mark.parametrize(
    "question",
    [
        "在 eICU demo 中，在入 ICU 第 24、48、72 小时，用到当时为止的生命体征和化验，预测之后 7 天内 ICU 死亡的效果如何？",
        "成人在 24 和 48 小时时仍在 ICU；使用此前 24 小时的乳酸预测之后 24 小时内 ICU 死亡。",
        "入 ICU 后每 24 小时，用到当时为止的数据预测接下来 48 小时内的死亡风险。",
        "In adults still in the ICU at 24, 48 and 72 hours, predict ICU death over the next 7 days from the vital signs available by then.",
        "At hours 24 and 48 after ICU admission, how well do the labs measured so far predict death in the following 3 days?",
        "On days 1, 2 and 3, predict hospital death within the next 7 days.",
    ],
)
def test_several_prediction_times_with_a_horizon_after_each_are_dynamic(question: str) -> None:
    assert landmark_prediction_framing(question)
    assert _family(question) == ("dynamic_prediction", FAMILY_BASIS_KEYWORD)


@pytest.mark.parametrize(
    "question",
    [
        # One prediction time.
        "在入 ICU 第 24 小时，用到当时为止的生命体征和化验，预测之后 7 天内 ICU 死亡的效果如何？",
        "At 24 hours after ICU admission, predict ICU death over the next 7 days.",
        # One look-back window.
        "用 eICU demo 中入 ICU 前 24 小时的数据建立一个预测 ICU 内死亡的模型，它的区分度和校准怎么样？",
        "Build a prediction model for in-hospital death using data from the first 24 hours after ICU admission.",
        # Listed horizons of one endpoint, from one time.
        "Predict mortality at 24 and 48 hours after ICU admission from admission labs.",
        # Times or a schedule that date a measurement, not the prediction.
        "Use lactate measured at 6 and 24 hours to predict death within the next 28 days.",
        "Every 4 hours glucose was measured; predict death in the next 7 days.",
        "用第 6、24 小时测得的乳酸预测之后 28 天内死亡。",
        "每 4 小时测量一次血糖；预测接下来 7 天内的死亡。",
    ],
)
def test_one_prediction_time_or_window_stays_a_static_model(question: str) -> None:
    assert not landmark_prediction_framing(question)
    assert _family(question)[0] == "prediction_model"


def test_several_times_without_a_prediction_are_not_a_prediction() -> None:
    question = "入 ICU 第 24、48、72 小时的乳酸与之后 7 天内死亡是否相关？"
    assert not landmark_prediction_framing(question)
    assert _family(question)[0] != "dynamic_prediction"
