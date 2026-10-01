"""An event that a follow-up handling clause lists is not a requested endpoint.

A question that says how follow-up ends ("处理死亡、出院与删失", "accounting for
death as a competing risk", "censored at death") lists censoring and competing
events.  The question reader used to read each such event as an endpoint the
researcher asks to analyse: a 90-day mortality question also requested generic
in-hospital death, and a length-of-stay question was read as a mortality
question.  Synthetic questions; none is a benchmark item.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("评估血管活性药物与90天死亡的关联；处理死亡、出院与删失。", ("mort_90d",)),
        (
            "Is early fluid balance associated with ICU length of stay, accounting for death "
            "as a competing risk?",
            ("los_icu",),
        ),
        (
            "Is lactate associated with ICU readmission, with patients censored at death or "
            "hospital discharge?",
            ("icu_readmission",),
        ),
        (
            "Is lactate associated with ICU readmission? Handle death and discharge as "
            "competing events.",
            ("icu_readmission",),
        ),
        (
            "Is lactate associated with ICU readmission? We handle death, discharge and "
            "censoring explicitly.",
            ("icu_readmission",),
        ),
        ("乳酸与 ICU 住院时长的关系，考虑死亡的竞争风险", ("los_icu",)),
        ("乳酸与 ICU 再入院的关系，死亡作为竞争事件", ("icu_readmission",)),
    ],
)
def test_events_that_end_follow_up_are_not_requested_endpoints(question, expected):
    assert study_intent.explicit_outcome_concepts(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is RRT associated with time to death or censoring at day 90?", ("death",)),
        ("Is RRT associated with time to death, censoring at hospital discharge?", ("death",)),
        ("Is RRT associated with 28-day mortality, with discharge as a censoring event?", ("mort_28d",)),
        ("Patients are censored at discharge and death is the primary outcome.", ("death",)),
        ("以28天死亡为主要结局，并在出院时删失", ("mort_28d",)),
        ("处理缺失数据后，研究院内死亡", ("death",)),
    ],
)
def test_an_endpoint_named_beside_its_censoring_rule_is_still_read(question, expected):
    assert study_intent.explicit_outcome_concepts(question) == expected


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        (
            "Is early fluid balance associated with ICU length of stay, accounting for death "
            "as a competing risk?",
            "fluid_balance",
            "los_icu",
        ),
        ("处理死亡与删失后，评估乳酸与 ICU 住院时长的关系", "lact", "los_icu"),
        ("Censored at death or discharge, is lactate associated with ICU readmission?", "lact", "icu_readmission"),
        ("乳酸与 ICU 再入院的关系，死亡作为竞争事件", "lact", "icu_readmission"),
    ],
)
def test_the_question_slots_skip_a_handled_event(question, exposure, outcome):
    slots = study_intent.deterministic_intent(question)["slots"]

    assert (slots["exposure"]["value"], slots["outcome"]["value"]) == (exposure, outcome)


def test_a_long_conjunction_chain_is_read_in_bounded_time():
    # Nested list quantifiers once took minutes on such a question; the reader
    # must stay linear in the length of what it is given.
    import subprocess
    import sys

    program = (
        "from easyicu.webserver import study_intent as s\n"
        "for q in ('death ' + 'and x ' * 80, '死亡' + '和甲乙' * 80,\n"
        "          'handle death' + ', x and y' * 60, 'death or ' * 100):\n"
        "    assert s.explicit_outcome_concepts(q) == ('death',), q[:20]\n"
        "    s.deterministic_intent(q)\n"
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)
