"""Every horizon a question states for death is read, and only the endpoint's.

The stated-horizon reader missed horizons a question states with a verb of
dying ("died within 48 hours of ICU admission"), with "in the first"
("mortality in the first 48 hours") or in a list ("48-hour and 28-day
mortality", "mortality at 24 and 48 hours", "28 天和 90 天死亡率"), where only the
horizon next to the endpoint word was read.  A question with an unread horizon
was planned for whole-stay in-hospital mortality with no finding, or for one
horizon of the list.  A labelled list of exclusion criteria ("Exclusion
criteria: death within 24 hours ... Outcome: in-hospital mortality.") named
the endpoint's horizon instead of the cohort's, and "patients studied within
28 days" read a 28-day horizon from "studied".

Each listed horizon is now read at its own place, a horizon a criteria list
names is the cohort's, and a landmark ("did not die within 24 hours") still
states none.  Synthetic texts and cohorts only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.outcome_availability import (
    mortality_horizon_spans,
    stated_mortality_horizon_mentions,
    stated_mortality_horizons,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.scientific_review import _endpoint_resolved
from easyicu.research_agent.research_context.builder import build_research_context


def _read(text: str) -> list[str]:
    return [horizon.adjective for horizon in stated_mortality_horizons(text)]


@pytest.mark.parametrize(
    ("text", "stated"),
    [
        # A verb of dying states a horizon in hours as in days.
        ("Is lactate associated with having died within 48 hours of ICU admission?", ["48-hour"]),
        ("Which patients die within 48 hours of ICU admission?", ["48-hour"]),
        ("Does early lactate predict dying within 72 hours?", ["72-hour"]),
        ("Which patients die within 28 days?", ["28-day"]),
        # "In the first" relates an endpoint to its horizon.
        ("Is ferritin associated with mortality in the first 48 hours?", ["48-hour"]),
        ("Is ferritin associated with death during the first 7 days?", ["7-day"]),
        # Every horizon of a list.
        ("Is SOFA associated with 48-hour and 28-day mortality?", ["48-hour", "28-day"]),
        ("Is SOFA associated with 24- and 48-hour mortality?", ["24-hour", "48-hour"]),
        ("Is SOFA associated with 28-day and 90-day mortality?", ["28-day", "90-day"]),
        ("Is SOFA associated with 28-, 90- and 180-day mortality?", ["28-day", "90-day", "180-day"]),
        ("Is SOFA associated with one- and two-year survival?", ["1-year", "2-year"]),
        ("Is SOFA associated with mortality at 28 and 90 days?", ["28-day", "90-day"]),
        ("Is SOFA associated with mortality at 24, 48 and 72 hours?", ["24-hour", "48-hour", "72-hour"]),
        ("Is SOFA associated with mortality at 28 days, 90 days, and one year?", ["28-day", "90-day", "1-year"]),
        ("Is SOFA associated with mortality at 48 hours and 28 days?", ["48-hour", "28-day"]),
        ("Is SOFA associated with survival to day 28 and day 90?", ["28-day", "90-day"]),
        ("SOFA 与 28 天和 90 天死亡率的关系", ["28-day", "90-day"]),
        ("SOFA 与 24 小时和 48 小时死亡的关系", ["24-hour", "48-hour"]),
        ("SOFA 与 28、90 天死亡率的关系", ["28-day", "90-day"]),
        ("SOFA 与 24 小时和 28 天死亡率的关系", ["24-hour", "28-day"]),
    ],
)
def test_a_horizon_stated_with_a_verb_a_first_period_or_a_list_is_read(text, stated):
    assert _read(text) == stated


@pytest.mark.parametrize(
    ("text", "stated"),
    [
        # A landmark states no horizon.
        ("Among patients who had not died within 24 hours, is lactate associated with 28-day mortality?", ["28-day"]),
        ("patients who did not die within 24 hours", []),
        ("Among those who never died within 24 hours", []),
        ("Among patients alive at 24 hours, is lactate associated with in-hospital mortality?", []),
        ("Among patients who survived the first 24 hours, is lactate associated with in-hospital mortality?", []),
        # Days and hours that are no endpoint's.
        ("Is there an association between lactate within 24 hours and 28-day mortality?", ["28-day"]),
        ("Is lactate clearance in the first 6 hours associated with 28-day mortality?", ["28-day"]),
        ("Is lactate associated with mortality within 28 days, 18 years or older?", ["28-day"]),
        ("Is age 65 and 28-day mortality related?", ["28-day"]),
        ("SOFA评分2和28天死亡的关系", ["28-day"]),
        ("入ICU第1天、28天死亡", ["28-day"]),
        ("24小时乳酸和28天死亡率的关系", ["28-day"]),
        ("patients studied within 28 days of admission", []),
        ("Is mortality in 2019 different from 2020?", []),
    ],
)
def test_a_landmark_or_another_variables_time_states_no_horizon(text, stated):
    assert _read(text) == stated


@pytest.mark.parametrize(
    ("text", "stated"),
    [
        ("Exclusion criteria: death within 24 hours of ICU admission. Outcome: in-hospital mortality.", []),
        ("Exclusion criteria: age under 18, death within 24 hours of ICU admission; Outcome: in-hospital mortality.", []),
        ("Exclusion criteria: death within 24 hours. Outcome: 28-day mortality.", ["28-day"]),
        ("Exclusion criteria:\n- death within 24 hours\n- age < 18\nOutcome: 28-day mortality", ["28-day"]),
        ("Exclusion criteria:\n- death within 24 hours\n\nIs lactate associated with 48-hour mortality?", ["48-hour"]),
        ("Exclusion criteria: death within 24 hours of ICU admission, age under 18. "
         "Is lactate associated with 90-day mortality?", ["90-day"]),
        ("Inclusion criteria: adults. Exclusion criteria: none. Primary outcome: 90-day mortality", ["90-day"]),
        ("排除标准：入ICU 24小时内死亡。结局：院内死亡。", []),
        ("排除标准：24小时内死亡、年龄<18岁；结局：28天死亡", ["28-day"]),
        ("研究乳酸与28天死亡的关系，排除标准：24小时内死亡", ["28-day"]),
        ("Patients who died within 24 hours of admission were not included.", []),
        ("Patients dying within 48 hours of ICU admission were not eligible.", []),
        ("The exclusion of patients who died within 24 hours; outcome 28-day mortality", ["28-day"]),
        ("excluding deaths within 48 hours and 7 days", []),
        # A label that lists no criteria governs nothing.
        ("Outcomes to include: 28-day mortality", ["28-day"]),
        ("Primary outcome: death within 48 hours of ICU admission; exclusion: age < 18", ["48-hour"]),
    ],
)
def test_a_horizon_an_exclusion_or_a_criteria_list_names_is_the_cohorts(text, stated):
    assert _read(text) == stated


@pytest.mark.parametrize(
    ("text", "places"),
    [
        ("Is SOFA associated with 28-day and 90-day mortality?", ["28-day", "90-day mortality"]),
        ("Is SOFA associated with 28- and 90-day mortality?", ["28", "90-day mortality"]),
        ("Is SOFA associated with mortality at 24 and 48 hours?", ["mortality at 24", "48 hours"]),
        ("SOFA 与 28 天和 90 天死亡率的关系", ["28 天", "90 天死亡"]),
    ],
)
def test_each_listed_horizon_has_its_own_place_and_one_covers_the_endpoint(text, places):
    assert [text[mention.start:mention.end] for mention in stated_mortality_horizon_mentions(text)] == places


def test_a_list_is_located_whole_read_or_excluded():
    text = "Excluding deaths within 24 and 48 hours, is SOFA associated with 28- and 90-day mortality?"

    assert [text[start:end] for start, end in mortality_horizon_spans(text)] == [
        "deaths within 24 and 48 hours",
        "28- and 90-day mortality",
    ]
    assert _read(text) == ["28-day", "90-day"]


def _context(question: str):
    built = build_research_context(
        research_question=question,
        cohort=pd.DataFrame({"stay_id": [1, 2, 3, 4], "hr": [80.0, 120.0, 95.0, 130.0], "death": [0, 1, 0, 1]}),
        cohort_name="synthetic", database="synthetic", target_outcome="death", primary_exposure="hr",
        endpoint=EndpointSpec(name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
    )
    return getattr(built, "context", built)


@pytest.mark.parametrize(
    ("question", "conflict"),
    [
        # A request for mortality within 48 hours is not in-hospital death.
        ("Is heart rate associated with having died within 48 hours of ICU admission?", True),
        ("Is heart rate associated with mortality in the first 48 hours?", True),
        ("Is heart rate associated with mortality at 24 and 48 hours?", True),
        # The criteria list's 24 hours are the cohort's: in-hospital death answers it.
        ("Exclusion criteria: death within 24 hours of ICU admission. "
         "Outcome: in-hospital mortality. Is heart rate associated with it?", False),
        ("排除标准：入ICU 24小时内死亡。结局：院内死亡。心率与其是否相关？", False),
    ],
)
def test_the_request_holds_the_horizons_the_question_states(question, conflict):
    context = _context(question)
    descriptor = context.variable("death")

    assert descriptor.description == "in hospital mortality"
    assert any("Endpoint-definition conflict" in note for note in descriptor.clinical_caveats) is conflict
    assert _endpoint_resolved(context) is not conflict


def test_a_long_list_or_label_run_is_read_in_bounded_time():
    import subprocess
    import sys

    program = (
        "from easyicu.outcome_availability import stated_mortality_horizons as read, mortality_horizon_spans as spans\n"
        "for q in ('28- and ' * 5000, '28-day and ' * 5000 + '90-day mortality', 'mortality at ' + '28 and ' * 5000,\n"
        "          'mortality at 28 days' + ' and 90 days' * 5000, 'mortality at ' + '28, ' * 5000 + 'and 90 days',\n"
        "          '28、' * 8000 + '90天死亡', '28天和' * 8000, '28-day mortality, ' * 5000,\n"
        "          'Exclusion criteria: ' * 5000 + 'death within 24 hours', '.' + ' ' * 20000 + 'Exclusion:',\n"
        "          ':' * 20000, 'not died within 24 hours ' * 3000, '28' + ' ' * 20000 + 'and 90-day mortality'):\n"
        "    read(q); spans(q)\n"
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)
