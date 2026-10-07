"""A horizon in hours is a stated horizon.

The horizons a question states for death or survival were read in days,
weeks, months and years only.  "48-hour mortality" stated no horizon, so the
context builder read the question as leaving mortality unspecified: the
in-hospital death concept kept its definition, the study was planned for
in-hospital mortality, and review passed the endpoint as resolved.  A Chinese
numeral was read from its last character ("二十八天死亡" as 8 days), and a
horizon an exclusion names ("deaths within 24 hours were excluded") was read
as the endpoint's.

Hours are now read next to a mortality, death or survival noun, so a request
for 48-hour mortality against in-hospital death is an endpoint-definition
conflict that review blocks.  A stated horizon stays part of the request when
the question or the outcome column also names a setting ("48-hour in-hospital
mortality"), and its hours are not read as the exposure window.  Hours that
name a landmark ("alive at 24 hours") and a horizon an exclusion names are
not read.  Synthetic texts and cohorts only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.outcome_availability import (
    StatedHorizon,
    fixed_horizon_mortality_endpoint_stated_by,
    mortality_horizon_spans,
    stated_mortality_horizons,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.scientific_review import _endpoint_resolved
from easyicu.research_agent.research_context.builder import (
    _enrich_target_outcome_descriptor,
    build_research_context,
)
from easyicu.research_agent.schema import ConceptDescriptor


@pytest.mark.parametrize(
    ("text", "stated"),
    [
        ("48-hour mortality", ["48-hour"]),
        ("72 h in-hospital mortality", ["72-hour"]),
        ("48h mortality", ["48-hour"]),
        ("24-hour survival", ["24-hour"]),
        ("mortality within 48 hours", ["48-hour"]),
        ("death within 24 h of ICU admission", ["24-hour"]),
        ("survival to 24 hours", ["24-hour"]),
        ("48 小时内死亡", ["48-hour"]),
        ("48 小时院内死亡率", ["48-hour"]),
        ("四十八小时病死率", ["48-hour"]),
        # A Chinese numeral is read whole.
        ("二十八天死亡率", ["28-day"]),
        ("九十天全因死亡", ["90-day"]),
        ("三百六十五天生存", ["365-day"]),
        # Never the tail of a numeral it cannot read.
        ("一百零五天死亡率", []),
        # Hours that qualify an exposure or name a landmark state no horizon.
        ("first-6h vital signs and in-hospital mortality", []),
        ("a 24-hour landmark survival analysis", []),
        ("patients alive at 24 hours", []),
        ("仅纳入 24 小时时仍存活的患者", []),
        ("报告 24 小时内早死的人数", []),
    ],
)
def test_the_horizons_a_question_states_in_hours(text, stated):
    assert [horizon.adjective for horizon in stated_mortality_horizons(text)] == stated


@pytest.mark.parametrize(
    ("text", "stated"),
    [
        ("patients who died within 24 hours were excluded", []),
        ("deaths within the first 24 hours were excluded", []),
        ("排除入 ICU 后 24 小时内死亡的患者", []),
        ("24 小时内死亡者被排除", []),
        ("excluding deaths within 24 hours, 28-day mortality", ["28-day"]),
        ("28-day mortality excluding deaths within 24 hours", ["28-day"]),
        ("Deaths within 24 hours of admission were excluded and 90-day mortality was the outcome", ["90-day"]),
        ("排除 24 小时内死亡的患者后的 28 天死亡", ["28-day"]),
        ("patients who died within 2 days were excluded; 28-day mortality", ["28-day"]),
        # An exclusion of something else leaves the endpoint's horizon.
        ("Excluding readmissions, is the exposure associated with 48-hour mortality?", ["48-hour"]),
        ("Is the exposure associated with 48-hour mortality after excluding readmissions?", ["48-hour"]),
        ("In stays excluding transfers is the exposure associated with 48-hour mortality?", ["48-hour"]),
    ],
)
def test_a_horizon_an_exclusion_names_is_the_cohorts(text, stated):
    assert [horizon.adjective for horizon in stated_mortality_horizons(text)] == stated


def test_every_horizon_phrase_is_located_read_or_excluded():
    text = "excluding deaths within 24 hours, 28-day mortality"

    assert [text[start:end] for start, end in mortality_horizon_spans(text)] == [
        "deaths within 24 hours",
        "28-day mortality",
    ]


def test_an_hour_horizon_admits_only_the_endpoint_of_that_many_hours():
    horizon = StatedHorizon(count=48, unit="hour")

    assert (horizon.adjective, horizon.noun, horizon.semantic_key) == ("48-hour", "48 hours", "mortality_48h")
    assert horizon.admits(2) and not horizon.admits(1) and not horizon.admits(28)
    # 36 hours is no day-long endpoint.
    assert not StatedHorizon(count=36, unit="hour").admits(1)
    assert fixed_horizon_mortality_endpoint_stated_by(horizon) is None
    assert fixed_horizon_mortality_endpoint_stated_by(StatedHorizon(count=672, unit="hour")).event_concept == "mort_28d"


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
        ("Can the first-6h heart rate discriminate 48-hour mortality in adult ICU stays?", True),
        ("Is heart rate associated with death within 24 hours of ICU admission?", True),
        ("首 6 小时心率能否区分 48 小时内死亡？", True),
        # A setting does not absorb the horizon stated with it.
        ("Is heart rate associated with 48-hour in-hospital mortality?", True),
        ("Is heart rate associated with in-hospital mortality within 48 hours?", True),
        ("心率与 48 小时院内死亡是否相关？", True),
        ("Is heart rate associated with 28-day in-hospital mortality?", True),
        ("Can the first-6h heart rate discriminate in-hospital mortality in adult ICU stays?", False),
        ("Among stays alive at 24 hours, is heart rate associated with in-hospital mortality?", False),
    ],
)
def test_an_hour_horizon_is_not_in_hospital_death(question, conflict):
    context = _context(question)
    descriptor = context.variable("death")

    # The owner's definition stands; a request for another horizon conflicts with it.
    assert descriptor.description == "in hospital mortality"
    assert any("Endpoint-definition conflict" in note for note in descriptor.clinical_caveats) is conflict
    assert _endpoint_resolved(context) is not conflict


@pytest.mark.parametrize(
    ("concept", "definition", "question", "conflict"),
    [
        # An outcome column that names a setting does not answer a stated horizon.
        ("hospital_death", "In-hospital death", "48-hour mortality?", True),
        ("hospital_death", "In-hospital death", "in-hospital mortality?", False),
        # Mortality in a setting at a horizon is not all-cause mortality at it.
        ("mort_28d", "28-day Mortality", "28-day in-hospital mortality?", True),
        ("mort_28d", "28-day Mortality", "28-day mortality?", False),
    ],
)
def test_a_setting_and_a_horizon_are_one_request(concept, definition, question, conflict):
    descriptor = ConceptDescriptor(name=concept, dtype="int64", source_concept=concept, description=definition)

    _enrich_target_outcome_descriptor(
        descriptors=[descriptor], research_question="Is heart rate associated with " + question,
        target_outcome=concept,
    )

    assert descriptor.description == definition
    assert any("Endpoint-definition conflict" in note for note in descriptor.clinical_caveats) is conflict


def test_a_long_question_with_hours_is_read_in_bounded_time():
    import subprocess
    import sys

    program = (
        "from easyicu.outcome_availability import stated_mortality_horizons as read\n"
        "for q in ('48' + ' ' * 20000 + 'h', 'mortality ' + 'within ' * 5000, '排除' * 8000 + '24 小时内死亡',\n"
        "          'deaths within 24 hours' + ' were' * 5000, 'excluding ' * 5000 + '48-hour mortality',\n"
        "          '二十' * 8000 + '天死亡', '48-hour ' + 'icu ' * 4000 + 'mortality'):\n"
        "    read(q)\n"
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)
