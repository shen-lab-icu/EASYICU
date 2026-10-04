"""A question's stated horizon names its fixed-horizon endpoint.

The question reader knew two fixed-horizon phrasings, "28-day mortality" and
"90-day mortality".  "1-year mortality", "1 年死亡率" and "随访一年" were read
as generic death, and a survival question that states its horizon without the
word mortality ("survival to day 28", "one-year survival") named no endpoint at
all.  The reader now takes the horizon from the vocabulary the context builder
compares endpoints with: a stated horizon names the one closed endpoint it
admits, and a horizon that none admits ("30-day mortality") stays generic
mortality.  Synthetic questions; none is a benchmark item.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is early RRT associated with 1-year mortality?", ("mort_365d",)),
        ("Is early RRT associated with one-year survival in a Cox model?", ("mort_365d",)),
        ("早期 RRT 与 1 年死亡率的关系", ("mort_365d",)),
        ("早期 RRT 与随访一年的生存是否相关？", ("mort_365d",)),
        ("Is early RRT associated with survival to day 28?", ("mort_28d",)),
        ("Is early RRT associated with 3-month mortality?", ("mort_90d",)),
        ("Is early RRT associated with 90-day mortality?", ("mort_90d",)),
        ("Is early RRT associated with 28-day mortality and one-year survival?", ("mort_28d", "mort_365d")),
        # No closed endpoint has these horizons.
        ("Is early RRT associated with 30-day mortality?", ("death",)),
        ("Is early RRT associated with 6-month mortality?", ("death",)),
    ],
)
def test_a_stated_horizon_names_the_closed_endpoint_it_admits(question, expected):
    slots = study_intent.deterministic_intent(question)["slots"]

    assert study_intent.explicit_outcome_concepts(question) == expected
    assert (slots["exposure"]["value"], slots["outcome"]["value"]) == ("rrt", expected[0])


def test_a_negated_horizon_names_no_endpoint():
    question = "Is early RRT associated with 90-day mortality rather than 1-year mortality?"

    assert study_intent.explicit_outcome_concepts(question) == ("mort_90d",)
