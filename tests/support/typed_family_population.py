"""Shared synthetic fixtures; no recorded research inputs."""
from __future__ import annotations

import json
from tests.research_agent.planning.family_spec_fixtures import _descriptive_context, _descriptive_payload

from tests.support.distribution_policies import EXCLUDE_MISSING

QUESTION = "Adults age 18 or older, first ICU within each hospitalization, described by group."


def context():
    original = _descriptive_context()
    constraints = json.loads(original.user_preferences.data_constraints)
    constraints["cohort"] = {"age_min": 18}
    return original.model_copy(update={
        "research_question": QUESTION,
        "cohort": original.cohort.model_copy(update={"database": "eicu_demo"}),
        "user_preferences": original.user_preferences.model_copy(update={
            "data_constraints": json.dumps(constraints),
        }),
    })


def typed_population():
    return {"criteria": [], "inclusion": [], "exclusion": [], "population_spec": {
        "criteria": [
            {"id": "c1", "kind": "age_years", "source": "question", "role": "include",
             "quote": "age 18 or older", "min_years": 18, "max_years": None},
            {"id": "c2", "kind": "first_icu_stay_per_hospitalization",
             "source": "question", "role": "include",
             "quote": "first ICU within each hospitalization"},
        ],
    }}


def payload(request, population=None):
    response = _descriptive_payload(request, baseline_variables=["age"])
    response["distribution_missing_policies"] = dict(EXCLUDE_MISSING)
    response["population"] = typed_population() if population is None else population
    return response
