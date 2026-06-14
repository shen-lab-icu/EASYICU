"""Schema-stability lock for :class:`ResearchContext`.

The agent layer's auditability claim rests on ResearchContext being a
fixed contract between EasyICU exports and the research-agent runtime.
This test fails when anyone adds, removes or renames a field without
bumping ``RESEARCH_CONTEXT_SCHEMA_VERSION``.

Bump-procedure for an intentional schema change:

1. Update ``RESEARCH_CONTEXT_FIELDS`` and the
   ``ResearchContext`` field list in :mod:`schema`.
2. Bump ``RESEARCH_CONTEXT_SCHEMA_VERSION`` (e.g.
   ``"easyicu.research_context/1"`` -> ``"easyicu.research_context/2"``).
3. Update ``EXPECTED_VERSION`` below.
4. If the change is breaking for downstream consumers (paper.py,
   context.py builders, webapp serialization), update them in the
   same commit.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import (
    RESEARCH_CONTEXT_FIELDS,
    RESEARCH_CONTEXT_SCHEMA_VERSION,
    ResearchContext,
)


EXPECTED_VERSION = "easyicu.research_context/1"

EXPECTED_FIELDS = (
    "schema_version",
    "research_question",
    "cohort",
    "variables",
    "time_windows",
    "temporal_constraints",
    "target_outcome",
    "primary_exposure",
    "cross_database_validation",
    "cohort_parquet",
    "user_preferences",
    "notes",
    "created_at",
)


def test_research_context_schema_version_is_locked():
    """If this fails: bump EXPECTED_VERSION here AND in schema.py."""
    assert RESEARCH_CONTEXT_SCHEMA_VERSION == EXPECTED_VERSION
    # The default field on the model must match the constant — otherwise
    # serialized contexts and the manifest will disagree.
    fields = ResearchContext.model_fields
    assert fields["schema_version"].default == EXPECTED_VERSION


def test_research_context_field_set_is_locked():
    """If this fails: a field was added/removed/renamed without a schema bump.

    Either:
    - bump the schema version (intentional change), update
      RESEARCH_CONTEXT_FIELDS and EXPECTED_FIELDS, AND update every
      ResearchContext(...) construction site to match, or
    - revert the field change.
    """
    actual = tuple(ResearchContext.model_fields.keys())
    assert actual == EXPECTED_FIELDS, (
        "ResearchContext field set drifted. "
        f"Expected: {EXPECTED_FIELDS}. "
        f"Got: {actual}. "
        "See the test docstring for the bump procedure."
    )
    assert RESEARCH_CONTEXT_FIELDS == EXPECTED_FIELDS


def test_research_context_is_immutable():
    """``frozen=True`` means downstream code cannot quietly rewrite
    cohort / variables / target_outcome between construction and the
    prompt-rendering site. Legitimate changes must go through
    ``model_copy(update=...)`` so the diff is auditable.
    """
    from easyicu.research_agent.schema import CohortDescriptor

    ctx = ResearchContext(
        research_question="Is age associated with mortality?",
        cohort=CohortDescriptor(
            cohort_name="test_cohort",
            database="miiv",
            n_patients=1,
            n_stays=1,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[],
    )
    with pytest.raises(Exception):  # pydantic ValidationError / TypeError
        ctx.research_question = "different question"

    # model_copy(update={...}) is the supported way to derive a new
    # immutable context — confirm that still works.
    ctx2 = ctx.model_copy(update={"research_question": "different question"})
    assert ctx2.research_question == "different question"
    assert ctx.research_question == "Is age associated with mortality?"
