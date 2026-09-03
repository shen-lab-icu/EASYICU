"""Schema-stability lock for :class:`ResearchContext`.

The agent layer's auditability claim rests on ResearchContext being a
fixed contract between EasyICU exports and the research-agent runtime.
This test fails when anyone adds, removes or renames a field without
bumping ``RESEARCH_CONTEXT_SCHEMA_VERSION``.

Bump-procedure for an intentional schema change:

1. Update ``RESEARCH_CONTEXT_FIELDS`` and the
   ``ResearchContext`` field list in :mod:`schema`.
2. Bump ``RESEARCH_CONTEXT_SCHEMA_VERSION`` **when the change can make an
   existing stored payload invalid or silently change its meaning** -- a
   removed field, a renamed field, a new required field, a narrowed type, a
   changed default. Update ``EXPECTED_VERSION`` below in the same edit.
3. Do **not** bump for a purely additive optional field with a default: no
   stored payload becomes invalid.
   ``test_literal_v1_payload_roundtrips_without_rewrite`` asserts that an
   archived payload still parses and no key it carried is rewritten.
4. A change that narrows a typed version's legal domain requires a new typed
   model and version. Keep the archived model parseable, add an explicit
   fail-closed migration when current execution needs the new facts, and make
   new builders write only the new version. V2 -> V3 window-role binding is the
   worked example.
5. If the change is breaking for downstream consumers (paper.py,
   context.py builders, webapp serialization), update them in the
   same commit.

``endpoint`` (2026-07-29, task O1) is the worked example of case 3: optional,
defaulted to ``None``, added so the study's endpoint can be *declared* instead
of guessed from a column-name suffix. No payload written before it becomes
invalid.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import (
    RESEARCH_CONTEXT_FIELDS,
    RESEARCH_CONTEXT_SCHEMA_VERSION,
    ResearchContext,
)
from easyicu.research_agent.research_context.typed import (
    RESEARCH_CONTEXT_V2_SCHEMA_VERSION,
    RESEARCH_CONTEXT_V3_SCHEMA_VERSION,
    ResearchContextV2,
    ResearchContextV3,
    parse_research_context,
    parse_research_context_json,
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
    "endpoint",
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


def test_literal_v1_payload_roundtrips_without_rewrite():
    payload = {
        "schema_version": "easyicu.research_context/1",
        "research_question": "Is age associated with mortality?",
        "cohort": {
            "cohort_name": "archived",
            "database": "miiv",
            "n_patients": 1,
            "n_stays": 1,
            "inclusion_criteria": [],
            "exclusion_criteria": [],
            "id_columns": ["stay_id"],
            "time_columns": [],
            "outcome_columns": ["death"],
            "provenance": {},
            "notes": None,
        },
        "variables": [],
        "time_windows": [],
        "temporal_constraints": [],
        "target_outcome": "death",
        "primary_exposure": "age",
        "cross_database_validation": [],
        "cohort_parquet": None,
        "user_preferences": None,
        "notes": None,
        "created_at": "2026-01-01T00:00:00Z",
    }
    raw = __import__("json").dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    parsed = parse_research_context_json(raw)
    assert type(parsed) is ResearchContext
    dumped = parsed.model_dump(mode="json")

    # The payload above is deliberately frozen at its archived shape and must
    # not be edited when a field is added. "Without rewrite" means every key the
    # archive carried keeps its exact value.
    assert {key: dumped[key] for key in payload} == payload

    # A purely additive optional field may appear. It must appear as None: a
    # field that materialised a *value* while reading an archive would mean the
    # reader inferred something the archive never said -- which for ``endpoint``
    # is the precise failure it was introduced to end.
    added = set(dumped) - set(payload)
    assert added == {"endpoint"}
    assert all(dumped[key] is None for key in added)

    assert "materialized_inputs" not in dumped


def test_unknown_research_context_version_is_rejected():
    with pytest.raises(ValueError, match="unsupported research context schema"):
        parse_research_context({"schema_version": "easyicu.research_context/999"})


def test_research_context_json_rejects_duplicate_keys_and_nonfinite_values():
    with pytest.raises(ValueError, match="duplicate research context JSON key"):
        parse_research_context_json(
            '{"schema_version":"easyicu.research_context/1",'
            '"schema_version":"easyicu.research_context/1"}'
        )
    with pytest.raises(ValueError, match="non-finite research context JSON"):
        parse_research_context_json(
            '{"schema_version":"easyicu.research_context/1","notes":NaN}'
        )


def test_v1_payload_cannot_smuggle_v2_fields():
    with pytest.raises(ValueError):
        parse_research_context(
            {
                "schema_version": "easyicu.research_context/1",
                "materialized_inputs": {},
            }
        )


def test_typed_research_context_versions_and_field_sets_are_locked():
    assert RESEARCH_CONTEXT_V2_SCHEMA_VERSION == "easyicu.research_context/2"
    assert RESEARCH_CONTEXT_V3_SCHEMA_VERSION == "easyicu.research_context/3"
    assert tuple(ResearchContext.model_fields) == EXPECTED_FIELDS
    assert tuple(ResearchContextV2.model_fields) == (
        *EXPECTED_FIELDS,
        "materialized_inputs",
    )
    assert tuple(ResearchContextV3.model_fields) == (
        *EXPECTED_FIELDS,
        "materialized_inputs",
    )
    assert (
        ResearchContextV2.model_fields["schema_version"].default
        == RESEARCH_CONTEXT_V2_SCHEMA_VERSION
    )
    assert (
        ResearchContextV3.model_fields["schema_version"].default
        == RESEARCH_CONTEXT_V3_SCHEMA_VERSION
    )
