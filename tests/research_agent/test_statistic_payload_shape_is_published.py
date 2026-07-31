"""A registered statistic must be written in a shape the host publishes.

The full03 M1 run reached step 10 of 12 -- past its cohort, its Table 1, its
primary adjusted association and two figures -- and then lost the step to

    invalid_statistic_shape in statistic:complete_case_n:
    A registered statistic JSON product was not an object.

The file it wrote was::

    [{"name": "complete_case_n", "value": 445, "n_full": 1000,
      "n_dropped": 555, "definition": "..."}]

which is the right object wrapped in a one-element list -- the same shape the
step writes every table in.  The canonical normalizer requires a bare object.
No host prompt had ever said so, so the shape was learnable only by dying, and
across the whole recorded corpus 16 of 17 registered statistics guessed right
and 1 guessed wrong and paid for it.

These tests hold the reader and the directive together: the example the host
publishes is fed to the real normalizer, and the shape it warns against is fed
to it too.
"""

from __future__ import annotations

import json
import re

import pytest

from easyicu.research_agent.agents.core import (
    _declared_output_scope_contract,
    _declared_statistic_products,
)
from easyicu.research_agent.contracts.result_envelope import (
    STATISTIC_PAYLOAD_KEY_ALIASES,
    StepArtifactRef,
    _DENOMINATOR_KEYS,
    _HIGH_KEYS,
    _LOW_KEYS,
    _NUMERATOR_KEYS,
    _P_VALUE_KEYS,
    _VALUE_KEYS,
    _parse_statistic,
)
from easyicu.research_agent.schema import AnalysisStep

STATISTIC = "statistic:complete_case_n"


def _step(*outputs: str) -> AnalysisStep:
    return AnalysisStep(
        step_id="10_robustness_replay",
        intent="Re-estimate the locked grid.",
        method="robustness_replay",
        expected_outputs=list(outputs),
        inputs=[],
    )


def _directive(*outputs: str) -> str:
    return _declared_output_scope_contract(_step(*outputs))


def _normalize(payload: object) -> tuple[object, list]:
    """Run the real reader over one statistic payload."""

    issues: list = []
    parsed = _parse_statistic(
        product_id=STATISTIC,
        statistic_name="complete_case_n",
        artifact=StepArtifactRef(
            product_id=STATISTIC,
            kind="statistic",
            name="complete_case_n",
            relative_path="complete_case_n.json",
            media_type="application/json",
            sha256="0" * 64,
            byte_size=1,
        ),
        artifact_bytes=json.dumps(payload).encode("utf-8"),
        receipts=[],
        issues=issues,
    )
    return parsed, issues


# --- the rule reaches exactly the steps it governs ---------------------------


def test_a_step_declaring_a_statistic_is_told_the_shape():
    assert "invalid_statistic_shape" in _directive("table:x", STATISTIC)


def test_a_step_declaring_no_statistic_is_not_told_about_one():
    assert "invalid_statistic_shape" not in _directive("table:table_one")


def test_every_declared_statistic_is_named_not_just_the_first():
    directive = _directive(STATISTIC, "statistic:primary_or")

    assert STATISTIC in directive
    assert "statistic:primary_or" in directive


def test_the_products_helper_reads_declarations_not_prose():
    step = _step("table:x", STATISTIC, STATISTIC, "figure:f")

    assert _declared_statistic_products(step) == (STATISTIC,)


# --- the published example is the shape the reader accepts -------------------


def test_the_example_the_host_publishes_is_accepted_by_the_real_reader():
    directive = _directive(STATISTIC)
    match = re.search(r"`(\{[^`]*\})`", directive)
    assert match is not None, "the directive no longer shows a JSON example"

    example = (
        match.group(1).replace("<name>", "complete_case_n").replace("<number>", "445")
    )
    parsed, issues = _normalize(json.loads(example))

    assert not issues
    assert parsed is not None


def test_the_shape_the_host_warns_against_really_is_refused():
    """Why the warning is worth its bytes -- this is the M1 file."""

    parsed, issues = _normalize(
        [
            {
                "name": "complete_case_n",
                "value": 445,
                "n_full": 1000,
                "n_dropped": 555,
            }
        ]
    )

    assert parsed is None
    assert [issue.code for issue in issues] == ["invalid_statistic_shape"]


def test_the_same_content_as_an_object_is_accepted():
    """The refusal is about the wrapper, not about the extra keys."""

    parsed, issues = _normalize(
        {
            "name": "complete_case_n",
            "value": 445,
            "n_full": 1000,
            "n_dropped": 555,
        }
    )

    assert not issues
    assert parsed is not None
    assert parsed.value == 445


def test_a_bare_number_is_refused_as_the_directive_says():
    parsed, issues = _normalize(445)

    assert parsed is None
    assert [issue.code for issue in issues] == ["invalid_statistic_shape"]


# --- the aliases are the reader's own, not a retelling of them ---------------


@pytest.mark.parametrize(
    "key",
    sorted(
        {
            key
            for keys in (
                _VALUE_KEYS,
                _LOW_KEYS,
                _HIGH_KEYS,
                _P_VALUE_KEYS,
                _NUMERATOR_KEYS,
                _DENOMINATOR_KEYS,
            )
            for key in keys
        }
    ),
)
def test_every_key_the_reader_accepts_appears_in_the_directive(key):
    """Anchored on the reader's own tuples, not on the published mapping.

    Deriving the cases from ``STATISTIC_PAYLOAD_KEY_ALIASES`` made this test
    agree with itself: dropping a family from the mapping dropped the cases
    that would have caught it, and the mutation survived.  The tuples below are
    what ``_parse_statistic`` actually consults, so unpublishing a family now
    fails here instead of quietly shrinking the suite.
    """

    assert key in _directive(STATISTIC)


def test_the_published_mapping_covers_every_field_the_reader_reads():
    published = {key for keys in STATISTIC_PAYLOAD_KEY_ALIASES.values() for key in keys}
    consulted = {
        key
        for keys in (
            _VALUE_KEYS,
            _LOW_KEYS,
            _HIGH_KEYS,
            _P_VALUE_KEYS,
            _NUMERATOR_KEYS,
            _DENOMINATOR_KEYS,
        )
        for key in keys
    }

    assert published == consulted


def test_a_name_that_contradicts_the_declared_product_is_refused():
    """The directive says an included name must match; the reader agrees."""

    parsed, issues = _normalize({"name": "something_else", "value": 1})

    assert parsed is None
    assert [issue.code for issue in issues] == ["conflicting_statistic_identity"]
