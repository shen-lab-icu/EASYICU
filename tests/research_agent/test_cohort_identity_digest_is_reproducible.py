"""The receipt's row-identity digest must be reproducible by whoever is asked
to check it.

A real E1 run (batch_20260731 ... full03) died at its very first step on

    ValueError: Loaded cohort identity order/content does not match the host receipt

with the correct 94,458 rows in the correct order.  The host publishes
``row_identity_sha256`` in the execution receipt and calls the receipt's
"counts and digests" integrity checks, but published no way to recompute the
digest, so the generated script invented a canonicalisation ("\\n".join of
str(value)) and was refused for a cohort that was exactly right.  Every step
after it was blocked, so one unpublished recipe cost the whole task.

The counts were always checkable -- ``EASYICU_COHORT_ROWS`` is an integer in
the environment.  The digest was not.  These tests hold the difference closed:
the recipe is now a callable the container can import, and the directive names
that exact callable.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import re

import pandas as pd
import pytest

from easyicu.research_agent.authority.coder_authority import HostCoderAuthority
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    _row_identity_sha256,
    cohort_row_identity_sha256,
)
from easyicu.research_agent.resources.coder import bind_primary_cohort_role

IDENTITY = "stay_id"


def _cohort(tmp_path, values):
    path = tmp_path / "cohort.parquet"
    pd.DataFrame({IDENTITY: list(values)}).to_parquet(path, index=False)
    return path


def _receipt_directive() -> str:
    """The exact attachment the host sends when it publishes the receipt."""

    authority = bind_primary_cohort_role(
        authority=HostCoderAuthority(),
        locked_cohort_payload='{"predicates": []}',
        materialized_execution_payload='{"authoritative_analysis_cohort": {}}',
    )
    return authority.render()


# --- the recipe is one implementation, not two -------------------------------


def test_the_published_helper_returns_what_the_host_wrote_into_the_receipt(tmp_path):
    path = _cohort(tmp_path, [30000153, 30000213, 30000484])

    host_side = _row_identity_sha256(path, identity_column=IDENTITY)
    container_side = cohort_row_identity_sha256(
        pd.read_parquet(path)[IDENTITY].tolist()
    )

    assert container_side == host_side


def test_the_digest_is_ordered_so_the_check_has_something_to_catch():
    forward = cohort_row_identity_sha256([1, 2, 3])
    reversed_ = cohort_row_identity_sha256([3, 2, 1])

    assert forward != reversed_


@pytest.mark.parametrize(
    "guess",
    [
        pytest.param(
            lambda values: hashlib.sha256(
                "\n".join(str(value) for value in values).encode("utf-8")
            ).hexdigest(),
            id="newline_joined_str__the_one_the_real_run_wrote",
        ),
        pytest.param(
            lambda values: hashlib.sha256(
                ",".join(str(value) for value in values).encode("utf-8")
            ).hexdigest(),
            id="comma_joined_str",
        ),
        pytest.param(
            lambda values: hashlib.sha256(
                repr(list(values)).encode("utf-8")
            ).hexdigest(),
            id="repr_of_the_list",
        ),
        pytest.param(
            lambda values: hashlib.sha256(
                pd.Series(list(values)).to_string().encode("utf-8")
            ).hexdigest(),
            id="pandas_to_string",
        ),
    ],
)
def test_no_reasonable_guess_reproduces_the_recipe(guess):
    """Why the recipe has to be published rather than described.

    Each of these is a defensible reading of "hash the identity column".  None
    of them is the host's.  A step that guesses is refused for being right.
    """

    values = [30000153, 30000213, 30000484]

    assert guess(values) != cohort_row_identity_sha256(values)


# --- the directive names the callable, and the name still resolves ------------


def test_the_directive_names_the_helper_it_expects_to_be_called():
    directive = _receipt_directive()

    assert "cohort_row_identity_sha256" in directive


def test_the_directive_says_a_hand_rolled_digest_is_wrong_not_merely_discouraged():
    directive = _receipt_directive()

    assert "never write one" in directive


def test_the_import_line_the_directive_publishes_actually_imports(tmp_path):
    """The anti-drift link: the prompt's import statement is executed here.

    If the helper is renamed, moved, or the sentence is reworded away from the
    real path, this fails -- rather than being discovered by a dead run.
    """

    directive = _receipt_directive()
    match = re.search(
        r"from ([\w.]+) import (cohort_row_identity_sha256)",
        directive.replace("\n", " "),
    )
    assert match is not None, "the directive no longer publishes an import line"

    module = importlib.import_module(match.group(1))
    published = getattr(module, match.group(2))

    path = _cohort(tmp_path, [7, 8, 9])
    assert published(pd.read_parquet(path)[IDENTITY].tolist()) == _row_identity_sha256(
        path, identity_column=IDENTITY
    )


def test_the_example_call_in_the_directive_is_the_call_that_works(tmp_path):
    """The directive shows `cohort_row_identity_sha256(df[col].tolist())`.

    Publishing an example that does not run is the same defect one level down,
    so the example is parsed out and evaluated against a real frame.
    """

    directive = _receipt_directive().replace("\n", " ")
    match = re.search(r"`(cohort_row_identity_sha256\([^`]*\))`", directive)
    assert match is not None, "the directive no longer shows an example call"

    expression = match.group(1)
    ast.parse(expression, mode="eval")

    path = _cohort(tmp_path, [11, 12])
    scope = {
        "cohort_row_identity_sha256": cohort_row_identity_sha256,
        "df": pd.read_parquet(path),
        "identity_column": IDENTITY,
    }
    assert eval(expression, scope) == _row_identity_sha256(  # noqa: S307
        path, identity_column=IDENTITY
    )


# --- publishing the recipe does not publish a way around its guards ----------


def test_a_duplicated_identity_still_fails_closed_through_the_public_helper():
    with pytest.raises(MaterializedMetadataError):
        cohort_row_identity_sha256([1, 1, 2])


def test_a_null_identity_still_fails_closed_through_the_public_helper():
    with pytest.raises(MaterializedMetadataError):
        cohort_row_identity_sha256([1, None, 2])


def test_a_value_that_is_not_canonically_encodable_fails_closed():
    with pytest.raises(MaterializedMetadataError):
        cohort_row_identity_sha256([1, float("nan")])
