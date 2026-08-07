"""The replay path looked for a filename no producer has ever written.

Robustness variants may be re-estimated only by replaying the primary model's
exact registered script, and that replay needs the term-level coefficients.
:func:`_find_structured_primary_model_source` finds them through the model
step's summary -- and it read one key, ``diagnostic_companions.coefficients``,
falling back to a fixed ``coefficients.csv`` when the key was absent.

Measured across every recorded run: 334 step summaries, 23 of them carrying
``model_contracts``.  ``coefficient_table`` appears 10 times (the deterministic
association owner writes it), ``coefficient_file`` 3 times (a Coder-written
summary), ``diagnostic_companions`` once, and a file named ``coefficients.csv``
exists zero times.  So the reader resolved 1 of 23; in the other 22 runs the
chain hit ``continue``, ``structured_source`` was ``None``, and the robustness
step blocked with "A completed primary estimate with point estimate and
confidence interval is required before robustness comparison" -- while the
estimate sat in the sibling file the summary did name.

That is the same defect as the cohort spelling: a reader that does not read
what the writers write.  The keys are now published once and the invented
default is gone -- a filename nobody declared must never be guessed, because a
guess that happens to exist would bind the replay to a table the plan never
promised.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.result_envelope import (
    MODEL_SUMMARY_COEFFICIENT_TABLE_KEYS,
    model_summary_coefficient_filename,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    _COEFFICIENT_FILENAME,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _coefficient_filename_from_summary,
)

# Written out rather than derived from the tuple under test.
_KEYS_PRODUCERS_ACTUALLY_WRITE = (
    "diagnostic_companions.coefficients",
    "coefficient_table",
    "coefficient_file",
)


def _owner_summary(out_dir) -> dict:
    """Fit the real model and return the summary the OWNER wrote."""

    rng = np.random.default_rng(20260731)
    n = 3000
    exposure = rng.integers(0, 2, n).astype(float)
    age = rng.normal(65.0, 12.0, n)
    charlson = rng.integers(0, 8, n).astype(float)
    logit = -3.0 + 1.4 * exposure + 0.03 * (age - 65.0) + 0.15 * charlson
    frame = pd.DataFrame(
        {
            "exposure": exposure,
            "age": age,
            "charlson": charlson,
            "death": (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float),
        }
    )

    previous = os.environ.get("STEP_OUT_DIR")
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    try:
        return run_adjusted_association_from_env(
            frame=frame,
            cohort_path=None,
            requirement_id="primary_logistic",
            exposure="exposure",
            outcome="death",
            covariates=["age", "charlson"],
            estimator_kind="logistic",
            analysis_set="source_aware",
            analysis_role="primary",
            method_family="binary_logistic_regression",
            model_terms=[
                {
                    "name": "exposure",
                    "role": "exposure",
                    "coding": "binary",
                    "levels": ["0", "1"],
                    "reference_level": "0",
                    "transform": "treatment_contrast",
                },
                {
                    "name": "age",
                    "role": "covariate",
                    "coding": "continuous",
                    "transform": "identity",
                },
                {
                    "name": "charlson",
                    "role": "covariate",
                    "coding": "continuous",
                    "transform": "identity",
                },
            ],
        )
    finally:
        if previous is None:
            os.environ.pop("STEP_OUT_DIR", None)
        else:
            os.environ["STEP_OUT_DIR"] = previous


def test_the_published_keys_are_the_ones_producers_write():
    assert MODEL_SUMMARY_COEFFICIENT_TABLE_KEYS == _KEYS_PRODUCERS_ACTUALLY_WRITE


@pytest.mark.parametrize(
    "summary",
    [
        {
            "diagnostic_companions": {
                "coefficients": "adjusted_association_coefficients.csv"
            }
        },
        {"coefficient_table": "adjusted_association_coefficients.csv"},
        {"coefficient_file": "adjusted_association_coefficients.csv"},
    ],
    ids=["diagnostic_companions", "coefficient_table", "coefficient_file"],
)
def test_every_spelling_a_real_run_used_is_read(summary):
    assert (
        model_summary_coefficient_filename(summary)
        == "adjusted_association_coefficients.csv"
    )


def test_the_deterministic_owner_writes_a_key_this_reader_reads(tmp_path):
    """The producer's REAL summary, not one this test wrote to look like it.

    An earlier version of this test built ``{"coefficient_table": ...}`` by
    hand and passed while the executor's key was renamed out from under it --
    which is precisely the drift the published vocabulary exists to stop. It
    now fits a model, reads the summary the executor actually emitted, and
    hands that to the reader.
    """

    summary = _owner_summary(tmp_path)

    filename = model_summary_coefficient_filename(summary)
    assert filename is not None, sorted(summary)
    assert (tmp_path / filename).is_file()
    assert filename == _COEFFICIENT_FILENAME


def test_a_summary_naming_no_companion_is_a_refusal_not_a_guess():
    """The removed default: a filename nobody declared must not be invented."""

    assert model_summary_coefficient_filename({}) is None
    assert model_summary_coefficient_filename({"model_contracts": []}) is None
    assert model_summary_coefficient_filename({"diagnostic_companions": {}}) is None
    assert model_summary_coefficient_filename({"diagnostic_companions": None}) is None


def test_the_dead_default_filename_is_gone():
    """`coefficients.csv` exists in zero recorded runs; nothing may assume it."""

    assert model_summary_coefficient_filename({"model_contracts": []}) != (
        "coefficients.csv"
    )
    assert model_summary_coefficient_filename({"status": "ok"}) is None


@pytest.mark.parametrize(
    "value",
    [
        123,
        ["coefficients.csv"],
        {"nested": "coefficients.csv"},
        "",
        "   ",
        "../outside.csv",
        "sub/dir/coefficients.csv",
        "coefficients.parquet",
        "coefficients",
    ],
)
def test_a_malformed_companion_fails_closed(value):
    assert model_summary_coefficient_filename({"coefficient_table": value}) is None


def test_a_malformed_nested_companion_fails_closed():
    assert (
        model_summary_coefficient_filename(
            {"diagnostic_companions": {"coefficients": 7}}
        )
        is None
    )
    assert (
        model_summary_coefficient_filename({"diagnostic_companions": "not-a-mapping"})
        is None
    )


@pytest.mark.parametrize(
    "summary",
    [
        {"coefficient_table": 7, "coefficient_file": "real.csv"},
        {"coefficient_table": "", "coefficient_file": "real.csv"},
        {"coefficient_table": "../escape.csv", "coefficient_file": "real.csv"},
        {"coefficient_table": "notes.txt", "coefficient_file": "real.csv"},
        {
            "diagnostic_companions": {"coefficients": 7},
            "coefficient_table": "real.csv",
        },
        {
            "diagnostic_companions": "not-a-mapping",
            "coefficient_table": "real.csv",
        },
    ],
    ids=[
        "non-string",
        "empty",
        "escapes-the-directory",
        "wrong-suffix",
        "nested-non-string",
        "nested-non-mapping",
    ],
)
def test_a_malformed_key_refuses_instead_of_trying_the_next_one(summary):
    """The distinction a same-key-only test cannot see.

    A summary that names a companion and names it badly is a broken
    declaration.  Skipping to the next spelling would silently bind the replay
    to a *different* table than the one the summary tried to point at -- and
    the fall-through is invisible, because the answer it returns is a real
    filename that really exists.
    """

    assert model_summary_coefficient_filename(summary) is None


def test_the_first_published_key_wins_when_a_summary_carries_two():
    summary = {
        "diagnostic_companions": {"coefficients": "first.csv"},
        "coefficient_table": "second.csv",
    }
    assert model_summary_coefficient_filename(summary) == "first.csv"


def test_a_non_mapping_summary_is_refused():
    assert model_summary_coefficient_filename(None) is None  # type: ignore[arg-type]
    assert model_summary_coefficient_filename([]) is None  # type: ignore[arg-type]


# --- one implementation, so the consumer cannot drift again -----------------


def test_the_robustness_reader_delegates_to_the_published_one():
    """Not "both give the same answer here" -- the same object, so they always do."""

    summary = {"coefficient_table": "adjusted_association_coefficients.csv"}
    assert _coefficient_filename_from_summary(
        summary
    ) == model_summary_coefficient_filename(summary)
    assert _coefficient_filename_from_summary({}) is None
