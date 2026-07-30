"""How a column is stored must not decide whether it has a category list.

MEASURED 2026-07-30 in ``e3_kdigo_gradient`` of batch
``..._88d3983_canonical9_full02``.  Step ``07_primary_adjusted_association_
models`` -- the study's primary result -- died with::

    RuntimeError: No closed category list is authorized for aki_stage_max

The generated code was right to raise.  The host's own coder authority says
``manifest['raw_input_contracts']['contracts']`` is "the sole executable domain
authority" and that absence "means no closed category list is authorized for
this step: do not recover one from prompt prose, the broader ResearchContext,
or the loaded frame".  The code read the contract, found no ``allowed_values``,
and refused rather than inventing the KDIGO stages.

``allowed_values`` was absent because ``observed_domain_for_series`` emitted
``levels`` for a text column with at most eight distinct values, but among
numeric columns only for a 0/1 binary.  So the recorded domain read::

    sex            {"n_unique": 2, "levels": ["Female", "Male"]}
    death          {"n_unique": 2, "levels": [0, 1]}
    aki_stage_max  {"n_unique": 4, "min": 0.0, "max": 3.0}      <- no levels

``aki_stage_max`` is the KDIGO stage: 0/1/2/3 across all 93,762 rows of the full
cohort.  Every ordinal clinical score is stored as a number, so the rule made a
storage accident decide whether a variable could be closed at all -- and one
missing list killed two steps in the same task, since the stage-stratified
figure was separately faulted by the concept auditor for not enforcing the four
stages and not displaying the zero-count one.

Two guards keep this from turning continuous measurements into categories, and
each has a test below.  Values must be integral: measured across three real
cohorts, not one non-integral column has eight or fewer distinct values, so the
requirement costs nothing and blocks a sparsely-sampled lab result.  And the
existing eight-value cap does the rest -- on the full E3 cohort
``aki_stage_max`` has 4 distinct values while ``aki_stage_first_time``, an hour
offset that is no kind of category, has 25 and stays open.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from easyicu.research_agent.cohort.artifact_facts import (  # noqa: E402
    observed_domain_for_series,
)
from easyicu.research_agent.research_context.typed import (  # noqa: E402
    _closed_observed_levels,
)


# ---------------------------------------------------------------------------
# The measured defect
# ---------------------------------------------------------------------------


def test_an_ordinal_score_stored_as_a_number_has_a_level_set():
    """The exact shape of ``aki_stage_max``: KDIGO stage 0/1/2/3 as float64."""

    stage = pd.Series([0.0, 1.0, 2.0, 3.0] * 25, dtype="float64")

    domain = observed_domain_for_series(stage)

    assert domain["levels"] == [0.0, 1.0, 2.0, 3.0]
    assert domain["is_binary"] is False


def test_the_level_set_reaches_the_executable_contract():
    """``levels`` is only useful if the generated code receives it.

    ``_closed_observed_levels`` is what turns the observed domain into the
    ``allowed_values`` the step manifest carries, and it is where the KDIGO
    stage came out ``None``.
    """

    stage = pd.Series([0.0, 1.0, 2.0, 3.0] * 25, dtype="float64")

    assert _closed_observed_levels(observed_domain_for_series(stage)) == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_storage_does_not_decide_whether_a_variable_can_be_closed():
    """The principle, stated directly: same categories, three spellings."""

    as_text = observed_domain_for_series(pd.Series(["a", "b", "c"] * 30))
    as_int = observed_domain_for_series(pd.Series([1, 2, 3] * 30, dtype="int64"))
    as_float = observed_domain_for_series(pd.Series([1.0, 2.0, 3.0] * 30))

    for domain in (as_text, as_int, as_float):
        assert "levels" in domain, domain
        assert len(domain["levels"]) == 3


def test_the_declared_values_keep_the_column_s_own_type():
    """``1`` and ``1.0`` are different declarations elsewhere in the schema.

    Asserted on ``type(...)``, not on ``==``: the first version of this test
    compared ``[0.0, 2.0, 4.0] == [0, 2, 4]``, which Python says is True, and
    it survived a mutation that made every numeric level a float.
    """

    from_int = observed_domain_for_series(pd.Series([0, 2, 4] * 20, dtype="int64"))[
        "levels"
    ]
    from_float = observed_domain_for_series(pd.Series([0.0, 2.0, 4.0] * 20))["levels"]
    from_bool = observed_domain_for_series(pd.Series([True, False] * 20, dtype="bool"))[
        "levels"
    ]

    assert [type(value).__name__ for value in from_int] == ["int", "int", "int"]
    assert [type(value).__name__ for value in from_float] == [
        "float",
        "float",
        "float",
    ]
    assert [type(value).__name__ for value in from_bool] == ["bool", "bool"]
    assert from_int == [0, 2, 4]
    assert from_float == [0.0, 2.0, 4.0]
    assert from_bool == [False, True]


# ---------------------------------------------------------------------------
# What must NOT become a category set
# ---------------------------------------------------------------------------


def test_a_continuous_measurement_is_not_a_category_set():
    """Few distinct values in one cohort is not a closed domain."""

    lactate = pd.Series([1.2, 3.4, 5.6, 2.1] * 25, dtype="float64")

    domain = observed_domain_for_series(lactate)

    assert "levels" not in domain
    assert domain["min"] == 1.2 and domain["max"] == 5.6


def test_a_wide_integer_column_stays_open():
    """The eight-value cap is what keeps an hour offset out.

    ``aki_stage_first_time`` has 25 distinct integral hour values on the full
    E3 cohort; the stage beside it has 4.
    """

    hours = pd.Series(list(range(25)) * 4, dtype="float64")

    assert "levels" not in observed_domain_for_series(hours)


def test_one_observed_value_is_still_not_a_domain():
    """A constant column cannot be told from a level this cohort just missed.

    The binary branch already refused this; the numeric branch must not
    reintroduce it through a different door.
    """

    assert "levels" not in observed_domain_for_series(
        pd.Series([1.0] * 40, dtype="float64")
    )
    assert "levels" not in observed_domain_for_series(
        pd.Series([7] * 40, dtype="int64")
    )


def test_an_unobserved_level_is_not_invented():
    """Reporting observations, not a codebook.

    ``aki_stage_min`` in the recorded development sample held 0/1/3 -- stage 2
    never appeared.  The domain says so; it does not fill the gap.  A study
    that needs the full declared list must get it from the Planner, which is
    what ``allowed_values_basis`` discloses.
    """

    domain = observed_domain_for_series(pd.Series([0.0, 1.0, 3.0] * 30))

    assert domain["levels"] == [0.0, 1.0, 3.0]


def test_the_binary_contract_is_untouched():
    """Changing what ``is_binary`` means would move far more than this fix."""

    for series, levels in (
        (pd.Series([0, 1] * 30, dtype="int64"), [0, 1]),
        (pd.Series([0.0, 1.0] * 30, dtype="float64"), [0.0, 1.0]),
    ):
        domain = observed_domain_for_series(series)
        assert domain["is_binary"] is True
        assert domain["levels"] == levels

    two_level_ordinal = observed_domain_for_series(
        pd.Series([1, 2] * 30, dtype="int64")
    )
    assert two_level_ordinal["is_binary"] is False
    assert two_level_ordinal["levels"] == [1, 2]
