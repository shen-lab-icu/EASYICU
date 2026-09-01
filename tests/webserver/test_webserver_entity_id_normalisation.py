"""One rule for turning a stored stay id into the string the app compares on.

Prepared exports store the stay key as int64, float64 or string depending on
the source database and the converter path, so `1`, `1.0` and `"1"` all have to
compare equal. The rule that makes that true lived as a private function in
`dataio` and was reached across module boundaries — `dataio._norm_id` — from
seven other modules including `cohort_review`, `patient_drilldown`, `catalog`,
`agent_outputs` and `ideas/mining`.

Nothing covered it, and it decides whether a cohort filter matches. These
tests pin the behaviour before the rule moves to `entity_ids`, which already
owns the canonical column name, the column resolver and the frame rename.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import entity_ids


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        # The case the rule exists for: the same stay written three ways.
        (7, "7"),
        (7.0, "7"),
        ("7", "7"),
        ("7.0", "7"),
        # Whitespace from a CSV export.
        ("  7  ", "7"),
        # Nothing to compare on.
        (None, ""),
        ("", ""),
        # A genuinely non-integer value keeps its own text.
        (3.7, "3.7"),
        ("3.7", "3.7"),
        ("stay_7", "stay_7"),
        ("abc", "abc"),
        (0, "0"),
        (-0.0, "0"),
        # Large ids must not go through scientific notation.
        (1e15, "1000000000000000"),
    ],
)
def test_ids_that_denote_the_same_stay_normalise_alike(stored: object, expected: str) -> None:
    assert entity_ids.normalize_entity_id(stored) == expected


def test_missing_values_normalise_to_empty_not_to_the_string_nan() -> None:
    """`str(float('nan'))` is `'nan'`, which would match a literal 'nan' id."""

    pd = pytest.importorskip("pandas")

    assert entity_ids.normalize_entity_id(float("nan")) == ""
    assert entity_ids.normalize_entity_id(pd.NA) == ""
    assert entity_ids.normalize_entity_id(pd.NaT) == ""
    assert entity_ids.normalize_entity_id(None) == ""


def test_a_float_column_of_whole_numbers_matches_an_int_column(tmp_path) -> None:
    """The cross-format case: one export stores int64, another float64."""

    pd = pytest.importorskip("pandas")

    as_int = pd.Series([1, 2, 3]).map(entity_ids.normalize_entity_id)
    as_float = pd.Series([1.0, 2.0, 3.0]).map(entity_ids.normalize_entity_id)
    as_text = pd.Series(["1", "2", "3"]).map(entity_ids.normalize_entity_id)

    assert list(as_int) == list(as_float) == list(as_text) == ["1", "2", "3"]


def test_the_normalisation_rule_is_not_reached_through_a_private_name() -> None:
    """It decides cohort membership; it is a contract, not a dataio detail."""

    from pathlib import Path

    webserver = Path(entity_ids.__file__).resolve().parent
    offenders = []
    for path in sorted(webserver.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "dataio._norm_id" in source:
            offenders.append(path.relative_to(webserver).as_posix())
    assert offenders == [], (
        "these modules reach into dataio for the id rule; call "
        f"entity_ids.normalize_entity_id instead: {offenders}"
    )


def test_the_contract_module_is_always_imported_under_an_alias() -> None:
    """`entity_ids` is a common local variable name on these surfaces.

    Importing the module under its bare name reads fine and lints clean, then
    fails at runtime the moment a function in the same file binds a local list
    or set called `entity_ids` — which several of them do. Ruff caught one of
    these (F823, a redefinition) and missed the other, where the shadow is a
    legal function parameter. An alias removes the class of bug rather than the
    instance.
    """

    from pathlib import Path
    import re

    webserver = Path(entity_ids.__file__).resolve().parent
    offenders = []
    for path in sorted(webserver.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if re.search(r"^from easyicu\.webserver import entity_ids$", source, re.M):
            offenders.append(path.relative_to(webserver).as_posix())
    assert offenders == [], (
        "import the entity-id contract under an alias "
        f"(`as entity_id_contract`): {offenders}"
    )
