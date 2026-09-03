"""A concept's levels describe its values, not a count of them.

When nothing else fixes a raw column's domain, the typed context falls back to
the source concept's declared dictionary levels.  For the concept's own column
that is right.  For a companion DERIVED from it -- a count of observations, a
0/1 status flag, the timestamp of the first one, a numeric summary -- it
publishes a contract the column cannot satisfy, and generated code that honours
the contract must fail.

MEASURED on h1_ventilation_survival, 2026-08-03 (``..._7ebee35_verify06``).
This was the first run in which h1 ever produced a plan; it reached step 03 and
died there::

    ValueError: mech_vent_n contains 92398 values outside the host-declared
    domain

``mech_vent_n`` is ``physical_role=count`` / ``window_nonnull_count`` / int64,
holding 20-25 observations per stay.  Its declared domain was
``['invasive', 'noninvasive']`` -- the levels of the concept it counts.  All
92,398 of 92,398 rows were outside it, and the generated validator raised on
exactly what the host had told it.

Its sibling ``mech_vent_measured`` escaped only by accident: its own metadata
already declared ``[0, 1]``, so the fallback never ran for it.  The count had
no metadata domain, so it took the concept's.

MEASURED over every recorded resolved-input contract: 6 publish a
concept-dictionary domain.  ONE is a column that really holds those values
(``sex``, ``stay_level_unique_value``, a str column holding Male/Female).  The
other five are ``mech_vent`` companions that cannot: a count, two numeric
aggregates that are float 0.0/1.0, and **two timestamps**.

The rule is read off the corpus rather than judged: 13 distinct
(physical_role, representation_transform) pairs exist and exactly one --
``stay_level_unique_value`` -- carries the value itself.  An unknown transform
answers no, because the failure being guarded is a contract nothing can
satisfy.

The ORDINAL branch is deliberately untouched: those levels come from the host's
own ICU rules, and a max over ordinal stages is still a stage.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.research_context.typed import (  # noqa: E402
    _VALUE_PRESERVING_TRANSFORMS,
    _transform_preserves_concept_values,
    ResearchContextV2,
    resolved_raw_input_contracts,
)

#: Every transform recorded on disk, with what its column actually holds.
#: Only the first carries the concept's own values.
_RECORDED_TRANSFORMS = [
    ("stay_level_unique_value", True, "the concept's own value"),
    ("window_nonnull_count", False, "a count of observations"),
    ("window_measurement_status", False, "a 0/1 flag"),
    ("window_presence_max", False, "a 0/1 flag"),
    ("whole_stay_any_truthy", False, "a 0/1 flag"),
    ("window_first_time", False, "a timestamp"),
    ("window_last_time", False, "a timestamp"),
    ("first_truthy_event_time", False, "a timestamp"),
    ("window_numeric_first", False, "a numeric summary"),
    ("window_numeric_max", False, "a numeric summary"),
    ("window_numeric_mean", False, "a numeric summary"),
    ("window_numeric_min", False, "a numeric summary"),
]


@pytest.mark.parametrize("transform,preserves,holds", _RECORDED_TRANSFORMS)
def test_only_the_value_transform_carries_the_concepts_values(
    transform: str, preserves: bool, holds: str
) -> None:
    assert _transform_preserves_concept_values(transform) is preserves, holds


@pytest.mark.parametrize("unknown", ["", None, "a_transform_added_next_year"])
def test_an_unknown_transform_does_not_inherit_a_domain(unknown) -> None:
    """Ambiguity fails closed: a new representation must claim the domain.

    The failure being guarded is a contract nothing can satisfy, so defaulting
    an unrecognised transform to "inherits" would re-open it for whatever is
    added next.
    """

    assert _transform_preserves_concept_values(unknown) is False


def test_the_value_preserving_set_is_not_silently_widened() -> None:
    """One entry, and the docstring above is the argument for it."""

    assert _VALUE_PRESERVING_TRANSFORMS == frozenset({"stay_level_unique_value"})


_CORPUS = pathlib.Path(
    "/Volumes/外置硬盘/easyicu_data/canonical9_runs"
)


def _h1_context() -> "ResearchContextV2":
    """The exact sealed context of the run that died on this."""

    recorded = sorted(
        _CORPUS.glob(
            "batch_*_verify06/h1_ventilation_survival/aware/run_*/research_context.json"
        )
    )
    if not recorded:
        pytest.skip("the h1 run that recorded this failure is not on disk")
    return ResearchContextV2.model_validate(
        json.loads(recorded[-1].read_text(encoding="utf-8"))
    )


def test_the_production_manifest_stops_publishing_the_impossible_domain() -> None:
    """Drives ``resolved_raw_input_contracts``, not the helper behind it.

    The first version of this file tested only
    ``_transform_preserves_concept_values``, and deleting the gate at the CALL
    SITE left every assertion green -- a load-bearing test that drove a helper
    instead of the function production calls.  This one asks the real entry
    point, with the real sealed context of the run that failed, and reads the
    manifest generated code actually executes against.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")
    contracts = resolved_raw_input_contracts(
        _h1_context(),
        ["mech_vent_n", "mech_vent_max", "mech_vent_first_time", "sex"],
    )["contracts"]

    # The count that killed the step: no borrowed domain at all.
    count = contracts["mech_vent_n"]
    assert count["representation_transform"] == "window_nonnull_count"
    assert count.get("allowed_values_basis") != "declared_concept_dictionary_levels"
    assert "invasive" not in str(count.get("allowed_values"))

    # A timestamp is not a ventilation mode either.
    moment = contracts["mech_vent_first_time"]
    assert moment.get("allowed_values_basis") != "declared_concept_dictionary_levels"

    # The aggregate falls through to what was actually observed, which is the
    # domain it really has -- a strictly better contract than the one it had.
    aggregate = contracts["mech_vent_max"]
    assert aggregate.get("allowed_values") == [0.0, 1.0]
    assert (
        aggregate.get("allowed_values_basis")
        == "sealed_research_context_observed_domain"
    )

    # And the column whose declared levels ARE its values keeps them.
    categorical = contracts["sex"]
    assert categorical["allowed_values"] == ["Female", "Male"]
    assert (
        categorical["allowed_values_basis"] == "declared_concept_dictionary_levels"
    )


def test_the_rule_keeps_the_categorical_column_and_drops_the_derived_ones() -> None:
    """Read off the recorded contracts, not restated from them.

    Stops being meaningful only if the corpus stops containing the case, which
    is why it skips rather than passes vacuously.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    kept: set[tuple[str, str]] = set()
    dropped: set[tuple[str, str]] = set()
    for path in _CORPUS.rglob("resolved_inputs/*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        contracts = (document.get("raw_input_contracts") or {}).get("contracts") or {}
        for column, contract in contracts.items():
            if not isinstance(contract, dict):
                continue
            if (
                contract.get("allowed_values_basis")
                != "declared_concept_dictionary_levels"
            ):
                continue
            transform = str(contract.get("representation_transform") or "")
            target = (
                kept
                if _transform_preserves_concept_values(transform)
                else dropped
            )
            target.add((str(column), transform))

    if not kept and not dropped:
        pytest.skip("no recorded contract publishes a concept-dictionary domain")

    # The one column whose declared levels ARE its values keeps them.
    assert ("sex", "stay_level_unique_value") in kept
    # Every dropped one is a derived companion, never a value column.
    assert dropped, "the defect must still be present in the corpus to be meaningful"
    for column, transform in dropped:
        assert transform not in _VALUE_PRESERVING_TRANSFORMS, column
    # The count that killed h1, and the two timestamps beside it.
    assert ("mech_vent_n", "window_nonnull_count") in dropped
    assert {transform for _, transform in dropped} >= {
        "window_first_time",
        "window_last_time",
    }
