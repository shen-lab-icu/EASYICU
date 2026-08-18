"""The endpoint is declared, never inferred (task O1).

The typed context could always say ``death`` is an OUTCOME of dtype int64 and
``death_time`` is a TIME of dtype float64.  It could never say whether *this
study's* endpoint is the binary flag or the time-to-event pair those two columns
would form -- that pairing did not exist in the type system, so it could only be
guessed from a ``_time`` suffix.  Four separately-patched defects were that one
missing type: a landmark phrase re-typing a mortality study as a survival study;
an undeclared outcome value silently counted as "did not happen"; a positive-only
event's absent rows read as real negatives; a time column used with no declared
origin.

Three properties are load-bearing here and each has its own test below:

1. **Fail closed on an incoherent declaration.**  A kind that needs a closed
   level set must have one; a kind that does not must not carry one.  A time
   axis (``time_column`` + ``time_origin``) and an event/censoring structure
   (``event_column`` + ``censoring_rule``) are *independent*: survival needs
   both, a repeated-measures endpoint needs only the first, a binary endpoint
   needs neither.
2. **Every case shape, not one.**  The first version of this spec was written
   against a binary mortality endpoint and could express neither a
   competing-risks endpoint (H4) nor a repeated-measures one (H3/H5).  Section
   1b covers the shapes E1 never exercises, because a type that fits one study
   is not a type.
3. **Never inferred.**  A builder that can see column names, dtypes and column
   order must not use any of them to fill this in -- those are exactly the
   signals that produced the four defects.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pydantic
import pytest

from easyicu.research_agent import build_research_context
from easyicu.research_agent.research_context.typed import ResearchContextV2
from easyicu.research_agent.schema import EndpointSpec

from tests.research_agent.test_materialized_column_metadata import _build_v2_context


BUILDER_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "research_agent"
    / "research_context"
    / "builder.py"
)


def _binary_death() -> EndpointSpec:
    return EndpointSpec(
        name="death",
        kind="binary",
        absence_semantics="no_absent_rows",
        levels=[0, 1],
    )


# ---------------------------------------------------------------------------
# 1. A coherent declaration of every kind is accepted
# ---------------------------------------------------------------------------


def test_each_kind_can_be_declared_coherently() -> None:
    binary = _binary_death()
    assert binary.declared_columns() == ("death",)

    ordinal = EndpointSpec(
        name="sofa",
        kind="ordinal",
        absence_semantics="absent_row_is_unmeasured",
        levels=[0, 1, 2, 3, 4],
    )
    assert ordinal.levels == [0, 1, 2, 3, 4]

    for kind in ("continuous", "count"):
        spec = EndpointSpec(
            name="los_icu",
            kind=kind,
            absence_semantics="absent_row_is_unmeasured",
        )
        assert spec.levels is None

    survival = EndpointSpec(
        name="death",
        kind="time_to_event",
        absence_semantics="absent_row_is_unmeasured",
        levels=[0, 1],
        event_column="death",
        time_column="death_time",
        time_origin="icu_admission",
        censoring_rule="administrative censoring at 28 days",
    )
    # Both bound columns are reported, de-duplicated, in declaration order.
    assert survival.declared_columns() == ("death", "death_time")


# ---------------------------------------------------------------------------
# 1b. The shapes the *other* benchmark cases need, not just a binary outcome
#
# A type designed against one mortality-association study is a type that fits
# one study.  Each assertion below is a case whose endpoint E1 never exercises;
# the first two were genuinely unrepresentable in the first version of this
# spec and are the reason the field split is what it is.
# ---------------------------------------------------------------------------


def test_a_competing_risks_endpoint_can_be_declared() -> None:
    """H4-shaped: more than one event type competes for the same clock.

    A single event flag plus a free-text censoring rule cannot say that the
    event column carries *which* event happened. Without the closed code set an
    unlisted code lands in the same silent non-event hole M1 closed for binary
    outcomes -- so the level set is required here too, and a competing-risks
    design is simply one that declares more than one event code.
    """
    competing = EndpointSpec(
        name="ventilation_outcome",
        kind="time_to_event",
        absence_semantics="absent_row_is_unmeasured",
        # 0 censored, 1 extubation, 2 death, 3 tracheostomy.
        levels=[0, 1, 2, 3],
        event_column="ventilation_outcome",
        time_column="ventilation_outcome_time",
        time_origin="intubation",
        censoring_rule="administrative censoring at 28 days; 0 denotes censored",
    )
    assert len(competing.levels or []) == 4
    assert competing.declared_columns() == (
        "ventilation_outcome",
        "ventilation_outcome_time",
    )


def test_a_repeated_measures_endpoint_can_declare_its_time_axis() -> None:
    """H3/H5-shaped: a longitudinal endpoint has a clock but no event.

    The first version forbade every time field on any non-survival kind, which
    left a repeated-measures endpoint unable to say which column its time axis
    is -- putting that column straight back into the guessing this type exists
    to end. A time axis and an event/censoring structure are independent.
    """
    trajectory = EndpointSpec(
        name="sofa_total",
        kind="repeated_measures",
        absence_semantics="absent_row_is_unmeasured",
        time_column="hours_from_admission",
        time_origin="icu_admission",
    )
    assert trajectory.event_column is None
    assert trajectory.censoring_rule is None
    assert trajectory.declared_columns() == ("sofa_total", "hours_from_admission")


def test_a_repeated_measures_endpoint_still_cannot_carry_censoring() -> None:
    """The two axes stay independent in both directions."""
    with pytest.raises(pydantic.ValidationError, match="no event/censoring structure"):
        EndpointSpec(
            name="sofa_total",
            kind="repeated_measures",
            absence_semantics="absent_row_is_unmeasured",
            time_column="hours_from_admission",
            time_origin="icu_admission",
            censoring_rule="administrative censoring at 28 days",
        )


def test_a_repeated_measures_endpoint_must_name_its_time_axis() -> None:
    with pytest.raises(pydantic.ValidationError, match="must declare time_column"):
        EndpointSpec(
            name="sofa_total",
            kind="repeated_measures",
            absence_semantics="absent_row_is_unmeasured",
        )


# ---------------------------------------------------------------------------
# 2. Incoherent declarations fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            dict(name="death", kind="binary", absence_semantics="no_absent_rows"),
            "must declare its closed level set",
        ),
        (
            dict(
                name="death",
                kind="binary",
                absence_semantics="no_absent_rows",
                levels=[0, 1, 2],
            ),
            "exactly two levels",
        ),
        (
            dict(
                name="sofa",
                kind="ordinal",
                absence_semantics="no_absent_rows",
                levels=[3],
            ),
            "at least two ordered levels",
        ),
        (
            dict(
                name="los_icu",
                kind="continuous",
                absence_semantics="no_absent_rows",
                levels=[0, 1],
            ),
            "has no closed level set",
        ),
        (
            dict(
                name="death",
                kind="binary",
                absence_semantics="no_absent_rows",
                levels=[0, 1],
                time_column="death_time",
            ),
            "has no time axis, so it must not declare time_column",
        ),
        (
            dict(
                name="death",
                kind="time_to_event",
                absence_semantics="no_absent_rows",
                levels=[0, 1],
                event_column="death",
            ),
            "must declare time_column, time_origin",
        ),
        (
            dict(
                name="death",
                kind="time_to_event",
                absence_semantics="no_absent_rows",
                levels=[0, 1],
                time_column="death_time",
                time_origin="icu_admission",
            ),
            "must declare censoring_rule, event_column",
        ),
        (
            dict(
                name="death",
                kind="time_to_event",
                absence_semantics="no_absent_rows",
                event_column="death",
                time_column="death_time",
                time_origin="icu_admission",
                censoring_rule="28d",
            ),
            "must declare its closed level set",
        ),
        (
            dict(
                name="death",
                kind="time_to_event",
                absence_semantics="no_absent_rows",
                levels=[1],
                event_column="death",
                time_column="death_time",
                time_origin="icu_admission",
                censoring_rule="28d",
            ),
            "censored code plus one code per",
        ),
        (
            dict(
                name="death",
                kind="time_to_event",
                absence_semantics="no_absent_rows",
                levels=[0, 1],
                event_column="death",
                time_column="death_time",
                time_origin="   ",
                censoring_rule="28d",
            ),
            "must declare time_origin",
        ),
        (
            dict(
                name="  ",
                kind="continuous",
                absence_semantics="no_absent_rows",
            ),
            "non-empty column name",
        ),
    ],
)
def test_incoherent_declarations_are_rejected(payload: dict, expected: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=expected):
        EndpointSpec(**payload)


def test_absence_semantics_has_no_default() -> None:
    """The one question a default would answer on the declarer's behalf.

    "The event did not occur" and "nobody looked" produce identically-shaped
    numbers and different science. A default here would silently pick one.
    """
    with pytest.raises(pydantic.ValidationError, match="absence_semantics"):
        EndpointSpec(name="death", kind="binary", levels=[0, 1])

    assert "absence_semantics" in EndpointSpec.model_fields
    assert EndpointSpec.model_fields["absence_semantics"].is_required()


def test_levels_that_compare_equal_are_rejected() -> None:
    """Typed-distinct is not enough for an endpoint: the data must separate them.

    ``_closed_table_one_levels`` keeps ``0`` and ``False`` apart by type, which
    is what M1 needs so an unexpected value cannot impersonate a declared one.
    A column compared against 0 still matches both, so one arm of the contrast
    would silently collapse.
    """
    for levels in ([0, False], [1, True]):
        with pytest.raises(pydantic.ValidationError, match="compare equal"):
            EndpointSpec(
                name="death",
                kind="binary",
                absence_semantics="no_absent_rows",
                levels=levels,
            )


def test_declaration_is_immutable() -> None:
    spec = _binary_death()
    with pytest.raises(pydantic.ValidationError):
        spec.kind = "time_to_event"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Never inferred
# ---------------------------------------------------------------------------


def _mortality_frame() -> pd.DataFrame:
    """A cohort carrying every signal the old guesser used to read."""
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [61.0, 74.0, 55.0, 80.0],
            # An int flag and a float ``*_time`` companion: exactly the pair a
            # suffix/dtype heuristic reads as "this is a survival study".
            "death": [0, 1, 0, 1],
            "death_time": [72.5, 18.0, 96.0, 4.5],
        }
    )


def test_an_undeclared_endpoint_stays_undeclared() -> None:
    context = build_research_context(
        research_question="Is age associated with ICU mortality?",
        cohort=_mortality_frame(),
        cohort_name="synthetic",
        database="synthetic",
        target_outcome="death",
    )
    # The cohort hands the builder a ``death``/``death_time`` pair and a
    # ``_time`` suffix. Nothing may turn that into an endpoint declaration.
    assert context.endpoint is None


def test_a_declaration_survives_a_cohort_that_suggests_otherwise() -> None:
    """The declaration wins over every signal the data offers."""
    context = build_research_context(
        research_question="Is age associated with ICU mortality?",
        cohort=_mortality_frame(),
        cohort_name="synthetic",
        database="synthetic",
        target_outcome="death",
        endpoint=_binary_death(),
    )
    assert context.endpoint == _binary_death()
    assert context.endpoint.kind == "binary"
    # Not rewritten into time_to_event despite death_time sitting right there.
    assert context.endpoint.time_column is None


def test_the_builder_never_reads_the_cohort_to_fill_the_endpoint() -> None:
    """Source-level: ``endpoint`` is only ever assigned from the parameter.

    A behavioural test can only show that today's frame is not misread. This
    shows the builder has no code path that could misread a different one:
    every assignment of the field is the pass-through of the declared value.
    """
    tree = ast.parse(BUILDER_SOURCE.read_text(encoding="utf-8"))

    assignments: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "endpoint":
            assignments.append(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "endpoint":
                    assignments.append(node.value)

    assert assignments, "builder no longer wires the endpoint at all"
    for value in assignments:
        assert isinstance(value, ast.Name) and value.id == "endpoint", (
            "endpoint must be passed through verbatim; "
            f"found {ast.dump(value)[:120]}"
        )


# ---------------------------------------------------------------------------
# 4. The receipt: a declaration is checked against the verified cohort
# ---------------------------------------------------------------------------


def _v2_with_endpoint(tmp_path: Path, endpoint: dict) -> ResearchContextV2:
    context = _build_v2_context(tmp_path)
    payload = context.model_dump(mode="python")
    payload["endpoint"] = endpoint
    return type(context).model_validate(payload)


def test_a_declaration_binding_real_columns_is_accepted(tmp_path: Path) -> None:
    context = _v2_with_endpoint(
        tmp_path,
        dict(
            name="death",
            kind="binary",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
        ),
    )
    assert context.endpoint is not None
    assert context.endpoint.name == "death"


def test_a_declaration_naming_an_absent_column_fails_closed(tmp_path: Path) -> None:
    """Without this the declaration is prose with a type annotation.

    The failure has to land here, at declaration time, and not at the step that
    first tries to read the column -- by then the run has spent provider budget
    and the error surfaces far from its owner.
    """
    with pytest.raises(pydantic.ValidationError, match="absent from the typed cohort"):
        _v2_with_endpoint(
            tmp_path,
            dict(
                name="death",
                kind="time_to_event",
                absence_semantics="absent_row_is_unmeasured",
                levels=[0, 1],
                event_column="death",
                time_column="death_time_that_does_not_exist",
                time_origin="icu_admission",
                censoring_rule="administrative censoring at 28 days",
            ),
        )


def test_a_context_without_a_declaration_still_validates(tmp_path: Path) -> None:
    """Optional, so every context written before O1 still loads."""
    context = _build_v2_context(tmp_path)
    assert context.endpoint is None
