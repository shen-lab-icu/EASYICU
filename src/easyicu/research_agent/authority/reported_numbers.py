"""Whether a sentence shows the numbers a scientific claim states.

Owner
-----
This module owns one comparison: the result numbers a claim states, each in
the unit its typed fields give (:meth:`.ScientificClaim.reported_numbers`),
against the numbers a sentence shows.  It never reads a claim's prose, never
guesses a unit from a value's size and never converts between units.

Two displays of one recorded value agree when they differ by at most half
the coarser of their last displayed places.  Each lies within half its own
place of the value and their difference is a whole number of the finer
place, so it cannot exceed half the coarser one; the edge is included, and
at equal precision the rule is equality.  A count agrees only with an equal
integer.  A percent agrees only with a number the sentence marks as a
percent, shown with a decimal place and two significant figures, so a
coarse display cannot agree with a small value by rounding it away; a zero
shown for an exact zero needs no second figure.  Each distinct number of the
claim must agree with a different number of the sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Sequence

#: Why a number a claim states is not shown by the sentence.
FACT_NUMBER_MISSING = "fact_number_missing"
FACT_PRECISION_TOO_COARSE = "fact_precision_too_coarse"
FACT_UNIT_MISMATCH = "fact_unit_mismatch"

_DISPLAY_RE = re.compile(r"-?\d+(?:\.\d+)?")
# A quoted reader name (“Sepsis-3 sepsis classification”) states no result.
_QUOTED_NAME_RE = re.compile(r"“[^”]*”")
# A sign only where a number may begin: in "10.12-14.57" and "Sepsis-3" the
# hyphen joins two terms.  A comma inside the number groups thousands, and
# digits inside an identifier ("sofa2", "claim_5") are not a number.
_SENTENCE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.,])(?P<sign>[-−])?"
    r"(?P<integer>\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.(?P<fraction>\d+))?"
    r"(?P<percent>\s?%|\s+percent\b)?"
)


@dataclass(frozen=True)
class ReportedNumber:
    """One displayed number: its exact value, last displayed place and unit.

    A number a claim states has the unit ``"count"`` or ``"percent"``.  A
    number read from a sentence has ``"percent"`` when the sentence marks it
    so and ``""`` otherwise.
    """

    value: Decimal
    quantum: Decimal
    unit: str

    @classmethod
    def displayed(cls, text: str, unit: str) -> "ReportedNumber":
        """Read one plain display such as ``"1,234"`` or ``"-12.30"``."""

        plain = str(text).replace(",", "").replace("−", "-")
        if _DISPLAY_RE.fullmatch(plain) is None:
            raise ValueError(f"not a plain decimal display: {text!r}")
        value = Decimal(plain)
        return cls(
            value=value, quantum=Decimal(1).scaleb(value.as_tuple().exponent), unit=unit
        )

    @classmethod
    def recorded(cls, text: str, unit: str, *, places: int) -> "ReportedNumber":
        """Read a value recorded to ``places`` decimals with its trailing
        zeros dropped: ``"10"`` recorded to six places is 10.000000."""

        number = cls.displayed(text, unit)
        return cls(
            value=number.value,
            quantum=min(number.quantum, Decimal(1).scaleb(-places)),
            unit=unit,
        )

    @property
    def places(self) -> int:
        return max(0, -int(self.quantum.as_tuple().exponent))

    @property
    def significant_figures(self) -> int:
        return 0 if self.value == 0 else len(self.value.as_tuple().digits)


@dataclass(frozen=True)
class NumberBinding:
    """Each distinct number a claim states, with the sentence number showing
    it or the reason none does."""

    bound: tuple[tuple[ReportedNumber, ReportedNumber], ...]
    unbound: tuple[tuple[ReportedNumber, str], ...]

    @property
    def complete(self) -> bool:
        return not self.unbound


def sentence_numbers(sentence: str) -> tuple[ReportedNumber, ...]:
    """The numbers a sentence shows outside its quoted names, in order.

    Digits that follow a comma or a point with no space between ("20" in
    "10,20") are not read; a host fact sentence never writes them so.
    """

    numbers = []
    for match in _SENTENCE_NUMBER_RE.finditer(_QUOTED_NAME_RE.sub(" ", sentence)):
        text = ("-" if match["sign"] else "") + match["integer"]
        if match["fraction"] is not None:
            text += "." + match["fraction"]
        numbers.append(
            ReportedNumber.displayed(text, "percent" if match["percent"] else "")
        )
    return tuple(numbers)


def _relation(stated: ReportedNumber, shown: ReportedNumber) -> str | None:
    """``"bound"``, why ``shown`` cannot show ``stated``, or ``None`` when its
    value is not the stated one at either precision."""

    if abs(stated.value - shown.value) > max(stated.quantum, shown.quantum) / 2:
        return None
    if stated.unit == "count":
        return (
            "bound" if shown.unit == "" and shown.quantum == 1 else FACT_UNIT_MISMATCH
        )
    if stated.unit != "percent" or shown.unit != "percent":
        return FACT_UNIT_MISMATCH
    exact_zero = stated.value == 0 and shown.value == 0
    if shown.places < 1 or (shown.significant_figures < 2 and not exact_zero):
        return FACT_PRECISION_TOO_COARSE
    return "bound"


def bind_reported_numbers(
    stated: Sequence[ReportedNumber], shown: Sequence[ReportedNumber]
) -> NumberBinding:
    """Bind each distinct stated number to a different shown number.

    A maximum matching (augmenting paths) decides; a number the claim states
    twice is bound once.  An unbound number names the closest failure: a
    shown number too coarse to show it, else one in another unit, else none.
    """

    first: dict[tuple[Decimal, str], ReportedNumber] = {}
    for number in stated:
        first.setdefault((number.value, number.unit), number)
    distinct = list(first.values())
    relations = [[_relation(number, other) for other in shown] for number in distinct]
    owner_of: dict[int, int] = {}

    def augment(index: int, visited: set[int]) -> bool:
        for position, relation in enumerate(relations[index]):
            if relation != "bound" or position in visited:
                continue
            visited.add(position)
            if position not in owner_of or augment(owner_of[position], visited):
                owner_of[position] = index
                return True
        return False

    for index in range(len(distinct)):
        augment(index, set())
    shown_for = {index: position for position, index in owner_of.items()}
    bound = tuple(
        (number, shown[shown_for[index]])
        for index, number in enumerate(distinct)
        if index in shown_for
    )
    unbound = tuple(
        (
            number,
            next(
                (
                    reason
                    for reason in (FACT_PRECISION_TOO_COARSE, FACT_UNIT_MISMATCH)
                    if reason in relations[index]
                ),
                FACT_NUMBER_MISSING,
            ),
        )
        for index, number in enumerate(distinct)
        if index not in shown_for
    )
    return NumberBinding(bound=bound, unbound=unbound)


__all__ = [
    "FACT_NUMBER_MISSING",
    "FACT_PRECISION_TOO_COARSE",
    "FACT_UNIT_MISMATCH",
    "NumberBinding",
    "ReportedNumber",
    "bind_reported_numbers",
    "sentence_numbers",
]
