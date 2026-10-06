"""Cohort eligibility must be decided by the analysis time zero.

A stay reaches a typed minimum ICU stay exactly when it is still in the ICU at
that hour, so a minimum beyond time zero selects on survival after it.  A
concept-derived population admits a stay on a positive row up to its window's
end, so a window ending after time zero selects on what happens after it; a
window ending at time zero has decided membership by then.

The family-spec request refuses such a selection before the Planner runs, and
the scientific review refuses a plan built on one, from the same rule here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..research_context.concept_population import ConceptCohortWindow


@dataclass(frozen=True)
class EligibilityAfterTimeZero:
    """One typed criterion that decides cohort membership after time zero."""

    criterion: Literal["minimum_icu_stay", "concept_population"]
    decided_by_hours: float
    time_zero_hours: float
    definition: str = ""

    def message(self) -> str:
        if self.criterion == "minimum_icu_stay":
            return (
                f"a minimum ICU stay of {self.decided_by_hours:g} h ends after the plan's "
                f"time zero at {self.time_zero_hours:g} h after ICU admission; eligibility "
                "would depend on survival after time zero"
            )
        return (
            f"the {self.definition} population admits a stay on a positive row up to "
            f"{self.decided_by_hours:g} h after ICU admission, after the plan's time zero "
            f"at {self.time_zero_hours:g} h; eligibility would depend on what happens "
            f"after time zero, so the study's cohort window must end by "
            f"{self.time_zero_hours:g} h"
        )


def eligibility_after_time_zero(
    *,
    time_zero_hours: float | None,
    minimum_icu_hours: float | None,
    concept_population: ConceptCohortWindow | None,
) -> tuple[EligibilityAfterTimeZero, ...]:
    """Each typed criterion decided after ``time_zero_hours``, minimum stay first.

    Without a time zero nothing is compared.  A criterion decided exactly at
    time zero passes.
    """

    if time_zero_hours is None:
        return ()
    found: list[EligibilityAfterTimeZero] = []
    if minimum_icu_hours is not None and minimum_icu_hours > time_zero_hours:
        found.append(
            EligibilityAfterTimeZero(
                criterion="minimum_icu_stay",
                decided_by_hours=minimum_icu_hours,
                time_zero_hours=time_zero_hours,
            )
        )
    if (
        concept_population is not None
        and concept_population.window_end_hours > time_zero_hours
    ):
        found.append(
            EligibilityAfterTimeZero(
                criterion="concept_population",
                decided_by_hours=concept_population.window_end_hours,
                time_zero_hours=time_zero_hours,
                definition=concept_population.definition,
            )
        )
    return tuple(found)


__all__ = ["EligibilityAfterTimeZero", "eligibility_after_time_zero"]
