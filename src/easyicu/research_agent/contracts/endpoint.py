"""Typed study-endpoint authority shared by context and planning facades."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .closed_levels import validate_closed_scalar_levels

EndpointKind = Literal[
    "binary",
    "continuous",
    "count",
    "ordinal",
    "time_to_event",
    "repeated_measures",
]

EndpointAbsenceSemantics = Literal[
    "absent_row_is_no_event",
    "absent_row_is_unmeasured",
    "no_absent_rows",
]


class EndpointSpec(BaseModel):
    """The study endpoint, declared once and never inferred from data shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: EndpointKind
    absence_semantics: EndpointAbsenceSemantics
    levels: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Closed level set. Required for binary (exactly two), ordinal, "
            "and time-to-event event codes; forbidden for other kinds."
        ),
    )
    event_column: Optional[str] = None
    time_column: Optional[str] = None
    time_origin: Optional[str] = Field(
        default=None,
        description="The declared meaning of t=0; never inferred.",
    )
    censoring_rule: Optional[str] = None

    @model_validator(mode="after")
    def _validate_kind_closure(self) -> "EndpointSpec":
        if not self.name.strip():
            raise ValueError("EndpointSpec.name must be a non-empty column name")

        time_axis_kinds = {"time_to_event", "repeated_measures"}
        event_kinds = {"time_to_event"}
        closed_level_kinds = {"binary", "ordinal", "time_to_event"}
        axis_fields = {"time_column": self.time_column, "time_origin": self.time_origin}
        event_fields = {
            "event_column": self.event_column,
            "censoring_rule": self.censoring_rule,
        }

        def blank(value: Optional[str]) -> bool:
            return value is None or not str(value).strip()

        if self.kind in time_axis_kinds:
            missing = sorted(
                field for field, value in axis_fields.items() if blank(value)
            )
            if missing:
                raise ValueError(
                    f"a {self.kind} endpoint must declare "
                    + ", ".join(missing)
                    + "; these are never inferred from column names or dtypes"
                )
        else:
            present = sorted(
                field for field, value in axis_fields.items() if value is not None
            )
            if present:
                raise ValueError(
                    f"a {self.kind} endpoint has no time axis, so it must not declare "
                    + ", ".join(present)
                )

        if self.kind in event_kinds:
            missing = sorted(
                field for field, value in event_fields.items() if blank(value)
            )
            if missing:
                raise ValueError(
                    f"a {self.kind} endpoint must declare "
                    + ", ".join(missing)
                    + "; these are never inferred from column names or dtypes"
                )
        else:
            present = sorted(
                field for field, value in event_fields.items() if value is not None
            )
            if present:
                raise ValueError(
                    f"a {self.kind} endpoint has no event/censoring structure, so it "
                    "must not declare " + ", ".join(present)
                )

        if self.kind in closed_level_kinds:
            if self.levels is None:
                raise ValueError(
                    f"a {self.kind} endpoint must declare its closed level set; "
                    "an undeclared value must stop the step, not be counted as a non-event"
                )
            levels = validate_closed_scalar_levels(
                self.levels, label=f"EndpointSpec({self.name}) levels"
            )
            minimum = 2
            if self.kind == "binary" and len(levels) != minimum:
                raise ValueError(
                    "a binary endpoint must declare exactly two levels, got "
                    f"{len(levels)}"
                )
            if self.kind == "ordinal" and len(levels) < minimum:
                raise ValueError(
                    "an ordinal endpoint must declare at least two ordered levels"
                )
            if self.kind == "time_to_event" and len(levels) < minimum:
                raise ValueError(
                    "a time_to_event endpoint must declare the closed set of "
                    "event_column codes: the censored code plus one code per event type "
                    "(a competing-risks design declares one each)"
                )
            for index, level in enumerate(levels):
                for other in levels[index + 1 :]:
                    try:
                        collides = bool(level == other)
                    except Exception:  # pragma: no cover - exotic __eq__
                        collides = False
                    if collides:
                        raise ValueError(
                            f"EndpointSpec({self.name}) declares levels {level!r} "
                            f"and {other!r} that compare equal; the data cannot tell them apart"
                        )
        elif self.levels is not None:
            raise ValueError(
                f"a {self.kind} endpoint has no closed level set; remove levels"
            )
        return self

    def declared_columns(self) -> tuple[str, ...]:
        """Return every cohort column bound by this declaration."""

        names = [self.name]
        names.extend(
            column
            for column in (self.event_column, self.time_column)
            if column is not None
        )
        return tuple(dict.fromkeys(names))


__all__ = ["EndpointAbsenceSemantics", "EndpointKind", "EndpointSpec"]
