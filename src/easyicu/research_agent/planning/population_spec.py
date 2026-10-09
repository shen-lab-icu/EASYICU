"""The population a study restricts itself to, as the Planner states it.

The Planner states whom the study includes as typed criteria, never as cohort
predicates: each criterion names the kind of restriction and its parameters,
and the host decides how it reaches the rows (:mod:`.population_compile`).
Only the host knows which column summarizes which window, what the export
already applied and which values a column can hold, so a criterion states
none of that.  A status has no threshold here: "Sepsis-3 within the first
24 h" is a condition that is present, so a test such as ``>= 2`` on a 0/1
column cannot be written.

Each criterion keeps the words it was stated in (``quote``) and where they
came from (``source``), so a reader can check the compiled population against
the study's own statement; one sentence may state several criteria, so they
can share a quote.  A restriction that no kind expresses is stated as
``not_typed`` with the reason; it is never dropped.

``role`` says whether the criterion keeps the stays that meet it (include) or
removes them (exclude).  An age, a stay length, the first ICU stay, survival
to an hour and the absence of an event state the stays kept, so they are
inclusions; "exclude patients younger than 18" is the inclusion of ages from
18.  Bounds are inclusive: ``min_years=18`` keeps ages of 18 and above.
Windows are hours after ICU admission, ``[start_hours, end_hours)``; a
condition or an event with no window (``null``) is read over the whole stay,
as its status records it.  A study whose time zero is another event is
version 2's.

A stated spec decides the plan's cohort (step 2b of the design), so the
planning foundation reads it strictly (:func:`read_stated_population_spec`)
and holds each criterion that cites the study's own words to them as written
(:func:`unquoted_criteria`): a paraphrase or a translation cannot be checked
against what the researcher asked.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, Union, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

POPULATION_SPEC_SCHEMA_VERSION = "easyicu.population_spec/1"
MAX_POPULATION_CRITERIA = 8
MAX_CONDITION_CONCEPTS = 4
MAX_DIAGNOSIS_CODES = 64
#: A year of hours: no population window or stay bound reaches further.
MAX_POPULATION_HOURS = 8760.0

CriterionSource = Literal["question", "study_wording", "outline", "preset"]
CriterionRole = Literal["include", "exclude"]
MeasurementSummary = Literal["max", "min", "first", "last", "mean"]
MeasurementOp = Literal[">", ">=", "<", "<=", "==", "!="]


class SpecWindow(BaseModel):
    """``[start_hours, end_hours)`` after ICU admission, both finite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_hours: float = Field(ge=-MAX_POPULATION_HOURS, le=MAX_POPULATION_HOURS)
    end_hours: float = Field(ge=-MAX_POPULATION_HOURS, le=MAX_POPULATION_HOURS)

    @model_validator(mode="after")
    def _ordered(self) -> "SpecWindow":
        if not self.end_hours > self.start_hours:
            raise ValueError("a population window must end after it starts")
        return self


#: The fields every criterion states, whatever its kind.
_COMMON_FIELDS = ("id", "quote", "source", "role", "kind")


class _Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Unique within the spec; the compiled record cites it.
    id: str = Field(pattern=r"^c[1-9][0-9]?$")
    #: The words the restriction is stated in, verbatim.
    quote: str = Field(min_length=2, max_length=160)
    source: CriterionSource
    role: CriterionRole

    @model_validator(mode="before")
    @classmethod
    def _fields_beside_kind(cls, data: Any) -> Any:
        # A kind's fields written under the kind's name are refused once, with
        # the shape this kind is read in, so the writer can move them.
        kind = data.get("kind") if isinstance(data, Mapping) else None
        if (
            isinstance(kind, str)
            and kind not in cls.model_fields
            and isinstance(data.get(kind), Mapping)
        ):
            fields = [name for name in cls.model_fields if name not in _COMMON_FIELDS]
            shape = json.dumps(
                {"kind": kind, **{name: "..." for name in fields}}, ensure_ascii=False
            )
            raise ValueError(
                f"the {kind} fields go beside kind, not nested under {kind!r}: "
                f"write {shape} next to id, quote, source and role"
            )
        return data


class _Kept(_Criterion):
    """A criterion that states the stays kept."""

    @model_validator(mode="after")
    def _included(self) -> "_Kept":
        if self.role != "include":
            raise ValueError(
                f"a {getattr(self, 'kind', 'population')} criterion states the stays "
                "kept: write it with role include"
            )
        return self


def _concept(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError("a population concept must be named")
    return cleaned


def _bounds(low: Optional[float], high: Optional[float], what: str) -> None:
    if low is None and high is None:
        raise ValueError(f"state a minimum or a maximum {what}")
    if low is not None and high is not None and not low < high:
        raise ValueError(f"the minimum {what} must be below the maximum")


class AgeYears(_Kept):
    """Age at ICU admission within ``[min_years, max_years]``."""

    kind: Literal["age_years"]
    min_years: Optional[float] = Field(default=None, ge=0, le=130)
    max_years: Optional[float] = Field(default=None, ge=0, le=130)

    @model_validator(mode="after")
    def _range(self) -> "AgeYears":
        _bounds(self.min_years, self.max_years, "age")
        return self


class FirstIcuStay(_Kept):
    """Each patient's first ICU stay."""

    kind: Literal["first_icu_stay"]


class IcuStayHours(_Kept):
    """An ICU length of stay within ``[min_hours, max_hours]``."""

    kind: Literal["icu_stay_hours"]
    min_hours: Optional[float] = Field(default=None, gt=0, le=MAX_POPULATION_HOURS)
    max_hours: Optional[float] = Field(default=None, gt=0, le=MAX_POPULATION_HOURS)

    @model_validator(mode="after")
    def _range(self) -> "IcuStayHours":
        _bounds(self.min_hours, self.max_hours, "ICU length of stay")
        return self


class ConditionPresent(_Criterion):
    """Each concept's status is present within the window: no threshold.

    An inclusion keeps the stays with every concept present; an exclusion
    removes the stays with its one concept present.  A stay is removed when
    it meets any exclusion, so an exclusion cannot name a combination.  With
    no window, the status counts over the whole stay.
    """

    kind: Literal["condition_present"]
    concepts_all_of: list[str] = Field(min_length=1, max_length=MAX_CONDITION_CONCEPTS)
    window: Optional[SpecWindow]

    @field_validator("concepts_all_of")
    @classmethod
    def _unique(cls, values: list[str]) -> list[str]:
        cleaned = [_concept(value) for value in values]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("a condition names each concept once")
        return cleaned

    @model_validator(mode="after")
    def _one_when_excluding(self) -> "ConditionPresent":
        if self.role == "exclude" and len(self.concepts_all_of) != 1:
            raise ValueError(
                "an excluding condition names one concept: a stay is removed when it "
                "meets any exclusion, so a combination cannot be excluded"
            )
        return self


class DiagnosisCodes(_Criterion):
    """A recorded diagnosis code begins with one of ``codes``."""

    kind: Literal["diagnosis_codes"]
    system: Literal["icd9", "icd10"]
    codes: list[Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.]{0,9}$")]] = (
        Field(min_length=1, max_length=MAX_DIAGNOSIS_CODES)
    )

    @field_validator("codes")
    @classmethod
    def _unique(cls, values: list[str]) -> list[str]:
        if len({diagnosis_code_token(value) for value in values}) != len(values):
            raise ValueError("diagnosis codes must be distinct")
        return values


class Measurement(_Criterion):
    """A value's summary over the window meets a threshold."""

    kind: Literal["measurement"]
    concept: str = Field(min_length=1, max_length=128)
    summary: MeasurementSummary
    window: SpecWindow
    op: MeasurementOp
    value: float = Field(allow_inf_nan=False)
    #: The threshold's unit when the statement gives one.
    unit: Optional[str] = Field(default=None, min_length=1, max_length=32)

    @field_validator("concept")
    @classmethod
    def _named(cls, value: str) -> str:
        return _concept(value)


class AliveAt(_Kept):
    """No death the input records happened before this hour after ICU admission."""

    kind: Literal["alive_at"]
    hours: float = Field(gt=0, le=MAX_POPULATION_HOURS)


class EventAbsent(_Kept):
    """The concept's event did not happen within the window, or in the stay."""

    kind: Literal["event_absent"]
    concept: str = Field(min_length=1, max_length=128)
    window: Optional[SpecWindow]

    @field_validator("concept")
    @classmethod
    def _named(cls, value: str) -> str:
        return _concept(value)


class NotTyped(_Criterion):
    """A restriction no kind expresses, kept with the reason."""

    kind: Literal["not_typed"]
    why: str = Field(min_length=8, max_length=240)


PopulationCriterion = Annotated[
    Union[
        AgeYears,
        FirstIcuStay,
        IcuStayHours,
        ConditionPresent,
        DiagnosisCodes,
        Measurement,
        AliveAt,
        EventAbsent,
        NotTyped,
    ],
    Field(discriminator="kind"),
]


#: Each criterion kind and the model that states it.
CRITERION_MODELS: Mapping[str, type[_Criterion]] = {
    get_args(model.model_fields["kind"].annotation)[0]: model
    for model in get_args(get_args(PopulationCriterion)[0])
}
#: The kinds that state the stays kept, so a criterion of one only includes.
KEPT_KINDS = frozenset(
    kind for kind, model in CRITERION_MODELS.items() if issubclass(model, _Kept)
)


class PopulationSpec(BaseModel):
    """Every restriction the study states on whom it includes; none for all rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria: list[PopulationCriterion] = Field(
        default_factory=list, max_length=MAX_POPULATION_CRITERIA
    )

    @model_validator(mode="after")
    def _distinct(self) -> "PopulationSpec":
        ids = [item.id for item in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("population criterion ids must be unique")
        restrictions = [
            json.dumps(
                item.model_dump(mode="json", exclude={"id", "quote", "source"}),
                sort_keys=True,
            )
            for item in self.criteria
        ]
        if len(restrictions) != len(set(restrictions)):
            raise ValueError("two population criteria state the same restriction")
        return self


def diagnosis_code_token(code: str) -> str:
    """A diagnosis code as an export matches it: upper case, without dots."""

    return str(code or "").strip().upper().replace(".", "")


#: The sources that cite the researcher's own words.  The others, the outline
#: and a preset, are the Planner's or the host's, with no text of the study's
#: to hold a quote against.
STUDY_WORDING_SOURCES = frozenset({"question", "study_wording"})
#: The owner's errors kept for a refused spec.
_MAX_SPEC_ERRORS = 20
_SPACE = re.compile(r"\s+")


class PopulationSpecRefused(ValueError):
    """The owner refuses a stated spec; ``errors`` locate why, without its input."""

    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        located = "; ".join(
            f"{item['loc'] or '<spec>'}: {item['msg']}" for item in errors
        )
        super().__init__(f"the population spec is refused: {located}")


def read_stated_population_spec(raw: Any) -> Optional[PopulationSpec]:
    """The spec as stated, read strictly; ``None`` when none is stated."""

    if raw is None:
        return None
    try:
        return PopulationSpec.model_validate(raw)
    except ValidationError as exc:
        raise PopulationSpecRefused(
            [
                {
                    "loc": ".".join(str(part) for part in error["loc"]),
                    "type": error["type"],
                    "msg": error["msg"],
                }
                for error in exc.errors(include_url=False, include_input=False)
            ][:_MAX_SPEC_ERRORS]
        ) from exc


def _words(text: Any) -> str:
    """``text`` as words are compared: one form per character, no case, no spacing."""

    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _SPACE.sub("", folded)


def unquoted_criteria(
    spec: PopulationSpec, study_texts: Sequence[str]
) -> tuple[PopulationCriterion, ...]:
    """The criteria citing the study's words whose quote is not written in them.

    ``study_texts`` are the question and the study's own statements of whom it
    includes.  Spacing, letter case and full-width forms are not words.
    """

    texts = [_words(text) for text in study_texts]
    return tuple(
        criterion
        for criterion in spec.criteria
        if criterion.source in STUDY_WORDING_SOURCES
        and not any(_words(criterion.quote) in text for text in texts)
    )


__all__ = [
    "CRITERION_MODELS",
    "KEPT_KINDS",
    "MAX_CONDITION_CONCEPTS",
    "MAX_DIAGNOSIS_CODES",
    "MAX_POPULATION_CRITERIA",
    "MAX_POPULATION_HOURS",
    "POPULATION_SPEC_SCHEMA_VERSION",
    "STUDY_WORDING_SOURCES",
    "AgeYears",
    "AliveAt",
    "ConditionPresent",
    "CriterionRole",
    "CriterionSource",
    "DiagnosisCodes",
    "EventAbsent",
    "FirstIcuStay",
    "IcuStayHours",
    "Measurement",
    "MeasurementOp",
    "MeasurementSummary",
    "NotTyped",
    "PopulationCriterion",
    "PopulationSpec",
    "PopulationSpecRefused",
    "SpecWindow",
    "diagnosis_code_token",
    "read_stated_population_spec",
    "unquoted_criteria",
]
