"""The target trial a study carries, as the researcher approved it.

Owner
-----
This module owns the ``target_trial_design`` section of a study's
configuration and its binding to a run.  The study setup states the trial
(:mod:`.target_trial_spec`) and the population it is eligible from
(:mod:`.population_spec`); the host compiles both
(:mod:`.target_trial_compile`) and keeps the record it compiled with the
population spec (:class:`TargetTrialCompileRecord`) in its own store, by the
record's digest.  The section keeps that digest and how many confirmation
lines the record lists: a study's configuration is small metadata, which a
record does not fit beside, and the digest binds the record to the study's
scientific configuration all the same.  The researcher's click on the
approval card adds the approval: an event id the host mints from the study,
the record's digest, the lines confirmed and the time of the click.  Only
that click writes an approval; a revision the system generates writes none,
so it cannot stand in for one.

A run binds the approved trial onto its research context before planning
(:func:`bind_confirmed_target_trial`): the host compiles the stated trial on
the context the run built, exactly as for the card, and requires the record
the researcher approved.  A difference -- in the data, the study or the
host's own rules -- stops the run before any model is called, and the trial
is confirmed again.  Nothing is relaxed for a difference that looks small.

Nothing here reads a patient row or chooses science.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..canonical_json import canonical_sha256
from ..schema import ResearchContext
from .population_compile import compile_population
from .population_spec import PopulationSpec
from .target_trial_compile import (
    CompiledTargetTrial,
    compile_record_sha256,
    compile_target_trial,
)
from .target_trial_spec import TargetTrialSpec

TARGET_TRIAL_DESIGN_SCHEMA_VERSION = "easyicu.target_trial_design/2"
TARGET_TRIAL_COMPILE_RECORD_SCHEMA_VERSION = "easyicu.target_trial_compile_record/1"
CONFIRMED_TARGET_TRIAL_SCHEMA_VERSION = "easyicu.confirmed_target_trial/1"
APPROVAL_EVENT_PREFIX = "approval:target-trial:"
#: The owner a run's typed stop names when the approved trial does not bind.
TARGET_TRIAL_CONFIRMATION_OWNER = "easyicu.planning.target_trial_confirmation_v1"
TARGET_TRIAL_COMPILE_DRIFTED = "target_trial_compile_drifted"
TARGET_TRIAL_CONFIRMATION_REASON_CODES = frozenset({TARGET_TRIAL_COMPILE_DRIFTED})

_SHA256 = r"^[0-9a-f]{64}$"
#: The host's own clock, to the second, in UTC.
_UTC_SECOND = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


class TargetTrialDesignError(ValueError):
    """A study's target trial section breaks its contract."""

    def __init__(self, code: str, message: str, *, field: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


class TargetTrialConfirmationError(ValueError):
    """A run's research context does not compile the trial the researcher approved."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.easyicu_safe_diagnostic = {
            "owner": TARGET_TRIAL_CONFIRMATION_OWNER,
            "reason_code": reason_code,
        }


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetTrialApproval(_Closed):
    """What the researcher's click on the approval card recorded."""

    approval_event_id: str = Field(pattern=r"^approval:target-trial:[0-9a-f]{16}$")
    #: The lines the card showed and the researcher confirmed.
    n_lines_confirmed: int = Field(ge=1)
    #: The compile record the card showed.
    confirmed_compile_sha256: str = Field(pattern=_SHA256)
    confirmed_at: str = Field(pattern=_UTC_SECOND)


def target_trial_approval_event_id(
    *,
    study_id: str,
    compile_sha256: str,
    n_lines_confirmed: int,
    confirmed_at: str,
) -> str:
    """The event id the host mints for one click on one study's card."""

    digest = canonical_sha256(
        {
            "study_id": study_id,
            "compile_sha256": compile_sha256,
            "n_lines_confirmed": n_lines_confirmed,
            "confirmed_at": confirmed_at,
        }
    )
    return f"{APPROVAL_EVENT_PREFIX}{digest[:16]}"


class TargetTrialCompileRecord(_Closed):
    """A compile record as the host keeps it, with the population it compiled.

    ``record`` is :meth:`CompiledTargetTrial.record` as the host compiled it
    for the card; ``compile_sha256`` is its digest, which names it.
    """

    schema_version: Literal["easyicu.target_trial_compile_record/1"]
    spec: TargetTrialSpec
    #: The population the trial is eligible from, compiled at its time zero.
    population_spec: PopulationSpec
    record: dict[str, Any]
    compile_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _one_record(self) -> "TargetTrialCompileRecord":
        record = self.record
        if compile_record_sha256(record) != self.compile_sha256:
            raise ValueError("the compile record does not have the digest kept with it")
        if record.get("spec") != self.spec.model_dump(mode="json"):
            raise ValueError("the compile record is of another spec")
        if not isinstance(record.get("confirmations"), list) or not isinstance(
            record.get("approvable"), bool
        ):
            raise ValueError("the compile record lists no confirmation lines")
        return self

    @classmethod
    def of(
        cls, compiled: CompiledTargetTrial, population_spec: PopulationSpec
    ) -> "TargetTrialCompileRecord":
        return cls(
            schema_version=TARGET_TRIAL_COMPILE_RECORD_SCHEMA_VERSION,
            spec=compiled.spec,
            population_spec=population_spec,
            record=compiled.record(),
            compile_sha256=compiled.sha256(),
        )

    @property
    def confirmation_lines(self) -> int:
        """The lines the card lists for the researcher to confirm."""

        return len(self.record["confirmations"])

    @property
    def approvable(self) -> bool:
        return self.record["approvable"] is True

    def design(self) -> dict[str, Any]:
        """The section that names this record, before any approval."""

        return {
            "schema_version": TARGET_TRIAL_DESIGN_SCHEMA_VERSION,
            "compile_sha256": self.compile_sha256,
            "confirmation_lines": self.confirmation_lines,
        }


class TargetTrialDesign(_Closed):
    """The ``target_trial_design`` section of a study's configuration."""

    schema_version: Literal["easyicu.target_trial_design/2"]
    #: The digest of the record the host compiled for the card and keeps.
    compile_sha256: str = Field(pattern=_SHA256)
    #: The lines that record lists for the researcher to confirm.
    confirmation_lines: int = Field(ge=1)
    approval: Optional[TargetTrialApproval] = None

    @model_validator(mode="after")
    def _approved_record(self) -> "TargetTrialDesign":
        approval = self.approval
        if approval is None:
            return self
        if approval.confirmed_compile_sha256 != self.compile_sha256:
            raise ValueError("the approval is for another compile record")
        if approval.n_lines_confirmed != self.confirmation_lines:
            raise ValueError(
                "the approval confirms another number of lines than the record lists"
            )
        return self

    def check_record(self, kept: TargetTrialCompileRecord) -> None:
        """Require ``kept`` to be the record this section names and approves.

        The record's own count of confirmation lines and the approval's count
        of lines confirmed come from different owners -- the compile and the
        click -- and must agree, not be copied from one another.
        """

        if kept.compile_sha256 != self.compile_sha256:
            raise TargetTrialDesignError(
                "target_trial_design_invalid",
                "the record kept is not the one the section names",
                field="target_trial_design.compile_sha256",
            )
        if kept.confirmation_lines != self.confirmation_lines:
            raise TargetTrialDesignError(
                "target_trial_design_invalid",
                "the confirmation lines kept differ from the lines the record lists",
                field="target_trial_design.confirmation_lines",
            )
        if self.approval is not None and not kept.approvable:
            raise TargetTrialDesignError(
                "target_trial_design_invalid",
                "an approval needs a record the host can approve",
                field="target_trial_design.approval",
            )

    def confirmed(
        self, kept: TargetTrialCompileRecord
    ) -> Optional["ConfirmedTargetTrial"]:
        """What a run binds, once the researcher approved the record ``kept``."""

        self.check_record(kept)
        if self.approval is None:
            return None
        return ConfirmedTargetTrial(
            schema_version=CONFIRMED_TARGET_TRIAL_SCHEMA_VERSION,
            spec=kept.spec,
            population_spec=kept.population_spec,
            compile_sha256=self.compile_sha256,
            confirmation_lines=self.confirmation_lines,
            approval=self.approval,
        )


class ConfirmedTargetTrial(_Closed):
    """The approved trial a run binds onto its research context."""

    schema_version: Literal["easyicu.confirmed_target_trial/1"]
    spec: TargetTrialSpec
    population_spec: PopulationSpec
    compile_sha256: str = Field(pattern=_SHA256)
    confirmation_lines: int = Field(ge=1)
    approval: TargetTrialApproval

    @model_validator(mode="after")
    def _approved_record(self) -> "ConfirmedTargetTrial":
        if self.approval.confirmed_compile_sha256 != self.compile_sha256:
            raise ValueError("the approval is for another compile record")
        if self.approval.n_lines_confirmed != self.confirmation_lines:
            raise ValueError(
                "the approval confirms another number of lines than the record lists"
            )
        return self


def _first_error_field(exc: ValidationError, *, root: str = "target_trial_design") -> str:
    for error in exc.errors(include_url=False, include_input=False):
        location = [str(part) for part in error.get("loc") or () if part != "__root__"]
        if location:
            return ".".join([root, *location[:3]])
    return root


def load_target_trial_compile_record(value: Any) -> TargetTrialCompileRecord:
    """A kept compile record, read back; the record names itself by its digest."""

    if not isinstance(value, Mapping):
        raise TargetTrialDesignError(
            "target_trial_record_invalid",
            "the kept compile record is not an object",
            field="target_trial_record",
        )
    try:
        return TargetTrialCompileRecord.model_validate(dict(value))
    except ValidationError as exc:
        raise TargetTrialDesignError(
            "target_trial_record_invalid",
            f"the kept compile record breaks its contract: {exc.error_count()} error(s)",
            field=_first_error_field(exc, root="target_trial_record"),
        ) from exc


def load_target_trial_design(
    value: Any, *, study_id: Optional[str] = None
) -> Optional[TargetTrialDesign]:
    """The study's section, or ``None`` for a study that states no trial.

    With ``study_id``, an approval must carry the event id the host mints for
    that study's click: an approval carried over from another study, or
    written by anything but the click, does not.
    """

    if value is None or (isinstance(value, Mapping) and not value):
        return None
    if not isinstance(value, Mapping):
        raise TargetTrialDesignError(
            "target_trial_design_invalid",
            "the target trial design is not an object",
            field="target_trial_design",
        )
    try:
        design = TargetTrialDesign.model_validate(dict(value))
    except ValidationError as exc:
        raise TargetTrialDesignError(
            "target_trial_design_invalid",
            f"the target trial design breaks its contract: {exc.error_count()} error(s)",
            field=_first_error_field(exc),
        ) from exc
    approval = design.approval
    if approval is not None and study_id is not None:
        expected = target_trial_approval_event_id(
            study_id=study_id,
            compile_sha256=approval.confirmed_compile_sha256,
            n_lines_confirmed=approval.n_lines_confirmed,
            confirmed_at=approval.confirmed_at,
        )
        if approval.approval_event_id != expected:
            raise TargetTrialDesignError(
                "target_trial_approval_event_mismatch",
                "the approval's event id is not the one the host mints for this "
                "study's click",
                field="target_trial_design.approval.approval_event_id",
            )
    return design


def normalize_target_trial_design(
    value: Any, *, study_id: Optional[str] = None
) -> dict[str, Any]:
    """The section as the study configuration stores it; ``{}`` for none."""

    design = load_target_trial_design(value, study_id=study_id)
    return {} if design is None else design.model_dump(mode="json")


def bind_confirmed_target_trial(
    context: ResearchContext, payload: Optional[Mapping[str, Any]]
) -> ResearchContext:
    """Require ``context`` to compile to the record the researcher approved.

    ``payload`` is the run's :class:`ConfirmedTargetTrial`, or ``None`` for a
    run that executes no trial.  The trial and its population are compiled on
    the run's own context -- the one its plan, checks and executor read -- as
    they were for the card; a restored context is held to the same record.
    The context is returned unchanged.
    """

    if payload is None:
        return context
    confirmed = ConfirmedTargetTrial.model_validate(payload)
    spec = confirmed.spec
    try:
        population = compile_population(
            confirmed.population_spec,
            context,
            time_zero_hours=spec.time_zero.hours_after_icu_admission,
        )
        compiled = compile_target_trial(spec, context, population=population)
    except ValueError as exc:
        raise TargetTrialConfirmationError(
            TARGET_TRIAL_COMPILE_DRIFTED,
            "the approved target trial does not compile on this run's data",
        ) from exc
    if (
        compiled.sha256() != confirmed.compile_sha256
        or len(compiled.confirmations) != confirmed.confirmation_lines
    ):
        raise TargetTrialConfirmationError(
            TARGET_TRIAL_COMPILE_DRIFTED,
            "the target trial compiles on this run's data to another record than "
            "the one the researcher approved",
        )
    return context


__all__ = [
    "APPROVAL_EVENT_PREFIX",
    "CONFIRMED_TARGET_TRIAL_SCHEMA_VERSION",
    "TARGET_TRIAL_COMPILE_DRIFTED",
    "TARGET_TRIAL_COMPILE_RECORD_SCHEMA_VERSION",
    "TARGET_TRIAL_CONFIRMATION_OWNER",
    "TARGET_TRIAL_CONFIRMATION_REASON_CODES",
    "TARGET_TRIAL_DESIGN_SCHEMA_VERSION",
    "ConfirmedTargetTrial",
    "TargetTrialApproval",
    "TargetTrialCompileRecord",
    "TargetTrialConfirmationError",
    "TargetTrialDesign",
    "TargetTrialDesignError",
    "bind_confirmed_target_trial",
    "load_target_trial_compile_record",
    "load_target_trial_design",
    "normalize_target_trial_design",
    "target_trial_approval_event_id",
]
