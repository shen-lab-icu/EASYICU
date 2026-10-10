"""Which plan steps need patient groups, and what planning knows of the source's grouping.

A step needs patient groups when it cannot run without them: the static
prediction owner splits development and validation rows by patient
(``contracts.prediction_execution``), and a model requirement that carries
within-patient dependence fits patient clusters (``contracts.dependence``).
The host's binding at execution, its approval check and the plan review read
this one rule, so none of them decides it again.

A metadata-only planning context binds no grouping unless its design reads
one, so what the source can provide is stated beside it, as one of
:data:`PATIENT_GROUPING_STATUSES` under :data:`PATIENT_GROUPING_STATUS_KEY`
(the host decides it; the planning catalog carries it):

* ``bound``: the planning context binds the source's verified grouping;
* ``available_unbound``: the source has a verified grouping that planning did
  not bind; execution binds it for the accepted steps that need it;
* ``not_carried_by_trajectory``: the source has one, but the study declares a
  design read from each stay's long trajectory, which a grouped
  materialization does not emit;
* ``source_has_none``: the host knows no verified grouping for the source;
* ``authority_invalid``: the source names a grouping authority that fails its
  checks; its typed code is stated beside it, under
  :data:`PATIENT_GROUPING_AUTHORITY_ERROR_KEY`, and only then.

Pure: it reads the plan it is given and imports no host module.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import ValidationError

from ..schema import AnalysisStep
from .prediction_execution import static_prediction_execution_verdict

PatientGroupingStatus = Literal[
    "bound",
    "available_unbound",
    "not_carried_by_trajectory",
    "source_has_none",
    "authority_invalid",
]
#: Every status, in the order above.  Stable: a published value never changes.
PATIENT_GROUPING_STATUSES: tuple[PatientGroupingStatus, ...] = (
    "bound",
    "available_unbound",
    "not_carried_by_trajectory",
    "source_has_none",
    "authority_invalid",
)
#: Where a metadata-only planning context states it (``cohort.provenance``).
PATIENT_GROUPING_STATUS_KEY = "patient_grouping_status"
#: The typed code of an authority that fails its checks, beside ``authority_invalid``.
PATIENT_GROUPING_AUTHORITY_ERROR_KEY = "patient_grouping_authority_error"


def _steps(plan: Any) -> list[AnalysisStep]:
    raw = (
        plan.get("steps") if isinstance(plan, Mapping) else getattr(plan, "steps", None)
    )
    steps: list[AnalysisStep] = []
    for item in raw or ():
        if isinstance(item, AnalysisStep):
            steps.append(item)
            continue
        try:
            steps.append(AnalysisStep.model_validate(item))
        except ValidationError:
            # A step that is not a typed step declares no requirement here;
            # the plan's own validation owns refusing it.
            continue
    return steps


def steps_needing_patient_groups(plan: Any) -> tuple[str, ...]:
    """The ids of a plan's steps that cannot run without patient groups, in plan order."""

    return tuple(
        dict.fromkeys(
            step.step_id
            for step in _steps(plan)
            if static_prediction_execution_verdict(step).claimed
            or any(
                getattr(requirement, "dependence", None) is not None
                for requirement in step.model_requirements or ()
            )
        )
    )


__all__ = [
    "PATIENT_GROUPING_AUTHORITY_ERROR_KEY",
    "PATIENT_GROUPING_STATUSES",
    "PATIENT_GROUPING_STATUS_KEY",
    "PatientGroupingStatus",
    "steps_needing_patient_groups",
]
