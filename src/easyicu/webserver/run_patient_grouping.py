"""When a run binds the source's verified patient grouping, and its typed stops.

A plan step needs patient groups when it cannot run without them
(``contracts.patient_grouping_need.steps_needing_patient_groups``, the one
rule the plan review reads too).  A study whose declared inference reads no
patient groups bound no grouping at execution, so such a step reached its
executor without the authority and failed there, after approval and spend,
with an untyped error.

A package-bound run now binds the source's verified grouping when the
accepted candidate plan has such a step.  A grouped materialization emits no
long trajectory, so a study that declares a design read from it stops before
anything is materialized instead.  At approval, a plan whose steps need
patient groups that its planned data does not hold stops before any step
runs, whether the source has no grouping or this run did not bind it.

A metadata-only planning context states what the source can provide
(:func:`planning_patient_grouping_status`), decided by the same resolver and
rule as the binding at execution, so planning and execution cannot disagree.
The run states within-patient dependence only for a design whose inference
reads the grouping (:func:`runtime_patient_dependence`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.patient_grouping_need import (
    PatientGroupingStatus,
    steps_needing_patient_groups,
)
from easyicu.research_agent.planning.dependence_authority import (
    context_patient_group_authority,
)
from easyicu.research_agent.schema import ResearchContext
from easyicu.webserver.research_launch_scientific import (
    declared_longitudinal_design,
    verified_patient_grouping,
)
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

PATIENT_GROUPING_REQUIRED = "research_pipeline_patient_grouping_required"
PATIENT_GROUPING_TRAJECTORY_CONFLICT = (
    "research_pipeline_patient_grouping_trajectory_conflict"
)


def execution_patient_grouping(
    study: Mapping[str, Any],
    *,
    step_ids: Sequence[str],
) -> Optional[PatientGroupingBinding]:
    """The grouping a package-bound run binds for the accepted plan's steps.

    ``None`` when no accepted step needs patient groups, or when the source
    has no verified grouping: the materialized rows may still carry a direct
    patient identifier, and the approval check stops a plan whose planned
    data holds neither.  A study that also declares a design read from the
    long trajectory, which a grouped materialization omits, stops before
    anything is materialized.
    """

    steps = list(dict.fromkeys(str(value) for value in step_ids if str(value)))
    if not steps:
        return None
    grouping = verified_patient_grouping(study)
    if grouping is None:
        return None
    longitudinal = declared_longitudinal_design(study)
    if longitudinal is not None:
        raise ResearchPipelineRunError(
            PATIENT_GROUPING_TRAJECTORY_CONFLICT,
            (
                "A step of the accepted plan needs patient groups, and the study "
                "declares a design read from each stay's long trajectory, which a "
                "patient-grouped materialization does not emit."
            ),
            details={"step_ids": steps, "longitudinal_design": longitudinal},
        )
    return grouping


def planning_patient_grouping_status(
    study: Mapping[str, Any],
    *,
    bound: Optional[PatientGroupingBinding],
) -> tuple[PatientGroupingStatus, Optional[str]]:
    """What a metadata-only planning context states of the source's grouping.

    ``bound`` is the grouping the planning context binds, if any.  Otherwise
    the rule is :func:`execution_patient_grouping`'s: the source's verified
    grouping, and whether a declared design reads the long trajectory a
    grouped materialization does not emit.  A grouping authority that fails
    its checks does not stop planning, which may need no groups: it is
    ``authority_invalid``, with its typed code as the second value, and
    execution refuses with that code if it binds.
    """

    if bound is not None:
        return "bound", None
    try:
        grouping = verified_patient_grouping(study)
    except ResearchPipelineRunError as exc:
        return "authority_invalid", exc.code
    if grouping is None:
        return "source_has_none", None
    if declared_longitudinal_design(study) is not None:
        return "not_carried_by_trajectory", None
    return "available_unbound", None


def runtime_patient_dependence(
    grouping: Optional[PatientGroupingBinding],
    validated_design: Mapping[str, Any],
) -> Optional[PlannedDependenceRequirement]:
    """The within-patient dependence a run's runtime authority states, if any.

    Only a design whose inference reads the grouping states it
    (``study_contexts.analysis_design_reads_patient_grouping``: cluster-robust
    variance, or a bootstrap that resamples patients).  A grouping bound only
    for a step that splits by patient leaves the variance the study declares.
    """

    if (
        grouping is None
        or not study_context_owner.analysis_design_reads_patient_grouping(
            validated_design
        )
    ):
        return None
    return PlannedDependenceRequirement(
        group_source=grouping.output_identity_column,
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )


def _planned_context(path: Path) -> Optional[ResearchContext]:
    try:
        return ResearchContext.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return None


def require_patient_groups_for_approval(plan: Any, *, context_path: Path) -> None:
    """Stop an approval whose plan needs patient groups its data does not hold.

    The executor would refuse such a step after approval; this stops before
    any step runs.  The planned context is read only when a step needs groups.
    """

    steps = steps_needing_patient_groups(plan)
    if not steps:
        return
    context = _planned_context(Path(context_path))
    if context is not None and context_patient_group_authority(context) is not None:
        return
    raise ResearchPipelineRunError(
        PATIENT_GROUPING_REQUIRED,
        (
            "A step of this plan needs patient groups, and the data it was "
            "planned on has no verified patient grouping."
        ),
        details={
            "step_ids": list(steps),
            "grouping_status": (
                "context_unreadable" if context is None else "not_in_planned_data"
            ),
            "stage": "approval",
        },
    )


__all__ = [
    "PATIENT_GROUPING_REQUIRED",
    "PATIENT_GROUPING_TRAJECTORY_CONFLICT",
    "execution_patient_grouping",
    "planning_patient_grouping_status",
    "require_patient_groups_for_approval",
]
