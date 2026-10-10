"""The exposure groups a study forms, staged before its plan is outlined.

Owner
-----
The plan phase builds the research context on the run's exact copy of its
input (``pipeline._run_plan_phase``).  When the host plans exposure groupings
(``PipelineConfig.enable_exposure_grouping``), this owner asks the Planner
whether the study forms its exposure by grouping one measured value
(``agents.exposure_grouping_planner``) and has the host compile what it
states (``planning.exposure_group_compile``).  When a grouping applies, the
run's cohort is staged again, in place, with each grouping's column of level
codes beside every source column (``intake.materialized_metadata``), and the
context is rebuilt on it.  The input capsule, the plan and the execution read
that cohort, so a grouped exposure is a closed-domain variable the outline
names like any other; a resumed run accepts it as its source's copy
(:func:`resumed_cohort_is_the_sources_copy`).  An input planned on metadata
alone holds no row: there each grouping's column is added empty and declared
in the input's planning authority (``contracts.exposure_group_rules``), and
the context types it as the prepared data's authority will.

What was stated and compiled is recorded (``exposure_groupings.json``)
before anything is staged, and the staged cohort's authority carries the
record's digest.  A run that plans groupings and asks nothing records why
(:data:`NOT_ASKED_REASONS`), so a run without the record did not plan
groupings at all.  A grouping the host cannot apply stops planning with a
typed code before the outline is requested: without the exposure the study
names, the outline would plan another question.  So does a stated group no
stay of the input falls in (``exposure_group_level_empty``): the plan would
compare groups the data do not hold, and a group is never dropped or merged
to make it compare others.  An input planned on metadata alone holds no
row, so no group is empty there.  The Planner is not asked
when no Provider call may be made yet (a capability review or a failed data
gate stops the run first), when a trajectory is staged (a grouped exposure
is no signed suite's input), or when the input holds no value a grouping
can read.

A run that follows an accepted candidate plan
(``PipelineConfig.bound_exposure_groupings``) is not asked either: the
groupings the candidate applied (:func:`candidate_exposure_groupings`) are
compiled again on this run's input, and each must read what it read for the
candidate, or planning stops (``exposure_group_candidate_drift``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

from ..agents.exposure_grouping_planner import (
    EXPOSURE_GROUPING_ROLE,
    ask_exposure_groupings,
)
from ..authority.evidence_store import sha256_of_file
from ..authority.runtime_artifacts import write_json_artifact
from ..contracts import exposure_group_rules
from ..contracts.exposure_group_rules import (
    ExposureGroupRuleError,
    planned_group_column,
    read_planned_group_columns,
)
from ..gates.preplan import preplan_data_findings
from ..intake import materialized_metadata
from ..intake.materialized_metadata import (
    EXPOSURE_GROUP_STAGE_PRODUCER,
    MaterializedMetadataError,
    VerifiedMaterializedCohortAuthority,
    implementation_bundle_sha256,
    stage_exposure_grouped_cohort_authority,
)
from ..planning.exposure_group_compile import (
    CandidateExposureGroupings,
    CompiledGrouping,
    compile_exposure_groupings,
    grouping_sources,
    grouping_variable_name,
    planned_on_metadata,
)
from ..planning.exposure_group_spec import (
    ExposureGroupings,
    read_stated_exposure_groupings,
)
from ..planning.progressive_contract import ProgressivePlanCompileError
from ..providers.structured_retry import StructuredResponseFailure
from ..schema import ResearchContext
from .config import PipelineConfig
from .reviewed_requirements import bind_reviewed_requirements

EXPOSURE_GROUPINGS_FILENAME = "exposure_groupings.json"
EXPOSURE_GROUPINGS_EVIDENCE_ID = "exposure_groupings"
EXPOSURE_GROUPINGS_RECORD_SCHEMA = "easyicu.exposure_groupings_record/1"
#: The parquet attribute a metadata-only planning input records itself in.
_PLANNING_AUTHORITY = "easyicu_planning_authority"
_MAX_RECORD_BYTES = 2 * 1024 * 1024

#: Why planning stops at the groupings.  Stable: a published code never changes.
EXPOSURE_GROUPING_STOPS = (
    "exposure_group_requires_extraction",
    "exposure_group_not_applied",
    "exposure_grouping_unanswered",
    "exposure_group_variable_unbound",
    "exposure_group_candidate_drift",
    "exposure_group_level_empty",
)


#: Why a run that plans groupings asked nothing.  Stable.
NOT_ASKED_REASONS = (
    "capability_review_pending",
    "trajectory_staged",
    "candidate_stated_none",
    "no_value_to_group",
    "data_gate_refused",
)


@dataclass(frozen=True)
class ExposureGroupingPhase:
    """The context the plan is made on."""

    context: ResearchContext
    #: The record of what the Planner stated, when it was asked.
    record_path: Optional[Path] = None


@dataclass(frozen=True)
class PlanningRunGroupings:
    """The groupings a planning run formed, for a run that follows it."""

    #: What the following run binds (``PipelineConfig.bound_exposure_groupings``).
    candidate: CandidateExposureGroupings
    #: The group columns the planning run's input declares.
    variables: tuple[str, ...]
    #: The concepts the groupings read, which the following run's data hold.
    concepts: tuple[str, ...]


def run_exposure_grouping_phase(
    *,
    context: ResearchContext,
    cohort_path: Path,
    run_dir: Path,
    planner: Any,
    rebuild_context: Callable[[Path], ResearchContext],
    capability_review_pending: bool,
    trajectory_staged: bool,
    emit_progress: Callable[..., None],
    candidate: Optional[Mapping[str, Any]] = None,
) -> ExposureGroupingPhase:
    """Ask for the study's groupings and stage the ones the host applies.

    ``cohort_path`` is the run's exact copy of its source, which a grouping
    restages in place; ``planner`` is the Planner's client, wrapped as every
    Planner call is; ``rebuild_context`` builds the context on the restaged
    cohort, bound as the first one was.  ``capability_review_pending`` says
    the run stops for a capability review before any Provider call.
    ``candidate`` is the accepted candidate plan's groupings
    (:class:`CandidateExposureGroupings`); with it, nothing is asked.
    """

    def not_asked(reason: str) -> ExposureGroupingPhase:
        return ExposureGroupingPhase(
            context=context, record_path=_record_not_asked(run_dir, reason)
        )

    accepted = (
        CandidateExposureGroupings.model_validate(candidate)
        if candidate is not None
        else None
    )
    if capability_review_pending:
        return not_asked("capability_review_pending")
    if trajectory_staged and accepted is None:
        return not_asked("trajectory_staged")
    if accepted is not None and not accepted.stated.groupings:
        return not_asked("candidate_stated_none")
    if trajectory_staged:
        raise _drift(
            "This run staged a trajectory, which a grouped exposure is not an input of."
        )
    sources = grouping_sources(context)
    if accepted is None and not sources:
        return not_asked("no_value_to_group")
    # A run the data gate refuses stops before any Provider call is made.
    if any(
        finding.severity == "error"
        for finding in preplan_data_findings(context=context, cohort_path=cohort_path)
    ):
        return not_asked("data_gate_refused")
    if accepted is None:
        emit_progress("context", "Asking whether the study groups a measured value.")
        stated, stated_by = _ask(planner, context=context, sources=sources)
    else:
        stated = accepted.stated
        stated_by = {
            "planner": None,
            "candidate": {
                "record_sha256": accepted.record_sha256,
                "stated": stated.model_dump(mode="json"),
            },
        }
    compiled = compile_exposure_groupings(stated, context)
    record_path = write_json_artifact(
        Path(run_dir) / EXPOSURE_GROUPINGS_FILENAME,
        {
            "schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA,
            "exposure_grouping_enabled": True,
            **stated_by,
            "compiled": compiled.record(),
            "compiled_sha256": compiled.sha256(),
        },
    )
    blocking = [item for item in compiled.groupings if not item.applied]
    if blocking:
        raise _stop(blocking)
    drifted = [
        item
        for item in compiled.applied
        if accepted is not None
        and item.derivation_sha256() != accepted.derivations[item.grouping.id]
    ]
    if drifted:
        raise _drift(
            "; ".join(
                f"{item.grouping.id} reads "
                + ", ".join(
                    f"{summary} from {column!r}"
                    for summary, column in sorted(item.source_columns.items())
                )
                + " here, not what the candidate's grouping read"
                for item in drifted
            )
            + ".",
            findings=[item.record() for item in drifted],
        )
    if not compiled.applied:
        return ExposureGroupingPhase(context=context, record_path=record_path)
    if planned_on_metadata(context):
        _declare_on_metadata(
            Path(cohort_path),
            compiled.applied,
            groupings_record_sha256=sha256_of_file(record_path),
        )
    else:
        _refuse_empty_groups(Path(cohort_path), compiled.applied)
        stage_exposure_grouped_cohort_authority(
            cohort_path,
            groupings=[_staged(item) for item in compiled.applied],
            groupings_record_sha256=sha256_of_file(record_path),
            producer_implementation_sha256=implementation_bundle_sha256(
                (
                    Path(__file__),
                    Path(exposure_group_rules.__file__),
                    Path(materialized_metadata.__file__),
                )
            ),
        )
    grouped = rebuild_context(Path(cohort_path))
    unbound = [
        str(item.variable)
        for item in compiled.applied
        if grouped.variable(str(item.variable)) is None
    ]
    if unbound:
        raise ProgressivePlanCompileError(
            "exposure_group_variable_unbound",
            "The context built on the grouped cohort holds no variable "
            + ", ".join(repr(name) for name in unbound)
            + ".",
            path="exposure_groupings",
        )
    emit_progress(
        "context",
        "Exposure groups staged: "
        + ", ".join(str(item.variable) for item in compiled.applied)
        + ".",
    )
    # The builder states each grouping's reference and contrast from its
    # declaration, so the plan compares the groups the study compares.
    return ExposureGroupingPhase(context=grouped, record_path=record_path)


def candidate_exposure_groupings(
    record: Mapping[str, Any], *, record_sha256: str
) -> CandidateExposureGroupings:
    """The groupings a candidate run's record says it stated and applied.

    ``record`` is the candidate run's ``exposure_groupings.json`` and
    ``record_sha256`` that file's digest, which the caller verified against
    the candidate's sealed input.  ``ValueError`` when the record is not one
    this owner wrote, or holds a grouping the candidate did not apply.
    """

    if record.get("schema_version") != EXPOSURE_GROUPINGS_RECORD_SCHEMA:
        raise ValueError("the record is not an exposure groupings record")
    stated_by = record.get("planner") or record.get("candidate")
    compiled = record.get("compiled")
    items = compiled.get("groupings") if isinstance(compiled, Mapping) else None
    if not isinstance(stated_by, Mapping) or not isinstance(items, list):
        raise ValueError("the record states no groupings")
    if any(
        not isinstance(item, Mapping) or item.get("disposition") != "applied"
        for item in items
    ):
        raise ValueError("the record holds a grouping the candidate did not apply")
    return CandidateExposureGroupings(
        record_sha256=record_sha256,
        stated=read_stated_exposure_groupings(stated_by.get("stated")),
        derivations={
            str(item.get("id")): str(item.get("derivation_sha256")) for item in items
        },
    )


def planning_run_groupings(
    run_dir: Path, *, cohort_sha256: str
) -> Optional[PlanningRunGroupings]:
    """The groupings a run planned on metadata alone formed, checked against its input.

    ``run_dir`` is the planning run's directory and ``cohort_sha256`` the
    digest its input capsule sealed for its ``cohort.parquet``, which the
    caller verified.  ``None`` when the run did not plan groupings or asked
    for none.  ``ValueError`` unless the input is the one sealed, holds no
    row, and declares exactly the group columns the record states, each as
    the host declares it from the record (``_declare_on_metadata``): its
    levels, labels, comparison and the record's digest.
    """

    run_dir = Path(run_dir)
    record_path = run_dir / EXPOSURE_GROUPINGS_FILENAME
    if not record_path.exists() and not record_path.is_symlink():
        return None
    cohort_path = run_dir / "cohort.parquet"
    if any(
        path.is_symlink() or not path.is_file() for path in (record_path, cohort_path)
    ):
        raise ValueError("the planning run's grouping record or input is not a file")
    if sha256_of_file(cohort_path) != cohort_sha256:
        raise ValueError("the planning run's input is not the one its capsule sealed")
    if pq.ParquetFile(cohort_path).metadata.num_rows:
        raise ValueError("a following run binds the groupings of an input with no row")
    raw = record_path.read_bytes()
    if len(raw) > _MAX_RECORD_BYTES:
        raise ValueError("the planning run's grouping record is too large")
    record_sha256 = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw)
    authority = pd.read_parquet(cohort_path).attrs.get(_PLANNING_AUTHORITY)
    raw_declared = (
        authority.get("exposure_groups") if isinstance(authority, Mapping) else None
    )
    declared = (
        read_planned_group_columns(raw_declared) if raw_declared is not None else ()
    )
    if isinstance(record, Mapping) and record.get("not_asked") is not None:
        if declared:
            raise ValueError(
                "the planning run asked for no grouping, but its input declares one"
            )
        return None
    candidate = candidate_exposure_groupings(record, record_sha256=record_sha256)
    variables = {grouping_variable_name(item) for item in candidate.stated.groupings}
    held = sorted(json.dumps(item.record(), sort_keys=True) for item in declared)
    if {item.variable for item in declared} != variables or held != _recorded_columns(
        record, record_sha256=record_sha256
    ):
        raise ValueError(
            "the planning run's input does not declare the group columns its record states"
        )
    return PlanningRunGroupings(
        candidate=candidate,
        variables=tuple(sorted(variables)),
        concepts=tuple(sorted({item.concept for item in candidate.stated.groupings})),
    )


def _recorded_columns(record: Mapping[str, Any], *, record_sha256: str) -> list[str]:
    """Each applied grouping's column as the host declares it from ``record``."""

    columns = []
    for item in record["compiled"]["groupings"]:
        try:
            column = planned_group_column(
                item["derivation"],
                variable=str(item["variable"]),
                labels=item["labels"],
                compared=item["compared"],
                groupings_record_sha256=record_sha256,
            )
        except (ExposureGroupRuleError, KeyError, TypeError) as exc:
            raise ValueError(f"the record's grouping is unreadable: {exc}") from exc
        columns.append(json.dumps(column.record(), sort_keys=True))
    return sorted(columns)


def resumed_cohort_is_the_sources_copy(
    staged: VerifiedMaterializedCohortAuthority,
    source: VerifiedMaterializedCohortAuthority,
) -> bool:
    """Whether a resumed run's staged cohort is its source's typed copy.

    A run stages an exact copy of its source.  A study that formed exposure
    groups restaged that copy with each grouping's column after the source
    columns, which the authority loader proved against the same parent; the
    restage names this source as its parent and its bytes as the ones it
    read.  Either way the rows are the source's, and every source column is
    bound as the source binds it.
    """

    held, origin = staged.authority, source.authority
    held_binding, origin_binding = staged.sidecar.files[0], source.sidecar.files[0]
    same_rows = (
        held.cohort_rows == origin.cohort_rows
        and held.row_identity_sha256 == origin.row_identity_sha256
        and held_binding.identity_column == origin_binding.identity_column
        and held_binding.time_coordinates == origin_binding.time_coordinates
    )
    if held.producer != EXPOSURE_GROUP_STAGE_PRODUCER:
        return (
            same_rows
            and held.cohort_sha256 == origin.cohort_sha256
            and held.cohort_size == origin.cohort_size
            and held.cohort_columns == origin.cohort_columns
            and held.cohort_schema_sha256 == origin.cohort_schema_sha256
            and held_binding.columns == origin_binding.columns
        )
    receipts = held.producer_parameters.get("groupings")
    variables = tuple(
        str(item.get("variable"))
        for item in (receipts if isinstance(receipts, (list, tuple)) else ())
        if isinstance(item, Mapping)
    )
    return (
        same_rows
        and bool(variables)
        and held.parent_authority_sha256 == source.reference.sha256
        and held.producer_parameters.get("source_cohort_sha256") == origin.cohort_sha256
        and held.cohort_columns == (*origin.cohort_columns, *variables)
        and {
            column: binding
            for column, binding in held_binding.columns.items()
            if column not in variables
        }
        == dict(origin_binding.columns)
    )


def bound_context_on(
    builder: Callable[..., ResearchContext],
    context_kwargs: Mapping[str, Any],
    config: PipelineConfig,
    cohort_path: Path,
) -> ResearchContext:
    """The plan phase's context on ``cohort_path``, built and bound as it was first."""

    return bind_reviewed_requirements(
        builder(**{**context_kwargs, "cohort": cohort_path}), config
    )


def register_exposure_groupings_record(evidence: Any, run_dir: Path) -> None:
    """Register the run's grouping record as evidence, once, when it was written."""

    path = Path(run_dir) / EXPOSURE_GROUPINGS_FILENAME
    if not path.is_file() or evidence.get(EXPOSURE_GROUPINGS_EVIDENCE_ID) is not None:
        return
    asked = "not_asked" not in json.loads(path.read_text(encoding="utf-8"))
    evidence.register_file(
        kind="log",
        description=(
            "Exposure groupings the Planner stated and the host compiled before "
            "the outline."
            if asked
            else "Why the run, which plans exposure groupings, asked for none."
        ),
        source_path=path,
        evidence_id=EXPOSURE_GROUPINGS_EVIDENCE_ID,
        producer="exposure_grouping_phase",
        generation_mode="llm" if asked else "system",
    )


def _record_not_asked(run_dir: Path, reason: str) -> Path:
    """Record that a run planning groupings asked nothing, and why."""

    return write_json_artifact(
        Path(run_dir) / EXPOSURE_GROUPINGS_FILENAME,
        {
            "schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA,
            "exposure_grouping_enabled": True,
            "not_asked": reason,
        },
    )


def _ask(
    planner: Any, *, context: ResearchContext, sources: Sequence[Any]
) -> tuple[ExposureGroupings, dict[str, Any]]:
    try:
        answer = ask_exposure_groupings(planner, context=context, sources=sources)
    except StructuredResponseFailure as exc:
        raise ProgressivePlanCompileError(
            "exposure_grouping_unanswered",
            "The Planner gave no exposure groupings the host could read "
            f"({len(exc.attempts)} attempts).",
            path="exposure_groupings",
            cause_code=EXPOSURE_GROUPING_ROLE,
        ) from exc
    return answer.groupings, {"planner": answer.record(), "candidate": None}


def _drift(
    detail: str, *, findings: Optional[list[dict[str, Any]]] = None
) -> ProgressivePlanCompileError:
    """Planning stops: the accepted candidate's groupings do not hold here."""

    return ProgressivePlanCompileError(
        "exposure_group_candidate_drift",
        "The exposure groupings of the accepted candidate plan cannot be formed "
        f"again on this input: {detail.rstrip()}",
        path="exposure_groupings",
        findings=findings or [],
    )


def _refuse_empty_groups(
    cohort_path: Path, applied: Sequence[CompiledGrouping]
) -> None:
    """Planning stops: a stated group holds no stay of the input's rows.

    The plan offered for approval compares every group its study stated, so
    the Planner revises a grouping whose group no stay falls in; the group is
    not dropped or merged here.  How many stays a group holds is a fact of the
    rows, not of the grouping, so it enters no grouping record: a candidate's
    record and its prepared run's still agree.
    """

    empty: list[str] = []
    for item in applied:
        rules = exposure_group_rules.read_grouping_rules(item.derivation())
        table = pq.read_table(cohort_path, columns=list(rules.read_columns))
        held = set(
            exposure_group_rules.evaluate_grouping(rules, table).drop_null().to_pylist()
        )
        labels = item.record()["labels"]
        empty.extend(
            f"{item.grouping.id} {labels[group_id]!r}"
            for group_id, _rule in rules.groups
            if rules.codes[group_id] not in held
        )
    if empty:
        raise ProgressivePlanCompileError(
            "exposure_group_level_empty",
            "A stated exposure group holds no ICU stay on this data: "
            + "; ".join(empty)
            + ". Revise the grouping's thresholds.",
            path="exposure_groupings",
        )


def _declare_on_metadata(
    cohort_path: Path,
    applied: Sequence[CompiledGrouping],
    *,
    groupings_record_sha256: str,
) -> None:
    """Add each grouping's column, empty, to the run's metadata-only input.

    The input holds no row, so no level is derived: its planning authority
    declares the codes each column will hold when the study's data are
    prepared.  The input is written whole and then replaced, so a cohort that
    loads holds every declared column or none.
    """

    if pq.ParquetFile(cohort_path).metadata.num_rows:
        raise MaterializedMetadataError(
            "exposure groups are declared on an input planned on metadata alone, "
            "which holds no row"
        )
    frame = pd.read_parquet(cohort_path)
    authority = frame.attrs.get(_PLANNING_AUTHORITY)
    if not isinstance(authority, Mapping) or "exposure_groups" in authority:
        raise MaterializedMetadataError(
            "exposure groups are declared once, on the run's metadata-only input"
        )
    declared = []
    for item in applied:
        variable = str(item.variable)
        frame[variable] = pd.Series(dtype="int64")
        record = item.record()
        declared.append(
            planned_group_column(
                item.derivation(),
                variable=variable,
                labels=record["labels"],
                compared=record["compared"],
                groupings_record_sha256=groupings_record_sha256,
            ).record()
        )
    frame.attrs[_PLANNING_AUTHORITY] = {**authority, "exposure_groups": declared}
    temporary = cohort_path.with_name(f".{cohort_path.name}.grouped")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, cohort_path)


def _staged(item: CompiledGrouping) -> dict[str, Any]:
    record = item.record()
    return {
        "variable": item.variable,
        "parameters": item.derivation(),
        "labels": record["labels"],
        "compared": record["compared"],
    }


def _stop(blocking: Sequence[CompiledGrouping]) -> ProgressivePlanCompileError:
    """Planning stops: a grouping waits for an extraction, or cannot apply."""

    code = (
        "exposure_group_not_applied"
        if any(item.disposition == "not_applied" for item in blocking)
        else "exposure_group_requires_extraction"
    )
    lines = "; ".join(
        f"{item.grouping.id} ({item.reason}): {item.detail.rstrip(' .')}"
        for item in blocking
    )
    return ProgressivePlanCompileError(
        code,
        f"The study's exposure grouping cannot be applied to this input: {lines}.",
        path="exposure_groupings",
        findings=[item.record() for item in blocking],
    )


__all__ = [
    "EXPOSURE_GROUPINGS_EVIDENCE_ID",
    "EXPOSURE_GROUPINGS_FILENAME",
    "EXPOSURE_GROUPINGS_RECORD_SCHEMA",
    "EXPOSURE_GROUPING_STOPS",
    "NOT_ASKED_REASONS",
    "ExposureGroupingPhase",
    "PlanningRunGroupings",
    "bound_context_on",
    "candidate_exposure_groupings",
    "planning_run_groupings",
    "register_exposure_groupings_record",
    "resumed_cohort_is_the_sources_copy",
    "run_exposure_grouping_phase",
]
