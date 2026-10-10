"""[Layer 4: Evidence & Provenance] Time-anchored cohort definitions.

CTAS (cohort time-aggregation schema) makes cohort predicates explicit:
concept, time window, aggregation, operator, and value. It is an audit
contract for the research-agent pipeline; it does not replace the broader
EasyICU concept loader.

The framework intentionally ships with an empty named-pattern registry.
Case-specific patterns, such as a benchmark cohort shortcut, must be registered
explicitly by the caller before planning. This keeps shared prompts and shared
agent code case-neutral.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..authority.evidence_snapshot import (
    EvidenceAuthorityIntegrityError,
    load_current_evidence_snapshot,
)
from ..authority.lock_contract import (
    LockAuthorityError,
    assert_lock_matches_evidence_anchor,
)
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..contracts.concept_values import columns_without_values
from ..planning.cohort_contract import (
    ALLOWED_CTAS_AGGREGATIONS,
    Aggregation,
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    PatternRegistry,
    PredicateOp,
    TimeAnchor,
    TimeWindow,
    UNIVERSAL_ANCHORS,
    _CONCEPT_DICT_PATH,
    _DEFAULT_PATTERN_REGISTRY,
    _EXTRA_COHORT_CONCEPT_IDS,
    clear_cohort_concept_ids,
    cohort_concept_id_scope,
    coerce_cohort_definition,
    cohort_definition_has_explicit_selection,
    cohort_definition_sha,
    concept_id_exists,
    default_pattern_registry,
    ensure_cohort_definition,
    event_status_reading,
    expand_named_cohort,
    known_concept_ids,
    register_cohort_concept_ids,
    register_pattern,
    register_patterns_from_file,
    reset_pattern_registry,
    sealed_cohort_concept_ids,
    validate_cohort_definition,
    validate_concept_predicate,
)
from ..research_context.materialization_window import (
    ColumnWindow,
    column_window_from_label,
    context_column_windows,
)
from ..research_context.stay_events import (
    event_times_typed_otherwise_than_hours,
    whole_stay_event_columns,
)
from ..research_context.typed import parse_research_context_json

COHORT_LOCK_FILENAME = "cohort_locked.json"
_IMPLEMENTED_AGGREGATIONS = set(ALLOWED_CTAS_AGGREGATIONS)


def materialized_cohort_concept_id_scope(
    path: Path, verified: Any = None
) -> Any:
    """Scope validation to columns proven by one materialized cohort."""

    if verified is not None:
        columns = verified.authority.cohort_columns
    else:
        import pyarrow.parquet as pq  # type: ignore

        columns = tuple(str(name) for name in pq.read_schema(path).names)
    return cohort_concept_id_scope(columns)


class CohortDataError(KeyError):
    """Raised when materialised data cannot satisfy a CTAS definition."""


class CohortAuthorityError(RuntimeError):
    """Raised when a locked cohort definition cannot be enforced on the data.

    Distinct from :class:`CohortDataError`, which is about one predicate being
    unsatisfiable. This one means the run declared a cohort and then could not
    apply it, so any downstream number would describe a population the plan
    did not authorise.
    """


@dataclass(frozen=True)
class MaterializedInputColumnAuthority:
    """Host-owned partition of sealed cohort columns.

    Identity and time coordinates remain available for cohort navigation but
    are not executable analysis variables.  Keeping that distinction in one
    owner prevents planners and validators from independently reconstructing
    the materialized-input namespace.
    """

    sealed_columns: tuple[str, ...]
    executable_columns: tuple[str, ...]
    reserved_navigation_coordinates: tuple[str, ...]


def materialized_input_column_authority(
    context: Any,
) -> MaterializedInputColumnAuthority:
    """Return the typed materialized-column partition, or an empty legacy one."""

    materialized_inputs = getattr(context, "materialized_inputs", None)
    typed_cohort = getattr(materialized_inputs, "cohort", None)
    sealed_columns = tuple(getattr(typed_cohort, "cohort_columns", ()) or ())
    if not sealed_columns:
        return MaterializedInputColumnAuthority((), (), ())
    bindings = getattr(typed_cohort, "column_bindings", {})
    executable_columns = tuple(bindings.keys()) if isinstance(bindings, Mapping) else ()
    return MaterializedInputColumnAuthority(
        sealed_columns=sealed_columns,
        executable_columns=executable_columns,
        reserved_navigation_coordinates=tuple(
            sorted(set(sealed_columns) - set(executable_columns))
        ),
    )


def context_materialized_columns(context: Any) -> tuple[str, ...]:
    """The columns the run's input carries, as the context records them.

    A typed context seals its cohort columns.  A legacy one lists its
    variables and the cohort's id, time and outcome columns.
    """

    sealed = materialized_input_column_authority(context).sealed_columns
    if sealed:
        return sealed
    cohort = getattr(context, "cohort", None)
    return tuple(
        dict.fromkeys(
            str(column)
            for column in (
                *(
                    getattr(variable, "name", "")
                    for variable in getattr(context, "variables", None) or ()
                ),
                *(getattr(cohort, "id_columns", None) or ()),
                *(getattr(cohort, "time_columns", None) or ()),
                *(getattr(cohort, "outcome_columns", None) or ()),
            )
            if str(column or "").strip()
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_run_cohort_concept_ids(run_dir: Path) -> tuple[str, ...]:
    """The sealed cohort roster of one run, from the context it registered.

    A stored cohort lock or plan may filter on a column the run materialized,
    which validation knows only with this roster
    (:func:`cohort_concept_id_scope`).  A run without a verifiable registered
    context yields no roster, so reading it behaves as it did before; a
    damaged evidence authority is reported by the reader's own anchor check.
    """

    root = Path(run_dir)
    try:
        records = list(load_current_evidence_snapshot(root).records)
    except (EvidenceAuthorityIntegrityError, OSError, ValueError):
        return ()
    for record in reversed(records):
        if not isinstance(record, Mapping) or str(
            record.get("evidence_id") or ""
        ) != "research_context":
            continue
        path = verified_run_evidence_path(root, record)
        if path is None:
            return ()
        try:
            context = parse_research_context_json(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return ()
        return sealed_cohort_concept_ids(context)
    return ()


def _load_locked_cohort_definition(run_dir: Path) -> CohortDefinition:
    path = Path(run_dir) / COHORT_LOCK_FILENAME
    if not path.exists():
        raise CohortSchemaError("cohort_locked.json is missing")
    if path.is_symlink() or not path.is_file():
        raise CohortSchemaError("cohort definition lock must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CohortSchemaError(f"cohort definition lock is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise CohortSchemaError("cohort definition lock has an invalid payload")
    raw_cohort = payload.get("cohort")
    # The lock may filter on a column the run materialized; validation knows
    # it only with the roster of the run's sealed context.
    cohort_concept_ids = registered_run_cohort_concept_ids(run_dir)
    with cohort_concept_id_scope(cohort_concept_ids):
        definition = coerce_cohort_definition(raw_cohort)
        if definition is None:
            raise CohortSchemaError("cohort definition lock has no cohort payload")
        validate_cohort_definition(definition)
    expected_sha = str(payload.get("cohort_sha256") or "").strip()
    observed_sha = cohort_definition_sha(definition)
    if not expected_sha or expected_sha != observed_sha:
        raise CohortSchemaError("cohort definition lock hash mismatch")
    try:
        assert_lock_matches_evidence_anchor(
            run_dir=run_dir,
            lock_path=path,
            evidence_id="cohort_locked",
            label="cohort definition lock",
        )
    except LockAuthorityError as original_exc:
        # A probe-only initial plan may have locked an empty placeholder before
        # the Planner supplied its first real cohort definition in a substantive
        # replan.  That one-way promotion is anchored under an id derived from
        # the promoted scientific digest; arbitrary lock rewrites still fail.
        revision_id = f"cohort_locked_revision_{observed_sha[:8]}"
        try:
            assert_lock_matches_evidence_anchor(
                run_dir=run_dir,
                lock_path=path,
                evidence_id=revision_id,
                label="promoted cohort definition lock",
            )
        except LockAuthorityError as revision_exc:
            raise CohortSchemaError(str(original_exc)) from revision_exc
    return definition


def write_locked_cohort_definition(
    *,
    run_dir: Path,
    plan: Any,
    evidence: Any,
    prompt_pack_version: Optional[str],
    llm_signature: str,
    allow_empty_promotion: bool = False,
    cohort_concept_ids: Sequence[str] = (),
) -> Path:
    # Materialized run columns are not package-level dictionary concepts.  The
    # plan lifecycle seals their exact roster, and callers must present that
    # roster again whenever a cohort payload is revalidated.  Keep the mutable
    # compatibility registry scoped to parsing only; never leak one run's
    # columns into another run or hold its lock during evidence I/O.
    with cohort_concept_id_scope(cohort_concept_ids):
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        if definition is None:
            definition = CohortDefinition(name="primary")
        validate_cohort_definition(definition)
        definition_sha = cohort_definition_sha(definition)
    path = run_dir / COHORT_LOCK_FILENAME
    if path.exists():
        with cohort_concept_id_scope(cohort_concept_ids):
            locked_definition = _load_locked_cohort_definition(run_dir)
            locked_sha = cohort_definition_sha(locked_definition)
        if definition_sha != locked_sha:
            locked_is_empty = not cohort_definition_has_explicit_selection(
                locked_definition
            )
            definition_is_real = cohort_definition_has_explicit_selection(definition)
            if not (allow_empty_promotion and locked_is_empty and definition_is_real):
                raise CohortSchemaError(
                    "cohort definition changed after plan lock; refusing to overwrite "
                    "the pre-specified execution contract"
                )

            # Preserve both authorities: the original empty plan-time lock stays
            # immutable in evidence, while the first real Agent-authored cohort
            # is registered as a digest-named revision before it becomes the live
            # execution lock.  No non-empty lock can ever be promoted again.
            payload = {
                "schema_version": "easyicu.cohort_definition/1",
                "locked_at": datetime.now(timezone.utc).isoformat(),
                "cohort_sha256": definition_sha,
                "cohort": definition.to_dict(),
            }
            revision_id = f"cohort_locked_revision_{definition_sha[:8]}"
            revision_path = run_dir / f"{revision_id}.json"
            revision_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            evidence.register_file(
                kind="log",
                description=(
                    "First substantive cohort definition promoted from the "
                    "probe-only empty plan lock."
                ),
                source_path=revision_path,
                evidence_id=revision_id,
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_pack_version,
                metadata={
                    "llm_signature": llm_signature,
                    "promotes_empty_lock": True,
                    "supersedes_evidence_id": "cohort_locked",
                },
            )
            from ..authority.evidence_store import _atomic_write_bytes

            _atomic_write_bytes(
                path,
                revision_path.read_bytes(),
                expected_root=Path(run_dir).resolve(),
            )
            return path
        if evidence.get("cohort_locked") is None:
            evidence.register_file(
                kind="log",
                description="Time-anchored cohort definition locked after planning.",
                source_path=path,
                evidence_id="cohort_locked",
                aliases=["cohort_locked"],
                producer="planner",
                generation_mode="system",
                prompt_pack_version=prompt_pack_version,
                metadata={"llm_signature": llm_signature, "lock_reused": True},
            )
        return path
    payload = {
        "schema_version": "easyicu.cohort_definition/1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": definition_sha,
        "cohort": definition.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if evidence.get("cohort_locked") is None:
        evidence.register_file(
            kind="log",
            description="Time-anchored cohort definition locked after planning.",
            source_path=path,
            evidence_id="cohort_locked",
            aliases=["cohort_locked"],
            producer="planner",
            generation_mode="system",
            prompt_pack_version=prompt_pack_version,
            metadata={"llm_signature": llm_signature},
        )
    return path


ANALYSIS_COHORT_FILENAME = "cohort_analysis.parquet"


def _declares_analysis_cohort(step: Any, *, plan: Any) -> bool:
    cohort_name = (
        str(getattr(getattr(plan, "cohort", None), "name", "") or "").strip().casefold()
    )
    for raw in getattr(step, "expected_outputs", ()) or ():
        kind, separator, name = str(raw or "").strip().casefold().partition(":")
        if not separator:
            continue
        if kind in {"artifact", "dataset", "table"} and name == "analysis_cohort":
            return True
        if kind == "cohort" and name in {"analysis_set", cohort_name}:
            return True
    return False


def _column_aggregation_matches(name: str, aggregation: str) -> bool:
    """Return whether a cross-name wide column declares the exact CTAS summary."""

    normalized_name = str(name or "").strip().casefold()
    normalized_aggregation = str(aggregation or "").strip().casefold()
    if normalized_aggregation == "count":
        # EasyICU's materialized concept contract names the observation-count
        # companion ``<concept>_n``.  It is the exact executable form of a
        # cohort predicate with ``aggregation='count'``; requiring only the
        # literal ``_count`` suffix makes valid exported count authority
        # impossible to bind.
        return normalized_name.endswith(("_count", "_n"))
    return bool(
        normalized_aggregation in ALLOWED_CTAS_AGGREGATIONS
        and normalized_name.endswith(f"_{normalized_aggregation}")
    )


def _descriptor_window_matches_predicate(value: Any, window: TimeWindow) -> bool:
    """Match a named descriptor window to one exact predicate window.

    Cross-name binding is an execution convenience, not permission to infer a
    temporal contract.  Accept only a label that names a window
    (``column_window_from_label``); missing or broad labels such as
    ``entire_stay`` fail closed.
    """

    column_window = column_window_from_label(value)
    return column_window is not None and column_window.is_window(
        window.anchor, window.start_offset_hours, window.end_offset_hours
    )


def _descriptor_aggregation_matches_predicate(
    *, descriptor_name: str, predicate: ConceptPredicate
) -> bool:
    """Require the declared summary, except for explicit missingness gates.

    ``aggregation='any'`` on a ``missing``/``not_missing`` predicate expresses
    availability of the Planner-selected value, not permission for the host to
    choose one of several summaries.  A unique, non-metadata descriptor from
    the analysis-cohort producer may therefore bind it; ambiguity is rejected
    by the caller.  Numeric/comparison predicates still require an exact
    aggregation suffix.
    """

    aggregation = str(predicate.aggregation or "").strip().casefold()
    op = str(predicate.op or "").strip().casefold()
    if aggregation == "any" and op in {"missing", "not_missing"}:
        return True
    return _column_aggregation_matches(descriptor_name, aggregation)


def _planner_declared_context_column_bindings(
    *,
    definition: CohortDefinition,
    plan: Any,
    context: Any,
    columns: Any,
    label: str = "cohort",
) -> Dict[str, str]:
    """Bind canonical predicate concepts to explicitly planned wide columns.

    The Planner still owns every predicate.  This helper only bridges a
    canonical ``concept_id`` to a materialised output column when all authority
    signals agree: exactly one analysis-cohort producer declares the column as
    an input, and its ResearchContext descriptor binds it to the same
    ``source_concept``, exact time window, and (except for an explicit
    missingness gate) exact aggregation.  This supports ordinary inclusion/QC
    variables without pretending they must be the primary exposure or outcome.
    A sibling output can never be selected by frame order: ambiguity fails
    closed, and no dtype or token fallback is allowed.
    """

    if context is None:
        return {}
    available = {str(column) for column in columns}
    descriptors_by_name: Dict[str, list[Any]] = {}
    for descriptor in getattr(context, "variables", ()) or ():
        name = str(getattr(descriptor, "name", "") or "").strip()
        if name and name in available:
            descriptors_by_name.setdefault(name, []).append(descriptor)

    # Exact/bare column resolution controls *which* column is used, but the
    # suffix alone cannot prove its scientific coordinate.  A direct column is
    # read over the window it was summarized over, its own label else the
    # host's materialization window, whatever window the predicate states;
    # and an event time is read in hours after ICU admission.  Validate a
    # direct column against its sealed descriptor even when the plan has no
    # separate cohort-materialisation step.  Cross-name bindings below remain
    # restricted to an explicit analysis-cohort producer.
    require_column_windows_readable(
        definition,
        columns=available,
        column_windows=context_column_windows(context),
        event_times_not_in_hours=event_times_typed_otherwise_than_hours(context),
        label=label,
    )
    for predicate in (*definition.inclusion, *definition.exclusion):
        direct_column = _resolve_predicate_column(
            columns,
            predicate.concept_id,
            predicate.aggregation,
        )
        direct_descriptors = [
            descriptor
            for descriptor in descriptors_by_name.get(str(direct_column or ""), ())
            if str(getattr(descriptor, "source_concept", "") or "").strip()
            == predicate.concept_id
        ]
        coordinate_descriptors = [
            descriptor
            for descriptor in direct_descriptors
            if str(getattr(descriptor, "analysis_window", "") or "").strip()
        ]
        if coordinate_descriptors and not any(
            _descriptor_aggregation_matches_predicate(
                descriptor_name=str(getattr(descriptor, "name", "") or ""),
                predicate=predicate,
            )
            and _descriptor_window_matches_predicate(
                getattr(descriptor, "analysis_window", None),
                predicate.time_window,
            )
            for descriptor in coordinate_descriptors
        ):
            sealed_windows = sorted(
                {
                    str(getattr(descriptor, "analysis_window", None) or "unknown")
                    for descriptor in coordinate_descriptors
                }
            )
            raise CohortDataError(
                "cohort predicate direct column has no sealed descriptor with "
                "proven matching aggregation and time window for concept "
                f"{predicate.concept_id!r}: requested="
                f"{predicate.time_window.anchor}["
                f"{predicate.time_window.start_offset_hours},"
                f"{predicate.time_window.end_offset_hours}]h/"
                f"{predicate.aggregation}, direct_column={direct_column!r}, "
                f"sealed_windows={sealed_windows!r}"
            )

    producers = [
        step
        for step in getattr(plan, "steps", ()) or ()
        if _declares_analysis_cohort(step, plan=plan)
    ]
    if len(producers) != 1:
        return {}
    declared_inputs = {
        str(value).strip()
        for value in getattr(producers[0], "inputs", ()) or ()
        if str(value or "").strip() in available and ":" not in str(value)
    }
    if not declared_inputs:
        return {}

    descriptors_by_source: Dict[str, list[Any]] = {}
    predicate_aggregations_by_concept: Dict[str, set[str]] = {}
    for predicate in (*definition.inclusion, *definition.exclusion):
        predicate_aggregations_by_concept.setdefault(
            predicate.concept_id,
            set(),
        ).add(str(predicate.aggregation or "").strip().casefold())
    for descriptor in getattr(context, "variables", ()) or ():
        name = str(getattr(descriptor, "name", "") or "").strip()
        source_concept = str(getattr(descriptor, "source_concept", "") or "").strip()
        role = getattr(descriptor, "role", "")
        role_value = str(getattr(role, "value", role) or "").strip().casefold()
        count_companion = bool(
            source_concept
            and predicate_aggregations_by_concept.get(source_concept) == {"count"}
            and _column_aggregation_matches(name, "count")
        )
        if (
            not name
            or not source_concept
            or name not in declared_inputs
            or role_value in {"id", "time"}
            or (role_value == "meta" and not count_companion)
        ):
            continue
        descriptors_by_source.setdefault(source_concept, []).append(descriptor)

    bindings: Dict[str, str] = {}
    predicate_concepts = {
        predicate.concept_id
        for predicate in (*definition.inclusion, *definition.exclusion)
    }
    directly_resolved_concepts = {
        concept_id
        for concept_id in predicate_concepts
        if all(
            _resolve_predicate_column(
                columns,
                predicate.concept_id,
                predicate.aggregation,
            )
            is not None
            for predicate in (*definition.inclusion, *definition.exclusion)
            if predicate.concept_id == concept_id
        )
    }
    for concept_id in sorted(predicate_concepts):
        if concept_id in directly_resolved_concepts:
            continue
        predicates = [
            predicate
            for predicate in (*definition.inclusion, *definition.exclusion)
            if predicate.concept_id == concept_id
        ]
        source_descriptors = descriptors_by_source.get(concept_id, ())
        candidates = sorted(
            str(getattr(descriptor, "name", "") or "").strip()
            for descriptor in source_descriptors
            if all(
                _descriptor_aggregation_matches_predicate(
                    descriptor_name=str(getattr(descriptor, "name", "") or ""),
                    predicate=predicate,
                )
                and _descriptor_window_matches_predicate(
                    getattr(descriptor, "analysis_window", None),
                    predicate.time_window,
                )
                for predicate in predicates
            )
        )
        if source_descriptors and not candidates:
            raise CohortDataError(
                "cohort predicate column binding has no Planner-declared "
                "operational column with proven matching aggregation and time "
                f"window for concept {concept_id!r}"
            )
        if len(candidates) > 1:
            raise CohortDataError(
                "cohort predicate column binding is ambiguous for concept "
                f"{concept_id!r}; Planner-declared ResearchContext candidates: "
                + ", ".join(repr(candidate) for candidate in candidates)
            )
        if len(candidates) == 1 and concept_id not in directly_resolved_concepts:
            bindings[concept_id] = candidates[0]
    return bindings


def _predicate_column_binding_records(
    definition: CohortDefinition,
    bindings: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": concept_id,
            "column": column,
            "basis": "planner_declared_context_input_source_concept",
            "predicate_contracts": [
                {
                    "aggregation": predicate.aggregation,
                    "time_window": predicate.time_window.to_dict(),
                }
                for predicate in (*definition.inclusion, *definition.exclusion)
                if predicate.concept_id == concept_id
            ],
        }
        for concept_id, column in sorted(bindings.items())
    ]


def analysis_cohort_authority_coordinates(
    *,
    plan: Any,
    context: Any,
    columns: Any,
    data: Any = None,
) -> dict[str, object]:
    """Recompute the science-owned coordinates bound by an analysis child."""

    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if not cohort_definition_has_explicit_selection(definition):
        raise CohortSchemaError(
            "analysis cohort authority requires an explicit locked selection"
        )
    bindings = _planner_declared_context_column_bindings(
        definition=definition,
        plan=plan,
        context=context,
        columns=columns,
    )
    coordinates: dict[str, object] = {
        "cohort_definition_sha256": cohort_definition_sha(definition),
        "predicate_column_bindings": _predicate_column_binding_records(
            definition, bindings
        ),
    }
    if data is not None:
        filter_input = data.reset_index(drop=True)
        selected = build_cohort(
            definition,
            filter_input,
            column_bindings=bindings,
        )
        positions = tuple(int(index) for index in selected.index.tolist())
        coordinates["selected_row_count"] = len(positions)
        coordinates["selected_row_positions_sha256"] = hashlib.sha256(
            json.dumps(list(positions), separators=(",", ":")).encode("ascii")
        ).hexdigest()
    return coordinates


def _raw_typed_plan_reference_issues(
    *,
    plan: Any,
    columns: tuple[str, ...],
    reserved_coordinates: tuple[str, ...] = (),
) -> list[str]:
    """Return Planner-owned raw fields absent from the sealed cohort.

    A typed run has no implicit variable namespace.  Raw dataframe fields must
    name an exact sealed column; upstream products use the explicit
    ``kind:name`` syntax and are resolved by the artifact graph instead.
    """

    available = set(columns)
    reserved = set(reserved_coordinates)
    invalid_locations: dict[str, list[str]] = {}

    def require_column(label: str, value: Any) -> None:
        name = str(value or "").strip()
        if name and ":" not in name and name not in available:
            invalid_locations.setdefault(name, []).append(label)

    for step_index, step in enumerate(getattr(plan, "steps", ()) or ()):
        step_id = str(getattr(step, "step_id", "") or step_index)
        # An outcome-by-cluster comparison names its row identity among its
        # inputs and reads it from the typed cohort product: the one reserved
        # navigation coordinate a step may declare, and only as that identity.
        comparison_spec = getattr(step, "phenotype_comparison_spec", None)
        comparison_identity = str(
            getattr(comparison_spec, "identity_column", "") or ""
        )
        for input_index, value in enumerate(getattr(step, "inputs", ()) or ()):
            if comparison_identity and value == comparison_identity and value in reserved:
                continue
            require_column(f"steps[{step_id}].inputs[{input_index}]", value)
        for requirement_index, requirement in enumerate(
            getattr(step, "model_requirements", ()) or ()
        ):
            require_column(
                f"steps[{step_id}].model_requirements" f"[{requirement_index}].outcome",
                getattr(requirement, "outcome", None),
            )
            require_column(
                f"steps[{step_id}].model_requirements"
                f"[{requirement_index}].exposure_source",
                getattr(requirement, "exposure_source", None),
            )

    for spec_index, spec in enumerate(getattr(plan, "robustness_specs", ()) or ()):
        spec_id = str(getattr(spec, "spec_id", "") or spec_index)
        missing = getattr(spec, "missing_override", None)
        if isinstance(missing, Mapping):
            for field in ("variables", "audit_flags"):
                values = missing.get(field)
                if isinstance(values, (list, tuple)):
                    for value_index, value in enumerate(values):
                        require_column(
                            f"robustness_specs[{spec_id}].missing_override."
                            f"{field}[{value_index}]",
                            value,
                        )
        outcome = getattr(spec, "outcome_override", None)
        if isinstance(outcome, Mapping):
            for field in (
                "column",
                "concept_id",
                "target",
                "event_time_column",
                "time_column",
            ):
                if outcome.get(field) is not None:
                    require_column(
                        f"robustness_specs[{spec_id}].outcome_override.{field}",
                        outcome.get(field),
                    )

    def location_categories(locations: list[str]) -> dict[str, int]:
        categories: dict[str, int] = {}
        for location in locations:
            if ".model_requirements" in location:
                category = (
                    "model outcomes"
                    if location.endswith(".outcome")
                    else "model exposures"
                )
            elif ".missing_override.variables" in location:
                category = "robustness missing variables"
            elif ".missing_override.audit_flags" in location:
                category = "robustness audit flags"
            elif ".outcome_override." in location:
                category = "robustness outcome fields"
            else:
                category = "step inputs"
            categories[category] = categories.get(category, 0) + 1
        return categories

    issues: list[str] = []
    for name, locations in invalid_locations.items():
        categories = location_categories(locations)
        if name in reserved:
            issues.append(
                f"raw name {name!r} is a sealed identity/time coordinate "
                "reserved for host navigation, not an executable analysis "
                f"field; locations={categories!r}"
            )
        else:
            issues.append(
                f"raw name {name!r} is not an exact executable sealed cohort "
                f"column; locations={categories!r}"
            )
    return issues


def _closed_observed_levels(variable: Any) -> list[Any]:
    """Return host-visible closed levels without exposing them in diagnostics."""

    if variable is None:
        return []
    domain = getattr(variable, "observed_domain", None)
    if not isinstance(domain, Mapping):
        return []
    levels = domain.get("levels")
    if isinstance(levels, list) and len(levels) >= 2:
        return list(levels)
    if not domain.get("is_binary"):
        return []
    dtype = str(getattr(variable, "dtype", "") or "").strip().casefold()
    if dtype.startswith(("int", "uint")):
        return [0, 1]
    if dtype.startswith(("float", "double")):
        return [0.0, 1.0]
    if dtype.startswith("bool"):
        return [False, True]
    return []


def predicate_accepts_closed_level(
    predicate: ConceptPredicate,
    level: Any,
) -> Optional[bool]:
    """Evaluate one typed predicate on a local closed level, if comparable.

    Each comparison is the one ``build_cohort`` applies to a column value
    (``_apply_op``), so a check that judges a predicate against a column's
    closed levels reads it as the cohort builder does.  ``None`` when the
    level and the value cannot be compared.
    """

    op = str(predicate.op or "").strip().casefold()
    target = predicate.value
    try:
        if op == "==":
            return bool(level == target)
        if op == "!=":
            return bool(level != target)
        if op == "<":
            return bool(level < target)
        if op == "<=":
            return bool(level <= target)
        if op == ">":
            return bool(level > target)
        if op == ">=":
            return bool(level >= target)
        if op == "in":
            values = target if isinstance(target, list) else [target]
            return bool(level in values)
        if op == "not_in":
            values = target if isinstance(target, list) else [target]
            return bool(level not in values)
        if op == "missing":
            return bool(
                level is None or (isinstance(level, float) and math.isnan(level))
            )
        if op == "not_missing":
            return not bool(
                level is None or (isinstance(level, float) and math.isnan(level))
            )
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def _primary_cohort_contrast_preservation_issues(
    *,
    plan: Any,
    context: Any,
    definition: CohortDefinition,
    columns: tuple[str, ...],
    bindings: Mapping[str, str],
) -> list[str]:
    """Reject a primary cohort that statically erases a planned contrast.

    This is a consistency check only.  The host does not choose eligibility or
    an estimand: it verifies that the Planner's own closed cohort predicates do
    not leave fewer than two levels of the same variable that the Planner later
    declares as a grouped comparison or required primary-model exposure.
    """

    variables = {
        str(getattr(variable, "name", "") or "").strip(): variable
        for variable in getattr(context, "variables", ()) or ()
    }
    targets: dict[str, list[Any]] = {}

    # Table 1 private execution bindings contain the locally observed labels;
    # public opaque tokens are never copied into this validation diagnostic.
    from ..authority.table_one_binding import table_one_execution_spec

    for step in getattr(plan, "steps", ()) or ():
        spec = table_one_execution_spec(step)
        if spec is not None and len(spec.group_levels) >= 2:
            targets.setdefault(str(spec.group_by), list(spec.group_levels))
        for requirement in getattr(step, "model_requirements", ()) or ():
            role = str(getattr(requirement, "analysis_role", "") or "").casefold()
            if role != "primary":
                continue
            exposure = str(getattr(requirement, "exposure_source", "") or "").strip()
            levels = _closed_observed_levels(variables.get(exposure))
            if exposure and len(levels) >= 2:
                targets.setdefault(exposure, levels)

    if not targets:
        return []

    predicates_by_column: dict[str, dict[str, list[ConceptPredicate]]] = {}
    for kind, predicates in (
        ("inclusion", definition.inclusion),
        ("exclusion", definition.exclusion),
    ):
        for predicate in predicates:
            column = _resolve_predicate_column(
                columns,
                predicate.concept_id,
                predicate.aggregation,
                column_bindings=dict(bindings),
            )
            if column:
                predicates_by_column.setdefault(column, {}).setdefault(kind, []).append(
                    predicate
                )

    issues: list[str] = []
    for column, levels in targets.items():
        predicate_sets = predicates_by_column.get(column)
        if not predicate_sets:
            continue
        retained = 0
        indeterminate = False
        for level in levels:
            include = True
            for predicate in predicate_sets.get("inclusion", ()):
                accepted = predicate_accepts_closed_level(predicate, level)
                if accepted is None:
                    indeterminate = True
                    break
                include = include and accepted
            if indeterminate:
                break
            for predicate in predicate_sets.get("exclusion", ()):
                excluded = predicate_accepts_closed_level(predicate, level)
                if excluded is None:
                    indeterminate = True
                    break
                include = include and not excluded
            if indeterminate:
                break
            retained += int(include)
        if not indeterminate and retained < 2:
            issues.append(
                "cohort: primary cohort predicates collapse a downstream closed "
                f"comparison on sealed column {column!r} below two retained "
                "levels. Revise the cohort eligibility or the downstream "
                "comparison/primary estimand so the plan is internally consistent."
            )
    return issues


def validate_plan_typed_bindings_against_context(
    *,
    plan: Any,
    context: Any,
) -> None:
    """Reject Planner references that cannot reach the sealed run input.

    Global dictionary membership is necessary but insufficient for a typed
    run: a legal EasyICU concept can still be absent from this immutable
    materialized cohort, or available only under a different sealed
    window/aggregation.  Likewise, a semantic label is not an executable
    dataframe column.  Validate cohort predicates, raw step inputs, model
    outcome/exposure fields, and robustness variables while the Planner's
    structured retry is active instead of failing later inside LangGraph
    execution.

    Legacy contexts retain their historical behavior because they do not carry
    a host-verified materialized column roster.
    """

    _require_primary_event_windows_readable(plan=plan, context=context)
    column_authority = materialized_input_column_authority(context)
    columns = column_authority.sealed_columns
    if not columns:
        return

    definitions: list[tuple[str, CohortDefinition]] = []
    primary = coerce_cohort_definition(getattr(plan, "cohort", None))
    if primary is not None and (primary.inclusion or primary.exclusion):
        definitions.append(("cohort", primary))
    for index, spec in enumerate(getattr(plan, "robustness_specs", ()) or ()):
        override = coerce_cohort_definition(getattr(spec, "cohort_override", None))
        if override is not None and (override.inclusion or override.exclusion):
            spec_id = str(getattr(spec, "spec_id", "") or index)
            definitions.append(
                (f"robustness_specs[{spec_id}].cohort_override", override)
            )

    # Identity/time coordinates are navigation metadata, not executable
    # analysis variables. Runtime raw-input contracts intentionally omit them,
    # so reject them while the Planner still has structured-retry authority.
    executable_columns = column_authority.executable_columns
    reserved_coordinates = column_authority.reserved_navigation_coordinates
    raw_issues = _raw_typed_plan_reference_issues(
        plan=plan,
        columns=executable_columns,
        reserved_coordinates=reserved_coordinates,
    )
    issues = list(raw_issues)
    unreadable_codes: set[str] = set()
    corrected_issues = 0
    primary_definition: Optional[CohortDefinition] = None
    primary_bindings: Dict[str, str] = {}
    for label, definition in definitions:
        try:
            bindings = _planner_declared_context_column_bindings(
                definition=definition,
                plan=plan,
                context=context,
                columns=columns,
                label=label,
            )
        except CohortDataError as exc:
            code = str(getattr(exc, "code", "") or "")
            issues.append(str(exc) if code else f"{label}: {exc}")
            if code in _UNREADABLE_WINDOW_CORRECTIONS:
                unreadable_codes.add(code)
                corrected_issues += 1
            continue
        if label == "cohort":
            primary_definition = definition
            primary_bindings = bindings
        for kind, predicates in (
            ("inclusion", definition.inclusion),
            ("exclusion", definition.exclusion),
        ):
            for index, predicate in enumerate(predicates):
                if (
                    _resolve_predicate_column(
                        columns,
                        predicate.concept_id,
                        predicate.aggregation,
                        column_bindings=bindings,
                    )
                    is None
                ):
                    issues.append(
                        f"{label}.{kind}[{index}] concept_id="
                        f"{predicate.concept_id!r}, aggregation="
                        f"{predicate.aggregation!r}, window="
                        f"{predicate.time_window.anchor}["
                        f"{predicate.time_window.start_offset_hours},"
                        f"{predicate.time_window.end_offset_hours}]h has no "
                        "exact or uniquely bound sealed column"
                    )
    if primary_definition is not None:
        issues.extend(
            _primary_cohort_contrast_preservation_issues(
                plan=plan,
                context=context,
                definition=primary_definition,
                columns=columns,
                bindings=primary_bindings,
            )
        )
    if not issues:
        return

    column_set = set(executable_columns)
    producer_columns = sorted(
        {
            str(value).strip()
            for step in getattr(plan, "steps", ()) or ()
            if _declares_analysis_cohort(step, plan=plan)
            for value in getattr(step, "inputs", ()) or ()
            if str(value or "").strip() in column_set and ":" not in str(value)
        }
    )
    typed_sources = sorted(
        {
            str(getattr(variable, "source_concept", "") or "").strip()
            for variable in getattr(context, "variables", ()) or ()
            if str(getattr(variable, "source_concept", "") or "").strip()
            and str(getattr(variable, "name", "") or "").strip() in producer_columns
        }
    )
    detail = "; ".join(issues[:4])
    corrections = [
        _UNREADABLE_WINDOW_CORRECTIONS[code] for code in sorted(unreadable_codes)
    ]
    if raw_issues:
        correction = (
            "For raw step inputs, Table 1, model requirements, and robustness "
            "fields, copy exact names from the executable materialized-input "
            "roster in the original prompt. Never list cohort id/time "
            "coordinates as analysis inputs; the host owns row navigation and "
            "cohort accounting. Concept ids are only valid inside typed cohort "
            "predicates, and kind:name is only valid for an explicit upstream "
            "product."
        )
    elif corrected_issues == len(issues):
        # Each issue has its own correction.
        correction = ""
    else:
        correction = (
            "Use an executable dictionary concept whose exact "
            "window/aggregation is bound by the declared analysis-cohort "
            f"columns={producer_columns!r} and source concepts={typed_sources!r}."
        )
    correction = " ".join(text for text in (correction, *corrections) if text)
    raise CohortSchemaError(
        "typed plan references are not executable against this sealed input. "
        f"Invalid references: {detail}. {correction} Additional binding context: "
        "declared "
        f"typed source concepts={typed_sources!r}; declared columns="
        f"{producer_columns!r}; executable cohort columns="
        f"{sorted(executable_columns)!r}; reserved navigation coordinates="
        f"{list(reserved_coordinates)!r}."
    )


#: Why a cohort predicate cannot be read over the window it states.
COHORT_COLUMN_WINDOW_MISMATCH = "cohort_column_window_mismatch"
#: Why a cohort predicate cannot read its event's time against its window.
COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION = (
    "cohort_event_time_not_hours_from_icu_admission"
)


#: How a plan states a criterion its cohort does not apply.
_NOT_APPLIED = (
    "List each such criterion in population_criteria with no concepts (a plan "
    "without population_criteria lists it in "
    "cohort.unapplied_population_criteria) and remove its predicate, so the "
    "results report it as not applied."
)
#: The correction for each reason the input cannot read a predicate's window.
_UNREADABLE_WINDOW_CORRECTIONS = {
    COHORT_COLUMN_WINDOW_MISMATCH: (
        f"For each {COHORT_COLUMN_WINDOW_MISMATCH}: the host filters a column as "
        "it was summarized, so a predicate reads only that column's window (an "
        "event read by its <concept>_time reads any window inside it).  When "
        "the question states no window for the criterion and the plan chose "
        "this one, restate the predicate's time_window as the column's window.  "
        "When the question states the window, do not restate it.  "
        + _NOT_APPLIED
        + " A column summarized over the stated window is the user's to extract."
    ),
    COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION: (
        f"For each {COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION}: the host "
        "compares an event's time with a window in hours after ICU admission, "
        "and this input types that time otherwise, or with only its origin or "
        "only its unit.  "
        + _NOT_APPLIED
        + " An event time in hours after ICU admission is the user's to extract."
    ),
}


def _require_primary_event_windows_readable(*, plan: Any, context: Any) -> None:
    """Refuse a primary cohort the builder would read over the whole stay.

    Legacy contexts too: without a sealed roster, the context's own columns
    stand in for the input's (``context_materialized_columns``).  The plan's
    cohort is read as the plan holds it, not validated again: its concepts
    were checked when the plan was built, in that plan's concept scope, and
    an ``AnalysisPlan`` always holds a ``CohortDefinition``.  A cohort whose
    columns do not bind is left to the checks that report it.
    """

    whole_stay = whole_stay_event_columns(context)
    definition = getattr(plan, "cohort", None)
    if not whole_stay or not isinstance(definition, CohortDefinition):
        return
    columns = context_materialized_columns(context)
    try:
        bindings = _planner_declared_context_column_bindings(
            definition=definition,
            plan=plan,
            context=context,
            columns=columns,
        )
    except CohortDataError:
        return
    found = predicates_read_over_the_whole_stay(
        definition,
        columns=columns,
        whole_stay_columns=whole_stay,
        column_bindings=bindings,
    )
    if found:
        raise CohortSchemaError(
            f"{COHORT_EVENT_WINDOW_UNREADABLE}: "
            + "; ".join(item.description() for item in found)
            + ". The cohort builder would read the whole stay instead. "
            + _NOT_APPLIED
            + " A reading bounded in time needs an input that records the "
            "event's time as <concept>_time, which is the user's to extract."
        )


def validate_plan_cohort_predicates_against_context(
    *,
    plan: Any,
    context: Any,
) -> None:
    """Compatibility alias for the expanded typed-plan binding gate."""

    validate_plan_typed_bindings_against_context(plan=plan, context=context)


def coerce_isfinite_safe_dtypes(frame: Any) -> Any:
    """Downcast pandas extension/object scalars to numpy ``isfinite``-safe dtypes.

    The universe builder emits per-concept aggregates as pandas *nullable*
    extension dtypes (``Int64`` / ``Float64`` / ``boolean``), or as object
    columns holding python bools, whenever the aggregate is mostly null.
    Generated causal / prediction code does ``design_df[col].to_numpy()`` and
    feeds the result to ``np.isfinite``; on a nullable or object array numpy
    raises ``ufunc 'isfinite' not supported for the input types`` and a primary
    estimate can be silently lost. Nullable numeric columns therefore become
    ``float64`` (NA -> NaN). Complete logical columns become numpy ``bool`` so
    their sealed boolean domain is not silently rewritten as numeric 0/1;
    logical columns with missing values still use ``float64`` because numpy has
    no non-object boolean representation with NA. Genuine string categoricals
    remain untouched for dummy-encoding.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        return frame

    to_coerce = []
    for col in frame.columns:
        series = frame[col]
        dtype = series.dtype
        if pd.api.types.is_extension_array_dtype(dtype) and (
            pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
        ):
            to_coerce.append(col)  # nullable Int64 / Float64 / boolean
        elif pd.api.types.is_object_dtype(dtype):
            non_null = series.dropna()
            if (
                len(non_null)
                and non_null.map(lambda v: isinstance(v, (bool, np.bool_))).all()
            ):
                to_coerce.append(col)  # object column holding python bools

    if not to_coerce:
        return frame

    out = frame.copy()
    for col in to_coerce:
        series = out[col]
        is_logical = pd.api.types.is_bool_dtype(series.dtype) or bool(
            len(series.dropna())
            and series.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all()
        )
        if is_logical and not bool(series.isna().any()):
            out[col] = series.astype("bool")
        else:
            out[col] = pd.to_numeric(series, errors="coerce").astype("float64")
    return out


def materialize_locked_analysis_cohort(
    *,
    run_dir: Path,
    plan: Any,
    universe_path: Path,
    context: Any = None,
    stem: str = "cohort_analysis",
    cohort_concept_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Apply the locked cohort definition to the universe → analysis cohort.

    This is the missing bridge between *declaring* a cohort (the locked
    ``CohortDefinition``, recorded for provenance) and *enforcing* it on the
    data the analysis steps consume. Without it, the universe-mode flow hands
    every step the unfiltered universe and silently relies on each LLM-generated
    step to re-apply inclusion/exclusion — which is unenforced and inconsistent.

    Reuses the deterministic, auditable ``build_cohort`` evaluator. Returns a
    result dict; ``status`` is one of ``applied`` (wrote ``<stem>.parquet`` +
    provenance), ``no_definition`` (nothing to apply → caller uses the universe),
    or ``error`` (predicates could not be evaluated → caller falls back to the
    universe so the run still proceeds).
    """
    result: Dict[str, Any] = {
        "status": "no_definition",
        "path": None,
        "flow_path": None,
        "authority_path": None,
        "authority_ref": None,
        "cohort_definition_sha256": None,
        "n_universe": None,
        "n_cohort": None,
        "error": None,
    }
    with cohort_concept_id_scope(cohort_concept_ids):
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        definition_sha = (
            cohort_definition_sha(definition) if definition is not None else None
        )
    if not cohort_definition_has_explicit_selection(definition):
        return result
    from ..intake.materialized_metadata import (
        MaterializedMetadataError,
        implementation_bundle_sha256,
        load_verified_materialized_cohort_authority,
        publish_ordered_subset_materialized_cohort,
        read_verified_materialized_cohort_table,
    )

    # Authority verification deliberately happens outside the legacy error
    # fallback below. A typed cohort that loses or corrupts its authority must
    # fail closed rather than silently becoming an untyped universe.
    typed_parent = load_verified_materialized_cohort_authority(universe_path)
    try:
        import pandas as pd  # type: ignore

        universe = (
            read_verified_materialized_cohort_table(
                universe_path,
                verified=typed_parent,
            ).to_pandas()
            if typed_parent is not None
            else pd.read_parquet(universe_path)
        )
        filter_input = (
            universe.reset_index(drop=True) if typed_parent is not None else universe
        )
        column_bindings = _planner_declared_context_column_bindings(
            definition=definition,
            plan=plan,
            context=context,
            columns=filter_input.columns,
        )
        cohort, cohort_flow = _build_cohort_with_flow(
            definition,
            filter_input,
            column_bindings=column_bindings,
            whole_stay_columns=whole_stay_event_columns(context),
        )
    except Exception as exc:
        if typed_parent is not None:
            raise MaterializedMetadataError(
                "typed cohort definition could not be applied to its sealed universe"
            ) from exc
        # Preserve the historical best-effort behavior only for legacy inputs.
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return result

    out_path = Path(run_dir) / f"{stem}.parquet"
    predicate_bindings = _predicate_column_binding_records(definition, column_bindings)
    semantic_provenance = {
        "schema_version": (
            "easyicu.analysis_cohort/2"
            if typed_parent is not None
            else "easyicu.analysis_cohort/1"
        ),
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "universe_parquet": str(universe_path),
        "cohort_definition": definition.to_dict(),
        "cohort_sha256": definition_sha,
        "n_universe": int(len(universe)),
        "n_analysis_cohort": int(len(cohort)),
        "predicate_column_bindings": predicate_bindings,
        "cohort_flow": cohort_flow,
    }
    authority_ref = None
    authority_path = None
    if typed_parent is not None:
        selected_positions = tuple(int(index) for index in cohort.index.tolist())
        verified_child = publish_ordered_subset_materialized_cohort(
            universe_path,
            out_path,
            selected_row_positions=selected_positions,
            semantic_provenance=semantic_provenance,
            producer_implementation_sha256=implementation_bundle_sha256(
                (
                    Path(__file__),
                    Path(__file__).resolve().parents[1]
                    / "planning"
                    / "cohort_contract.py",
                    Path(__file__).resolve().parents[1]
                    / "intake"
                    / "materialized_metadata.py",
                )
            ),
            producer_parameters={
                "cohort_definition": definition.to_dict(),
                "cohort_definition_sha256": definition_sha,
                "predicate_column_bindings": predicate_bindings,
                "stem": stem,
            },
            expected_parent_authority=typed_parent.reference,
        )
        if verified_child is None:  # pragma: no cover - typed parent selected above
            raise RuntimeError("typed analysis cohort publication lost authority")
        authority_ref = verified_child.reference.to_dict()
        authority_path = out_path.parent / verified_child.reference.file
    else:
        cohort = coerce_isfinite_safe_dtypes(cohort).reset_index(drop=True)
        cohort.to_parquet(out_path, index=False)
        # Anchor the exact parquet bytes in the ledger: without this, a later
        # plan-phase adoption could only verify definition digest + row count,
        # leaving same-row-count content drift undetectable on the legacy /1
        # branch (the typed-parent branch is anchored by its authority sidecar).
        semantic_provenance["cohort_parquet_sha256"] = _file_sha256(out_path)
        (Path(run_dir) / f"{stem}_provenance.json").write_text(
            json.dumps(semantic_provenance, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    flow_path = Path(run_dir) / f"{stem}_flow.csv"
    pd.DataFrame(cohort_flow).to_csv(flow_path, index=False)
    result.update(
        status="applied",
        path=out_path,
        flow_path=flow_path,
        authority_path=authority_path,
        authority_ref=authority_ref,
        cohort_definition_sha256=definition_sha,
        n_universe=int(len(universe)),
        n_cohort=int(len(cohort)),
    )
    return result


def load_materialized_analysis_cohort_result(
    *,
    run_dir: Path,
    plan: Any,
    stem: str = "cohort_analysis",
    cohort_concept_ids: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """Recover a plan-phase materialization only from its closed host ledger."""

    cohort_path = Path(run_dir) / f"{stem}.parquet"
    flow_path = Path(run_dir) / f"{stem}_flow.csv"
    provenance_path = Path(run_dir) / f"{stem}_provenance.json"
    if not (
        cohort_path.is_file() and flow_path.is_file() and provenance_path.is_file()
    ):
        return None
    with cohort_concept_id_scope(cohort_concept_ids):
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        if definition is None:
            return None
        expected_definition_sha = cohort_definition_sha(definition)
    try:
        import pandas as pd  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("cohort_sha256") != expected_definition_sha:
            return None
        recorded_parquet_sha = str(
            provenance.get("cohort_parquet_sha256") or ""
        ).strip()
        # This is an authority recovery path, not a best-effort cache: the
        # bytes on disk must be proved to be the ones the materialization
        # closed, or adoption is refused.  There are two proofs because there
        # are two ledgers.  ``cohort_parquet_sha256`` anchors the untyped
        # ``analysis_cohort/1`` ledger, which has nothing else.  The typed
        # ``/2`` branch never writes that key -- it publishes a content-
        # addressed authority sidecar instead -- so requiring the key alone
        # refused every typed materialization the host had just performed.
        # Measured over the recorded corpus: 164 of 164 ledgers are ``/2`` and
        # none carries the key, so this recovery had never once succeeded, and
        # the cohort-definition step it exists to adopt was written by the
        # Coder in 127 of 127 runs.
        verified_authority = None
        if recorded_parquet_sha:
            if _file_sha256(cohort_path) != recorded_parquet_sha:
                return None
        else:
            from ..intake.materialized_metadata import (
                load_verified_materialized_cohort_authority,
            )

            # Strictly stronger than the digest it stands in for: this pins the
            # parquet bytes, size, row count, column list, schema digest and a
            # per-row identity digest, and requires the sidecar's semantic
            # provenance to equal this ledger.  A missing or broken authority
            # yields None (or raises into the handler below), so a ledger with
            # neither proof still fails closed.
            verified_authority = load_verified_materialized_cohort_authority(
                cohort_path
            )
            if verified_authority is None:
                return None
        flow = pd.read_csv(flow_path)
        if flow.empty:
            return None
        flow_records = (
            flow.astype(object).where(pd.notna(flow), None).to_dict(orient="records")
        )
        n_universe = int(provenance["n_universe"])
        n_cohort = int(provenance["n_analysis_cohort"])
        if (
            flow_records != provenance.get("cohort_flow")
            or provenance.get("cohort_definition") != definition.to_dict()
            or int(flow.iloc[0]["n_before"]) != n_universe
            or int(flow.iloc[-1]["n_remaining"]) != n_cohort
            or int(pq.ParquetFile(cohort_path).metadata.num_rows) != n_cohort
        ):
            return None
    except Exception:
        # This is an authority recovery path: malformed JSON/CSV/Parquet or a
        # missing optional reader must disable adoption, never weaken it.
        return None
    return {
        "status": "applied",
        "path": cohort_path,
        "flow_path": flow_path,
        # Report the authority this adoption actually verified, so a recovered
        # result carries the same reference a fresh materialization would.
        "authority_path": (
            cohort_path.parent / verified_authority.reference.file
            if verified_authority is not None
            else None
        ),
        "authority_ref": (
            verified_authority.reference.to_dict()
            if verified_authority is not None
            else provenance.get("materialized_cohort_authority_ref")
        ),
        "cohort_definition_sha256": expected_definition_sha,
        "n_universe": n_universe,
        "n_cohort": n_cohort,
        "error": None,
    }


def assert_cohort_definition_locked(
    *,
    run_dir: Path,
    plan: Any,
    cohort_concept_ids: Sequence[str] = (),
) -> None:
    with cohort_concept_id_scope(cohort_concept_ids):
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        if definition is None:
            definition = CohortDefinition(name="primary")
        locked_definition = _load_locked_cohort_definition(run_dir)
        definition_sha = cohort_definition_sha(definition)
        locked_sha = cohort_definition_sha(locked_definition)
    if locked_sha != definition_sha:
        raise CohortSchemaError(
            "cohort definition changed after plan lock; execute phase refuses "
            "to run an unlocked cohort"
        )


def build_cohort(
    definition: CohortDefinition,
    data: Any = None,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
    whole_stay_columns: Sequence[str] = (),
) -> Any:
    """Apply a CTAS definition to a stay-level dataframe.

    This MVP intentionally supports a small deterministic surface. The broader
    EasyICU concept loader remains responsible for extracting time-series
    concepts; this function filters already-materialised columns. The CTAS
    ``time_window`` and ``aggregation`` are locked for audit, but this filter
    step does not re-verify that an upstream loader materialised the column with
    the declared window/aggregation.
    """

    if data is None:
        raise NotImplementedError(
            "build_cohort currently requires a materialised dataframe; "
            "time-series concept extraction is handled by EasyICU loaders"
        )
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - pandas is a project dependency
        raise NotImplementedError(
            "pandas is required for CTAS dataframe filtering"
        ) from exc

    if not isinstance(data, pd.DataFrame):
        raise TypeError("build_cohort data must be a pandas DataFrame")
    cohort, _ = _build_cohort_with_flow(
        definition,
        data,
        column_bindings=column_bindings,
        whole_stay_columns=whole_stay_columns,
    )
    return cohort


def _build_cohort_with_flow(
    definition: CohortDefinition,
    data: Any,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
    whole_stay_columns: Sequence[str] = (),
) -> tuple[Any, list[Dict[str, Any]]]:
    """Apply locked predicates once and return their exact attrition ledger.

    ``whole_stay_columns`` are the columns of ``data`` that record an event
    over the whole stay (``whole_stay_event_columns``); a predicate that would
    read one over a finite window is refused before any is applied.
    """

    import pandas as pd  # type: ignore

    require_event_windows_readable(
        definition,
        columns=data.columns,
        whole_stay_columns=whole_stay_columns,
        column_bindings=column_bindings,
    )
    mask = pd.Series(True, index=data.index)
    flow: list[Dict[str, Any]] = [
        {
            "step_order": 0,
            "predicate_kind": "universe",
            "concept_id": None,
            "resolved_column": None,
            "aggregation": None,
            "op": None,
            "value": None,
            "n_before": int(len(data)),
            "n_excluded": 0,
            "n_remaining": int(len(data)),
            "n_excluded_missing": 0,
            **_event_time_flow_fields(None),
        }
    ]
    ordered = [
        *(("inclusion", predicate) for predicate in definition.inclusion),
        *(("exclusion", predicate) for predicate in definition.exclusion),
    ]
    for order, (kind, predicate) in enumerate(ordered, start=1):
        before = int(mask.sum())
        column = _resolve_predicate_column(
            data.columns,
            predicate.concept_id,
            predicate.aggregation,
            column_bindings=column_bindings,
        )
        # A column no stay of the input holds a value of reads nothing; one
        # the stays left at this criterion happen to miss is a missing value.
        if before and column is not None and columns_without_values(data, [column]):
            raise CohortPredicateColumnWithoutValuesError(
                kind, predicate.concept_id, column, int(len(data))
            )
        predicate_mask, event_time_window, unrecorded = _predicate_mask(
            data,
            predicate,
            column_bindings=column_bindings,
        )
        keep = predicate_mask if kind == "inclusion" else ~predicate_mask
        excluded = mask & ~keep
        mask &= keep
        remaining = int(mask.sum())
        flow.append(
            {
                "step_order": order,
                "predicate_kind": kind,
                "concept_id": predicate.concept_id,
                "resolved_column": column,
                "aggregation": predicate.aggregation,
                "op": predicate.op,
                "value": predicate.value,
                "n_before": before,
                "n_excluded": before - remaining,
                "n_remaining": remaining,
                # Of those excluded, the stays the predicate read without a
                # recorded value: a criterion that could not be read is not
                # one that was unmet, and a reader of the ledger can tell them
                # apart only here.
                "n_excluded_missing": int((excluded & unrecorded).sum()),
                # The mask above is the only authority on what this predicate
                # did; the same call that built it reports the window it used,
                # so the ledger cannot describe a filter that was not applied.
                **_event_time_flow_fields(event_time_window),
            }
        )
    return data.loc[mask].copy(), flow


#: Why a cohort predicate cannot be read over its window on this input.
COHORT_EVENT_WINDOW_UNREADABLE = "cohort_event_window_unreadable"


@dataclass(frozen=True)
class WholeStayEventWindow:
    """A predicate that reads an event over a finite window from a whole-stay status.

    ``column`` records whether the event happened at any time in the ICU
    stay, and the input has no ``<concept>_time`` to place it in the window.
    """

    label: str
    concept_id: str
    column: str
    anchor: str
    start_offset_hours: float
    end_offset_hours: float

    def description(self) -> str:
        return (
            f"{self.label} reads whether the event of {self.concept_id!r} happened "
            f"within {self.anchor}[{self.start_offset_hours:g}, "
            f"{self.end_offset_hours:g}) h, but this input records {self.column!r} "
            "over the whole ICU stay and has no "
            f"{self.concept_id + '_time'!r} to place the event in that window"
        )


#: Why a cohort predicate reads nothing on this input: its column holds no
#: value in the stays it is applied to (``contracts.concept_values``).
COHORT_PREDICATE_COLUMN_WITHOUT_VALUES = "cohort_predicate_column_without_values"


class CohortPredicateColumnWithoutValuesError(CohortDataError):
    """A cohort predicate's column holds no value in any stay of the input.

    Applied, it would read nothing: an inclusion would keep no stay and an
    exclusion would exclude none, while the plan and its ledger state a
    criterion that was applied.
    """

    code = COHORT_PREDICATE_COLUMN_WITHOUT_VALUES

    def __init__(self, kind: str, concept_id: str, column: str, stays: int) -> None:
        self.kind = kind
        self.concept_id = concept_id
        self.column = column
        self.stays = stays
        super().__init__(
            f"{COHORT_PREDICATE_COLUMN_WITHOUT_VALUES}: the {kind} over "
            f"{concept_id!r} reads {column!r}, which holds no value in any of the "
            f"input's {stays} stays"
        )

    def __str__(self) -> str:
        # KeyError would quote the message.
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.kind, self.concept_id, self.column, self.stays)


class CohortEventWindowUnreadableError(CohortDataError):
    """The cohort reads an event over a window its input cannot place it in."""

    code = COHORT_EVENT_WINDOW_UNREADABLE

    def __init__(self, windows: Sequence[WholeStayEventWindow]) -> None:
        self.windows = tuple(windows)
        super().__init__(
            f"{COHORT_EVENT_WINDOW_UNREADABLE}: "
            + "; ".join(window.description() for window in self.windows)
        )

    def __str__(self) -> str:
        # KeyError would quote the message.
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.windows,)


def predicates_read_over_the_whole_stay(
    definition: CohortDefinition,
    *,
    columns: Any,
    whole_stay_columns: Any,
    column_bindings: Optional[Dict[str, str]] = None,
    label: str = "cohort",
) -> tuple[WholeStayEventWindow, ...]:
    """The predicates that read an event over a finite window from a whole-stay status.

    The builder reads such a window by the event's own time, the
    ``<concept>_time`` column beside the status
    (``_refine_occurrence_mask_by_event_time``).  Without that column it reads
    the status as it is, over the whole stay: an exclusion of the event
    within 24 h would remove every stay with the event.  ``columns`` and
    ``column_bindings`` are those the builder resolves a predicate with, and
    ``whole_stay_columns`` the columns among them that record an event over
    the whole stay (``whole_stay_event_columns``).
    """

    whole_stay = {str(column) for column in whole_stay_columns or ()}
    if not whole_stay:
        return ()
    available = {str(column) for column in columns}
    found: list[WholeStayEventWindow] = []
    for kind, predicates in (
        ("inclusion", definition.inclusion),
        ("exclusion", definition.exclusion),
    ):
        for index, predicate in enumerate(predicates):
            window = predicate.time_window
            end = float(window.end_offset_hours)
            if (
                _event_time_reading(predicate) is None
                or not math.isfinite(end)
                or f"{predicate.concept_id}_time" in available
            ):
                continue
            column = _resolve_predicate_column(
                available,
                predicate.concept_id,
                predicate.aggregation,
                column_bindings=column_bindings,
            )
            if column in whole_stay:
                found.append(
                    WholeStayEventWindow(
                        label=f"{label}.{kind}[{index}]",
                        concept_id=predicate.concept_id,
                        column=column,
                        anchor=str(window.anchor),
                        start_offset_hours=float(window.start_offset_hours),
                        end_offset_hours=end,
                    )
                )
    return tuple(found)


def require_event_windows_readable(
    definition: CohortDefinition,
    *,
    columns: Any,
    whole_stay_columns: Any,
    column_bindings: Optional[Dict[str, str]] = None,
    label: str = "cohort",
) -> None:
    """Refuse a cohort the builder would read over the whole stay."""

    found = predicates_read_over_the_whole_stay(
        definition,
        columns=columns,
        whole_stay_columns=whole_stay_columns,
        column_bindings=column_bindings,
        label=label,
    )
    if found:
        raise CohortEventWindowUnreadableError(found)


@dataclass(frozen=True)
class ColumnWindowMismatch:
    """A predicate stating another window than its column was summarized over.

    ``by_event_time`` marks an event read by the event's own time, which
    reads any window inside the column's; any other predicate reads exactly
    the column's window.
    """

    label: str
    concept_id: str
    column: str
    column_window: str
    anchor: str
    start_offset_hours: float
    end_offset_hours: float
    by_event_time: bool = False

    def description(self) -> str:
        stated = (
            f"{self.anchor}[{self.start_offset_hours:g}, "
            f"{self.end_offset_hours:g}) h"
        )
        if self.by_event_time:
            return (
                f"{self.label} reads the event of {self.concept_id!r} within "
                f"{stated} by its time, but this input records {self.column!r} "
                f"only over {self.column_window}"
            )
        return (
            f"{self.label} reads {self.column!r} within {stated}, but this input "
            f"summarizes {self.column!r} over {self.column_window}"
        )


class CohortColumnWindowMismatchError(CohortDataError):
    """The cohort reads a column over another window than it was summarized over."""

    code = COHORT_COLUMN_WINDOW_MISMATCH

    def __init__(self, windows: Sequence[ColumnWindowMismatch]) -> None:
        self.windows = tuple(windows)
        super().__init__(
            f"{COHORT_COLUMN_WINDOW_MISMATCH}: "
            + "; ".join(window.description() for window in self.windows)
        )

    def __str__(self) -> str:
        # KeyError would quote the message.
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.windows,)


@dataclass(frozen=True)
class EventTimeNotHoursFromAdmission:
    """A predicate that reads its event's time, which the input times otherwise.

    ``origin`` and ``unit`` are those the input types for the time, ``None``
    for one it does not state.
    """

    label: str
    concept_id: str
    time_column: str
    origin: Optional[str] = None
    unit: Optional[str] = None

    def description(self) -> str:
        if self.origin and self.unit:
            typed = f"in {self.unit!r} from {self.origin!r}"
        elif self.unit:
            typed = f"in {self.unit!r} from an origin it does not state"
        elif self.origin:
            typed = f"from {self.origin!r} in a unit it does not state"
        else:
            typed = "in other than hours after ICU admission"
        return (
            f"{self.label} reads the event of {self.concept_id!r} within its "
            f"window in hours after ICU admission by {self.time_column!r}, which "
            f"this input times {typed}"
        )


class CohortEventTimeNotHoursError(CohortDataError):
    """The cohort compares an event time in other units with a window in hours."""

    code = COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION

    def __init__(self, times: Sequence[EventTimeNotHoursFromAdmission]) -> None:
        self.times = tuple(times)
        super().__init__(
            f"{COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION}: "
            + "; ".join(item.description() for item in self.times)
        )

    def __str__(self) -> str:
        return str(self.args[0])

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.times,)


def predicates_read_through_another_window(
    definition: CohortDefinition,
    *,
    columns: Any,
    column_windows: Mapping[str, ColumnWindow],
    column_bindings: Optional[Dict[str, str]] = None,
    label: str = "cohort",
) -> tuple[ColumnWindowMismatch, ...]:
    """The predicates that state another window than their column was summarized over.

    The builder filters a column as it was summarized; a predicate's window is
    locked for audit only.  So a predicate reads exactly its column's window,
    whatever window it states.  An event the builder reads by its own time,
    the ``<concept>_time`` beside its status, reads any window inside the one
    its status and its time were recorded over.  ``column_windows`` are the
    windows the input's columns were summarized over
    (``context_column_windows``); a column without one is not judged here.
    """

    if not column_windows:
        return ()
    available = {str(column) for column in columns}
    found: list[ColumnWindowMismatch] = []
    for kind, predicates in (
        ("inclusion", definition.inclusion),
        ("exclusion", definition.exclusion),
    ):
        for index, predicate in enumerate(predicates):
            column = _resolve_predicate_column(
                available,
                predicate.concept_id,
                predicate.aggregation,
                column_bindings=column_bindings,
            )
            window = column_windows.get(str(column)) if column is not None else None
            if window is None:
                continue
            stated = predicate.time_window
            start = float(stated.start_offset_hours)
            end = float(stated.end_offset_hours)
            time_column = f"{predicate.concept_id}_time"
            by_event_time = (
                _event_time_reading(predicate) is not None
                and math.isfinite(end)
                and time_column in available
            )
            if by_event_time:
                time_window = column_windows.get(time_column)
                readable = window.contains(stated.anchor, start, end) and (
                    time_window is None
                    or time_window.contains(stated.anchor, start, end)
                )
            else:
                readable = window.is_window(stated.anchor, start, end)
            if not readable:
                found.append(
                    ColumnWindowMismatch(
                        label=f"{label}.{kind}[{index}]",
                        concept_id=predicate.concept_id,
                        column=str(column),
                        column_window=window.description(),
                        anchor=str(stated.anchor),
                        start_offset_hours=start,
                        end_offset_hours=end,
                        by_event_time=by_event_time,
                    )
                )
    return tuple(found)


def predicates_read_by_an_event_time_not_in_hours(
    definition: CohortDefinition,
    *,
    columns: Any,
    event_times_not_in_hours: Any,
    label: str = "cohort",
) -> tuple[EventTimeNotHoursFromAdmission, ...]:
    """The predicates the builder would read by an event time the input times otherwise.

    The builder compares ``<concept>_time`` with a window in hours after ICU
    admission (``_refine_occurrence_mask_by_event_time``).  An input that
    types that column in days, in minutes, from another origin, or with only
    one of the two (``event_times_typed_otherwise_than_hours``, which maps
    each to the origin and unit typed for it) would have it read as hours
    after ICU admission.
    """

    other = {str(column) for column in event_times_not_in_hours or ()}
    if not other:
        return ()
    typed = (
        event_times_not_in_hours
        if isinstance(event_times_not_in_hours, Mapping)
        else {}
    )
    available = {str(column) for column in columns}
    found: list[EventTimeNotHoursFromAdmission] = []
    for kind, predicates in (
        ("inclusion", definition.inclusion),
        ("exclusion", definition.exclusion),
    ):
        for index, predicate in enumerate(predicates):
            time_column = f"{predicate.concept_id}_time"
            if (
                _event_time_reading(predicate) is None
                or not math.isfinite(float(predicate.time_window.end_offset_hours))
                or time_column not in available
                or time_column not in other
            ):
                continue
            origin, unit = typed.get(time_column) or (None, None)
            found.append(
                EventTimeNotHoursFromAdmission(
                    label=f"{label}.{kind}[{index}]",
                    concept_id=predicate.concept_id,
                    time_column=time_column,
                    origin=origin,
                    unit=unit,
                )
            )
    return tuple(found)


def require_column_windows_readable(
    definition: CohortDefinition,
    *,
    columns: Any,
    column_windows: Mapping[str, ColumnWindow],
    event_times_not_in_hours: Any = (),
    column_bindings: Optional[Dict[str, str]] = None,
    label: str = "cohort",
) -> None:
    """Refuse a cohort the builder would read over another window than it states."""

    found = predicates_read_through_another_window(
        definition,
        columns=columns,
        column_windows=column_windows,
        column_bindings=column_bindings,
        label=label,
    )
    if found:
        raise CohortColumnWindowMismatchError(found)
    times = predicates_read_by_an_event_time_not_in_hours(
        definition,
        columns=columns,
        event_times_not_in_hours=event_times_not_in_hours,
        label=label,
    )
    if times:
        raise CohortEventTimeNotHoursError(times)


def _catalog_output_stems(concept_id: str) -> tuple[str, ...]:
    """Return catalog-owned output stems for one extraction source.

    Composite-loader output names belong to the EasyICU concept catalog, not
    the research-agent cohort engine.  Import lazily to keep this execution leaf
    free of a module-import dependency on the catalog/UI layer.
    """

    from easyicu.concept_output_sources import COMPOSITE_CONCEPT_OUTPUT_SOURCES

    return tuple(
        sorted(
            output
            for output, source in COMPOSITE_CONCEPT_OUTPUT_SOURCES.items()
            if str(source).strip() == str(concept_id).strip()
        )
    )


def _resolve_predicate_column(
    columns: Any,
    concept_id: str,
    aggregation: str,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a predicate ``concept_id`` to an actual universe column.

    The universe wide table names id-level concepts bare (``age``, ``los_icu``,
    ``death``) and time-series concepts as ``<output>_<aggregation>``
    (``aki_stage_reference_max`` …). A predicate carries the *dictionary*
    ``concept_id``
    plus the requested ``aggregation``; resolve against the columns present,
    trying in order: an explicit Planner/context binding, the bare id, the wide
    ``<concept_id>_<aggregation>`` form, and unambiguous catalog-owned composite
    outputs. Return ``None`` when no unique column honours the contract, so the
    caller can fail loudly rather than silently choose a sibling output.
    """
    cols = set(columns)
    if concept_id in cols:
        return concept_id
    aggregated = f"{concept_id}_{aggregation}"
    if aggregated in cols:
        return aggregated
    bound = str((column_bindings or {}).get(concept_id) or "").strip()
    if bound and bound in cols:
        return bound
    catalog_candidates: set[str] = set()
    for stem in _catalog_output_stems(concept_id):
        if stem in cols:
            catalog_candidates.add(stem)
        stem_aggregated = f"{stem}_{aggregation}"
        if stem_aggregated in cols:
            catalog_candidates.add(stem_aggregated)
    return next(iter(catalog_candidates)) if len(catalog_candidates) == 1 else None


def _predicate_mask(
    data: Any,
    pred: ConceptPredicate,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
) -> tuple[Any, Optional["AppliedEventTimeWindow"], Any]:
    """The predicate's mask, its event-time window and its unrecorded rows.

    The last is ``_unrecorded_value_mask``: the rows read without a value.
    """

    if pred.aggregation not in _IMPLEMENTED_AGGREGATIONS:
        raise NotImplementedError(
            f"aggregation {pred.aggregation!r} is not implemented by the CTAS "
            "dataframe builder"
        )
    column = _resolve_predicate_column(
        data.columns,
        pred.concept_id,
        pred.aggregation,
        column_bindings=column_bindings,
    )
    if column is None:
        raise CohortDataError(
            f"cohort dataframe is missing concept column {pred.concept_id!r} "
            f"(also tried {pred.concept_id}_{pred.aggregation}, an explicit "
            "Planner binding, and unambiguous catalog outputs)"
        )
    series = data[column]
    mask = _apply_op(series, pred.op, pred.value)
    mask, window = _refine_occurrence_mask_by_event_time(
        data, pred, mask, status=series
    )
    return mask, window, _unrecorded_value_mask(data, series, window)


@dataclass(frozen=True)
class AppliedEventTimeWindow:
    """The event-time window a predicate mask actually consulted.

    The attrition ledger publishes ``resolved_column``, ``op`` and ``value``,
    and the Coder authority prompt tells the Coder to reproduce the recorded
    before/excluded/remaining counts from them. For a predicate that
    ``_refine_occurrence_mask_by_event_time`` narrowed, those three fields are
    not the whole predicate: the mask also consulted a second column. Correct
    generated code then computes a different count and fails closed, which is
    the right behaviour against a receipt that under-describes its own filter.

    So the owner that applies the refinement is the owner that describes it:
    this record is produced by the same call that builds the mask and is
    written straight into the ledger row, leaving no second place for the two
    to drift apart. ``None`` means the predicate was applied exactly as the
    ledger's ordinary fields state.

    ``reading`` says which question the window answered: ``"occurrence"``
    (the event happened within it) or ``"absence"`` (it did not).
    """

    event_time_column: str
    start_offset_hours: float
    end_offset_hours: float
    reading: str


def _event_time_flow_fields(
    refinement: Optional[AppliedEventTimeWindow],
) -> Dict[str, Any]:
    """Render one refinement as flat ledger fields.

    Flat rather than nested because every other field of a flow row is flat and
    the same rows are written verbatim to ``<stem>_flow.csv`` through
    ``pd.DataFrame``; a nested object would land in that CSV as a repr string.
    Unrefined predicates carry the keys with ``None`` so the ledger's schema
    does not depend on which predicates a plan happened to declare.
    """

    return {
        "event_time_column": refinement.event_time_column if refinement else None,
        "event_time_start_hours": (
            float(refinement.start_offset_hours) if refinement else None
        ),
        "event_time_end_hours": (
            float(refinement.end_offset_hours) if refinement else None
        ),
        "event_time_reading": refinement.reading if refinement else None,
    }


#: Every ``<concept>_time`` column is in hours from ICU admission.
_EVENT_TIME_ANCHORS = frozenset({"icu_admit", "icu_admission"})


def _event_time_reading(pred: ConceptPredicate) -> Optional[str]:
    """Whether a predicate asks that an event happened, or that it did not.

    The planning contract owns the rule (``event_status_reading``); the
    time-zero rule judges a windowed event predicate by the same one.
    """

    return event_status_reading(pred.op, pred.value)


def _refine_occurrence_mask_by_event_time(
    data: Any,
    pred: ConceptPredicate,
    mask: Any,
    *,
    status: Any,
) -> tuple[Any, Optional[AppliedEventTimeWindow]]:
    """Read an event predicate over its window by the event's own time.

    ``build_cohort`` filters an already-materialised wide table and, by design,
    does not re-window the summary columns. That is correct for a concept whose
    column was summarised WITHIN the predicate window, but an OUTCOME concept is
    materialised whole-stay (``death`` is 1 whenever the patient ever died)
    alongside an event-time column (``death_time`` = hours from ICU admission).
    A bounded-window predicate on such a concept -- a landmark exclusion written
    to avoid immortal-time bias, or an inclusion of the stays that survived the
    window -- must therefore consult the event time. Otherwise the whole-stay
    flag decides for every event, not just the in-window ones: "no death within
    24 h" would keep only the stays that never died.

    A predicate that names one event level (``_event_time_reading``) over a
    finite window on a concept carrying a ``<concept>_time`` sibling is read by
    that time:

    - ``occurrence``: the op and value hold AND the event time lies within the
      window;
    - ``absence``: the op and value hold OR the stay's event (status 1) lies
      outside the window.

    The window is ``[start_offset_hours, end_offset_hours)``, as
    ``TimeWindow`` states it.  An event time may be a row's chart time on the
    hourly grid, where a row at hour ``h`` was charted in ``[h, h + 1)``, so an
    event at the window's end happened after it, as an event at a landmark
    happens after the landmark.  A missing event time lies outside every window. A window that no recorded
    time could place is refused rather than read as "no event in it": every
    event in the table lacks a time (a source that records none), or the
    predicate is anchored elsewhere than at ICU admission, the origin of every
    ``<concept>_time``.

    Magnitude filters (age>=18, los>=1), missingness checks and concepts without
    an event-time column are untouched, so association runs with no event-time
    columns behave exactly as before.

    Returns the mask together with the window it consulted, or ``None`` when the
    predicate was left exactly as its ordinary fields describe it. Every early
    return below is a case the ledger's ``resolved_column``/``op``/``value``
    already describe on their own.

    ``pred.time_window`` and its ``end_offset_hours`` are taken as given:
    ``ConceptPredicate.__post_init__`` refuses a predicate without a window and
    ``TimeWindow.end_offset_hours`` is a required float, so guarding them here
    only hid a broken invariant behind a silently unrefined mask. An infinite
    end is a different matter -- ``_coerce_offset`` accepts ``"inf"`` for a
    deliberately unbounded window, which refines nothing and could not be
    published as a finite bound.
    """
    reading = _event_time_reading(pred)
    if reading is None:
        return mask, None
    tw = pred.time_window
    end = float(tw.end_offset_hours)
    if not math.isfinite(end):
        return mask, None
    start = float(tw.start_offset_hours)
    event_time_col = f"{pred.concept_id}_time"
    if event_time_col not in data.columns:
        return mask, None
    if str(tw.anchor).strip().lower() not in _EVENT_TIME_ANCHORS:
        raise CohortDataError(
            f"cohort predicate on {pred.concept_id!r} is anchored at "
            f"{tw.anchor!r}, but {event_time_col!r} is in hours from ICU admission"
        )
    event = _event_occurred(status)
    event_time = data[event_time_col]
    if bool(event.any()) and not bool(event_time[event].notna().any()):
        raise CohortDataError(
            f"cohort predicate on {pred.concept_id!r} reads its event within "
            f"[{start:g}, {end:g}) h, but no event in {event_time_col!r} has a "
            "recorded time"
        )
    in_window = ((event_time >= start) & (event_time < end)).fillna(False)
    refined = (
        mask & in_window if reading == "occurrence" else mask | (event & ~in_window)
    )
    return refined, AppliedEventTimeWindow(
        event_time_column=event_time_col,
        start_offset_hours=start,
        end_offset_hours=end,
        reading=reading,
    )


def _event_occurred(status: Any) -> Any:
    """The stays whose event status records the event (level 1)."""

    return (status == 1).fillna(False).astype(bool)


def _unrecorded_value_mask(
    data: Any,
    series: Any,
    window: Optional[AppliedEventTimeWindow],
) -> Any:
    """The rows a predicate read without a recorded value.

    Whatever the operator, such a row was placed by the value's absence,
    not by a value: a comparison reads a missing value as unmet (``!=`` and
    ``not_in`` as met) and a missingness check reads nothing else. A
    predicate read over an event-time window (``window``) also reads the
    event's time, and an event without one lies outside every window; a
    stay without the event needs no time. The attrition ledger counts
    these rows among each step's exclusions.
    """

    unrecorded = series.isna()
    if window is not None:
        unrecorded = unrecorded | (
            _event_occurred(series) & data[window.event_time_column].isna()
        )
    return unrecorded


def _apply_op(series: Any, op: str, value: Any) -> Any:
    if op == "==":
        return series == value
    if op == "!=":
        return series != value
    if op == "<":
        return series < value
    if op == "<=":
        return series <= value
    if op == ">":
        return series > value
    if op == ">=":
        return series >= value
    if op == "in":
        values = value if isinstance(value, list) else [value]
        return series.isin(values)
    if op == "not_in":
        values = value if isinstance(value, list) else [value]
        return ~series.isin(values)
    if op == "missing":
        return series.isna()
    if op == "not_missing":
        return series.notna()
    raise CohortSchemaError(f"unsupported predicate operator: {op}")


__all__ = [
    "ALLOWED_CTAS_AGGREGATIONS",
    "COHORT_COLUMN_WINDOW_MISMATCH",
    "COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION",
    "COHORT_EVENT_WINDOW_UNREADABLE",
    "COHORT_LOCK_FILENAME",
    "COHORT_PREDICATE_COLUMN_WITHOUT_VALUES",
    "CohortColumnWindowMismatchError",
    "CohortEventTimeNotHoursError",
    "ColumnWindowMismatch",
    "EventTimeNotHoursFromAdmission",
    "CohortAuthorityError",
    "CohortDefinition",
    "CohortDataError",
    "CohortEventWindowUnreadableError",
    "CohortPredicateColumnWithoutValuesError",
    "CohortSchemaError",
    "ConceptPredicate",
    "MaterializedInputColumnAuthority",
    "PatternRegistry",
    "TimeWindow",
    "UNIVERSAL_ANCHORS",
    "WholeStayEventWindow",
    "assert_cohort_definition_locked",
    "build_cohort",
    "coerce_cohort_definition",
    "clear_cohort_concept_ids",
    "cohort_concept_id_scope",
    "context_materialized_columns",
    "materialized_cohort_concept_id_scope",
    "cohort_definition_sha",
    "concept_id_exists",
    "default_pattern_registry",
    "ensure_cohort_definition",
    "expand_named_cohort",
    "known_concept_ids",
    "materialized_input_column_authority",
    "predicate_accepts_closed_level",
    "predicates_read_by_an_event_time_not_in_hours",
    "predicates_read_over_the_whole_stay",
    "predicates_read_through_another_window",
    "register_cohort_concept_ids",
    "registered_run_cohort_concept_ids",
    "register_pattern",
    "register_patterns_from_file",
    "require_column_windows_readable",
    "require_event_windows_readable",
    "reset_pattern_registry",
    "validate_cohort_definition",
    "validate_plan_cohort_predicates_against_context",
    "validate_plan_typed_bindings_against_context",
    "validate_concept_predicate",
    "write_locked_cohort_definition",
]
