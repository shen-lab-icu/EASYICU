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
from typing import Any, Dict, Mapping, Optional

from ..authority.lock_contract import (
    LockAuthorityError,
    assert_lock_matches_evidence_anchor,
)
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
    expand_named_cohort,
    known_concept_ids,
    register_cohort_concept_ids,
    register_pattern,
    register_patterns_from_file,
    reset_pattern_registry,
    validate_cohort_definition,
    validate_concept_predicate,
)

COHORT_LOCK_FILENAME = "cohort_locked.json"
_IMPLEMENTED_AGGREGATIONS = set(ALLOWED_CTAS_AGGREGATIONS)


class CohortDataError(KeyError):
    """Raised when materialised data cannot satisfy a CTAS definition."""


class CohortAuthorityError(RuntimeError):
    """Raised when a locked cohort definition cannot be enforced on the data.

    Distinct from :class:`CohortDataError`, which is about one predicate being
    unsatisfiable. This one means the run declared a cohort and then could not
    apply it, so any downstream number would describe a population the plan
    did not authorise.
    """


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> Path:
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if definition is None:
        definition = CohortDefinition(name="primary")
    validate_cohort_definition(definition)
    path = run_dir / COHORT_LOCK_FILENAME
    if path.exists():
        locked_definition = _load_locked_cohort_definition(run_dir)
        definition_sha = cohort_definition_sha(definition)
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
        "cohort_sha256": cohort_definition_sha(definition),
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
    return bool(
        normalized_aggregation in ALLOWED_CTAS_AGGREGATIONS
        and normalized_name.endswith(f"_{normalized_aggregation}")
    )


def _descriptor_window_matches_predicate(value: Any, window: TimeWindow) -> bool:
    """Match a named descriptor window to one exact predicate window.

    Cross-name binding is an execution convenience, not permission to infer a
    temporal contract.  Accept only explicit canonical names; missing or broad
    labels such as ``entire_stay`` fail closed.
    """

    raw = str(value or "").strip().casefold()
    if not raw:
        return False

    def _number(raw: float) -> str:
        number = float(raw)
        return str(int(number)) if number.is_integer() else f"{number:g}"

    anchor_aliases = {
        "icu_admit": {"icu_admit", "icu_admission"},
        "hospital_admit": {"hospital_admit", "hospital_admission"},
        "index_time": {"index_time"},
    }
    anchors = anchor_aliases.get(str(window.anchor).casefold(), {str(window.anchor)})
    start = _number(window.start_offset_hours)
    end = _number(window.end_offset_hours)
    accepted = {
        candidate
        for anchor in anchors
        for candidate in (
            f"{anchor}_{start}_{end}h",
            f"{anchor}[{start},{end}]h",
        )
    }
    normalized = re.sub(r"[^a-z0-9.]+", "_", raw).strip("_")
    normalized_accepted = {
        re.sub(r"[^a-z0-9.]+", "_", candidate).strip("_") for candidate in accepted
    }
    return raw in accepted or normalized in normalized_accepted


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
    # suffix alone cannot prove its scientific coordinate.  Validate a direct
    # column against its sealed descriptor even when the plan has no separate
    # cohort-materialisation step.  Cross-name bindings below remain restricted
    # to an explicit analysis-cohort producer.
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
    for descriptor in getattr(context, "variables", ()) or ():
        name = str(getattr(descriptor, "name", "") or "").strip()
        source_concept = str(getattr(descriptor, "source_concept", "") or "").strip()
        role = getattr(descriptor, "role", "")
        role_value = str(getattr(role, "value", role) or "").strip().casefold()
        if (
            not name
            or not source_concept
            or name not in declared_inputs
            or role_value in {"id", "meta", "time"}
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
        for input_index, value in enumerate(getattr(step, "inputs", ()) or ()):
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


def _predicate_accepts_closed_level(
    predicate: ConceptPredicate,
    level: Any,
) -> Optional[bool]:
    """Evaluate one typed predicate on a local closed level, if comparable."""

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
                accepted = _predicate_accepts_closed_level(predicate, level)
                if accepted is None:
                    indeterminate = True
                    break
                include = include and accepted
            if indeterminate:
                break
            for predicate in predicate_sets.get("exclusion", ()):
                excluded = _predicate_accepts_closed_level(predicate, level)
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

    materialized_inputs = getattr(context, "materialized_inputs", None)
    typed_cohort = getattr(materialized_inputs, "cohort", None)
    columns = tuple(getattr(typed_cohort, "cohort_columns", ()) or ())
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
    executable_columns = tuple(getattr(typed_cohort, "column_bindings", {}).keys())
    reserved_coordinates = tuple(sorted(set(columns) - set(executable_columns)))
    raw_issues = _raw_typed_plan_reference_issues(
        plan=plan,
        columns=executable_columns,
        reserved_coordinates=reserved_coordinates,
    )
    issues = list(raw_issues)
    primary_definition: Optional[CohortDefinition] = None
    primary_bindings: Dict[str, str] = {}
    for label, definition in definitions:
        try:
            bindings = _planner_declared_context_column_bindings(
                definition=definition,
                plan=plan,
                context=context,
                columns=columns,
            )
        except CohortDataError as exc:
            issues.append(f"{label}: {exc}")
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
    else:
        correction = (
            "Use an executable dictionary concept whose exact "
            "window/aggregation is bound by the declared analysis-cohort "
            f"columns={producer_columns!r} and source concepts={typed_sources!r}."
        )
    raise CohortSchemaError(
        "typed plan references are not executable against this sealed input. "
        f"Invalid references: {detail}. {correction} Additional binding context: "
        "declared "
        f"typed source concepts={typed_sources!r}; declared columns="
        f"{producer_columns!r}; executable cohort columns="
        f"{sorted(executable_columns)!r}; reserved navigation coordinates="
        f"{list(reserved_coordinates)!r}."
    )


def validate_plan_cohort_predicates_against_context(
    *,
    plan: Any,
    context: Any,
) -> None:
    """Compatibility alias for the expanded typed-plan binding gate."""

    validate_plan_typed_bindings_against_context(plan=plan, context=context)


def coerce_isfinite_safe_dtypes(frame: Any) -> Any:
    """Downcast pandas nullable-extension and boolean-object columns to numpy
    ``float64`` so downstream ``np.isfinite`` / ``to_numpy()`` in generated
    analysis code never receives an object or extension array.

    The universe builder emits per-concept aggregates as pandas *nullable*
    extension dtypes (``Int64`` / ``Float64`` / ``boolean``), or as object
    columns holding python bools, whenever the aggregate is mostly null.
    Generated causal / prediction code does ``design_df[col].to_numpy()`` and
    feeds the result to ``np.isfinite``; on a nullable or object array numpy
    raises ``ufunc 'isfinite' not supported for the input types`` and a primary
    estimate can be silently lost. Coercing these to ``float64`` (NA -> NaN) at
    cohort-materialisation time leaves every column as either a numpy numeric or
    a genuine string categorical -- the two shapes generated code already
    handles. True string/categorical object columns (for example a demographic
    category or admission type) are left untouched for dummy-encoding.
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
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def materialize_locked_analysis_cohort(
    *,
    run_dir: Path,
    plan: Any,
    universe_path: Path,
    context: Any = None,
    stem: str = "cohort_analysis",
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
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
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
        "cohort_sha256": cohort_definition_sha(definition),
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
                "cohort_definition_sha256": cohort_definition_sha(definition),
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
        cohort_definition_sha256=cohort_definition_sha(definition),
        n_universe=int(len(universe)),
        n_cohort=int(len(cohort)),
    )
    return result


def load_materialized_analysis_cohort_result(
    *,
    run_dir: Path,
    plan: Any,
    stem: str = "cohort_analysis",
) -> Optional[Dict[str, Any]]:
    """Recover a plan-phase materialization only from its closed host ledger."""

    cohort_path = Path(run_dir) / f"{stem}.parquet"
    flow_path = Path(run_dir) / f"{stem}_flow.csv"
    provenance_path = Path(run_dir) / f"{stem}_provenance.json"
    if not (
        cohort_path.is_file() and flow_path.is_file() and provenance_path.is_file()
    ):
        return None
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if definition is None:
        return None
    try:
        import pandas as pd  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_definition_sha = cohort_definition_sha(definition)
        if provenance.get("cohort_sha256") != expected_definition_sha:
            return None
        recorded_parquet_sha = str(
            provenance.get("cohort_parquet_sha256") or ""
        ).strip()
        # This is an authority recovery path, not a best-effort cache.  A
        # pre-digest ledger cannot prove which parquet bytes originally closed
        # the plan-phase materialization, so it must not be promoted into a new
        # successful checkpoint.  Fresh runs deterministically rematerialize
        # the cohort and write the digest; legacy runs fail closed instead of
        # silently blessing same-row-count content drift.
        if not recorded_parquet_sha:
            return None
        if _file_sha256(cohort_path) != recorded_parquet_sha:
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
        "authority_path": None,
        "authority_ref": provenance.get("materialized_cohort_authority_ref"),
        "cohort_definition_sha256": expected_definition_sha,
        "n_universe": n_universe,
        "n_cohort": n_cohort,
        "error": None,
    }


def assert_cohort_definition_locked(*, run_dir: Path, plan: Any) -> None:
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if definition is None:
        definition = CohortDefinition(name="primary")
    locked_definition = _load_locked_cohort_definition(run_dir)
    if cohort_definition_sha(locked_definition) != cohort_definition_sha(definition):
        raise CohortSchemaError(
            "cohort definition changed after plan lock; execute phase refuses "
            "to run an unlocked cohort"
        )


def build_cohort(
    definition: CohortDefinition,
    data: Any = None,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
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
    )
    return cohort


def _build_cohort_with_flow(
    definition: CohortDefinition,
    data: Any,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
) -> tuple[Any, list[Dict[str, Any]]]:
    """Apply locked predicates once and return their exact attrition ledger."""

    import pandas as pd  # type: ignore

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
            **_event_time_flow_fields(None),
        }
    ]
    ordered = [
        *(("inclusion", predicate) for predicate in definition.inclusion),
        *(("exclusion", predicate) for predicate in definition.exclusion),
    ]
    for order, (kind, predicate) in enumerate(ordered, start=1):
        before = int(mask.sum())
        predicate_mask, event_time_window = _predicate_mask(
            data,
            predicate,
            column_bindings=column_bindings,
        )
        keep = predicate_mask if kind == "inclusion" else ~predicate_mask
        mask &= keep
        remaining = int(mask.sum())
        flow.append(
            {
                "step_order": order,
                "predicate_kind": kind,
                "concept_id": predicate.concept_id,
                "resolved_column": _resolve_predicate_column(
                    data.columns,
                    predicate.concept_id,
                    predicate.aggregation,
                    column_bindings=column_bindings,
                ),
                "aggregation": predicate.aggregation,
                "op": predicate.op,
                "value": predicate.value,
                "n_before": before,
                "n_excluded": before - remaining,
                "n_remaining": remaining,
                # The mask above is the only authority on what this predicate
                # did; the same call that built it reports the window it used,
                # so the ledger cannot describe a filter that was not applied.
                **_event_time_flow_fields(event_time_window),
            }
        )
    return data.loc[mask].copy(), flow


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
    (``aki_stage_max`` …). A predicate carries the *dictionary* ``concept_id``
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
) -> tuple[Any, Optional["AppliedEventTimeWindow"]]:
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
    return _refine_occurrence_mask_by_event_time(data, pred, mask)


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
    """

    event_time_column: str
    start_offset_hours: float
    end_offset_hours: float


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
    }


def _refine_occurrence_mask_by_event_time(
    data: Any, pred: ConceptPredicate, mask: Any
) -> tuple[Any, Optional[AppliedEventTimeWindow]]:
    """Intersect an event-occurrence predicate with its event-time window.

    ``build_cohort`` filters an already-materialised wide table and, by design,
    does not re-window the summary columns. That is correct for a concept whose
    column was summarised WITHIN the predicate window, but an OUTCOME concept is
    materialised whole-stay (``death`` is 1 whenever the patient ever died)
    alongside an event-time column (``death_time`` = hours from the anchor). A
    bounded-window occurrence predicate on such a concept — for example, a
    landmark exclusion written to avoid immortal-time bias — must therefore
    consult the event time. Otherwise the whole-stay flag drops every event,
    not just the in-window ones.

    Scope is deliberately narrow: only a truthy ``==`` occurrence check over a
    finite window on a concept that actually carries a ``<concept>_time`` sibling
    column is refined. Magnitude filters (age>=18, los>=1) and concepts without
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
    tw = pred.time_window
    if pred.op != "==" or pred.value in (0, 0.0, False, None):
        return mask, None
    end = float(tw.end_offset_hours)
    if not math.isfinite(end):
        return mask, None
    event_time_col = f"{pred.concept_id}_time"
    if event_time_col not in data.columns:
        return mask, None
    event_time = data[event_time_col]
    start = float(tw.start_offset_hours)
    in_window = (event_time >= start) & (event_time <= end)
    # NaN event time (no event) -> not in window; keep the row's occurrence flag
    # from deciding membership only when the event genuinely falls in the window.
    try:
        in_window = in_window.fillna(False)
    except Exception:
        pass
    return mask & in_window, AppliedEventTimeWindow(
        event_time_column=event_time_col,
        start_offset_hours=start,
        end_offset_hours=end,
    )


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
    "COHORT_LOCK_FILENAME",
    "CohortAuthorityError",
    "CohortDefinition",
    "CohortDataError",
    "CohortSchemaError",
    "ConceptPredicate",
    "PatternRegistry",
    "TimeWindow",
    "UNIVERSAL_ANCHORS",
    "assert_cohort_definition_locked",
    "build_cohort",
    "coerce_cohort_definition",
    "clear_cohort_concept_ids",
    "cohort_concept_id_scope",
    "cohort_definition_sha",
    "concept_id_exists",
    "default_pattern_registry",
    "ensure_cohort_definition",
    "expand_named_cohort",
    "known_concept_ids",
    "register_cohort_concept_ids",
    "register_pattern",
    "register_patterns_from_file",
    "reset_pattern_registry",
    "validate_cohort_definition",
    "validate_plan_cohort_predicates_against_context",
    "validate_plan_typed_bindings_against_context",
    "validate_concept_predicate",
    "write_locked_cohort_definition",
]
