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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .lock_authority import (
    LockAuthorityError,
    assert_lock_matches_evidence_anchor,
    rehydrate_timestamp_only_legacy_lock,
)
from .planning.cohort_contract import (
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
    coerce_cohort_definition,
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
    # Compatibility for locks written before cohort hashes canonicalised
    # integer/float time-window offsets.  This does not weaken modern evidence
    # authority: the complete lock bytes must still match the immutable anchor.
    legacy_payload_sha = hashlib.sha256(
        json.dumps(
            raw_cohort,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not expected_sha or expected_sha not in {observed_sha, legacy_payload_sha}:
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
        try:
            repair = rehydrate_timestamp_only_legacy_lock(
                run_dir=run_dir,
                lock_path=path,
                evidence_id="cohort_locked",
                label="cohort definition lock",
            )
        except LockAuthorityError as exc:
            raise CohortSchemaError(str(exc)) from exc
        if (
            repair is not None
            and evidence.get("cohort_lock_resume_rehydration") is None
        ):
            evidence.register_json(
                kind="log",
                description=(
                    "Resume compatibility repair: restored the cohort lock from "
                    "its verified plan-time evidence anchor after a legacy "
                    "timestamp-only rewrite."
                ),
                payload=repair,
                filename="cohort_lock_resume_rehydration.json",
                evidence_id="cohort_lock_resume_rehydration",
                producer="planner",
                generation_mode="system",
                prompt_pack_version=prompt_pack_version,
                metadata={"llm_signature": llm_signature},
            )
        locked_definition = _load_locked_cohort_definition(run_dir)
        definition_sha = cohort_definition_sha(definition)
        locked_sha = cohort_definition_sha(locked_definition)
        if definition_sha != locked_sha:
            locked_is_empty = not (
                locked_definition.inclusion or locked_definition.exclusion
            )
            definition_is_real = bool(definition.inclusion or definition.exclusion)
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
            from .evidence import _atomic_write_bytes

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


def _declares_analysis_cohort(step: Any) -> bool:
    for raw in getattr(step, "expected_outputs", ()) or ():
        kind, separator, name = str(raw or "").strip().casefold().partition(":")
        if (
            separator
            and kind in {"artifact", "dataset", "table"}
            and name == ("analysis_cohort")
        ):
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

    normalized = re.sub(r"[^a-z0-9.]+", "_", str(value or "").casefold()).strip("_")
    if not normalized:
        return False

    def _number(raw: float) -> str:
        number = float(raw)
        return str(int(number)) if number.is_integer() else f"{number:g}"

    anchor = re.sub(r"[^a-z0-9]+", "_", str(window.anchor).casefold()).strip("_")
    start = _number(window.start_offset_hours)
    end = _number(window.end_offset_hours)
    return normalized == f"{anchor}_{start}_{end}h"


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
    an input, the ResearchContext names it as the operational exposure or
    outcome, and its descriptor binds it to the same ``source_concept``, exact
    aggregation, and exact time window.  This prevents a sibling output from
    the same composite loader from masquerading as the selected analysis
    variable or silently changing Planner-owned temporal/aggregation semantics.
    Ambiguity fails closed; no dtype, token, or frame-order fallback is allowed.
    """

    if context is None:
        return {}
    producers = [
        step
        for step in getattr(plan, "steps", ()) or ()
        if _declares_analysis_cohort(step)
    ]
    if len(producers) != 1:
        return {}
    available = {str(column) for column in columns}
    operational_outputs = {
        str(getattr(context, field, "") or "").strip()
        for field in ("primary_exposure", "target_outcome")
        if str(getattr(context, field, "") or "").strip() in available
    }
    if not operational_outputs:
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
            or name not in operational_outputs
            or role_value in {"id", "meta", "time"}
        ):
            continue
        descriptors_by_source.setdefault(source_concept, []).append(descriptor)

    bindings: Dict[str, str] = {}
    predicate_concepts = {
        predicate.concept_id
        for predicate in (*definition.inclusion, *definition.exclusion)
        if _resolve_predicate_column(
            columns,
            predicate.concept_id,
            predicate.aggregation,
        )
        is None
    }
    for concept_id in sorted(predicate_concepts):
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
                _column_aggregation_matches(
                    str(getattr(descriptor, "name", "") or ""),
                    predicate.aggregation,
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
        if len(candidates) == 1:
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
            "basis": "planner_declared_operational_output_source_concept",
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
    if definition is None or not (definition.inclusion or definition.exclusion):
        raise CohortSchemaError("analysis cohort authority requires locked predicates")
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


def coerce_isfinite_safe_dtypes(frame: Any) -> Any:
    """Downcast pandas nullable-extension and boolean-object columns to numpy
    ``float64`` so downstream ``np.isfinite`` / ``to_numpy()`` in generated
    analysis code never receives an object or extension array.

    The universe builder emits per-concept aggregates as pandas *nullable*
    extension dtypes (``Int64`` / ``Float64`` / ``boolean``), or as object
    columns holding python bools, whenever the aggregate is mostly null.
    Generated causal / prediction code does ``design_df[col].to_numpy()`` and
    feeds the result to ``np.isfinite``; on a nullable or object array numpy
    raises ``ufunc 'isfinite' not supported for the input types`` and the whole
    primary estimate is silently lost (H2 vasopressor causal: the readmission
    aggregates came through as ``boolean`` / ``Float64`` / ``Int64`` and crashed
    the propensity balance table -> ``adjusted_effect=None``). Coercing these to
    ``float64`` (NA -> NaN) at cohort-materialisation time leaves every column as
    either a numpy numeric or a genuine string categorical -- the two shapes the
    generated code already handles. True string/categorical object columns (e.g.
    ``sex``, admission type) are left untouched for dummy-encoding.
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
        "authority_path": None,
        "authority_ref": None,
        "cohort_definition_sha256": None,
        "n_universe": None,
        "n_cohort": None,
        "error": None,
    }
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if definition is None or not (definition.inclusion or definition.exclusion):
        return result
    from .intake.materialized_metadata import (
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
        cohort = build_cohort(
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
                    Path(__file__).resolve().parent
                    / "planning"
                    / "cohort_contract.py",
                    Path(__file__).resolve().parent
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
        (Path(run_dir) / f"{stem}_provenance.json").write_text(
            json.dumps(semantic_provenance, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    result.update(
        status="applied",
        path=out_path,
        authority_path=authority_path,
        authority_ref=authority_ref,
        cohort_definition_sha256=cohort_definition_sha(definition),
        n_universe=int(len(universe)),
        n_cohort=int(len(cohort)),
    )
    return result


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
    mask = pd.Series(True, index=data.index)
    for pred in definition.inclusion:
        mask &= _predicate_mask(data, pred, column_bindings=column_bindings)
    for pred in definition.exclusion:
        mask &= ~_predicate_mask(data, pred, column_bindings=column_bindings)
    return data.loc[mask].copy()


# A few EasyICU concepts materialise their value under an output-column name
# that differs from the dictionary ``concept_id`` because the concept's callback
# emits a clinically-named column (e.g. the ``kdigo_aki`` concept emits
# ``aki_stage``; see ``kdigo_aki.py`` and ``api.py``'s SPECIAL_CONCEPTS dispatch
# ``_KDIGO_OUTPUTS``/``_CIRC_OUTPUTS``). A planner that references the cohort
# concept by its *dictionary id* (the canonical, cross-database way per the
# concept layer) then names a predicate whose ``concept_id`` never appears as a
# universe column, even though the data is present under the output name. This
# is a general EasyICU concept-layer fact, not a benchmark-specific alias: the
# mapping holds for every database and every analysis that uses these concepts.
_CONCEPT_OUTPUT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "kdigo_aki": ("aki_stage",),
    "kdigo_creat": ("aki_stage_creat",),
    "kdigo_uo": ("aki_stage_uo",),
    "circ_failure": ("circ_failure", "circ_event"),
}


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
    trying in order: the bare id, the wide ``<concept_id>_<aggregation>`` form,
    and the concept's known output-column alias(es) (bare and aggregated). Return
    ``None`` when no column honours the requested aggregation, so the caller can
    fail loudly rather than silently skip an unenforceable predicate.
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
    for stem in _CONCEPT_OUTPUT_COLUMN_ALIASES.get(concept_id, ()):
        if stem in cols:
            return stem
        stem_aggregated = f"{stem}_{aggregation}"
        if stem_aggregated in cols:
            return stem_aggregated
    return None


def _predicate_mask(
    data: Any,
    pred: ConceptPredicate,
    *,
    column_bindings: Optional[Dict[str, str]] = None,
) -> Any:
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
            f"(also tried {pred.concept_id}_{pred.aggregation} and known output "
            "aliases)"
        )
    series = data[column]
    mask = _apply_op(series, pred.op, pred.value)
    return _refine_occurrence_mask_by_event_time(data, pred, mask)


def _refine_occurrence_mask_by_event_time(
    data: Any, pred: ConceptPredicate, mask: Any
) -> Any:
    """Intersect an event-occurrence predicate with its event-time window.

    ``build_cohort`` filters an already-materialised wide table and, by design,
    does not re-window the summary columns. That is correct for a concept whose
    column was summarised WITHIN the predicate window, but an OUTCOME concept is
    materialised whole-stay (``death`` is 1 whenever the patient ever died)
    alongside an event-time column (``death_time`` = hours from the anchor). A
    bounded-window occurrence predicate on such a concept — e.g. the landmark
    exclusion "died within the first 24h" that a survival design writes to avoid
    immortal-time bias — must therefore consult the event time. Otherwise the
    whole-stay flag drops EVERY event, not just the in-window ones (H1 survival
    regression: all 9,466 deaths excluded -> 0 events -> "survival infeasible").

    Scope is deliberately narrow: only a truthy ``==`` occurrence check over a
    finite window on a concept that actually carries a ``<concept>_time`` sibling
    column is refined. Magnitude filters (age>=18, los>=1) and concepts without
    an event-time column are untouched, so association runs with no event-time
    columns (e.g. E3) behave exactly as before.
    """
    tw = pred.time_window
    if tw is None:
        return mask
    if pred.op != "==" or pred.value in (0, 0.0, False, None):
        return mask
    end = tw.end_offset_hours
    if end is None or not math.isfinite(float(end)):
        return mask
    event_time_col = f"{pred.concept_id}_time"
    if event_time_col not in data.columns:
        return mask
    event_time = data[event_time_col]
    in_window = (event_time >= float(tw.start_offset_hours)) & (
        event_time <= float(end)
    )
    # NaN event time (no event) -> not in window; keep the row's occurrence flag
    # from deciding membership only when the event genuinely falls in the window.
    try:
        in_window = in_window.fillna(False)
    except Exception:
        pass
    return mask & in_window


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
    "validate_concept_predicate",
    "write_locked_cohort_definition",
]
