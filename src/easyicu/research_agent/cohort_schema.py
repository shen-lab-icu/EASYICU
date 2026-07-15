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
from functools import lru_cache
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from .lock_authority import (
    LockAuthorityError,
    assert_lock_matches_evidence_anchor,
    rehydrate_timestamp_only_legacy_lock,
)


# Framework-owned anchors stay deliberately small and generic. Disease- or
# intervention-specific anchors such as "sepsis_onset" or "vent_start" are
# case-owned strings declared by the benchmark/run protocol, not by this shared
# schema module.
UNIVERSAL_ANCHORS = frozenset({"icu_admit", "hospital_admit", "index_time"})
TimeAnchor = str
ALLOWED_CTAS_AGGREGATIONS = (
    "max",
    "min",
    "mean",
    "median",
    "last",
    "first",
    "any",
    "all",
    "count",
    "sum",
)
Aggregation = Literal[
    "max",
    "min",
    "mean",
    "median",
    "last",
    "first",
    "any",
    "all",
    "count",
    "sum",
]
PredicateOp = Literal[
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not_in",
    "missing",
    "not_missing",
]

COHORT_LOCK_FILENAME = "cohort_locked.json"
_CONCEPT_DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "concept-dict.json"
_ANY_ALL_ALLOWED_OPS = {"==", "!=", "missing", "not_missing"}
_IMPLEMENTED_AGGREGATIONS = set(ALLOWED_CTAS_AGGREGATIONS)


class CohortSchemaError(ValueError):
    """Raised when a cohort definition is ambiguous or invalid."""


class CohortDataError(KeyError):
    """Raised when materialised data cannot satisfy a CTAS definition."""


@dataclass(frozen=True)
class TimeWindow:
    anchor: TimeAnchor
    start_offset_hours: float
    end_offset_hours: float

    def __post_init__(self) -> None:
        if not self.anchor:
            raise CohortSchemaError("time_window.anchor is required")
        if self.end_offset_hours <= self.start_offset_hours:
            raise CohortSchemaError("time_window.end_offset_hours must be > start")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["start_offset_hours"] = _offset_to_json(self.start_offset_hours)
        data["end_offset_hours"] = _offset_to_json(self.end_offset_hours)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeWindow":
        if not isinstance(data, dict):
            raise CohortSchemaError("time_window must be an object")
        required = ("anchor", "start_offset_hours", "end_offset_hours")
        missing = [key for key in required if key not in data]
        if missing:
            raise CohortSchemaError(
                "time_window missing required field(s): " + ", ".join(missing)
            )
        return cls(
            anchor=str(data["anchor"]),
            start_offset_hours=_coerce_offset(data["start_offset_hours"]),
            end_offset_hours=_coerce_offset(data["end_offset_hours"]),
        )


@dataclass(frozen=True)
class ConceptPredicate:
    concept_id: str
    time_window: TimeWindow
    aggregation: Aggregation
    op: PredicateOp
    value: Any = None

    def __post_init__(self) -> None:
        validate_concept_predicate(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "time_window": self.time_window.to_dict(),
            "aggregation": self.aggregation,
            "op": self.op,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptPredicate":
        if not isinstance(data, dict):
            raise CohortSchemaError("concept predicate must be an object")
        required = ("concept_id", "time_window", "aggregation", "op")
        missing = [key for key in required if key not in data]
        if missing:
            raise CohortSchemaError(
                "concept predicate missing required field(s): " + ", ".join(missing)
            )
        return cls(
            concept_id=str(data["concept_id"]),
            time_window=TimeWindow.from_dict(data["time_window"]),
            aggregation=str(data["aggregation"]),  # type: ignore[arg-type]
            op=str(data["op"]),  # type: ignore[arg-type]
            value=data.get("value"),
        )


@dataclass(frozen=True)
class CohortDefinition:
    name: str
    inclusion: tuple[ConceptPredicate, ...] = ()
    exclusion: tuple[ConceptPredicate, ...] = ()
    derived_from_named: Optional[str] = None
    locked_at: str = "not_locked"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inclusion": [pred.to_dict() for pred in self.inclusion],
            "exclusion": [pred.to_dict() for pred in self.exclusion],
            "derived_from_named": self.derived_from_named,
            "locked_at": self.locked_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohortDefinition":
        if not isinstance(data, dict):
            raise CohortSchemaError("cohort definition must be an object")
        if "from_named" in data:
            return expand_named_cohort(str(data["from_named"]))
        return cls(
            name=str(data.get("name") or "primary"),
            inclusion=tuple(
                ConceptPredicate.from_dict(item)
                for item in data.get("inclusion", []) or []
            ),
            exclusion=tuple(
                ConceptPredicate.from_dict(item)
                for item in data.get("exclusion", []) or []
            ),
            derived_from_named=(
                str(data["derived_from_named"])
                if data.get("derived_from_named") is not None
                else None
            ),
            locked_at=str(data.get("locked_at") or "not_locked"),
        )


class PatternRegistry:
    """Runtime registry for caller-supplied named cohort patterns."""

    def __init__(self) -> None:
        self._patterns: Dict[str, CohortDefinition] = {}
        self._provenance: Dict[str, str] = {}

    def register(
        self,
        name: str,
        definition: CohortDefinition,
        *,
        provenance: str,
    ) -> None:
        name = str(name).strip()
        if not name:
            raise CohortSchemaError("named cohort pattern requires a name")
        if not provenance:
            raise CohortSchemaError(f"named cohort pattern {name!r} lacks provenance")
        validate_cohort_definition(definition)
        self._patterns[name] = definition
        self._provenance[name] = provenance

    def register_from_file(self, path: Path | str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_entries = payload.get("patterns", payload.get("cohorts", payload))
        if not isinstance(raw_entries, dict):
            raise CohortSchemaError("pattern file must contain an object of patterns")
        for name, raw in raw_entries.items():
            if not isinstance(raw, dict):
                raise CohortSchemaError(f"named pattern {name!r} must be an object")
            provenance = str(raw.get("provenance") or "").strip()
            definition_payload = raw.get("definition")
            if definition_payload is None:
                definition_payload = {
                    key: value for key, value in raw.items() if key != "provenance"
                }
            definition = CohortDefinition.from_dict(definition_payload)
            self.register(str(name), definition, provenance=provenance)

    def expand(self, name: str) -> CohortDefinition:
        name = str(name).strip()
        definition = self._patterns.get(name)
        if definition is None:
            raise CohortSchemaError(
                f"unknown named cohort pattern: {name}; register it before planning "
                "or supply the full time-window/aggregation predicate tuple"
            )
        return replace(definition, derived_from_named=name)

    def provenance_for(self, name: str) -> Optional[str]:
        return self._provenance.get(name)

    def clear(self) -> None:
        self._patterns.clear()
        self._provenance.clear()


_DEFAULT_PATTERN_REGISTRY = PatternRegistry()


def default_pattern_registry() -> PatternRegistry:
    return _DEFAULT_PATTERN_REGISTRY


def register_pattern(
    name: str,
    definition: CohortDefinition,
    *,
    provenance: str,
    registry: Optional[PatternRegistry] = None,
) -> None:
    (registry or _DEFAULT_PATTERN_REGISTRY).register(
        name,
        definition,
        provenance=provenance,
    )


def register_patterns_from_file(
    path: Path | str,
    *,
    registry: Optional[PatternRegistry] = None,
) -> None:
    (registry or _DEFAULT_PATTERN_REGISTRY).register_from_file(path)


def reset_pattern_registry() -> None:
    """Clear the process-local default registry.

    Intended for tests and runner setup boundaries. Production code should
    prefer explicit registration before invoking the planner.
    """

    _DEFAULT_PATTERN_REGISTRY.clear()


def coerce_cohort_definition(value: Any) -> Optional[CohortDefinition]:
    if value is None:
        return None
    if isinstance(value, CohortDefinition):
        validate_cohort_definition(value)
        return value
    if isinstance(value, str):
        raise CohortSchemaError(
            "cohort must be a CohortDefinition object or {from_named: ...}; "
            "free-text cohort strings are not allowed"
        )
    if isinstance(value, dict):
        definition = CohortDefinition.from_dict(value)
        validate_cohort_definition(definition)
        return definition
    raise CohortSchemaError(f"unsupported cohort definition type: {type(value)!r}")


def ensure_cohort_definition(plan: Any) -> Any:
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if definition is None:
        definition = CohortDefinition(name="primary")
    return plan.model_copy(update={"cohort": definition})


def validate_concept_predicate(predicate: ConceptPredicate) -> None:
    if not predicate.concept_id:
        raise CohortSchemaError("concept_id is required")
    if not concept_id_exists(predicate.concept_id):
        raise CohortSchemaError(f"unknown concept_id: {predicate.concept_id}")
    if predicate.time_window is None:
        raise CohortSchemaError("time_window is required")
    if not predicate.aggregation:
        raise CohortSchemaError("aggregation is required")
    if (
        predicate.aggregation in {"any", "all"}
        and predicate.op not in _ANY_ALL_ALLOWED_OPS
    ):
        raise CohortSchemaError(
            f"aggregation={predicate.aggregation!r} only supports "
            f"operators {sorted(_ANY_ALL_ALLOWED_OPS)}"
        )


def validate_cohort_definition(definition: CohortDefinition) -> None:
    if not definition.name:
        raise CohortSchemaError("cohort.name is required")
    for pred in [*definition.inclusion, *definition.exclusion]:
        validate_concept_predicate(pred)


def expand_named_cohort(
    name: str, registry: Optional[PatternRegistry] = None
) -> CohortDefinition:
    definition = (registry or _DEFAULT_PATTERN_REGISTRY).expand(name)
    validate_cohort_definition(definition)
    return definition


def cohort_definition_sha(definition: CohortDefinition) -> str:
    # Round-trip through the parser so equivalent integer/float time-window
    # literals (``24`` versus ``24.0``) have one durable scientific digest.
    # Without this, a freshly written lock could fail its own next resume after
    # JSON parsing normalised the offsets to floats.
    canonical = CohortDefinition.from_dict(definition.to_dict()).to_dict()
    raw = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
        if separator and kind in {"artifact", "dataset", "table"} and name == (
            "analysis_cohort"
        ):
            return True
    return False


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
    outcome, and its descriptor binds it to the same ``source_concept``.  This
    prevents a sibling output from the same composite loader from masquerading
    as the selected analysis variable.  Ambiguity fails closed; no dtype,
    token, or frame-order fallback is allowed.
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

    descriptors_by_source: Dict[str, set[str]] = {}
    for descriptor in getattr(context, "variables", ()) or ():
        name = str(getattr(descriptor, "name", "") or "").strip()
        source_concept = str(
            getattr(descriptor, "source_concept", "") or ""
        ).strip()
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
        descriptors_by_source.setdefault(source_concept, set()).add(name)

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
        candidates = sorted(descriptors_by_source.get(concept_id, ()))
        if len(candidates) > 1:
            raise CohortDataError(
                "cohort predicate column binding is ambiguous for concept "
                f"{concept_id!r}; Planner-declared ResearchContext candidates: "
                + ", ".join(repr(candidate) for candidate in candidates)
            )
        if len(candidates) == 1:
            bindings[concept_id] = candidates[0]
    return bindings


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
        "n_universe": None,
        "n_cohort": None,
        "error": None,
    }
    definition = coerce_cohort_definition(getattr(plan, "cohort", None))
    if definition is None or not (definition.inclusion or definition.exclusion):
        return result
    try:
        import pandas as pd  # type: ignore

        universe = pd.read_parquet(universe_path)
        column_bindings = _planner_declared_context_column_bindings(
            definition=definition,
            plan=plan,
            context=context,
            columns=universe.columns,
        )
        cohort = build_cohort(
            definition,
            universe,
            column_bindings=column_bindings,
        ).reset_index(drop=True)
    except Exception as exc:  # fall back to the universe; never break the run
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return result

    cohort = coerce_isfinite_safe_dtypes(cohort)
    out_path = Path(run_dir) / f"{stem}.parquet"
    cohort.to_parquet(out_path, index=False)
    provenance = {
        "schema_version": "easyicu.analysis_cohort/1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "universe_parquet": str(universe_path),
        "cohort_definition": definition.to_dict(),
        "cohort_sha256": cohort_definition_sha(definition),
        "n_universe": int(len(universe)),
        "n_analysis_cohort": int(len(cohort)),
        "predicate_column_bindings": [
            {
                "concept_id": concept_id,
                "column": column,
                "basis": "planner_declared_operational_output_source_concept",
            }
            for concept_id, column in sorted(column_bindings.items())
        ],
    }
    (Path(run_dir) / f"{stem}_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result.update(
        status="applied",
        path=out_path,
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


# Columns of an externally provided, already-materialised cohort (e.g. the
# EHRFlowBench path or cohort_materializer output). They are not dictionary
# concepts, but the data is already present, so a planner may legitimately
# reference them in a CTAS predicate — `_predicate_mask` reads them straight
# from `data.columns`. Registered per run so the static planner validation does
# not reject pre-materialised covariates as "unknown concept_id".
_EXTRA_COHORT_CONCEPT_IDS: set[str] = set()


def register_cohort_concept_ids(concept_ids: Any) -> None:
    """Allow these ids in CTAS predicate validation (pre-materialised columns)."""
    _EXTRA_COHORT_CONCEPT_IDS.update(str(c) for c in concept_ids)


def clear_cohort_concept_ids() -> None:
    _EXTRA_COHORT_CONCEPT_IDS.clear()


def concept_id_exists(concept_id: str) -> bool:
    return concept_id in known_concept_ids() or concept_id in _EXTRA_COHORT_CONCEPT_IDS


@lru_cache(maxsize=1)
def known_concept_ids() -> set[str]:
    try:
        payload = json.loads(_CONCEPT_DICT_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return set(payload.keys())


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


def _coerce_offset(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "+inf", "infinity"}:
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CohortSchemaError(f"invalid time offset: {value!r}") from exc


def _offset_to_json(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


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
