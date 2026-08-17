"""Pure planning contract for time-anchored cohort definitions.

This module owns the serializable CTAS types, validation rules, named-pattern
registry, and concept-id registry used while planning a cohort. It deliberately
does not read or write run locks, materialize dataframes, or promote evidence;
those lifecycle responsibilities remain in :mod:`..cohort.schema`.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional

# Framework-owned anchors stay deliberately small and generic. Disease- or
# intervention-specific anchors such as "sepsis_onset" or "vent_start" are
# case-owned strings declared by the benchmark/run protocol, not by this shared
# contract module.
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
CohortSelectionMode = Literal["predicate_filtered", "all_input_rows"]
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

_CONCEPT_DICT_PATH = Path(__file__).resolve().parents[2] / "data" / "concept-dict.json"
_SOFA2_DICT_PATH = Path(__file__).resolve().parents[2] / "data" / "sofa2-dict.json"
_CONCEPT_DICT_PATHS = (_CONCEPT_DICT_PATH, _SOFA2_DICT_PATH)
_ANY_ALL_ALLOWED_OPS = {"==", "!=", "missing", "not_missing"}


class CohortSchemaError(ValueError):
    """Raised when a cohort definition is ambiguous or invalid."""


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
    selection_mode: CohortSelectionMode = "predicate_filtered"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "inclusion": [pred.to_dict() for pred in self.inclusion],
            "exclusion": [pred.to_dict() for pred in self.exclusion],
            "derived_from_named": self.derived_from_named,
            "locked_at": self.locked_at,
        }
        # Preserve every legacy predicate-filtered authority digest. The new
        # coordinate is serialized only for an explicit all-row decision.
        if self.selection_mode != "predicate_filtered":
            payload["selection_mode"] = self.selection_mode
        return payload

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
            selection_mode=str(
                data.get("selection_mode") or "predicate_filtered"
            ),  # type: ignore[arg-type]
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
    """Clear the process-local default registry at a setup/test boundary."""

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


def _context_cohort_concept_ids(context: Any) -> tuple[str, ...]:
    values: list[str] = []
    for variable in getattr(context, "variables", ()) or ():
        values.extend(
            str(value).strip()
            for value in (
                getattr(variable, "name", None),
                getattr(variable, "source_concept", None),
                *(getattr(variable, "derived_from_concepts", ()) or ()),
            )
            if str(value or "").strip()
        )
    return tuple(dict.fromkeys(values))


def ensure_cohort_definition(plan: Any, *, context: Any | None = None) -> Any:
    """Normalize a plan cohort against its sealed run concept authority.

    A typed plan can legitimately name a physical column materialized only for
    this run. Re-validating it against the packaged dictionary alone would
    reject the exact same cohort that Planner compilation already accepted.
    The optional context provides a temporary, non-leaking registry scope; a
    caller without run authority retains the original dictionary-only gate.
    """

    def normalized() -> CohortDefinition:
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        return definition or CohortDefinition(name="primary")

    if context is None:
        definition = normalized()
    else:
        with cohort_concept_id_scope(_context_cohort_concept_ids(context)):
            definition = normalized()
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
    if definition.selection_mode not in {"predicate_filtered", "all_input_rows"}:
        raise CohortSchemaError("cohort.selection_mode is invalid")
    if definition.selection_mode == "all_input_rows" and (
        definition.inclusion or definition.exclusion
    ):
        raise CohortSchemaError(
            "cohort.selection_mode='all_input_rows' requires empty inclusion "
            "and exclusion predicates"
        )
    for pred in [*definition.inclusion, *definition.exclusion]:
        validate_concept_predicate(pred)


def cohort_definition_has_explicit_selection(
    definition: CohortDefinition | None,
) -> bool:
    """Whether a plan selected all rows or supplied filtering predicates."""

    return bool(
        definition is not None
        and (
            definition.selection_mode == "all_input_rows"
            or definition.inclusion
            or definition.exclusion
        )
    )


def expand_named_cohort(
    name: str, registry: Optional[PatternRegistry] = None
) -> CohortDefinition:
    definition = (registry or _DEFAULT_PATTERN_REGISTRY).expand(name)
    validate_cohort_definition(definition)
    return definition


def cohort_definition_sha(definition: CohortDefinition) -> str:
    # Round-trip through the parser so equivalent integer/float time-window
    # literals (``24`` versus ``24.0``) have one durable scientific digest.
    canonical = CohortDefinition.from_dict(definition.to_dict()).to_dict()
    raw = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Pre-materialised cohort columns are registered per run so planning validation
# can accept them without pretending they are packaged dictionary concepts.
#
# KNOWN DEBT -- the lock below does NOT make this safe for concurrent runs.
# ``register_cohort_concept_ids`` is a permanent process-wide registration, so
# two runs in one process accumulate each other's cohort columns and each will
# validate a plan naming a column only the *other* run materialised.  The lock
# fixes the scoped-replay race (interleaved snapshot/restore); it cannot fix
# cross-run mixing, because there is only one set to mix into.  The real fix is
# an explicit immutable registry threaded through planning validation rather
# than this module-level global.
_EXTRA_COHORT_CONCEPT_IDS: set[str] = set()
#: Guards every mutation of the set above. Re-entrant so a nested scope in
#: the same thread is not a deadlock.
_EXTRA_COHORT_CONCEPT_IDS_LOCK = threading.RLock()


def register_cohort_concept_ids(concept_ids: Any) -> None:
    """Allow ids backed by pre-materialised columns in CTAS validation."""

    with _EXTRA_COHORT_CONCEPT_IDS_LOCK:
        _EXTRA_COHORT_CONCEPT_IDS.update(str(c) for c in concept_ids)


def clear_cohort_concept_ids() -> None:
    with _EXTRA_COHORT_CONCEPT_IDS_LOCK:
        _EXTRA_COHORT_CONCEPT_IDS.clear()


@contextmanager
def cohort_concept_id_scope(concept_ids: Any) -> Iterator[None]:
    """Register ``concept_ids`` for the duration of the block, then restore.

    ``register_cohort_concept_ids`` mutates process-global state and
    ``clear_cohort_concept_ids`` empties it wholesale, so a caller that only
    wants to ask a question ("would this plan validate if these columns
    existed?") has no way to ask it without either leaking its answer into
    every later validation or destroying a registration someone else owns.
    This restores the exact prior set, including ids that were already present
    before the block.

    **The scope is mutually exclusive, and it has to be.** Two threads whose
    snapshot-and-restore interleave restore each other's snapshots: thread A
    snapshots, B snapshots (now including A's ids), A restores (dropping its
    own), B restores (re-adding them permanently). The set is process-global,
    so the only sound answer is that one scoped question runs at a time; the
    lock is re-entrant so nesting in one thread still works. Callers that need
    genuine parallelism want an explicit immutable registry passed down rather
    than this shared set -- this makes the shared-state cost visible instead of
    silently corrupting.
    """

    with _EXTRA_COHORT_CONCEPT_IDS_LOCK:
        previous = set(_EXTRA_COHORT_CONCEPT_IDS)
        _EXTRA_COHORT_CONCEPT_IDS.update(str(c) for c in concept_ids)
        try:
            yield
        finally:
            _EXTRA_COHORT_CONCEPT_IDS.clear()
            _EXTRA_COHORT_CONCEPT_IDS.update(previous)


def concept_id_exists(concept_id: str) -> bool:
    """Answer from a consistent view of the registry.

    Guarding only the writers left the readers outside: a thread that never
    enters a scope could still observe another thread's temporary ids, so a
    hypothetical asked in one place became a real answer in another.  The read
    takes the same lock, which also means a bare reader blocks for the duration
    of a scope rather than seeing half of it.
    """

    if concept_id in known_concept_ids():
        return True
    with _EXTRA_COHORT_CONCEPT_IDS_LOCK:
        return concept_id in _EXTRA_COHORT_CONCEPT_IDS


@lru_cache(maxsize=1)
def known_concept_ids() -> set[str]:
    concept_ids: set[str] = set()
    for path in _CONCEPT_DICT_PATHS:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            concept_ids.update(str(key) for key in payload)
    return concept_ids


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
    "Aggregation",
    "CohortDefinition",
    "CohortSchemaError",
    "ConceptPredicate",
    "PatternRegistry",
    "PredicateOp",
    "TimeAnchor",
    "TimeWindow",
    "UNIVERSAL_ANCHORS",
    "clear_cohort_concept_ids",
    "coerce_cohort_definition",
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
]
