"""Cross-database EasyICU standardized extraction availability.

This module is intentionally built around EasyICU's standardized extraction
surface, not SQL-level access. It answers:
"Can this EasyICU concept be derived for this database, and if not, which
concept dependency blocks or degrades it?"

The implementation reads EasyICU's packaged concept dictionary and data-source
registry. Recursive and callback-derived concepts are resolved through their
declared dependencies so external agents can reason over the cross-database
standardization layer before calling ``easyicu.load_concepts``, without seeing
database-specific item ids or tables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

PUBLIC_DATABASES = ("mimic", "miiv", "eicu", "aumc", "hirid", "sic")

_OUTCOME_BLIND_FORBIDDEN_FIELDS = (
    "outcome_rate",
    "stratified_outcome",
    "effect_estimate",
    "p_value",
    "odds_ratio",
    "hazard_ratio",
    "risk_difference",
)

_DATABASE_ALIASES = {
    "miii": "mimic",
    "mimiciii": "mimic",
    "mimic-iii": "mimic",
    "mimic3": "mimic",
    "mimiciv": "miiv",
    "mimic-iv": "miiv",
    "miv": "miiv",
    "sicdb": "sic",
}

_CONCEPT_ALIASES = {
    "creatinine": "crea",
    "creat": "crea",
    "lactate": "lact",
    "urine_output": "urine",
    "uo": "urine",
    "aki": "kdigo_aki",
    "kdigo": "kdigo_aki",
    "kdigo_stage": "kdigo_aki",
    "aki_stage": "kdigo_aki",
    "sofa-2": "sofa2",
    "mortality": "death",
    "hospital_mortality": "death",
    "icu_mortality": "death",
}

# Sparse, full-cohort event indicators where an exported wide-table NaN can
# mean "event absent" rather than "not observed". This is intentionally narrower
# than generic binary-determinable concepts: screened/assessed concepts must not
# default missing values to false.
_DEFAULT_EVENT_DEFAULT_FALSE_CONCEPTS = frozenset(
    {
        "aki",
        "kdigo_aki",
        "rrt",
        "mech_vent",
        "death",
        "vaso_ind",
        "circ_event",
        "circ_failure",
        "sep3",
    }
)


class ConceptDatabaseAvailability(BaseModel):
    """Availability of one EasyICU concept on one database."""

    model_config = ConfigDict(extra="forbid")

    concept: str
    requested_concept: str
    database: str
    status: str = Field(description="full, degraded, or blocked")
    available: bool = False
    direct_source: bool = False
    reason: Optional[str] = None
    runtime_reason: Optional[str] = None
    source_missing_tables: List[str] = Field(default_factory=list)
    structural_unavailable: bool = False
    available_dependencies: List[str] = Field(default_factory=list)
    degraded_dependencies: List[str] = Field(default_factory=list)
    missing_dependencies: List[str] = Field(default_factory=list)


def concept_database_availability_from_load_record(
    record: Any,
    *,
    requested_concept: Optional[str] = None,
) -> ConceptDatabaseAvailability:
    """Map a runtime load availability record onto the RA availability model."""

    reason = str(getattr(record, "reason"))
    status = str(getattr(record, "status"))
    structural = reason in {"unmapped", "source_unavailable"}
    return ConceptDatabaseAvailability(
        concept=str(getattr(record, "concept")),
        requested_concept=str(requested_concept or getattr(record, "concept")),
        database=str(getattr(record, "database")),
        status=status,
        available=status != "blocked",
        direct_source=reason in {"mapped_present", "data_missing"},
        reason=reason,
        runtime_reason=reason,
        source_missing_tables=list(getattr(record, "missing_tables", ()) or ()),
        structural_unavailable=structural,
    )


class RealDataConceptFeasibility(BaseModel):
    """Outcome-blind real-data availability summary for one concept.

    This model is deliberately limited to denominator and missingness signals.
    It never carries outcome rates, stratum outcomes, p-values, or effect
    estimates; those belong to a registered analysis after a human gate.
    """

    model_config = ConfigDict(extra="forbid")

    concept: str
    database: str
    analytic_unit: Literal["stay", "patient"] = "stay"
    denominator_n: int = Field(ge=0)
    n_present: int = Field(ge=0)
    fraction_missing: float = Field(ge=0.0, le=1.0)
    n_joint_complete: int = Field(ge=0)
    joint_fraction_complete: float = Field(ge=0.0, le=1.0)
    missingness_applicable: bool = True
    structural_unavailable: bool = False
    availability_status: Optional[str] = None
    availability_reason: Optional[str] = None
    source_missing_tables: List[str] = Field(default_factory=list)
    structural_unavailable_concepts: List[str] = Field(default_factory=list)
    joint_denominator_concepts: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    time_window_requested: Optional[str] = None
    aggregation_requested: Optional[str] = None
    cohort_filter_summary: str = "none"
    missingness_severity: Literal["low", "medium", "high"]
    value_contrast_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    non_outcome_blind_fields_checked: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts_do_not_exceed_denominator(self) -> "RealDataConceptFeasibility":
        if self.n_present > self.denominator_n:
            raise ValueError("n_present cannot exceed denominator_n")
        if self.n_joint_complete > self.denominator_n:
            raise ValueError("n_joint_complete cannot exceed denominator_n")
        return self

    @model_validator(mode="after")
    def _record_outcome_blind_guard_fields(self) -> "RealDataConceptFeasibility":
        checked = list(
            dict.fromkeys(
                [
                    *self.non_outcome_blind_fields_checked,
                    *_OUTCOME_BLIND_FORBIDDEN_FIELDS,
                ]
            )
        )
        self.non_outcome_blind_fields_checked = checked
        return self


def normalize_database_name(database: str) -> str:
    key = (database or "").strip().lower().replace("_", "-")
    return _DATABASE_ALIASES.get(key, key)


def normalize_concept_name(concept: str) -> str:
    key = (concept or "").strip().lower().replace(" ", "_")
    return _CONCEPT_ALIASES.get(key, key)


def default_public_databases() -> List[str]:
    return list(PUBLIC_DATABASES)


def cross_database_concept_availability(
    *,
    concepts: Sequence[str],
    databases: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return concept-level availability for concepts x databases.

    The returned structure is JSON-serialisable and stable for MCP clients:

    ``{requested_concept: {database: ConceptDatabaseAvailability.dict()}}``.
    """

    dbs = _normalise_database_list(databases)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for requested in concepts:
        canonical = normalize_concept_name(requested)
        per_db: Dict[str, Dict[str, Any]] = {}
        for db in dbs:
            cell = explain_concept_availability(
                concept=canonical,
                database=db,
                requested_concept=requested,
            )
            per_db[db] = cell.model_dump(mode="json")
        out[str(requested)] = per_db
    return out


def explain_concept_availability(
    *,
    concept: str,
    database: str,
    requested_concept: Optional[str] = None,
) -> ConceptDatabaseAvailability:
    db = normalize_database_name(database)
    requested = requested_concept or concept
    canonical = normalize_concept_name(concept)
    return _explain_concept_availability_cached(canonical, db, str(requested))


def hypothesis_cross_database_feasibility(
    *,
    concepts: Sequence[str],
    databases: Sequence[str],
) -> Dict[str, Any]:
    """Summarise concept availability into hypothesis-level DB feasibility."""

    dbs = _normalise_database_list(databases)
    deps = _unique(normalize_concept_name(c) for c in concepts if c)
    availability = cross_database_concept_availability(
        concepts=deps,
        databases=dbs,
    )
    feasibility: Dict[str, str] = {}
    degraded_reason: Dict[str, str] = {}
    for db in dbs:
        cells = [
            availability[concept][db] for concept in deps if concept in availability
        ]
        if not cells:
            feasibility[db] = "blocked"
            degraded_reason[db] = "No concept dependencies were available to assess."
            continue
        statuses = [str(cell.get("status")) for cell in cells]
        if all(status == "full" for status in statuses):
            feasibility[db] = "full"
            continue
        if any(status == "blocked" for status in statuses):
            feasibility[db] = "blocked"
        else:
            feasibility[db] = "degraded"
        degraded_reason[db] = "; ".join(
            _reason_for_cell(cell)
            for cell in cells
            if str(cell.get("status")) != "full"
        )
    return {
        "concept_dependencies": deps,
        "cross_database_feasibility": feasibility,
        "degraded_reason": degraded_reason,
        "availability": availability,
    }


def real_data_concept_feasibility(
    concepts: Sequence[str],
    database: str,
    data_path: str | Path,
    *,
    cohort: Optional[Mapping[str, Any]] = None,
    time_window: Optional[str] = None,
    aggregation: Optional[str] = None,
    analytic_unit: Literal["stay", "patient"] = "stay",
    event_default_false_concepts: Optional[Iterable[str]] = None,
    contrast_concepts: Optional[Iterable[str]] = None,
) -> Dict[str, RealDataConceptFeasibility]:
    """Return outcome-blind data-layer feasibility for EasyICU concepts.

    The dictionary layer is checked first. Concepts that are blocked by the
    EasyICU concept dictionary short-circuit without reading ``data_path``.
    When at least one concept is dictionary-feasible, ``data_path`` is treated
    as a single exported wide cohort table, not as a raw database directory or
    a sharded ricu-style concept store. The table is inspected for denominator
    counts, per-concept non-missing counts, and joint completeness across the
    requested concept set. ``time_window`` and ``aggregation`` are recorded as
    requested metadata for downstream triage; S1 does not enforce temporal
    filtering or aggregation rules.

    ``event_default_false_concepts`` identifies sparse event/logical concepts
    whose present column covers the full cohort even when event-negative rows
    are encoded as ``NaN``. Measurement concepts keep the default ``NaN`` means
    missing semantics.

    ``contrast_concepts`` opts specific concepts into the exposure-side
    answerability signal ``value_contrast_fraction`` (1 - modal share over
    non-missing analytic units). This stays outcome-blind: callers must pass
    ONLY predictor/exposure concepts, never the outcome, because a binary
    outcome's modal share equals its event rate (a forbidden outcome field).
    """

    db = normalize_database_name(database)
    unit = _normalise_analytic_unit(analytic_unit)
    requested = [str(c) for c in concepts if str(c or "").strip()]
    event_default_false = _normalise_event_default_false_concepts(
        event_default_false_concepts
    )
    contrast_set = {
        normalize_concept_name(str(c))
        for c in (contrast_concepts or ())
        if str(c or "").strip()
    }
    cells = {
        concept: explain_concept_availability(
            concept=normalize_concept_name(concept),
            database=db,
            requested_concept=concept,
        )
        for concept in requested
    }
    if not cells:
        return {}

    structural_concepts = {
        concept
        for concept, cell in cells.items()
        if _cell_is_structural_unavailable(cell)
    }
    data_concepts = [
        concept for concept in requested if concept not in structural_concepts
    ]

    needs_data = bool(data_concepts)
    if not needs_data:
        return {
            concept: _blocked_real_data_feasibility(
                concept=concept,
                database=db,
                analytic_unit=unit,
                reason=_reason_for_cell(cells[concept].model_dump(mode="json")),
                time_window=time_window,
                aggregation=aggregation,
                availability_cell=cells[concept],
            )
            for concept in requested
        }

    frame = _read_prepared_frame(data_path)
    non_structural_blocked = [
        concept for concept in data_concepts if cells[concept].status == "blocked"
    ]
    if non_structural_blocked:
        joint = _zero_joint_from_frame(
            frame,
            analytic_unit=unit,
            cohort=cohort,
        )
    else:
        joint = _probe_joint_from_frame(
            frame,
            concepts=data_concepts,
            analytic_unit=unit,
            cohort=cohort,
            event_default_false_concepts=event_default_false,
        )

    out: Dict[str, RealDataConceptFeasibility] = {}
    structural_note = _structural_unavailable_note(structural_concepts)
    joint_denominator_concepts = [
        normalize_concept_name(concept) for concept in data_concepts
    ]
    for concept in requested:
        cell = cells[concept]
        if _cell_is_structural_unavailable(cell):
            out[concept] = _blocked_real_data_feasibility(
                concept=concept,
                database=db,
                analytic_unit=unit,
                reason=_reason_for_cell(cell.model_dump(mode="json")),
                time_window=time_window,
                aggregation=aggregation,
                availability_cell=cell,
            )
            continue
        single = _probe_single_concept_from_frame(
            frame,
            concept=concept,
            analytic_unit=unit,
            cohort=cohort,
            event_default_false_concepts=event_default_false,
            compute_contrast=normalize_concept_name(concept) in contrast_set,
        )
        note = _join_notes(
            structural_note,
            _column_resolution_note(
                cell=cell,
                concept=concept,
                single=single,
            ),
        )
        out[concept] = RealDataConceptFeasibility(
            concept=normalize_concept_name(concept),
            database=db,
            analytic_unit=unit,
            denominator_n=single["denominator_n"],
            n_present=single["n_present"],
            fraction_missing=single["fraction_missing"],
            n_joint_complete=joint["n_joint_complete"],
            joint_fraction_complete=joint["joint_fraction_complete"],
            missingness_applicable=True,
            structural_unavailable=False,
            availability_status=cell.status,
            availability_reason=cell.reason,
            source_missing_tables=list(cell.source_missing_tables),
            structural_unavailable_concepts=sorted(
                normalize_concept_name(concept) for concept in structural_concepts
            ),
            joint_denominator_concepts=joint_denominator_concepts,
            note=note,
            time_window_requested=str(time_window) if time_window is not None else None,
            aggregation_requested=str(aggregation) if aggregation is not None else None,
            cohort_filter_summary=single["cohort_filter_summary"],
            missingness_severity=_missingness_severity(single["fraction_missing"]),
            value_contrast_fraction=single.get("value_contrast_fraction"),
        )
    return out


@lru_cache(maxsize=4096)
def _explain_concept_availability_cached(
    concept: str,
    database: str,
    requested_concept: str,
) -> ConceptDatabaseAvailability:
    from easyicu.resources import load_data_sources, load_dictionary

    db = normalize_database_name(database)
    dictionary = load_dictionary(include_sofa2=True)
    registry = load_data_sources()
    canonical = normalize_concept_name(concept)
    definition = dictionary.get(canonical)
    if definition is None:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="blocked",
            available=False,
            reason="concept_not_found",
        )

    try:
        config = registry.get(db)
    except KeyError:
        config = None

    direct_source = False
    if config is not None:
        try:
            direct_source = bool(definition.for_data_source(config))
        except Exception:
            direct_source = False
    else:
        direct_source = db in getattr(definition, "sources", {})

    if direct_source:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="full",
            available=True,
            direct_source=True,
            reason="direct_source_available",
        )

    dependencies = _concept_dependencies(definition)
    if not dependencies:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="blocked",
            available=False,
            reason="no_direct_source_or_dependencies",
        )

    dep_cells = [
        _explain_dependency(dep, db)
        for dep in dependencies
        if normalize_concept_name(dep) != canonical
    ]
    if not dep_cells:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="blocked",
            available=False,
            reason="recursive_dependency_cycle_or_empty_dependency_set",
        )

    available_dependencies = [
        cell.concept for cell in dep_cells if cell.status == "full"
    ]
    degraded_dependencies = [
        cell.concept for cell in dep_cells if cell.status == "degraded"
    ]
    missing_dependencies = [
        cell.concept for cell in dep_cells if cell.status == "blocked"
    ]
    if not missing_dependencies and not degraded_dependencies:
        status = "full"
        reason = "all_dependencies_available"
    elif available_dependencies or degraded_dependencies:
        status = "degraded"
        reason = "partial_dependency_availability"
    else:
        status = "blocked"
        reason = "all_dependencies_blocked"

    return ConceptDatabaseAvailability(
        concept=canonical,
        requested_concept=requested_concept,
        database=db,
        status=status,
        available=status != "blocked",
        direct_source=False,
        reason=reason,
        available_dependencies=available_dependencies,
        degraded_dependencies=degraded_dependencies,
        missing_dependencies=missing_dependencies,
    )


def _explain_dependency(dep: str, database: str) -> ConceptDatabaseAvailability:
    canonical = normalize_concept_name(dep)
    return _explain_concept_availability_cached(canonical, database, dep)


def _concept_dependencies(definition: Any) -> List[str]:
    deps: List[str] = []
    for attr in ("sub_concepts", "depends_on"):
        values = getattr(definition, attr, None) or []
        if isinstance(values, str):
            deps.append(values)
        elif isinstance(values, Iterable):
            deps.extend(str(v) for v in values)
    return _unique(deps)


def _normalise_database_list(databases: Optional[Sequence[str]]) -> List[str]:
    if not databases:
        return default_public_databases()
    return _unique(normalize_database_name(db) for db in databases if db)


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _reason_for_cell(cell: Mapping[str, Any]) -> str:
    concept = str(cell.get("concept") or cell.get("requested_concept") or "concept")
    status = str(cell.get("status") or "unknown")
    missing = cell.get("missing_dependencies") or []
    degraded = cell.get("degraded_dependencies") or []
    reason = str(cell.get("reason") or "")
    parts = [f"{concept}={status}"]
    if missing:
        parts.append("missing=" + ",".join(map(str, missing[:6])))
    if degraded:
        parts.append("degraded=" + ",".join(map(str, degraded[:6])))
    if reason and reason not in {
        "direct_source_available",
        "all_dependencies_available",
    }:
        parts.append(reason)
    return " ".join(parts)


def _cell_is_structural_unavailable(cell: ConceptDatabaseAvailability) -> bool:
    if cell.structural_unavailable:
        return True
    if cell.status != "blocked":
        return False
    return (cell.reason or "") in {
        "concept_not_found",
        "no_direct_source_or_dependencies",
        "recursive_dependency_cycle_or_empty_dependency_set",
        "all_dependencies_blocked",
        "source_unavailable",
        "unmapped",
    }


def _structural_unavailable_note(concepts: Iterable[str]) -> Optional[str]:
    unavailable = sorted(
        normalize_concept_name(concept)
        for concept in concepts
        if str(concept or "").strip()
    )
    if not unavailable:
        return None
    return (
        "structural unavailable concept(s) excluded from missingness denominator: "
        + ", ".join(unavailable)
    )


def _join_notes(*notes: Optional[str]) -> Optional[str]:
    clean = [note for note in notes if note]
    if not clean:
        return None
    return " ".join(clean)


def _column_resolution_note(
    *,
    cell: ConceptDatabaseAvailability,
    concept: str,
    single: Mapping[str, Any],
) -> Optional[str]:
    if cell.status != "full":
        return None
    if single.get("column_resolved"):
        return None
    if int(single.get("denominator_n") or 0) <= 0:
        return None
    if int(single.get("n_present") or 0) != 0:
        return None
    candidates = ", ".join(map(str, single.get("column_candidates") or []))
    suffix = f" checked aliases: {candidates}" if candidates else ""
    return (
        "available concept has no matching exported wide-table column: "
        f"{normalize_concept_name(concept)}.{suffix}"
    )


def _read_prepared_frame(data_path: str | Path) -> Any:
    path = Path(data_path)
    suffix = path.suffix.lower()
    import pandas as pd

    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _normalise_analytic_unit(
    analytic_unit: Literal["stay", "patient"],
) -> Literal["stay", "patient"]:
    unit = str(analytic_unit or "stay").strip().lower()
    if unit not in {"stay", "patient"}:
        raise ValueError("analytic_unit must be 'stay' or 'patient'")
    return unit  # type: ignore[return-value]


def _apply_cohort_filter(
    frame: Any,
    cohort: Optional[Mapping[str, Any]],
) -> tuple[Any, str]:
    if not cohort:
        return frame, "none (all rows)"
    mask = None
    parts: List[str] = []
    for column, allowed in cohort.items():
        if column not in frame.columns:
            current = frame.index == "__missing_cohort_column__"
            parts.append(f"{column}=<missing column>")
        elif isinstance(allowed, (list, tuple, set, frozenset)):
            current = frame[column].isin(list(allowed))
            preview = ",".join(map(str, list(allowed)[:5]))
            parts.append(f"{column} in [{preview}]")
        else:
            current = frame[column] == allowed
            parts.append(f"{column}={allowed}")
        mask = current if mask is None else (mask & current)
    if mask is None:
        return frame, "none (empty cohort filter)"
    return frame.loc[mask].copy(), "; ".join(parts)


def _probe_single_concept_from_frame(
    frame: Any,
    *,
    concept: str,
    analytic_unit: Literal["stay", "patient"],
    cohort: Optional[Mapping[str, Any]],
    event_default_false_concepts: Optional[Iterable[str]] = None,
    compute_contrast: bool = False,
) -> Dict[str, Any]:
    filtered, cohort_summary = _apply_cohort_filter(frame, cohort)
    unit = _normalise_analytic_unit(analytic_unit)
    denominator = _denominator_n(filtered, unit)
    column = _resolve_concept_column(filtered, concept)
    column_candidates = _concept_column_candidates(concept)
    event_default_false = _normalise_event_default_false_concepts(
        event_default_false_concepts
    )
    n_present = (
        _n_present(
            filtered,
            column,
            unit,
            event_default_false=_is_event_default_false(
                concept,
                event_default_false,
            ),
        )
        if column is not None
        else 0
    )
    fraction_missing = _fraction_missing(n_present=n_present, denominator_n=denominator)
    value_contrast_fraction = (
        _value_contrast_fraction(
            filtered,
            column,
            event_default_false=_is_event_default_false(concept, event_default_false),
        )
        if compute_contrast and column is not None
        else None
    )
    return {
        "concept": normalize_concept_name(concept),
        "denominator_n": denominator,
        "n_present": n_present,
        "fraction_missing": fraction_missing,
        "cohort_filter_summary": cohort_summary,
        "column_resolved": column is not None,
        "column_candidates": column_candidates,
        "value_contrast_fraction": value_contrast_fraction,
    }


def _value_contrast_fraction(
    frame: Any, column: Any, *, event_default_false: bool = False
) -> Optional[float]:
    """Exposure-side answerability: ``1 - modal share`` over the values.

    A near-zero value means the predictor is essentially constant in the cohort
    (e.g. an intervention present in ~0% or ~100% of units), so there is no
    exposure contrast to estimate an association from no matter how complete the
    data is. ``0.0`` is a degenerate (single-valued) exposure. Returns ``None``
    when nothing is observed (missingness already captures that).

    Computed over the prepared wide cohort table (≈one row per analytic unit).
    Callers gate this to predictor/exposure concepts only: a binary outcome's
    modal share equals its event rate, which the outcome-blind guard forbids.

    For ``event_default_false`` concepts (mech_vent, rrt, ...), event-absent
    units are encoded as NaN — real observed negatives, not missing data (see
    ``_present_mask``). ``dropna()`` would strip them and report a balanced 30/70
    binary exposure as single-valued (contrast 0.0 -> a false-infeasible verdict
    at the human gate). Treat NaN as the absent class over the full denominator
    instead; only measurement concepts drop missing rows.
    """
    if event_default_false:
        series = frame[column].fillna(0)
    else:
        series = frame[column].dropna()
    n = int(len(series))
    if n == 0:
        return None
    modal_count = int(series.value_counts().iloc[0])
    return max(0.0, min(1.0, 1.0 - modal_count / n))


def _probe_joint_from_frame(
    frame: Any,
    *,
    concepts: Sequence[str],
    analytic_unit: Literal["stay", "patient"],
    cohort: Optional[Mapping[str, Any]],
    event_default_false_concepts: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    filtered, _cohort_summary = _apply_cohort_filter(frame, cohort)
    unit = _normalise_analytic_unit(analytic_unit)
    denominator = _denominator_n(filtered, unit)
    unique_concepts = _unique_concepts_preserving_requested_alias(concepts)
    resolved_columns = [
        _resolve_concept_column(filtered, concept) for concept in unique_concepts
    ]
    if (
        denominator == 0
        or not resolved_columns
        or any(col is None for col in resolved_columns)
    ):
        return {
            "n_joint_complete": 0,
            "joint_fraction_complete": 0.0,
        }
    event_default_false = _normalise_event_default_false_concepts(
        event_default_false_concepts
    )
    complete = None
    for concept, column in zip(unique_concepts, resolved_columns):
        assert column is not None
        current = _present_mask(
            filtered,
            column,
            event_default_false=_is_event_default_false(
                concept,
                event_default_false,
            ),
        )
        complete = current if complete is None else (complete & current)
    if complete is None:
        return {
            "n_joint_complete": 0,
            "joint_fraction_complete": 0.0,
        }
    if unit == "patient":
        id_col = _patient_id_column(filtered)
        n_joint = int(filtered.loc[complete, id_col].dropna().nunique())
    else:
        n_joint = int(complete.sum())
    return {
        "n_joint_complete": n_joint,
        "joint_fraction_complete": _fraction_present(
            n_present=n_joint,
            denominator_n=denominator,
        ),
    }


def _zero_joint_from_frame(
    frame: Any,
    *,
    analytic_unit: Literal["stay", "patient"],
    cohort: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    filtered, _cohort_summary = _apply_cohort_filter(frame, cohort)
    _denominator_n(filtered, _normalise_analytic_unit(analytic_unit))
    return {
        "n_joint_complete": 0,
        "joint_fraction_complete": 0.0,
    }


def _blocked_real_data_feasibility(
    *,
    concept: str,
    database: str,
    analytic_unit: Literal["stay", "patient"],
    reason: str,
    time_window: Optional[str],
    aggregation: Optional[str],
    availability_cell: Optional[ConceptDatabaseAvailability] = None,
) -> RealDataConceptFeasibility:
    return RealDataConceptFeasibility(
        concept=normalize_concept_name(concept),
        database=database,
        analytic_unit=analytic_unit,
        denominator_n=0,
        n_present=0,
        fraction_missing=0.0,
        n_joint_complete=0,
        joint_fraction_complete=0.0,
        missingness_applicable=False,
        structural_unavailable=True,
        availability_status=(
            availability_cell.status if availability_cell else "blocked"
        ),
        availability_reason=availability_cell.reason if availability_cell else reason,
        source_missing_tables=(
            list(availability_cell.source_missing_tables) if availability_cell else []
        ),
        structural_unavailable_concepts=[normalize_concept_name(concept)],
        joint_denominator_concepts=[],
        note=_structural_unavailable_note([concept]),
        time_window_requested=str(time_window) if time_window is not None else None,
        aggregation_requested=str(aggregation) if aggregation is not None else None,
        cohort_filter_summary=f"structural_unavailable: {reason}",
        missingness_severity="low",
    )


def _resolve_concept_column(frame: Any, concept: str) -> Optional[Any]:
    candidates = _concept_column_candidates(concept)
    columns_by_key = {_normalise_column_key(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = columns_by_key.get(_normalise_column_key(candidate))
        if match is not None:
            return match
    return None


def _concept_column_candidates(concept: str) -> List[str]:
    requested = str(concept or "").strip()
    canonical = normalize_concept_name(requested)
    candidates = [requested, canonical]
    candidates.extend(_reverse_concept_aliases().get(canonical, ()))
    return _unique(candidates)


@lru_cache(maxsize=1)
def _reverse_concept_aliases() -> Dict[str, tuple[str, ...]]:
    aliases: Dict[str, List[str]] = {}
    for alias, canonical in _CONCEPT_ALIASES.items():
        key = normalize_concept_name(canonical)
        aliases.setdefault(key, []).append(alias)
    return {key: tuple(_unique(values)) for key, values in aliases.items()}


def _normalise_column_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _denominator_n(frame: Any, analytic_unit: Literal["stay", "patient"]) -> int:
    if analytic_unit == "patient":
        id_col = _patient_id_column(frame)
        return int(frame[id_col].dropna().nunique())
    return int(len(frame))


def _n_present(
    frame: Any,
    column: Any,
    analytic_unit: Literal["stay", "patient"],
    *,
    event_default_false: bool = False,
) -> int:
    present = _present_mask(
        frame,
        column,
        event_default_false=event_default_false,
    )
    if analytic_unit == "patient":
        id_col = _patient_id_column(frame)
        return int(frame.loc[present, id_col].dropna().nunique())
    return int(present.sum())


def _present_mask(
    frame: Any,
    column: Any,
    *,
    event_default_false: bool,
) -> Any:
    if event_default_false:
        import pandas as pd

        return pd.Series(True, index=frame.index)
    return frame[column].notna()


def _normalise_event_default_false_concepts(
    concepts: Optional[Iterable[str]],
) -> set[str]:
    raw = _DEFAULT_EVENT_DEFAULT_FALSE_CONCEPTS if concepts is None else concepts
    out: set[str] = set()
    for concept in raw:
        key = str(concept or "").strip()
        if not key:
            continue
        out.add(key)
        out.add(normalize_concept_name(key))
    return out


def _is_event_default_false(concept: str, concepts: set[str]) -> bool:
    key = str(concept or "").strip()
    return bool(key) and (key in concepts or normalize_concept_name(key) in concepts)


def _unique_concepts_preserving_requested_alias(concepts: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for concept in concepts:
        requested = str(concept or "").strip()
        if not requested:
            continue
        canonical = normalize_concept_name(requested)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(requested)
    return out


def _patient_id_column(frame: Any) -> str:
    for column in ("patient_id", "subject_id", "uniquepid"):
        if column in frame.columns:
            return column
    raise ValueError(
        "analytic_unit='patient' requires a patient_id, subject_id, or uniquepid column"
    )


def _fraction_missing(*, n_present: int, denominator_n: int) -> float:
    if denominator_n <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (float(n_present) / float(denominator_n))))


def _fraction_present(*, n_present: int, denominator_n: int) -> float:
    if denominator_n <= 0:
        return 0.0
    return max(0.0, min(1.0, float(n_present) / float(denominator_n)))


def _missingness_severity(fraction_missing: float) -> Literal["low", "medium", "high"]:
    if fraction_missing <= 0.10:
        return "low"
    if fraction_missing <= 0.30:
        return "medium"
    return "high"


__all__ = [
    "ConceptDatabaseAvailability",
    "PUBLIC_DATABASES",
    "RealDataConceptFeasibility",
    "concept_database_availability_from_load_record",
    "cross_database_concept_availability",
    "default_public_databases",
    "explain_concept_availability",
    "hypothesis_cross_database_feasibility",
    "normalize_concept_name",
    "normalize_database_name",
    "real_data_concept_feasibility",
]
