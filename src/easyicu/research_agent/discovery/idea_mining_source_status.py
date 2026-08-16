"""Provider-free source-status profiling for row-wise derived concepts.

Idea Mining must distinguish a concept that is absent from a database source
from one that is present but sparsely measured.  This module profiles that
distinction directly from an existing EasyICU prepared-data root.  It never
extracts data, calls a provider, or authorizes a scientific analysis.

The profiler is deliberately formula-agnostic.  A caller supplies a typed
specification and a host-owned Arrow computation.  This keeps disease- or
concept-specific formulas out of shared prompts while preserving an auditable
formula identifier in the result.
"""

from __future__ import annotations

from ..canonical_json import sha256_file as _sha256_file

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from easyicu.databases.profiles import get_database_profile

SOURCE_STATUS_SCHEMA_VERSION = "easyicu.idea_mining_source_status/1"

ArrowFormula = Callable[[Mapping[str, pa.ChunkedArray]], pa.Array | pa.ChunkedArray]


class ComparisonSourceSpec(BaseModel):
    """Optional measured alternative used to contextualize a derived concept."""

    table: str
    column: str
    valid_range: tuple[float, float] | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class RowwiseDerivedConceptSpec(BaseModel):
    """Data-only contract for a row-wise derived measurement."""

    concept_name: str
    denominator_table: str = "demographics"
    source_table: str
    time_column: str = "charttime"
    component_columns: tuple[str, ...]
    formula_id: str
    valid_range: tuple[float, float]
    materialized_column: str | None = None
    predictor_authority: Literal["host_recomputed", "materialized_column"] = (
        "host_recomputed"
    )
    materialized_comparison_semantics: Literal[
        "same_row_expected", "nonlinear_post_aggregation_not_equivalent"
    ] = "same_row_expected"
    comparison_source: ComparisonSourceSpec | None = None
    formula_tolerance: float = Field(default=1e-6, ge=0.0)
    material_difference_threshold: float = Field(default=0.1, gt=0.0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator(
        "concept_name",
        "denominator_table",
        "source_table",
        "time_column",
        "formula_id",
        "materialized_column",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("source-status text fields must be non-empty")
        return text

    @field_validator("component_columns", mode="before")
    @classmethod
    def _normalize_components(cls, value: object) -> tuple[str, ...]:
        values = tuple(str(item).strip() for item in value)  # type: ignore[arg-type]
        if len(values) < 2 or any(not item for item in values):
            raise ValueError("at least two non-empty component columns are required")
        if len(set(values)) != len(values):
            raise ValueError("component columns must be unique")
        return values

    @model_validator(mode="after")
    def _validate_range(self) -> "RowwiseDerivedConceptSpec":
        if self.valid_range[0] >= self.valid_range[1]:
            raise ValueError("valid_range lower bound must be less than upper bound")
        if self.material_difference_threshold < self.formula_tolerance:
            raise ValueError(
                "material_difference_threshold must be at least formula_tolerance"
            )
        if (
            self.predictor_authority == "materialized_column"
            and not self.materialized_column
        ):
            raise ValueError(
                "materialized_column is required when it is the predictor authority"
            )
        return self


class PreparedFileBinding(BaseModel):
    """Byte and schema binding for one prepared parquet input."""

    relative_path: str
    sha256: str
    size_bytes: int
    parquet_rows: int
    schema_sha256: str
    selected_column_types: dict[str, str]

    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnCoverage(BaseModel):
    """Coverage of one source column against the demographic stay denominator."""

    column: str
    source_present: bool
    non_null_rows: int
    observed_stays: int
    denominator_stays: int
    observed_fraction: float
    dtype: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceStatusPartition(BaseModel):
    """Exclusive stay-level states for the derived construct."""

    structural_no_source: int
    source_present_unmeasured: int
    contradictory_or_out_of_range: int
    valid_observed: int

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def total(self) -> int:
        return (
            self.structural_no_source
            + self.source_present_unmeasured
            + self.contradictory_or_out_of_range
            + self.valid_observed
        )


class FormulaAgreement(BaseModel):
    """Agreement of a materialized column with the host-owned recomputation."""

    comparable_rows: int
    within_tolerance_rows: int
    mismatch_rows: int
    material_difference_threshold: float
    material_difference_rows: int
    material_difference_fraction: float
    max_absolute_difference: float | None
    mean_absolute_difference: float | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class PredictorOutcomePairCoverage(BaseModel):
    """Stay-level overlap between a derived predictor and comparison outcome."""

    predictor_valid_stays: int
    outcome_valid_stays: int
    joint_valid_stays: int
    denominator_stays: int
    joint_fraction: float

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatabaseDerivedConceptProfile(BaseModel):
    """One database's exact source-status profile."""

    database: str
    stay_id_column: str
    denominator_rows: int
    denominator_stays: int
    missing_denominator_ids: int
    duplicate_denominator_ids: int
    source_rows: int
    source_stays_outside_denominator: int
    component_coverage: tuple[ColumnCoverage, ...]
    exact_component_rows: int
    exact_component_stays: int
    recomputed_valid_rows: int
    recomputed_valid_stays: int
    materialized_coverage: ColumnCoverage | None
    comparison_coverage: ColumnCoverage | None
    predictor_outcome_pair_coverage: PredictorOutcomePairCoverage | None = None
    source_status: SourceStatusPartition
    formula_agreement: FormulaAgreement | None
    data_readiness: Literal[
        "ready", "partial", "structural_no_source", "invalid_denominator"
    ]
    warnings: tuple[str, ...] = ()
    input_files: tuple[PreparedFileBinding, ...]

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _partition_matches_denominator(self) -> "DatabaseDerivedConceptProfile":
        if self.source_status.total != self.denominator_stays:
            raise ValueError("source-status partition must equal denominator stays")
        return self


class MeasurementAuditCriteria(BaseModel):
    """Predeclared data-answerability criteria for a measurement audit."""

    min_databases_with_valid_observations: int = Field(default=3, ge=2)
    min_valid_stays_per_database: int = Field(default=500, ge=1)
    min_cross_database_coverage_range: float = Field(default=0.20, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class MeasurementAuditAnswerability(BaseModel):
    """Deterministic review eligibility, not analysis or paper authorization."""

    status: Literal[
        "answerable_requires_human_confirmation",
        "insufficient_database_coverage",
        "insufficient_cross_database_variation",
    ]
    criteria: MeasurementAuditCriteria
    eligible_databases: tuple[str, ...]
    excluded_databases: tuple[str, ...]
    minimum_coverage_fraction: float | None
    maximum_coverage_fraction: float | None
    coverage_range: float | None
    reason: str
    requires_human_confirmation: bool = True
    analysis_authorized: bool = False
    paper_authorized: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class PairAnswerabilityCriteria(BaseModel):
    """Predeclared feasibility floor for a predictor/outcome pair."""

    min_databases_with_joint_observations: int = Field(default=3, ge=1)
    min_joint_stays_per_database: int = Field(default=500, ge=1)
    min_joint_fraction_per_database: float = Field(default=0.01, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class PairAnswerability(BaseModel):
    """Data feasibility only; timing and scientific design remain unresolved."""

    status: Literal[
        "answerable_requires_temporal_protocol",
        "insufficient_joint_coverage",
        "comparison_source_not_configured",
    ]
    criteria: PairAnswerabilityCriteria
    eligible_databases: tuple[str, ...]
    excluded_databases: tuple[str, ...]
    reason: str
    requires_temporal_protocol: bool = True
    requires_human_confirmation: bool = True
    analysis_authorized: bool = False
    paper_authorized: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class CrossDatabaseDerivedConceptProfile(BaseModel):
    """Auditable cross-database result; never an analysis authorization."""

    schema_version: str = SOURCE_STATUS_SCHEMA_VERSION
    export_root: str
    concept_spec: RowwiseDerivedConceptSpec
    databases: tuple[DatabaseDerivedConceptProfile, ...]
    n_databases_ready: int
    n_databases_profiled: int
    measurement_audit_answerability: MeasurementAuditAnswerability | None = None
    pair_answerability: PairAnswerability | None = None
    analysis_authorized: bool = False
    paper_authorized: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


def assess_pair_answerability(
    databases: Sequence[DatabaseDerivedConceptProfile],
    *,
    criteria: PairAnswerabilityCriteria,
) -> PairAnswerability:
    """Require observed predictor/outcome overlap in multiple databases.

    This gate deliberately does not infer a predictor window, outcome onset,
    eligibility criteria, estimand, or causal direction.  Passing it only
    means the existing exports contain enough stay-level overlap to justify
    human design of a temporal protocol.
    """

    configured = [
        row for row in databases if row.predictor_outcome_pair_coverage is not None
    ]
    if not configured:
        return PairAnswerability(
            status="comparison_source_not_configured",
            criteria=criteria,
            eligible_databases=(),
            excluded_databases=tuple(row.database for row in databases),
            reason="No comparison outcome source was configured for this profile.",
        )

    eligible: list[str] = []
    excluded: list[str] = []
    for row in databases:
        pair = row.predictor_outcome_pair_coverage
        if (
            pair is not None
            and pair.joint_valid_stays >= criteria.min_joint_stays_per_database
            and pair.joint_fraction >= criteria.min_joint_fraction_per_database
        ):
            eligible.append(row.database)
        else:
            excluded.append(row.database)
    if len(eligible) < criteria.min_databases_with_joint_observations:
        return PairAnswerability(
            status="insufficient_joint_coverage",
            criteria=criteria,
            eligible_databases=tuple(eligible),
            excluded_databases=tuple(excluded),
            reason=(
                f"Only {len(eligible)} databases meet the predeclared joint "
                f"coverage floor; {criteria.min_databases_with_joint_observations} "
                "are required."
            ),
        )
    return PairAnswerability(
        status="answerable_requires_temporal_protocol",
        criteria=criteria,
        eligible_databases=tuple(eligible),
        excluded_databases=tuple(excluded),
        reason=(
            f"{len(eligible)} databases meet the predictor/outcome overlap floor; "
            "a pre-outcome predictor window and outcome-onset protocol remain required."
        ),
    )


def assess_measurement_audit_answerability(
    databases: Sequence[DatabaseDerivedConceptProfile],
    *,
    criteria: MeasurementAuditCriteria,
) -> MeasurementAuditAnswerability:
    """Check whether a cross-database source-status audit is estimable.

    This gate intentionally does not ask whether an association is novel or
    positive.  It only requires enough observed stays in enough databases and
    a predeclared coverage contrast large enough to support a measurement-
    availability comparison.
    """

    eligible: list[tuple[str, float]] = []
    excluded: list[str] = []
    for row in databases:
        if (
            row.denominator_stays > 0
            and row.recomputed_valid_stays >= criteria.min_valid_stays_per_database
        ):
            eligible.append(
                (
                    row.database,
                    row.recomputed_valid_stays / row.denominator_stays,
                )
            )
        else:
            excluded.append(row.database)

    fractions = [fraction for _, fraction in eligible]
    minimum = min(fractions) if fractions else None
    maximum = max(fractions) if fractions else None
    coverage_range = (
        maximum - minimum if minimum is not None and maximum is not None else None
    )
    if len(eligible) < criteria.min_databases_with_valid_observations:
        status = "insufficient_database_coverage"
        reason = (
            f"Only {len(eligible)} databases meet the predeclared minimum of "
            f"{criteria.min_valid_stays_per_database} valid stays; "
            f"{criteria.min_databases_with_valid_observations} are required."
        )
    elif (
        coverage_range is None
        or coverage_range < criteria.min_cross_database_coverage_range
    ):
        status = "insufficient_cross_database_variation"
        reason = (
            "Observed cross-database coverage range is below the predeclared "
            f"minimum of {criteria.min_cross_database_coverage_range:.3f}."
        )
    else:
        status = "answerable_requires_human_confirmation"
        reason = (
            f"{len(eligible)} databases meet the stay-count requirement and "
            f"the observed coverage range is {coverage_range:.3f}; literature, "
            "source-semantics, and human review remain required."
        )
    return MeasurementAuditAnswerability(
        status=status,
        criteria=criteria,
        eligible_databases=tuple(database for database, _ in eligible),
        excluded_databases=tuple(excluded),
        minimum_coverage_fraction=minimum,
        maximum_coverage_fraction=maximum,
        coverage_range=coverage_range,
        reason=reason,
    )


def _schema_sha256(schema: pa.Schema) -> str:
    payload = [(field.name, str(field.type), field.nullable) for field in schema]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _file_binding(
    path: Path,
    *,
    export_root: Path,
    selected_columns: Sequence[str],
) -> PreparedFileBinding:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return PreparedFileBinding(
        relative_path=path.relative_to(export_root).as_posix(),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        parquet_rows=parquet.metadata.num_rows,
        schema_sha256=_schema_sha256(schema),
        selected_column_types={
            column: str(schema.field(column).type)
            for column in selected_columns
            if column in schema.names
        },
    )


def _true_count(mask: pa.Array | pa.ChunkedArray) -> int:
    return int(pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0)


def _non_null_unique_values(
    values: pa.Array | pa.ChunkedArray,
    mask: pa.Array | pa.ChunkedArray | None = None,
) -> set[object]:
    selected = (
        pc.filter(values, pc.fill_null(mask, False)) if mask is not None else values
    )
    return {item for item in pc.unique(pc.drop_null(selected)).to_pylist()}


def _normalized_source_ids(
    source_ids: pa.ChunkedArray,
    denominator_ids: pa.ChunkedArray,
) -> tuple[pa.ChunkedArray, pa.Array]:
    """Cast denominator IDs to the source type without stringifying millions of rows."""

    try:
        denominator_set = pc.cast(
            pc.unique(pc.drop_null(denominator_ids)), source_ids.type, safe=True
        )
        return source_ids, denominator_set
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return (
            pc.cast(source_ids, pa.string()),
            pc.cast(pc.unique(pc.drop_null(denominator_ids)), pa.string()),
        )


def _in_denominator_mask(
    source_ids: pa.ChunkedArray,
    denominator_ids: pa.ChunkedArray,
) -> tuple[pa.ChunkedArray, pa.Array | pa.ChunkedArray]:
    normalized, value_set = _normalized_source_ids(source_ids, denominator_ids)
    return normalized, pc.is_in(normalized, value_set=value_set)


def _coverage(
    *,
    column: str,
    table: pa.Table,
    id_column: str,
    denominator_ids: pa.ChunkedArray,
) -> ColumnCoverage:
    source_ids, in_denominator = _in_denominator_mask(table[id_column], denominator_ids)
    valid = pc.and_(pc.is_valid(table[column]), in_denominator)
    observed = _non_null_unique_values(source_ids, valid)
    denominator_n = len(_non_null_unique_values(denominator_ids))
    return ColumnCoverage(
        column=column,
        source_present=True,
        non_null_rows=_true_count(valid),
        observed_stays=len(observed),
        denominator_stays=denominator_n,
        observed_fraction=(len(observed) / denominator_n if denominator_n else 0.0),
        dtype=str(table[column].type),
    )


def _missing_coverage(column: str, denominator_n: int) -> ColumnCoverage:
    return ColumnCoverage(
        column=column,
        source_present=False,
        non_null_rows=0,
        observed_stays=0,
        denominator_stays=denominator_n,
        observed_fraction=0.0,
    )


def _numeric_valid_mask(
    values: pa.Array | pa.ChunkedArray,
    valid_range: tuple[float, float],
) -> pa.Array | pa.ChunkedArray:
    numeric = pc.cast(values, pa.float64(), safe=False)
    return pc.and_(
        pc.is_finite(numeric),
        pc.and_(
            pc.greater_equal(numeric, valid_range[0]),
            pc.less_equal(numeric, valid_range[1]),
        ),
    )


def _formula_agreement(
    recomputed: pa.Array | pa.ChunkedArray,
    materialized: pa.Array | pa.ChunkedArray,
    *,
    comparable_mask: pa.Array | pa.ChunkedArray,
    tolerance: float,
    material_difference_threshold: float,
) -> FormulaAgreement:
    differences = pc.filter(
        pc.abs(
            pc.subtract(
                pc.cast(recomputed, pa.float64(), safe=False),
                pc.cast(materialized, pa.float64(), safe=False),
            )
        ),
        pc.fill_null(comparable_mask, False),
    )
    comparable = len(differences)
    if not comparable:
        return FormulaAgreement(
            comparable_rows=0,
            within_tolerance_rows=0,
            mismatch_rows=0,
            material_difference_threshold=material_difference_threshold,
            material_difference_rows=0,
            material_difference_fraction=0.0,
            max_absolute_difference=None,
            mean_absolute_difference=None,
        )
    within = _true_count(pc.less_equal(differences, tolerance))
    material = _true_count(pc.greater(differences, material_difference_threshold))
    return FormulaAgreement(
        comparable_rows=comparable,
        within_tolerance_rows=within,
        mismatch_rows=comparable - within,
        material_difference_threshold=material_difference_threshold,
        material_difference_rows=material,
        material_difference_fraction=material / comparable,
        max_absolute_difference=float(pc.max(differences).as_py()),
        mean_absolute_difference=float(pc.mean(differences).as_py()),
    )


def _comparison_profile(
    *,
    database_dir: Path,
    export_root: Path,
    id_column: str,
    denominator_ids: pa.ChunkedArray,
    spec: ComparisonSourceSpec,
) -> tuple[
    ColumnCoverage,
    set[object],
    PreparedFileBinding | None,
    tuple[str, ...],
]:
    path = database_dir / f"{spec.table}.parquet"
    denominator_n = len(_non_null_unique_values(denominator_ids))
    if not path.is_file():
        return (
            _missing_coverage(spec.column, denominator_n),
            set(),
            None,
            (f"comparison table missing: {path.name}",),
        )
    schema = pq.read_schema(path)
    required = [id_column, spec.column]
    if any(column not in schema.names for column in required):
        return (
            _missing_coverage(spec.column, denominator_n),
            set(),
            _file_binding(path, export_root=export_root, selected_columns=required),
            (f"comparison column missing: {spec.column}",),
        )
    table = pq.read_table(path, columns=required)
    source_ids, in_denominator = _in_denominator_mask(table[id_column], denominator_ids)
    valid = pc.and_(pc.is_valid(table[spec.column]), in_denominator)
    if spec.valid_range is not None:
        valid = pc.and_(
            valid, _numeric_valid_mask(table[spec.column], spec.valid_range)
        )
    observed = _non_null_unique_values(source_ids, valid)
    coverage = ColumnCoverage(
        column=spec.column,
        source_present=True,
        non_null_rows=_true_count(valid),
        observed_stays=len(observed),
        denominator_stays=denominator_n,
        observed_fraction=(len(observed) / denominator_n if denominator_n else 0.0),
        dtype=str(table[spec.column].type),
    )
    return (
        coverage,
        observed,
        _file_binding(path, export_root=export_root, selected_columns=required),
        (),
    )


def _profile_database(
    *,
    export_root: Path,
    database: str,
    spec: RowwiseDerivedConceptSpec,
    formula: ArrowFormula,
) -> DatabaseDerivedConceptProfile:
    profile = get_database_profile(database)
    database_dir = export_root / profile.key
    denominator_path = database_dir / f"{spec.denominator_table}.parquet"
    source_path = database_dir / f"{spec.source_table}.parquet"
    if not denominator_path.is_file():
        raise FileNotFoundError(f"denominator table missing: {denominator_path}")

    denominator_schema = pq.read_schema(denominator_path)
    if profile.stay_id_col not in denominator_schema.names:
        raise ValueError(
            f"denominator ID column '{profile.stay_id_col}' missing in {denominator_path}"
        )
    denominator = pq.read_table(denominator_path, columns=[profile.stay_id_col])
    denominator_ids = denominator[profile.stay_id_col]
    denominator_rows = len(denominator)
    missing_denominator_ids = denominator_ids.null_count
    denominator_values = _non_null_unique_values(denominator_ids)
    denominator_n = len(denominator_values)
    duplicate_denominator_ids = (
        denominator_rows - missing_denominator_ids - denominator_n
    )
    files = [
        _file_binding(
            denominator_path,
            export_root=export_root,
            selected_columns=[profile.stay_id_col],
        )
    ]
    warnings: list[str] = []
    if missing_denominator_ids:
        warnings.append("denominator contains missing stay identifiers")
    if duplicate_denominator_ids:
        warnings.append("denominator contains duplicate stay identifiers")

    required = [
        profile.stay_id_col,
        spec.time_column,
        *spec.component_columns,
    ]
    if spec.materialized_column:
        required.append(spec.materialized_column)
    if not source_path.is_file():
        component_coverage = tuple(
            _missing_coverage(column, denominator_n)
            for column in spec.component_columns
        )
        comparison_coverage: ColumnCoverage | None = None
        if spec.comparison_source is not None:
            (
                comparison_coverage,
                comparison_ids,
                comparison_binding,
                comparison_warnings,
            ) = _comparison_profile(
                database_dir=database_dir,
                export_root=export_root,
                id_column=profile.stay_id_col,
                denominator_ids=denominator_ids,
                spec=spec.comparison_source,
            )
            if comparison_binding is not None:
                files.append(comparison_binding)
            warnings.extend(comparison_warnings)
        else:
            comparison_ids = set()
        return DatabaseDerivedConceptProfile(
            database=profile.key,
            stay_id_column=profile.stay_id_col,
            denominator_rows=denominator_rows,
            denominator_stays=denominator_n,
            missing_denominator_ids=missing_denominator_ids,
            duplicate_denominator_ids=duplicate_denominator_ids,
            source_rows=0,
            source_stays_outside_denominator=0,
            component_coverage=component_coverage,
            exact_component_rows=0,
            exact_component_stays=0,
            recomputed_valid_rows=0,
            recomputed_valid_stays=0,
            materialized_coverage=(
                _missing_coverage(spec.materialized_column, denominator_n)
                if spec.materialized_column
                else None
            ),
            comparison_coverage=comparison_coverage,
            predictor_outcome_pair_coverage=(
                PredictorOutcomePairCoverage(
                    predictor_valid_stays=0,
                    outcome_valid_stays=len(comparison_ids),
                    joint_valid_stays=0,
                    denominator_stays=denominator_n,
                    joint_fraction=0.0,
                )
                if spec.comparison_source is not None
                else None
            ),
            source_status=SourceStatusPartition(
                structural_no_source=denominator_n,
                source_present_unmeasured=0,
                contradictory_or_out_of_range=0,
                valid_observed=0,
            ),
            formula_agreement=None,
            data_readiness=(
                "invalid_denominator" if not denominator_n else "structural_no_source"
            ),
            warnings=tuple([*warnings, f"source table missing: {source_path.name}"]),
            input_files=tuple(files),
        )

    source_schema = pq.read_schema(source_path)
    files.append(
        _file_binding(source_path, export_root=export_root, selected_columns=required)
    )
    required_for_recomputation = [
        profile.stay_id_col,
        spec.time_column,
        *spec.component_columns,
    ]
    missing_columns = [
        column
        for column in required_for_recomputation
        if column not in source_schema.names
    ]
    available_components = [
        column for column in spec.component_columns if column in source_schema.names
    ]
    if missing_columns:
        warnings.extend(
            f"required source column missing: {column}" for column in missing_columns
        )
        inspectable = [
            column
            for column in [
                profile.stay_id_col,
                *available_components,
                spec.materialized_column,
            ]
            if column and column in source_schema.names
        ]
        partial_source = (
            pq.read_table(source_path, columns=list(dict.fromkeys(inspectable)))
            if inspectable
            else None
        )
        can_bind_to_denominator = (
            partial_source is not None
            and profile.stay_id_col in partial_source.column_names
        )
        partial_coverages: list[ColumnCoverage] = []
        for column in spec.component_columns:
            if (
                can_bind_to_denominator
                and partial_source is not None
                and column in partial_source.column_names
            ):
                partial_coverages.append(
                    _coverage(
                        column=column,
                        table=partial_source,
                        id_column=profile.stay_id_col,
                        denominator_ids=denominator_ids,
                    )
                )
            elif column in source_schema.names:
                partial_coverages.append(
                    ColumnCoverage(
                        column=column,
                        source_present=True,
                        non_null_rows=0,
                        observed_stays=0,
                        denominator_stays=denominator_n,
                        observed_fraction=0.0,
                        dtype=str(source_schema.field(column).type),
                    )
                )
            else:
                partial_coverages.append(_missing_coverage(column, denominator_n))
        partial_materialized: ColumnCoverage | None = None
        if spec.materialized_column:
            if (
                can_bind_to_denominator
                and partial_source is not None
                and spec.materialized_column in partial_source.column_names
            ):
                partial_materialized = _coverage(
                    column=spec.materialized_column,
                    table=partial_source,
                    id_column=profile.stay_id_col,
                    denominator_ids=denominator_ids,
                )
            elif spec.materialized_column in source_schema.names:
                partial_materialized = ColumnCoverage(
                    column=spec.materialized_column,
                    source_present=True,
                    non_null_rows=0,
                    observed_stays=0,
                    denominator_stays=denominator_n,
                    observed_fraction=0.0,
                    dtype=str(source_schema.field(spec.materialized_column).type),
                )
            else:
                partial_materialized = _missing_coverage(
                    spec.materialized_column, denominator_n
                )
        partial_comparison: ColumnCoverage | None = None
        if spec.comparison_source is not None:
            (
                partial_comparison,
                comparison_ids,
                comparison_binding,
                comparison_warnings,
            ) = _comparison_profile(
                database_dir=database_dir,
                export_root=export_root,
                id_column=profile.stay_id_col,
                denominator_ids=denominator_ids,
                spec=spec.comparison_source,
            )
            if comparison_binding is not None:
                files.append(comparison_binding)
            warnings.extend(comparison_warnings)
        else:
            comparison_ids = set()
        return DatabaseDerivedConceptProfile(
            database=profile.key,
            stay_id_column=profile.stay_id_col,
            denominator_rows=denominator_rows,
            denominator_stays=denominator_n,
            missing_denominator_ids=missing_denominator_ids,
            duplicate_denominator_ids=duplicate_denominator_ids,
            source_rows=pq.ParquetFile(source_path).metadata.num_rows,
            source_stays_outside_denominator=0,
            component_coverage=tuple(partial_coverages),
            exact_component_rows=0,
            exact_component_stays=0,
            recomputed_valid_rows=0,
            recomputed_valid_stays=0,
            materialized_coverage=partial_materialized,
            comparison_coverage=partial_comparison,
            predictor_outcome_pair_coverage=(
                PredictorOutcomePairCoverage(
                    predictor_valid_stays=0,
                    outcome_valid_stays=len(comparison_ids),
                    joint_valid_stays=0,
                    denominator_stays=denominator_n,
                    joint_fraction=0.0,
                )
                if spec.comparison_source is not None
                else None
            ),
            source_status=SourceStatusPartition(
                structural_no_source=denominator_n,
                source_present_unmeasured=0,
                contradictory_or_out_of_range=0,
                valid_observed=0,
            ),
            formula_agreement=None,
            data_readiness=(
                "invalid_denominator" if not denominator_n else "structural_no_source"
            ),
            warnings=tuple(warnings),
            input_files=tuple(files),
        )

    read_columns = list(
        dict.fromkeys(column for column in required if column in source_schema.names)
    )
    source = pq.read_table(source_path, columns=read_columns)
    source_ids, in_denominator = _in_denominator_mask(
        source[profile.stay_id_col], denominator_ids
    )
    # Compare through the same normalized mask that _in_denominator_mask
    # produced, never as a raw Python set difference: when source and
    # denominator ID types differ and the normalizer falls back to strings,
    # a set difference between original-typed values counts every stay as
    # "outside".
    outside_mask = pc.invert(pc.fill_null(in_denominator, False))
    source_stays_outside = len(
        _non_null_unique_values(source_ids, outside_mask)
    )
    component_coverage = tuple(
        _coverage(
            column=column,
            table=source,
            id_column=profile.stay_id_col,
            denominator_ids=denominator_ids,
        )
        for column in spec.component_columns
    )

    exact_mask: pa.Array | pa.ChunkedArray = pc.and_(
        pc.and_(pc.is_valid(source[profile.stay_id_col]), in_denominator),
        pc.is_valid(source[spec.time_column]),
    )
    for column in spec.component_columns:
        exact_mask = pc.and_(exact_mask, pc.is_valid(source[column]))

    recomputed = formula({column: source[column] for column in spec.component_columns})
    if len(recomputed) != len(source):
        raise ValueError("host formula output length does not match source rows")
    valid_recomputed = pc.and_(
        exact_mask, _numeric_valid_mask(recomputed, spec.valid_range)
    )
    exact_ids = _non_null_unique_values(source_ids, exact_mask)
    valid_ids = _non_null_unique_values(source_ids, valid_recomputed)
    invalid_ids = _non_null_unique_values(
        source_ids,
        pc.and_(exact_mask, pc.invert(pc.fill_null(valid_recomputed, False))),
    )
    contradictory_ids = invalid_ids - valid_ids
    unmeasured = denominator_n - len(valid_ids) - len(contradictory_ids)

    materialized_coverage: ColumnCoverage | None = None
    agreement: FormulaAgreement | None = None
    materialized_valid_ids: set[object] = set()
    if spec.materialized_column:
        if spec.materialized_column in source_schema.names:
            materialized_coverage = _coverage(
                column=spec.materialized_column,
                table=source,
                id_column=profile.stay_id_col,
                denominator_ids=denominator_ids,
            )
            comparable = pc.and_(
                valid_recomputed,
                _numeric_valid_mask(source[spec.materialized_column], spec.valid_range),
            )
            materialized_valid = pc.and_(
                pc.and_(pc.is_valid(source[profile.stay_id_col]), in_denominator),
                _numeric_valid_mask(source[spec.materialized_column], spec.valid_range),
            )
            materialized_valid_ids = _non_null_unique_values(
                source_ids, materialized_valid
            )
            agreement = _formula_agreement(
                recomputed,
                source[spec.materialized_column],
                comparable_mask=comparable,
                tolerance=spec.formula_tolerance,
                material_difference_threshold=spec.material_difference_threshold,
            )
            if (
                agreement.material_difference_rows
                and spec.materialized_comparison_semantics == "same_row_expected"
            ):
                warnings.append(
                    "materialized values materially differ from host recomputation"
                )
            elif (
                agreement.material_difference_rows
                and spec.materialized_comparison_semantics
                == "nonlinear_post_aggregation_not_equivalent"
            ):
                warnings.append(
                    "materialized/recomputed comparison is descriptive only because "
                    "the nonlinear derivation precedes wide-table aggregation"
                )
        else:
            materialized_coverage = _missing_coverage(
                spec.materialized_column, denominator_n
            )
            warnings.append(f"materialized column missing: {spec.materialized_column}")

    comparison_coverage: ColumnCoverage | None = None
    if spec.comparison_source is not None:
        (
            comparison_coverage,
            comparison_ids,
            comparison_binding,
            comparison_warnings,
        ) = _comparison_profile(
            database_dir=database_dir,
            export_root=export_root,
            id_column=profile.stay_id_col,
            denominator_ids=denominator_ids,
            spec=spec.comparison_source,
        )
        if comparison_binding is not None:
            files.append(comparison_binding)
        warnings.extend(comparison_warnings)
    else:
        comparison_ids = set()

    pair_predictor_ids = (
        materialized_valid_ids
        if spec.predictor_authority == "materialized_column"
        else valid_ids
    )
    joint_ids = pair_predictor_ids & comparison_ids

    if not denominator_n:
        readiness = "invalid_denominator"
    elif valid_ids:
        readiness = "ready"
    else:
        readiness = "partial"
    return DatabaseDerivedConceptProfile(
        database=profile.key,
        stay_id_column=profile.stay_id_col,
        denominator_rows=denominator_rows,
        denominator_stays=denominator_n,
        missing_denominator_ids=missing_denominator_ids,
        duplicate_denominator_ids=duplicate_denominator_ids,
        source_rows=len(source),
        source_stays_outside_denominator=source_stays_outside,
        component_coverage=component_coverage,
        exact_component_rows=_true_count(exact_mask),
        exact_component_stays=len(exact_ids),
        recomputed_valid_rows=_true_count(valid_recomputed),
        recomputed_valid_stays=len(valid_ids),
        materialized_coverage=materialized_coverage,
        comparison_coverage=comparison_coverage,
        predictor_outcome_pair_coverage=(
            PredictorOutcomePairCoverage(
                predictor_valid_stays=len(pair_predictor_ids),
                outcome_valid_stays=len(comparison_ids),
                joint_valid_stays=len(joint_ids),
                denominator_stays=denominator_n,
                joint_fraction=(
                    len(joint_ids) / denominator_n if denominator_n else 0.0
                ),
            )
            if spec.comparison_source is not None
            else None
        ),
        source_status=SourceStatusPartition(
            structural_no_source=0,
            source_present_unmeasured=max(0, unmeasured),
            contradictory_or_out_of_range=len(contradictory_ids),
            valid_observed=len(valid_ids),
        ),
        formula_agreement=agreement,
        data_readiness=readiness,
        warnings=tuple(warnings),
        input_files=tuple(files),
    )


def profile_rowwise_derived_concept(
    export_root: str | Path,
    *,
    databases: Sequence[str],
    spec: RowwiseDerivedConceptSpec,
    formula: ArrowFormula,
    measurement_audit_criteria: MeasurementAuditCriteria | None = None,
    pair_answerability_criteria: PairAnswerabilityCriteria | None = None,
) -> CrossDatabaseDerivedConceptProfile:
    """Profile one host-defined derived construct over existing prepared data."""

    root = Path(export_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"prepared-data root not found: {root}")
    requested = tuple(str(database).strip() for database in databases)
    if not requested or any(not database for database in requested):
        raise ValueError("at least one database is required")
    if len(set(requested)) != len(requested):
        raise ValueError("database list must not contain duplicates")

    rows = tuple(
        _profile_database(
            export_root=root,
            database=database,
            spec=spec,
            formula=formula,
        )
        for database in requested
    )
    answerability = (
        assess_measurement_audit_answerability(
            rows, criteria=measurement_audit_criteria
        )
        if measurement_audit_criteria is not None
        else None
    )
    pair_answerability = (
        assess_pair_answerability(rows, criteria=pair_answerability_criteria)
        if pair_answerability_criteria is not None
        else None
    )
    return CrossDatabaseDerivedConceptProfile(
        export_root=str(root),
        concept_spec=spec,
        databases=rows,
        n_databases_ready=sum(row.data_readiness == "ready" for row in rows),
        n_databases_profiled=len(rows),
        measurement_audit_answerability=answerability,
        pair_answerability=pair_answerability,
    )
