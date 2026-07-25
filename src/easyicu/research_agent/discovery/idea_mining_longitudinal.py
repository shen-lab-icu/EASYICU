"""Host-owned longitudinal readiness for design-first Idea Mining.

The ordinary data-first route enumerates predictor/outcome pairs.  That shape
cannot represent a question such as whether a repeated physiologic trajectory
is reproducible across databases.  This leaf supplies the missing, case-neutral
route: verify that a named value is genuinely longitudinal in several prepared
database artifacts, then emit a *human-review candidate* for cross-database
trajectory transportability.

The output is feasibility evidence, never a novelty or scientific-result claim.
In particular, the mere presence of similarly named parquet files is
insufficient: every admitted database must expose an explicit unit identifier,
time coordinate, non-null value support, and repeated observations in the
bounded data sample.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..concept_availability import normalize_concept_name

LONGITUDINAL_DISCOVERY_SCHEMA_VERSION = "easyicu.longitudinal_discovery/1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LongitudinalArtifactProfile:
    """Verified bounded profile of one value in one prepared database table."""

    concept: str
    database: str
    artifact_path: str
    artifact_sha256: str
    row_count: int
    id_column: str
    time_column: str
    value_column: str
    sample_row_count: int
    sample_unit_count: int
    sample_distinct_time_count: int
    sample_units_with_repeats: int
    sample_repeated_unit_fraction: float
    sample_median_observations_per_unit: float
    sample_value_nonnull_fraction: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LongitudinalTransportabilityCandidate:
    """One evidence-bound cross-database trajectory review candidate."""

    concept: str
    analysis_family: str
    design_archetype: str
    ready_database_count: int
    ready_databases: tuple[str, ...]
    total_databases_profiled: int
    artifact_profiles: tuple[LongitudinalArtifactProfile, ...]
    differentiator_note: str
    requires_human_confirmation: bool = True
    novelty_claimed: bool = False
    scientific_result_claimed: bool = False
    paper_authorized: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["artifact_profiles"] = [
            profile.to_dict() for profile in self.artifact_profiles
        ]
        return payload


def _bounded_batches(
    parquet_file: pq.ParquetFile,
    *,
    columns: Sequence[str],
    sample_rows: int,
) -> pa.RecordBatch:
    if sample_rows <= 0:
        return pa.RecordBatch.from_arrays(
            [
                pa.array([], type=parquet_file.schema_arrow.field(name).type)
                for name in columns
            ],
            names=list(columns),
        )
    batches: List[pa.RecordBatch] = []
    remaining = sample_rows
    for batch in parquet_file.iter_batches(
        batch_size=min(sample_rows, 100_000), columns=list(columns)
    ):
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        remaining -= batch.num_rows
    if not batches:
        return pa.RecordBatch.from_arrays(
            [
                pa.array([], type=parquet_file.schema_arrow.field(name).type)
                for name in columns
            ],
            names=list(columns),
        )
    table = pa.Table.from_batches(batches).combine_chunks()
    return table.to_batches(max_chunksize=table.num_rows)[0]


def profile_longitudinal_table(
    *,
    path: str | Path,
    database: str,
    id_column: str,
    time_column: str,
    value_columns: Sequence[str],
    concept_by_value_column: Optional[Mapping[str, str]] = None,
    sample_rows: int = 100_000,
) -> List[LongitudinalArtifactProfile]:
    """Profile explicit longitudinal columns without loading the whole table.

    ``id_column`` and ``time_column`` are host-declared coordinates; this
    function never guesses them from similarly named columns.  A bounded prefix
    is used only for readiness triage.  The full artifact bytes and full parquet
    row count remain bound into each returned profile.
    """

    artifact = Path(path).resolve()
    if not artifact.is_file() or artifact.is_symlink():
        raise ValueError(f"longitudinal artifact must be a regular file: {artifact}")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    requested_values = list(dict.fromkeys(str(item) for item in value_columns))
    if not requested_values:
        return []

    parquet_file = pq.ParquetFile(artifact)
    schema = parquet_file.schema_arrow
    required = [id_column, time_column, *requested_values]
    missing = [name for name in required if name not in schema.names]
    if missing:
        raise ValueError(
            "longitudinal artifact is missing declared column(s): " + ", ".join(missing)
        )
    if not (
        pa.types.is_integer(schema.field(id_column).type)
        or pa.types.is_string(schema.field(id_column).type)
    ):
        raise ValueError("declared longitudinal id column has unsupported type")
    if not (
        pa.types.is_integer(schema.field(time_column).type)
        or pa.types.is_floating(schema.field(time_column).type)
        or pa.types.is_timestamp(schema.field(time_column).type)
    ):
        raise ValueError("declared longitudinal time column has unsupported type")
    for value_column in requested_values:
        value_type = schema.field(value_column).type
        if not (
            pa.types.is_integer(value_type)
            or pa.types.is_floating(value_type)
            or pa.types.is_boolean(value_type)
        ):
            raise ValueError(
                f"declared longitudinal value column is not scalar numeric: {value_column}"
            )

    batch = _bounded_batches(
        parquet_file,
        columns=required,
        sample_rows=min(sample_rows, parquet_file.metadata.num_rows),
    )
    sample_n = batch.num_rows
    ids = batch.column(batch.schema.get_field_index(id_column)).to_pylist()
    times = batch.column(batch.schema.get_field_index(time_column)).to_pylist()
    valid_pairs = [
        (unit, when)
        for unit, when in zip(ids, times)
        if unit is not None and when is not None
    ]
    per_unit = Counter(unit for unit, _ in valid_pairs)
    unit_count = len(per_unit)
    units_with_repeats = sum(count >= 2 for count in per_unit.values())
    repeated_fraction = units_with_repeats / unit_count if unit_count else 0.0
    observations = list(per_unit.values())
    median_observations = float(median(observations)) if observations else 0.0
    distinct_times = len({when for _, when in valid_pairs})
    artifact_sha256 = _sha256_file(artifact)
    concept_names = concept_by_value_column or {}

    profiles: List[LongitudinalArtifactProfile] = []
    for value_column in requested_values:
        values = batch.column(batch.schema.get_field_index(value_column))
        nonnull = int(pc.count(values).as_py()) if sample_n else 0
        nonnull_fraction = nonnull / sample_n if sample_n else 0.0
        concept = normalize_concept_name(
            str(concept_names.get(value_column, value_column))
        )
        profiles.append(
            LongitudinalArtifactProfile(
                concept=concept,
                database=str(database),
                artifact_path=str(artifact),
                artifact_sha256=artifact_sha256,
                row_count=int(parquet_file.metadata.num_rows),
                id_column=id_column,
                time_column=time_column,
                value_column=value_column,
                sample_row_count=sample_n,
                sample_unit_count=unit_count,
                sample_distinct_time_count=distinct_times,
                sample_units_with_repeats=units_with_repeats,
                sample_repeated_unit_fraction=round(repeated_fraction, 8),
                sample_median_observations_per_unit=round(median_observations, 8),
                sample_value_nonnull_fraction=round(nonnull_fraction, 8),
            )
        )
    return profiles


def _profile_is_longitudinally_ready(
    profile: LongitudinalArtifactProfile,
    *,
    min_distinct_times: int,
    min_repeated_unit_fraction: float,
    min_value_nonnull_fraction: float,
) -> bool:
    values = (
        profile.sample_repeated_unit_fraction,
        profile.sample_value_nonnull_fraction,
        profile.sample_median_observations_per_unit,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return bool(
        profile.row_count > 0
        and profile.sample_unit_count > 0
        and profile.sample_distinct_time_count >= min_distinct_times
        and profile.sample_repeated_unit_fraction >= min_repeated_unit_fraction
        and profile.sample_median_observations_per_unit >= 2.0
        and profile.sample_value_nonnull_fraction >= min_value_nonnull_fraction
    )


def generate_longitudinal_transportability_candidates(
    *,
    profiles: Iterable[LongitudinalArtifactProfile],
    min_ready_databases: int = 4,
    min_distinct_times: int = 3,
    min_repeated_unit_fraction: float = 0.50,
    min_value_nonnull_fraction: float = 0.50,
    limit: int = 25,
) -> List[LongitudinalTransportabilityCandidate]:
    """Emit bounded trajectory candidates from verified database profiles."""

    if min_ready_databases <= 0:
        raise ValueError("min_ready_databases must be positive")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    grouped: Dict[str, List[LongitudinalArtifactProfile]] = {}
    for profile in profiles:
        grouped.setdefault(normalize_concept_name(profile.concept), []).append(profile)

    candidates: List[LongitudinalTransportabilityCandidate] = []
    for concept, concept_profiles in grouped.items():
        # One authoritative profile per database; ambiguity is not silently
        # resolved because two artifacts claiming the same concept would need a
        # host selection contract first.
        databases = [profile.database for profile in concept_profiles]
        if len(databases) != len(set(databases)):
            continue
        ready = [
            profile
            for profile in concept_profiles
            if _profile_is_longitudinally_ready(
                profile,
                min_distinct_times=min_distinct_times,
                min_repeated_unit_fraction=min_repeated_unit_fraction,
                min_value_nonnull_fraction=min_value_nonnull_fraction,
            )
        ]
        ready.sort(key=lambda item: item.database)
        if len(ready) < min_ready_databases:
            continue
        min_repeat = min(item.sample_repeated_unit_fraction for item in ready)
        min_nonnull = min(item.sample_value_nonnull_fraction for item in ready)
        db_names = tuple(item.database for item in ready)
        note = (
            f"{concept} has verified repeated time coordinates in "
            f"{len(ready)}/{len(concept_profiles)} profiled databases "
            f"({', '.join(db_names)}); minimum sampled repeated-unit fraction "
            f"{min_repeat:.3f}, minimum sampled value coverage {min_nonnull:.3f}. "
            "Candidate question: are prespecified longitudinal trajectory "
            "features/classes reproducible and transportable across databases? "
            "Human protocol and prior-art review required; this is not a novelty "
            "or scientific-result claim."
        )
        candidates.append(
            LongitudinalTransportabilityCandidate(
                concept=concept,
                analysis_family="trajectory_clustering",
                design_archetype="cross_database_trajectory_transportability",
                ready_database_count=len(ready),
                ready_databases=db_names,
                total_databases_profiled=len(concept_profiles),
                artifact_profiles=tuple(ready),
                differentiator_note=note,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.ready_database_count,
            -min(
                profile.sample_repeated_unit_fraction
                for profile in item.artifact_profiles
            ),
            item.concept,
        )
    )
    return candidates[:limit]


__all__ = [
    "LONGITUDINAL_DISCOVERY_SCHEMA_VERSION",
    "LongitudinalArtifactProfile",
    "LongitudinalTransportabilityCandidate",
    "generate_longitudinal_transportability_candidates",
    "profile_longitudinal_table",
]
