"""Raw EHR hashing and run-bound extraction metadata provenance.

The metadata projection at the end of this module publishes only captured
dictionary facts from a verified cohort/context, including explicit absence
for historical runs. It never reconstructs an extraction from today's catalog.

Before the research-agent pipeline runs, it normally receives a
materialised cohort (``cohort.parquet``). That cohort was itself
derived from a pipeline built on top of raw EHR sources (MIMIC /
eICU / HiRID / local extracts). Evidence-chain completeness requires
that the raw sources are hashed and registered too, so a reviewer
can ask "where did this cohort come from" and trace back through
three layers without trusting prose.

What this module provides:

* :class:`SourceFileRecord` — hashed descriptor for one raw file
  (sha256, size, mtime, relative role).
* :func:`hash_sources` — walks a list of paths (files or
  directories), skipping files over a configurable size cap so we
  don't accidentally stream TBs of MIMIC tables through sha256.
* :func:`build_provenance_bundle` — turns the hashes into a single
  JSON artefact (``provenance_sources.json``) that the pipeline can
  register as evidence.

Constraints:

* **PHI-safe.** We never copy source bytes anywhere; we compute
  sha256 in streaming 1 MiB chunks and record only the hash, byte
  size, mtime, and a user-supplied ``role`` label. Filenames are
  kept because reviewers ask about MIMIC tables by name; no per-row
  content is echoed.
* **Raw-file hashing is opt-in.** Users pass ``source_files=[...]`` to
  ``pipeline.run``. The ordinary metadata artifact reads registered context
  bytes using the existing typed metadata and evidence owners.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, model_validator

from easyicu.concept.provenance import Digest, ExportDataProvenance
from ..intake.materialized_metadata import VerifiedMaterializedCohortAuthority
from .evidence_store import EvidenceStore


# Streaming chunk. 1 MiB is a sweet spot — smaller than typical
# MIMIC table shards (10–500 MiB), larger than hashlib's internal
# block, and small enough that sha256 overhead doesn't dominate
# I/O.
_HASH_CHUNK = 1 << 20  # 1 MiB

# Default skip cap. Reviewer wants reproducibility, not a 10 TB
# hash. Raise at call time if the user really wants to hash
# everything.
_DEFAULT_MAX_BYTES = 2 * (1 << 30)  # 2 GiB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceFileRecord:
    """sha256-hashed provenance for a single raw EHR source file."""

    relative_path: str
    absolute_path: str
    size_bytes: int
    sha256: Optional[str]
    mtime_iso: str
    role: Optional[str] = None
    database: Optional[str] = None
    skipped_reason: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class ProvenanceBundle:
    """A collection of :class:`SourceFileRecord` plus aggregates."""

    records: List[SourceFileRecord] = field(default_factory=list)
    root: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        total_size = sum(r.size_bytes for r in self.records if r.size_bytes)
        hashed = sum(1 for r in self.records if r.sha256)
        skipped = [r for r in self.records if r.skipped_reason]
        databases = sorted({r.database for r in self.records if r.database})
        return {
            "n_sources": len(self.records),
            "n_hashed": hashed,
            "n_skipped": len(skipped),
            "total_size_bytes": total_size,
            "databases": databases,
        }

    def to_disk(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "easyicu.provenance_sources/1",
                    "root": self.root,
                    "summary": self.summary(),
                    "records": [r.to_json() for r in self.records],
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_entries(
    entries: Sequence[Any], *, root: Optional[Path] = None
) -> List[Tuple[Path, Optional[str], Optional[str]]]:
    """Normalise the user-supplied list into (path, role, database) triples.

    Each entry can be:

    * a path (string / ``Path``) — role and database are inferred from
      the directory name (``mimic4_icu/`` -> database=``miiv``);
    * a dict ``{"path": ..., "role": ..., "database": ...}``;
    * a tuple ``(path, role, database)``.
    """
    resolved: List[Tuple[Path, Optional[str], Optional[str]]] = []
    for entry in entries:
        role: Optional[str] = None
        database: Optional[str] = None
        if isinstance(entry, dict):
            raw_path = entry.get("path")
            role = entry.get("role")
            database = entry.get("database")
        elif isinstance(entry, tuple) and len(entry) in (2, 3):
            raw_path = entry[0]
            role = entry[1] if len(entry) >= 2 else None
            database = entry[2] if len(entry) == 3 else None
        else:
            raw_path = entry
        if raw_path is None:
            continue
        p = Path(raw_path)
        if not p.is_absolute() and root is not None:
            p = (root / p).resolve()
        resolved.append((p, role, database))
    return resolved


def _walk_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def hash_sources(
    entries: Sequence[Any],
    *,
    max_bytes_per_file: int = _DEFAULT_MAX_BYTES,
    root: Optional[Path] = None,
) -> ProvenanceBundle:
    """Hash every source file (files or directories).

    Args:
        entries: paths / dicts / tuples; see :func:`_resolve_entries`.
        max_bytes_per_file: per-file cap; files over the cap are
            recorded with ``sha256=None`` and ``skipped_reason="exceeds_cap"``.
        root: if set, relative ``entries`` resolve underneath this path.

    Returns a :class:`ProvenanceBundle` whose ``records`` list has
    one row per file (directories are expanded).
    """
    bundle = ProvenanceBundle(root=str(root) if root else None)
    for abs_path, role, database in _resolve_entries(entries, root=root):
        if not abs_path.exists():
            bundle.records.append(
                SourceFileRecord(
                    relative_path=str(abs_path),
                    absolute_path=str(abs_path),
                    size_bytes=0,
                    sha256=None,
                    mtime_iso="",
                    role=role,
                    database=database,
                    skipped_reason="not_found",
                )
            )
            continue
        for fp in _walk_files(abs_path):
            try:
                st = fp.stat()
            except OSError as exc:
                bundle.records.append(
                    SourceFileRecord(
                        relative_path=str(fp),
                        absolute_path=str(fp),
                        size_bytes=0,
                        sha256=None,
                        mtime_iso="",
                        role=role,
                        database=database,
                        skipped_reason=f"stat_error:{exc.errno}",
                    )
                )
                continue
            size = int(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            rel = fp.relative_to(abs_path.parent).as_posix() if abs_path.is_dir() else fp.name
            if size > max_bytes_per_file:
                bundle.records.append(
                    SourceFileRecord(
                        relative_path=rel,
                        absolute_path=str(fp),
                        size_bytes=size,
                        sha256=None,
                        mtime_iso=mtime,
                        role=role,
                        database=database,
                        skipped_reason=f"exceeds_cap:{max_bytes_per_file}",
                    )
                )
                continue
            try:
                digest = sha256_file(fp)
            except OSError as exc:
                bundle.records.append(
                    SourceFileRecord(
                        relative_path=rel,
                        absolute_path=str(fp),
                        size_bytes=size,
                        sha256=None,
                        mtime_iso=mtime,
                        role=role,
                        database=database,
                        skipped_reason=f"read_error:{exc.errno}",
                    )
                )
                continue
            bundle.records.append(
                SourceFileRecord(
                    relative_path=rel,
                    absolute_path=str(fp),
                    size_bytes=size,
                    sha256=digest,
                    mtime_iso=mtime,
                    role=role,
                    database=database,
                )
            )
    return bundle


def build_provenance_bundle(
    *,
    cohort_path: Path,
    source_files: Optional[Sequence[Any]] = None,
    root: Optional[Path] = None,
    max_bytes_per_file: int = _DEFAULT_MAX_BYTES,
) -> ProvenanceBundle:
    """Top-level driver.

    Always hashes the cohort parquet; adds raw EHR source files if the
    user provides them. The cohort hash matches the one already written
    into the evidence store, but recording it here gives reviewers a
    single JSON to quote in the supplement.
    """
    entries: List[Any] = []
    if source_files:
        entries.extend(source_files)
    entries.append({"path": str(cohort_path), "role": "cohort", "database": None})
    return hash_sources(
        entries, max_bytes_per_file=max_bytes_per_file, root=root,
    )


__all__ = [
    "ProvenanceBundle",
    "SourceFileRecord",
    "build_provenance_bundle",
    "hash_sources",
    "sha256_file",
]


class RunVariableProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    column: str
    concept_id: str | None


class RunPredicateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    anchor: str
    start_offset_hours: float | str
    end_offset_hours: float | str


class RunCohortPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    concept_id: str
    time_window: RunPredicateWindow
    aggregation: str
    op: str
    value: str | int | float | bool | tuple[str | int | float | bool, ...] | None = None


class RunCohortSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    inclusion: tuple[RunCohortPredicate, ...] = ()
    exclusion: tuple[RunCohortPredicate, ...] = ()
    selection_mode: str = "predicate_filtered"


class DataProvenanceEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: Literal["data_provenance"] = "data_provenance"
    sha256: Digest


class RunDataProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["easyicu.run-data-provenance/1"] = "easyicu.run-data-provenance/1"
    run_id: str
    status: Literal["recorded", "not_recorded"]
    research_context_sha256: Digest | None = None
    export_authority_sha256: Digest | None = None
    materialization_sha256: Digest | None = None
    cohort_file_sha256: Digest | None = None
    data_provenance: ExportDataProvenance | None = None
    variables: tuple[RunVariableProvenance, ...] = ()
    cohort_selection: RunCohortSelection | None = None
    evidence_ref: DataProvenanceEvidenceRef | None = None

    @model_validator(mode="after")
    def check_binding(self):
        if self.status == "recorded" and (
            self.data_provenance is None or not self.data_provenance.snapshots
            or self.research_context_sha256 is None
            or (self.export_authority_sha256 is None and (
                self.materialization_sha256 is None or self.cohort_file_sha256 is None
            ))
        ):
            raise ValueError("recorded provenance requires frozen input bindings")
        if self.status == "not_recorded" and self.data_provenance is not None:
            raise ValueError("unrecorded provenance cannot contain invented inputs")
        return self


class VerifiedMaterializationSummary(BaseModel):
    """Facts copied from a verified materialization authority, not inferred lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    source_mode: Literal["export", "converted"]
    cohort_window_hours: tuple[float, float]
    output_columns: tuple[str, ...]
    producer_implementation_sha256: Digest

    @model_validator(mode="after")
    def check_bounds(self):
        if (self.cohort_window_hours[0] > self.cohort_window_hours[1]
                or not 1 <= len(self.output_columns) <= 256
                or any(not isinstance(name, str) or not 1 <= len(name) <= 128
                       for name in self.output_columns)
                or len(set(self.output_columns)) != len(self.output_columns)):
            raise ValueError("materialization summary is outside the bounded metadata contract")
        return self


class RunDataProvenanceV2(RunDataProvenance):
    schema_version: Literal["easyicu.run-data-provenance/2"] = "easyicu.run-data-provenance/2"
    materialization: VerifiedMaterializationSummary

    @model_validator(mode="after")
    def check_materialization_binding(self):
        if self.materialization_sha256 is None or self.cohort_file_sha256 is None:
            raise ValueError("materialization facts require verified cohort and receipt bindings")
        return self


def provenance_from_verified_cohort(
    verified: VerifiedMaterializedCohortAuthority | None,
    legacy_provenance: dict | None = None,
) -> dict:
    """Read only the already verified semantic receipt, never the live catalog."""
    if verified is None and legacy_provenance is None:
        return {}
    semantic = verified.authority.to_dict()["semantic_provenance"] if verified else legacy_provenance
    export = semantic.get("export_authority") or {}
    captured = export.get("data_provenance") or semantic.get("data_provenance")
    snapshot = ExportDataProvenance.model_validate(captured) if captured is not None else None
    if (snapshot is None or not snapshot.snapshots) and verified is None:
        return {}
    result = {
        **({"data_provenance": snapshot.model_dump(mode="json")} if snapshot and snapshot.snapshots else {}),
        "data_provenance_export_authority_sha256": export.get("authority_sha256"),
        "data_provenance_materialization_sha256": (
            verified.reference.sha256 if verified else legacy_provenance["provenance_sha256"]
        ),
        "data_provenance_cohort_file_sha256": semantic["cohort_file_sha256"],
        "data_provenance_cohort_selection": (
            RunCohortSelection.model_validate({
                key: value for key, value in semantic["cohort_definition"].items()
                if key in RunCohortSelection.model_fields
            }).model_dump(mode="json")
            if semantic.get("cohort_definition") else None
        ),
    }
    if verified is not None:
        authority = verified.authority
        # The frozen semantic receipt must describe exactly the sealed output;
        # never use a present-day dictionary or infer a mapping from names.
        if tuple(semantic.get("columns") or ()) == tuple(authority.cohort_columns):
            try:
                summary = VerifiedMaterializationSummary.model_validate({
                    "source_mode": semantic.get("source_mode"),
                    "cohort_window_hours": semantic.get("cohort_window_hours"),
                    "output_columns": semantic.get("columns"),
                    "producer_implementation_sha256": authority.producer_implementation_sha256,
                })
            except ValueError:
                pass
            else:
                result["data_provenance_materialization"] = summary.model_dump(mode="json")
    return result


def publish_run_data_provenance(*, run_dir: Path, evidence: EvidenceStore):
    """Register one immutable metadata artifact, including explicit legacy absence."""
    context_record = evidence.get("research_context")
    if context_record is None:
        raise ValueError("data provenance requires registered research context")
    from .runtime_artifacts import verified_run_evidence_path

    context_path = verified_run_evidence_path(run_dir, context_record)
    if context_path is None:
        raise ValueError("registered research context digest mismatch")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    provenance = context.get("cohort", {}).get("provenance") or {}
    captured = provenance.get("data_provenance")
    materialization = provenance.get("data_provenance_materialization")
    if materialization is not None:
        from ..intake.materialized_metadata import (
            MaterializedMetadataError, load_verified_materialized_cohort_authority,
        )

        # Context is merely an evidence-bound claim. Re-establish authority
        # from the run-owned staged cohort, never a context-supplied path.
        try:
            cohort = run_dir / "cohort.parquet"
            verified = None if cohort.is_symlink() else load_verified_materialized_cohort_authority(cohort)
            actual = provenance_from_verified_cohort(verified) if verified is not None else {}
            bound = (
                verified is not None
                and context.get("cohort", {}).get("database") == verified.sidecar.source_database
                and all(provenance.get(key) == actual.get(key) for key in (
                    "data_provenance_materialization", "data_provenance_materialization_sha256",
                    "data_provenance_cohort_file_sha256", "data_provenance_export_authority_sha256",
                    "data_provenance",
                ))
                and actual.get("data_provenance_materialization") is not None
            )
        except (OSError, ValueError, MaterializedMetadataError):
            bound = False
        if not bound:
            materialization = None
            captured = None
            provenance = {}
    if captured is not None:
        captured = ExportDataProvenance.model_validate(captured)
        if not captured.snapshots:
            captured = None
    payload_type = RunDataProvenanceV2 if materialization is not None else RunDataProvenance
    payload = payload_type(
        run_id=run_dir.name, status="recorded" if captured else "not_recorded",
        research_context_sha256=context_record.sha256,
        export_authority_sha256=provenance.get("data_provenance_export_authority_sha256"),
        materialization_sha256=provenance.get("data_provenance_materialization_sha256"),
        cohort_file_sha256=provenance.get("data_provenance_cohort_file_sha256"),
        data_provenance=captured,
        cohort_selection=provenance.get("data_provenance_cohort_selection"),
        variables=tuple(RunVariableProvenance(
            column=variable["name"], concept_id=variable.get("source_concept"),
        ) for variable in context.get("variables", [])),
        **({"materialization": materialization} if materialization is not None else {}),
    )
    raw = payload.model_dump_json().encode("utf-8")
    existing = evidence.get("data_provenance")
    if existing is not None:
        if existing.sha256 != hashlib.sha256(raw).hexdigest():
            raise ValueError("run data provenance changed from its registered digest")
        if verified_run_evidence_path(run_dir, existing) is None:
            raise ValueError("registered data provenance digest mismatch")
        return existing
    path = run_dir / "data_provenance.json"
    path.write_bytes(raw)
    return evidence.register_file(
        kind="log", description="Frozen metadata-only extraction provenance for this run.",
        source_path=path, evidence_id="data_provenance", inputs=["research_context"],
        producer="pipeline", generation_mode="system",
    )


def register_run_metadata(
    *, run_dir, evidence, write_fingerprint, assert_matches,
    expected_concept_dict_sha, expected_sofa2_dict_sha,
):
    """Keep dictionary drift validation and run metadata registration together."""
    path = run_dir / "concept_dict_fingerprint.json"
    fingerprint = write_fingerprint(path)
    assert_matches(
        fingerprint, expected_concept_dict_sha=expected_concept_dict_sha,
        expected_sofa2_dict_sha=expected_sofa2_dict_sha, mode="strict",
    )
    if evidence.get("concept_dict_fingerprint") is None:
        evidence.register_file(
            kind="log", description="SHA-256 fingerprint for EasyICU concept dictionaries.",
            source_path=path, evidence_id="concept_dict_fingerprint",
            producer="pipeline", generation_mode="system",
        )
    publish_run_data_provenance(run_dir=run_dir, evidence=evidence)
