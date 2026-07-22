"""Structural typed RETROFIT seal for an untyped EasyICU module export.

Repository-local paper infrastructure — NEVER imported by the research Agent.

An older module export (e.g. ``full6_20260717/miiv``) carries empty
``concept_meta`` and no ``column_metadata`` sidecar, so it is a *legacy* export
package: the official typed-authority path cannot seal a cohort/trajectory
authority from it. Re-running the official typed exporter is impossible (it
recomputes concepts from raw source tables the module export does not contain)
and would recompute values under the *current* dictionary, changing the data.

This module RETROFITS the existing export in place. It writes ONLY:
  * a content-addressed ``column_metadata.sha256-*.json`` sidecar, and
  * a native ``_manifest.json``
alongside the EXISTING parquet files — **whose bytes are never touched**.

It seals PROVABLE structure only — column identity, dtype, role, unit, time
coordinate, concept id, and file SHA — and DELIBERATELY OMITS
``extraction_bounds``.  The current dictionary's numeric ranges are NOT a
generation authority for values extracted under an older dictionary vintage:
re-imposing them would (a) misrepresent provenance and (b) fail-close on
legitimate vintage values (measured: full6 ``po2`` has 6.4% of values below the
current floor of 40).  Bounds authority is recorded as ``unavailable``; anomalous
values are PRESERVED for downstream host-owned QC and never filtered here.

    seal_kind = retrofitted_structural_typed_export

The seal produces a valid *typed* export package so ``materialize_to_parquet``
can seal cohort/trajectory authorities, while keeping the honest separation the
owner asked for: this tool answers "is the data's structure/identity trustworthy",
NOT "are the values clinically in range" (that is downstream scientific QC).
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    project_concept_column_metadata,
)
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.config import load_src_cfg
from easyicu.research_agent.intake.export_package import (
    IDENTIFIER_COLUMNS,
    NATIVE_MANIFEST,
    TIME_COLUMNS,
)
from easyicu.resources import load_dictionary
from easyicu.webserver.dataio import (
    _NATIVE_EXPORT_SCHEMA_V2,
    _build_export_file_metadata_binding,
    _metadata_definition_for_export,
)

SEAL_KIND = "retrofitted_structural_typed_export"
BOUNDS_AUTHORITY_UNAVAILABLE = "unavailable"

# npj_dm/20260718 packaged-dictionary identity used for the metadata PROJECTION
# (dtype/role/unit/time).  Recorded so downstream can see exactly which dictionary
# produced the sealed STRUCTURE — it does NOT make the current dict an authority
# over the older-vintage VALUES (bounds stay unavailable).
SUBMISSION_PROFILE_REF = "npj_dm/20260718"
CONCEPT_DICT_SHA256 = "fccadc53622dc82fe1dc8696617e52044168b6a84a9255e97e59df9e53bc5803"
SOFA2_DICT_SHA256 = "61f37a41083cd96df49a2e61d26c682e9d090d0a22d05ff97ba85a966b165b1c"

# Concepts each Canonical9 case in this batch (e1/m1/m2/m3/h1/h3) structurally
# requires per benchmarks/figure2_canonical9/evaluator/suite.py.  The flat cases'
# additional predictor features are LLM-selected downstream from the available
# typed pool, so they are intentionally NOT enumerated here (this map is an
# advisory annotation, not a gate).
_ALL_CASES = ("e1", "m1", "m2", "m3", "h1", "h3")
_CASE_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "age": _ALL_CASES,
    "sex": _ALL_CASES,
    "death": _ALL_CASES,
    "los_icu": ("h1",),
    "sep3_sofa2": ("e1",),
    "sofa2": ("e1", "h3"),
    "sofa2_resp": ("e1", "h3"),
    "sofa2_cardio": ("e1", "h3"),
    "sofa2_cns": ("e1", "h3"),
    "sofa2_coag": ("e1", "h3"),
    "sofa2_liver": ("e1", "m1", "h3"),
    "sofa2_renal": ("e1", "h3"),
    "lact": ("h3",),
    "bili": ("m1",),
}


class TypedRetrofitSealError(RuntimeError):
    """The structural retrofit seal could not be produced safely."""


@dataclasses.dataclass(frozen=True)
class ColumnCompat:
    """One physical column's compatibility verdict (constraint: no silent skips)."""

    file: str
    column: str
    dtype: str
    # bound | unbound | semantic_conflict
    status: str
    role: Optional[str] = None
    concept_id: Optional[str] = None
    required_by_case: Tuple[str, ...] = ()
    reason: Optional[str] = None
    # Advisory ONLY — how the CURRENT dict's bounds would score these VINTAGE
    # values.  Nothing is filtered; this is a QC signal + preserved denominator.
    current_dict_bounds_advisory: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "column": self.column,
            "dtype": self.dtype,
            "status": self.status,
            "role": self.role,
            "concept_id": self.concept_id,
            "required_by_case": list(self.required_by_case),
            "reason": self.reason,
            "current_dict_bounds_advisory": self.current_dict_bounds_advisory,
        }


@dataclasses.dataclass(frozen=True)
class SealResult:
    export_dir: str
    seal_kind: str
    value_vintage: str
    bounds_authority: str
    metadata_projection_dict: Dict[str, str]
    patient_identity: Dict[str, Any]
    sidecar_file: Optional[str]
    manifest_path: Optional[str]
    files: List[Dict[str, Any]]
    columns: List[ColumnCompat]
    parquet_immutability_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_dir": self.export_dir,
            "seal_kind": self.seal_kind,
            "value_vintage": self.value_vintage,
            "bounds_authority": self.bounds_authority,
            "metadata_projection_dict": self.metadata_projection_dict,
            "patient_identity": self.patient_identity,
            "sidecar_file": self.sidecar_file,
            "manifest_path": self.manifest_path,
            "parquet_immutability_verified": self.parquet_immutability_verified,
            "compat_summary": self.compat_summary(),
            "files": self.files,
            "columns": [c.to_dict() for c in self.columns],
        }

    def compat_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.columns:
            out[c.status] = out.get(c.status, 0) + 1
        return out


def _sha256_size(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def _strip_extraction_bounds(
    file_binding: ColumnMetadataFileBinding,
) -> ColumnMetadataFileBinding:
    """Return a copy of ``file_binding`` with every column's bounds set to None.

    The current dictionary's ``extraction_bounds`` are not an authority over the
    older-vintage values, so they are omitted from the sealed structure.
    """

    new_columns = {}
    for column, binding in file_binding.columns.items():
        stripped_meta = dataclasses.replace(binding.metadata, extraction_bounds=None)
        new_columns[column] = dataclasses.replace(binding, metadata=stripped_meta)
    return dataclasses.replace(file_binding, columns=new_columns)


def _bounds_advisory(
    concept: str, series: pd.Series, dictionary: Any, prefixes: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Score VINTAGE values against the CURRENT dict bounds — advisory only.

    Preserves the denominator (``n_total``) and the count that WOULD be excluded
    (``n_below`` / ``n_above``) so downstream host-owned QC can decide.  Never
    filters or mutates the data.
    """

    try:
        definition = _metadata_definition_for_export(concept, "x", dictionary)
        md = project_concept_column_metadata(
            definition,
            spec=ColumnProjectionSpec(
                column_name=concept,
                source_concept=concept,
                role=ConceptColumnRole.VALUE,
            ),
            source_database="miiv",
            source_database_class_prefixes=tuple(prefixes),
        )
    except Exception:
        return None
    bounds = getattr(md, "extraction_bounds", None)
    if bounds is None:
        return None
    lo = getattr(bounds, "minimum", None)
    hi = getattr(bounds, "maximum", None)
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    n_total = int(numeric.shape[0])
    n_below = int((numeric < float(lo)).sum()) if lo is not None else 0
    n_above = int((numeric > float(hi)).sum()) if hi is not None else 0
    if n_below == 0 and n_above == 0:
        return None  # conforms to current bounds; nothing to advise
    return {
        "current_dict_minimum": None if lo is None else float(lo),
        "current_dict_maximum": None if hi is None else float(hi),
        "observed_minimum": float(numeric.min()) if n_total else None,
        "observed_maximum": float(numeric.max()) if n_total else None,
        "n_total": n_total,
        "n_below_current_min": n_below,
        "n_above_current_max": n_above,
        "note": (
            "vintage values scored against CURRENT dict bounds; NOT filtered; "
            "left for downstream host-owned QC"
        ),
    }


def _concept_role_name(concept: str, series: pd.Series, dictionary: Any) -> str:
    """Provable role: EVENT_STATUS iff logical/categorical-boolean, else VALUE."""

    definition = _metadata_definition_for_export(concept, "x", dictionary)
    physical_is_bool = pd.api.types.is_bool_dtype(series)
    is_logical = definition.class_name == "lgl_cncpt"
    categorical_boolean = physical_is_bool and definition.class_name == "fct_cncpt"
    if is_logical or categorical_boolean:
        return ConceptColumnRole.EVENT_STATUS.value
    return ConceptColumnRole.VALUE.value


def _classify_columns(
    module: str,
    frame: pd.DataFrame,
    dictionary: Any,
    prefixes: Sequence[str],
    relative_path: str,
) -> Tuple[List[str], List[ColumnCompat]]:
    """Classify every physical value column; return (bindable_concepts, compat)."""

    value_cols = [
        c
        for c in frame.columns
        if c not in IDENTIFIER_COLUMNS and c not in TIME_COLUMNS
    ]
    bindable: List[str] = []
    compat: List[ColumnCompat] = []
    for column in value_cols:
        series = frame[column]
        dtype = str(series.dtype)
        required = _CASE_REQUIRED.get(column, ())
        # Is it dict-resolvable at all?
        try:
            definition = _metadata_definition_for_export(column, "x", dictionary)
        except Exception as exc:  # noqa: BLE001 - reported, not silenced
            compat.append(
                ColumnCompat(
                    file=relative_path,
                    column=column,
                    dtype=dtype,
                    status="unbound",
                    concept_id=None,
                    required_by_case=required,
                    reason=f"concept not resolvable in dictionary: {type(exc).__name__}",
                )
            )
            continue
        # Bool physical value + non-logical concept = semantic conflict (do NOT
        # silently skip — constraint 7).
        physical_is_bool = pd.api.types.is_bool_dtype(series)
        logical = definition.class_name in {"lgl_cncpt", "fct_cncpt"}
        if physical_is_bool and not logical:
            compat.append(
                ColumnCompat(
                    file=relative_path,
                    column=column,
                    dtype=dtype,
                    status="semantic_conflict",
                    concept_id=column,
                    required_by_case=required,
                    reason=(
                        f"physical boolean but dict class {definition.class_name!r} "
                        "is not logical/categorical"
                    ),
                )
            )
            continue
        bindable.append(column)
        compat.append(
            ColumnCompat(
                file=relative_path,
                column=column,
                dtype=dtype,
                status="bound",
                role=_concept_role_name(column, series, dictionary),
                concept_id=column,
                required_by_case=required,
                current_dict_bounds_advisory=_bounds_advisory(
                    column, series, dictionary, prefixes
                ),
            )
        )
    return bindable, compat


def _module_name_for(parquet_path: Path) -> str:
    """Module name from the sibling per-file manifest, else the filename stem."""

    manifest = parquet_path.with_suffix("").with_suffix(".manifest.json")
    alt = parquet_path.parent / f"{parquet_path.stem}.manifest.json"
    for candidate in (manifest, alt):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                name = str(data.get("module") or "").strip()
                if name:
                    return name
            except (json.JSONDecodeError, OSError):
                pass
    return parquet_path.stem


def seal_export_structural_typed(
    export_dir: str | Path,
    *,
    database: str = "miiv",
    value_vintage: str = "20260717",
    dictionary: Any = None,
) -> SealResult:
    """Retrofit ``export_dir`` into a native typed export IN PLACE (data untouched).

    Writes only ``_manifest.json`` + a content-addressed ``column_metadata``
    sidecar next to the existing parquets.  Verifies every parquet's SHA/size is
    identical before and after.  Raises before writing if any parquet would change.
    """

    root = Path(export_dir).expanduser()
    if not root.is_dir():
        raise TypedRetrofitSealError(f"export dir is not a directory: {root}")
    if (root / NATIVE_MANIFEST).exists():
        raise TypedRetrofitSealError(
            f"{NATIVE_MANIFEST} already present; refusing to overwrite a native "
            "manifest — remove it explicitly to re-seal"
        )
    if dictionary is None:
        dictionary = load_dictionary(include_sofa2=True)
    prefixes = tuple(
        str(v).strip().lower()
        for v in load_src_cfg(database).class_prefix
        if str(v).strip()
    )

    parquet_paths = sorted(
        p for p in root.glob("*.parquet") if p.is_file() and not p.name.startswith(".")
    )
    if not parquet_paths:
        raise TypedRetrofitSealError(f"no parquet files found under {root}")

    # Pre-write immutability baseline (constraint 2).
    pre_fingerprint = {p.name: _sha256_size(p) for p in parquet_paths}

    file_bindings: List[ColumnMetadataFileBinding] = []
    files_meta: List[Dict[str, Any]] = []
    all_compat: List[ColumnCompat] = []
    subject_id_seen = False

    for path in parquet_paths:
        module = _module_name_for(path)
        frame = pd.read_parquet(path)
        if "subject_id" in frame.columns:
            subject_id_seen = True
        bindable, compat = _classify_columns(
            module, frame, dictionary, prefixes, path.name
        )
        all_compat.extend(compat)
        if not bindable:
            # No typed columns — record it, do not seal this file.
            del frame
            gc.collect()
            continue
        binding = _build_export_file_metadata_binding(
            relative_path=path.name,
            module=module,
            frame=frame,
            concept_ids=bindable,
            database=database.strip().lower(),
            database_class_prefixes=prefixes,
            dictionary=dictionary,
        )
        binding = _strip_extraction_bounds(binding)
        file_bindings.append(binding)
        sha, size = pre_fingerprint[path.name]
        files_meta.append(
            {
                "file": path.name,
                "module": module,
                "concepts": len(bindable),
                "concept_ids": sorted(binding.columns),
                "rows": int(frame.shape[0]),
                "columns": list(frame.columns),
                "sha256": sha,
                "size_bytes": size,
                "column_metadata_columns": sorted(binding.columns),
            }
        )
        del frame
        gc.collect()

    if not file_bindings:
        raise TypedRetrofitSealError(
            "no bindable typed columns across the export; nothing to seal"
        )

    sidecar = ColumnMetadataSidecar(
        source_database=database.strip().lower(),
        source_database_class_prefixes=prefixes,
        scope=EXPORT_PHYSICAL_SCOPE,
        files=tuple(file_bindings),
    )
    metadata_ref = write_content_addressed_sidecar(root, sidecar)

    patient_identity = {
        "subject_id_present": subject_id_seen,
        "row_identity": "stay_id",
        "patient_level_uniqueness_verified": False,
        "first_icu_stay_verified": False,
        "blocker": None if subject_id_seen else "patient_identity_unavailable",
    }

    manifest = {
        "schema_version": _NATIVE_EXPORT_SCHEMA_V2,
        "database": database.strip().lower(),
        "format": "parquet",
        "seal_kind": SEAL_KIND,
        "value_vintage": value_vintage,
        "bounds_authority": BOUNDS_AUTHORITY_UNAVAILABLE,
        "metadata_projection_dict": {
            "submission_profile": SUBMISSION_PROFILE_REF,
            "concept_dict_sha256": CONCEPT_DICT_SHA256,
            "sofa2_dict_sha256": SOFA2_DICT_SHA256,
        },
        "patient_identity": patient_identity,
        "retrofit_note": (
            "Structural typed retrofit of an untyped module export. Parquet bytes "
            "unchanged. extraction_bounds OMITTED — current-dict ranges are not an "
            "authority over vintage values; anomalies preserved for downstream QC."
        ),
        "concept_selection": {
            "mode": "all_bindable_in_modules",
            "modules": {fm["module"]: fm["concept_ids"] for fm in files_meta},
        },
        "files": files_meta,
        "feature_definitions": {"included": False},
        "column_metadata": metadata_ref.to_dict(),
        "compatibility_report": [c.to_dict() for c in all_compat],
    }
    manifest_path = root / NATIVE_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Post-write immutability verification (constraint 2): parquets unchanged.
    immutable = True
    for path in parquet_paths:
        sha, size = _sha256_size(path)
        if (sha, size) != pre_fingerprint[path.name]:
            immutable = False
            raise TypedRetrofitSealError(
                f"parquet mutated during seal (must never happen): {path.name}"
            )

    return SealResult(
        export_dir=str(root),
        seal_kind=SEAL_KIND,
        value_vintage=value_vintage,
        bounds_authority=BOUNDS_AUTHORITY_UNAVAILABLE,
        metadata_projection_dict=manifest["metadata_projection_dict"],
        patient_identity=patient_identity,
        sidecar_file=metadata_ref.file,
        manifest_path=str(manifest_path),
        files=files_meta,
        columns=all_compat,
        parquet_immutability_verified=immutable,
    )


__all__ = [
    "SEAL_KIND",
    "BOUNDS_AUTHORITY_UNAVAILABLE",
    "ColumnCompat",
    "SealResult",
    "TypedRetrofitSealError",
    "seal_export_structural_typed",
]
