"""Structural typed RETROFIT seal for an untyped EasyICU module export.

Repository-local paper infrastructure — NEVER imported by the research Agent.

An older module export (e.g. ``full6_20260717/miiv``) carries empty
``concept_meta`` and no ``column_metadata`` sidecar, so it is a *legacy* export
package: the official typed-authority path cannot seal a cohort/trajectory
authority from it, and re-running the official typed exporter is impossible (it
recomputes concepts from raw source tables the module export does not contain).

This module RETROFITS the existing export by writing ONLY a native
``_manifest.json`` + a content-addressed ``column_metadata`` sidecar alongside
the EXISTING parquet files — **whose bytes are never touched** (verified by
per-file SHA/size before and after).

HONEST PROVENANCE BOUNDARY (the whole point):

* It seals structure only — column identity, dtype, role, unit, and time
  coordinate — and these are a PROJECTION of the *current* packaged dictionary,
  NOT facts recorded at the older extraction. Every sealed structural field is
  labelled ``current_dictionary_projection_not_extraction_provenance`` and the
  six-case required concepts get a per-field semantic review table that keeps
  ``paper_authorized = false`` until a human signs it off.
* It OMITS every numeric value-range claim (``extraction_bounds`` AND
  ``analysis_plausibility_range``): the current dictionary's ranges are not an
  authority over older-vintage values and would fail-close on legitimate vintage
  data (measured: full6 ``po2`` 6.4% below the current floor). ``bounds_authority``
  is recorded as ``unavailable``; anomalies are PRESERVED for downstream
  host-owned QC (denominator + would-exclude counts kept), never filtered.

    seal_kind = retrofitted_structural_typed_export

The seal produces a valid *typed* export package so ``materialize_to_parquet``
can seal cohort/trajectory authorities, while answering only "is the data's
structure/identity trustworthy" — NOT "are the values clinically in range"
(downstream scientific QC).
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from easyicu.concept.export_column_binding import (
    build_export_file_metadata_binding,
    metadata_definition_for_export,
)
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    SidecarRef,
    canonical_sidecar_bytes,
    write_content_addressed_sidecar,
)
from easyicu.config import load_src_cfg
from easyicu.research_agent.concept_dict_audit import (
    assert_dict_matches,
    compute_concept_dict_fingerprint,
)
from easyicu.research_agent.intake.export_package import (
    IDENTIFIER_COLUMNS,
    LEGACY_MANIFEST,
    NATIVE_MANIFEST,
    NATIVE_MANIFEST_SCHEMA_V2,
    TIME_COLUMNS,
)
from easyicu.research_agent.orchestration.profiles import NPJ_DM_2026_07_18
from easyicu.resources import load_dictionary

SEAL_KIND = "retrofitted_structural_typed_export"
BOUNDS_AUTHORITY_UNAVAILABLE = "unavailable"
METADATA_PROVENANCE = "current_dictionary_projection_not_extraction_provenance"

_ALL_CASES = ("e1", "m1", "m2", "m3", "h1", "h3")
# Concepts each Canonical9 case in this batch structurally requires per
# benchmarks/figure2_canonical9/evaluator/suite.py.  Flat cases' additional
# predictor features are LLM-selected downstream from the typed pool, so they are
# intentionally not enumerated (this is an advisory annotation, not a gate).
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
    """One physical column's compatibility verdict (no silent skips)."""

    file: str
    column: str
    dtype: str
    status: str  # bound | unbound | semantic_conflict
    role: Optional[str] = None
    canonical_unit: Optional[str] = None
    concept_id: Optional[str] = None
    required_by_case: Tuple[str, ...] = ()
    reason: Optional[str] = None
    # Advisory ONLY — how the CURRENT dict bounds would score these VINTAGE
    # values.  Nothing is filtered; this is a QC signal + preserved denominator.
    current_dict_bounds_advisory: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self) | {
            "required_by_case": list(self.required_by_case)
        }


@dataclasses.dataclass(frozen=True)
class SealResult:
    export_dir: str
    seal_kind: str
    dry_run: bool
    value_vintage: str
    value_vintage_basis: str
    bounds_authority: str
    metadata_provenance: str
    dict_fingerprint: Dict[str, str]
    source_evidence: Dict[str, Any]
    patient_identity: Dict[str, Any]
    semantic_review: Dict[str, Any]
    paper_ready: bool
    sidecar_ref: Dict[str, Any]
    sidecar_file: Optional[str]
    manifest_path: Optional[str]
    files: List[Dict[str, Any]]
    columns: List[ColumnCompat]
    parquet_immutability_verified: bool

    def compat_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.columns:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_dir": self.export_dir,
            "seal_kind": self.seal_kind,
            "dry_run": self.dry_run,
            "value_vintage": self.value_vintage,
            "value_vintage_basis": self.value_vintage_basis,
            "bounds_authority": self.bounds_authority,
            "metadata_provenance": self.metadata_provenance,
            "dict_fingerprint": self.dict_fingerprint,
            "source_evidence": self.source_evidence,
            "patient_identity": self.patient_identity,
            "semantic_review": self.semantic_review,
            "paper_ready": self.paper_ready,
            "sidecar_ref": self.sidecar_ref,
            "sidecar_file": self.sidecar_file,
            "manifest_path": self.manifest_path,
            "parquet_immutability_verified": self.parquet_immutability_verified,
            "compat_summary": self.compat_summary(),
            "files": self.files,
            "columns": [c.to_dict() for c in self.columns],
        }


def _sha256_size(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def _sha256_file(path: Path) -> str:
    return _sha256_size(path)[0]


def _canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic bytes for digesting a manifest sub-block."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sealer_source_files() -> List[Path]:
    """The exact source files whose bytes constitute the sealing implementation."""

    import easyicu.concept.export_column_binding as leaf

    return sorted({Path(__file__).resolve(), Path(leaf.__file__).resolve()})


def _sealer_code_sha256() -> str:
    """SHA of the sealing implementation (this module + the shared binding leaf)."""

    digest = hashlib.sha256()
    for source in _sealer_source_files():
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _git_head(repo_hint: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_hint), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_paths_dirty(repo_hint: Path, paths: Sequence[Path]) -> Optional[bool]:
    """Whether the given paths differ from HEAD (staged, unstaged, or untracked).

    Returns ``None`` when git cannot answer (not a repo, git absent). A dirty
    result means HEAD does NOT faithfully describe those files' running bytes.
    """

    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(repo_hint),
                "status",
                "--porcelain",
                "--",
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _sealer_git_provenance() -> Dict[str, Any]:
    """Honest HEAD/dirty provenance for the running sealer code.

    ``sealer_code_sha256`` is the byte-exact authority; the git commit is only a
    faithful cross-reference when the sealer's OWN source files are clean at
    HEAD. A dirty (or untracked) sealer file means the running bytes are NOT the
    committed ones, so ``head_describes_running_code`` is False and the bare
    commit must never be read as the provenance of this seal.
    """

    sources = _sealer_source_files()
    repo_hint = sources[0].parent
    head = _git_head(repo_hint)
    dirty = _git_paths_dirty(repo_hint, sources)
    return {
        "head": head,
        "sealer_paths_dirty": dirty,
        "head_describes_running_code": bool(head) and dirty is False,
    }


def _strip_value_range_claims(
    file_binding: ColumnMetadataFileBinding,
) -> ColumnMetadataFileBinding:
    """Copy ``file_binding`` with every column's numeric value-range claims removed.

    The current dictionary's ``extraction_bounds`` and
    ``analysis_plausibility_range`` are not an authority over older-vintage
    values, so they are omitted from the sealed structure.
    """

    new_columns = {}
    for column, binding in file_binding.columns.items():
        stripped_meta = dataclasses.replace(
            binding.metadata,
            extraction_bounds=None,
            analysis_plausibility_range=None,
        )
        new_columns[column] = dataclasses.replace(binding, metadata=stripped_meta)
    return dataclasses.replace(file_binding, columns=new_columns)


def _bounds_advisory(
    concept: str, series: pd.Series, dictionary: Any, prefixes: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Score VINTAGE values against CURRENT dict bounds — advisory only.

    Preserves the denominator and the would-exclude counts so downstream
    host-owned QC can decide.  Never filters or mutates data.
    """

    from easyicu.concept.metadata_projection import (
        ColumnProjectionSpec,
        ConceptColumnRole,
        project_concept_column_metadata,
    )

    try:
        definition = metadata_definition_for_export(concept, "x", dictionary)
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
        return None
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


def _definition_unit(definition: Any) -> Optional[str]:
    unit = getattr(definition, "unit", None)
    if isinstance(unit, (list, tuple)) and unit:
        return str(unit[0])
    if isinstance(unit, str) and unit:
        return unit
    return None


def _classify_columns(
    frame: pd.DataFrame,
    dictionary: Any,
    prefixes: Sequence[str],
    relative_path: str,
) -> Tuple[List[str], List[ColumnCompat]]:
    """Classify every physical value column; return (bindable_concepts, compat)."""

    from easyicu.concept.metadata_projection import ConceptColumnRole

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
        try:
            definition = metadata_definition_for_export(column, "x", dictionary)
        except Exception as exc:  # noqa: BLE001 - reported, not silenced
            compat.append(
                ColumnCompat(
                    file=relative_path,
                    column=column,
                    dtype=dtype,
                    status="unbound",
                    required_by_case=required,
                    reason=(
                        f"concept not resolvable in dictionary: {type(exc).__name__}"
                    ),
                )
            )
            continue
        physical_is_bool = pd.api.types.is_bool_dtype(series)
        # Match the official binding's scalar ``==`` semantics: a list/multi-class
        # ``class_name`` is treated as non-logical (tuple membership, not a set,
        # so an unhashable list never raises).
        logical = definition.class_name in ("lgl_cncpt", "fct_cncpt")
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
        role = (
            ConceptColumnRole.EVENT_STATUS.value
            if logical
            else ConceptColumnRole.VALUE.value
        )
        compat.append(
            ColumnCompat(
                file=relative_path,
                column=column,
                dtype=dtype,
                status="bound",
                role=role,
                canonical_unit=_definition_unit(definition),
                concept_id=column,
                required_by_case=required,
                current_dict_bounds_advisory=_bounds_advisory(
                    column, series, dictionary, prefixes
                ),
            )
        )
    return bindable, compat


def _subject_identity(
    parquet_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Verify cross-file stay_id -> subject_id consistency; never fabricate.

    Presence of ``subject_id`` in one file does not clear the blocker.  Every
    file carrying both columns must agree; any conflict fails closed.
    """

    import pyarrow.parquet as pq

    stay_to_subject: Dict[Any, Any] = {}
    files_with_subject: List[str] = []
    conflicts: List[Dict[str, Any]] = []
    for path in parquet_paths:
        cols = set(pq.read_schema(path).names)  # metadata only, no row read
        if "subject_id" not in cols or "stay_id" not in cols:
            continue
        files_with_subject.append(path.name)
        pair = pd.read_parquet(path, columns=["stay_id", "subject_id"]).dropna()
        for stay, subject in zip(pair["stay_id"].tolist(), pair["subject_id"].tolist()):
            prior = stay_to_subject.get(stay)
            if prior is None:
                stay_to_subject[stay] = subject
            elif prior != subject:
                conflicts.append(
                    {"stay_id": stay, "subject_ids": sorted({str(prior), str(subject)})}
                )
    if not files_with_subject:
        return {
            "subject_id_present": False,
            "row_identity": "stay_id",
            "patient_level_uniqueness_verified": False,
            "first_icu_stay_verified": False,
            "blocker": "patient_identity_unavailable",
        }
    if conflicts:
        raise TypedRetrofitSealError(
            "subject_id cross-file conflict "
            f"({len(conflicts)} stay_id(s) map to multiple subject_id): "
            f"{conflicts[:3]}"
        )
    return {
        "subject_id_present": True,
        "row_identity": "stay_id",
        "files_with_subject_id": sorted(files_with_subject),
        "stay_to_subject_consistent": True,
        "n_stays_with_subject": len(stay_to_subject),
        "patient_level_uniqueness_verified": True,
        "first_icu_stay_verified": False,
        "blocker": None,
    }


def _bind_value_vintage(export_dir: Path, value_vintage: str) -> Tuple[str, str]:
    """Anchor ``value_vintage`` to an evidence token; fail closed on a forged tag.

    If the export path embeds a YYYYMMDD token, ``value_vintage`` MUST equal it.
    Otherwise the vintage is operator-asserted and recorded as unverified.
    """

    tokens = re.findall(r"(?<!\d)(\d{8})(?!\d)", str(export_dir))
    if tokens:
        token = tokens[-1]
        if value_vintage != token:
            raise TypedRetrofitSealError(
                f"value_vintage {value_vintage!r} does not match export-path date "
                f"token {token!r} (forged vintage)"
            )
        return value_vintage, f"export_path_date_token:{token}"
    return value_vintage, "operator_asserted_no_path_token"


def _source_evidence(root: Path, parquet_paths: Sequence[Path]) -> Dict[str, Any]:
    legacy = root / LEGACY_MANIFEST
    per_module = {
        p.name: _sha256_file(p)
        for p in sorted(root.glob("*.manifest.json"))
        if p.is_file()
    }
    return {
        # The legacy export manifest IS the outer extraction-run manifest.
        "extraction_run_manifest": LEGACY_MANIFEST,
        "extraction_run_manifest_sha256": (
            _sha256_file(legacy) if legacy.exists() else None
        ),
        "per_module_manifest_sha256": per_module,
        "sealer_code_sha256": _sealer_code_sha256(),
        # HEAD/dirty provenance of the SEALER's code repo (not the export data
        # dir, which lives on an external drive and is not a git repo). The bare
        # HEAD is only a faithful pointer when the sealer's own files are clean —
        # see ``head_describes_running_code``. ``sealer_code_sha256`` above is the
        # byte-exact authority regardless.
        "sealer_git": _sealer_git_provenance(),
    }


def _semantic_review(
    file_bindings: Sequence[ColumnMetadataFileBinding],
) -> Dict[str, Any]:
    """Per-field review table for the six-case required concepts (paper-gated)."""

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for fb in file_bindings:
        coord = fb.time_coordinates[0] if fb.time_coordinates else None
        for column, binding in fb.columns.items():
            if column not in _CASE_REQUIRED or column in seen:
                continue
            seen.add(column)
            md = binding.metadata
            rows.append(
                {
                    "concept": column,
                    "file": fb.relative_path,
                    "required_by_case": list(_CASE_REQUIRED[column]),
                    "sealed_role": md.role.value,
                    "sealed_canonical_unit": md.canonical_unit,
                    "sealed_time_origin": getattr(coord, "origin", None),
                    "sealed_time_unit": getattr(coord, "unit", None),
                    "provenance": METADATA_PROVENANCE,
                    "reviewed": False,
                }
            )
    missing = sorted(set(_CASE_REQUIRED) - seen)
    return {
        "paper_authorized": False,
        "reviewed": False,
        "note": (
            "role/unit/time are a CURRENT-dictionary projection, not facts recorded "
            "at the older extraction. A human must sign off each field before any "
            "sealed input is paper-authorized."
        ),
        "unbound_required_concepts": missing,
        "review_table": rows,
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish ``path`` atomically (temp + fsync + rename); the last commit point."""

    raw = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _module_name_for(parquet_path: Path) -> str:
    alt = parquet_path.parent / f"{parquet_path.stem}.manifest.json"
    if alt.exists():
        try:
            data = json.loads(alt.read_text(encoding="utf-8"))
            name = str(data.get("module") or "").strip()
            if name:
                return name
        except (json.JSONDecodeError, OSError):
            pass
    return parquet_path.stem


def _patient_identity_sufficient(identity: Dict[str, Any]) -> bool:
    """A retrofit export has sufficient identity only if subject_id is proven and
    no blocker remains. Stay-level-only identity is NOT sufficient for paper use."""

    return bool(identity.get("subject_id_present")) and not identity.get("blocker")


def _paper_ready(
    semantic_review: Dict[str, Any], patient_identity: Dict[str, Any]
) -> bool:
    """The single producer/consumer predicate: a retrofit seal is paper-ready only
    when a human has authorized the projected structure AND identity is sufficient."""

    return semantic_review.get("paper_authorized") is True and (
        _patient_identity_sufficient(patient_identity)
    )


def assert_sealed_export_paper_ready(export_dir: str | Path) -> Dict[str, Any]:
    """Consumer half of the retrofit producer→consumer gate — fail-closed.

    A paper-facing run over a RETROFIT-sealed export MUST route through this gate
    before using the sealed metadata as authoritative. It reads the native
    ``_manifest.json`` the seal wrote and, for a retrofit seal, raises unless the
    seal is paper-ready by the SAME predicate the producer records:

    * the per-field semantic review has been human-signed
      (``semantic_review.paper_authorized is True``), AND
    * patient identity is sufficient (``subject_id`` proven, no blocker).

    A non-retrofit native export (no ``seal_kind``) is governed by the official
    typed-authority path, so this gate returns its manifest unchanged. Returns
    the verified manifest on success.
    """

    root = Path(export_dir).expanduser()
    manifest_path = root / NATIVE_MANIFEST
    if not manifest_path.is_file():
        raise TypedRetrofitSealError(
            f"no {NATIVE_MANIFEST} at {root}: export is not sealed; refusing "
            "paper-facing use"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TypedRetrofitSealError(
            f"unreadable native manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise TypedRetrofitSealError(
            f"native manifest is not an object: {manifest_path}"
        )

    if manifest.get("seal_kind") != SEAL_KIND:
        # Not a retrofit seal — the official typed-authority path governs it.
        return manifest

    review = manifest.get("semantic_review")
    identity = manifest.get("patient_identity")
    if not isinstance(review, dict) or not isinstance(identity, dict):
        raise TypedRetrofitSealError(
            "retrofit seal is missing its semantic_review / patient_identity "
            "provenance; refusing paper-facing use"
        )
    if manifest.get("paper_ready") is not _paper_ready(review, identity):
        raise TypedRetrofitSealError(
            "retrofit manifest paper_ready flag disagrees with its own "
            "semantic_review/patient_identity; refusing paper-facing use"
        )
    if not _paper_ready(review, identity):
        reasons: List[str] = []
        if review.get("paper_authorized") is not True:
            reasons.append(
                f"semantic review unsigned (paper_authorized="
                f"{review.get('paper_authorized')!r})"
            )
        if not _patient_identity_sufficient(identity):
            reasons.append(
                f"patient identity insufficient (blocker={identity.get('blocker')!r}, "
                f"subject_id_present={identity.get('subject_id_present')!r})"
            )
        raise TypedRetrofitSealError(
            "retrofit-sealed export is NOT paper-authorized: "
            + "; ".join(reasons)
            + " — a human must review before any paper-facing run"
        )
    return manifest


RETROFIT_REVIEW_ATTESTATION_SCHEMA = "easyicu.retrofit_review_attestation/1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def build_retrofit_review_attestation(
    export_dir: str | Path,
    *,
    reviewer: str,
    reviewed_at: str,
) -> Dict[str, Any]:
    """Mint a frozen paper-readiness attestation for a retrofit-sealed export.

    This is the ONLY sanctioned producer of an attestation and it FAILS CLOSED
    through :func:`assert_sealed_export_paper_ready`: an export that is unreviewed
    or whose patient identity is insufficient (e.g. full6 ->
    ``patient_identity_unavailable``) cannot yield an attestation, so it can never
    become a paper-ready authority. Call this BEFORE freezing a task binding or
    materialising a paper authority from the sealed export. The attestation binds
    the source manifest + sidecar + review + identity digests so a downstream
    verifier can re-derive and detect tamper/drift.
    """

    root = Path(export_dir).expanduser()
    manifest = assert_sealed_export_paper_ready(root)  # fail-close before minting
    manifest_path = root / NATIVE_MANIFEST
    sidecar_ref = manifest.get("column_metadata") or {}
    sidecar_file = str(sidecar_ref.get("file") or "")
    sidecar_path = root / sidecar_file
    if not sidecar_file or Path(sidecar_file).name != sidecar_file:
        raise TypedRetrofitSealError(
            "retrofit manifest references no single-component sidecar; cannot attest"
        )
    if not sidecar_path.is_file():
        raise TypedRetrofitSealError(
            f"retrofit manifest sidecar is missing: {sidecar_file}; cannot attest"
        )
    reviewer = reviewer.strip()
    reviewed_at = reviewed_at.strip()
    if not reviewer or not reviewed_at:
        raise TypedRetrofitSealError(
            "attestation requires a non-empty reviewer and reviewed_at"
        )
    return {
        "schema_version": RETROFIT_REVIEW_ATTESTATION_SCHEMA,
        "seal_kind": SEAL_KIND,
        "value_vintage": str(manifest.get("value_vintage") or ""),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_sidecar_file": sidecar_file,
        "source_sidecar_sha256": _sha256_file(sidecar_path),
        "semantic_review_sha256": hashlib.sha256(
            _canonical_json_bytes(manifest.get("semantic_review"))
        ).hexdigest(),
        "patient_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(manifest.get("patient_identity"))
        ).hexdigest(),
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "paper_ready": True,
    }


def verify_retrofit_review_attestation(
    attestation: Mapping[str, Any],
    *,
    export_dir: str | Path | None = None,
) -> None:
    """Re-verify a bound attestation; fail-closed (raises ``TypedRetrofitSealError``).

    Offline structural + semantic checks always run (schema, seal kind,
    ``paper_ready``, digest shape, reviewer). When ``export_dir`` is provided the
    live artifacts are re-derived and the paper-readiness gate is re-run, so a
    tampered or drifted manifest/sidecar digest fails closed.
    """

    if not isinstance(attestation, Mapping):
        raise TypedRetrofitSealError("attestation must be a mapping")
    if attestation.get("schema_version") != RETROFIT_REVIEW_ATTESTATION_SCHEMA:
        raise TypedRetrofitSealError("unknown retrofit attestation schema")
    if attestation.get("seal_kind") != SEAL_KIND:
        raise TypedRetrofitSealError("attestation seal_kind is not a retrofit seal")
    if attestation.get("paper_ready") is not True:
        raise TypedRetrofitSealError("attestation is not paper_ready; fail-closed")
    for key in (
        "source_manifest_sha256",
        "source_sidecar_sha256",
        "semantic_review_sha256",
        "patient_identity_sha256",
    ):
        value = attestation.get(key)
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise TypedRetrofitSealError(f"attestation {key} is not a sha256 digest")
    if not str(attestation.get("reviewer") or "").strip():
        raise TypedRetrofitSealError("attestation has no reviewer")
    if export_dir is None:
        return

    # Live re-derivation: re-run the gate and re-compute every bound digest.
    root = Path(export_dir).expanduser()
    manifest = assert_sealed_export_paper_ready(root)  # fail-close on drift
    if _sha256_file(root / NATIVE_MANIFEST) != attestation["source_manifest_sha256"]:
        raise TypedRetrofitSealError(
            "attestation source_manifest_sha256 mismatch (tampered or drifted)"
        )
    sidecar_file = str((manifest.get("column_metadata") or {}).get("file") or "")
    if sidecar_file != attestation.get("source_sidecar_file"):
        raise TypedRetrofitSealError("attestation source_sidecar_file mismatch")
    if _sha256_file(root / sidecar_file) != attestation["source_sidecar_sha256"]:
        raise TypedRetrofitSealError(
            "attestation source_sidecar_sha256 mismatch (tampered or drifted)"
        )
    if (
        hashlib.sha256(
            _canonical_json_bytes(manifest.get("semantic_review"))
        ).hexdigest()
        != attestation["semantic_review_sha256"]
    ):
        raise TypedRetrofitSealError("attestation semantic_review_sha256 mismatch")
    if (
        hashlib.sha256(
            _canonical_json_bytes(manifest.get("patient_identity"))
        ).hexdigest()
        != attestation["patient_identity_sha256"]
    ):
        raise TypedRetrofitSealError("attestation patient_identity_sha256 mismatch")


def seal_export_structural_typed(
    export_dir: str | Path,
    *,
    database: str = "miiv",
    value_vintage: str = "20260717",
    dictionary: Any = None,
    dry_run: bool = False,
    submission_profile: Any = NPJ_DM_2026_07_18,
) -> SealResult:
    """Retrofit ``export_dir`` into a native typed export (data untouched).

    ``dry_run=True`` performs the FULL scan + compatibility report over the real
    export and computes the would-be sidecar/manifest, but writes NOTHING.
    Otherwise it writes only the sidecar + ``_manifest.json`` (both atomic), and
    verifies every parquet SHA/size is identical before and after.
    """

    root = Path(export_dir).expanduser()
    if not root.is_dir():
        raise TypedRetrofitSealError(f"export dir is not a directory: {root}")
    if (root / NATIVE_MANIFEST).exists():
        raise TypedRetrofitSealError(
            f"{NATIVE_MANIFEST} already present; refusing to overwrite a native "
            "manifest — remove it explicitly to re-seal"
        )

    # (#1) Runtime dictionary SHA must match the declared profile, or fail closed.
    fingerprint = compute_concept_dict_fingerprint()
    assert_dict_matches(
        fingerprint,
        expected_concept_dict_sha=submission_profile.expected_concept_dict_sha,
        expected_sofa2_dict_sha=submission_profile.expected_sofa2_dict_sha,
        mode="strict",
    )
    if dictionary is None:
        dictionary = load_dictionary(include_sofa2=True)
    prefixes = tuple(
        str(v).strip().lower()
        for v in load_src_cfg(database).class_prefix
        if str(v).strip()
    )

    bound_vintage, vintage_basis = _bind_value_vintage(root, value_vintage)

    parquet_paths = sorted(
        p for p in root.glob("*.parquet") if p.is_file() and not p.name.startswith(".")
    )
    if not parquet_paths:
        raise TypedRetrofitSealError(f"no parquet files found under {root}")

    pre_fingerprint = {p.name: _sha256_size(p) for p in parquet_paths}
    patient_identity = _subject_identity(parquet_paths)

    file_bindings: List[ColumnMetadataFileBinding] = []
    files_meta: List[Dict[str, Any]] = []
    all_compat: List[ColumnCompat] = []

    for path in parquet_paths:
        module = _module_name_for(path)
        frame = pd.read_parquet(path)
        bindable, compat = _classify_columns(frame, dictionary, prefixes, path.name)
        binding: Optional[ColumnMetadataFileBinding] = None
        if bindable:
            try:
                binding = _strip_value_range_claims(
                    build_export_file_metadata_binding(
                        relative_path=path.name,
                        module=module,
                        frame=frame,
                        concept_ids=bindable,
                        database=database.strip().lower(),
                        database_class_prefixes=prefixes,
                        dictionary=dictionary,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - recorded per-column, not silenced
                # Non-fatal for a full-export scan: downgrade this file's bound
                # columns to ``unbound`` with the reason, never a silent skip.
                reason = f"binding failed: {type(exc).__name__}: {exc}"[:200]
                compat = [
                    (
                        dataclasses.replace(
                            c, status="unbound", role=None, reason=reason
                        )
                        if c.status == "bound"
                        else c
                    )
                    for c in compat
                ]
                bindable = []
        all_compat.extend(compat)
        if not bindable or binding is None:
            del frame
            gc.collect()
            continue
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
    sidecar_bytes = canonical_sidecar_bytes(sidecar)
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
    sidecar_name = f"column_metadata.sha256-{sidecar_sha}.json"
    sidecar_ref = SidecarRef(
        file=sidecar_name,
        sha256=sidecar_sha,
        size=len(sidecar_bytes),
        record_count=sidecar.record_count,
    ).to_dict()

    dict_fp = {
        "submission_profile": f"{submission_profile.name}/{submission_profile.version}",
        "concept_dict_sha256": fingerprint.concept_dict_sha,
        "sofa2_dict_sha256": fingerprint.sofa2_dict_sha,
        "verified_against_profile": True,
    }
    source_evidence = _source_evidence(root, parquet_paths)
    semantic_review = _semantic_review(file_bindings)

    manifest = {
        "schema_version": NATIVE_MANIFEST_SCHEMA_V2,
        "database": database.strip().lower(),
        "format": "parquet",
        "seal_kind": SEAL_KIND,
        "value_vintage": bound_vintage,
        "value_vintage_basis": vintage_basis,
        "bounds_authority": BOUNDS_AUTHORITY_UNAVAILABLE,
        "metadata_provenance": METADATA_PROVENANCE,
        "metadata_projection_dict": dict_fp,
        "source_evidence": source_evidence,
        "patient_identity": patient_identity,
        "semantic_review": semantic_review,
        # The producer's own paper-readiness verdict, enforced by the consumer
        # gate ``assert_sealed_export_paper_ready``. False until a human signs the
        # semantic review AND patient identity is sufficient.
        "paper_ready": _paper_ready(semantic_review, patient_identity),
        "retrofit_note": (
            "Structural typed retrofit of an untyped module export. Parquet bytes "
            "unchanged. role/unit/time are a CURRENT-dictionary projection (not "
            "extraction provenance); extraction_bounds AND analysis_plausibility_"
            "range omitted; anomalies preserved for downstream QC."
        ),
        "concept_selection": {
            "mode": "all_bindable_in_modules",
            "modules": {fm["module"]: fm["concept_ids"] for fm in files_meta},
        },
        "files": files_meta,
        "feature_definitions": {"included": False},
        "column_metadata": sidecar_ref,
        "compatibility_report": [c.to_dict() for c in all_compat],
    }
    manifest_path = root / NATIVE_MANIFEST

    written_sidecar: Optional[str] = None
    immutable = True
    if not dry_run:
        # Sidecar first (content-addressed, self-atomic), manifest LAST (the
        # single commit point, published atomically).
        ref = write_content_addressed_sidecar(root, sidecar)
        assert ref.file == sidecar_name and ref.sha256 == sidecar_sha
        written_sidecar = ref.file
        _atomic_write_json(manifest_path, manifest)
        for path in parquet_paths:
            if _sha256_size(path) != pre_fingerprint[path.name]:
                immutable = False
                raise TypedRetrofitSealError(
                    f"parquet mutated during seal (must never happen): {path.name}"
                )

    return SealResult(
        export_dir=str(root),
        seal_kind=SEAL_KIND,
        dry_run=dry_run,
        value_vintage=bound_vintage,
        value_vintage_basis=vintage_basis,
        bounds_authority=BOUNDS_AUTHORITY_UNAVAILABLE,
        metadata_provenance=METADATA_PROVENANCE,
        dict_fingerprint=dict_fp,
        source_evidence=source_evidence,
        patient_identity=patient_identity,
        semantic_review=semantic_review,
        paper_ready=_paper_ready(semantic_review, patient_identity),
        sidecar_ref=sidecar_ref,
        sidecar_file=written_sidecar,
        manifest_path=None if dry_run else str(manifest_path),
        files=files_meta,
        columns=all_compat,
        parquet_immutability_verified=immutable and not dry_run,
    )


__all__ = [
    "SEAL_KIND",
    "BOUNDS_AUTHORITY_UNAVAILABLE",
    "METADATA_PROVENANCE",
    "RETROFIT_REVIEW_ATTESTATION_SCHEMA",
    "ColumnCompat",
    "SealResult",
    "TypedRetrofitSealError",
    "assert_sealed_export_paper_ready",
    "build_retrofit_review_attestation",
    "seal_export_structural_typed",
    "verify_retrofit_review_attestation",
]
