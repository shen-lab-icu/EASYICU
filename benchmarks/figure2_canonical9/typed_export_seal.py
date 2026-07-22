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
import tempfile
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
from easyicu.research_agent.graph import HumanReviewDecision, HumanReviewRequest
from benchmarks.figure2_canonical9.retrofit_hitl import (
    checkpoint_receipt_sha256,
    run_human_review_interrupt,
    verify_checkpoint_receipt_binds_request,
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
            "n_stays_with_subject": 0,
            "n_subjects": 0,
            "n_multi_stay_patients": 0,
            "max_stays_per_subject": 0,
            "multi_stay_patients_present": False,
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
    # Honest patient-level structure: a consistent stay->subject mapping does NOT
    # prove patient-level uniqueness. Count subjects and multi-stay patients from
    # the actual rows; ``patient_level_uniqueness_verified`` is True ONLY when every
    # subject maps to exactly one stay. First-ICU-stay ordering cannot be proven
    # without admission times, so it stays unverified.
    stays_per_subject: Dict[Any, int] = {}
    for subject in stay_to_subject.values():
        stays_per_subject[subject] = stays_per_subject.get(subject, 0) + 1
    n_stays = len(stay_to_subject)
    n_subjects = len(stays_per_subject)
    n_multi_stay = sum(1 for count in stays_per_subject.values() if count > 1)
    max_stays = max(stays_per_subject.values()) if stays_per_subject else 0
    return {
        "subject_id_present": True,
        "row_identity": "stay_id",
        "files_with_subject_id": sorted(files_with_subject),
        "stay_to_subject_consistent": True,
        "n_stays_with_subject": n_stays,
        "n_subjects": n_subjects,
        "n_multi_stay_patients": n_multi_stay,
        "max_stays_per_subject": max_stays,
        "multi_stay_patients_present": n_multi_stay > 0,
        # True only when subjects and stays are 1:1 — never inferred from a merely
        # consistent mapping.
        "patient_level_uniqueness_verified": n_subjects == n_stays,
        # Cannot be proven from this export (no admission ordering); stays False so
        # the task cohort-identity policy must resolve first-ICU-stay downstream.
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


# --------------------------------------------------------------------------- #
# Task-level cohort identity policy
# --------------------------------------------------------------------------- #
# ``subject_id`` present is only the FLOOR. Whether a source is paper-ready for a
# task depends on what that task's cohort demands of patient identity. A source
# with repeat ICU admissions (or unverifiable first-stay ordering) must NOT be
# paper-ready for a one-stay-per-patient or first-ICU-stay task — it may only be
# used when the task explicitly permits repeat admissions with patient clustering.
COHORT_IDENTITY_POLICIES = (
    "unique_stay_per_patient",  # exactly one ICU stay per patient (n_subjects==n_stays)
    "first_icu_stay",  # first ICU stay per patient (needs verified admission ordering)
    "repeat_admissions_clustered",  # multi-stay allowed; patient-clustered downstream
)
CohortIdentityPolicy = str  # one of COHORT_IDENTITY_POLICIES (kept as a str alias)
DEFAULT_COHORT_IDENTITY_POLICY = "unique_stay_per_patient"


def _patient_identity_sufficient(identity: Dict[str, Any]) -> bool:
    """A retrofit export has sufficient identity only if subject_id is proven and
    no blocker remains. Stay-level-only identity is NOT sufficient for paper use.

    This is the FLOOR, not the whole gate: a task's cohort identity policy
    (:func:`_identity_satisfies_cohort_policy`) decides whether the *structure*
    (repeat admissions, first-stay ordering) is acceptable for that task."""

    return bool(identity.get("subject_id_present")) and not identity.get("blocker")


def _identity_satisfies_cohort_policy(
    identity: Mapping[str, Any], policy: str
) -> Tuple[bool, Optional[str]]:
    """Does the re-derived identity structure satisfy the task cohort policy?

    Returns ``(ok, reason_if_not)``. Never fabricates: a merely consistent
    stay->subject mapping does NOT satisfy ``unique_stay_per_patient`` unless the
    counts prove 1:1, and ``first_icu_stay`` cannot be satisfied without verified
    admission ordering (which the retrofit export does not carry), so it fails
    closed by design until that ordering is proven upstream.
    """

    if policy not in COHORT_IDENTITY_POLICIES:
        return False, f"unknown_cohort_identity_policy:{policy}"
    if not _patient_identity_sufficient(identity):
        return False, str(identity.get("blocker") or "patient_identity_insufficient")
    if policy == "unique_stay_per_patient":
        if identity.get("patient_level_uniqueness_verified") is not True:
            return False, "repeat_icu_admissions_present"
        return True, None
    if policy == "first_icu_stay":
        if identity.get("first_icu_stay_verified") is not True:
            return False, "first_icu_stay_unverified"
        return True, None
    # repeat_admissions_clustered: subject_id present is enough; the task owns the
    # patient-clustered handling of the multiple stays downstream.
    return True, None


def _paper_ready(
    semantic_review: Dict[str, Any], patient_identity: Dict[str, Any]
) -> bool:
    """The single producer/consumer predicate: a retrofit seal is paper-ready only
    when a human has authorized the projected structure AND identity is sufficient."""

    return semantic_review.get("paper_authorized") is True and (
        _patient_identity_sufficient(patient_identity)
    )


RETROFIT_REVIEW_DECISION_SCHEMA = "easyicu.retrofit_review_decision/2"
RETROFIT_DECISION_FILE = "retrofit_review_decision.json"
RETROFIT_REVIEW_KIND = "protocol_claim"


def _write_once_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish a decision receipt exactly once; a differing rewrite fails closed.

    A HITL decision cannot be silently re-decided: if the file already exists
    with different bytes the write is rejected, so any content change to the
    reviewed artifacts must produce a NEW review, never overwrite the old one.
    """

    raw = _canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise TypedRetrofitSealError(
                f"retrofit review decision already exists with different bytes at "
                f"{path}; a reviewed change requires a new review, not an overwrite"
            )
        return
    fd, tmp = tempfile.mkstemp(prefix=".retrofit-decision-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise TypedRetrofitSealError(
                    f"retrofit review decision raced with different bytes at {path}"
                ) from None
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _rederive_patient_identity_authority(root: Path) -> Tuple[Dict[str, Any], str]:
    """Re-derive patient identity from the ACTUAL parquet columns, never the
    manifest boolean, and return (authority, its digest)."""

    parquet_paths = sorted(
        p for p in root.glob("*.parquet") if p.is_file() and not p.name.startswith(".")
    )
    if not parquet_paths:
        raise TypedRetrofitSealError(f"no parquet files under {root}; cannot re-derive")
    identity = _subject_identity(parquet_paths)  # cross-file stay_id -> subject_id
    digest = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return identity, digest


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_retrofit_review_request(
    export_dir: str | Path,
    *,
    cohort_identity_policy: str = DEFAULT_COHORT_IDENTITY_POLICY,
) -> Tuple["HumanReviewRequest", Dict[str, Any], Dict[str, Any]]:
    """Build the digest-bound Framework v2 review request for a retrofit source.

    The reviewed authority is the tuple of live artifact digests (manifest,
    sidecar, column-derived patient identity) AND the ``cohort_identity_policy``
    the source is reviewed FOR. Its ``review_id`` is DERIVED from that authority by
    ``HumanReviewRequest`` — a caller cannot choose it — so an operator
    ``HumanReviewDecision`` is only valid for THIS exact source AND THIS policy: a
    review for ``repeat_admissions_clustered`` cannot be silently reused for a task
    that needs ``unique_stay_per_patient`` (different digest, different review id).

    Fails closed if the export is not a retrofit seal, identity is insufficient, or
    the re-derived identity structure does not satisfy ``cohort_identity_policy``.
    Returns ``(request, authority, identity)``.
    """

    root = Path(export_dir).expanduser()
    manifest_path = root / NATIVE_MANIFEST
    if not manifest_path.is_file():
        raise TypedRetrofitSealError(f"no {NATIVE_MANIFEST} at {root}; cannot review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("seal_kind") != SEAL_KIND:
        raise TypedRetrofitSealError("not a retrofit seal; cannot review")
    if cohort_identity_policy not in COHORT_IDENTITY_POLICIES:
        raise TypedRetrofitSealError(
            f"unknown cohort identity policy: {cohort_identity_policy!r}"
        )
    sidecar_file = str((manifest.get("column_metadata") or {}).get("file") or "")
    if not sidecar_file or Path(sidecar_file).name != sidecar_file:
        raise TypedRetrofitSealError(
            "retrofit manifest has no single-component sidecar"
        )
    sidecar_path = root / sidecar_file
    if not sidecar_path.is_file():
        raise TypedRetrofitSealError(f"retrofit sidecar missing: {sidecar_file}")

    identity, identity_digest = _rederive_patient_identity_authority(root)
    if not _patient_identity_sufficient(identity):
        raise TypedRetrofitSealError(
            "cannot review: patient identity insufficient, re-derived from parquet "
            f"columns (blocker={identity.get('blocker')!r}) — this export can only be "
            "a development input, never a paper-ready authority"
        )
    ok, reason = _identity_satisfies_cohort_policy(identity, cohort_identity_policy)
    if not ok:
        raise TypedRetrofitSealError(
            f"cannot review for cohort policy {cohort_identity_policy!r}: {reason} "
            f"(n_subjects={identity.get('n_subjects')}, "
            f"n_stays={identity.get('n_stays_with_subject')}, "
            f"multi_stay={identity.get('multi_stay_patients_present')}, "
            f"first_icu_stay_verified={identity.get('first_icu_stay_verified')}) — "
            "review this source only for a task whose cohort policy it satisfies"
        )

    authority = {
        "seal_kind": SEAL_KIND,
        "value_vintage": str(manifest.get("value_vintage") or ""),
        "cohort_identity_policy": cohort_identity_policy,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_sidecar_file": sidecar_file,
        "source_sidecar_sha256": _sha256_file(sidecar_path),
        "patient_identity_authority_sha256": identity_digest,
    }
    authority_sha256 = _canonical_sha(authority)
    request = HumanReviewRequest.create(
        kind=RETROFIT_REVIEW_KIND,
        summary=(
            f"retrofit typed-seal paper-readiness review: {root.name} "
            f"[{cohort_identity_policy}]"
        ),
        authority_sha256=authority_sha256,
        payload=authority,
    )
    return request, authority, identity


def write_retrofit_review_decision(
    export_dir: str | Path,
    *,
    decision: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    cohort_identity_policy: str = DEFAULT_COHORT_IDENTITY_POLICY,
) -> Path:
    """Record a write-once retrofit review from a Framework v2 HumanReviewDecision.

    ``decision`` MUST be a ``HumanReviewDecision`` bound to the digest-derived
    ``HumanReviewRequest`` for this exact source AND ``cohort_identity_policy``:
    its ``review_id`` and ``authority_sha256`` must match, and ``decision`` must be
    ``approved``. A caller cannot ``reviewer='me', review_id='whatever'`` its way to
    approval — the review id is derived from the reviewed authority, not free text.

    ``checkpoint_receipt`` MUST be a :class:`HumanReviewCheckpointReceipt` proving
    the decision flowed through a real LangGraph interrupt + checkpoint resume (see
    :mod:`retrofit_hitl`): a bare constructed decision is not enough. It is bound to
    this request + decision (interrupt/resume digests re-derived and matched).

    The write-once receipt binds the request, decision, and checkpoint canonical
    SHAs. (Honest boundary: the interrupt/checkpoint prove the pause/resume
    MECHANISM; the operator identity remains a trusted local claim.)
    """

    root = Path(export_dir).expanduser()
    request, authority, identity = build_retrofit_review_request(
        root, cohort_identity_policy=cohort_identity_policy
    )
    try:
        parsed = HumanReviewDecision.model_validate(dict(decision))
    except Exception as exc:  # noqa: BLE001 - surfaced as a fail-close
        raise TypedRetrofitSealError(
            f"review decision is not a valid HumanReviewDecision: {exc}"
        ) from exc
    if parsed.review_id != request.review_id:
        raise TypedRetrofitSealError(
            "review decision review_id does not bind this source's derived request"
        )
    if parsed.authority_sha256 != request.authority_sha256:
        raise TypedRetrofitSealError("review decision authority_sha256 mismatch")
    if parsed.decision != "approved":
        raise TypedRetrofitSealError(
            f"review decision is not approved (decision={parsed.decision!r})"
        )
    try:
        checkpoint = verify_checkpoint_receipt_binds_request(
            checkpoint_receipt, request=request, decision=parsed
        )
    except ValueError as exc:
        raise TypedRetrofitSealError(
            f"human review checkpoint receipt does not bind this decision: {exc}"
        ) from exc

    request_json = request.model_dump(mode="json")
    decision_json = parsed.model_dump(mode="json")
    checkpoint_json = checkpoint.model_dump(mode="json")
    receipt: Dict[str, Any] = {
        "schema_version": RETROFIT_REVIEW_DECISION_SCHEMA,
        "seal_kind": SEAL_KIND,
        "value_vintage": authority["value_vintage"],
        "cohort_identity_policy": cohort_identity_policy,
        "review_id": request.review_id,
        "reviewer": parsed.reviewer,
        "reviewed_at": parsed.decided_at,
        "authority_sha256": request.authority_sha256,
        "source_manifest_sha256": authority["source_manifest_sha256"],
        "source_sidecar_file": authority["source_sidecar_file"],
        "source_sidecar_sha256": authority["source_sidecar_sha256"],
        "patient_identity_authority_sha256": authority[
            "patient_identity_authority_sha256"
        ],
        # Honest identity facts, so the task-level consumer can enforce its policy.
        "n_subjects": int(identity.get("n_subjects") or 0),
        "n_stays_with_subject": int(identity.get("n_stays_with_subject") or 0),
        "multi_stay_patients_present": bool(
            identity.get("multi_stay_patients_present")
        ),
        "first_icu_stay_verified": bool(identity.get("first_icu_stay_verified")),
        "review_request": request_json,
        "review_decision": decision_json,
        "human_review_checkpoint": checkpoint_json,
        "request_sha256": _canonical_sha(request_json),
        "decision_sha256": _canonical_sha(decision_json),
        "checkpoint_receipt_sha256": checkpoint_receipt_sha256(checkpoint_json),
    }
    path = root / RETROFIT_DECISION_FILE
    _write_once_json(path, receipt)
    return path


def review_retrofit_export(
    export_dir: str | Path,
    *,
    reviewer: str,
    decided_at: str,
    cohort_identity_policy: str = DEFAULT_COHORT_IDENTITY_POLICY,
    note: str = "",
) -> Path:
    """Run a real interrupt-backed HITL review and record the write-once decision.

    This is the operator entry point: it builds the digest-bound request for
    ``cohort_identity_policy``, drives a genuine LangGraph interrupt + checkpoint
    resume (:func:`retrofit_hitl.run_human_review_interrupt`) with the operator's
    approval, and writes the write-once decision receipt binding the checkpoint.
    Fails closed if the source cannot be reviewed for the policy.
    """

    root = Path(export_dir).expanduser()
    request, _authority, _identity = build_retrofit_review_request(
        root, cohort_identity_policy=cohort_identity_policy
    )

    def _decide(
        requests: Tuple[HumanReviewRequest, ...],
    ) -> List[HumanReviewDecision]:
        req = requests[0]
        return [
            HumanReviewDecision(
                review_id=req.review_id,
                authority_sha256=req.authority_sha256,
                decision="approved",
                reviewer=reviewer,
                decided_at=decided_at,
                note=note,
            )
        ]

    decisions, checkpoint = run_human_review_interrupt(
        [request],
        decide=_decide,
        thread_id="retrofit-" + request.authority_sha256[:16],
    )
    return write_retrofit_review_decision(
        root,
        decision=decisions[0].model_dump(mode="json"),
        checkpoint_receipt=checkpoint.model_dump(mode="json"),
        cohort_identity_policy=cohort_identity_policy,
    )


def _reconcile_embedded_review(
    receipt: Mapping[str, Any],
    *,
    expected_request: "HumanReviewRequest",
    manifest_sha: str,
    sidecar_file: str,
    sidecar_sha: str,
    identity_digest: str,
) -> None:
    """Re-validate the embedded HITL artifacts against a source authority.

    Shared by the live gate (authority = live parquet columns + files) and the
    content-addressed staged gate (authority = staged, SHA-verified blobs). Checks
    the embedded request binds ``expected_request``, the decision binds the request
    and is approved, the checkpoint receipt binds both, and every recorded source
    digest equals the caller-provided (re-derived / re-digested) value. Fails closed.
    """

    try:
        req = HumanReviewRequest.model_validate(receipt.get("review_request"))
        dec = HumanReviewDecision.model_validate(receipt.get("review_decision"))
    except Exception as exc:  # noqa: BLE001 - surfaced as a fail-close
        raise TypedRetrofitSealError(
            f"review request/decision is not a valid Framework v2 artifact: {exc}"
        ) from exc
    if req.review_id != expected_request.review_id:
        raise TypedRetrofitSealError(
            "review request does not bind this source (review_id mismatch — the "
            "reviewed authority differs from the resolved artifacts)"
        )
    if req.authority_sha256 != expected_request.authority_sha256:
        raise TypedRetrofitSealError("review request authority_sha256 mismatch")
    if dec.review_id != req.review_id or dec.authority_sha256 != req.authority_sha256:
        raise TypedRetrofitSealError("review decision is not bound to the request")
    if dec.decision != "approved":
        raise TypedRetrofitSealError(
            f"review decision is not approved (decision={dec.decision!r})"
        )
    if receipt.get("request_sha256") != _canonical_sha(req.model_dump(mode="json")):
        raise TypedRetrofitSealError("review request canonical sha mismatch (tampered)")
    if receipt.get("decision_sha256") != _canonical_sha(dec.model_dump(mode="json")):
        raise TypedRetrofitSealError(
            "review decision canonical sha mismatch (tampered)"
        )
    # The decision must have flowed through a real LangGraph interrupt + checkpoint
    # resume: a bare (non-interrupt) decision has no valid checkpoint receipt.
    try:
        checkpoint = verify_checkpoint_receipt_binds_request(
            receipt.get("human_review_checkpoint") or {}, request=req, decision=dec
        )
    except ValueError as exc:
        raise TypedRetrofitSealError(
            f"review checkpoint receipt does not bind this decision: {exc}"
        ) from exc
    if receipt.get("checkpoint_receipt_sha256") != checkpoint_receipt_sha256(
        checkpoint.model_dump(mode="json")
    ):
        raise TypedRetrofitSealError(
            "review checkpoint receipt canonical sha mismatch (tampered)"
        )
    if receipt.get("source_manifest_sha256") != manifest_sha:
        raise TypedRetrofitSealError(
            "review decision source_manifest_sha256 mismatch (manifest changed since "
            "review)"
        )
    if receipt.get("source_sidecar_file") != sidecar_file:
        raise TypedRetrofitSealError("review decision sidecar reference mismatch")
    if receipt.get("source_sidecar_sha256") != sidecar_sha:
        raise TypedRetrofitSealError(
            "review decision source_sidecar_sha256 mismatch (sidecar changed since "
            "review)"
        )
    if receipt.get("patient_identity_authority_sha256") != identity_digest:
        raise TypedRetrofitSealError(
            "review decision patient_identity_authority_sha256 mismatch (identity "
            "differs from the reviewed authority)"
        )


def assert_sealed_export_paper_ready(export_dir: str | Path) -> Dict[str, Any]:
    """Consumer half of the retrofit producer→consumer gate — fail-closed.

    This performs live content-addressed re-validation: it re-reads the manifest
    and sidecar and RE-COMPUTES their digests, RE-DERIVES patient identity from
    the actual parquet columns (never the manifest boolean), and requires a valid
    write-once HITL review decision receipt whose bound digests match every
    re-derived value. No hand-editable manifest field can make an export
    paper-ready. A non-retrofit native export (no ``seal_kind``) is governed by
    the official typed-authority path and is returned unchanged.
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

    # (req 3) Re-derive patient identity from the actual columns, fail closed.
    identity, identity_digest = _rederive_patient_identity_authority(root)
    if not _patient_identity_sufficient(identity):
        raise TypedRetrofitSealError(
            "retrofit export patient identity is insufficient, re-derived from "
            f"parquet columns (blocker={identity.get('blocker')!r}); refusing "
            "paper-facing use — development input only"
        )

    # (req 1+2) A valid write-once Framework v2 HITL decision is REQUIRED. Rebuild
    # the digest-bound request from the re-derived authority and re-validate the
    # embedded HumanReviewRequest + HumanReviewDecision — never trust flags.
    decision_path = root / RETROFIT_DECISION_FILE
    if not decision_path.is_file():
        raise TypedRetrofitSealError(
            "retrofit export has no write-once HITL review decision receipt; "
            "refusing paper-facing use (paper_authorized manifest flags are not "
            "trusted)"
        )
    try:
        receipt = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TypedRetrofitSealError(f"unreadable review decision: {exc}") from exc
    if not isinstance(receipt, dict):
        raise TypedRetrofitSealError("review decision is not an object")
    if receipt.get("schema_version") != RETROFIT_REVIEW_DECISION_SCHEMA:
        raise TypedRetrofitSealError("unknown review decision schema")

    # (req 2) The cohort identity policy the source was reviewed FOR travels in the
    # receipt; rebuild the request with it so the derived review_id matches, and
    # re-check the live identity still satisfies it (build_retrofit_review_request
    # fail-closes if a repeat-admissions/unverified-first source no longer meets the
    # reviewed policy). The review_id binds the policy, so a downgrade fails closed.
    reviewed_policy = receipt.get("cohort_identity_policy")
    if reviewed_policy not in COHORT_IDENTITY_POLICIES:
        raise TypedRetrofitSealError(
            f"review decision has an unknown cohort identity policy: "
            f"{reviewed_policy!r}"
        )
    # The request the operator MUST have signed, rebuilt from the LIVE artifacts.
    # Its review_id is derived from the reviewed authority, so any manifest /
    # sidecar / identity / policy drift changes review_id and fails closed here.
    expected_request, _authority, _live_identity = build_retrofit_review_request(
        root, cohort_identity_policy=reviewed_policy
    )

    # (req 1) Recompute every bound source digest from the LIVE artifacts, then
    # re-validate the embedded request / decision / checkpoint against them.
    sidecar_file = str((manifest.get("column_metadata") or {}).get("file") or "")
    if not sidecar_file or not (root / sidecar_file).is_file():
        raise TypedRetrofitSealError("review decision sidecar reference mismatch")
    _reconcile_embedded_review(
        receipt,
        expected_request=expected_request,
        manifest_sha=_sha256_file(manifest_path),
        sidecar_file=sidecar_file,
        sidecar_sha=_sha256_file(root / sidecar_file),
        identity_digest=identity_digest,
    )
    return manifest


RETROFIT_REVIEW_ATTESTATION_SCHEMA = "easyicu.retrofit_review_attestation/1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_ATTESTATION_RECEIPT_FIELDS = (
    "cohort_identity_policy",
    "review_id",
    "reviewer",
    "reviewed_at",
    "authority_sha256",
    "request_sha256",
    "decision_sha256",
    "checkpoint_receipt_sha256",
    "source_manifest_sha256",
    "source_sidecar_file",
    "source_sidecar_sha256",
    "patient_identity_authority_sha256",
    "n_subjects",
    "n_stays_with_subject",
    "multi_stay_patients_present",
    "first_icu_stay_verified",
)


def build_retrofit_review_attestation(export_dir: str | Path) -> Dict[str, Any]:
    """Mint a frozen paper-readiness attestation from the write-once review decision.

    FAILS CLOSED through :func:`assert_sealed_export_paper_ready`, which re-derives
    patient identity from columns and requires a valid write-once HITL decision
    receipt whose digests match the live artifacts. An export that is unreviewed or
    identity-insufficient (e.g. full6) cannot yield an attestation. The attestation
    is a frozen mirror of the decision receipt, bound into a task binding so
    acceptance can reconcile it against the live receipt.
    """

    root = Path(export_dir).expanduser()
    manifest = assert_sealed_export_paper_ready(root)  # fail-close (receipt-backed)
    receipt = json.loads((root / RETROFIT_DECISION_FILE).read_text(encoding="utf-8"))
    attestation = {
        "schema_version": RETROFIT_REVIEW_ATTESTATION_SCHEMA,
        "seal_kind": SEAL_KIND,
        "value_vintage": str(manifest.get("value_vintage") or ""),
        "paper_ready": True,
    }
    for field in _ATTESTATION_RECEIPT_FIELDS:
        attestation[field] = receipt[field]
    return attestation


def verify_retrofit_review_attestation(
    attestation: Mapping[str, Any],
    *,
    export_dir: str | Path | None = None,
) -> None:
    """Re-verify a bound attestation; fail-closed (raises ``TypedRetrofitSealError``).

    Offline structural checks always run (schema, seal kind, ``paper_ready``, digest
    shape, reviewer/review id). When ``export_dir`` is provided the receipt-backed
    gate is re-run (identity re-derived from columns, live digests recomputed) and
    the attestation is reconciled field-by-field against the live decision receipt,
    so any tamper or drift fails closed.
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
        "patient_identity_authority_sha256",
        "authority_sha256",
        "request_sha256",
        "decision_sha256",
        "checkpoint_receipt_sha256",
    ):
        value = attestation.get(key)
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise TypedRetrofitSealError(f"attestation {key} is not a sha256 digest")
    if not str(attestation.get("reviewer") or "").strip():
        raise TypedRetrofitSealError("attestation has no reviewer")
    if not str(attestation.get("review_id") or "").strip():
        raise TypedRetrofitSealError("attestation has no review_id")
    if attestation.get("cohort_identity_policy") not in COHORT_IDENTITY_POLICIES:
        raise TypedRetrofitSealError("attestation has no known cohort identity policy")
    if export_dir is None:
        return

    # Live: re-run the receipt-backed gate, then reconcile the frozen attestation
    # against the live decision receipt field-by-field.
    root = Path(export_dir).expanduser()
    assert_sealed_export_paper_ready(root)  # fail-close: identity + receipt re-derived
    receipt = json.loads((root / RETROFIT_DECISION_FILE).read_text(encoding="utf-8"))
    for field in _ATTESTATION_RECEIPT_FIELDS:
        if attestation.get(field) != receipt.get(field):
            raise TypedRetrofitSealError(
                f"attestation {field} disagrees with the live decision receipt "
                "(tampered or drifted)"
            )


# --------------------------------------------------------------------------- #
# Content-addressed source-authority staging (production acceptance without the
# live external parquet export dir)
# --------------------------------------------------------------------------- #
STAGED_AUTHORITY_SCHEMA = "easyicu.retrofit_staged_authority/1"
STAGED_AUTHORITY_INDEX = "staged_authority_index.json"


def _stage_blob(path: Path, data: bytes) -> None:
    """Write a content-addressed blob once; a byte-differing collision fails closed."""

    if path.exists():
        if path.read_bytes() != data:
            raise TypedRetrofitSealError(f"staged blob sha collision at {path}")
        return
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def stage_retrofit_source_authority(
    export_dir: str | Path, dest_dir: str | Path
) -> Dict[str, Any]:
    """Register a reviewed retrofit source as content-addressed authority blobs.

    Fails closed through :func:`assert_sealed_export_paper_ready` (only a
    legitimately reviewed, identity-sufficient source can be staged), then writes
    the manifest, sidecar, column-derived patient-identity authority, and write-once
    HITL decision receipt as SHA-named blobs plus a canonical index. Production
    acceptance can then re-digest these blobs WITHOUT the live external export dir —
    the trust root becomes the reviewed, content-addressed authority, not a mutable
    path. Returns the staged index. Writes nothing to ``export_dir``.
    """

    root = Path(export_dir).expanduser()
    manifest = assert_sealed_export_paper_ready(root)  # fail-close (receipt-backed)
    if manifest.get("seal_kind") != SEAL_KIND:
        raise TypedRetrofitSealError("cannot stage a non-retrofit export")
    manifest_path = root / NATIVE_MANIFEST
    sidecar_file = str((manifest.get("column_metadata") or {}).get("file") or "")
    sidecar_path = root / sidecar_file
    receipt_path = root / RETROFIT_DECISION_FILE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity, identity_digest = _rederive_patient_identity_authority(root)

    manifest_bytes = manifest_path.read_bytes()
    sidecar_bytes = sidecar_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    identity_bytes = _canonical_json_bytes(identity)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()

    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    names = {
        "source_manifest_file": f"{manifest_sha}.manifest.json",
        "source_sidecar_blob": f"{sidecar_sha}.sidecar.json",
        "patient_identity_blob": f"{identity_digest}.identity.json",
        "decision_receipt_blob": f"{receipt_sha}.decision.json",
    }
    _stage_blob(dest / names["source_manifest_file"], manifest_bytes)
    _stage_blob(dest / names["source_sidecar_blob"], sidecar_bytes)
    _stage_blob(dest / names["patient_identity_blob"], identity_bytes)
    _stage_blob(dest / names["decision_receipt_blob"], receipt_bytes)

    index = {
        "schema_version": STAGED_AUTHORITY_SCHEMA,
        "seal_kind": SEAL_KIND,
        "value_vintage": str(manifest.get("value_vintage") or ""),
        "cohort_identity_policy": receipt.get("cohort_identity_policy"),
        "source_manifest_sha256": manifest_sha,
        "source_manifest_file": names["source_manifest_file"],
        "source_sidecar_file": sidecar_file,
        "source_sidecar_sha256": sidecar_sha,
        "source_sidecar_blob": names["source_sidecar_blob"],
        "patient_identity_authority_sha256": identity_digest,
        "patient_identity_blob": names["patient_identity_blob"],
        "decision_receipt_sha256": receipt_sha,
        "decision_receipt_blob": names["decision_receipt_blob"],
    }
    _atomic_write_json(dest / STAGED_AUTHORITY_INDEX, index)
    return index


def verify_retrofit_review_attestation_from_staged(
    attestation: Mapping[str, Any], staged_dir: str | Path
) -> None:
    """Re-verify a bound attestation from content-addressed staged authority.

    The paper-facing production analogue of :func:`verify_retrofit_review_attestation`
    that needs no live external export dir. Every staged blob is loaded by SHA
    (content-addressed integrity), the operator-signed review authority (bound into
    the derived ``review_id``) must reference exactly those staged digests, the
    decision + checkpoint are re-validated, the staged identity must still satisfy
    the reviewed cohort policy, and the frozen attestation is reconciled against the
    staged decision receipt. Any tamper or drift fails closed.
    """

    verify_retrofit_review_attestation(attestation)  # offline shape checks (no dir)
    staged = Path(staged_dir).expanduser()
    index_path = staged / STAGED_AUTHORITY_INDEX
    if not index_path.is_file():
        raise TypedRetrofitSealError(f"no staged authority index at {staged}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TypedRetrofitSealError(
            f"unreadable staged authority index: {exc}"
        ) from exc
    if not isinstance(index, dict):
        raise TypedRetrofitSealError("staged authority index is not an object")
    if index.get("schema_version") != STAGED_AUTHORITY_SCHEMA:
        raise TypedRetrofitSealError("unknown staged authority schema")
    if index.get("seal_kind") != SEAL_KIND:
        raise TypedRetrofitSealError("staged authority is not a retrofit seal")

    def _load(sha: Any, blob_name: Any) -> bytes:
        name = str(blob_name or "")
        if not name or Path(name).name != name:
            raise TypedRetrofitSealError(
                f"staged blob reference invalid: {blob_name!r}"
            )
        path = staged / name
        if not path.is_file():
            raise TypedRetrofitSealError(f"staged blob missing: {name}")
        data = path.read_bytes()
        if not isinstance(sha, str) or hashlib.sha256(data).hexdigest() != sha:
            raise TypedRetrofitSealError(f"staged blob sha mismatch: {name}")
        return data

    manifest_sha = index.get("source_manifest_sha256")
    sidecar_sha = index.get("source_sidecar_sha256")
    identity_sha = index.get("patient_identity_authority_sha256")
    receipt_sha = index.get("decision_receipt_sha256")
    manifest_bytes = _load(manifest_sha, index.get("source_manifest_file"))
    _sidecar_bytes = _load(sidecar_sha, index.get("source_sidecar_blob"))
    identity_bytes = _load(identity_sha, index.get("patient_identity_blob"))
    receipt_bytes = _load(receipt_sha, index.get("decision_receipt_blob"))

    staged_manifest = json.loads(manifest_bytes)
    identity = json.loads(identity_bytes)
    if hashlib.sha256(_canonical_json_bytes(identity)).hexdigest() != identity_sha:
        raise TypedRetrofitSealError("staged identity authority is not canonical")
    receipt = json.loads(receipt_bytes)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != RETROFIT_REVIEW_DECISION_SCHEMA
    ):
        raise TypedRetrofitSealError("staged decision receipt schema mismatch")

    policy = index.get("cohort_identity_policy")
    if policy not in COHORT_IDENTITY_POLICIES:
        raise TypedRetrofitSealError("staged authority has an unknown cohort policy")
    ok, reason = _identity_satisfies_cohort_policy(identity, policy)
    if not ok:
        raise TypedRetrofitSealError(
            f"staged identity does not satisfy cohort policy {policy!r}: {reason}"
        )

    # The operator-signed review authority (its review_id binds the whole payload)
    # must reference EXACTLY the staged, SHA-verified content — this is what ties
    # the human decision to the content address instead of a mutable path.
    try:
        req = HumanReviewRequest.model_validate(receipt.get("review_request"))
    except Exception as exc:  # noqa: BLE001 - surfaced as a fail-close
        raise TypedRetrofitSealError(
            f"staged review request is not a valid Framework v2 artifact: {exc}"
        ) from exc
    payload = req.payload
    if req.authority_sha256 != _canonical_sha(payload):
        raise TypedRetrofitSealError(
            "staged review request authority_sha256 does not digest its payload"
        )
    sidecar_file = str((staged_manifest.get("column_metadata") or {}).get("file") or "")
    if (
        payload.get("seal_kind") != SEAL_KIND
        or payload.get("cohort_identity_policy") != policy
        or payload.get("source_manifest_sha256") != manifest_sha
        or payload.get("source_sidecar_sha256") != sidecar_sha
        or payload.get("source_sidecar_file") != sidecar_file
        or payload.get("patient_identity_authority_sha256") != identity_sha
    ):
        raise TypedRetrofitSealError(
            "staged review authority does not match the content-addressed source"
        )

    _reconcile_embedded_review(
        receipt,
        expected_request=req,
        manifest_sha=str(manifest_sha),
        sidecar_file=sidecar_file,
        sidecar_sha=str(sidecar_sha),
        identity_digest=str(identity_sha),
    )

    for field in _ATTESTATION_RECEIPT_FIELDS:
        if attestation.get(field) != receipt.get(field):
            raise TypedRetrofitSealError(
                f"attestation {field} disagrees with the staged decision receipt "
                "(tampered or drifted)"
            )
    if (
        attestation.get("source_manifest_sha256") != manifest_sha
        or attestation.get("source_sidecar_sha256") != sidecar_sha
        or attestation.get("patient_identity_authority_sha256") != identity_sha
    ):
        raise TypedRetrofitSealError(
            "attestation source digests differ from the staged content-addressed source"
        )


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
    "RETROFIT_REVIEW_DECISION_SCHEMA",
    "ColumnCompat",
    "SealResult",
    "TypedRetrofitSealError",
    "assert_sealed_export_paper_ready",
    "build_retrofit_review_attestation",
    "build_retrofit_review_request",
    "seal_export_structural_typed",
    "verify_retrofit_review_attestation",
    "write_retrofit_review_decision",
]
