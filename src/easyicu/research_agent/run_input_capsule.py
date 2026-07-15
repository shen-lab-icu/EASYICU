"""Immutable scientific-input authority for research-agent resume.

The ordinary run directory contains mutable working copies (``cohort.parquet``
and ``research_context.json``).  A resume must therefore prove that the caller
is continuing the same study *before* touching either file.  This module keeps
that boundary small and independent from the execute loop:

* :class:`RunInputCapsule` freezes the scientific request plus the staged cohort
  and context digests;
* resume verifies the capsule and all current successful step evidence without
  mutating the run directory;
* model/code/prompt/validator drift is recorded in a new attempt receipt.  It is
  not treated as a different study because framework fixes must be able to
  resume an unfinished step, but no old audit cache is authoritative under a
  different environment hash.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .evidence import EvidenceStore, sha256_of_file
from .prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .runtime_artifacts import (
    current_successful_step_records,
    current_step_records,
    verified_run_evidence_path,
)
from .schema import ResearchContext, TimeWindow


RUN_INPUT_CAPSULE_FILENAME = "run_input_capsule.json"
RUN_INPUT_CAPSULE_EVIDENCE_ID = "run_input_capsule"
RUN_INPUT_CAPSULE_SCHEMA_VERSION = "easyicu.run_input_capsule/1"
RESUME_ENVIRONMENT_SCHEMA_VERSION = "easyicu.resume_environment_receipt/1"


class RunInputIdentityError(ValueError):
    """The requested resume cannot be proven to be the same study."""


class RunInputCapsule(BaseModel):
    """Immutable identity and input-evidence seal for one run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = RUN_INPUT_CAPSULE_SCHEMA_VERSION
    scientific_identity: Dict[str, Any]
    scientific_identity_sha256: str
    context_evidence_id: str = "research_context"
    context_sha256: str
    context_relative_path: str = "research_context.json"
    cohort_relative_path: str = "cohort.parquet"
    cohort_sha256: str
    trajectory_relative_path: Optional[str] = None
    trajectory_sha256: Optional[str] = None
    experiment_spec_evidence_id: Optional[str] = None
    experiment_spec_sha256: Optional[str] = None
    experiment_spec_relative_path: Optional[str] = None
    initial_environment: Dict[str, Any]
    legacy_adopted: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ResumeInputAuthority:
    """Verified paths and records returned without changing the run directory."""

    capsule: RunInputCapsule
    context_evidence_path: Path
    experiment_spec_evidence_path: Optional[Path]
    evidence_records: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class PreparedResumeInput:
    """Outcome of pre-write resume identity preparation."""

    resume_state: Dict[str, Any]
    input_verified: bool
    context_evidence_path: Optional[Path]
    cohort_path: Optional[Path]
    experiment_spec_path: Optional[Path]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _dataframe_content_sha256(frame: pd.DataFrame) -> str:
    """Stable value+schema identity for an in-memory cohort."""

    digest = hashlib.sha256()
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "n_rows": int(len(frame)),
    }
    digest.update(_canonical_json_bytes(schema))
    try:
        hashed = pd.util.hash_pandas_object(
            frame,
            index=False,
            categorize=True,
        )
        digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    except (TypeError, ValueError):
        # ICU cohorts are scalar tables, but retain a deterministic fallback for
        # extension/object columns that pandas cannot hash directly.
        digest.update(
            frame.to_json(
                orient="split",
                date_format="iso",
                date_unit="ns",
                default_handler=str,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def cohort_input_identity(
    cohort: Union[str, Path, pd.DataFrame],
) -> Dict[str, Any]:
    """Return a PHI-safe byte/value identity without staging the cohort."""

    if isinstance(cohort, (str, Path)):
        source = Path(cohort).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Cohort path not found: {source}")
        payload: Dict[str, Any] = {
            "kind": "file",
            "format": source.suffix.lower(),
            "size_bytes": int(source.stat().st_size),
            "sha256": sha256_of_file(source),
        }
        trajectory = source.with_name(f"{source.stem}_trajectory.parquet")
        if trajectory.is_file():
            payload["trajectory"] = {
                "size_bytes": int(trajectory.stat().st_size),
                "sha256": sha256_of_file(trajectory),
            }
        else:
            payload["trajectory"] = None
        return payload
    if isinstance(cohort, pd.DataFrame):
        return {
            "kind": "dataframe",
            "n_rows": int(len(cohort)),
            "n_columns": int(len(cohort.columns)),
            "sha256": _dataframe_content_sha256(cohort),
            "trajectory": None,
        }
    raise TypeError("cohort must be a path or a pandas DataFrame")


def _cohort_frame(cohort: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(cohort, pd.DataFrame):
        return cohort
    path = Path(cohort).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise RunInputIdentityError(
        f"Cannot compare legacy cohort content for unsupported format {suffix!r}."
    )


def _source_file_identities(source_files: Optional[Sequence[Any]]) -> list[Dict[str, Any]]:
    identities: list[Dict[str, Any]] = []
    for entry in source_files or []:
        role = None
        database = None
        raw_path: Any = entry
        if isinstance(entry, Mapping):
            raw_path = entry.get("path")
            role = entry.get("role")
            database = entry.get("database")
        if raw_path is None:
            identities.append(
                {"sha256": None, "role": role, "database": database, "missing": True}
            )
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            identities.append(
                {"sha256": None, "role": role, "database": database, "missing": True}
            )
            continue
        identities.append(
            {
                "sha256": sha256_of_file(path),
                "size_bytes": int(path.stat().st_size),
                "role": role,
                "database": database,
                "missing": False,
            }
        )
    return identities


def build_scientific_identity(
    *,
    cohort: Union[str, Path, pd.DataFrame],
    question: str,
    cohort_name: str,
    database: str,
    target_outcome: Optional[str],
    primary_exposure: Optional[str],
    cross_database_validation: Optional[Sequence[str]],
    inclusion_criteria: Optional[Sequence[str]],
    exclusion_criteria: Optional[Sequence[str]],
    id_columns: Optional[Sequence[str]],
    time_columns: Optional[Sequence[str]],
    outcome_columns: Optional[Sequence[str]],
    time_windows: Optional[Sequence[TimeWindow]],
    concept_descriptions: Optional[Dict[str, str]],
    user_preferences: Optional[Dict[str, Any]],
    notes: Optional[str],
    skill_key: Optional[str],
    experiment_spec: Optional[BaseModel],
    source_files: Optional[Sequence[Any]],
    disable_icu_context: bool,
) -> Dict[str, Any]:
    """Canonical scientific request; execution-only knobs are excluded."""

    return _jsonable(
        {
            "cohort": cohort_input_identity(cohort),
            "question": question,
            "cohort_name": cohort_name,
            "database": database,
            "target_outcome": target_outcome,
            "primary_exposure": primary_exposure,
            "cross_database_validation": list(cross_database_validation or []),
            "inclusion_criteria": list(inclusion_criteria or []),
            "exclusion_criteria": list(exclusion_criteria or []),
            "id_columns": list(id_columns or []),
            "time_columns": list(time_columns or []),
            "outcome_columns": list(outcome_columns or []),
            "time_windows": (
                [window.model_dump(mode="json") for window in time_windows]
                if time_windows is not None
                else None
            ),
            "concept_descriptions": dict(concept_descriptions or {}),
            "user_preferences": dict(user_preferences or {}),
            "notes": notes,
            "skill_key": skill_key,
            "experiment_spec": (
                experiment_spec.model_dump(mode="json")
                if experiment_spec is not None
                else None
            ),
            "source_files": _source_file_identities(source_files),
            "disable_icu_context": bool(disable_icu_context),
        }
    )


def _tree_sha256(root: Path, *, relative_paths: Optional[Sequence[Path]] = None) -> str:
    digest = hashlib.sha256()
    paths = (
        sorted(relative_paths, key=lambda path: str(path))
        if relative_paths is not None
        else sorted(root.rglob("*.py"))
    )
    for path in paths:
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        raw = path.read_bytes()
        digest.update(str(relative).encode("utf-8"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def engine_code_sha256() -> str:
    return _tree_sha256(Path(__file__).resolve().parent)


@lru_cache(maxsize=1)
def validator_code_sha256() -> str:
    root = Path(__file__).resolve().parent
    paths = [
        root / "code_preflight.py",
        root / "declared_product_contract.py",
        root / "audits" / "validators.py",
        root / "audits" / "step_summary_integrity.py",
    ]
    return _tree_sha256(root, relative_paths=paths)


def build_environment_identity(*, llm_signature: str) -> Dict[str, Any]:
    prompt_files = prompt_pack_files()
    return {
        "llm_signature": llm_signature,
        "llm_signature_sha256": canonical_sha256(llm_signature),
        "engine_code_sha256": engine_code_sha256(),
        "validator_code_sha256": validator_code_sha256(),
        "prompt_pack_version": PROMPT_PACK_VERSION,
        "prompt_pack_files": prompt_files,
        "prompt_pack_sha256": canonical_sha256(prompt_files),
    }


def _records_from_index(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    index_path = Path(run_dir) / "evidence" / "evidence_index.json"
    if not index_path.is_file():
        raise RunInputIdentityError(
            "Cannot resume safely: evidence_index.json is missing."
        )
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RunInputIdentityError(
            "Cannot resume safely: evidence_index.json is corrupt."
        ) from exc
    if not isinstance(payload, list):
        raise RunInputIdentityError(
            "Cannot resume safely: evidence_index.json is not a record list."
        )
    records: Dict[str, Dict[str, Any]] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            raise RunInputIdentityError(
                "Cannot resume safely: evidence index contains a non-object record."
            )
        evidence_id = str(raw.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in records:
            raise RunInputIdentityError(
                "Cannot resume safely: evidence ids are missing or duplicated."
            )
        records[evidence_id] = dict(raw)
    return records


def _verified_record_path(
    *,
    run_dir: Path,
    records: Mapping[str, Dict[str, Any]],
    evidence_id: str,
    expected_sha256: Optional[str] = None,
) -> Path:
    record = records.get(evidence_id)
    if record is None:
        raise RunInputIdentityError(
            f"Cannot resume safely: required evidence {evidence_id!r} is missing."
        )
    path = verified_run_evidence_path(run_dir, record)
    if path is None:
        raise RunInputIdentityError(
            f"Cannot resume safely: required evidence {evidence_id!r} failed digest verification."
        )
    if expected_sha256 is not None and str(record.get("sha256")) != expected_sha256:
        raise RunInputIdentityError(
            f"Cannot resume safely: required evidence {evidence_id!r} has the wrong digest."
        )
    return path


def seal_run_input_capsule(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    scientific_identity: Dict[str, Any],
    initial_environment: Dict[str, Any],
    context_path: Path,
    cohort_path: Path,
    experiment_spec_path: Optional[Path],
    legacy_adopted: bool = False,
) -> RunInputCapsule:
    """Write and evidence-register the capsule exactly once."""

    capsule_path = Path(run_dir) / RUN_INPUT_CAPSULE_FILENAME
    existing = evidence.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
    if capsule_path.exists() or existing is not None:
        raise RunInputIdentityError("Run input capsule is immutable and already exists.")
    context_record = evidence.get("research_context")
    if context_record is None:
        raise RunInputIdentityError(
            "Cannot seal run input capsule without research_context evidence."
        )
    experiment_record = evidence.get("experiment_spec")
    trajectory_path = Path(run_dir) / "cohort_trajectory.parquet"
    capsule = RunInputCapsule(
        scientific_identity=scientific_identity,
        scientific_identity_sha256=canonical_sha256(scientific_identity),
        context_sha256=str(context_record.sha256),
        cohort_sha256=sha256_of_file(cohort_path),
        trajectory_relative_path=(
            trajectory_path.name if trajectory_path.is_file() else None
        ),
        trajectory_sha256=(
            sha256_of_file(trajectory_path) if trajectory_path.is_file() else None
        ),
        experiment_spec_evidence_id=(
            str(experiment_record.evidence_id) if experiment_record is not None else None
        ),
        experiment_spec_sha256=(
            str(experiment_record.sha256) if experiment_record is not None else None
        ),
        experiment_spec_relative_path=(
            str(experiment_spec_path.relative_to(run_dir))
            if experiment_record is not None and experiment_spec_path is not None
            else None
        ),
        initial_environment=initial_environment,
        legacy_adopted=legacy_adopted,
    )
    capsule_path.write_text(capsule.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Immutable scientific-input, context, cohort, model, prompt, validator, "
            "and engine identity for safe step-level resume."
        ),
        source_path=capsule_path,
        evidence_id=RUN_INPUT_CAPSULE_EVIDENCE_ID,
        producer="pipeline",
        generation_mode="system",
    )
    return capsule


def adopt_verified_legacy_run_input_capsule(
    *,
    run_dir: Path,
    cohort: Union[str, Path, pd.DataFrame],
    scientific_identity: Dict[str, Any],
    initial_environment: Dict[str, Any],
    enforcement_mode: Any,
) -> ResumeInputAuthority:
    """One-time migration for a pre-capsule run with completed evidence.

    Adoption trusts only old digest-bound authorities: ``research_context`` and
    ``provenance_sources``. The incoming cohort must be value-identical to the
    staged cohort and every reconstructible study coordinate must agree with the
    sealed context. A legacy run lacking either authority is not safely
    adoptable and remains fail-closed.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    records = _records_from_index(run_dir)
    context_path = _verified_record_path(
        run_dir=run_dir,
        records=records,
        evidence_id="research_context",
    )
    provenance_path = _verified_record_path(
        run_dir=run_dir,
        records=records,
        evidence_id="provenance_sources",
    )
    try:
        context = ResearchContext.model_validate_json(
            context_path.read_text(encoding="utf-8")
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RunInputIdentityError(
            "Cannot adopt legacy resume: context/provenance evidence is invalid."
        ) from exc

    expected_pairs = {
        "question": context.research_question,
        "cohort_name": context.cohort.cohort_name,
        "database": context.cohort.database,
        "target_outcome": context.target_outcome,
        "primary_exposure": context.primary_exposure,
        "cross_database_validation": list(context.cross_database_validation),
        "inclusion_criteria": list(context.cohort.inclusion_criteria),
        "exclusion_criteria": list(context.cohort.exclusion_criteria),
        "notes": context.notes,
    }
    mismatched = sorted(
        key
        for key, expected in expected_pairs.items()
        if scientific_identity.get(key) != _jsonable(expected)
    )
    for key, expected in (
        ("id_columns", list(context.cohort.id_columns)),
        ("time_columns", list(context.cohort.time_columns)),
        ("outcome_columns", list(context.cohort.outcome_columns)),
    ):
        requested = list(scientific_identity.get(key) or [])
        if requested and requested != expected:
            mismatched.append(key)
    requested_windows = scientific_identity.get("time_windows")
    if requested_windows is not None and requested_windows != [
        window.model_dump(mode="json") for window in context.time_windows
    ]:
        mismatched.append("time_windows")
    requested_preferences = scientific_identity.get("user_preferences") or {}
    context_preferences = (
        context.user_preferences.model_dump(mode="json")
        if context.user_preferences is not None
        else {}
    )
    if requested_preferences != context_preferences:
        mismatched.append("user_preferences")
    requested_descriptions = scientific_identity.get("concept_descriptions") or {}
    context_variables = {variable.name: variable for variable in context.variables}
    for name, description in requested_descriptions.items():
        variable = context_variables.get(name)
        if variable is None or str(variable.description or "") != str(description or ""):
            mismatched.append("concept_descriptions")
            break
    if mismatched:
        raise RunInputIdentityError(
            "Cannot adopt legacy resume because the requested scientific identity "
            "differs from sealed context fields: " + ", ".join(sorted(set(mismatched)))
        )

    staged_cohort = run_dir / "cohort.parquet"
    if not staged_cohort.is_file() or staged_cohort.is_symlink():
        raise RunInputIdentityError(
            "Cannot adopt legacy resume: staged cohort is missing."
        )
    raw_records = provenance.get("records") if isinstance(provenance, dict) else None
    cohort_records = [
        record
        for record in (raw_records or [])
        if isinstance(record, dict)
        and str(record.get("role") or "") == "cohort"
        and str(record.get("relative_path") or "") == "cohort.parquet"
    ]
    if len(cohort_records) != 1:
        raise RunInputIdentityError(
            "Cannot adopt legacy resume: provenance has no unique cohort authority."
        )
    expected_cohort_sha = str(cohort_records[0].get("sha256") or "")
    if len(expected_cohort_sha) != 64 or sha256_of_file(staged_cohort) != expected_cohort_sha:
        raise RunInputIdentityError(
            "Cannot adopt legacy resume: staged cohort no longer matches its "
            "original provenance digest."
        )
    incoming_frame = _cohort_frame(cohort)
    staged_frame = pd.read_parquet(staged_cohort)
    if _dataframe_content_sha256(incoming_frame) != _dataframe_content_sha256(
        staged_frame
    ):
        raise RunInputIdentityError(
            "Cannot adopt legacy resume: incoming cohort values differ from the "
            "sealed staged cohort."
        )

    existing_experiment = records.get("experiment_spec")
    requested_experiment = scientific_identity.get("experiment_spec")
    if bool(existing_experiment) != bool(requested_experiment):
        raise RunInputIdentityError(
            "Cannot adopt legacy resume: experiment specification presence changed."
        )
    experiment_path: Optional[Path] = None
    if existing_experiment is not None:
        experiment_path = _verified_record_path(
            run_dir=run_dir,
            records=records,
            evidence_id="experiment_spec",
        )
        try:
            import yaml

            stored_experiment = yaml.safe_load(
                experiment_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError) as exc:
            raise RunInputIdentityError(
                "Cannot adopt legacy resume: experiment spec evidence is invalid."
            ) from exc
        if _jsonable(stored_experiment) != requested_experiment:
            raise RunInputIdentityError(
                "Cannot adopt legacy resume: experiment specification changed."
            )

    evidence = EvidenceStore(root=run_dir, enforcement_mode=enforcement_mode)
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=scientific_identity,
        initial_environment=initial_environment,
        context_path=context_path,
        cohort_path=staged_cohort,
        experiment_spec_path=(
            run_dir / "experiment_spec.yaml" if experiment_path is not None else None
        ),
        legacy_adopted=True,
    )
    return load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=scientific_identity,
    )


def load_verified_run_input_capsule(
    *,
    run_dir: Path,
    scientific_identity: Dict[str, Any],
) -> ResumeInputAuthority:
    """Verify capsule and immutable inputs without writing to the run."""

    run_dir = Path(run_dir).expanduser().resolve()
    records = _records_from_index(run_dir)
    capsule_evidence_path = _verified_record_path(
        run_dir=run_dir,
        records=records,
        evidence_id=RUN_INPUT_CAPSULE_EVIDENCE_ID,
    )
    capsule_path = run_dir / RUN_INPUT_CAPSULE_FILENAME
    if not capsule_path.is_file() or capsule_path.is_symlink():
        raise RunInputIdentityError(
            "Cannot resume safely: immutable run_input_capsule.json is missing."
        )
    capsule_record = records[RUN_INPUT_CAPSULE_EVIDENCE_ID]
    if sha256_of_file(capsule_path) != str(capsule_record.get("sha256")):
        raise RunInputIdentityError(
            "Cannot resume safely: run_input_capsule.json was modified."
        )
    if capsule_path.read_bytes() != capsule_evidence_path.read_bytes():
        raise RunInputIdentityError(
            "Cannot resume safely: capsule working copy differs from sealed evidence."
        )
    try:
        capsule = RunInputCapsule.model_validate_json(
            capsule_evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as exc:
        raise RunInputIdentityError(
            "Cannot resume safely: run input capsule is invalid."
        ) from exc
    if canonical_sha256(capsule.scientific_identity) != (
        capsule.scientific_identity_sha256
    ):
        raise RunInputIdentityError(
            "Cannot resume safely: capsule scientific identity digest is invalid."
        )
    if capsule.scientific_identity != scientific_identity:
        keys = sorted(
            key
            for key in set(capsule.scientific_identity) | set(scientific_identity)
            if capsule.scientific_identity.get(key) != scientific_identity.get(key)
        )
        raise RunInputIdentityError(
            "Resume request belongs to a different scientific input identity; "
            f"changed fields: {', '.join(keys) or 'unknown'}."
        )

    context_evidence_path = _verified_record_path(
        run_dir=run_dir,
        records=records,
        evidence_id=capsule.context_evidence_id,
        expected_sha256=capsule.context_sha256,
    )
    try:
        ResearchContext.model_validate_json(
            context_evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as exc:
        raise RunInputIdentityError(
            "Cannot resume safely: sealed research context is invalid."
        ) from exc

    cohort_path = run_dir / capsule.cohort_relative_path
    if (
        not cohort_path.is_file()
        or cohort_path.is_symlink()
        or sha256_of_file(cohort_path) != capsule.cohort_sha256
    ):
        raise RunInputIdentityError(
            "Cannot resume safely: staged cohort bytes are missing or changed."
        )
    if capsule.trajectory_relative_path is not None:
        trajectory_path = run_dir / capsule.trajectory_relative_path
        if (
            not trajectory_path.is_file()
            or trajectory_path.is_symlink()
            or sha256_of_file(trajectory_path) != capsule.trajectory_sha256
        ):
            raise RunInputIdentityError(
                "Cannot resume safely: staged trajectory bytes are missing or changed."
            )

    experiment_path: Optional[Path] = None
    if capsule.experiment_spec_evidence_id is not None:
        experiment_path = _verified_record_path(
            run_dir=run_dir,
            records=records,
            evidence_id=capsule.experiment_spec_evidence_id,
            expected_sha256=capsule.experiment_spec_sha256,
        )
    return ResumeInputAuthority(
        capsule=capsule,
        context_evidence_path=context_evidence_path,
        experiment_spec_evidence_path=experiment_path,
        evidence_records=records,
    )


def _evidence_closure_error(
    *,
    evidence_id: str,
    step_id: str,
    run_dir: Path,
    records: Mapping[str, Dict[str, Any]],
    visited: set[str],
    require_step_owner: bool = True,
) -> Optional[str]:
    if evidence_id in visited:
        return None
    visited.add(evidence_id)
    record = records.get(evidence_id)
    if record is None:
        return f"missing evidence record {evidence_id}"
    producer = str(record.get("produced_by_step") or "").strip()
    if require_step_owner and producer and producer != step_id:
        return f"evidence {evidence_id} belongs to step {producer}"
    if verified_run_evidence_path(run_dir, record) is None:
        return f"evidence {evidence_id} failed path/digest verification"
    dependencies = [
        str(value)
        for value in (record.get("inputs") or [])
        if str(value).strip()
    ]
    script_id = str(record.get("script_evidence_id") or "").strip()
    if script_id:
        dependencies.append(script_id)
    for dependency_id in dependencies:
        error = _evidence_closure_error(
            evidence_id=dependency_id,
            step_id=step_id,
            run_dir=run_dir,
            records=records,
            visited=visited,
            # Inputs legitimately come from upstream steps. Their producer may
            # differ, but their complete digest-bound closure remains required.
            require_step_owner=False,
        )
        if error is not None:
            return f"evidence {evidence_id} has invalid input {dependency_id}: {error}"
    return None


def invalidate_unverified_successful_steps(
    *,
    run_dir: Path,
    resume_state: Dict[str, Any],
    records: Mapping[str, Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Append fail-closed checkpoints for current successes with bad evidence."""

    state = dict(resume_state)
    history = [
        dict(record)
        for record in (resume_state.get("per_step_records") or [])
        if isinstance(record, Mapping)
    ]
    invalidated: Dict[str, str] = {}
    for record in current_step_records(history):
        if str(record.get("status") or "").strip().lower() != "ok":
            continue
        step_id = str(record.get("step_id") or "").strip()
        evidence_ids = [
            str(value)
            for value in (record.get("evidence_ids") or [])
            if str(value).strip()
        ]
        error = None
        if not evidence_ids:
            error = "successful checkpoint has no evidence_ids"
        else:
            for evidence_id in evidence_ids:
                error = _evidence_closure_error(
                    evidence_id=evidence_id,
                    step_id=step_id,
                    run_dir=run_dir,
                    records=records,
                    visited=set(),
                )
                if error is not None:
                    break
        if error is None:
            continue
        invalidated[step_id] = error
        history.append(
            {
                "step_id": step_id,
                "status": "resume_evidence_invalid",
                "resume_invalidation_reason": error,
                "evidence_ids": [],
            }
        )

    if invalidated:
        findings = list(resume_state.get("findings") or [])
        for step_id, reason in invalidated.items():
            findings.append(
                {
                    "validator": "resume_evidence_integrity",
                    "severity": "warning",
                    "message": (
                        f"Prior success for step {step_id} was invalidated before "
                        "resume because its evidence closure is no longer verifiable."
                    ),
                    "detail": {
                        "step_id": step_id,
                        "reason": reason,
                        "requires_reexecution": True,
                    },
                }
            )
        state["findings"] = findings
    state["per_step_records"] = history
    return state, invalidated


def write_resume_environment_receipt(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    capsule: RunInputCapsule,
    current_environment: Dict[str, Any],
    invalidated_step_ids: Sequence[str],
) -> Path:
    """Record one immutable resume attempt after scientific identity passes."""

    existing = sorted(Path(run_dir).glob("resume_environment_receipt_*.json"))
    sequence = 1
    if existing:
        try:
            sequence = max(int(path.stem.rsplit("_", 1)[-1]) for path in existing) + 1
        except ValueError:
            sequence = len(existing) + 1
    changed_fields = sorted(
        key
        for key in set(capsule.initial_environment) | set(current_environment)
        if capsule.initial_environment.get(key) != current_environment.get(key)
    )
    payload = {
        "schema_version": RESUME_ENVIRONMENT_SCHEMA_VERSION,
        "attempt_sequence": sequence,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scientific_identity_sha256": capsule.scientific_identity_sha256,
        "initial_environment": capsule.initial_environment,
        "current_environment": current_environment,
        "changed_fields": changed_fields,
        "environment_drift": bool(changed_fields),
        "invalidated_step_ids": sorted(set(invalidated_step_ids)),
        "audit_cache_authority": (
            "current_environment_hashes_only"
            if changed_fields
            else "unchanged_environment"
        ),
    }
    path = Path(run_dir) / f"resume_environment_receipt_{sequence:04d}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Digest-bound model/code/prompt/validator environment receipt for "
            "one resume attempt."
        ),
        source_path=path,
        evidence_id=f"resume_environment_receipt_{sequence:04d}",
        producer="pipeline",
        generation_mode="system",
        metadata={
            "environment_drift": bool(changed_fields),
            "changed_fields": changed_fields,
        },
    )
    return path


def prepare_existing_resume_input(
    *,
    run_dir: Path,
    resume_state: Dict[str, Any],
    scientific_identity: Dict[str, Any],
    current_environment: Dict[str, Any],
    cohort: Union[str, Path, pd.DataFrame],
    question: str,
    resume_from_step_id: Optional[str],
    enforcement_mode: Any,
    load_compatible_plan: Callable[..., Any],
) -> PreparedResumeInput:
    """Validate/adopt a checkpoint and create its environment receipt.

    All study-identity comparisons happen before the first write. A receipt is
    written only after the capsule, context, cohort, and current step evidence
    have been verified.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    capsule_path = run_dir / RUN_INPUT_CAPSULE_FILENAME
    prior_successes = current_successful_step_records(
        resume_state.get("per_step_records") or []
    )
    if capsule_path.is_file():
        authority = load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=scientific_identity,
        )
    elif prior_successes:
        prior_llm_signature = str(
            resume_state.get("llm_signature")
            or current_environment["llm_signature"]
        )
        legacy_prompt_files = dict(resume_state.get("prompt_pack_files") or {})
        legacy_environment = {
            **current_environment,
            "llm_signature": prior_llm_signature,
            "llm_signature_sha256": canonical_sha256(prior_llm_signature),
            "engine_code_sha256": "legacy_unknown",
            "validator_code_sha256": "legacy_unknown",
            "prompt_pack_version": str(
                resume_state.get("prompt_pack_version") or "legacy_unknown"
            ),
            "prompt_pack_files": legacy_prompt_files,
            "prompt_pack_sha256": canonical_sha256(legacy_prompt_files),
        }
        authority = adopt_verified_legacy_run_input_capsule(
            run_dir=run_dir,
            cohort=cohort,
            scientific_identity=scientific_identity,
            initial_environment=legacy_environment,
            enforcement_mode=enforcement_mode,
        )
    else:
        # An unstarted legacy checkpoint has no result evidence to mix. Keep its
        # verified plan, but establish context/cohort/capsule from the current
        # request exactly once in the ordinary fresh-input path.
        attempted_steps = current_step_records(
            resume_state.get("per_step_records") or []
        )
        if attempted_steps:
            raise RunInputIdentityError(
                "Cannot bootstrap a legacy resume after any step attempt without "
                "an immutable run input capsule; start a new run instead."
            )
        plan, _ = load_compatible_plan(run_dir=run_dir, resume_state=resume_state)
        if (
            plan is None
            or " ".join(plan.research_question.split())
            != " ".join(question.split())
        ):
            raise RunInputIdentityError(
                "Cannot bootstrap legacy resume: verified plan question does "
                "not match the requested study."
            )
        return PreparedResumeInput(
            resume_state=resume_state,
            input_verified=False,
            context_evidence_path=None,
            cohort_path=None,
            experiment_spec_path=None,
        )

    prepared_state, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=resume_state,
        records=authority.evidence_records,
    )
    if resume_from_step_id and invalidated:
        plan, _ = load_compatible_plan(
            run_dir=run_dir,
            resume_state=prepared_state,
        )
        if plan is None:
            raise RunInputIdentityError(
                "Cannot resume safely after evidence invalidation: no compatible "
                "verified plan remains."
            )
        order = {step.step_id: index for index, step in enumerate(plan.steps)}
        cut = order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in invalidated
            if cut is not None and order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after invalidated upstream evidence; resume "
                "at or before: " + ", ".join(earlier_invalid)
            )

    receipt_store = EvidenceStore(root=run_dir, enforcement_mode=enforcement_mode)
    receipt_path = write_resume_environment_receipt(
        run_dir=run_dir,
        evidence=receipt_store,
        capsule=authority.capsule,
        current_environment=current_environment,
        invalidated_step_ids=tuple(invalidated),
    )
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    prepared_state = {
        **prepared_state,
        "resume_environment_receipt_path": receipt_path.name,
        "resume_environment_drift": bool(receipt_payload.get("environment_drift")),
        "resume_environment_changed_fields": list(
            receipt_payload.get("changed_fields") or []
        ),
    }
    experiment_spec_path: Optional[Path] = None
    if authority.experiment_spec_evidence_path is not None:
        experiment_spec_path = run_dir / "experiment_spec.yaml"
        if (
            not experiment_spec_path.is_file()
            or sha256_of_file(experiment_spec_path)
            != sha256_of_file(authority.experiment_spec_evidence_path)
        ):
            shutil.copy2(
                authority.experiment_spec_evidence_path,
                experiment_spec_path,
            )
    return PreparedResumeInput(
        resume_state=prepared_state,
        input_verified=True,
        context_evidence_path=authority.context_evidence_path,
        cohort_path=run_dir / authority.capsule.cohort_relative_path,
        experiment_spec_path=experiment_spec_path,
    )


__all__ = [
    "RUN_INPUT_CAPSULE_FILENAME",
    "RUN_INPUT_CAPSULE_EVIDENCE_ID",
    "RunInputCapsule",
    "RunInputIdentityError",
    "ResumeInputAuthority",
    "PreparedResumeInput",
    "adopt_verified_legacy_run_input_capsule",
    "build_environment_identity",
    "build_scientific_identity",
    "canonical_sha256",
    "invalidate_unverified_successful_steps",
    "load_verified_run_input_capsule",
    "prepare_existing_resume_input",
    "seal_run_input_capsule",
    "write_resume_environment_receipt",
]
