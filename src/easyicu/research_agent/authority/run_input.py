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
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .filesystem import AnchoredDirectory, AuthorityFilesystemError
from .evidence_store import EvidenceStore, sha256_of_file
from .evidence_snapshot import (
    EvidenceAuthorityIntegrityError,
    load_current_evidence_snapshot,
)
from ..intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
    VerifiedMaterializedCohortAuthority,
    load_verified_materialized_cohort_authority,
    read_verified_materialized_cohort_table,
)
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
    TrajectoryConceptBinding,
    VerifiedLegacyTrajectoryCapsuleReceipt,
    VerifiedMaterializedTrajectoryAuthority,
    load_verified_materialized_trajectory_authority,
)
from ..providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files
from ..research_context.implementation_identity import metadata_implementation_identity
from .runtime_artifacts import (
    current_successful_step_records,
    current_step_records,
    verified_run_evidence_path,
)
from ..schema import ResearchContext, TimeWindow
from ..research_context.typed import (
    MaterializedResearchInputs,
    ResearchContextV2,
    binding_preserves_analysis_range,
    materialized_research_inputs_from_authority,
    parse_research_context_json,
)
from ..cohort_artifact_facts import observed_domain_for_series

RUN_INPUT_CAPSULE_FILENAME = "run_input_capsule.json"
RUN_INPUT_CAPSULE_EVIDENCE_ID = "run_input_capsule"
RUN_INPUT_CAPSULE_SCHEMA_VERSION = "easyicu.run_input_capsule/1"
RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2 = "easyicu.run_input_capsule/2"
RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3 = "easyicu.run_input_capsule/3"
RESUME_ENVIRONMENT_SCHEMA_VERSION = "easyicu.resume_environment_receipt/1"
_MAX_RUN_INPUT_CAPSULE_BYTES = 64 * 1024 * 1024


class RunInputIdentityError(ValueError):
    """The requested resume cannot be proven to be the same study."""


def _validate_v2_context_input_authority(
    context: ResearchContext,
    *,
    cohort_path: Path,
    cohort: VerifiedMaterializedCohortAuthority,
    trajectory: Optional[VerifiedMaterializedTrajectoryAuthority],
    allow_v1: bool,
    require_current_implementation: bool,
) -> None:
    """Join every V2 redundant fact to capsule-selected typed inputs."""

    if not isinstance(context, ResearchContextV2):
        if allow_v1:
            # Archived typed capsules may contain an already sealed V1 context.
            return
        raise RunInputIdentityError(
            "Fresh typed inputs require a ResearchContext v2 authority."
        )
    typed = context.materialized_inputs
    if typed.cohort.projection_scope != "full" or (
        typed.trajectory is not None and typed.trajectory.projection_scope != "full"
    ):
        raise RunInputIdentityError(
            "Sealed ResearchContext typed input authority must be full."
        )
    matching_file_bindings = tuple(
        item
        for item in cohort.sidecar.files
        if item.relative_path == cohort.authority.cohort_file
    )
    if len(matching_file_bindings) != 1:
        raise RunInputIdentityError("Typed cohort has no unique sidecar file binding.")
    file_binding = matching_file_bindings[0]
    if require_current_implementation:
        # The V2 typed binding closes lineage/unit/range facts.  Dtype and
        # observed values live only in the capsule-selected parquet bytes, and
        # source files live in the sealed derivation receipts.  Reconstruct all
        # three once here before the first seal; never trust the redundant
        # legacy descriptor fields as an independent authority.
        frame = read_verified_materialized_cohort_table(
            cohort_path,
            verified=cohort,
        ).to_pandas()
        descriptors = {item.name: item for item in context.variables}
        derivations = {
            item.output_column: item for item in cohort.authority.output_derivations
        }
        for column in cohort.authority.cohort_columns:
            descriptor = descriptors.get(column)
            if descriptor is None:
                raise RunInputIdentityError(
                    f"ResearchContext descriptor is absent for {column!r}."
                )
            derivation = derivations.get(column)
            expected_source_files = sorted(
                {source.file for source in derivation.sources}
                if derivation is not None
                else set()
            )
            expected_facts = {
                "dtype": str(frame[column].dtype),
                "observed_domain": observed_domain_for_series(frame[column]),
                "source_files": expected_source_files,
            }
            for field_name, expected_value in expected_facts.items():
                if getattr(descriptor, field_name) != expected_value:
                    raise RunInputIdentityError(
                        "ResearchContext descriptor artifact fact does not match "
                        f"the staged cohort: {column}.{field_name}"
                    )
    expected = materialized_research_inputs_from_authority(
        cohort=cohort,
        trajectory=trajectory,
    )
    if not require_current_implementation:
        # Resume must preserve the immutable implementation coordinates sealed
        # with the old context while recording current drift in the environment
        # receipt. All artifact/sidecar-derived facts are still reconstructed
        # and compared below.
        expected_payload = expected.model_dump(mode="python")
        expected_payload["cohort"][
            "metadata_projection_sha256"
        ] = typed.cohort.metadata_projection_sha256
        expected_payload["cohort"][
            "metadata_sidecar_sha256"
        ] = typed.cohort.metadata_sidecar_sha256
        expected_payload["cohort"]["icu_rules_sha256"] = typed.cohort.icu_rules_sha256
        expected_payload["cohort"][
            "metadata_implementation_bundle_sha256"
        ] = typed.cohort.metadata_implementation_bundle_sha256
        # ICU-rule drift must trigger deterministic revalidation, not make an
        # otherwise immutable run impossible to resume. Preserve only the
        # fallback ranges sealed by the old implementation. Explicit sidecar
        # ranges remain reconstructed from the staged authority and therefore
        # retain exact tamper detection.
        expected_bindings = expected_payload["cohort"]["column_bindings"]
        for column, source_binding in file_binding.columns.items():
            if (
                source_binding.metadata.analysis_plausibility_range is None
                and binding_preserves_analysis_range(source_binding)
            ):
                expected_bindings[column]["analysis_plausibility_range"] = (
                    typed.cohort.column_bindings[column].analysis_plausibility_range
                )
        if (
            trajectory is not None
            and typed.trajectory is not None
            and expected_payload.get("trajectory") is not None
        ):
            expected_ranges = expected_payload["trajectory"][
                "concept_analysis_plausibility_ranges"
            ]
            for concept, raw_binding in typed.trajectory.concept_bindings.items():
                parsed = TrajectoryConceptBinding.from_dict(raw_binding)
                if (
                    parsed.binding.metadata.analysis_plausibility_range is None
                    and binding_preserves_analysis_range(parsed.binding)
                ):
                    expected_ranges[concept] = (
                        typed.trajectory.concept_analysis_plausibility_ranges[concept]
                    )
        expected = MaterializedResearchInputs.model_validate(expected_payload)
    if typed != expected:
        raise RunInputIdentityError(
            "ResearchContext typed input facts do not match staged authority."
        )


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


class RunInputCapsuleV2(RunInputCapsule):
    """Fresh typed-input capsule with an exact staged authority selector."""

    schema_version: Literal["easyicu.run_input_capsule/2"] = (
        RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2
    )
    materialized_cohort_authority_required: Literal[True] = True
    materialized_cohort_authority_ref: Dict[str, Any]


class RunInputCapsuleV3(RunInputCapsuleV2):
    """Typed cohort plus exact staged trajectory authority."""

    schema_version: Literal["easyicu.run_input_capsule/3"] = (
        RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3
    )
    trajectory_relative_path: Literal["cohort_trajectory.parquet"]
    trajectory_sha256: str
    materialized_trajectory_authority_required: Literal[True]
    materialized_trajectory_authority_ref: Dict[str, Any]


RunInputCapsuleAuthority = Union[RunInputCapsule, RunInputCapsuleV2, RunInputCapsuleV3]


@dataclass(frozen=True)
class ResumeInputAuthority:
    """Verified paths and records returned without changing the run directory."""

    capsule: RunInputCapsuleAuthority
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
    trajectory_binding: Optional[StagedTrajectoryBinding]
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


def _scientific_trajectory_envelope(
    scientific_identity: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    top_level = scientific_identity.get("trajectory")
    cohort_identity = scientific_identity.get("cohort")
    nested = (
        cohort_identity.get("trajectory")
        if isinstance(cohort_identity, Mapping)
        else None
    )
    raw = top_level if top_level is not None else nested
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RunInputIdentityError("Scientific trajectory identity must be an object.")
    sha256 = raw.get("sha256")
    size = raw.get("size_bytes")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise RunInputIdentityError(
            "Scientific trajectory identity has invalid byte coordinates."
        )
    envelope = {"sha256": sha256, "size_bytes": size}
    if top_level is not None and nested is not None:
        if not isinstance(nested, Mapping):
            raise RunInputIdentityError(
                "Nested scientific trajectory identity must be an object."
            )
        nested_sha256 = nested.get("sha256")
        nested_size = nested.get("size_bytes")
        if nested_sha256 != sha256 or nested_size != size:
            raise RunInputIdentityError(
                "Scientific trajectory identities select conflicting bytes."
            )
    return envelope


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


def _source_file_identities(
    source_files: Optional[Sequence[Any]],
) -> list[Dict[str, Any]]:
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
    materialized_cohort_authority_ref: Optional[Mapping[str, Any]] = None,
    trajectory_path: Optional[Path] = None,
    materialized_trajectory_authority_ref: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical scientific request; execution-only knobs are excluded."""

    payload: Dict[str, Any] = {
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
    if materialized_cohort_authority_ref is not None:
        payload["materialized_cohort_authority_ref"] = dict(
            materialized_cohort_authority_ref
        )
    if trajectory_path is not None:
        raw_selected_trajectory = Path(trajectory_path).expanduser()
        if (
            raw_selected_trajectory.is_symlink()
            or not raw_selected_trajectory.is_file()
        ):
            raise RunInputIdentityError(
                "Scientific trajectory input is missing or unsafe."
            )
        selected_trajectory = raw_selected_trajectory.resolve(strict=True)
        canonical_sibling = None
        if isinstance(cohort, (str, Path)):
            cohort_path = Path(cohort).expanduser().resolve()
            canonical_sibling = cohort_path.with_name(
                f"{cohort_path.stem}_trajectory.parquet"
            )
        # Historical v1/v2 identities already bind the canonical sibling under
        # ``cohort.trajectory``.  Do not add a second top-level coordinate for
        # the same bytes; doing so would make archived capsules impossible to
        # resume.  A non-sibling trajectory needs its own coordinate.
        if canonical_sibling is None or selected_trajectory != canonical_sibling:
            # ``cohort_input_identity`` records an ambient canonical sibling for
            # historical v1/v2 compatibility.  When the caller explicitly
            # selects a different trajectory, that sibling is *not* a science
            # input and must not survive as a second conflicting coordinate.
            cohort_identity = payload.get("cohort")
            if isinstance(cohort_identity, Mapping) and "trajectory" in cohort_identity:
                selected_cohort_identity = dict(cohort_identity)
                selected_cohort_identity.pop("trajectory", None)
                payload["cohort"] = selected_cohort_identity
            payload["trajectory"] = {
                "format": selected_trajectory.suffix.lower(),
                "size_bytes": int(selected_trajectory.stat().st_size),
                "sha256": sha256_of_file(selected_trajectory),
            }
    if materialized_trajectory_authority_ref is not None:
        payload["materialized_trajectory_authority_ref"] = dict(
            materialized_trajectory_authority_ref
        )
    return _jsonable(payload)


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
    return canonical_sha256(
        {
            "research_agent_tree_sha256": _tree_sha256(
                Path(__file__).resolve().parents[1]
            ),
            "metadata_implementation": dict(metadata_implementation_identity()),
        }
    )


@lru_cache(maxsize=1)
def validator_code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "gates" / "preflight.py",
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
        **dict(metadata_implementation_identity()),
    }


def _records_from_index(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    try:
        snapshot = load_current_evidence_snapshot(Path(run_dir))
    except EvidenceAuthorityIntegrityError as exc:
        raise RunInputIdentityError(
            "Cannot resume safely: evidence authority is corrupt."
        ) from exc
    payload = list(snapshot.records)
    if not payload:
        raise RunInputIdentityError(
            "Cannot resume safely: the selected evidence authority has no records."
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
        raise RunInputIdentityError(
            "Run input capsule is immutable and already exists."
        )
    context_record = evidence.get("research_context")
    if context_record is None:
        raise RunInputIdentityError(
            "Cannot seal run input capsule without research_context evidence."
        )
    try:
        if sha256_of_file(context_path) != str(context_record.sha256):
            raise RunInputIdentityError(
                "ResearchContext working copy differs from sealed evidence."
            )
        sealed_context = parse_research_context_json(
            context_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as exc:
        raise RunInputIdentityError(
            "Cannot seal run input capsule with an invalid ResearchContext."
        ) from exc
    experiment_record = evidence.get("experiment_spec")
    trajectory_path = Path(run_dir) / "cohort_trajectory.parquet"
    trajectory_envelope = _scientific_trajectory_envelope(scientific_identity)
    if trajectory_path.is_file() != (trajectory_envelope is not None):
        raise RunInputIdentityError(
            "Scientific trajectory identity and staged trajectory presence differ."
        )
    if trajectory_envelope is not None and (
        sha256_of_file(trajectory_path) != trajectory_envelope["sha256"]
        or int(trajectory_path.stat().st_size) != trajectory_envelope["size_bytes"]
    ):
        raise RunInputIdentityError(
            "Staged trajectory bytes do not match scientific identity."
        )
    staged_materialized_authority = load_verified_materialized_cohort_authority(
        cohort_path
    )
    capsule_fields: Dict[str, Any] = {
        "scientific_identity": scientific_identity,
        "scientific_identity_sha256": canonical_sha256(scientific_identity),
        "context_sha256": str(context_record.sha256),
        "cohort_sha256": sha256_of_file(cohort_path),
        "trajectory_relative_path": (
            trajectory_path.name if trajectory_path.is_file() else None
        ),
        "trajectory_sha256": (
            sha256_of_file(trajectory_path) if trajectory_path.is_file() else None
        ),
        "experiment_spec_evidence_id": (
            str(experiment_record.evidence_id)
            if experiment_record is not None
            else None
        ),
        "experiment_spec_sha256": (
            str(experiment_record.sha256) if experiment_record is not None else None
        ),
        "experiment_spec_relative_path": (
            str(experiment_spec_path.relative_to(run_dir))
            if experiment_record is not None and experiment_spec_path is not None
            else None
        ),
        "initial_environment": initial_environment,
        "legacy_adopted": legacy_adopted,
    }
    staged_trajectory_authority = None
    if trajectory_path.is_file():
        try:
            staged_trajectory_authority = (
                load_verified_materialized_trajectory_authority(trajectory_path)
            )
        except MaterializedTrajectoryError as exc:
            raise RunInputIdentityError(
                "Staged trajectory authority is invalid."
            ) from exc
    if staged_materialized_authority is not None:
        raw_source_ref = scientific_identity.get("materialized_cohort_authority_ref")
        if not isinstance(raw_source_ref, Mapping):
            raise RunInputIdentityError(
                "Typed staged cohort is absent from scientific identity."
            )
        try:
            source_ref = MaterializedCohortAuthorityRef.from_dict(raw_source_ref)
        except (MaterializedMetadataError, TypeError, ValueError) as exc:
            raise RunInputIdentityError(
                "Typed scientific identity contains an invalid source authority."
            ) from exc
        if (
            staged_materialized_authority.authority.parent_authority_sha256
            != source_ref.sha256
        ):
            raise RunInputIdentityError(
                "Staged cohort does not descend from the scientific source authority."
            )
        raw_source_trajectory_ref = scientific_identity.get(
            "materialized_trajectory_authority_ref"
        )
        if trajectory_path.is_file() and staged_trajectory_authority is None:
            raise RunInputIdentityError(
                "Typed staged trajectory lacks a sealed authority."
            )
        if staged_trajectory_authority is not None:
            if not isinstance(raw_source_trajectory_ref, Mapping):
                raise RunInputIdentityError(
                    "Typed staged trajectory is absent from scientific identity."
                )
            try:
                source_trajectory_ref = MaterializedTrajectoryAuthorityRef.from_dict(
                    raw_source_trajectory_ref
                )
            except (MaterializedTrajectoryError, TypeError, ValueError) as exc:
                raise RunInputIdentityError(
                    "Typed scientific identity contains an invalid trajectory "
                    "authority."
                ) from exc
            try:
                staged_trajectory_authority = (
                    load_verified_materialized_trajectory_authority(
                        trajectory_path,
                        expected_authority=staged_trajectory_authority.reference,
                        expected_universe_authority=(
                            staged_materialized_authority.reference
                        ),
                        expected_parent_universe_authority=source_ref,
                    )
                )
            except MaterializedTrajectoryError as exc:
                raise RunInputIdentityError(
                    "Staged trajectory parent does not match its scientific cohort."
                ) from exc
            if staged_trajectory_authority is None:  # pragma: no cover
                raise RunInputIdentityError(
                    "Staged trajectory authority disappeared during sealing."
                )
            if (
                staged_trajectory_authority.authority.parent_trajectory_authority
                != source_trajectory_ref
                or staged_trajectory_authority.authority.bound_universe_authority
                != staged_materialized_authority.reference
                or staged_trajectory_authority.authority.trajectory_file
                != trajectory_path.name
                or staged_trajectory_authority.authority.trajectory_sha256
                != capsule_fields["trajectory_sha256"]
            ):
                raise RunInputIdentityError(
                    "Staged trajectory does not close its source/cohort authority."
                )
            _validate_v2_context_input_authority(
                sealed_context,
                cohort_path=cohort_path,
                cohort=staged_materialized_authority,
                trajectory=staged_trajectory_authority,
                allow_v1=legacy_adopted,
                require_current_implementation=True,
            )
            capsule = RunInputCapsuleV3(
                **capsule_fields,
                materialized_trajectory_authority_required=True,
                materialized_cohort_authority_ref=(
                    staged_materialized_authority.reference.to_dict()
                ),
                materialized_trajectory_authority_ref=(
                    staged_trajectory_authority.reference.to_dict()
                ),
            )
        else:
            if raw_source_trajectory_ref is not None:
                raise RunInputIdentityError(
                    "Typed scientific identity trajectory was not staged."
                )
            _validate_v2_context_input_authority(
                sealed_context,
                cohort_path=cohort_path,
                cohort=staged_materialized_authority,
                trajectory=None,
                allow_v1=legacy_adopted,
                require_current_implementation=True,
            )
            capsule = RunInputCapsuleV2(
                **capsule_fields,
                materialized_cohort_authority_ref=(
                    staged_materialized_authority.reference.to_dict()
                ),
            )
    else:
        if staged_trajectory_authority is not None:
            raise RunInputIdentityError(
                "Typed staged trajectory cannot bind an untyped cohort."
            )
        if scientific_identity.get("materialized_trajectory_authority_ref") is not None:
            raise RunInputIdentityError(
                "Typed trajectory source authority cannot bind an untyped cohort."
            )
        capsule = RunInputCapsule(**capsule_fields)
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
        context = parse_research_context_json(context_path.read_text(encoding="utf-8"))
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
        if variable is None or str(variable.description or "") != str(
            description or ""
        ):
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
    if (
        len(expected_cohort_sha) != 64
        or sha256_of_file(staged_cohort) != expected_cohort_sha
    ):
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
        capsule_raw = capsule_evidence_path.read_text(encoding="utf-8")
        capsule_payload = json.loads(capsule_raw)
        if not isinstance(capsule_payload, Mapping):
            raise ValueError("capsule must be an object")
        schema_version = capsule_payload.get("schema_version")
        if schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION:
            capsule: RunInputCapsuleAuthority = RunInputCapsule.model_validate(
                capsule_payload
            )
        elif schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2:
            capsule = RunInputCapsuleV2.model_validate(capsule_payload)
        elif schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3:
            capsule = RunInputCapsuleV3.model_validate(capsule_payload)
        else:
            raise ValueError("unsupported run input capsule schema")
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
    trajectory_envelope = _scientific_trajectory_envelope(capsule.scientific_identity)
    if (capsule.trajectory_relative_path is not None) != (
        trajectory_envelope is not None
    ):
        raise RunInputIdentityError(
            "Cannot resume safely: trajectory identity and capsule presence differ."
        )
    if trajectory_envelope is not None and (
        capsule.trajectory_sha256 != trajectory_envelope["sha256"]
    ):
        raise RunInputIdentityError(
            "Cannot resume safely: trajectory capsule digest is inconsistent."
        )

    context_evidence_path = _verified_record_path(
        run_dir=run_dir,
        records=records,
        evidence_id=capsule.context_evidence_id,
        expected_sha256=capsule.context_sha256,
    )
    try:
        sealed_context = parse_research_context_json(
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
    if isinstance(capsule, RunInputCapsuleV2):
        try:
            expected_staged_ref = MaterializedCohortAuthorityRef.from_dict(
                capsule.materialized_cohort_authority_ref
            )
            staged_authority = load_verified_materialized_cohort_authority(
                cohort_path,
                expected_authority=expected_staged_ref,
            )
        except (MaterializedMetadataError, TypeError, ValueError) as exc:
            raise RunInputIdentityError(
                "Cannot resume safely: staged cohort authority is missing or changed."
            ) from exc
        if staged_authority is None:  # pragma: no cover - exact ref forbids legacy
            raise RunInputIdentityError(
                "Cannot resume safely: staged cohort authority is missing."
            )
        raw_source_ref = capsule.scientific_identity.get(
            "materialized_cohort_authority_ref"
        )
        if not isinstance(raw_source_ref, Mapping):
            raise RunInputIdentityError(
                "Cannot resume safely: typed source authority is absent."
            )
        try:
            source_ref = MaterializedCohortAuthorityRef.from_dict(raw_source_ref)
        except (MaterializedMetadataError, TypeError, ValueError) as exc:
            raise RunInputIdentityError(
                "Cannot resume safely: typed source authority is invalid."
            ) from exc
        if staged_authority.authority.parent_authority_sha256 != source_ref.sha256:
            raise RunInputIdentityError(
                "Cannot resume safely: staged cohort has the wrong source authority."
            )
    if capsule.trajectory_relative_path is not None:
        if (
            Path(capsule.trajectory_relative_path).name
            != capsule.trajectory_relative_path
        ):
            raise RunInputIdentityError(
                "Cannot resume safely: trajectory path escapes the run root."
            )
        trajectory_path = run_dir / capsule.trajectory_relative_path
        if (
            not trajectory_path.is_file()
            or trajectory_path.is_symlink()
            or sha256_of_file(trajectory_path) != capsule.trajectory_sha256
            or int(trajectory_path.stat().st_size) != trajectory_envelope["size_bytes"]
        ):
            raise RunInputIdentityError(
                "Cannot resume safely: staged trajectory bytes are missing or changed."
            )
    staged_trajectory: Optional[VerifiedMaterializedTrajectoryAuthority] = None
    if isinstance(capsule, RunInputCapsuleV3):
        if capsule.trajectory_relative_path is None:
            raise RunInputIdentityError(
                "Cannot resume safely: typed trajectory path is absent."
            )
        try:
            staged_trajectory_ref = MaterializedTrajectoryAuthorityRef.from_dict(
                capsule.materialized_trajectory_authority_ref
            )
            staged_trajectory = load_verified_materialized_trajectory_authority(
                run_dir / capsule.trajectory_relative_path,
                expected_authority=staged_trajectory_ref,
                expected_universe_authority=MaterializedCohortAuthorityRef.from_dict(
                    capsule.materialized_cohort_authority_ref
                ),
                expected_parent_universe_authority=source_ref,
            )
            raw_source_trajectory_ref = capsule.scientific_identity.get(
                "materialized_trajectory_authority_ref"
            )
            if not isinstance(raw_source_trajectory_ref, Mapping):
                raise MaterializedTrajectoryError(
                    "typed source trajectory authority is absent"
                )
            source_trajectory_ref = MaterializedTrajectoryAuthorityRef.from_dict(
                raw_source_trajectory_ref
            )
        except (
            MaterializedMetadataError,
            MaterializedTrajectoryError,
            TypeError,
            ValueError,
        ) as exc:
            raise RunInputIdentityError(
                "Cannot resume safely: staged trajectory authority is invalid."
            ) from exc
        if (
            staged_trajectory is None
            or staged_trajectory.authority.parent_trajectory_authority
            != source_trajectory_ref
            or staged_trajectory.authority.trajectory_sha256
            != capsule.trajectory_sha256
            or staged_trajectory.authority.bound_universe_file
            != capsule.cohort_relative_path
        ):
            raise RunInputIdentityError(
                "Cannot resume safely: staged trajectory has the wrong authority."
            )

    if isinstance(capsule, RunInputCapsuleV2):
        try:
            _validate_v2_context_input_authority(
                sealed_context,
                cohort_path=cohort_path,
                cohort=staged_authority,
                trajectory=staged_trajectory,
                allow_v1=True,
                require_current_implementation=False,
            )
        except (
            MaterializedMetadataError,
            MaterializedTrajectoryError,
            ValueError,
        ) as exc:
            raise RunInputIdentityError(
                "Cannot resume safely: ResearchContext typed input authority is invalid."
            ) from exc

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
    if require_step_owner and producer != step_id:
        owner = producer or "<run-level>"
        return f"evidence {evidence_id} belongs to step {owner}"
    if verified_run_evidence_path(run_dir, record) is None:
        return f"evidence {evidence_id} failed path/digest verification"
    dependencies = [
        str(value) for value in (record.get("inputs") or []) if str(value).strip()
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


_REQUIRED_STEP_AUTHORITY_KINDS = {
    "step_summary_evidence_id",
    "script_evidence_id",
}
_STEP_AUTHORITY_EXPECTED_KINDS = {
    "step_summary_evidence_id": "statistic",
    "script_evidence_id": "code",
    "interpretation_evidence_id": "log",
}
_RESUME_AUTHORITY_MIGRATION_SCHEMA_VERSION = "easyicu.resume_step_authority_migration/1"
_HOST_PROBE_STEP_ID = "00_probe"
_HOST_PROBE_AUTHORITY_KIND = "host_deterministic_probe"
_HOST_PROBE_AUTHORITIES = {
    "probe_summary_evidence_id": ("statistic", "probe_summary.json"),
    "probe_table_evidence_id": ("table", "probe_variable_profile.csv"),
}
_HOST_COHORT_MATERIALIZER_GENERATION_MODE = "deterministic_cohort_materializer"
_HOST_COHORT_MATERIALIZER_AUTHORITY_KIND = "host_deterministic_cohort_materializer"
_HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD = "cohort_table_evidence_id"
_HOST_COHORT_MATERIALIZER_EVIDENCE_ID = "analysis_cohort_execute_repair"
_HOST_COHORT_MATERIALIZER_SOURCE_NAME = "cohort_analysis.parquet"


def _registered_source_name(
    record: Mapping[str, Any],
) -> Optional[str]:
    """Recover ``filename`` from EvidenceStore's ``<id>__<filename>`` path."""

    evidence_id = str(record.get("evidence_id") or "").strip()
    relative_path = str(record.get("relative_path") or "").strip()
    if not evidence_id or not relative_path:
        return None
    name = Path(relative_path).name
    prefix = f"{evidence_id}__"
    if not name.startswith(prefix):
        return None
    source_name = name[len(prefix) :]
    return source_name or None


def _host_probe_authority_error(
    *,
    record: Mapping[str, Any],
    evidence_ids: Sequence[str],
    step_id: str,
    run_dir: Path,
    records: Mapping[str, Dict[str, Any]],
) -> Optional[str]:
    """Validate the script-free, host-owned deterministic probe checkpoint."""

    if record.get("step_authority_kind") != _HOST_PROBE_AUTHORITY_KIND:
        return "successful host probe checkpoint lacks migrated probe authority"
    listed = set(evidence_ids)
    for field, (expected_kind, expected_source_name) in _HOST_PROBE_AUTHORITIES.items():
        evidence_id = str(record.get(field) or "").strip()
        if not evidence_id:
            return f"successful host probe checkpoint is missing required {field}"
        if evidence_id not in listed:
            return (
                f"successful host probe {field} {evidence_id} is absent from "
                "evidence_ids"
            )
        authority = records.get(evidence_id)
        if not isinstance(authority, Mapping):
            return f"successful host probe {field} references missing {evidence_id}"
        if str(authority.get("produced_by_step") or "").strip() != step_id:
            return f"successful host probe {field} is not owned by step {step_id}"
        if str(authority.get("kind") or "").strip().lower() != expected_kind:
            return (
                f"successful host probe {field} has wrong evidence kind; "
                f"expected {expected_kind}"
            )
        if (
            str(authority.get("producer") or "").strip().lower() != "pipeline"
            or str(authority.get("generation_mode") or "").strip().lower()
            != "deterministic_probe"
        ):
            return f"successful host probe {field} is not host-owned probe evidence"
        if _registered_source_name(authority) != expected_source_name:
            return (
                f"successful host probe {field} does not name "
                f"{expected_source_name}"
            )
        error = _evidence_closure_error(
            evidence_id=evidence_id,
            step_id=step_id,
            run_dir=run_dir,
            records=records,
            visited=set(),
        )
        if error is not None:
            return error
    return None


def _host_cohort_materializer_authority_error(
    *,
    record: Mapping[str, Any],
    evidence_ids: Sequence[str],
    step_id: str,
    run_dir: Path,
    records: Mapping[str, Dict[str, Any]],
) -> Optional[str]:
    """Validate the single-product, script-free cohort materializer.

    This is deliberately a separate closed contract from the probe and from
    ordinary deterministic records.  The host only owns the mechanical
    materialization of the Agent-selected cohort; the exact registered table
    remains the sole authority for that planned producer step.
    """

    prefix = "successful host cohort materializer"
    if (
        str(record.get("generation_mode") or "").strip().lower()
        != _HOST_COHORT_MATERIALIZER_GENERATION_MODE
        or record.get("step_authority_kind") != _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
    ):
        return f"{prefix} checkpoint lacks migrated cohort authority"

    evidence_id = str(
        record.get(_HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD) or ""
    ).strip()
    if evidence_id != _HOST_COHORT_MATERIALIZER_EVIDENCE_ID:
        return (
            f"{prefix} checkpoint is missing exact "
            f"{_HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD}"
        )
    listed = [str(value).strip() for value in evidence_ids if str(value).strip()]
    if listed != [evidence_id]:
        return f"{prefix} checkpoint must list only its cohort table authority"

    step_summary = record.get("step_summary")
    if not isinstance(step_summary, Mapping):
        return f"{prefix} checkpoint lacks its inline product receipt"
    output_files = step_summary.get("output_files")
    if not isinstance(output_files, Mapping) or dict(output_files) != {
        "table:analysis_cohort": _HOST_COHORT_MATERIALIZER_SOURCE_NAME
    }:
        return f"{prefix} checkpoint does not declare the analysis cohort product"
    n_universe = step_summary.get("n_universe")
    n_cohort = step_summary.get("n_analysis_cohort")
    if (
        not isinstance(n_universe, int)
        or isinstance(n_universe, bool)
        or not isinstance(n_cohort, int)
        or isinstance(n_cohort, bool)
        or n_universe < 0
        or n_cohort < 0
        or n_cohort > n_universe
    ):
        return f"{prefix} checkpoint has invalid cohort accounting"

    authority = records.get(evidence_id)
    if not isinstance(authority, Mapping):
        return f"{prefix} checkpoint references missing {evidence_id}"
    if str(authority.get("evidence_id") or "").strip() != evidence_id:
        return f"{prefix} authority has a mismatched evidence identity"
    if str(authority.get("produced_by_step") or "").strip() != step_id:
        return f"{prefix} authority is not owned by step {step_id}"
    if str(authority.get("kind") or "").strip().lower() != "table":
        return f"{prefix} authority is not a table"
    if (
        str(authority.get("producer") or "").strip().lower() != "cohort_repair"
        or str(authority.get("generation_mode") or "").strip().lower() != "llm"
    ):
        return f"{prefix} authority is not the host cohort-repair product"
    if _registered_source_name(authority) != _HOST_COHORT_MATERIALIZER_SOURCE_NAME:
        return f"{prefix} authority does not name the canonical cohort product"
    if str(authority.get("script_evidence_id") or "").strip() or list(
        authority.get("inputs") or []
    ):
        return f"{prefix} authority has an unexpected executable dependency"
    metadata = authority.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or not str(metadata.get("reason") or "").strip()
    ):
        return f"{prefix} authority lacks its materialization reason"

    closure_error = _evidence_closure_error(
        evidence_id=evidence_id,
        step_id=step_id,
        run_dir=run_dir,
        records=records,
        visited=set(),
    )
    if closure_error is not None:
        return closure_error

    canonical_path = run_dir / _HOST_COHORT_MATERIALIZER_SOURCE_NAME
    try:
        resolved_root = run_dir.resolve(strict=True)
        if canonical_path.is_symlink() or not canonical_path.is_file():
            return f"{prefix} canonical cohort is missing or not a regular file"
        resolved_canonical = canonical_path.resolve(strict=True)
        resolved_canonical.relative_to(resolved_root)
        expected_digest = str(authority.get("sha256") or "").strip().lower()
        if sha256_of_file(resolved_canonical).lower() != expected_digest:
            return f"{prefix} canonical cohort differs from sealed evidence"
        try:
            import pyarrow.parquet as pq  # type: ignore

            actual_rows = int(pq.ParquetFile(resolved_canonical).metadata.num_rows)
        except ImportError:
            actual_rows = int(len(pd.read_parquet(resolved_canonical)))
    except (FileNotFoundError, OSError, ValueError):
        return f"{prefix} canonical cohort failed path verification"
    except Exception as exc:
        return (
            f"{prefix} canonical cohort row count is unreadable: "
            f"{type(exc).__name__}"
        )
    if actual_rows != n_cohort:
        return (
            f"{prefix} canonical cohort row count {actual_rows} does not match "
            f"checkpoint {n_cohort}"
        )

    raw_materialized_ref = step_summary.get("materialized_cohort_authority_ref")
    cohort_definition_sha256 = str(
        step_summary.get("cohort_definition_sha256") or ""
    ).strip()
    metadata_ref = (
        metadata.get("materialized_cohort_authority_ref")
        if isinstance(metadata, Mapping)
        else None
    )
    metadata_definition_sha256 = str(
        metadata.get("cohort_definition_sha256") or ""
        if isinstance(metadata, Mapping)
        else ""
    ).strip()
    from ..cohort_schema import (
        CohortSchemaError,
        _load_locked_cohort_definition,
        cohort_definition_sha,
    )

    try:
        from ..intake.materialized_metadata import (
            MaterializedCohortAuthorityRef,
            MaterializedMetadataError,
            load_verified_materialized_cohort_authority,
        )

        if raw_materialized_ref is not None:
            if not isinstance(raw_materialized_ref, Mapping):
                return f"{prefix} typed authority reference is not an object"
            typed_ref = MaterializedCohortAuthorityRef.from_dict(raw_materialized_ref)
            if metadata_ref != raw_materialized_ref:
                return f"{prefix} evidence does not bind the typed authority reference"
            if (
                not cohort_definition_sha256
                or metadata_definition_sha256 != cohort_definition_sha256
            ):
                return f"{prefix} typed cohort definition digest is not sealed"
            locked_definition_sha256 = cohort_definition_sha(
                _load_locked_cohort_definition(run_dir)
            )
            if locked_definition_sha256 != cohort_definition_sha256:
                return f"{prefix} typed cohort does not match the locked definition"
            verified_materialized = load_verified_materialized_cohort_authority(
                canonical_path,
                expected_authority=typed_ref,
            )
            if verified_materialized is None:  # pragma: no cover - expected ref
                return f"{prefix} typed cohort authority is missing"
            typed_authority = verified_materialized.authority
            if (
                typed_authority.producer != "analysis_cohort_ordered_subset"
                or typed_authority.cohort_rows != n_cohort
                or typed_authority.cohort_sha256 != expected_digest
                or typed_authority.producer_parameters.get("cohort_definition_sha256")
                != cohort_definition_sha256
                or typed_authority.semantic_provenance.get("cohort_sha256")
                != cohort_definition_sha256
            ):
                return f"{prefix} typed cohort authority does not match the receipt"
        else:
            verified_materialized = load_verified_materialized_cohort_authority(
                canonical_path
            )
            if verified_materialized is not None:
                return f"{prefix} checkpoint omits its typed cohort authority"
            if metadata_ref is not None or metadata_definition_sha256:
                return f"{prefix} legacy receipt contains partial typed authority"
    except (MaterializedMetadataError, CohortSchemaError) as exc:
        return f"{prefix} typed cohort authority is invalid: {exc}"
    return None


def _migrated_legacy_step_authority(
    *,
    record: Mapping[str, Any],
    run_dir: Path,
    records: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return one proven legacy-authority checkpoint, never a guessed one.

    Historical executable checkpoints already listed the exact code evidence,
    and their explicit summary evidence named that code in
    ``script_evidence_id``.  The outer checkpoint merely lacked the duplicated
    convenience field.  Migrate only that closed chain.  The deterministic
    probe and cohort materializer are script-free and therefore use their own
    exact product contracts.
    """

    if str(record.get("status") or "").strip().lower() != "ok":
        return None
    step_id = str(record.get("step_id") or "").strip()
    evidence_ids = [
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    ]
    if not step_id or not evidence_ids:
        return None

    if step_id == _HOST_PROBE_STEP_ID:
        if record.get("step_authority_kind") == _HOST_PROBE_AUTHORITY_KIND:
            return None
        migrated_fields: Dict[str, str] = {}
        for field, (
            expected_kind,
            expected_source_name,
        ) in _HOST_PROBE_AUTHORITIES.items():
            candidates: list[str] = []
            for evidence_id in evidence_ids:
                authority = records.get(evidence_id)
                if not isinstance(authority, Mapping):
                    continue
                if (
                    str(authority.get("produced_by_step") or "").strip() != step_id
                    or str(authority.get("kind") or "").strip().lower() != expected_kind
                    or str(authority.get("producer") or "").strip().lower()
                    != "pipeline"
                    or str(authority.get("generation_mode") or "").strip().lower()
                    != "deterministic_probe"
                    or _registered_source_name(authority) != expected_source_name
                ):
                    continue
                if (
                    _evidence_closure_error(
                        evidence_id=evidence_id,
                        step_id=step_id,
                        run_dir=run_dir,
                        records=records,
                        visited=set(),
                    )
                    is None
                ):
                    candidates.append(evidence_id)
            if len(candidates) != 1:
                return None
            migrated_fields[field] = candidates[0]
        migrated = dict(record)
        migrated.update(migrated_fields)
        migrated.update(
            {
                "step_authority_kind": _HOST_PROBE_AUTHORITY_KIND,
                "resume_authority_migration_schema_version": (
                    _RESUME_AUTHORITY_MIGRATION_SCHEMA_VERSION
                ),
                "resume_authority_migrated_fields": sorted(migrated_fields),
            }
        )
        return migrated

    if (
        str(record.get("generation_mode") or "").strip().lower()
        == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
    ):
        if (
            record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        ):
            return None
        migrated = dict(record)
        migrated.update(
            {
                "step_authority_kind": (_HOST_COHORT_MATERIALIZER_AUTHORITY_KIND),
                _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD: (
                    _HOST_COHORT_MATERIALIZER_EVIDENCE_ID
                ),
            }
        )
        if (
            _host_cohort_materializer_authority_error(
                record=migrated,
                evidence_ids=evidence_ids,
                step_id=step_id,
                run_dir=run_dir,
                records=records,
            )
            is not None
        ):
            return None
        migrated.update(
            {
                "resume_authority_migration_schema_version": (
                    _RESUME_AUTHORITY_MIGRATION_SCHEMA_VERSION
                ),
                "resume_authority_migrated_fields": [
                    _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD
                ],
            }
        )
        return migrated

    if str(record.get("script_evidence_id") or "").strip():
        return None
    summary_id = str(record.get("step_summary_evidence_id") or "").strip()
    if not summary_id or summary_id not in evidence_ids:
        return None
    summary = records.get(summary_id)
    if not isinstance(summary, Mapping):
        return None
    if (
        str(summary.get("produced_by_step") or "").strip() != step_id
        or str(summary.get("kind") or "").strip().lower() != "statistic"
        or _evidence_closure_error(
            evidence_id=summary_id,
            step_id=step_id,
            run_dir=run_dir,
            records=records,
            visited=set(),
        )
        is not None
    ):
        return None
    script_id = str(summary.get("script_evidence_id") or "").strip()
    if not script_id or script_id not in evidence_ids:
        return None
    active_same_step_codes = [
        evidence_id
        for evidence_id in evidence_ids
        if isinstance((authority := records.get(evidence_id)), Mapping)
        and str(authority.get("produced_by_step") or "").strip() == step_id
        and str(authority.get("kind") or "").strip().lower() == "code"
        and _evidence_closure_error(
            evidence_id=evidence_id,
            step_id=step_id,
            run_dir=run_dir,
            records=records,
            visited=set(),
        )
        is None
    ]
    if active_same_step_codes != [script_id]:
        return None
    migrated = dict(record)
    migrated.update(
        {
            "script_evidence_id": script_id,
            "resume_authority_migration_schema_version": (
                _RESUME_AUTHORITY_MIGRATION_SCHEMA_VERSION
            ),
            "resume_authority_migrated_fields": ["script_evidence_id"],
        }
    )
    return migrated


def _interpretation_authority_is_applicable(
    *,
    record: Mapping[str, Any],
    step_id: str,
    records: Mapping[str, Dict[str, Any]],
) -> bool:
    """Return whether this checkpoint completed an Analyzer interpretation.

    Current executor checkpoints name the interpretation explicitly.  Looking
    for the immutable analyzer-owned evidence as well prevents a mutable
    checkpoint from evading the requirement by deleting only that field.
    Legacy/system steps that never created analyzer evidence remain eligible
    for the two mandatory execution authorities below.
    """

    if "interpretation_evidence_id" in record:
        return True
    return any(
        str(candidate.get("produced_by_step") or "").strip() == step_id
        and str(candidate.get("producer") or "").strip().lower() == "analyzer"
        for candidate in records.values()
        if isinstance(candidate, Mapping)
    )


def _explicit_step_authority_error(
    *,
    record: Mapping[str, Any],
    evidence_ids: Sequence[str],
    step_id: str,
    run_dir: Path,
    records: Mapping[str, Dict[str, Any]],
) -> Optional[str]:
    """Require host-owned step authorities, not an arbitrary evidence blob.

    A successful executable step always has a machine-readable summary and the
    exact script that produced it.  Analyzer interpretation is additionally
    required when that stage ran.  Merely listing some digest-valid run-level
    evidence cannot substitute for those role-specific authorities.
    """

    if step_id == _HOST_PROBE_STEP_ID:
        return _host_probe_authority_error(
            record=record,
            evidence_ids=evidence_ids,
            step_id=step_id,
            run_dir=run_dir,
            records=records,
        )

    if (
        str(record.get("generation_mode") or "").strip().lower()
        == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
    ):
        return _host_cohort_materializer_authority_error(
            record=record,
            evidence_ids=evidence_ids,
            step_id=step_id,
            run_dir=run_dir,
            records=records,
        )

    listed = set(evidence_ids)
    required_fields = set(_REQUIRED_STEP_AUTHORITY_KINDS)
    if _interpretation_authority_is_applicable(
        record=record,
        step_id=step_id,
        records=records,
    ):
        required_fields.add("interpretation_evidence_id")

    for field in _STEP_AUTHORITY_EXPECTED_KINDS:
        if field not in required_fields and field not in record:
            continue
        raw_value = record.get(field)
        if not isinstance(raw_value, str) or not raw_value.strip():
            return f"successful checkpoint is missing required {field}"
        evidence_id = raw_value.strip()
        if evidence_id not in listed:
            return (
                f"successful checkpoint {field} {evidence_id} is absent from "
                "evidence_ids"
            )
        authority = records.get(evidence_id)
        if authority is None:
            return f"successful checkpoint {field} references missing {evidence_id}"
        owner = str(authority.get("produced_by_step") or "").strip()
        if owner != step_id:
            return (
                f"successful checkpoint {field} {evidence_id} is not owned by "
                f"step {step_id}"
            )
        expected_kind = _STEP_AUTHORITY_EXPECTED_KINDS[field]
        actual_kind = str(authority.get("kind") or "").strip().lower()
        if actual_kind != expected_kind:
            return (
                f"successful checkpoint {field} {evidence_id} has kind "
                f"{actual_kind or '<missing>'}, expected {expected_kind}"
            )
    return None


def _external_evidence_dependencies(
    *,
    record: Mapping[str, Any],
    step_id: str,
    records: Mapping[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Map upstream producer steps to evidence consumed by ``step_id``.

    Output records carry digest-bound ``inputs`` while the checkpoint also
    stores host-resolved typed inputs.  Inspect both and walk their closures so
    invalidation propagates even when the downstream bytes themselves remain
    intact.
    """

    roots = [
        str(value).strip()
        for value in (
            list(record.get("evidence_ids") or [])
            + list(record.get("resolved_input_evidence_ids") or [])
        )
        if str(value).strip()
    ]
    dependencies: Dict[str, str] = {}
    visited: set[str] = set()
    pending = list(dict.fromkeys(roots))
    while pending:
        evidence_id = pending.pop()
        if evidence_id in visited:
            continue
        visited.add(evidence_id)
        evidence_record = records.get(evidence_id)
        if not isinstance(evidence_record, Mapping):
            continue
        producer = str(evidence_record.get("produced_by_step") or "").strip()
        if producer and producer != step_id:
            dependencies.setdefault(producer, evidence_id)
            # This is the immediate authority edge. Its own upstream closure
            # belongs to that producer and will be propagated in the next
            # fixed-point iteration, preserving the actual A -> B -> C chain.
            continue
        nested = [
            str(value).strip()
            for value in (evidence_record.get("inputs") or [])
            if str(value).strip()
        ]
        script_id = str(evidence_record.get("script_evidence_id") or "").strip()
        if script_id:
            nested.append(script_id)
        pending.extend(nested)
    return dependencies


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
    # Current checkpoints written before the explicit authority fields were
    # introduced may still carry a complete, digest-bound authority chain.
    # Append a migration checkpoint only when that chain proves the missing
    # field exactly; the historical record remains immutable in the ledger.
    for record in list(current_step_records(history)):
        migrated = _migrated_legacy_step_authority(
            record=record,
            run_dir=run_dir,
            records=records,
        )
        if migrated is not None:
            history.append(migrated)
    current_records = list(current_step_records(history))
    successful_records = [
        record
        for record in current_records
        if str(record.get("status") or "").strip().lower() == "ok"
    ]
    invalidated: Dict[str, str] = {}
    for record in successful_records:
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
            error = _explicit_step_authority_error(
                record=record,
                evidence_ids=evidence_ids,
                step_id=step_id,
                run_dir=run_dir,
                records=records,
            )
            if error is None:
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

    # A digest-valid downstream file is not current authority when the step
    # that supplied one of its inputs has just been invalidated.  Iterate to a
    # fixed point so A -> B -> C invalidates both B and C in the same resume
    # preparation pass, even when only A's checkpoint metadata was damaged.
    previously_invalid = {
        str(record.get("step_id") or "").strip()
        for record in current_records
        if str(record.get("status") or "").strip().lower() == "resume_evidence_invalid"
        and str(record.get("step_id") or "").strip()
    }
    dependency_map = {
        str(record.get("step_id") or "").strip(): _external_evidence_dependencies(
            record=record,
            step_id=str(record.get("step_id") or "").strip(),
            records=records,
        )
        for record in successful_records
    }
    while True:
        changed = False
        unavailable_steps = previously_invalid | set(invalidated)
        for record in successful_records:
            step_id = str(record.get("step_id") or "").strip()
            if step_id in invalidated:
                continue
            dependencies = dependency_map.get(step_id, {})
            invalid_dependency = next(
                (
                    (producer_step, evidence_id)
                    for producer_step, evidence_id in dependencies.items()
                    if producer_step in unavailable_steps
                ),
                None,
            )
            if invalid_dependency is None:
                continue
            producer_step, evidence_id = invalid_dependency
            invalidated[step_id] = (
                "successful checkpoint depends on invalidated step "
                f"{producer_step} via evidence {evidence_id}"
            )
            changed = True
        if not changed:
            break

    for record in successful_records:
        step_id = str(record.get("step_id") or "").strip()
        error = invalidated.get(step_id)
        if error is None:
            continue
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


def verify_legacy_trajectory_capsule_receipt(
    *,
    run_dir: Path,
    trajectory_path: Path,
    receipt: VerifiedLegacyTrajectoryCapsuleReceipt,
    expected_universe_authority: MaterializedCohortAuthorityRef,
) -> tuple[str, int]:
    """Recheck the one compatibility exception for a typed v2 resume.

    A v2 capsule could seal a typed cohort while its trajectory was still a
    digest-bound raw sibling. Modern fresh runs may not create that shape, but
    an archived v2 run must remain executable after its full capsule has been
    verified. This check reads the explicitly selected capsule and never scans
    for an alternate authority.
    """

    if not isinstance(receipt, VerifiedLegacyTrajectoryCapsuleReceipt):
        raise RunInputIdentityError("legacy trajectory receipt is invalid")
    run_dir = Path(run_dir).expanduser().absolute()
    selected_trajectory = Path(trajectory_path).expanduser().absolute()
    expected_trajectory = run_dir / receipt.trajectory_relative_path
    try:
        records = _records_from_index(run_dir)
        capsule_record = records.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
        if (
            capsule_record is None
            or str(capsule_record.get("sha256") or "") != receipt.capsule_sha256
        ):
            raise RunInputIdentityError(
                "legacy trajectory receipt is not selected by evidence authority"
            )
        sealed_capsule_path = _verified_record_path(
            run_dir=run_dir,
            records=records,
            evidence_id=RUN_INPUT_CAPSULE_EVIDENCE_ID,
            expected_sha256=receipt.capsule_sha256,
        )
        with AnchoredDirectory.open(sealed_capsule_path.parent) as evidence_root:
            sealed_capsule_bytes = evidence_root.read_bytes(
                sealed_capsule_path.name,
                max_bytes=_MAX_RUN_INPUT_CAPSULE_BYTES,
                expected_sha256=receipt.capsule_sha256,
            )
            evidence_root.assert_still_selected()
        with AnchoredDirectory.open(run_dir) as run_root:
            capsule_bytes = run_root.read_bytes(
                RUN_INPUT_CAPSULE_FILENAME,
                max_bytes=_MAX_RUN_INPUT_CAPSULE_BYTES,
                expected_size=len(sealed_capsule_bytes),
                expected_sha256=receipt.capsule_sha256,
            )
            if capsule_bytes != sealed_capsule_bytes:
                raise RunInputIdentityError(
                    "legacy trajectory capsule differs from sealed evidence"
                )
            trajectory_bytes = run_root.read_bytes(
                receipt.trajectory_relative_path,
                max_bytes=max(receipt.trajectory_size, 1),
                expected_size=receipt.trajectory_size,
                expected_sha256=receipt.trajectory_sha256,
            )
            run_root.assert_still_selected()
        if len(trajectory_bytes) != receipt.trajectory_size:
            raise RunInputIdentityError(
                "legacy trajectory receipt size changed during verification"
            )
        records_after = _records_from_index(run_dir)
        if records_after.get(RUN_INPUT_CAPSULE_EVIDENCE_ID) != capsule_record:
            raise RunInputIdentityError(
                "legacy trajectory evidence selection changed during verification"
            )
        raw = json.loads(capsule_bytes.decode("utf-8"))
        capsule = RunInputCapsuleV2.model_validate(raw)
        trajectory_envelope = _scientific_trajectory_envelope(
            capsule.scientific_identity
        )
        universe_ref = MaterializedCohortAuthorityRef.from_dict(
            capsule.materialized_cohort_authority_ref
        )
        if (
            canonical_sha256(capsule.scientific_identity)
            != capsule.scientific_identity_sha256
        ):
            raise RunInputIdentityError(
                "legacy trajectory capsule scientific identity is invalid"
            )
        if universe_ref != expected_universe_authority:
            raise RunInputIdentityError(
                "legacy trajectory capsule selected a different staged universe"
            )
        staged_cohort = load_verified_materialized_cohort_authority(
            run_dir / capsule.cohort_relative_path,
            expected_authority=universe_ref,
        )
        raw_source_ref = capsule.scientific_identity.get(
            "materialized_cohort_authority_ref"
        )
        if staged_cohort is None or not isinstance(raw_source_ref, Mapping):
            raise RunInputIdentityError(
                "legacy trajectory capsule lost typed cohort lineage"
            )
        source_ref = MaterializedCohortAuthorityRef.from_dict(raw_source_ref)
        if (
            staged_cohort.authority.cohort_sha256 != capsule.cohort_sha256
            or staged_cohort.authority.parent_authority_sha256 != source_ref.sha256
        ):
            raise RunInputIdentityError(
                "legacy trajectory capsule has invalid typed cohort lineage"
            )
    except RunInputIdentityError:
        raise
    except (
        AuthorityFilesystemError,
        EvidenceAuthorityIntegrityError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        MaterializedMetadataError,
    ) as exc:
        raise RunInputIdentityError(
            "legacy trajectory receipt references an invalid v2 capsule"
        ) from exc
    if (
        selected_trajectory != expected_trajectory
        or capsule.trajectory_relative_path != receipt.trajectory_relative_path
        or capsule.trajectory_sha256 != receipt.trajectory_sha256
        or trajectory_envelope is None
        or trajectory_envelope["sha256"] != receipt.trajectory_sha256
        or trajectory_envelope["size_bytes"] != receipt.trajectory_size
        or universe_ref != expected_universe_authority
        or universe_ref.sha256 != receipt.universe_authority_sha256
    ):
        raise RunInputIdentityError(
            "legacy trajectory receipt coordinates do not match the selected inputs"
        )
    return receipt.trajectory_sha256, receipt.trajectory_size


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
            resume_state.get("llm_signature") or current_environment["llm_signature"]
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
            "metadata_projection_sha256": "legacy_unknown",
            "metadata_sidecar_sha256": "legacy_unknown",
            "icu_rules_sha256": "legacy_unknown",
            "metadata_implementation_bundle_sha256": "legacy_unknown",
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
        if plan is None or " ".join(plan.research_question.split()) != " ".join(
            question.split()
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
            trajectory_binding=None,
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

    # Evidence revalidation can be arbitrarily expensive and may invoke legacy
    # migration hooks.  Re-read the complete input authority immediately before
    # the first resume write so a selector/artifact swap during that interval
    # cannot be followed by an authoritative resume receipt.
    reverified_authority = load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=scientific_identity,
    )
    if (
        reverified_authority.capsule != authority.capsule
        or reverified_authority.evidence_records != authority.evidence_records
    ):
        raise RunInputIdentityError(
            "Cannot resume safely: input authority changed during revalidation."
        )
    authority = reverified_authority

    receipt_store = EvidenceStore(root=run_dir, enforcement_mode=enforcement_mode)
    receipt_path = write_resume_environment_receipt(
        run_dir=run_dir,
        evidence=receipt_store,
        capsule=authority.capsule,
        current_environment=current_environment,
        invalidated_step_ids=tuple(invalidated),
    )
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    # The receipt itself legitimately advances evidence authority.  Verify the
    # selected capsule and all sealed input bytes once more before returning any
    # path to the caller; corruption is therefore never reported as verified.
    post_receipt_authority = load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=scientific_identity,
    )
    if post_receipt_authority.capsule != authority.capsule:
        raise RunInputIdentityError(
            "Cannot resume safely: input authority changed while recording resume."
        )
    authority = post_receipt_authority
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
        if not experiment_spec_path.is_file() or sha256_of_file(
            experiment_spec_path
        ) != sha256_of_file(authority.experiment_spec_evidence_path):
            shutil.copy2(
                authority.experiment_spec_evidence_path,
                experiment_spec_path,
            )
    trajectory_binding = None
    if authority.capsule.trajectory_relative_path is not None:
        trajectory_path = run_dir / authority.capsule.trajectory_relative_path
        legacy_receipt = None
        if type(authority.capsule) is RunInputCapsuleV2:
            capsule_record = authority.evidence_records.get(
                RUN_INPUT_CAPSULE_EVIDENCE_ID
            )
            capsule_sha256 = (
                str(capsule_record.get("sha256") or "")
                if isinstance(capsule_record, Mapping)
                else ""
            )
            staged_universe_ref = MaterializedCohortAuthorityRef.from_dict(
                authority.capsule.materialized_cohort_authority_ref
            )
            legacy_receipt = VerifiedLegacyTrajectoryCapsuleReceipt(
                capsule_sha256=capsule_sha256,
                trajectory_relative_path=(authority.capsule.trajectory_relative_path),
                trajectory_sha256=str(authority.capsule.trajectory_sha256),
                trajectory_size=int(trajectory_path.stat().st_size),
                universe_authority_sha256=staged_universe_ref.sha256,
            )
        trajectory_binding = StagedTrajectoryBinding(
            path=trajectory_path,
            sha256=str(authority.capsule.trajectory_sha256),
            size=int(trajectory_path.stat().st_size),
            authority_ref=(
                MaterializedTrajectoryAuthorityRef.from_dict(
                    authority.capsule.materialized_trajectory_authority_ref
                )
                if isinstance(authority.capsule, RunInputCapsuleV3)
                else None
            ),
            legacy_capsule_receipt=legacy_receipt,
        )
    return PreparedResumeInput(
        resume_state=prepared_state,
        input_verified=True,
        context_evidence_path=authority.context_evidence_path,
        cohort_path=run_dir / authority.capsule.cohort_relative_path,
        trajectory_binding=trajectory_binding,
        experiment_spec_path=experiment_spec_path,
    )


__all__ = [
    "RUN_INPUT_CAPSULE_FILENAME",
    "RUN_INPUT_CAPSULE_EVIDENCE_ID",
    "RunInputCapsule",
    "RunInputCapsuleV2",
    "RunInputCapsuleV3",
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
    "verify_legacy_trajectory_capsule_receipt",
    "write_resume_environment_receipt",
]
