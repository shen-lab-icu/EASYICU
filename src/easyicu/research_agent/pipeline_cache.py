"""Pipeline cache: deterministic key derivation + on-disk index.

Extracted from :mod:`easyicu.research_agent.pipeline` so the cache
surface can be reasoned about (and tested) without dragging the
full ``ResearchAgentPipeline`` class in.

The cache key is a sha256 hex digest of a canonical JSON payload that
combines:

* the cohort bytes hash;
* normalised run knobs (question, target_outcome, skill, database, ...);
* a bag of caller-supplied flags (``enable_*``, ``max_*``, ``latex_*``,
  ``context_top_k``, ``disable_icu_context``, ...);
* caller-supplied scientific inputs that are not represented by the cohort;
* the configured LLM topology and model(s);
* hashes of the engine, validators, prompts, and concept dictionaries.

Cache hits are fail-closed.  A prior run is reusable only when its final
manifest and run-status gates prove that it is a complete, manuscript-ready
run.  Paused, partial, aborted, or blocked runs are never cache entries.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .evidence_authority import EvidenceAuthorityIntegrityError
from .llm import MockLLMClient
from .metadata_implementation_identity import metadata_implementation_identity
from .prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .run_input_capsule import (
    RunInputIdentityError,
    invalidate_unverified_successful_steps,
    load_verified_run_input_capsule,
)
from .runtime_artifacts import verified_run_evidence_path
from .schema import AnalysisManifest, PipelineResult

_CACHE_KEY_SCHEMA_VERSION = "easyicu.pipeline_cache_key/2"
_CACHE_READY_STATUSES = frozenset({"manuscript_ready", "publication_ready"})
_CACHE_REQUIRED_GATES = (
    "execution_complete",
    "evidence_complete",
    "numeric_verified",
    "analysis_validated",
    "manuscript_ready",
)
_NON_COMPLETE_NOTE_TOKENS = (
    "paused_after_analysis",
    "pipeline aborted",
    "aborted:",
    "stopped after",
)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def hash_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Streaming sha256 of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _canonical_sha256(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tree_sha256(root: Path, paths: Sequence[Path]) -> str:
    return _canonical_sha256(
        {
            str(path.relative_to(root)): hash_file(path)
            for path in sorted(paths)
            if path.is_file()
        }
    )


def runtime_identity() -> Dict[str, Any]:
    """Return code/prompt/concept identities that affect cached output.

    This is evaluated at key construction time rather than captured as a
    module constant, so a long-lived service cannot reuse a pre-change key
    after a hot deployment.
    """

    package_root = Path(__file__).resolve().parent
    engine_paths = list(package_root.rglob("*.py"))
    validator_paths = [
        package_root / "gates" / "preflight.py",
        package_root / "declared_product_contract.py",
        package_root / "audits" / "base.py",
        package_root / "audits" / "patterns.py",
        package_root / "audits" / "validators.py",
        package_root / "audits" / "step_summary_integrity.py",
    ]
    prompt_files = prompt_pack_files()
    data_root = package_root.parent / "data"
    concept_files = {
        name: hash_file(data_root / name) if (data_root / name).is_file() else "missing"
        for name in ("concept-dict.json", "sofa2-dict.json")
    }
    return {
        "engine_code_sha256": _tree_sha256(package_root, engine_paths),
        "validator_code_sha256": _tree_sha256(package_root, validator_paths),
        "prompt_pack_version": PROMPT_PACK_VERSION,
        "prompt_pack_sha256": _canonical_sha256(prompt_files),
        "concept_dictionary_sha256": _canonical_sha256(concept_files),
        **dict(metadata_implementation_identity()),
    }


def llm_signature(llm: Any) -> str:
    """Return a short string identifying the configured LLM(s).

    Two runs with different LLMs *must* invalidate the cache because the
    bound manuscript / generated code / chosen plan could differ. The
    signature binds role mapping and ordered fallback topology because both can
    change which provider produces an answer.
    """
    if llm is None:
        return "unconfigured"
    if isinstance(llm, MockLLMClient):
        return "mock"
    if isinstance(getattr(llm, "_clients", None), (list, tuple)):
        payload = {
            "schema": "easyicu.llm_fallback_authority/1",
            "class": f"{type(llm).__module__}.{type(llm).__qualname__}",
            "clients": [llm_signature(client) for client in llm._clients],
        }
        return f"fallback-authority:{_canonical_sha256(payload)}"
    if hasattr(llm, "for_role") and isinstance(getattr(llm, "_roles", None), dict):
        role_signatures: Dict[str, str] = {}
        for role in getattr(llm, "_roles"):
            try:
                role_signatures[str(role)] = llm_signature(llm.for_role(role))
            except KeyError:
                role_signatures[str(role)] = "unconfigured"
        payload = {
            "schema": "easyicu.llm_router_authority/1",
            "class": f"{type(llm).__module__}.{type(llm).__qualname__}",
            "default": llm_signature(getattr(llm, "_default", None)),
            "roles": role_signatures,
        }
        return f"router-authority:{_canonical_sha256(payload)}"
    if hasattr(llm, "iter_clients"):
        sigs = [llm_signature(c) for c in llm.iter_clients()]
        return "client-topology:" + _canonical_sha256(sigs)
    model = getattr(llm, "_model", None)
    cls = getattr(llm, "name", llm.__class__.__name__)
    endpoint = (
        str(
            getattr(llm, "_resolved_base_url", None)
            or getattr(llm, "_base_url", None)
            or ""
        )
        .strip()
        .rstrip("/")
    )
    extra_body = getattr(llm, "_extra_body", None)
    payload = {
        "schema": "easyicu.llm_client_authority/1",
        "class": f"{type(llm).__module__}.{type(llm).__qualname__}",
        "name": str(cls),
        "model": str(model or ""),
        # Only digests are persisted: endpoint topology and output-affecting
        # request options are bound without leaking credentials or headers.
        "endpoint_sha256": _canonical_sha256(endpoint),
        "extra_body_sha256": _canonical_sha256(extra_body or {}),
    }
    return f"{cls}:{model or ''}:authority:{_canonical_sha256(payload)}"


def iter_mock_clients(llm: Any):
    """Yield every :class:`MockLLMClient` reachable through ``llm``.

    For a plain client this is just ``llm`` itself when it's a Mock;
    for an :class:`LLMRouter` we walk the router's child clients.
    Idempotent for non-Mock clients.
    """
    if llm is None:
        return
    if isinstance(llm, MockLLMClient):
        yield llm
        return
    if hasattr(llm, "iter_clients"):
        for child in llm.iter_clients():
            if isinstance(child, MockLLMClient):
                yield child


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _resolve_run_artifact(
    run_dir: Path,
    relative_value: Any,
    *,
    default: Optional[str] = None,
) -> Optional[Path]:
    relative_text = str(relative_value or default or "").strip()
    relative = Path(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    candidate = run_dir / relative
    try:
        candidate.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() and not candidate.is_symlink() else None


def _completed_run_payload(
    *,
    run_dir: Path,
    expected_run_id: str,
    scientific_identity: Mapping[str, Any],
) -> Optional[tuple[Dict[str, Path], Dict[str, Any]]]:
    """Return verified result paths only for a complete reusable run."""

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None
    manifest = _read_json_object(manifest_path)
    if manifest is None or str(manifest.get("run_id") or "") != expected_run_id:
        return None
    try:
        AnalysisManifest.model_validate(manifest)
        input_authority = load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=dict(scientific_identity),
        )
    except (
        EvidenceAuthorityIntegrityError,
        RunInputIdentityError,
        TypeError,
        ValueError,
    ):
        return None
    if not manifest.get("finished_at"):
        return None
    notes = str(manifest.get("notes") or "").lower()
    if any(token in notes for token in _NON_COMPLETE_NOTE_TOKENS):
        return None
    if bool(manifest.get("writer_probe_mode")):
        return None

    # A newer partial checkpoint means a resume has superseded this final
    # manifest.  Corrupt checkpoint state is never authority to reuse old work.
    partial_path = run_dir / "manifest_partial.json"
    if partial_path.exists():
        partial = _read_json_object(partial_path)
        if partial is None:
            return None
        final_seq = manifest.get("checkpoint_sequence")
        partial_seq = partial.get("checkpoint_sequence")
        if isinstance(final_seq, int) or isinstance(partial_seq, int):
            if not (
                isinstance(final_seq, int)
                and isinstance(partial_seq, int)
                and final_seq > partial_seq
            ):
                return None
        elif partial_path.stat().st_mtime_ns > manifest_path.stat().st_mtime_ns:
            return None

    readiness = manifest.get("readiness")
    if not isinstance(readiness, Mapping) or any(
        readiness.get(gate) is not True for gate in _CACHE_REQUIRED_GATES
    ):
        return None

    records = manifest.get("per_step_records")
    if not isinstance(records, list) or not records:
        return None
    latest: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return None
        step_id = str(record.get("step_id") or "").strip()
        if step_id:
            latest[step_id] = str(record.get("status") or "").strip().lower()
    if not latest or set(latest.values()) != {"ok"}:
        return None

    manifest_evidence = manifest.get("evidence")
    if not isinstance(manifest_evidence, list):
        return None
    manifest_records: Dict[str, Dict[str, Any]] = {}
    for raw_record in manifest_evidence:
        if not isinstance(raw_record, Mapping):
            return None
        evidence_id = str(raw_record.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in manifest_records:
            return None
        manifest_records[evidence_id] = dict(raw_record)
    if manifest_records != input_authority.evidence_records:
        return None
    if any(
        verified_run_evidence_path(run_dir, record) is None
        for record in input_authority.evidence_records.values()
    ):
        return None
    _, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=manifest,
        records=input_authority.evidence_records,
    )
    if invalidated:
        return None

    paths: Dict[str, Path] = {}
    for key, manifest_key, default in (
        ("context", "context_path", "research_context.json"),
        ("plan", "plan_path", "analysis_plan.json"),
        ("report", "report_path", "results_report.md"),
        ("manuscript", "manuscript_path", "manuscript_scaffold_bound.md"),
    ):
        resolved = _resolve_run_artifact(
            run_dir,
            manifest.get(manifest_key),
            default=default,
        )
        if resolved is None:
            return None
        paths[key] = resolved

    artifact_paths = manifest.get("artifact_paths")
    status_relative = (
        artifact_paths.get("run_status")
        if isinstance(artifact_paths, Mapping)
        else None
    )
    status_path = _resolve_run_artifact(
        run_dir,
        status_relative,
        default="run_status.json",
    )
    if status_path is None:
        return None
    root_evidence_pairs = {
        "context": "research_context",
        "plan": "analysis_plan",
        "manuscript": "manuscript_scaffold_bound",
    }
    for path_key, evidence_id in root_evidence_pairs.items():
        record = input_authority.evidence_records.get(evidence_id)
        if (
            record is None
            or verified_run_evidence_path(run_dir, record) is None
            or hash_file(paths[path_key]) != str(record.get("sha256") or "")
        ):
            return None
    status_record = input_authority.evidence_records.get("run_status")
    if status_record is None:
        return None
    status_evidence_path = verified_run_evidence_path(run_dir, status_record)
    if status_evidence_path is None or hash_file(status_path) != str(
        status_record.get("sha256") or ""
    ):
        return None
    run_status = _read_json_object(status_evidence_path)
    if run_status is None or str(run_status.get("status") or "") not in (
        _CACHE_READY_STATUSES
    ):
        return None
    if run_status.get("gates") != dict(readiness):
        return None
    if str(run_status.get("research_question") or "") != str(
        manifest.get("research_question") or ""
    ):
        return None
    return paths, manifest


# ---------------------------------------------------------------------------
# PipelineCache class — wraps the on-disk index
# ---------------------------------------------------------------------------


class PipelineCache:
    """Encapsulates cache key generation and the on-disk cache index.

    The caller (typically the pipeline) constructs this with a target
    directory and then calls :meth:`compute_key` / :meth:`lookup` /
    :meth:`record_hit` directly. Configuration knobs that participate
    in the cache key are passed in as a ``flags`` mapping so the cache
    module stays decoupled from the pipeline's specific flag set.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)

    # -- index path -----------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self._cache_dir / "cache_index.json"

    # -- key derivation -------------------------------------------------

    def compute_key(
        self,
        *,
        cohort_path: Path,
        question: Optional[str],
        target_outcome: Optional[str],
        skill_key: Optional[str],
        database: Optional[str],
        llm: Any,
        stop_after_analysis: bool,
        manuscript_language: str,
        flags: Mapping[str, Any],
        science_inputs: Optional[Mapping[str, Any]] = None,
        identity_hashes: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Compose the deterministic cache key for this run.

        ``flags`` is a free-form mapping of additional configuration
        knobs that should invalidate the cache when changed.

        ``science_inputs`` is the host-owned identity envelope for inputs not
        represented by the materialised cohort itself (for example exposure,
        eligibility criteria, time windows, context settings, and experiment
        specification).  It is optional for API compatibility, but production
        callers should pass the complete run-request envelope.

        ``identity_hashes`` permits a caller to add or override environment
        identities.  The default always includes current engine, validator,
        prompt, and concept-dictionary hashes.
        """
        environment = runtime_identity()
        environment.update(dict(identity_hashes or {}))

        # Experiment specifications are materialised before the current cache
        # lookup.  Bind their bytes even for legacy callers that have not yet
        # adopted ``science_inputs``.
        spec_path = Path(cohort_path).parent / "experiment_spec.yaml"
        payload: Dict[str, Any] = {
            "schema_version": _CACHE_KEY_SCHEMA_VERSION,
            "cohort_sha256": hash_file(cohort_path),
            "question": (question or "").strip(),
            "target_outcome": (target_outcome or "").strip(),
            "skill": (skill_key or "").strip(),
            "database": (database or "").strip(),
            "stop_after_analysis": bool(stop_after_analysis),
            "manuscript_language": manuscript_language,
            "llm": llm_signature(llm),
            "experiment_spec_sha256": (
                hash_file(spec_path) if spec_path.is_file() else None
            ),
            "science_inputs": dict(science_inputs or {}),
            "flags": dict(flags),
            "environment": environment,
        }
        return _canonical_sha256(payload)

    # -- index storage --------------------------------------------------

    def load_index(self) -> Dict[str, Dict[str, str]]:
        path = self.index_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}

    def save_index(self, index: Dict[str, Dict[str, str]]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    # -- query / record -------------------------------------------------

    def lookup(
        self,
        cache_key: str,
        *,
        scientific_identity: Mapping[str, Any],
    ) -> Optional[PipelineResult]:
        """Return a prior complete :class:`PipelineResult`, otherwise ``None``.

        Existence alone is not completion.  The final manifest, current step
        ledger, plan coverage, readiness gates, and run status must agree that
        the run reached a manuscript-ready terminal state.  Invalid entries
        are evicted so repeated calls do not repeatedly inspect them.
        """
        index = self.load_index()
        entry = index.get(cache_key)
        if not entry:
            return None
        run_id = entry.get("run_id")
        workdir = entry.get("workdir")
        if not run_id or not workdir:
            index.pop(cache_key, None)
            self.save_index(index)
            return None
        run_dir = Path(workdir)
        manifest = run_dir / "manifest.json"
        completed = _completed_run_payload(
            run_dir=run_dir,
            expected_run_id=run_id,
            scientific_identity=scientific_identity,
        )
        if completed is None:
            index.pop(cache_key, None)
            self.save_index(index)
            return None
        paths, manifest_payload = completed
        try:
            return PipelineResult(
                run_id=run_id,
                workdir=str(run_dir),
                context_path=str(paths["context"]),
                plan_path=str(paths["plan"]),
                manifest_path=str(manifest),
                report_path=str(paths["report"]),
                manuscript_path=str(paths["manuscript"]),
                evidence_count=len(manifest_payload.get("evidence") or []),
                findings_count=len(manifest_payload.get("findings") or []),
            )
        except Exception:
            return None

    def record_hit(
        self,
        cache_key: str,
        result: PipelineResult,
        *,
        scientific_identity: Mapping[str, Any],
    ) -> None:
        """Record ``result`` only when host-owned completion gates pass."""

        run_dir = Path(result.workdir)
        if (
            _completed_run_payload(
                run_dir=run_dir,
                expected_run_id=str(result.run_id),
                scientific_identity=scientific_identity,
            )
            is None
        ):
            # If a key was previously present, a newer partial/blocked result
            # must not leave the old authority reachable under that key.
            index = self.load_index()
            if cache_key in index:
                index.pop(cache_key, None)
                self.save_index(index)
            return
        index = self.load_index()
        index[cache_key] = {
            "run_id": result.run_id,
            "workdir": result.workdir,
            "evidence_count": str(result.evidence_count),
            "findings_count": str(result.findings_count),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_index(index)


__all__ = [
    "PipelineCache",
    "hash_file",
    "iter_mock_clients",
    "llm_signature",
]
