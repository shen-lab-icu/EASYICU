"""Host-reviewed, source-bound preparation of the conservative eICU subset.

This adapter is not registered as a user entry point. It grants preparation
only; research execution still requires the complete plan's existing grant.
The attestation protects against untrusted request data, not same-UID code.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import FirstObservedIcuCoordinate
from easyicu.research_agent.acquisition.eicu_first_observed_candidate import REQUIRED_COLUMNS
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint, verify_raw_preparation_review
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.webserver import sources, state_paths
from easyicu.webserver.demo_source_storage import verified_public_demo_raw
from easyicu.webserver.eicu_first_observed_source import derive_from_verified_eicu_demo
from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
from easyicu.webserver.raw_preparation_decisions import (
    RawPreparationHostBinding, _publish_once, _signature, consume_host_review, verify_consumed_host_review,
)
from easyicu.webserver.raw_source_authority import resolve_raw_source_identity_binding

_SCOPE_KEY = "first_observed_icu_preparation_scope"
_SUMMARY = (
    "Prepare the source-visible adult first ICU subset from the complete eICU Demo patient table. "
    "Read patient, hospitalization, unit order, unit type and age fields; exclude people with "
    "multiple visible hospitalizations or unverified identity, order or age. This is not a "
    "lifetime first ICU guarantee and does not authorize analysis."
)
_ADAPTER = "eicu-demo-first-observed-single-hospital-adult/1"


def _host_inputs(*, run_dir: Path, request: Any) -> tuple[dict[str, Any], Path]:
    """Construct scope from bound plan, registered source, and metadata only."""
    raw = request.payload.get("plan_review_authority")
    review = PlanReviewAuthority.model_validate(raw)
    if review.execution is not None:
        raise ValueError("first_observed_preparation_review_not_proposal")
    projection = reviewed_population_spec(run_dir=run_dir, pending=SimpleNamespace(requests=[request]))
    if projection is None or projection["plan_sha256"] != review.plan_sha256:
        raise ValueError("first_observed_preparation_population_unverified")
    matches = [item["criterion"] for item in projection["criteria"]
               if item["criterion"].get("kind") == "first_observed_icu_single_hospital_adult"
               and item["criterion"].get("role") == "include"]
    if len(matches) != 1:
        raise ValueError("first_observed_preparation_scope_ambiguous")
    with AnchoredDirectory.open(run_dir) as directory:
        raw_context = directory.read_bytes("research_context.json", max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    context_sha = hashlib.sha256(raw_context).hexdigest()
    if review.evidence_sha256.get("research_context") != context_sha:
        raise ValueError("first_observed_preparation_context_mismatch")
    context = parse_research_context_json(raw_context.decode("utf-8"))
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if identity is None or identity.database != "eicu_demo":
        raise ValueError("first_observed_preparation_source_unavailable")
    registered = [item for item in sources.load_registry().get("sources", [])
                  if isinstance(item, Mapping) and item.get("path")
                  and sources.registered_source_identity(str(item["path"])) == identity]
    if len(registered) != 1:
        raise ValueError("first_observed_preparation_source_unavailable")
    export = Path(str(registered[0]["path"]))
    binding = resolve_raw_source_identity_binding(export_path=export, database="eicu_demo")
    if binding is None or binding.family != "eicu" or "patient" not in binding.table_paths:
        raise ValueError("first_observed_preparation_source_unavailable")
    source = verified_public_demo_raw(Path(binding.table_paths["patient"]).parent)
    if source is None or source.database != "eicu_demo" or not source.archive_sha256:
        raise ValueError("first_observed_preparation_source_unavailable")
    if identity.archive_sha256 != source.archive_sha256:
        raise ValueError("first_observed_preparation_source_mismatch")
    scientific_sha = review.evidence_sha256.get("scientific_plan_review")
    if not scientific_sha:
        raise ValueError("first_observed_preparation_scientific_review_missing")
    return ({"adapter_id": _ADAPTER, "purpose": "prepare_only", "analysis_authorized": False,
             "database": "eicu_demo", "source_identity_sha256": canonical_sha256(identity.to_dict()),
             "plan_sha256": review.plan_sha256, "context_sha256": context_sha,
             "scientific_review_sha256": scientific_sha, "criterion": matches[0],
             "source_table": "patient", "source_columns": list(REQUIRED_COLUMNS),
             "population_definition": "adult_first_observed_icu_in_single_visible_hospitalization",
             "unknown_identity_or_order_policy": "exclude_or_refuse_source",
             "multiple_hospitalization_policy": "exclude_patient",
             "missing_age_policy": "exclude_patient", "not_lifetime_first_icu": True,
             "read_outcome_columns": False, "public_coordinate_authorized": False,
             "export_manifest_sha256": binding.export_manifest_sha256,
             "patient_table_sha256": binding.table_sha256["patient"],
             "archive_sha256": source.archive_sha256}, export)


def proposed_first_observed_scope(*, run_dir: Path, request: Any) -> dict[str, Any]:
    """Return an exact reviewable scope, without reading patient rows."""
    scope, _ = _host_inputs(run_dir=run_dir, request=request)
    return scope


def _private_root(*, create: bool) -> Path:
    root = state_paths.state_root() / "private-first-observed"
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("first_observed_private_directory_invalid")
    return root


def prepare_first_observed_coordinate(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                      binding: RawPreparationHostBinding,
                                      scope: Mapping[str, Any]) -> FirstObservedIcuCoordinate:
    """Consume host approval before reading source rows and attest exact bytes."""
    expected, export = _host_inputs(run_dir=run_dir, request=checkpoint.request)
    scope = dict(scope)
    if scope != expected:
        raise ValueError("first_observed_preparation_scope_mismatch")
    verify_raw_preparation_review(checkpoint=checkpoint, requested_scope=scope, scope_key=_SCOPE_KEY,
        expected_summary=_SUMMARY, registered_database="eicu_demo",
        registered_identity_sha256=scope["source_identity_sha256"], plan_sha256=scope["plan_sha256"],
        context_sha256=scope["context_sha256"], scientific_review_sha256=scope["scientific_review_sha256"])
    consume_host_review(checkpoint=checkpoint, scope=scope, **binding.model_dump(mode="python"))
    candidate = derive_from_verified_eicu_demo(export)
    proof = candidate.receipt.get("source_binding", {})
    completeness = proof.get("complete_patient_source", {})
    if (proof.get("official_release_independently_verified") is not True
        or completeness.get("schema_version") != "easyicu.eicu_demo_complete_patient_source/1"
        or completeness.get("all_source_rows_compared") is not True
        or completeness.get("archive_sha256") != scope["archive_sha256"]
        or proof.get("patient_table_sha256") != scope["patient_table_sha256"]
        or proof.get("export_manifest_sha256") != scope["export_manifest_sha256"]):
        raise ValueError("first_observed_preparation_source_proof_mismatch")
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(candidate.frame, preserve_index=False), buffer)
    data = buffer.getvalue()
    body = {"schema_version": "easyicu.first-observed-coordinate-attestation/1",
            "scope_sha256": canonical_sha256(scope), "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "binding_sha256": canonical_sha256(binding.model_dump(mode="json")),
            "coordinate_sha256": hashlib.sha256(data).hexdigest(), "source_complete_six_columns": True,
            "analysis_authorized": False}
    document = canonical_json({"body": body, "signature": _signature(body)}).encode()
    root = _private_root(create=True)
    with AnchoredDirectory.open(root) as directory:
        _publish_once(directory, binding.nonce + ".parquet", data)
        _publish_once(directory, binding.nonce + ".json", document)
        directory.assert_still_selected()
    return _coordinate(root, binding.nonce, scope, checkpoint, body)


def _coordinate(root: Path, nonce: str, scope: Mapping[str, Any], checkpoint: RawPreparationCheckpoint,
                body: Mapping[str, Any]) -> FirstObservedIcuCoordinate:
    return FirstObservedIcuCoordinate(root / (nonce + ".parquet"), str(body["coordinate_sha256"]),
        scope["archive_sha256"], scope["patient_table_sha256"], scope["export_manifest_sha256"],
        scope["plan_sha256"], checkpoint.checkpoint_sha256)


def resolve_prepared_first_observed_coordinate(*, run_dir: Path, checkpoint: RawPreparationCheckpoint,
                                               binding: RawPreparationHostBinding,
                                               scope: Mapping[str, Any]) -> FirstObservedIcuCoordinate:
    """Reverify the host decision, reviewed scope, signed result and exact bytes."""
    expected, _ = _host_inputs(run_dir=run_dir, request=checkpoint.request)
    scope = dict(scope)
    if scope != expected or checkpoint.request.payload.get(_SCOPE_KEY) != scope:
        raise ValueError("first_observed_preparation_scope_mismatch")
    verify_consumed_host_review(checkpoint=checkpoint, binding=binding, scope=scope)
    root = _private_root(create=False)
    with AnchoredDirectory.open(root) as directory:
        document = json.loads(directory.read_bytes(binding.nonce + ".json", max_bytes=64 * 1024))
        with directory.open_regular(binding.nonce + ".parquet") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        directory.assert_still_selected()
    body, signature = document["body"], document["signature"]
    if not isinstance(signature, str) or not hmac.compare_digest(signature, _signature(body)):
        raise ValueError("first_observed_attestation_invalid")
    if body != {"schema_version": "easyicu.first-observed-coordinate-attestation/1",
                "scope_sha256": canonical_sha256(scope), "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "binding_sha256": canonical_sha256(binding.model_dump(mode="json")), "coordinate_sha256": digest,
                "source_complete_six_columns": True, "analysis_authorized": False}:
        raise ValueError("first_observed_attestation_binding_mismatch")
    return _coordinate(root, binding.nonce, scope, checkpoint, body)
