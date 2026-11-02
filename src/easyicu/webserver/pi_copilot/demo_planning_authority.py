"""Host-only, source-bound receipt for a confirmed official Demo planning choice."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.canonical_json import canonical_json
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import dataio, demo_source_storage, sources, study_contexts
from easyicu.webserver.raw_preparation_decisions import _host_key, _private_directory, _publish_once

_DOMAIN = b"easyicu.demo-planning-source-choice/1\0"


def _signature(body: Mapping[str, Any]) -> str:
    return hmac.new(_host_key(), _DOMAIN + canonical_json(dict(body)).encode(), hashlib.sha256).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("demo_planning_source_changed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_facts(export: str, raw: str, database: str) -> dict[str, Any]:
    historical = demo_source_storage.verified_public_demo_replanning_export(export)
    if historical is None:
        raise ValueError("demo_planning_source_unregistered")
    candidate, paths = historical
    normalized_export = sources._norm_path(export)
    identity = SourceIdentity(database=candidate.database, registration_id=sources._source_id(normalized_export),
                              version=candidate.version, source_id=candidate.id,
                              archive_sha256=candidate.archive_sha256)
    if sources.stored_source_path_for_identity(identity) != normalized_export:
        raise ValueError("demo_planning_source_unregistered")
    if (candidate.database != database or identity.database != database
            or identity.source_id != candidate.id or identity.version != candidate.version
            or identity.archive_sha256 != candidate.archive_sha256
            or demo_source_storage.verified_public_demo_raw(raw) != candidate):
        raise ValueError("demo_planning_source_changed")
    binding = dataio.resolve_registered_export_binding(export, database)
    if Path(binding["source_data_path"]) != Path(raw) or Path(raw) != paths.raw:
        raise ValueError("demo_planning_source_changed")
    registry = Path(sources._CONFIG_PATH)
    registry_sha = _sha(registry)
    registry_stat = registry.stat()
    return {
        "identity": identity.to_dict(),
        "registry_sha256": registry_sha,
        "registry_inode": registry_stat.st_ino,
        "registry_mtime_ns": registry_stat.st_mtime_ns,
        "export_path_sha256": hashlib.sha256(export.encode()).hexdigest(),
        "raw_path_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "manifest_sha256": _sha(Path(export) / "_manifest.json"),
        "conversion_sha256": _sha(Path(raw) / "conversion_manifest.json"),
        "prepared_marker_sha256": _sha(paths.prepared_marker),
        "extracted_marker_sha256": _sha(paths.extracted_marker),
        "converted_marker_sha256": _sha(paths.converted_marker),
    }


def publish_choice(*, study: Mapping[str, Any], new_source: Mapping[str, Any],
                   confirmations: Mapping[str, Any], offer_sha256: str) -> str:
    source = study.get("data_source") or {}
    facts = _source_facts(str(source.get("path") or ""), str(new_source.get("path") or ""),
                          str(new_source.get("database") or ""))
    from .demo_planning_source import resolve_demo_planning_source

    rechecked = resolve_demo_planning_source(study)
    if (rechecked is None or rechecked.raw_path != new_source.get("path")
            or not hmac.compare_digest(str(rechecked.offer.get("offer_sha256") or ""), offer_sha256)
            or _source_facts(str(source.get("path") or ""), str(new_source.get("path") or ""),
                             str(new_source.get("database") or "")) != facts):
        raise ValueError("demo_planning_source_changed")
    proposed = {**dict(study), "data_source": dict(new_source), "confirmations": dict(confirmations)}
    body = {"schema_version": "easyicu.demo-planning-source-choice/1",
            "study_id": study["id"], "revision": study["revision"] + 1,
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(proposed),
            "offer_sha256": offer_sha256, "nonce": secrets.token_hex(32), **facts}
    receipt = {"body": body, "signature": _signature(body)}
    raw = canonical_json(receipt).encode()
    ref = hashlib.sha256(raw).hexdigest()
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, f"demo-planning-{ref}.json", raw)
        directory.assert_still_selected()
    return ref


def resolve_choice(study: Mapping[str, Any]) -> Any:
    ref = str(study.get("demo_planning_authority_ref") or "")
    if not ref:
        return None
    if len(ref) != 64 or any(ch not in "0123456789abcdef" for ch in ref):
        raise ValueError("demo_planning_authority_invalid")
    with AnchoredDirectory.open(_private_directory()) as directory:
        raw = directory.read_bytes(f"demo-planning-{ref}.json", max_bytes=65536)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != ref:
        raise ValueError("demo_planning_authority_invalid")
    receipt = json.loads(raw)
    body, signature = receipt["body"], receipt["signature"]
    if not isinstance(body, dict) or not isinstance(signature, str) or not hmac.compare_digest(_signature(body), signature):
        raise ValueError("demo_planning_authority_invalid")
    source = study.get("data_source") or {}
    if (body.get("schema_version") != "easyicu.demo-planning-source-choice/1"
            or body.get("study_id") != study.get("id") or type(body.get("revision")) is not int
            or type(study.get("revision")) is not int or study["revision"] < body["revision"]
            or body.get("scientific_configuration_sha256") != study_contexts.scientific_configuration_sha256(dict(study))):
        raise ValueError("demo_planning_authority_stale")
    identity_record = body.get("identity")
    identity = SourceIdentity.from_record(identity_record)
    if identity is None or identity.to_dict() != identity_record:
        raise ValueError("demo_planning_authority_invalid")
    export_path = sources.stored_source_path_for_identity(identity)
    if export_path is None:
        raise ValueError("demo_planning_source_unregistered")
    facts = _source_facts(export_path, str(source.get("path") or ""), str(source.get("database") or ""))
    if any(body.get(key) != value for key, value in facts.items()):
        raise ValueError("demo_planning_source_changed")
    return identity
