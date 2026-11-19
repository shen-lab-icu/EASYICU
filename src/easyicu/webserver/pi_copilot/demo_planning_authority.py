"""Host-only, source-bound receipt for a confirmed official Demo planning choice."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
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
    # Bind this source's own registration, not the whole registry file: other
    # registry writes (selecting or comparing sources, a finished extraction)
    # must not invalidate a confirmed choice (M2, 2026-11-19).
    registration_sha = sources.stored_registration_sha256(identity)
    if registration_sha is None:
        raise ValueError("demo_planning_source_unregistered")
    return {
        "identity": identity.to_dict(),
        "registration_sha256": registration_sha,
        "export_path_sha256": hashlib.sha256(export.encode()).hexdigest(),
        "raw_path_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "manifest_sha256": _sha(Path(export) / "_manifest.json"),
        "conversion_sha256": _sha(Path(raw) / "conversion_manifest.json"),
        "prepared_marker_sha256": _sha(paths.prepared_marker),
        "extracted_marker_sha256": _sha(paths.extracted_marker),
        "converted_marker_sha256": _sha(paths.converted_marker),
    }


def _source_study_sha256(study: Mapping[str, Any]) -> str:
    # A document choice changes the review candidate, while this receipt
    # grants access to a specific scientific study and registered source.
    return study_contexts.scientific_configuration_sha256(
        {key: value for key, value in study.items() if key != "plan_deliverables"}
    )


def publish_choice(*, study: Mapping[str, Any], new_source: Mapping[str, Any],
                   confirmations: Mapping[str, Any], offer_sha256: str,
                   drafted_scope: Mapping[str, Any] | None = None, again: bool = False) -> str:
    source = study.get("data_source") or {}
    facts = _source_facts(str(source.get("path") or ""), str(new_source.get("path") or ""),
                          str(new_source.get("database") or ""))
    from .demo_planning_source import resolve_demo_planning_source

    rechecked = resolve_demo_planning_source(study, again=again)
    if (rechecked is None or rechecked.raw_path != new_source.get("path")
            or not hmac.compare_digest(str(rechecked.offer.get("offer_sha256") or ""), offer_sha256)
            or _source_facts(str(source.get("path") or ""), str(new_source.get("path") or ""),
                             str(new_source.get("database") or "")) != facts):
        raise ValueError("demo_planning_source_changed")
    proposed = {**dict(study), **dict(drafted_scope or {}),
                "data_source": dict(new_source), "confirmations": dict(confirmations)}
    body = {"schema_version": "easyicu.demo-planning-source-choice/1",
            "study_id": study["id"], "revision": study["revision"] + 1,
            "scientific_configuration_sha256": _source_study_sha256(proposed),
            "offer_sha256": offer_sha256, "nonce": secrets.token_hex(32), **facts}
    receipt = {"body": body, "signature": _signature(body)}
    raw = canonical_json(receipt).encode()
    ref = hashlib.sha256(raw).hexdigest()
    with AnchoredDirectory.open(_private_directory()) as directory:
        _publish_once(directory, f"demo-planning-{ref}.json", raw)
        directory.assert_still_selected()
    return ref


def _signed_body(ref: str) -> dict[str, Any]:
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
    return body


def resolve_choice(study: Mapping[str, Any]) -> Any:
    ref = str(study.get("demo_planning_authority_ref") or "")
    if not ref:
        return None
    body = _signed_body(ref)
    source = study.get("data_source") or {}
    if (body.get("schema_version") != "easyicu.demo-planning-source-choice/1"
            or body.get("study_id") != study.get("id") or type(body.get("revision")) is not int
            or type(study.get("revision")) is not int or study["revision"] < body["revision"]
            or body.get("scientific_configuration_sha256") != _source_study_sha256(study)):
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


def signed_choice_revision(study: Mapping[str, Any]) -> int | None:
    """The study revision the study's current, valid choice was written at.

    ``None`` without a choice that resolves for this study now. Read only.
    """

    from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
    from easyicu.webserver import dataio

    try:
        if resolve_choice(study) is None:
            return None
        revision = _signed_body(str(study.get("demo_planning_authority_ref") or "")).get("revision")
    except (ValueError, OSError, KeyError, TypeError, dataio.ExportCohortError, AuthorityFilesystemError):
        return None
    return revision if type(revision) is int else None


_RECEIPT_NAME = re.compile(r"demo-planning-([0-9a-f]{64})\.json")
_MAX_RECEIPTS = 4096


def prior_choice_export(study: Mapping[str, Any]) -> str | None:
    """The registered export named by this study's latest signed raw choice.

    A write that changes the scientific configuration clears the choice
    (study_contexts) while the study keeps the raw path, e.g. applying a
    reviewed plan's configuration (UX leg 3, 2026-11-19). The researcher's
    earlier receipt for the same study and raw folder names the export the
    choice was offered from, so the choice can be offered again on the
    current configuration. This grants nothing: only a new confirmation
    signs a new receipt. Unreadable receipts are skipped; too many, or none,
    is no offer.
    """
    source = study.get("data_source") or {}
    raw = str(source.get("path") or "") if isinstance(source, Mapping) else ""
    study_id = study.get("id")
    if not raw or not study_id:
        return None
    raw_sha = hashlib.sha256(raw.encode()).hexdigest()
    try:
        names = [name for name in os.listdir(_private_directory()) if _RECEIPT_NAME.fullmatch(name)]
    except (OSError, ValueError, AuthorityFilesystemError):
        return None
    if len(names) > _MAX_RECEIPTS:
        return None
    latest = None
    for name in names:
        try:
            body = _signed_body(_RECEIPT_NAME.fullmatch(name).group(1))
            identity = SourceIdentity.from_record(body.get("identity"))
        except (ValueError, OSError, KeyError, TypeError, AuthorityFilesystemError):
            continue
        revision = body.get("revision")
        if (body.get("schema_version") != "easyicu.demo-planning-source-choice/1"
                or body.get("study_id") != study_id or body.get("raw_path_sha256") != raw_sha
                or type(revision) is not int or identity is None
                or identity.database != str(source.get("database") or "")):
            continue
        if latest is None or revision > latest[0]:
            latest = (revision, identity)
    return None if latest is None else sources.stored_source_path_for_identity(latest[1])


def follow_launch_timing(before: Mapping[str, Any], landmark_patch: Mapping[str, Any]) -> str:
    """Carry a confirmed choice onto the 24 h landmark the host adds at launch.

    The choice signs the study's scientific configuration. At launch,
    plan_decisions.record_post_baseline_landmark (the only caller) derives the
    agent landmark from that same study, so the runner's recheck found the
    receipt stale and every raw Demo plan with a first-day exposure and a death
    outcome stopped (UX leg 3, 2026-11-19).

    This re-signs the same choice (identity, offer, source facts, revision) for
    ``before`` plus ``landmark_patch`` only when the original receipt resolves
    for ``before`` now, is not itself a follow, ``before`` has no timing row,
    and the patch equals, by value, the host's agent landmark for ``before``:
    that one sensitivity row and its three plan_timing_* confirmations. The
    follow is deterministic, so a receipt has exactly one.
    """

    from .plan_decisions import agent_post_baseline_landmark_patch

    ref = str(before.get("demo_planning_authority_ref") or "")
    if not ref:
        raise ValueError("demo_planning_authority_stale")
    original = _signed_body(ref)
    rows = before.get("sensitivity_specs") or []
    if ("follows" in original or resolve_choice(before) is None or not isinstance(rows, list)
            or any(not isinstance(row, Mapping) or row.get("axis") == "timing" for row in rows)):
        raise ValueError("demo_planning_authority_stale")
    host_landmark = agent_post_baseline_landmark_patch(before)
    if canonical_json(dict(landmark_patch)) != canonical_json(host_landmark):
        raise ValueError("demo_planning_authority_stale")
    body = {**original, "scientific_configuration_sha256": _source_study_sha256({**dict(before), **host_landmark}),
            "nonce": hashlib.sha256(b"easyicu.demo-planning-follow/1\0" + ref.encode()).hexdigest(),
            "follows": ref}
    raw = canonical_json({"body": body, "signature": _signature(body)}).encode()
    followed = hashlib.sha256(raw).hexdigest()
    with AnchoredDirectory.open(_private_directory()) as directory:
        try:
            _publish_once(directory, f"demo-planning-{followed}.json", raw)
        except FileExistsError:
            # A refused launch left this same follow; the name is its content hash.
            if directory.read_bytes(f"demo-planning-{followed}.json", max_bytes=65536) != raw:
                raise ValueError("demo_planning_authority_invalid") from None
        directory.assert_still_selected()
    return followed
