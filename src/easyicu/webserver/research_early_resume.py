"""Durable seed for the narrow pre-Provider metadata-planning boundary.

The seed carries only public connection identity and source metadata. It does
not assert that an unknown request has finished or authorize replay of one.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.run_input import canonical_sha256

SCHEMA = "easyicu.research-early-planning-seed/1"
FILENAME = "early_planning_recovery_seed.json"
_PUBLIC_IDENTITY = (
    "provider", "model", "credential_fingerprint", "endpoint_fingerprint",
    "credential_header", "base_url_endpoint", "session_binding_sha256",
)


def fresh_seed_scope(*, scope: Any, acquisition: Any, literature: Any) -> bool:
    """A fresh seed cannot replace a continuation's bound literature."""
    return scope is None and acquisition is None and literature is None


def metadata_binding(database: str, export_path: str) -> dict[str, Any]:
    """Bind the registered source and the exact value-free planning menu."""
    from easyicu.webserver import agent_pipeline_runs, sources

    path = Path(export_path).expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError("early_source_invalid")
    identity = sources.registered_source_identity(str(path))
    if identity is None:
        raise ValueError("early_source_identity_missing")
    catalog = agent_pipeline_runs._metadata_only_planning_catalog(
        database=database, export_path=path,
    )
    return {
        "export_path": str(path.resolve()),
        "database": database,
        "source_identity": identity.to_dict(),
        "catalog_sha256": canonical_sha256(asdict(catalog)),
        "without_values": list(agent_pipeline_runs._source_concepts_without_values(path)),
    }


def create_seed(
    *, wrapper: Path, study_id: str, scientific_sha256: str,
    credential_source: str, budget_mode: str, provider_public: Mapping[str, Any],
    database: str, export_path: str, runner_image: str | None,
    original_source_identity: Mapping[str, Any],
    hard_stop_ledger_path: str, hard_stop_task_id: str,
    hard_stop_declaration_sha256: str,
) -> None:
    """Persist before first provider dispatch; never overwrite earlier evidence."""
    binding = metadata_binding(database, export_path)
    if binding["source_identity"] != original_source_identity:
        raise ValueError("early_source_identity_changed")
    payload = {
        "schema_version": SCHEMA, "study_id": study_id,
        "scientific_configuration_sha256": scientific_sha256,
        "credential_source": credential_source, "budget_mode": budget_mode,
        "provider_public": {key: provider_public.get(key) for key in _PUBLIC_IDENTITY},
        "source": binding, "runner_image": runner_image,
        "hard_stop_ledger_path": hard_stop_ledger_path,
        "hard_stop_task_id": hard_stop_task_id,
        "hard_stop_declaration_sha256": hard_stop_declaration_sha256,
    }
    payload["sha256"] = canonical_sha256(payload)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > 65536:
        raise ValueError("early_seed_too_large")
    target = wrapper / FILENAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if hasattr(os, "O_DIRECTORY"):
        directory_fd = os.open(wrapper, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def load_seed(wrapper: Path) -> dict[str, Any] | None:
    target = wrapper / FILENAME
    if target.is_symlink():
        raise ValueError("early_seed_invalid")
    try:
        if target.stat().st_size > 65536:
            raise ValueError("early_seed_invalid")
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("early_seed_invalid")
    digest = payload.pop("sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("early_seed_invalid")
    return payload
