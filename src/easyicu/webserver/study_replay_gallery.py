"""Static gallery from explicitly registered, digest-pinned recorded runs."""

from __future__ import annotations

import json
from pathlib import Path

from .study_replay import build_study_replay
from .study_replay_io import (
    MAX_BUNDLE_BYTES,
    digest,
    identity,
    parse_json,
    read_snapshot,
    relative_parts,
    write_bundle,
)
from .study_replay_policy import ReplayExportError, canonical_json, public_text


def _build_replay_gallery(root: Path, registry: str) -> dict[str, bytes]:
    """Build gallery.json plus replays/<id>/; no study execution or discovery.

    Registry entries pin run_id and source_run_manifest.json SHA-256. Source
    eligibility comes from the run, never from a registry/caller override.
    A fixture/demo cannot establish acceptance on real data, so every entry
    carries real_data_acceptance=not_accepted. The site owns all user copy.
    """
    registered = parse_json(read_snapshot(root, registry))
    if (
        not isinstance(registered, dict)
        or registered.get("schema_version") != "easyicu.replay-gallery-registry/1"
        or not isinstance(registered.get("entries"), list)
        or not 1 <= len(registered["entries"]) <= 50
    ):
        raise ReplayExportError("gallery_registry_invalid")
    entries, files, seen = [], {}, set()
    for row in registered["entries"]:
        if not isinstance(row, dict):
            raise ReplayExportError("gallery_registry_invalid")
        ident = identity(row.get("id"))
        if ident in seen:
            raise ReplayExportError("gallery_registry_invalid")
        seen.add(ident)
        relative = row.get("run_dir")
        relative_parts(relative)
        manifest = read_snapshot(root, relative + "/source_run_manifest.json")
        if digest(manifest) != row.get("source_manifest_sha256"):
            raise ReplayExportError("gallery_registration_mismatch")
        # The exporter owns privacy even for a registered synthetic run.
        bundle = build_study_replay(Path(root) / relative)
        receipt = json.loads(bundle["manifest.json"])
        if receipt["run_id"] != row.get("run_id") or not any(
            item["record"] == "source_run_manifest.json"
            and item["sha256"] == row["source_manifest_sha256"]
            for item in receipt["inputs"]
        ):
            raise ReplayExportError("gallery_registration_mismatch")
        source = receipt["source"]
        public_databases = {
            "synthetic",
            "test",
            "mock",
            "fixture",
            "mimic_demo",
            "eicu_demo",
            source["database"],
        }
        if source["kind"] not in {"synthetic", "public_demo"} or not set(
            receipt["source_databases"]
        ).issubset(public_databases):
            raise ReplayExportError("gallery_source_forbidden")
        replay = json.loads(bundle["replay.json"])
        entries.append(
            {
                "id": ident,
                "title": public_text(row.get("title")),
                "question": replay["question"],
                "database": source["database"],
                "source_kind": source["kind"],
                "study_type": public_text(row.get("study_type")),
                "real_data_acceptance": "not_accepted",
                "run_id": receipt["run_id"],
                "replay": f"replays/{ident}/index.html",
                "replay_manifest_sha256": digest(bundle["manifest.json"]),
            }
        )
        files.update({f"replays/{ident}/{name}": raw for name, raw in bundle.items()})
        if sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
            raise ReplayExportError("replay_size_limit")
    files["gallery.json"] = canonical_json(
        {
            "schema_version": "easyicu.replay-gallery/1",
            "entries": sorted(entries, key=lambda row: row["id"]),
        }
    )
    return dict(sorted(files.items()))


def build_replay_gallery(root: Path, registry: str) -> dict[str, bytes]:
    """Read an explicit registration file and return deterministic static files."""
    try:
        return _build_replay_gallery(root, registry)
    except ReplayExportError:
        raise
    except (
        TypeError,
        ValueError,
        KeyError,
        UnicodeError,
        RecursionError,
        OverflowError,
    ):
        raise ReplayExportError("gallery_registry_invalid") from None


def export_replay_gallery(root: Path, registry: str, destination: Path) -> Path:
    """Publish a new directory for website/; existing site files are untouched."""
    write_bundle(destination, build_replay_gallery(root, registry))
    return Path(destination)
