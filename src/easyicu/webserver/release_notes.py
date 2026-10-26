"""Versioned bilingual release notes; acknowledging a version is a client action."""

from __future__ import annotations

from datetime import date
from importlib.resources import files
import json
import re
from typing import Any


def _version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]{0,5})(?:\.(?:0|[1-9][0-9]{0,5})){2}", value):
        raise ValueError("invalid_version")
    return tuple(map(int, value.split(".")))


def parse_release_notes(raw: str) -> list[dict[str, Any]]:
    """Require one complete zh/en entry per release, newest first on output."""
    try:
        source = json.loads(raw)
        if set(source) != {"schema_version", "entries"} or source["schema_version"] != "easyicu.release-notes/1":
            raise ValueError("schema")
        entries = source["entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("entries")
        versions = set()
        for entry in entries:
            if set(entry) != {"version", "date", "changes"}:
                raise ValueError("entry")
            version = _version(entry["version"])
            if version in versions:
                raise ValueError("duplicate")
            versions.add(version)
            date.fromisoformat(entry["date"])
            if not isinstance(entry["changes"], list) or not entry["changes"]:
                raise ValueError("changes")
            codes = set()
            for change in entry["changes"]:
                if set(change) != {"code", "zh", "en"} or any(
                    not isinstance(value, str) or not value.strip() for value in change.values()
                ) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", change["code"]) or change["code"] in codes:
                    raise ValueError("translation")
                codes.add(change["code"])
        return sorted(entries, key=lambda entry: _version(entry["version"]), reverse=True)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("release_notes_invalid") from exc


def whats_new(last_seen_version: str | None, *, entries: list | None = None) -> dict[str, Any]:
    try:
        seen = _version(last_seen_version) if last_seen_version is not None else None
    except ValueError as exc:
        raise ValueError("last_seen_version_invalid") from exc
    if entries is None:
        entries = parse_release_notes(files("easyicu.data").joinpath("release_notes.json").read_text(encoding="utf-8"))
    newer = [entry for entry in entries if seen is None or _version(entry["version"]) > seen]
    return {"schema_version": "easyicu.whats-new/1", "latest_version": entries[0]["version"],
            "entries": newer, "show_after_update": bool(newer)}
