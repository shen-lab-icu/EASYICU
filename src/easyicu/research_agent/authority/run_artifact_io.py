"""File access for run validity evidence and artifact aliases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from .runtime_artifacts import (
    current_run_evidence_paths,
    current_successful_step_records,
    load_run_artifact_authority,
)


def iter_step_summaries(run_dir: Path) -> list[Mapping[str, object]]:
    authority = load_run_artifact_authority(run_dir)
    if authority is not None:
        records = authority.get("per_step_records")
        records = records if isinstance(records, list) else []
        return [
            summary
            for record in current_successful_step_records(records)
            if isinstance((summary := record.get("step_summary")), Mapping)
        ]
    summaries: list[Mapping[str, object]] = []
    for pattern in ("steps/*/outputs/step_summary.json", "steps/*/step_summary.json"):
        for path in run_dir.glob(pattern):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(summary, dict):
                summaries.append(summary)
    return summaries


def find_artifacts(run_dir: Path, *substrings: str) -> list[Path]:
    needles = tuple(substring.lower() for substring in substrings)
    hits: list[Path] = []
    current_paths = current_run_evidence_paths(run_dir)
    candidates = current_paths if current_paths is not None else run_dir.rglob("*")
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
            continue
        name = path.name.lower()
        if any(needle in name for needle in needles):
            hits.append(path)
    return hits


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, ValueError):
        return []


def write_artifact_aliases(run_dir: Path, aliases: Mapping[str, str]) -> None:
    for alias, relative_path in aliases.items():
        source = run_dir / relative_path
        (run_dir / alias).write_bytes(source.read_bytes())
