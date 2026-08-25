#!/usr/bin/env python3
"""Audit existing manuscripts without modifying their source run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
    render_reader_manuscript,
)


def _resolve_source(raw: str) -> tuple[str, Path]:
    label, separator, path_text = raw.partition("=")
    if not separator:
        path_text = label
        label = Path(path_text).name
    source = Path(path_text).expanduser().resolve()
    if source.is_dir():
        source = source / "manuscript_scaffold_bound.md"
    if not source.is_file():
        raise FileNotFoundError(f"Bound manuscript not found: {source}")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "manuscript"
    return safe_label, source


def _render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Manuscript quality replay",
        "",
        "This is a deterministic, provider-free writing audit. It does not grant publication authority.",
        "",
        "| Manuscript | Status | Errors | Main codes |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {status} | {error_count} | {codes} |".format(
                label=row["label"],
                status=row["status"],
                error_count=row["error_count"],
                codes=", ".join(row["error_codes"]) or "none",
            )
        )
    lines.extend(["", "Provider calls: 0", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Manuscript file/run directory, optionally LABEL=PATH.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for raw in args.inputs:
        label, source = _resolve_source(raw)
        text = source.read_text(encoding="utf-8")
        audit = audit_manuscript_quality(text)
        audit_path = output_dir / f"{label}_manuscript_quality_audit.json"
        reader_path = output_dir / f"{label}_manuscript_reader.md"
        audit_path.write_text(
            json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        reader_path.write_text(render_reader_manuscript(text), encoding="utf-8")
        errors = [finding for finding in audit.findings if finding.severity == "error"]
        rows.append(
            {
                "label": label,
                "source": str(source),
                "source_sha256": audit.source_sha256,
                "reader_sha256": audit.reader_sha256,
                "status": audit.status,
                "error_count": len(errors),
                "error_codes": sorted({finding.code for finding in errors}),
                "audit": str(audit_path),
                "reader": str(reader_path),
            }
        )

    payload = {
        "schema_version": "manuscript-quality-replay-v1",
        "provider_calls": 0,
        "publication_authority": False,
        "manuscripts": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_summary(rows), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
