"""Copy recorded run metadata into the existing wrapper manifest on completion.

No source export or patient table is opened. Versions come from execution
receipts, never from the machine currently serving an old result. Missing
metadata remains missing, including source identity fields absent before ec2.
"""

from __future__ import annotations

import json

from easyicu.research_agent.authority.runtime_artifacts import current_successful_step_records

from .research_evidence_preview import EvidencePreviewError, _read_regular_file
from .scientific_review_items import mapping, pointer, rows, safe_value, token

_FIELDS = {"package_versions", "python_version", "data_property_notes", "limitations", "diagnostics", "claim_ceiling"}
MAX_RUNTIME_ROWS = 128


def _recorded_fields(value, path=(), depth=0):
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FIELDS:
                yield (*path, key), child
            elif isinstance(child, (dict, list)):
                yield from _recorded_fields(child, (*path, key), depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _recorded_fields(child, (*path, index), depth + 1)


def recorded_review_metadata(run_dir, source) -> dict:
    """Bound the current successful summaries already held by the run ledger."""
    source = mapping(source)
    identity = {key: safe_value(source[key]) for key in ("database", "version", "export_id") if source.get(key)}
    if "version" not in identity and source.get("database_version"):
        identity["version"] = safe_value(source["database_version"])
    result = {"source": identity, "runtime": [], "status": "unavailable", "truncated": False}
    if run_dir is None:
        return result
    try:
        manifest = json.loads(_read_regular_file(run_dir / "manifest.json", max_bytes=4 * 1024 * 1024))
    except (EvidencePreviewError, ValueError, UnicodeDecodeError):
        return result
    records = rows(mapping(manifest).get("per_step_records"))
    current = current_successful_step_records([row for row in records if isinstance(row, dict)])
    for record in current:
        for path, value in _recorded_fields(mapping(record.get("step_summary"))):
            if len(result["runtime"]) == MAX_RUNTIME_ROWS:
                result["truncated"] = True
                break
            result["runtime"].append({"field": path[-1], "value": safe_value(value),
                                      "step_id": token(record.get("step_id")), "summary_pointer": pointer(*path)})
    result["status"] = "recorded"
    return result
