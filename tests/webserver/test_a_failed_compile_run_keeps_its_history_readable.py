"""A run stopped at progressive compile keeps the run history readable.

UX walk (2026-11-19): /api/agent-runs/history returned HTTP 500 for every
study with a progressive-compile stop, so none of that study's run files
loaded. The stop's gate detail nests a ``plan_repair`` receipt; the history
row copied the frozen gate payload shallowly, so the nested read-only mapping
reached the JSON response, which cannot encode it. Synthetic run; no
benchmark item.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic_core import to_json

from easyicu.webserver import agent_runs

_RUN = "run_synthetic_compile_stop"
_STUDY = "study_synthetic_compile_stop"


def _write(run_dir: Path, name: str, payload: dict) -> dict:
    data = json.dumps(payload, sort_keys=True).encode()
    (run_dir / name).write_bytes(data)
    return {"name": name, "kind": "json", "media_type": "application/json",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _compile_stop(project_root: Path) -> None:
    run_dir = project_root / _STUDY / _RUN
    run_dir.mkdir(parents=True)
    engine = "easyicu.research_agent.pipeline"
    artifacts = [
        _write(run_dir, "run_context.json", {
            "engine": engine, "mode": "research_agent_pipeline", "run_id": _RUN, "run_type": "full",
            "scientific_configuration_sha256": "a" * 64, "study_id": _STUDY,
            "summary": {"analysis_started": False, "execution_complete": False},
        }),
        _write(run_dir, "quality_gate.json", {"gate": {
            "checks": [{"id": "research_pipeline_execution", "label": "Research Agent pipeline reached a governed result",
                        "passed": False, "reason_code": "research_pipeline_progressive_compile_failed"}],
            "detail": {"plan_repair": {"limit": 2, "outcome": "repeated_failure", "repairs": 1},
                       "reason_code": "progressive_article_required_roles_missing", "step_id": "summarize"},
            "draft_unlocked": False, "next_actions": [], "reason": "research_pipeline_progressive_compile_failed",
            "reportable": False, "status": "blocked",
        }, "quality": []}),
        _write(run_dir, "source_run_manifest.json", {
            "analysis_started": False, "engine": engine, "failure_code": "research_pipeline_progressive_compile_failed",
            "failure_type": "error", "path_values_returned": False, "publication_authorized": False,
            "run_id": _RUN, "status": "failed",
        }),
    ]
    _write(run_dir, "evidence_ledger.json", {
        "artifacts": artifacts, "engine": engine, "pipeline_evidence_count": 0,
        "privacy": {"path_values_returned": False, "patient_rows_in_projection": False, "projection_scan_passed": True},
        "run_id": _RUN, "run_type": "full", "schema_version": "easyicu.web-research-pipeline-ledger/1", "status": "blocked",
    })


def test_a_compile_stop_with_a_repair_receipt_serializes_in_the_history(tmp_path: Path) -> None:
    _compile_stop(tmp_path)
    history = agent_runs.list_run_history(study_id=_STUDY, project_root=str(tmp_path))
    (row,) = history["runs"]
    assert row["gate_detail"]["plan_repair"] == {"limit": 2, "outcome": "repeated_failure", "repairs": 1}
    assert type(row["gate_detail"]["plan_repair"]) is dict  # a fresh container, not the frozen snapshot
    json.loads(to_json(history))  # what the route's JSON response encodes
