from __future__ import annotations

import hashlib
import json

from easyicu.webserver import agent_pipeline_runs, agent_runs


def _payload(manuscript: bytes) -> dict:
    return {
        "schema_version": "easyicu.manuscript-provenance/1",
        "artifact_kind": "evidence_bound_manuscript_reader",
        "manuscript_sha256": hashlib.sha256(manuscript).hexdigest(),
        "claim_count": 1,
        "claim_ceiling": "analysis_only",
        "publication_authorized": False,
        "article_blocks": [
            {
                "kind": "paragraph",
                "segments": [{"kind": "claim", "text": "0.5", "claim_id": "claim_1"}],
            }
        ],
        "claims": [
            {
                "claim_id": "claim_1",
                "display_value": "0.5",
                "source_value": "0.5",
                "canonical_value": 0.5,
                "step_id": "primary",
                "source_field": "runtime.spline_knot_quantiles[1]",
                "source_json_pointer": "/runtime/spline_knot_quantiles[1]",
                "evidence": {"evidence_id": "summary", "sha256": "a" * 64},
                "related_artifacts": [],
            }
        ],
        "integrity": {
            "path_values_returned": False,
            "patient_rows_returned": False,
            "raw_data_returned": False,
            "numeric_claims_verified": True,
        },
    }


def test_manuscript_provenance_is_a_fixed_public_json_artifact(tmp_path) -> None:
    manuscript = b"# Bound manuscript\n"
    (tmp_path / "manuscript_scaffold_bound.md").write_bytes(manuscript)
    payload = _payload(manuscript)
    (tmp_path / "manuscript_provenance.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    projected = agent_pipeline_runs._manuscript_provenance_projection(tmp_path)
    loaded = agent_runs.read_run_artifact(str(tmp_path), "manuscript_provenance.json")

    assert projected == payload
    assert loaded["ok"] is True
    assert loaded["payload"] == payload
    assert "manuscript_provenance.json" in agent_runs._RUN_ARTIFACT_NAMES


def test_manuscript_provenance_projection_rejects_stale_source_digest(
    tmp_path,
) -> None:
    manuscript = b"# Bound manuscript\n"
    (tmp_path / "manuscript_scaffold_bound.md").write_bytes(manuscript)
    payload = _payload(manuscript)
    payload["manuscript_sha256"] = "0" * 64
    (tmp_path / "manuscript_provenance.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert agent_pipeline_runs._manuscript_provenance_projection(tmp_path) == {}
