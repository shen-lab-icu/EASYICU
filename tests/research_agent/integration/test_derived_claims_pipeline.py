"""Exercise frozen numeric sources through the actual execution phase."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore


def test_pipeline_derives_from_prior_and_own_current_summaries(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).with_name("test_trajectory_stability_pipeline_success.py")
    spec = importlib.util.spec_from_file_location("_derived_trajectory_fixture", path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    original = fixture._HybridTrajectoryRunner._write_characterization

    def write_with_derivation(self, out_dir, *, resolved_inputs_path):
        original(self, out_dir, resolved_inputs_path=resolved_inputs_path)
        summary_path = out_dir / "step_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["derived_claims"] = [{
            "name": "sum_cluster_counts", "formula": "a + b", "explanation": "sum",
            "sources": {
                "a": {"step_id": "02_candidates", "field": "n_clusters"},
                "b": "n_clusters",
            },
        }]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(fixture._HybridTrajectoryRunner, "_write_characterization", write_with_derivation)
    fixture.test_typed_trajectory_stability_success_is_evidence_bound_and_continues(
        ra, tmp_path, monkeypatch,
    )
    manifests = list(tmp_path.rglob("manifest_partial.json"))
    assert len(manifests) == 1
    evidence = EvidenceStore(manifests[0].parent)
    claims = [claim for claim in evidence.numeric_claims()
              if claim.step_id == "04_characterization" and claim.is_derived]
    assert [(claim.source_field, claim.canonical) for claim in claims] == [
        ("sum_cluster_counts", 4.0),
    ]
