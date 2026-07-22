from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.discovery.longitudinal_handoff import (
    DEFAULT_LONGITUDINAL_PROTOCOL_CONFIRMATIONS,
    build_longitudinal_analysis_task_pack,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    profiles = []
    for database, id_col in (("miiv", "stay_id"), ("eicu", "patientunitstayid")):
        artifact = tmp_path / database / "sofa2_score.parquet"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(database.encode("utf-8"))
        profiles.append(
            {
                "concept": "sofa2",
                "database": database,
                "artifact_path": str(artifact),
                "artifact_sha256": _sha(artifact),
                "row_count": 10,
                "id_column": id_col,
                "time_column": "charttime",
                "value_column": "sofa2",
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.longitudinal_discovery/1",
                "candidates": [
                    {
                        "concept": "sofa2",
                        "analysis_family": "trajectory_clustering",
                        "artifact_profiles": profiles,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_longitudinal_discovery_builds_six_database_style_child_handoffs(
    tmp_path: Path,
):
    source = _manifest(tmp_path)

    pack = build_longitudinal_analysis_task_pack(
        source,
        output_dir=tmp_path / "task",
        concept="sofa2",
    )

    assert pack.analysis_family == "trajectory_clustering"
    assert pack.go_no_go == "hold"
    assert pack.protocol_status == "awaiting_human_confirmation"
    assert pack.paper_authorized is False
    assert pack.required_protocol_confirmations == list(
        DEFAULT_LONGITUDINAL_PROTOCOL_CONFIRMATIONS
    )
    assert {task.database for task in pack.database_tasks} == {"miiv", "eicu"}
    for task in pack.database_tasks:
        handoff = json.loads(Path(task.handoff_path).read_text(encoding="utf-8"))
        assert handoff["analysis_family"] == "trajectory_clustering"
        assert handoff["resolved_analysis_concepts"] == ["sofa2"]
        assert handoff["target_outcome"] is None
        assert handoff["human_confirmed"] is False
        assert handoff["go_no_go"] == "hold"


def test_longitudinal_task_pack_does_not_guess_protocol_choices(tmp_path: Path):
    pack = build_longitudinal_analysis_task_pack(
        _manifest(tmp_path),
        output_dir=tmp_path / "task",
    )

    serialized = pack.model_dump_json()
    assert "72h" not in serialized
    assert "kmeans" not in serialized.lower()
    assert "time_zero" in pack.required_protocol_confirmations
    assert "cross_database_matching_and_transportability_metric" in (
        pack.required_protocol_confirmations
    )


def test_longitudinal_task_pack_rejects_unknown_concept(tmp_path: Path):
    with pytest.raises(ValueError, match="expected one longitudinal candidate"):
        build_longitudinal_analysis_task_pack(
            _manifest(tmp_path),
            output_dir=tmp_path / "task",
            concept="peep",
        )
