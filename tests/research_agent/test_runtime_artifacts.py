from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


def test_workflow_graph_and_replay_bundle_build(ra, tmp_path: Path):
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "age": [60, 70, 80],
            "death": [0, 1, 0],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="Predict death",
        cohort=df,
        cohort_name="demo",
        database="synthetic",
        target_outcome="death",
    )
    plan = ra.schema.AnalysisPlan(
        research_question="Predict death",
        steps=[
            ra.schema.AnalysisStep(step_id="01_table_one", intent="Describe cohort")
        ],
    )
    records = [
        {
            "step_id": "01_table_one",
            "status": "ok",
            "generation_mode": "llm",
            "evidence_ids": ["table_one"],
        }
    ]
    graph = ra.build_workflow_graph(
        run_id="run_demo",
        context=ctx,
        plan=plan,
        per_step_records=records,
        paused_after_analysis=True,
    )
    assert any(n.node_id == "context" for n in graph.nodes)
    mermaid = ra.render_workflow_graph_mermaid(graph)
    assert "flowchart TD" in mermaid

    replay = ra.build_execution_replay(
        run_id="run_demo",
        cohort_path=cohort_path,
        context_path="research_context.json",
        plan_path="analysis_plan.json",
        llm_signature="mock",
        prompt_pack_version="v1",
        per_step_records=records,
        findings=[],
        evidence_ids=["table_one"],
    )
    assert replay.run_id == "run_demo"
    assert replay.steps[0].step_id == "01_table_one"


def test_capture_code_version_reports_git_and_package_identity(ra):
    """The run manifest must be tie-able back to the code that produced it.

    In this repo checkout git identity is available; assert the shape and
    that it degrades to a dict (never raises). package_version comes from
    the installed easyicu metadata."""
    from easyicu.research_agent.authority.runtime_artifacts import capture_code_version

    cv = capture_code_version()
    # In a git checkout with the package installed, capture returns a dict.
    assert cv is not None
    assert set(cv.keys()) == {
        "git_sha",
        "git_branch",
        "git_dirty",
        "package_version",
    }
    # git_dirty is a bool (or None if git was unavailable, but here it is a
    # real checkout so it must be a concrete bool).
    assert isinstance(cv["git_dirty"], bool)
    # A sha, when present, is a 40-char hex string.
    if cv["git_sha"] is not None:
        assert len(cv["git_sha"]) == 40
        int(cv["git_sha"], 16)  # parses as hex


def test_code_version_manifest_field_wraps_capture(ra):
    from easyicu.research_agent.orchestration.finalize import (
        _code_version_manifest_fields,
    )

    fields = _code_version_manifest_fields()
    assert "code_version" in fields
    # Either a populated dict or None — but the key is always present so the
    # manifest schema field is populated deterministically.
    assert fields["code_version"] is None or isinstance(fields["code_version"], dict)


def test_current_artifact_authority_uses_latest_outer_step_status():
    from easyicu.research_agent.authority.runtime_artifacts import (
        active_step_evidence_ids,
        current_evidence_records,
        current_step_records,
        current_successful_step_ids,
    )

    records = [
        {"step_id": "01_current", "status": "ok", "evidence_ids": ["keep"]},
        {"step_id": "02_retried", "status": "ok", "evidence_ids": ["old_ok"]},
        {
            "step_id": "02_retried",
            "status": "blocked_by_concept_audit",
            "evidence_ids": ["rejected_checkpoint"],
        },
    ]

    latest = {record["step_id"]: record for record in current_step_records(records)}
    assert latest["02_retried"]["status"] == "blocked_by_concept_audit"
    assert current_successful_step_ids(records) == {"01_current"}
    assert active_step_evidence_ids(records) == {"keep"}

    evidence = [
        {"evidence_id": "run_context", "produced_by_step": None},
        {"evidence_id": "keep", "produced_by_step": "01_current"},
        {"evidence_id": "old_ok", "produced_by_step": "02_retried"},
        {
            "evidence_id": "rejected_checkpoint",
            "produced_by_step": "02_retried",
        },
    ]
    current = current_evidence_records(evidence, records)
    assert {record["evidence_id"] for record in current} == {
        "run_context",
        "keep",
    }


def test_current_evidence_requires_its_own_current_producer_binding():
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_evidence_records,
    )

    records = [
        {
            "step_id": "01_current",
            "status": "ok",
            "evidence_ids": ["cross_owned"],
        },
        {
            "step_id": "02_failed",
            "status": "contract_failed",
            "evidence_ids": [],
        },
    ]
    evidence = [
        {
            "evidence_id": "cross_owned",
            "produced_by_step": "02_failed",
        }
    ]

    assert current_evidence_records(evidence, records) == []


def test_newer_final_manifest_supersedes_stale_partial_authority(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import (
        load_run_artifact_authority,
    )

    partial = tmp_path / "manifest_partial.json"
    final = tmp_path / "manifest.json"
    partial.write_text(
        json.dumps({"per_step_records": [{"step_id": "01_model", "status": "ok"}]}),
        encoding="utf-8",
    )
    final.write_text(
        json.dumps(
            {"per_step_records": [{"step_id": "01_model", "status": "contract_failed"}]}
        ),
        encoding="utf-8",
    )
    os.utime(partial, ns=(1_000_000_000, 1_000_000_000))
    os.utime(final, ns=(2_000_000_000, 2_000_000_000))

    authority = load_run_artifact_authority(tmp_path)

    assert authority is not None
    assert authority["per_step_records"][-1]["status"] == "contract_failed"


def test_corrupt_newest_manifest_cannot_replay_older_success(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import (
        RunArtifactAuthorityError,
        load_run_artifact_authority,
    )

    final = tmp_path / "manifest.json"
    partial = tmp_path / "manifest_partial.json"
    final.write_text(
        json.dumps({"per_step_records": [{"step_id": "01_model", "status": "ok"}]}),
        encoding="utf-8",
    )
    partial.write_text("{corrupt newest checkpoint", encoding="utf-8")
    os.utime(final, ns=(1_000_000_000, 1_000_000_000))
    os.utime(partial, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(
        RunArtifactAuthorityError,
        match=r"newest checkpoint.*manifest_partial\.json.*corrupt",
    ):
        load_run_artifact_authority(tmp_path)


def test_corrupt_only_manifest_is_not_reported_as_legacy(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import (
        RunArtifactAuthorityError,
        current_run_evidence_records,
    )

    (tmp_path / "manifest_partial.json").write_text(
        "{corrupt checkpoint",
        encoding="utf-8",
    )

    with pytest.raises(RunArtifactAuthorityError, match="corrupt"):
        current_run_evidence_records(tmp_path)


def test_newest_manifest_missing_ledger_cannot_replay_older_success(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import (
        RunArtifactAuthorityError,
        load_run_artifact_authority,
    )

    final = tmp_path / "manifest.json"
    partial = tmp_path / "manifest_partial.json"
    final.write_text(
        json.dumps({"per_step_records": [{"step_id": "01_model", "status": "ok"}]}),
        encoding="utf-8",
    )
    partial.write_text(json.dumps({"evidence": []}), encoding="utf-8")
    os.utime(final, ns=(1_000_000_000, 1_000_000_000))
    os.utime(partial, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(
        RunArtifactAuthorityError,
        match=r"newest checkpoint.*does not declare per_step_records",
    ):
        load_run_artifact_authority(tmp_path)


def test_missing_manifests_keep_legacy_authority_signal(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_run_evidence_records,
        load_run_artifact_authority,
    )

    assert load_run_artifact_authority(tmp_path) is None
    assert current_run_evidence_records(tmp_path) is None


def test_checkpoint_writer_is_atomic_and_monotonically_sequenced(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import write_run_checkpoint

    first = write_run_checkpoint(
        tmp_path / "manifest_partial.json",
        {"per_step_records": [{"step_id": "01", "status": "ok"}]},
    )
    second = write_run_checkpoint(
        tmp_path / "manifest.json",
        {"per_step_records": [{"step_id": "01", "status": "contract_failed"}]},
    )

    assert (first, second) == (1, 2)
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["checkpoint_sequence"] == 2
    assert not list(tmp_path.glob(".manifest*.tmp"))


def test_checkpoint_sequence_outranks_mtime_for_current_authority(tmp_path: Path):
    from easyicu.research_agent.authority.runtime_artifacts import (
        load_run_artifact_authority,
    )

    partial = tmp_path / "manifest_partial.json"
    final = tmp_path / "manifest.json"
    partial.write_text(
        json.dumps(
            {
                "checkpoint_sequence": 3,
                "per_step_records": [{"step_id": "01", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    final.write_text(
        json.dumps(
            {
                "checkpoint_sequence": 4,
                "per_step_records": [{"step_id": "01", "status": "contract_failed"}],
            }
        ),
        encoding="utf-8",
    )
    os.utime(final, ns=(1_000_000_000, 1_000_000_000))
    os.utime(partial, ns=(2_000_000_000, 2_000_000_000))

    authority = load_run_artifact_authority(tmp_path)

    assert authority is not None
    assert authority["checkpoint_sequence"] == 4
    assert authority["per_step_records"][0]["status"] == "contract_failed"


def test_final_manifest_external_history_is_digest_verified_and_hydrated(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import (
        STEP_ATTEMPT_HISTORY_REF_SCHEMA,
        encode_step_attempt_history_jsonl,
        load_run_artifact_authority,
        write_run_checkpoint,
    )

    history = [
        {"step_id": "01_model", "status": "candidate_checkpointed", "large": "x" * 500},
        {"step_id": "01_model", "status": "ok", "large": "y" * 500},
    ]
    evidence = EvidenceStore(tmp_path)
    record = evidence.register_text(
        kind="log",
        description="External append-only step attempt history.",
        text=encode_step_attempt_history_jsonl(history),
        filename="step_attempt_history.jsonl",
        evidence_id="step_attempt_history",
        producer="pipeline",
        generation_mode="system",
        publish_aliases=False,
    )
    write_run_checkpoint(
        tmp_path / "manifest.json",
        {
            "per_step_records": [history[-1]],
            "step_attempt_history": [],
            "step_attempt_history_ref": {
                "schema_version": STEP_ATTEMPT_HISTORY_REF_SCHEMA,
                "format": "jsonl",
                "evidence_id": record.evidence_id,
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "record_count": len(history),
            },
            "evidence": [record.model_dump(mode="json")],
        },
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_attempt_history"] == []
    authority = load_run_artifact_authority(tmp_path)
    assert authority is not None
    assert authority["step_attempt_history"] == history


def test_external_history_tampering_fails_closed(tmp_path: Path) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import (
        STEP_ATTEMPT_HISTORY_REF_SCHEMA,
        RunArtifactAuthorityError,
        encode_step_attempt_history_jsonl,
        load_run_artifact_authority,
        write_run_checkpoint,
    )

    history = [{"step_id": "01_model", "status": "ok"}]
    evidence = EvidenceStore(tmp_path)
    record = evidence.register_text(
        kind="log",
        description="External append-only step attempt history.",
        text=encode_step_attempt_history_jsonl(history),
        filename="step_attempt_history.jsonl",
        evidence_id="step_attempt_history",
        producer="pipeline",
        generation_mode="system",
        publish_aliases=False,
    )
    write_run_checkpoint(
        tmp_path / "manifest.json",
        {
            "per_step_records": history,
            "step_attempt_history": [],
            "step_attempt_history_ref": {
                "schema_version": STEP_ATTEMPT_HISTORY_REF_SCHEMA,
                "format": "jsonl",
                "evidence_id": record.evidence_id,
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "record_count": 1,
            },
            "evidence": [record.model_dump(mode="json")],
        },
    )
    (tmp_path / record.relative_path).write_text(
        '{"step_id":"01_model","status":"forged"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RunArtifactAuthorityError, match="digest verification"):
        load_run_artifact_authority(tmp_path)
