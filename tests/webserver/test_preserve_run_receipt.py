"""Contract tests for tools/preserve_run_receipt.py.

The tool exists so a pruned run stays provable. These tests lock the two
properties that make that true: the receipt copies the run's own governance
verdict verbatim (it never upgrades a status), and verification fails closed
when the run tree it describes has moved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority import run_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "preserve_run_receipt.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("preserve_run_receipt", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A minimal run tree shaped like a real research-agent run directory."""

    run = tmp_path / "run_20260831T034515_94e17c"
    _write_json(
        run / "manifest.json",
        {
            "schema_version": "easyicu.research_manifest/1",
            "run_id": "run_20260831T034515_94e17c",
            "research_question": "Is SOFA associated with ICU mortality?",
            "started_at": "2026-08-31T03:45:15.348113Z",
            "finished_at": "2026-08-31T03:45:17.082194Z",
            "current_plan_authority": {"evidence_id": "analysis_plan"},
        },
    )
    _write_json(
        run / "run_status.json",
        {
            "schema_version": "easyicu.run_status/2",
            "status": "analysis_only",
            "strict_fail_closed": True,
            "gates": {"execution_complete": True, "failed_steps": []},
            "code_version": {"git_sha": "82896fb", "git_dirty": True},
        },
    )
    _write_json(
        run / ".easyicu_evidence_authority_head.json",
        {"generation": 89, "head_sha256": "671da4cd"},
    )
    _write_json(run / "steps" / "01_table_one" / "step_summary.json", {"status": "ok"})
    (run / "evidence").mkdir(parents=True, exist_ok=True)
    (run / "evidence" / "analysis_plan__analysis_plan.json").write_text(
        '{"plan": 1}', encoding="utf-8"
    )
    return run


def test_receipt_copies_the_runs_own_governance_verdict(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)

    assert receipt["run_id"] == "run_20260831T034515_94e17c"
    assert receipt["status"] == "analysis_only"
    assert receipt["strict_fail_closed"] is True
    assert receipt["gates"] == {"execution_complete": True, "failed_steps": []}
    assert receipt["code_version"]["git_sha"] == "82896fb"
    assert receipt["evidence_authority_head"]["generation"] == 89
    assert receipt["steps"] == [{"step": "01_table_one", "status": "ok"}]


def test_receipt_never_upgrades_a_status(run_dir: Path) -> None:
    """A receipt records readiness; it must not manufacture it."""

    receipt = tool.build_receipt(run_dir)

    assert receipt["status"] == "analysis_only"
    assert "publication_ready" not in receipt
    assert "paper_authorized" not in receipt


def test_receipt_inventories_every_artifact_with_a_digest(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)

    paths = {row["path"] for row in receipt["artifacts"]}
    assert "manifest.json" in paths
    assert "evidence/analysis_plan__analysis_plan.json" in paths
    assert "steps/01_table_one/step_summary.json" in paths
    assert receipt["artifact_count"] == len(receipt["artifacts"])
    assert receipt["evidence_file_count"] == 1
    assert all(len(row["sha256"]) == 64 for row in receipt["artifacts"])


def test_receipt_is_byte_stable_across_builds(run_dir: Path) -> None:
    first = tool.build_receipt(run_dir)
    second = tool.build_receipt(run_dir)

    assert first == second
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_verify_accepts_an_unchanged_run_tree(run_dir: Path) -> None:
    assert tool.verify_receipt(run_dir, tool.build_receipt(run_dir)) == []


def test_verify_fails_closed_when_the_receipt_was_edited(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)
    receipt["status"] = "publication_ready"

    findings = tool.verify_receipt(run_dir, receipt)

    assert any(
        finding.startswith(tool.RUN_RECEIPT_SELF_DIGEST_MISMATCH)
        for finding in findings
    )


def test_verify_fails_closed_when_an_artifact_changed(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)
    (run_dir / "evidence" / "analysis_plan__analysis_plan.json").write_text(
        '{"plan": 2}', encoding="utf-8"
    )

    findings = tool.verify_receipt(run_dir, receipt)

    assert len(findings) == 1
    assert findings[0].startswith(tool.RUN_RECEIPT_DIGEST_MISMATCH)


def test_verify_fails_closed_when_an_artifact_vanished(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)
    (run_dir / "evidence" / "analysis_plan__analysis_plan.json").unlink()

    findings = tool.verify_receipt(run_dir, receipt)

    assert len(findings) == 1
    assert findings[0].startswith(tool.RUN_RECEIPT_ARTIFACT_MISSING)


def test_verify_fails_closed_when_an_unrecorded_artifact_appears(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)
    (run_dir / "unexpected.json").write_text("{}", encoding="utf-8")

    findings = tool.verify_receipt(run_dir, receipt)

    assert findings == [f"{tool.RUN_RECEIPT_ARTIFACT_UNRECORDED}: unexpected.json"]


def test_verify_rejects_unsafe_artifact_paths_without_reading_them(
    run_dir: Path,
) -> None:
    receipt = tool.build_receipt(run_dir)
    receipt["artifacts"][0]["path"] = "../../outside.json"
    receipt["receipt_sha256"] = tool._receipt_sha256(receipt)

    findings = tool.verify_receipt(run_dir, receipt)

    assert any(
        finding.startswith(tool.RUN_RECEIPT_ARTIFACT_PATH_UNSAFE)
        for finding in findings
    )


def test_verify_rejects_an_unsupported_receipt_schema(run_dir: Path) -> None:
    receipt = tool.build_receipt(run_dir)
    receipt["schema_version"] = "easyicu.run_receipt/999"

    findings = tool.verify_receipt(run_dir, receipt)

    assert findings and findings[0].startswith(tool.RUN_RECEIPT_SCHEMA_UNSUPPORTED)


def test_build_names_the_missing_file_it_needs(run_dir: Path) -> None:
    (run_dir / "manifest.json").unlink()

    with pytest.raises(tool.ReceiptError) as excinfo:
        tool.build_receipt(run_dir)

    assert excinfo.value.code == tool.RUN_RECEIPT_MANIFEST_MISSING


def test_build_rejects_a_run_dir_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(tool.ReceiptError) as excinfo:
        tool.build_receipt(tmp_path / "nope")

    assert excinfo.value.code == tool.RUN_RECEIPT_RUN_DIR_MISSING


def test_cli_round_trips_build_then_verify(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "receipts" / "run.json"

    assert tool.main([str(run_dir), "--out", str(out)]) == 0
    assert out.is_file()
    assert tool.main([str(run_dir), "--verify", str(out)]) == 0

    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    assert tool.main([str(run_dir), "--verify", str(out)]) == 1


def test_terminal_receipts_are_durable_immutable_versions(run_dir, tmp_path):
    root = tmp_path / "durable"
    first = run_receipt.preserve_terminal_run_receipt(run_dir, destination_root=root)
    original = first.read_bytes()
    assert first == run_receipt.preserve_terminal_run_receipt(run_dir, destination_root=root)
    assert not first.is_relative_to(run_dir)
    _write_json(run_dir / "later_signoff.json", {"scope": "analysis_only"})
    second = run_receipt.preserve_terminal_run_receipt(run_dir, destination_root=root)
    assert second != first
    assert first.read_bytes() == original
    assert tool.verify_receipt(run_dir, json.loads(second.read_bytes())) == []


def test_receipt_cannot_overwrite_an_existing_different_receipt(run_dir, tmp_path):
    out = tmp_path / "receipt.json"
    assert tool.main([str(run_dir), "--out", str(out)]) == 0
    original = out.read_bytes()
    _write_json(run_dir / "added.json", {"status": "new"})
    assert tool.main([str(run_dir), "--out", str(out)]) == 1
    assert out.read_bytes() == original


def test_receipt_cannot_be_written_inside_its_own_inventory(run_dir):
    assert tool.main([str(run_dir), "--out", str(run_dir / "receipt.json")]) == 1
    assert not (run_dir / "receipt.json").exists()


def test_receipt_facts_are_checked_even_when_self_digest_was_recomputed(run_dir):
    receipt = tool.build_receipt(run_dir)
    receipt["status"] = "publication_ready"
    receipt["receipt_sha256"] = tool._receipt_sha256(receipt)
    assert any(run_receipt.RUN_RECEIPT_FACTS_MISMATCH in item for item in tool.verify_receipt(run_dir, receipt))


def test_receipt_refuses_corrupt_present_authority_head(run_dir):
    (run_dir / run_receipt.AUTHORITY_HEAD_NAME).write_text("broken")
    with pytest.raises(tool.ReceiptError, match="RUN_RECEIPT_UNREADABLE_JSON"):
        tool.build_receipt(run_dir)


def test_receipt_refuses_source_drift_during_capture(run_dir, monkeypatch):
    read = run_receipt._read_json
    def changed_read(path, **kwargs):
        result = read(path, **kwargs)
        if path.name == "manifest.json":
            path.write_text("{}")
        return result
    monkeypatch.setattr(run_receipt, "_read_json", changed_read)
    with pytest.raises(tool.ReceiptError, match="RUN_RECEIPT_SOURCE_CHANGED"):
        tool.build_receipt(run_dir)


def test_pipeline_completion_automatically_preserves_terminal_receipt(run_dir, tmp_path, monkeypatch):
    from easyicu.research_agent.pipeline import PipelineResult, ResearchAgentPipeline
    from easyicu.research_agent.orchestration.workflow import WorkflowCompleted

    root = tmp_path / "retained"
    monkeypatch.setenv("EASYICU_RUN_RECEIPT_ROOT", str(root))
    pipeline = object.__new__(ResearchAgentPipeline)
    result = PipelineResult(
        run_id=run_dir.name, workdir=str(run_dir), context_path="", plan_path="",
        manifest_path=str(run_dir / "manifest.json"), report_path="", manuscript_path="",
        evidence_count=1, findings_count=0,
    )
    actual = pipeline._pipeline_result_or_pending(
        WorkflowCompleted(final_result=result), workflow=None,
        run_id=run_dir.name, run_dir=run_dir,
    )
    assert actual is result
    saved, = (root / run_dir.name).glob("*.json")
    assert tool.verify_receipt(run_dir, json.loads(saved.read_bytes())) == []
