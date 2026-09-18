"""Boundary checks for the extracted write-phase module."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_writer_checkpoint_survives_finalizer_plan_order_without_losing_attempt_authority(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting import write_phase

    records = [
        {"step_id": "b", "status": "ok", "step_summary": {"n": 20}},
        {"step_id": "a", "status": "ok", "step_summary": {"n": 30}},
    ]
    store = EvidenceStore(tmp_path)
    eid = write_phase._preserve_writer_checkpoint(
        "# Partial draft\n\n## Methods\n\nBaseline description.",
        evidence=store, per_step_records=records,
    )
    finalized = list(reversed(records))
    result = write_phase._verified_resume_writer_scaffold_for_quality_migration(
        resume_state={"per_step_records": finalized}, evidence=store,
        run_dir=tmp_path, per_step_records=finalized,
    )
    assert result is not None and result[1]["source_evidence_id"] == eid
    assert write_phase._writer_execution_checkpoint_sha256(records) != write_phase._writer_execution_checkpoint_sha256(
        records + [{"step_id": "b", "status": "failed"}],
    )


@pytest.mark.parametrize('changed_checkpoint,tamper', [(False, False), (True, False), (False, True)])
def test_rejected_writer_candidate_only_resumes_as_verified_repair(tmp_path, changed_checkpoint, tamper):
    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.reporting.manuscript_sections import ManuscriptReaderQualityContractError
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    store = EvidenceStore(tmp_path)
    records = [{'step_id': 'baseline', 'status': 'ok', 'step_summary': {'n': 20}}]
    exc = ManuscriptReaderQualityContractError(
        findings=(('MISSING', 'Methods', 'Admission type omitted'),),
        manuscript='# Diagnostic draft\n\n## Methods\n\nIncomplete prose.',
    )
    eid = write_phase._preserve_rejected_writer_candidate(exc, evidence=store, per_step_records=records)
    record = store.get(eid)
    assert record.metadata['publication_authorized'] is False
    assert record.finding_severity == 'error'
    assert eid not in write_phase._preferred_writer_evidence_names(store, records)
    assert store.get('manuscript_scaffold_raw') is None
    assert write_phase._verified_resume_writer_scaffold(
        resume_state={'per_step_records': records}, evidence=store,
        run_dir=tmp_path, per_step_records=records,
    ) is None
    if tamper:
        (tmp_path / record.relative_path).write_text('altered')
    current = [{'step_id': 'baseline', 'status': 'ok', 'step_summary': {'n': 21}}] if changed_checkpoint else records
    result = write_phase._verified_resume_writer_scaffold_for_quality_migration(
        resume_state={'per_step_records': records}, evidence=store,
        run_dir=tmp_path, per_step_records=current,
    )
    if changed_checkpoint or tamper:
        assert result is None
    else:
        assert result[0] == exc.manuscript
        assert result[1]['source_evidence_id'] == eid


def test_failed_quality_migration_preserves_verified_prior_scaffold(
    monkeypatch,
) -> None:
    from easyicu.research_agent.reporting import write_phase

    prior = "# Prior valid scaffold\n\n## Abstract\n\nPreserved prose."
    detail = {
        "source_evidence_id": "manuscript_scaffold_raw__prior",
        "source_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        write_phase,
        "_verified_resume_writer_scaffold",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        write_phase,
        "_verified_resume_writer_scaffold_for_quality_migration",
        lambda **_kwargs: (prior, detail),
    )
    monkeypatch.setattr(
        write_phase,
        "load_manuscript_administrative_authority",
        lambda _run_dir: None,
    )
    monkeypatch.setattr(
        write_phase,
        "render_writer_literature_digest",
        lambda _literature, **_kwargs: "literature",
    )

    class FailingWriter:
        def repair_existing(self, *_args, **_kwargs):
            raise RuntimeError("bounded provider repair exhausted")

    findings = []
    observed = write_phase._render_or_resume_writer_scaffold(
        writer=FailingWriter(),
        resume_state={},
        evidence=object(),
        run_dir=Path("/tmp/run"),
        per_step_records=(),
        execute_result=SimpleNamespace(plan=object()),
        literature=None,
        agent_context=object(),
        preferred_evidence_names=(),
        writer_evidence_digest="digest",
        findings=findings,
    )

    assert observed == prior
    assert findings[-1].severity == "error"
    assert (
        findings[-1].detail["reason_code"]
        == "WRITER_QUALITY_MIGRATION_FAILED_PRIOR_PRESERVED"
    )


def test_reporting_write_phase_entrypoint_is_importable() -> None:
    from easyicu.research_agent.reporting.write_phase import run_write_phase

    assert callable(run_write_phase)


def test_reporting_write_phase_does_not_import_pipeline_at_module_top() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "easyicu"
        / "research_agent"
        / "reporting"
        / "write_phase.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"pipeline", "easyicu.research_agent.pipeline"}
        for node in top_imports
    )


def test_write_phase_keeps_stages_bounded() -> None:
    """The public phase remains orchestration, not another monolithic owner."""

    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "easyicu"
        / "research_agent"
        / "reporting"
        / "write_phase.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions["run_write_phase"].end_lineno - functions["run_write_phase"].lineno < 300
    for name in (
        "_activate_publication_figure",
        "_activate_publication_inputs",
        "_draft_manuscript",
        "_bind_and_review_manuscript",
        "_publish_and_audit_manuscript",
        "_write_reproducibility_artifacts",
    ):
        function = functions[name]
        assert function.end_lineno - function.lineno < 500, name


def test_publication_figure_activation_precedes_analysis_pause() -> None:
    """A Writer pause must not skip the deterministic article-display suffix."""

    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "easyicu"
        / "research_agent"
        / "reporting"
        / "write_phase.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    run_write_phase = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_write_phase"
    )
    figure_calls = [
        node
        for node in ast.walk(run_write_phase)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_activate_publication_figure"
    ]
    completed_pause = next(
        node
        for node in run_write_phase.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "stop_after_analysis"
    )
    assert len(figure_calls) == 1
    assert figure_calls[0].lineno < completed_pause.lineno
