from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.schema import ValidationFinding


def _finding(validator: str) -> ValidationFinding:
    return ValidationFinding(
        validator=validator,
        severity="warning",
        message=f"{validator} finding",
    )


def _install_successful_article_contract(
    monkeypatch: pytest.MonkeyPatch,
    article_audit: object,
) -> None:
    monkeypatch.setattr(
        article_audit,
        "summarize_article_contract_coverage",
        lambda **_kwargs: {"complete": True},
    )
    monkeypatch.setattr(
        article_audit,
        "article_contract_audit_payload",
        lambda status: {"status": status},
    )
    monkeypatch.setattr(
        article_audit,
        "validate_run_against_article_contract",
        lambda **_kwargs: [_finding("article_result")],
    )


def test_collect_run_article_audits_combines_findings_and_binds_final_plan_family(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import article_audit

    _install_successful_article_contract(monkeypatch, article_audit)
    family_calls: list[str] = []
    figure_calls: list[object] = []

    def resolve_family(analysis_type: str) -> str:
        family_calls.append(analysis_type)
        return "final-plan-family"

    def validate_figures(**kwargs: object) -> list[ValidationFinding]:
        figure_calls.append(kwargs["analysis_family"])
        return [_finding("figure_result")]

    monkeypatch.setattr(
        article_audit,
        "study_design_family_for_analysis_type",
        resolve_family,
    )
    monkeypatch.setattr(
        article_audit,
        "validate_run_against_article_figure_strategy",
        validate_figures,
    )

    result = article_audit.collect_run_article_audits(
        context=object(),
        plan=SimpleNamespace(analysis_type="final-analysis-type"),
        evidence_records=(),
        per_step_records=(),
        run_dir=tmp_path,
    )

    assert [item.validator for item in result.findings] == [
        "article_result",
        "figure_result",
    ]
    assert family_calls == ["final-analysis-type"]
    assert figure_calls == ["final-plan-family"]
    assert result.artifact is not None
    assert result.artifact.evidence_id == "article_contract_audit"
    assert result.manifest_items == (
        ("article_contract_audit", "article_contract_audit.json"),
    )
    assert json.loads(result.artifact.source_path.read_text(encoding="utf-8")) == {
        "status": {"complete": True}
    }


def test_article_contract_failure_does_not_suppress_figure_audit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import article_audit

    def fail_article(**_kwargs: object) -> None:
        raise RuntimeError("article boom")

    monkeypatch.setattr(
        article_audit,
        "summarize_article_contract_coverage",
        fail_article,
    )
    monkeypatch.setattr(
        article_audit,
        "validate_run_against_article_figure_strategy",
        lambda **_kwargs: [_finding("figure_result")],
    )

    result = article_audit.collect_run_article_audits(
        context=object(),
        plan=None,
        evidence_records=(),
        per_step_records=(),
        run_dir=tmp_path,
    )

    assert [item.validator for item in result.findings] == [
        "article_analysis_contract",
        "figure_result",
    ]
    assert "RuntimeError: article boom" in result.findings[0].message
    assert result.artifact is None
    assert result.manifest_items == ()


def test_figure_audit_failure_preserves_article_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import article_audit

    _install_successful_article_contract(monkeypatch, article_audit)

    def fail_figure(**_kwargs: object) -> None:
        raise ValueError("figure boom")

    monkeypatch.setattr(
        article_audit,
        "validate_run_against_article_figure_strategy",
        fail_figure,
    )

    result = article_audit.collect_run_article_audits(
        context=object(),
        plan=None,
        evidence_records=(),
        per_step_records=(),
        run_dir=tmp_path,
    )

    assert [item.validator for item in result.findings] == [
        "article_result",
        "article_figure_strategy",
    ]
    assert "ValueError: figure boom" in result.findings[1].message
    assert result.artifact is not None
    assert result.manifest_items


def test_execute_phase_delegates_article_audits_to_the_owner_module() -> None:
    from easyicu.research_agent.execution.phase import run_execute_phase

    source = inspect.getsource(run_execute_phase)
    assert "collect_run_article_audits(" in source
    assert "summarize_article_contract_coverage(" not in source
    assert "validate_run_against_article_figure_strategy(" not in source


def test_execute_phase_host_persists_the_owner_artifact_and_manifest(tmp_path) -> None:
    from easyicu.research_agent.execution.article_audit import (
        ArticleAuditArtifact,
        RunArticleAuditResult,
    )
    from easyicu.research_agent.execution.phase import (
        _persist_run_article_audit_result,
    )

    registered: list[dict[str, object]] = []
    flushed: list[dict[str, object]] = []

    class Store:
        def get(self, evidence_id: str) -> None:
            assert evidence_id == "article_contract_audit"
            return None

        def register_file(self, **kwargs: object) -> None:
            registered.append(kwargs)

    result = RunArticleAuditResult(
        findings=(),
        artifact=ArticleAuditArtifact(
            evidence_id="article_contract_audit",
            kind="log",
            description="article audit",
            source_path=Path(tmp_path / "article_contract_audit.json"),
            producer="article_contract",
            generation_mode="system",
        ),
        manifest_items=(("article_contract_audit", "article_contract_audit.json"),),
    )

    persistence_findings = _persist_run_article_audit_result(
        result=result,
        evidence_store=Store(),
        flush_partial_manifest=flushed.append,
    )

    assert persistence_findings == ()
    assert registered[0]["evidence_id"] == "article_contract_audit"
    assert flushed == [{"article_contract_audit": "article_contract_audit.json"}]
