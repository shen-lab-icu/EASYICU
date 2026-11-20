"""A result whose context is typed (V2/V3) can be extended to an article (B34b).

UX on rc_a12: after the B34 seal, 扩展成完整论文 answered 409
``result_append_unavailable``.  ``ResultAppendRuntimeAuthority`` re-read the
sealed context with the plain V1 model, which refuses ``materialized_inputs``,
and the route hid the cause.  The authority now reads it through the versioned
owner, as do the replication reader and both multi-database readers, and the
route logs the cause behind its stable code.  The synthetic fixture writes V1,
so each reader's dispatch is pinned with a spy around the real owner; the typed
case was checked read-only against the UX run.
"""

from __future__ import annotations

import inspect
import logging

import pytest
from fastapi import HTTPException

from easyicu.research_agent.planning.family_spec import baseline_association_template as template
from easyicu.research_agent.replication import metrics as replication_metrics
from easyicu.research_agent.research_context import typed
from easyicu.webserver import research_multi_pipeline, research_multi_results
from easyicu.webserver.routes import research_append as append_route
from tests.support.result_append import completed_source


def _spy(module, name, monkeypatch):
    calls = []
    owner = getattr(typed, name)

    def versioned(raw):
        calls.append(raw)
        return owner(raw)

    monkeypatch.setattr(module, name, versioned)
    return calls


@pytest.mark.parametrize("kind", ["article", "subgroup"])
def test_the_append_authority_reads_its_sealed_context_by_version(tmp_path, monkeypatch, kind) -> None:
    _wrapper, _run, plan, context, baseline = completed_source(tmp_path)
    calls = _spy(template, "parse_research_context_json", monkeypatch)

    authority = template.ResultAppendRuntimeAuthority.create(
        kind=kind, source_run_id="source", source_run_sha256="a" * 64,
        source_plan=plan, context=context, baseline=baseline,
    )
    restored = template.ResultAppendRuntimeAuthority.model_validate_json(authority.model_dump_json())

    assert calls and set(calls) == {context.model_dump_json()}
    assert restored.append_plan_sha256 == authority.append_plan_sha256


def test_the_multi_database_source_reads_its_context_by_version(tmp_path, monkeypatch) -> None:
    _wrapper, _run, plan, context, _ = completed_source(tmp_path)
    design = {"run_id": "source", "source_sha256": "a" * 64,
              "plan": plan.model_dump(mode="json"), "context": context.model_dump(mode="json")}
    calls = _spy(research_multi_pipeline, "parse_research_context", monkeypatch)

    result = research_multi_pipeline.source_result(design)

    assert calls == [design["context"]]
    assert result.context == context


def test_the_other_run_context_readers_use_the_versioned_owner() -> None:
    results = inspect.getsource(research_multi_results)
    replication = inspect.getsource(replication_metrics)

    assert 'parse_research_context(design["context"])' in results
    assert "ResearchContext.model_validate(" not in results
    assert "parse_research_context_json(verified.context_evidence_path.read_bytes())" in replication
    assert "ResearchContext.model_validate_json(" not in replication


def test_a_refused_append_keeps_its_code_and_logs_the_cause(caplog) -> None:
    def refuses(_service):
        raise ValueError("1 validation error for ResultAppendRuntimeAuthority\nmaterialized_inputs\n  secret input")

    with caplog.at_level(logging.WARNING, logger=append_route.__name__), pytest.raises(HTTPException) as refused:
        append_route._call(refuses)  # noqa: SLF001

    assert refused.value.status_code == 409
    assert refused.value.detail == {"error": "result_append_unavailable", "cause": "ValueError"}
    assert "ResultAppendRuntimeAuthority | materialized_inputs" in caplog.text
    assert "secret input" not in caplog.text
