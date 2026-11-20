"""A finished run's typed research context seals as an append source (B34, UX 2026-11-20).

UX on rc_78a: a baseline association study completed (B27 PASS), but 「扩展成完整论文」
never appeared.  The completion hook sealed the source by reading the run's
research context with the plain V1 model, while every executed run writes its
typed V2/V3 context with ``materialized_inputs``: "Extra inputs are not
permitted", swallowed as an unsupported source, and the article append was
unavailable for every executed run.  The reader now uses the versioned owner.
The synthetic fixture writes a V1 context, so the dispatch is pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from easyicu.research_agent.research_context import typed
from easyicu.research_agent.schema import ResearchContext
from easyicu.webserver import research_append_source
from easyicu.webserver.research_append_source import seal_completed_source
from tests.support.result_append import completed_source


def test_the_plain_model_refuses_a_typed_context(tmp_path) -> None:
    _wrapper, _run, _plan, context, _baseline = completed_source(tmp_path)
    payload = context.model_dump(mode="json")
    with pytest.raises(pydantic.ValidationError, match="materialized_inputs"):
        ResearchContext.model_validate({**payload, "materialized_inputs": {}})


def test_the_source_context_is_read_by_the_versioned_owner(tmp_path, monkeypatch) -> None:
    wrapper, run, _plan, context, baseline = completed_source(tmp_path)
    calls = []

    def versioned(raw):
        calls.append(raw)
        return typed.parse_research_context_json(raw)

    monkeypatch.setattr(research_append_source, "parse_research_context_json", versioned)
    source = seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
                                   baseline=baseline.model_dump(mode="json"),
                                   prepared_package_binding={"synthetic": True})

    assert calls and calls[0] == (run / "research_context.json").read_bytes()
    assert source.context.model_dump(mode="json") == context.model_dump(mode="json")
    assert research_append_source.load_append_source(wrapper).digest == source.digest


# --- S1745: a completed baseline run offers the article; a failed seal is kept and made again.

def _seeded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from easyicu.webserver import agent_review_recovery, study_contexts

    wrapper, run, _plan, _context, baseline = completed_source(tmp_path)
    seed = SimpleNamespace(study={"id": "study"}, prepared_package_binding={"synthetic": True},
                           pipeline_config={"current_case_scientific_runtime_authority": baseline.model_dump(mode="json")})
    monkeypatch.setattr(agent_review_recovery, "load_recovery_seed", lambda _wrapper: seed)
    monkeypatch.setattr(study_contexts, "get_context", lambda _id: {"id": "study"})
    service = SimpleNamespace(_research_run_row=lambda project, run_id: {"project_dir": str(wrapper), "study_id": "study"})
    return wrapper, run, service


def _options(service, run):
    from easyicu.webserver.research_append_actions import append_options

    return append_options(service, project_id="project", run_id=run.name)


def _finish(wrapper, run, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs

    removed = []
    monkeypatch.setattr(agent_pipeline_runs, "_remove_local_recovery", removed.append)
    agent_pipeline_runs._cleanup_recovery_after_projection(  # noqa: SLF001
        wrapper_dir=wrapper, projection={"run_id": run.name, "gate": {
            "reason": "research_agent_pipeline_complete_human_interpretation_required"}})
    return removed


def test_a_completed_baseline_run_offers_the_article(tmp_path, monkeypatch) -> None:
    wrapper, run, service = _seeded(tmp_path, monkeypatch)

    assert _finish(wrapper, run, monkeypatch) == [wrapper]
    options = _options(service, run)

    assert {row["kind"] for row in options["options"] if row["available"]} >= {"article"}
    assert not (wrapper / research_append_source.SEAL_FAILURE_NAME).exists()


def test_a_failed_seal_is_recorded_keeps_the_seed_and_is_made_again(tmp_path, monkeypatch) -> None:
    wrapper, run, service = _seeded(tmp_path, monkeypatch)

    def unreadable(_raw):
        raise ValueError("ResearchContext.materialized_inputs: Extra inputs are not permitted")

    with monkeypatch.context() as broken:
        broken.setattr(research_append_source, "parse_research_context_json", unreadable)
        assert _finish(wrapper, run, monkeypatch) == []  # the seed is kept
        failed = _options(service, run)
    assert failed["options"] == []
    assert failed["unavailable_reason"] == "result_append_source_seal_failed"
    assert "materialized_inputs" in failed["unavailable_detail"]
    assert not (wrapper / research_append_source.SEAL_NAME).exists()

    # The reader is fixed (an upgrade): opening the result seals it from the kept seed.
    resealed = _options(service, run)
    assert {row["kind"] for row in resealed["options"] if row["available"]} >= {"article"}
    assert (wrapper / research_append_source.SEAL_NAME).exists()
    assert not (wrapper / research_append_source.SEAL_FAILURE_NAME).exists()


def test_the_result_card_says_the_seal_failed(tmp_path) -> None:
    import json as _json
    import shutil

    from easyicu.webserver.routes import agent as agent_routes
    from tests.support.node import run_node

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    static = Path(agent_routes.__file__).resolve().parents[1] / "static" / "js"
    script = r"""
const lang = process.argv[3];
global.window = { EU_LANG: lang };
const path = require('node:path');
const dir = path.dirname(process.argv[1]);
require(path.join(dir, 'html-escape.js'));
require(path.join(dir, 'screens-guided-pi-modules.js'));
require(process.argv[1]);
const result = JSON.parse(process.argv[2]);
const owner = window.EasyICU.guidedPi.require('resultAppend').create({
  tr: (en, zh) => (lang === 'zh' ? zh : en), esc: window.EU_HTML.esc,
  projectId: () => 'project', canPreview: () => true, busy: () => false,
  api: () => ({ loadPiCopilotAppendOptions: async () => result }), rerender: () => {},
});
(async () => {
  await owner.load({ run_id: 'source' });
  process.stdout.write(owner.render({ run_id: 'source' }));
})();
"""
    payload = {"run_id": "source", "options": [], "unavailable_reason": "result_append_source_seal_failed",
               "unavailable_detail": "Extra inputs are not permitted"}
    outputs = {}
    for lang in ("zh", "en"):
        result = run_node(node, script, str(static / "screens-guided-pi-result-append.js"),
                          _json.dumps(payload), lang, check=False)
        assert result.returncode == 0, result.stderr or result.stdout
        outputs[lang] = result.stdout
    assert "未能把这次结果登记为追加分析的来源" in outputs["zh"] and "Extra inputs are not permitted" in outputs["zh"]
    assert "could not record this result as the source for additions" in outputs["en"]
