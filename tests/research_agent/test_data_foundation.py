"""Tests for the L2 data-foundation agent (concept selection + acquisition)."""
from __future__ import annotations

import easyicu.research_agent.data_foundation as df_mod
from easyicu.research_agent.data_catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.data_foundation import (
    DataFoundationAgent,
    _extract_json,
    acquire_universe_for_question,
)


class _StubLLM:
    """Returns a fixed completion string regardless of the prompt."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, messages, *, max_tokens=2048, temperature=0.1, **kw) -> str:
        return self._response


def _catalog(*ids: str) -> AvailableCatalog:
    return AvailableCatalog(
        source="mem", concepts=[CatalogConcept(concept_id=i) for i in ids]
    )


def test_extract_json_handles_fenced_and_bare():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prose {"a": 2} trailing') == {"a": 2}
    assert _extract_json("not json at all") is None


def test_agent_selects_concepts_and_reports_coverage():
    llm = _StubLLM(
        '{"selected_concepts": ["sep3_sofa2", "sofa2", "death", "made_up"], '
        '"inclusion_exclusion": ["adults", "ICU LoS >= 1 day"], '
        '"rationale": "sepsis exposure + severity"}'
    )
    cat = _catalog("sep3_sofa2", "sofa2", "death", "age", "los_icu")
    sel = DataFoundationAgent(llm).select_concepts(
        question="sepsis-3 vs mortality", catalog=cat, target_outcome="death"
    )
    assert "sep3_sofa2" in sel.selected_concepts
    assert sel.inclusion_exclusion == ["adults", "ICU LoS >= 1 day"]
    # coverage flags the hallucinated concept, but not the real ones
    assert sel.coverage is not None
    assert sel.coverage.missing == ["made_up"]
    assert "sofa2" in sel.coverage.available


def test_agent_empty_or_garbage_response_is_safe():
    sel = DataFoundationAgent(_StubLLM("garbage")).select_concepts(
        question="q", catalog=_catalog("death")
    )
    assert sel.selected_concepts == []
    assert sel.coverage is not None and sel.coverage.sufficient  # nothing requested


def test_acquire_blocks_when_outcome_missing(monkeypatch):
    # If the outcome concept itself is not in the data, hard-block with advice.
    called = {"materialize": False}

    def _fake_materialize(**kwargs):
        called["materialize"] = True
        return {"parquet": "x.parquet", "provenance": "x.json"}

    monkeypatch.setattr(df_mod, "build_available_catalog", lambda _d: _catalog("lact", "sofa2"))
    # patch the lazily-imported materializer symbol
    import easyicu.research_agent.cohort_materializer as cm
    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_StubLLM('{"selected_concepts": ["lact"]}'),
        output_dir="/tmp/x",
        outcome_concepts=["death"],
    )
    assert res.blocked
    assert not called["materialize"]
    assert "outcome" in res.note.lower()


def test_acquire_proceeds_on_available_subset_when_outcome_present(monkeypatch):
    captured = {}

    def _fake_materialize(**kwargs):
        captured.update(kwargs)
        return {"parquet": "u.parquet", "provenance": "u.json"}

    monkeypatch.setattr(
        df_mod, "build_available_catalog",
        lambda _d: _catalog("sofa2", "lact", "death", "age", "sex", "los_icu"),
    )
    import easyicu.research_agent.cohort_materializer as cm
    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_StubLLM('{"selected_concepts": ["sofa2", "lact", "made_up", "death"]}'),
        output_dir="/tmp/x",
        outcome_concepts=["death"],
        static_concepts=["age", "sex", "los_icu"],
    )
    assert not res.blocked
    # outcome/demographics are passed via dedicated args, not feature_concepts;
    # the hallucinated concept is dropped; only available features remain.
    assert set(captured["feature_concepts"]) == {"sofa2", "lact"}
    assert res.coverage.missing == ["made_up"]
    assert "re-extract" in res.note.lower()


def test_acquire_captures_selection_token_usage_and_cost(monkeypatch):
    # A metered client exposes last_usage + model; the selection's token cost
    # is recorded on the result (it runs as a pre-sandbox stage).
    class _MeteredStub(_StubLLM):
        # OpenAIClient stores the model id as the private ``_model``.
        _model = "deepseek-chat"
        last_usage = {"prompt_tokens": 1000, "completion_tokens": 200,
                      "total_tokens": 1200}

    monkeypatch.setattr(
        df_mod, "build_available_catalog",
        lambda _d: _catalog("sofa2", "death", "age", "sex", "los_icu"),
    )
    import easyicu.research_agent.cohort_materializer as cm
    monkeypatch.setattr(
        cm, "materialize_to_parquet",
        lambda **kw: {"parquet": "u.parquet", "provenance": "u.json"},
    )

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_MeteredStub('{"selected_concepts": ["sofa2", "death"]}'),
        output_dir="/tmp/x",
        outcome_concepts=["death"],
    )
    assert res.selection_model == "deepseek-chat"
    assert res.selection_usage == {"prompt_tokens": 1000, "completion_tokens": 200,
                                   "total_tokens": 1200}
    # deepseek-chat priced at (0.27, 1.10)/1M -> 0.001*0.27 + 0.0002*1.10
    assert res.selection_cost_usd is not None and res.selection_cost_usd > 0
