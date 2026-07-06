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


def test_default_static_concepts_carry_survival_censoring():
    """Survival-readiness contract: the universe must carry both LOS concepts.

    Time-to-event designs (H1 ventilation survival, etc.) need an event time AND
    a survivor follow-up end. The materializer emits ``death_time`` (event time)
    from the outcome's timestamp; the survivor censoring time comes from
    ``los_hosp`` (hospital length of stay). Both LOS concepts must stay in the
    default static set, or a regenerated universe silently reverts to a timeless
    binary outcome and blocks survival analysis again.
    """
    import inspect

    default = inspect.signature(
        acquire_universe_for_question
    ).parameters["static_concepts"].default
    assert "los_icu" in default
    assert "los_hosp" in default


def test_augment_certified_followup_columns_builds_clean_survival_time(tmp_path):
    """The data-foundation layer certifies an ICU-anchored follow-up so a
    survival step can run KM/Cox instead of declining on censoring. death_time
    (hours) is the event time; survivors are censored at los_hosp*24; negative /
    post-discharge / non-positive artifacts are repaired."""
    import pandas as pd
    from easyicu.research_agent.data_foundation import _augment_certified_followup_columns

    p = tmp_path / "universe.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "death": [1, 0, 1, 1, 0],
            "death_time": [50.0, None, -23.0, 9000.0, None],  # hours; -23 & huge are artifacts
            "los_hosp": [3.0, 5.0, 2.0, 4.0, 10.0],           # DAYS
        }
    ).to_parquet(p, index=False)

    prov = _augment_certified_followup_columns(p)
    out = pd.read_parquet(p)
    ft = out["followup_time_hours"]

    assert prov["n_event_observed"] == 3
    assert out["event_observed"].tolist() == [1, 0, 1, 1, 0]
    # stay 1: valid death at 50h -> 50
    assert ft.iloc[0] == 50.0
    # stay 2: survivor -> los_hosp*24 = 120
    assert ft.iloc[1] == 120.0
    # stay 3: negative death_time artifact -> hospital-discharge proxy 2*24 = 48
    assert ft.iloc[2] == 48.0
    # stay 4: death_time 9000 > los_hosp*24 (96) -> capped at 96 (no post-discharge death)
    assert ft.iloc[3] == 96.0
    # stay 5: survivor -> 240
    assert ft.iloc[4] == 240.0
    # every follow-up strictly positive
    assert (ft > 0).all()


def test_augment_certified_followup_is_noop_without_event_time():
    """Prediction/association universes (no death_time) are untouched."""
    import pandas as pd
    import tempfile
    from pathlib import Path
    from easyicu.research_agent.data_foundation import _augment_certified_followup_columns

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "u.parquet"
        pd.DataFrame({"stay_id": [1, 2], "death": [1, 0], "los_hosp": [3.0, 5.0]}).to_parquet(p, index=False)
        prov = _augment_certified_followup_columns(p)
        out = pd.read_parquet(p)
    assert prov is None
    assert "followup_time_hours" not in out.columns
    assert "event_observed" not in out.columns
