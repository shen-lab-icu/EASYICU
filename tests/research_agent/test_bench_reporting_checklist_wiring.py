"""The bench runner must force the KIND-matched reporting checklist (G-2).

Detector/emitter contract mismatch found by the G-2 sweep: the scorecard READS
the reporting checklist by ``task.kind`` (``mortality_prediction`` -> TRIPOD+AI),
but the bench runner never told the pipeline which checklist to EMIT, so it fell
back to free-text analysis-family inference and emitted only STROBE for the
prediction task. The clean M2 run reached the write phase yet
``reporting_completeness`` was silently NA. The fix wires the single source of
truth (``checklist_names_for_kind``) into the pipeline construction; these tests
pin that wiring without a full (~40 min) run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _item(kind: str):
    return SimpleNamespace(
        key="K1",
        name="demo",
        research_question="Build a model.",
        target_outcome="death",
        primary_predictor="",
        inclusion_criteria=[],
        kind=kind,
    )


def _run_and_capture(monkeypatch, tmp_path, kind: str):
    import easyicu.research_agent as rapkg
    import tools.run_research_agent_bench as bench

    captured: dict = {}

    class CapturePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            captured["pipeline_run_kwargs"] = kwargs
            return SimpleNamespace(workdir=str(tmp_path))

    monkeypatch.setattr(rapkg, "ResearchAgentPipeline", CapturePipeline)
    # Avoid the heavy scorecard read; we only care about construction kwargs.
    monkeypatch.setattr(bench, "_score_arm", lambda **k: {})

    bench._run_one_arm(
        item=_item(kind),
        cohort=SimpleNamespace(columns=["age", "death", "hr_first"]),
        workdir=tmp_path,
        disable_icu_context=True,
        label="aware",
        llm=object(),
    )
    return captured


def test_prediction_kind_forces_tripod_ai_checklist(monkeypatch, tmp_path):
    captured = _run_and_capture(monkeypatch, tmp_path, "mortality_prediction")
    assert captured.get("reporting_checklist_names") == ["strobe", "tripod_ai"]


def test_clustering_kind_forces_internal_phenotype_checklist(monkeypatch, tmp_path):
    captured = _run_and_capture(monkeypatch, tmp_path, "subphenotype_clustering")
    assert captured.get("reporting_checklist_names") == ["internal_phenotype"]


def test_association_kind_forces_strobe(monkeypatch, tmp_path):
    captured = _run_and_capture(monkeypatch, tmp_path, "descriptive_association")
    assert captured.get("reporting_checklist_names") == ["strobe"]


def test_explicit_pipeline_option_overrides_kind_default(monkeypatch, tmp_path):
    # An explicit pipeline_options value must win over the kind default.
    import easyicu.research_agent as rapkg
    import tools.run_research_agent_bench as bench

    captured: dict = {}

    class CapturePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return SimpleNamespace(workdir=str(tmp_path))

    monkeypatch.setattr(rapkg, "ResearchAgentPipeline", CapturePipeline)
    monkeypatch.setattr(bench, "_score_arm", lambda **k: {})

    bench._run_one_arm(
        item=_item("mortality_prediction"),
        cohort=SimpleNamespace(columns=["age", "death"]),
        workdir=tmp_path,
        disable_icu_context=True,
        label="aware",
        llm=object(),
        pipeline_options={"reporting_checklist_names": ["strobe"]},
    )
    assert captured.get("reporting_checklist_names") == ["strobe"]


def test_external_execution_uses_database_and_operational_exposure_not_scoring_key(
    monkeypatch, tmp_path
):
    item = _item("descriptive_association")
    item.database = "eicu"
    item.primary_predictor = "concept_level_scoring_key"
    item.operational_exposure = "materialized_exposure_column"
    item.gold_answer = {"numeric_targets": {"hidden_metric": {"lower": 0.0}}}
    item.semantic_guardrails = ["Evaluator-only audit key."]

    import easyicu.research_agent as rapkg
    import tools.run_research_agent_bench as bench

    captured: dict = {}

    class CapturePipeline:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run(self, **kwargs):
            captured["run"] = kwargs
            return SimpleNamespace(workdir=str(tmp_path))

    monkeypatch.setattr(rapkg, "ResearchAgentPipeline", CapturePipeline)
    monkeypatch.setattr(bench, "_score_arm", lambda **kwargs: {})

    bench._run_one_arm(
        item=item,
        cohort=SimpleNamespace(
            columns=["materialized_exposure_column", "death"]
        ),
        workdir=tmp_path,
        disable_icu_context=False,
        label="aware",
        llm=object(),
    )

    assert captured["run"]["database"] == "eicu"
    assert captured["run"]["primary_exposure"] == "materialized_exposure_column"
    assert item.primary_predictor == "concept_level_scoring_key"
    assert "gold_answer" not in captured["run"]
    assert "semantic_guardrails" not in captured["run"]
    assert captured["run"]["question"] == "Build a model."
