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


def _run_and_capture(
    monkeypatch,
    tmp_path,
    kind: str,
    *,
    scientific_contract: dict[str, object] | None = None,
):
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

    item = _item(kind)
    item.scientific_acceptance_contract = scientific_contract
    bench._run_one_arm(
        item=item,
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


def test_scientific_contract_binds_primary_cohort_mode(monkeypatch, tmp_path):
    captured = _run_and_capture(
        monkeypatch,
        tmp_path,
        "descriptive_association",
        scientific_contract={"primary_cohort_selection_mode": "all_input_rows"},
    )

    assert captured["required_primary_cohort_selection_mode"] == "all_input_rows"


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
    item.notes = "Use the audited first-24-hour materialization."
    item.expected_outputs = ["table:adjusted_association"]
    item.semantic_guardrails = ["Preserve the declared temporal window."]

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
    assert captured["run"]["concept_descriptions"] is None
    assert item.primary_predictor == "concept_level_scoring_key"
    assert "gold_answer" not in captured["run"]
    assert "semantic_guardrails" not in captured["run"]
    preferences = captured["run"]["user_preferences"]
    assert "Use the audited first-24-hour materialization." in preferences[
        "data_constraints"
    ]
    assert "table:adjusted_association" in preferences["must_have_outputs"]
    assert "Preserve the declared temporal window." in preferences[
        "evaluation_focus"
    ]
    assert captured["run"]["question"] == "Build a model."


def test_question_exposed_exposure_label_is_typed_context_metadata(
    monkeypatch, tmp_path
):
    item = _item("descriptive_association")
    item.research_question = "Is admission lactate associated with mortality?"
    item.primary_predictor = "lactate"
    item.operational_exposure = "lact_max"

    import easyicu.research_agent as rapkg
    import tools.run_research_agent_bench as bench

    captured: dict = {}

    class CapturePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(workdir=str(tmp_path))

    monkeypatch.setattr(rapkg, "ResearchAgentPipeline", CapturePipeline)
    monkeypatch.setattr(bench, "_score_arm", lambda **kwargs: {})

    bench._run_one_arm(
        item=item,
        cohort=SimpleNamespace(columns=["lact_max", "death"]),
        workdir=tmp_path,
        disable_icu_context=False,
        label="aware",
        llm=object(),
    )

    assert captured["primary_exposure"] == "lact_max"
    assert captured["concept_descriptions"] == {"lact_max": "lactate"}
