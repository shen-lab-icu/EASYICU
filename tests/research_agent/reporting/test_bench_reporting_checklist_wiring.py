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

import hashlib
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


def test_planner_only_human_review_pause_is_a_completed_planning_outcome(
    monkeypatch, tmp_path
):
    import easyicu.research_agent as rapkg
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRequest,
    )
    import tools.run_research_agent_bench as bench

    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()
    (run_dir / "analysis_plan.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "human_review_checkpoint.json").write_text("{}\n", encoding="utf-8")
    request = HumanReviewRequest.create(
        kind="protocol_claim",
        authority_sha256="a" * 64,
        summary="Review the locked plan.",
        payload={"run_id": "run_demo"},
    )

    class PausingPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return HumanReviewPending(
                run_id="run_demo",
                thread_id="thread_demo",
                run_dir=str(run_dir),
                requests=(request,),
            )

    monkeypatch.setattr(rapkg, "ResearchAgentPipeline", PausingPipeline)

    score = bench._run_one_arm(
        item=_item("descriptive_association"),
        cohort=SimpleNamespace(columns=["age", "death"]),
        workdir=tmp_path,
        disable_icu_context=False,
        label="aware",
        llm=object(),
        pipeline_options={"planner_only": True},
    )

    assert score["status"] == "human_review_pending"
    assert score["planner_only_complete"] is True
    assert score["execution_complete"] is False
    assert bench._score_execution_failures({"aware": score}) == []

    hard_stop = SimpleNamespace(finish=lambda **kwargs: pytest.fail("must stay paused"))
    bench._finish_task_on_execution_outcome(hard_stop, {"aware": score})


def test_outline_only_design_canary_is_completed_without_plan_or_execution(
    monkeypatch, tmp_path
):
    import easyicu.research_agent as rapkg
    from easyicu.research_agent.orchestration.workflow import (
        PlannerDesignCanaryComplete,
    )
    from easyicu.research_agent.planning.progressive_artifacts import (
        ProgressiveDesignCanaryReceipt,
    )
    import tools.run_research_agent_bench as bench

    run_dir = tmp_path / "run_design_canary"
    run_dir.mkdir()
    checkpoint_path = run_dir / "progressive_planner_checkpoint_000.json"
    checkpoint_path.write_text("{}\n", encoding="utf-8")
    receipt = ProgressiveDesignCanaryReceipt(
        request_authority_sha256="a" * 64,
        checkpoint_sequence=0,
        checkpoint_sha256="b" * 64,
        outline_sha256="c" * 64,
        outline_step_count=5,
        candidate_design_count=3,
        selected_design_ids=["selected_design"],
        rejected_design_ids=["alternative_one", "alternative_two"],
        selected_literature_dimension_count=7,
        selected_literature_citation_keys=["reviewed_card"],
        planner_efficiency={"calls": 1, "reported_tokens": 1200},
        cost_summary={"total_tokens": 1200, "total_cost_usd": 0.02},
    )
    receipt_path = run_dir / "progressive_design_canary_receipt.json"
    receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    class DesignCanaryPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return PlannerDesignCanaryComplete(
                run_id="run_design_canary",
                run_dir=str(run_dir),
                receipt_path=str(receipt_path),
                receipt_sha256=receipt_sha256,
                candidate_design_count=3,
                rejected_design_count=2,
                selected_literature_dimension_count=7,
                provider_calls=1,
                reported_tokens=1200,
                estimated_cost_usd=0.02,
            )

    monkeypatch.setattr(rapkg, "ResearchAgentPipeline", DesignCanaryPipeline)

    score = bench._run_one_arm(
        item=_item("descriptive_association"),
        cohort=SimpleNamespace(columns=["age", "death"]),
        workdir=tmp_path,
        disable_icu_context=False,
        label="aware",
        llm=object(),
        pipeline_options={"planner_only": True},
    )

    assert score["status"] == "design_outline_complete"
    assert score["candidate_design_count"] == 3
    assert score["selected_literature_dimension_count"] == 7
    assert score["execution_complete"] is False
    assert score["paper_authorized"] is False
    assert bench._score_execution_failures({"aware": score}) == []
    assert bench._aggregate([{"aware": score}])["aware"]["n_items"] == 1


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


def test_source_feasibility_authority_does_not_declare_a_runtime_contrast(
    monkeypatch,
    tmp_path,
):
    from benchmarks.figure2_canonical9.case_scientific_protocol import (
        build_runtime_scientific_projection,
        load_default_case_protocol,
    )

    item = _item("causal_inference")
    item.key = "h2_vasopressor_causal"
    item.research_question = "Estimate an early vasopressor effect."
    item.primary_predictor = "vasopressor"
    item.operational_exposure = "vaso_ind_max"
    projection = build_runtime_scientific_projection(
        load_default_case_protocol("h2_vasopressor_causal")
    )
    item.runtime_scientific_projection = projection.model_dump(mode="json")
    item.runtime_scientific_projection_sha256 = (
        projection.runtime_projection_sha256
    )
    item.case_scientific_protocol_sha256 = projection.protocol_content_sha256

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
        cohort=SimpleNamespace(columns=["vaso_ind_max", "death"]),
        workdir=tmp_path,
        disable_icu_context=False,
        label="aware",
        llm=object(),
        pipeline_options={"development_diagnostic": True},
    )

    assert captured["run"]["primary_exposure"] is None
    assert captured["run"]["target_outcome"] is None
    assert captured["run"]["concept_descriptions"] is None
    authority = captured["init"]["current_case_scientific_runtime_authority"]
    assert authority["binary_control_arm_authorized"] is False
    assert authority["causal_contrast_authorized"] is False


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


def test_external_scientific_authority_reaches_context_builder(monkeypatch, tmp_path):
    from easyicu.research_agent.contracts.endpoint import EndpointSpec

    item = _item("descriptive_association")
    item.exclusion_criteria = ["Exclude events before the landmark."]
    item.endpoint = EndpointSpec(
        name="death",
        kind="binary",
        absence_semantics="no_absent_rows",
        levels=[0, 1],
    )
    item.concept_descriptions = {
        "death": "Documented in-hospital mortality through discharge."
    }
    item.user_preferences = {
        "covariates": ["age"],
        "covariate_selection": "exact",
        "covariate_rationales": {
            "age": "Age is a baseline confounder of exposure and outcome."
        },
        "covariate_temporal_roles": {"age": "baseline_static"},
    }
    item.time_columns = ["followup_hour"]
    item.outcome_columns = ["death"]
    item.time_windows = [
        {
            "name": "followup",
            "anchor": "icu_admission",
            "start_hours": 24.0,
            "end_hours": 168.0,
            "rationale": "Prespecified post-landmark follow-up.",
        }
    ]

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
        cohort=SimpleNamespace(columns=["age", "death"]),
        workdir=tmp_path,
        disable_icu_context=False,
        label="aware",
        llm=object(),
    )

    assert captured["endpoint"] == item.endpoint
    assert captured["exclusion_criteria"] == item.exclusion_criteria
    assert captured["concept_descriptions"] == item.concept_descriptions
    assert captured["user_preferences"]["covariate_selection"] == "exact"
    assert "K1" in captured["user_preferences"]["data_constraints"]
    assert captured["time_columns"] == ["followup_hour"]
    assert captured["outcome_columns"] == ["death"]
    assert captured["time_windows"][0]["anchor"] == "icu_admission"
