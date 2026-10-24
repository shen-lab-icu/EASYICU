"""Complete study populations survive the bounded constraints transport."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.agents.plan_payload import planner_structured_output_request
from easyicu.research_agent.agents.planner import PlannerAgent
from easyicu.research_agent.agents.progressive_payload import (
    progressive_outline_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.run_input import (
    build_scientific_identity,
    load_verified_run_input_capsule,
    seal_run_input_capsule,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.prompt_budget import budgeted_client
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.concept_population import context_data_constraints
from easyicu.research_agent.research_context.outbound import outbound_safe_context_payload
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.webserver import agent_pipeline_runs, primary_cohort
from easyicu.webserver.research_launch_scientific import (
    _require_export_holds_study_cohort,
)


def _ordinary_constraints() -> dict:
    """Synthetic native-export receipt with the full ordinary population."""

    return {
        "cohort": {
            "preset": "all_icu",
            "age_min": 18,
            "age_max": 100,
            "min_icu_los_hours": 24,
            "exclude_readmissions": False,
            "review": (
                "Adults admitted to an ICU, retaining the first ICU stay in each "
                "hospitalization. Exclude step-down unit stays and stays whose "
                "prior-unit history is incomplete. Read age from the demographic "
                "record at hospital admission. Apply the same rules to every "
                "selected module before planning; the outcome horizon remains "
                "separate from the feature materialization window."
            ),
            "exclusion_statement": (
                "Later ICU stays in the same hospitalization, step-down units, "
                "incomplete prior-unit records, age below 18 years, and ICU stays "
                "shorter than 24 hours are outside the declared population."
            ),
        },
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "confirmations": {
            "feature_time_window": True,
            "export_format": True,
            "extraction_completed": True,
            "repeated_icu_stays_retained": True,
        },
        "materialization_window": {
            "role": "outer_observation_window",
            "hours": 24,
            "anchor": "ICU admission",
        },
        "event_time_semantics": {"death_time": "recorded_time"},
        "source_selection": {
            "basis": "export_contract",
            "host_applied": [
                "Keep only the first ICU stay per hospitalization, using the "
                "verified hospital and ICU admission coordinates before planning.",
                "Exclude step-down unit stays using the bound source's unit "
                "classification; unknown classifications cannot prove eligibility.",
                "Require a complete prior-unit history for the hospitalization "
                "before determining which ICU admission is the first eligible stay.",
                "Use age at hospital admission from the source demographic "
                "record; missing age does not establish adult eligibility.",
            ],
            "export_report": {
                "mode": "all_icu",
                "count_unit": "icu_stay",
                "source_total": 12,
                "selected": 4,
                "defaults_locked": True,
                "demographic_steps": [
                    {
                        "criterion": "age",
                        "parameters": {"age_min": 18, "age_max": 100},
                        "n_before": 12,
                        "n_excluded": 2,
                        "n_remaining": 10,
                        "n_excluded_missing": 1,
                    },
                    {
                        "criterion": "minimum_icu_duration",
                        "parameters": {"min_icu_los_hours": 24},
                        "n_before": 10,
                        "n_excluded": 2,
                        "n_remaining": 8,
                        "n_excluded_missing": 0,
                    },
                ],
                "host_steps": [
                    {
                        "criterion": "first_icu_stay_per_hospitalization",
                        "n_before": 8,
                        "n_excluded": 1,
                        "n_remaining": 7,
                    },
                    {
                        "criterion": "exclude_step_down_units",
                        "n_before": 7,
                        "n_excluded": 1,
                        "n_remaining": 6,
                    },
                    {
                        "criterion": "complete_prior_unit_history",
                        "n_before": 6,
                        "n_excluded": 2,
                        "n_remaining": 4,
                    },
                ],
                "module_defaults": {
                    "demographics": {"age": "hospital_admission", "sex": "static"},
                    "vitals": {
                        "window_hours": 24,
                        "aggregation": "mean",
                        "missingness_policy": "retain stays and report missing measurements",
                    },
                    "outcome": {"death": "hospital_discharge", "los_icu": "days"},
                },
                "selection_receipt": (
                    "The native export locks the selected module defaults and "
                    "records the population selection before feature extraction. "
                    "Counts refer to ICU stays in this synthetic source and are "
                    "reported in their execution order. Hospitalization selection, "
                    "unit eligibility, prior-unit completeness and demographic age "
                    "use source coordinates verified by the host. Repeated stays "
                    "across hospitalizations remain grouped by patient for inference."
                ),
            },
        },
    }


def test_an_ordinary_population_is_transported_without_elision() -> None:
    constraints = _ordinary_constraints()
    expected = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    assert 3_100 <= len(expected) <= 3_300

    assert agent_pipeline_runs._compile_data_constraints(constraints) == expected


@pytest.mark.parametrize("size", [4_000, 4_096])
@pytest.mark.parametrize("language", ["english", "chinese"])
def test_complete_web_constraints_reach_planner_context_and_sealed_run_inputs(
    tmp_path: Path, size: int, language: str,
) -> None:
    """Exercise real transport owners without ASGI, providers or a research run."""

    constraints = _ordinary_constraints()
    constraints["source_selection"]["host_applied"] = []
    constraints["cohort"]["review"] += (
        " Verify the selected population against the native export receipt before "
        "constructing an analysis cohort. Preserve the distinction between a stay "
        "excluded by the source and a row with a missing baseline measurement. "
        "Do not exclude a stay solely because one requested measurement is absent; "
        "report measurement availability and the population used by each analysis. "
        "Keep the feature window anchored to ICU admission and describe outcome "
        "follow-up independently. Read recorded death times using their declared "
        "event-time semantics; a discharge label cannot establish the hour of death. "
        "Check that exclusions are applied in their recorded order and reconcile "
        "the remaining stays with the export report. Retain the stated patient "
        "grouping across separate hospitalizations for uncertainty estimates. "
        "Do not reinterpret the exported module defaults as user approval for "
        "additional endpoints, exposure groups or subgroup comparisons. Keep the "
        "original eligibility wording alongside the executable population so "
        "reviewers can identify any restriction that the available source cannot "
        "prove. State unresolved source limitations before interpreting results."
    )
    if language == "chinese":
        constraints["cohort"]["review"] += " 保留全部入组条件、缺失值规则及重复住院的患者聚类要求。"
    # Only whitespace pads the exact boundary; the substantive receipt above
    # remains realistic and every section must survive byte for byte.
    serialized = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    assert 3_800 <= len(serialized) <= size
    constraints["cohort"]["review"] += " " * (size - len(serialized))
    expected = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    assert len(expected) == size
    if language == "chinese":
        assert len(expected.encode("utf-8")) > size

    preferences = agent_pipeline_runs._research_user_preferences(
        {
            "cohort": constraints["cohort"],
            "analysis_design": constraints["analysis_design"],
            "confirmations": constraints["confirmations"],
            "time_window": {"hours": 24, "anchor": "ICU admission"},
        },
        source_selection_basis="export_contract",
        source_selection_report=constraints["source_selection"]["export_report"],
        event_time_semantics=constraints["event_time_semantics"],
    )
    assert preferences["data_constraints"] == expected

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3, 4], "age": [24, 37, 58, 76]}).to_parquet(
        cohort_path, index=False,
    )
    request = {
        "cohort": cohort_path,
        "cohort_name": "synthetic_adults",
        "database": "synthetic",
        "target_outcome": None,
        "primary_exposure": None,
        "cross_database_validation": None,
        "inclusion_criteria": ["age range: 18 to 100"],
        "exclusion_criteria": [],
        "id_columns": ["stay_id"],
        "time_columns": None,
        "outcome_columns": None,
        "time_windows": None,
        "concept_descriptions": None,
        "user_preferences": preferences,
        "notes": None,
    }
    question = "Describe the age distribution of eligible adult ICU stays."
    identity = build_scientific_identity(
        question=question, skill_key=None, experiment_spec=None,
        source_files=None, disable_icu_context=False, **request,
    )
    assert identity["user_preferences"]["data_constraints"] == expected
    context = build_research_context(research_question=question, **request)
    context_path = tmp_path / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    restored = parse_research_context_json(context_path.read_text(encoding="utf-8"))
    assert restored.user_preferences.data_constraints == expected
    assert context_data_constraints(restored) == constraints
    assert outbound_safe_context_payload(restored)["study_preferences"]["data_constraints"] == expected

    evidence = EvidenceStore(tmp_path)
    evidence.register_file(
        kind="log", description="Synthetic research context.",
        source_path=context_path, evidence_id="research_context",
        producer="pipeline", generation_mode="system",
    )
    seal_run_input_capsule(
        run_dir=tmp_path, evidence=evidence, scientific_identity=identity,
        initial_environment={}, context_path=context_path,
        cohort_path=cohort_path, experiment_spec_path=None,
    )
    authority = load_verified_run_input_capsule(run_dir=tmp_path, scientific_identity=identity)
    assert authority.capsule.scientific_identity["user_preferences"]["data_constraints"] == expected
    sealed_context = parse_research_context_json(
        authority.context_evidence_path.read_text(encoding="utf-8")
    )
    assert sealed_context.user_preferences.data_constraints == expected

    class RecordingTransport:
        def complete(self, messages, **kwargs):
            assert kwargs["structured_output"].payload_bytes > 0
            self.messages = messages
            return "captured"

    for planner in (PlannerAgent, ProgressivePlannerAgent):
        if planner is PlannerAgent:
            schema = planner_structured_output_request()
        else:
            families, variables, actions, _ = planner._request_authorities(sealed_context)
            schema = progressive_outline_structured_output_request(
                analysis_types=families, variable_names=variables,
                scientific_action_ids=actions,
            )
        messages = planner.request_messages(
            sealed_context, strict_transport_schema=True, structured_output=schema,
        )
        transport = RecordingTransport()
        client = budgeted_client(transport, "planner", "planner_plan_generation")
        assert client.complete(messages, structured_output=schema) == "captured"
        content = transport.messages[-1].content
        assert '"data_constraints":' in content, planner.__name__
        received, _ = json.JSONDecoder().raw_decode(
            content.split('"data_constraints":', 1)[1].lstrip()
        )
        assert received == expected, planner.__name__
        assert json.loads(received) == constraints

    refused_transport = RecordingTransport()
    bounded = budgeted_client(
        refused_transport, "planner", "planner_plan_generation", limit_tokens=1_024,
    )
    with pytest.raises(ProgressivePlanCompileError) as raised:
        ProgressivePlannerAgent(bounded).run_attempt(sealed_context, stop_after_outline=True)
    assert raised.value.code == "progressive_prompt_budget_exceeded"
    assert raised.value.path == "planner_request"
    assert not hasattr(refused_transport, "messages")


@pytest.mark.parametrize("size", [4_096, 4_097])
def test_the_transport_boundary_preserves_or_refuses_the_complete_payload(size: int) -> None:
    constraints = _ordinary_constraints()
    empty_size = len(json.dumps(constraints, ensure_ascii=False, sort_keys=True))
    constraints["cohort"]["review"] += "x" * (size - empty_size)
    payload = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    assert len(payload) == size

    if size == 4_096:
        assert agent_pipeline_runs._compile_data_constraints(constraints) == payload
    else:
        with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
            agent_pipeline_runs._compile_data_constraints(constraints)
        assert raised.value.code == "research_pipeline_data_constraints_too_large"
        assert raised.value.details["limit_chars"] == 4_096
        assert raised.value.details["serialized_chars"] == size
        assert raised.value.details["field"] == "data_constraints"
        assert raised.value.details["section_chars"]["cohort"] == len(
            json.dumps(constraints["cohort"], ensure_ascii=False, sort_keys=True)
        )


def test_a_list_cannot_be_elided_to_fit_the_transport() -> None:
    constraints = _ordinary_constraints()
    constraints["source_selection"]["host_applied"] += [
        f"Synthetic eligibility criterion {index}: require complete source coordinates."
        for index in range(24)
    ]
    assert len(json.dumps(constraints, ensure_ascii=False, sort_keys=True)) > 4_096

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._compile_data_constraints(constraints)

    assert raised.value.code == "research_pipeline_data_constraints_too_large"


@pytest.mark.parametrize(
    "cohort",
    [
        {"preset": "unregistered", "exclude_readmissions": True},
        {"icd_enabled": "invalid", "exclude_readmissions": True},
        {"exclude_readmissions": "invalid"},
    ],
)
def test_an_invalid_cohort_declares_no_applied_restriction_and_still_refuses_execution(
    tmp_path: Path, cohort: dict,
) -> None:
    study = {"cohort": cohort}
    assert agent_pipeline_runs._host_applied_criteria(study) == []
    assert agent_pipeline_runs._exclusion_criteria(study) == []

    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"files": [], "cohort_contract": {"preset": "all_icu"}}),
        encoding="utf-8",
    )
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        _require_export_holds_study_cohort(study, str(export))

    assert raised.value.code == "research_pipeline_export_cohort_invalid"
    expected = "unsupported_cohort_preset" if "preset" in cohort else "invalid_cohort_boolean"
    assert raised.value.details == {"reason_code": expected}


def test_a_cohort_owner_error_does_not_crash_the_diagnostic_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid(_cohort: object) -> bool:
        raise primary_cohort.PrimaryCohortContractError("invalid_cohort_boolean")

    monkeypatch.setattr(primary_cohort, "first_icu_stay_only", invalid)

    assert agent_pipeline_runs._host_applied_criteria({"cohort": {}}) == []
