"""Scoped Coder guidance for the canonical primary-cohort product schema."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.core import (
    CoderAgent,
    _cohort_predicate_partition_safety_contract,
    _primary_analysis_cohort_output_contract,
)
from easyicu.research_agent.plan_utils import _step_contract_repair_guidance
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.resources.coder import (
    bind_execution_cohort_runtime,
    bind_primary_cohort_role,
)
from easyicu.research_agent.authority.coder_authority import HostCoderAuthority


def _CaptureLLM(responses: list[str]):  # noqa: N802
    return ScriptedMockLLMClient(responses)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Apply the Planner-owned eligibility definition.",
        cohort=CohortDescriptor(
            cohort_name="planned_cohort",
            database="synthetic",
            n_stays=10,
            n_patients=9,
        ),
        variables=[],
    )


def _primary_cohort_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="01_cohort",
        intent="Materialize the planned cohort and report attrition.",
        inputs=["stay_id", "registered_eligibility_concept"],
        expected_outputs=[
            "artifact:analysis_cohort",
            "table:cohort_flow",
            "table:cohort_attrition",
        ],
        method="cohort_definition_and_attrition",
    )


def _assert_canonical_schema_guidance(text: str) -> None:
    assert "PRIMARY ANALYSIS-COHORT PRODUCT SCHEMA" in text or (
        "exact top-level integer fields `n_universe`" in text
    )
    assert "`n_final_analysis_cohort`" in text
    assert "`criterion_id`" in text
    assert "`n_at_start_rows`" in text
    assert "`n_remaining_rows`" in text
    assert "`n_excluded_rows`" in text
    assert "`{include|exclude}_{order:02d}_{normalized_concept_id}`" in text
    assert "Do not split a predicate" in text


def _assert_partition_safety_guidance(text: str) -> None:
    assert "COHORT-PREDICATE PARTITION SAFETY" in text or (
        "build a finite-value mask" in text
    )
    assert "positive or negative infinity" in text
    assert "Never allow a missing, unparseable, or non-finite value" in text
    assert "mutually exclusive and exhaustive" in text
    assert "n_at_start_rows = n_remaining_rows + n_excluded_rows" in text
    assert "fail the cohort step closed" in text


def test_initial_coder_prompt_receives_primary_cohort_canonical_schema() -> None:
    llm = _CaptureLLM(["import os\nresult = 1\n"])

    CoderAgent(llm).run(context=_context(), step=_primary_cohort_step())

    assert len(llm.calls) == 1
    _assert_canonical_schema_guidance(llm.calls[0][0][-1].content)
    _assert_partition_safety_guidance(llm.calls[0][0][-1].content)


def test_primary_cohort_role_binds_resolved_predicate_receipt() -> None:
    receipt = json.dumps(
        {
            "schema_version": "easyicu.primary_cohort_execution_prompt/1",
            "raw_universe": {"rows": 10, "sha256": "a" * 64},
            "authoritative_analysis_cohort": {
                "rows": 8,
                "sha256": "b" * 64,
            },
            "ordered_predicate_flow": [
                {
                    "predicate_kind": "inclusion",
                    "concept_id": "registered_eligibility_concept",
                    "resolved_column": "eligibility_flag",
                    "op": "not_missing",
                    "n_before": 10,
                    "n_excluded": 2,
                    "n_remaining": 8,
                }
            ],
        },
        sort_keys=True,
    )
    authority = bind_primary_cohort_role(
        authority=HostCoderAuthority(),
        locked_cohort_payload='{"name":"planned_cohort"}',
        materialized_execution_payload=receipt,
    )
    text = authority.render()

    assert "it is not already filtered" in text
    assert "HOST-VERIFIED COHORT EXECUTION RECEIPT" in text
    assert '"resolved_column": "eligibility_flag"' in text
    assert "every non-missing value is exactly in {0, 1}" in text
    assert "A threshold check alone is not a domain check" in text
    assert "not permission to select rows by position" in text
    assert "`resolved_column` entries are the only raw predicate coordinates" in text
    assert "Do not read or audit related measured/count/status/timing siblings" in text
    assert "`manifest['host_verified_cohort_execution_receipt']`" in text
    assert "do not expect an alias or reconstruct the receipt" in text


def test_execution_cohort_runtime_uses_only_step_raw_domain_contract() -> None:
    text = bind_execution_cohort_runtime(authority=HostCoderAuthority()).render()

    assert "raw_input_contracts" in text
    assert "sole executable domain authority" in text
    assert "Use its exact allowed_values when present" in text
    assert "do not recover one from prompt prose, the broader ResearchContext" in text
    assert "ResearchContext JSON uses observed_domain.levels" not in text


def test_repair_prompt_and_contract_guidance_share_primary_cohort_schema() -> None:
    patch = json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [
                {
                    "old": "result = 1",
                    "new": "result = 2",
                    "expected_count": 1,
                }
            ],
        }
    )
    llm = _CaptureLLM([patch])
    step = _primary_cohort_step()
    repair_guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "contract_failed"},
        code="import os\nresult = 1\n",
    )

    repaired = CoderAgent(llm).repair(
        context=_context(),
        step=step,
        code="import os\nresult = 1\n",
        run_log=repair_guidance,
    )

    assert repaired.strip().endswith("result = 2")
    _assert_canonical_schema_guidance(repair_guidance)
    _assert_canonical_schema_guidance(llm.calls[0][0][-1].content)
    _assert_partition_safety_guidance(repair_guidance)
    _assert_partition_safety_guidance(llm.calls[0][0][-1].content)


def test_primary_cohort_schema_guidance_tracks_host_product_aliases() -> None:
    step = _primary_cohort_step().model_copy(
        update={
            "expected_outputs": [
                "dataset:analysis_cohort",
                "table:attrition",
            ]
        }
    )

    _assert_canonical_schema_guidance(_primary_analysis_cohort_output_contract(step))


def test_primary_cohort_schema_guidance_accepts_analysis_set_alias() -> None:
    step = _primary_cohort_step().model_copy(
        update={
            "method": "cohort_definition_with_attrition",
            "expected_outputs": [
                "cohort:analysis_set",
                "table:cohort_flow",
                "table:attrition",
            ],
        }
    )

    guidance = _primary_analysis_cohort_output_contract(step)

    _assert_canonical_schema_guidance(guidance)
    assert "every physical column" in guidance
    assert "ordered row identity" in guidance


def test_primary_cohort_guidance_accepts_agent_semantic_method_label() -> None:
    step = _primary_cohort_step().model_copy(
        update={
            "method": "explicit_eligibility_filter_with_attrition",
            "expected_outputs": [
                "cohort:analysis_set",
                "table:cohort_flow",
                "table:cohort_attrition",
            ],
        }
    )

    _assert_canonical_schema_guidance(_primary_analysis_cohort_output_contract(step))
    _assert_partition_safety_guidance(_cohort_predicate_partition_safety_contract(step))


@pytest.mark.parametrize(
    "output",
    ["artifact:analysis_set", "table:analysis_set", "dataset:analysis_set"],
)
def test_primary_cohort_schema_guidance_rejects_foreign_analysis_set_namespaces(
    output: str,
) -> None:
    step = _primary_cohort_step().model_copy(
        update={
            "expected_outputs": [output, "table:cohort_flow"],
        }
    )

    assert _primary_analysis_cohort_output_contract(step) == ""


def test_primary_cohort_schema_guidance_requires_host_method_family() -> None:
    step = _primary_cohort_step().model_copy(
        update={"method": "mixed_effects_regression"}
    )

    assert _primary_analysis_cohort_output_contract(step) == ""


def test_generic_cohort_flow_contract_prevents_nonfinite_threshold_admission() -> None:
    step = AnalysisStep(
        step_id="01_eligibility",
        intent="Apply the declared eligibility rule and account for every row.",
        inputs=["eligibility_value"],
        expected_outputs=["cohort:eligible_records", "table:cohort_flow"],
        method="cohort_definition_and_attrition",
    )
    llm = _CaptureLLM(["import os\nresult = 1\n"])

    direct_contract = _cohort_predicate_partition_safety_contract(step)
    repair_guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "contract_failed"},
        code="import os\nresult = 1\n",
    )
    CoderAgent(llm).run(context=_context(), step=step)

    _assert_partition_safety_guidance(direct_contract)
    _assert_partition_safety_guidance(repair_guidance)
    _assert_partition_safety_guidance(llm.calls[0][0][-1].content)
    combined = "\n".join(
        (direct_contract, repair_guidance, llm.calls[0][0][-1].content)
    )
    for case_term in ("lactate", "kdigo", "mimic", "e2_lactate"):
        assert case_term not in combined.lower()


@pytest.mark.parametrize(
    ("method", "outputs"),
    [
        ("mixed_effects_regression", ["cohort:eligible_records", "table:cohort_flow"]),
        ("cohort_definition_and_attrition", ["cohort:eligible_records"]),
        ("cohort_definition_and_attrition", ["table:ordinary_summary"]),
    ],
)
def test_cohort_partition_safety_requires_method_and_structured_flow_product(
    method: str,
    outputs: list[str],
) -> None:
    step = AnalysisStep(
        step_id="cohort_words_are_only_prose",
        intent="Mention cohort flow and attrition in prose.",
        expected_outputs=outputs,
        method=method,
    )

    assert _cohort_predicate_partition_safety_contract(step) == ""
    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "contract_failed"},
        code="",
    )
    assert "finite-value mask" not in guidance


@pytest.mark.parametrize(
    "outputs",
    [
        ["artifact:analysis_cohort"],
        ["table:cohort_flow", "table:cohort_attrition"],
        ["artifact:adult_analysis_cohort", "table:cohort_flow"],
        ["artifact:analysis_cohort", "table:table_one"],
        ["table:ordinary_summary"],
    ],
)
def test_primary_cohort_schema_guidance_does_not_leak_to_other_products(
    outputs: list[str],
) -> None:
    step = AnalysisStep(
        step_id="cohort_words_are_not_authority",
        intent=(
            "Mention analysis_cohort, cohort_flow, canonical attrition, and "
            "missingness in prose only."
        ),
        expected_outputs=outputs,
        method="cohort_definition_and_attrition",
    )

    assert _primary_analysis_cohort_output_contract(step) == ""
    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "contract_failed"},
        code="",
    )
    assert "n_final_analysis_cohort" not in guidance
    assert "normalized_concept_id" not in guidance


def test_contract_repair_guidance_forbids_raw_receipts_without_typed_inputs() -> None:
    step = AnalysisStep(
        step_id="define_analysis_cohort",
        intent="Materialize the analysis cohort.",
        inputs=["age"],
        expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
        method="cohort_definition_and_attrition",
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"input_bindings": [{"input_key": "raw:age"}]},
        code="",
        input_bindings={},
    )

    assert "no host-resolved typed inputs" in guidance
    assert "`raw:<column>` receipts" in guidance
