"""ResearchContext v2 prompt facts stay bounded and role-scoped.

The materialized-input projection is host-owned physical/lineage authority.  It
may help Planner/Replanner understand what is actually available, but it must
not become a second source of scientific choices or leak into downstream prose
prompts by default.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from easyicu.concept.metadata_sidecar import (
    ColumnMetadataBinding,
    binding_payload_sha256,
)
from easyicu.research_agent.agents.core import (
    AnalyzerAgent,
    PlannerAgent,
    ReplannerAgent,
    WriterAgent,
)
from easyicu.research_agent.research_context.typed import (
    ResearchContextV2,
    materialized_input_prompt_attachment,
)
from easyicu.research_agent.authority.coder_authority import HostCoderAuthority
from easyicu.research_agent.resources.coder import bind_materialized_coder_authority
from tests.research_agent.test_research_context_v2_authority_join import (
    _prepare_typed_run,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.research_agent.test_materialized_column_metadata import (
    _build_v2_context,
)

_MATERIALIZED_HEADING = "MATERIALIZED INPUT FACTS (host-verified;"
_FORBIDDEN_SCIENTIFIC_SELECTION_KEYS = {
    "analysis_method",
    "analysis_set",
    "covariates",
    "estimand",
    "exposure",
    "inclusion_criteria",
    "method",
    "outcome",
    "primary_exposure",
    "study_cohort",
    "target_outcome",
}


def _CapturingLLM(response: str):
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    return ScriptedMockLLMClient([response], repeat_last=True)


def _last_user_prompt(client) -> str:
    assert client.calls
    messages, _kwargs = client.calls[-1]
    return next(
        message.content for message in reversed(messages) if message.role == "user"
    )


@pytest.fixture
def v2_context(tmp_path: Path) -> ResearchContextV2:
    return _build_v2_context(tmp_path)


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe age while retaining the declared effect model.",
        analysis_type="descriptive_epidemiology",
        steps=[
            AnalysisStep(
                step_id="01_summary",
                intent="Summarize the declared variables.",
                inputs=["age", "lact_max", "death"],
                expected_outputs=["table:summary"],
                method="descriptive",
            )
        ],
        rationale="Use the declared descriptive step.",
    )


def test_step_scoped_v2_facts_bind_the_shared_coder_authority(
    v2_context: ResearchContextV2,
) -> None:
    step = _plan().steps[0]
    base_authority = HostCoderAuthority().append("existing host receipt")

    scoped_context, bound_authority = bind_materialized_coder_authority(
        context=v2_context,
        step=step,
        authority=base_authority,
    )

    assert isinstance(scoped_context, ResearchContextV2)
    attachment = materialized_input_prompt_attachment(scoped_context)
    assert attachment
    assert bound_authority.attachments == (*base_authority.attachments, attachment)
    assert set(scoped_context.materialized_inputs.cohort.column_bindings).issubset(
        {variable.name for variable in scoped_context.variables}
    )


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(
            *(_recursive_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(item) for item in value), set())
    return set()


def _wide_context_with_late_primary(context: ResearchContextV2) -> ResearchContextV2:
    """Add enough internally valid bindings to force prompt projection trimming."""

    payload = context.model_dump(mode="python")
    cohort = payload["materialized_inputs"]["cohort"]
    cohort["projection_scope"] = "scoped"
    base = ColumnMetadataBinding.from_dict(cohort["column_bindings"]["age"]["binding"])
    canonical_bindings = {
        column: ColumnMetadataBinding.from_dict(binding["binding"])
        for column, binding in cohort["column_bindings"].items()
    }

    for index in range(55):
        column = f"dummy_{index:02d}"
        binding = replace(
            base,
            metadata=replace(base.metadata, column_name=column),
        )
        canonical_bindings[column] = binding
        cohort["column_bindings"][column] = {
            "binding": binding.to_dict(),
            "binding_sha256": binding_payload_sha256({column: binding}),
            "analysis_plausibility_range": cohort["column_bindings"]["age"][
                "analysis_plausibility_range"
            ],
        }
        cohort["cohort_columns"].append(column)

    primary = "zz_primary"
    primary_binding = replace(
        base,
        metadata=replace(base.metadata, column_name=primary),
    )
    canonical_bindings[primary] = primary_binding
    cohort["column_bindings"][primary] = {
        "binding": primary_binding.to_dict(),
        "binding_sha256": binding_payload_sha256({primary: primary_binding}),
        "analysis_plausibility_range": cohort["column_bindings"]["age"][
            "analysis_plausibility_range"
        ],
    }
    cohort["cohort_columns"].append(primary)
    cohort["column_binding_payload_sha256"] = binding_payload_sha256(canonical_bindings)

    age = next(variable for variable in context.variables if variable.name == "age")
    payload["variables"].append(
        age.model_copy(update={"name": primary}).model_dump(mode="python")
    )
    payload["primary_exposure"] = primary
    return type(context).model_validate(payload)


def test_planner_and_replanner_use_unified_outbound_safe_projection(
    v2_context: ResearchContextV2,
) -> None:
    plan = _plan()

    planner_llm = _CapturingLLM(plan.model_dump_json())
    PlannerAgent(planner_llm).run(v2_context)
    planner_prompt = _last_user_prompt(planner_llm)
    assert _MATERIALIZED_HEADING not in planner_prompt
    assert "age" in planner_prompt
    assert "easyicu.outbound_safe_context/1" in planner_prompt

    replanner_llm = _CapturingLLM(plan.model_dump_json())
    ReplannerAgent(replanner_llm).run(
        context=v2_context,
        current_plan=plan,
    )
    replanner_prompt = _last_user_prompt(replanner_llm)
    assert _MATERIALIZED_HEADING not in replanner_prompt
    assert "age" in replanner_prompt
    assert "easyicu.outbound_safe_context/1" in replanner_prompt


def test_analyzer_and_writer_do_not_receive_materialized_block_by_default(
    v2_context: ResearchContextV2,
) -> None:
    step = _plan().steps[0]

    analyzer_llm = _CapturingLLM("No numeric interpretation is available.")
    AnalyzerAgent(analyzer_llm).run(
        context=v2_context,
        step=step,
        step_summary={},
        evidence_ids=[],
    )
    assert _MATERIALIZED_HEADING not in _last_user_prompt(analyzer_llm)

    writer_llm = _CapturingLLM("No supported prose.")
    WriterAgent(writer_llm)._call_section(
        section_name="Results",
        instruction="Omit unsupported results.",
        context=v2_context,
        evidence_ids=[],
        evidence_digest=None,
    )
    assert _MATERIALIZED_HEADING not in _last_user_prompt(writer_llm)


def test_writer_prompt_forbids_unregistered_numeric_derivation(
    v2_context: ResearchContextV2,
) -> None:
    writer_llm = _CapturingLLM("No supported prose.")
    WriterAgent(writer_llm)._call_section(
        section_name="Results",
        instruction="Report only registered results.",
        context=v2_context,
        evidence_ids=["registered_result"],
        evidence_digest=(
            "Registered value: 0.25 {evidence:registered_result}"
        ),
    )

    prompt = _last_user_prompt(writer_llm)
    assert "Copy a current-study number only when that exact literal value" in prompt
    assert "Do not calculate, infer, transform, round, or reconstruct" in prompt
    assert "RESEARCH CONTEXT supplies study semantics only" in prompt
    assert "host has registered it explicitly in the evidence digest" in prompt


def test_wide_materialized_projection_keeps_primary_and_no_science_choices(
    v2_context: ResearchContextV2,
) -> None:
    context = _wide_context_with_late_primary(v2_context)
    attachment = materialized_input_prompt_attachment(context)

    assert len(attachment.encode("utf-8")) <= 4 * 1024
    payload = json.loads(attachment.split("\n", 1)[1])
    columns = payload["cohort"]["column_bindings"]
    assert any(item["column"] == "zz_primary" for item in columns)
    assert payload["cohort"]["column_binding_total_count"] > 49
    assert payload["cohort"]["column_binding_omitted_count"] > 0

    assert payload["authority_scope"] == (
        "physical_representation_availability_and_lineage_only"
    )
    materialized_cohort = context.materialized_inputs.cohort
    assert payload["schema_version"] == "easyicu.materialized_input_prompt_facts/2"
    assert payload["metadata_implementation"] == {
        "metadata_implementation_bundle_sha256": (
            materialized_cohort.metadata_implementation_bundle_sha256
        )
    }
    assert "retain cohort, exposure, outcome, method" in payload["scientific_ownership"]
    assert not (_recursive_keys(payload) & _FORBIDDEN_SCIENTIFIC_SELECTION_KEYS)


def test_typed_trajectory_prompt_is_bounded_and_preserves_primary_source_facts(
    tmp_path: Path,
) -> None:
    _run_dir, _cohort_path, context, *_rest = _prepare_typed_run(tmp_path)

    attachment = materialized_input_prompt_attachment(context)
    assert len(attachment.encode("utf-8")) <= 4 * 1024
    payload = json.loads(attachment.split("\n", 1)[1])
    cohort_columns = {item["column"] for item in payload["cohort"]["column_bindings"]}
    assert {"lact_max", "death"}.issubset(cohort_columns)
    trajectory = payload["trajectory"]
    assert trajectory["concept_total_count"] == (
        len(trajectory["concepts"]) + trajectory["concept_omitted_count"]
    )
    lact = next(item for item in trajectory["concepts"] if item["concept"] == "lact")
    assert lact["status"] == "materialized"
    assert lact["physical_binding"]["source_concept"] == "lact"
    assert lact["physical_binding"]["canonical_unit"] == "mmol/L"
    assert lact["physical_binding"]["analysis_plausibility_range"] == {
        "minimum": 0.0,
        "maximum": 30.0,
    }
    assert lact["source"]["file"]
    assert lact["source"]["column"]
    assert not (_recursive_keys(payload) & _FORBIDDEN_SCIENTIFIC_SELECTION_KEYS)


def test_typed_trajectory_prompt_reads_only_sealed_fallback_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_dir, _cohort_path, context, *_rest = _prepare_typed_run(tmp_path)
    before = materialized_input_prompt_attachment(context)

    def must_not_recompute(_binding):
        raise AssertionError("prompt rendering must not recompute sealed ranges")

    monkeypatch.setattr(
        "easyicu.research_agent.research_context.typed.effective_analysis_plausibility_range",
        must_not_recompute,
    )
    after = materialized_input_prompt_attachment(context)

    assert after == before
