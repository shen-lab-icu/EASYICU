"""Pre-v1 top-level import paths are deliberately retired.

The public pre-release experiments are preserved by the
``archive/pre-v1-agent-compat-20260719`` Git tag.  Current releases expose the
canonical responsibility packages only; fresh experiments must not recreate
the deleted facade files.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

RETIRED_TOP_LEVEL_MODULES: dict[str, str | None] = {
    "agentic_coder": "agents.agentic_coder",
    "analysis_blueprint": "planning.analysis_blueprint",
    "analysis_method_suite": "planning.analysis_method_suite",
    "analysis_types": "planning.analysis_types",
    "article_contract": "reporting.article_contract",
    "authority_fs": "authority.filesystem",
    "bibtex": "reporting.bibtex",
    "capability_registry": "planning.capability_registry",
    "case_contexts": "case_plugins.contexts",
    "causal_audit": "review.causal_audit",
    "code_hygiene": "execution.code_hygiene",
    "code_patch": "repairs.patch",
    "code_preflight": "gates.preflight",
    "code_repair": "repairs.source",
    "code_repair_helpers": "repairs.helpers",
    "coder_authority_notes": "authority.coder_authority",
    "coder_context": "research_context.prompt_scope",
    "cohort_artifact_facts": "cohort.artifact_facts",
    "cohort_materializer": "cohort.materializer",
    "cohort_repair": "cohort.repair",
    "cohort_schema": "cohort.schema",
    "concept_audit_execution": "execution.concept_audit",
    "concept_audit_cache": "execution.concept_audit_cache",
    "concept_gate": "gates.concept",
    "concept_proposal": "discovery.concept_proposal",
    "context": "research_context.builder",
    "context_numeric": "authority.context_numeric_claims",
    "contract_gate": "gates.contract",
    "cost": "providers.cost",
    "cross_model_panel": "evaluation.cross_model_panel",
    "data_catalog": "acquisition.catalog",
    "data_foundation": "acquisition.foundation",
    "declared_product_contract": "contracts.declared_product",
    "deterministic_causal": None,
    "deterministic_clustering": None,
    "deterministic_cohort_flow": None,
    "deterministic_descriptive": "execution.runners.deterministic_descriptive",
    "deterministic_missingness": "execution.runners.deterministic_missingness",
    "deterministic_ordinal": None,
    "deterministic_robustness": "execution.runners.deterministic_robustness",
    "deterministic_sensitivity": None,
    "deterministic_survival": None,
    "discovery_handoff": "discovery.discovery_handoff",
    "discovery_package": "discovery.discovery_package",
    "discovery_story_figure": "discovery.discovery_story_figure",
    "display_suite": "reporting.display_suite",
    "easyicu_case_builder": "case_plugins.builder",
    "evidence": "authority.evidence_store",
    "evidence_authority": "authority.evidence_snapshot",
    "evidence_registration": "authority.registration",
    "experiment_spec": "orchestration.experiment_spec",
    "estimators": "robustness.estimators",
    "experience": "learning.experience",
    "figure_contract": "figures.publication",
    "figure_contracts": "figures.contracts",
    "figure_skill": "figures.skill",
    "figure_contract_preparation": "execution.figure_preparation",
    "figure_strategy": "planning.figure_strategy",
    "gate_evaluator": "gates.visual",
    "gate_semantics": "gates.semantics",
    "hypothesis_generator": "discovery.hypothesis_generator",
    "idea_mining": "discovery.idea_mining",
    "idea_mining_data_first": "discovery.idea_mining_data_first",
    "idea_mining_eval": "discovery.idea_mining_eval",
    "idea_mining_extended_feasibility": ("discovery.idea_mining_extended_feasibility"),
    "idea_mining_feasibility_tier": "discovery.idea_mining_feasibility_tier",
    "idea_mining_funnel": "discovery.idea_mining_funnel",
    "idea_mining_priorart": "discovery.idea_mining_priorart",
    "idea_mining_pubmed": "discovery.idea_mining_pubmed",
    "idea_mining_schema": "discovery.idea_mining_schema",
    "idea_registry": "discovery.idea_registry",
    "idea_scope": "discovery.idea_scope",
    "latex": "reporting.latex",
    "legacy_code_migrations": None,
    "lock_authority": "authority.lock_contract",
    "llm": "providers.llm",
    "llm_mocks": "providers.mocks",
    "manuscript_post": "reporting.manuscript_post",
    "method_compatibility": "gates.method_compatibility",
    "methodological_rigor": "review.methodological_rigor",
    "metadata_implementation_identity": "research_context.implementation_identity",
    "memory": "learning.memory",
    "method_capabilities": "execution.method_capabilities",
    "ordered_stratified_contract": "contracts.ordered_stratified",
    "pdf_render": "reporting.pdf_render",
    "pipeline_config": "orchestration.config",
    "pipeline_cross_db": "replication.report",
    "pipeline_execute": "execution.phase",
    "pipeline_primary_effect": "robustness.primary_effect",
    "pipeline_package": "orchestration.finalize",
    "pipeline_phases": None,
    "pipeline_profiles": "orchestration.profiles",
    "pipeline_report": "reporting.readiness",
    "pipeline_resume": "orchestration.resume",
    "pipeline_state": None,
    "pipeline_write": "reporting.write_phase",
    "pipeline_writer_aux": "reporting.writer_evidence",
    "pipeline_cache": "authority.pipeline_cache",
    "provider_budget": "authority.provider_budget",
    "prompts": "providers.prompts",
    "publication_figure_execution": "execution.publication_figure",
    "publication_figures": "figures.publication",
    "provenance": "authority.provenance",
    "repair_coordination": "repairs.coordination",
    "repair_reasons": "repairs.reasons",
    "reporting_checklist": "reporting.reporting_checklist",
    "research_context_v2": "research_context.typed",
    "review_artifacts": "reporting.review_artifacts",
    "reviewer": "reporting.reviewer",
    "run_input_capsule": "authority.run_input",
    "run_coordination": "execution.run_coordination",
    "run_lock": "authority.run_lock",
    "robustness_panel": "robustness.panel",
    "robustness_execution_contract": "contracts.robustness_execution",
    "runner": "execution.runner",
    "runtime_artifacts": "authority.runtime_artifacts",
    "step_execution": "execution.step_execution",
    "step_summary": None,
    "side_findings": "reporting.side_findings",
    "study_design": "planning.study_design",
    "study_design_playbook": "planning.study_design_playbook",
    "structured_retry": "providers.structured_retry",
    "step_attempt_authority": "authority.step_attempt",
    "step_authority_capsule": "authority.step_capsule",
    "step_authority_runtime": "authority.step_runtime",
    "step_worker_state": "execution.step_worker_state",
    "summary_repair": "repairs.summary",
    "temporal_semantics": "research_context.temporal_semantics",
    "temporal_features": "methods.temporal_features",
    "tier2_jury": "evaluation.tier2_jury",
    "tier2_rubric": "evaluation.tier2_rubric",
    "trajectory_stability_executor": (
        "execution.runners.trajectory_stability_executor"
    ),
    "trajectory_bundle": "trajectory.bundle",
    "trajectory_contract": "trajectory.contract",
    "trajectory_plan_contract": "trajectory.plan_contract",
    "trajectory_resume_schema": None,
    "visual_qa": "gates.visual_qa",
}


@pytest.mark.parametrize("leaf", sorted(RETIRED_TOP_LEVEL_MODULES))
def test_retired_top_level_module_is_absent(leaf: str) -> None:
    package_root = (
        Path(__file__).resolve().parents[2] / "src" / "easyicu" / "research_agent"
    )
    assert not (package_root / f"{leaf}.py").exists()
    assert importlib.util.find_spec(f"easyicu.research_agent.{leaf}") is None


@pytest.mark.parametrize(
    "target",
    sorted(
        target for target in RETIRED_TOP_LEVEL_MODULES.values() if target is not None
    ),
)
def test_canonical_replacement_is_importable(target: str) -> None:
    imported = importlib.import_module(f"easyicu.research_agent.{target}")
    assert imported.__name__ == f"easyicu.research_agent.{target}"


@pytest.mark.parametrize(
    ("name", "target"),
    [
        ("VisualQAAuditor", "gates.visual_qa"),
        ("VLMVisualQAAdapter", "gates.visual_qa"),
        ("PublicationFigureSkill", "figures.skill"),
        ("PublicationFigureSkillResult", "figures.skill"),
        ("FigureContract", "figures.publication"),
        ("PanelSpec", "figures.publication"),
        ("make_figure_contract", "figures.publication"),
        ("audit_figure_contract", "figures.publication"),
        ("apply_publication_style", "figures.publication"),
        ("save_publication_figure", "figures.publication"),
        ("audit_publication_exports", "figures.publication"),
    ],
)
def test_root_figure_api_preserves_canonical_object_identity(
    name: str,
    target: str,
) -> None:
    root = importlib.import_module("easyicu.research_agent")
    canonical = importlib.import_module(f"easyicu.research_agent.{target}")
    assert getattr(root, name) is getattr(canonical, name)


@pytest.mark.parametrize(
    "name",
    [
        "os_environ",
        "step_summary",
        "PlanPhaseState",
        "ExecutePhaseState",
        "WritePhaseState",
        "PlanPhaseRunner",
        "ExecutePhaseRunner",
        "WritePhaseRunner",
    ],
)
def test_retired_root_attribute_is_absent(name: str) -> None:
    root = importlib.import_module("easyicu.research_agent")
    assert not hasattr(root, name)
