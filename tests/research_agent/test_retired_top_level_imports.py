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
    "analysis_method_suite": "planning.analysis_method_suite",
    "analysis_types": "planning.analysis_types",
    "article_contract": "reporting.article_contract",
    "bibtex": "reporting.bibtex",
    "capability_registry": "planning.capability_registry",
    "case_contexts": "case_plugins.contexts",
    "causal_audit": "review.causal_audit",
    "code_patch": "repairs.patch",
    "code_preflight": "gates.preflight",
    "code_repair": "repairs.source",
    "code_repair_helpers": "repairs.helpers",
    "coder_authority_notes": "authority.coder_authority",
    "coder_context": "research_context.prompt_scope",
    "concept_audit_execution": "execution.concept_audit",
    "concept_gate": "gates.concept",
    "concept_proposal": "discovery.concept_proposal",
    "context": "research_context.builder",
    "contract_gate": "gates.contract",
    "cross_model_panel": "evaluation.cross_model_panel",
    "deterministic_causal": "execution.runners.deterministic_causal",
    "deterministic_clustering": "execution.runners.deterministic_clustering",
    "deterministic_cohort_flow": "execution.runners.deterministic_cohort_flow",
    "deterministic_descriptive": "execution.runners.deterministic_descriptive",
    "deterministic_missingness": "execution.runners.deterministic_missingness",
    "deterministic_ordinal": "execution.runners.deterministic_ordinal",
    "deterministic_robustness": "execution.runners.deterministic_robustness",
    "deterministic_sensitivity": "execution.runners.deterministic_sensitivity",
    "deterministic_survival": "execution.runners.deterministic_survival",
    "discovery_handoff": "discovery.discovery_handoff",
    "discovery_package": "discovery.discovery_package",
    "discovery_story_figure": "discovery.discovery_story_figure",
    "display_suite": "reporting.display_suite",
    "easyicu_case_builder": "case_plugins.builder",
    "evidence_registration": "authority.registration",
    "experience": "learning.experience",
    "figure_contract": "publication_figures",
    "figure_contract_preparation": "execution.figure_preparation",
    "figure_strategy": "planning.figure_strategy",
    "gate_evaluator": "gates.visual",
    "hypothesis_generator": "discovery.hypothesis_generator",
    "idea_mining": "discovery.idea_mining",
    "idea_mining_data_first": "discovery.idea_mining_data_first",
    "idea_mining_eval": "discovery.idea_mining_eval",
    "idea_mining_extended_feasibility": (
        "discovery.idea_mining_extended_feasibility"
    ),
    "idea_mining_feasibility_tier": "discovery.idea_mining_feasibility_tier",
    "idea_mining_funnel": "discovery.idea_mining_funnel",
    "idea_mining_priorart": "discovery.idea_mining_priorart",
    "idea_mining_pubmed": "discovery.idea_mining_pubmed",
    "idea_mining_schema": "discovery.idea_mining_schema",
    "idea_registry": "discovery.idea_registry",
    "idea_scope": "discovery.idea_scope",
    "latex": "reporting.latex",
    "manuscript_post": "reporting.manuscript_post",
    "method_compatibility": "gates.method_compatibility",
    "methodological_rigor": "review.methodological_rigor",
    "memory": "learning.memory",
    "pdf_render": "reporting.pdf_render",
    "provider_budget": "authority.provider_budget",
    "publication_figure_execution": "execution.publication_figure",
    "repair_coordination": "repairs.coordination",
    "repair_reasons": "repairs.reasons",
    "reporting_checklist": "reporting.reporting_checklist",
    "research_context_v2": "research_context.typed",
    "review_artifacts": "reporting.review_artifacts",
    "reviewer": "reporting.reviewer",
    "step_summary": None,
    "study_design": "planning.study_design",
    "study_design_playbook": "planning.study_design_playbook",
    "summary_repair": "repairs.summary",
    "temporal_features": "methods.temporal_features",
    "tier2_jury": "evaluation.tier2_jury",
    "tier2_rubric": "evaluation.tier2_rubric",
    "trajectory_stability_executor": (
        "execution.runners.trajectory_stability_executor"
    ),
}


@pytest.mark.parametrize("leaf", sorted(RETIRED_TOP_LEVEL_MODULES))
def test_retired_top_level_module_is_absent(leaf: str) -> None:
    package_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
    )
    assert not (package_root / f"{leaf}.py").exists()
    assert importlib.util.find_spec(f"easyicu.research_agent.{leaf}") is None


@pytest.mark.parametrize(
    "target",
    sorted(
        target
        for target in RETIRED_TOP_LEVEL_MODULES.values()
        if target is not None
    ),
)
def test_canonical_replacement_is_importable(target: str) -> None:
    imported = importlib.import_module(f"easyicu.research_agent.{target}")
    assert imported.__name__ == f"easyicu.research_agent.{target}"
