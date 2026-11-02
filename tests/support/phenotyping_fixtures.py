"""Synthetic phenotyping contexts, artifact bindings and planner skeleton."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import PHENOTYPE_ASSIGNMENTS_PRODUCT, PHENOTYPE_PROFILES_PRODUCT
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.schema import AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext, VariableRole
from tests.research_agent.planning.progressive_planner_fixtures import _payload


def _context(n: int) -> ResearchContext:
    return ResearchContext(
        research_question="Discover early cross-sectional phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="phenotype_fixture",
            database="synthetic",
            n_stays=n,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="str"),
            ConceptDescriptor(name="marker_a", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(
                name="marker_b", role=VariableRole.VITAL, dtype="float64"
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
        ],
        target_outcome="death",
    )

def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 360
    group = np.repeat(np.arange(3), n // 3)
    return pd.DataFrame(
        {
            "stay_id": [f"stay_{index}" for index in range(n)],
            "marker_a": rng.normal(group * 2.5, 0.5),
            "marker_b": rng.normal((2 - group) * 2.0, 0.6),
            "death": rng.binomial(1, 0.2 + group * 0.1),
        }
    )

def _binding(
    key: str, frame: pd.DataFrame, path: Path, step_id: str
) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": str(path.relative_to(path.parents[1])),
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "produced_by_step": f"producer_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
        "identity_row": {
            "input_key": key,
            "declared_kind": "table",
            "product": product,
            "evidence_id": f"evidence_{product}",
            "sha256": digest,
        },
        "step_id": step_id,
    }

def _primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_phenotypes",
        planned_analysis_role="primary",
        intent="Fit the Planner-selected early phenotype roster.",
        inputs=["stay_id", "marker_a", "marker_b", "death", "artifact:analysis_cohort"],
        expected_outputs=[PHENOTYPE_PROFILES_PRODUCT, PHENOTYPE_ASSIGNMENTS_PRODUCT],
        method="cross-sectional phenotyping",
        scientific_action_id="phenotyping.cluster_solution",
        phenotyping_feature_columns=["marker_a", "marker_b"],
    )

def _skeleton(features):
    payload = _payload()
    primary = payload["steps"][5]
    primary.update(
        step_id="cluster", planned_analysis_role="primary", module_id="custom_analysis",
        depends_on=["01_cohort"], raw_inputs=["marker_a", "marker_b", "death"], product_inputs=[],
        outputs=[{"product_id": product, "semantic_role": "custom"} for product in
                 ("table:phenotype_profiles", "table:phenotype_assignments")],
        scientific_action_id="phenotyping.cluster_solution", custom_method="cross_sectional_phenotyping",
        sensitivity_spec_ids=[], phenotyping_feature_columns=features,
    )
    payload.update(analysis_type="trajectory_clustering", steps=[payload["steps"][0], primary],
                   display_labels=[], robustness_intents=[])
    return ProgressivePlanSkeleton.model_validate(payload)
