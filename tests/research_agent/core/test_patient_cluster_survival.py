"""Synthetic grouping checks: repeated copies remain one sampling unit."""

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.methods.patient_cluster_survival import (
    bind_groups,
    cox_fit_arguments,
    group_structure,
    jackknife_covariance,
)


def test_patient_jackknife_keeps_duplicate_stays_together():
    frame = pd.DataFrame(
        {"patient": [1, 1, 2, 2, 3, 3], "value": [0.0, 0.0, 3.0, 3.0, 9.0, 9.0]}
    )
    bind_groups(
        frame,
        frame,
        PlannedDependenceRequirement(
            group_source="patient", group_derivation="identity"
        ),
    )
    variance = jackknife_covariance(frame, lambda subset: [subset.value.mean()])
    # The mean is over three patients, not six independent stays.
    assert variance[0, 0] == pytest.approx(np.var([0.0, 3.0, 9.0], ddof=1) / 3)
    assert group_structure(frame)["patients_with_multiple_stays"] == 3
    subset = frame.iloc[2:]
    assert group_structure(subset)["n_patients"] == 2
    fitted, kwargs = cox_fit_arguments(subset)
    assert kwargs == {"cluster_col": "__host_patient_cluster", "robust": True}
    assert fitted[kwargs["cluster_col"]].nunique() == 2


def test_a_failed_patient_deletion_never_produces_successful_only_uncertainty():
    frame = pd.DataFrame({"patient": [1, 2, 3], "value": [0.0, 3.0, 9.0]})
    bind_groups(
        frame,
        frame,
        PlannedDependenceRequirement(
            group_source="patient", group_derivation="identity"
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        jackknife_covariance(
            frame, lambda subset: [np.nan if 0 not in subset.index else 1.0]
        )


def test_the_signed_patient_rule_is_part_of_plan_resume_identity():
    from easyicu.research_agent.authority.plan_scope import plan_signature
    from easyicu.research_agent.contracts.runtime_dependence import (
        RuntimeDependenceContract,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    step = AnalysisStep(
        step_id="primary",
        intent="Run the signed suite",
        method="signed_landmark_survival_suite",
    )
    original = AnalysisPlan(research_question="Synthetic survival", steps=[step])
    contract = RuntimeDependenceContract(
        owner_ref="scientific_runtime_contract:" + "a" * 64,
        dependence=PlannedDependenceRequirement(
            group_source="patient", group_derivation="identity"
        ),
    )
    revised = original.model_copy(
        update={
            "steps": [step.model_copy(update={"runtime_dependence_contract": contract})]
        }
    )
    assert plan_signature(original) != plan_signature(revised)
