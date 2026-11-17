"""Synthetic signed-suite execution retains every stay and clusters uncertainty."""

import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    _adjusted_cox_fit,
)
from easyicu.research_agent.methods.patient_cluster_survival import (
    bind_groups,
    clustered_ph_test,
)
from tests.support.survival_sealed import (
    sealed_survival,
    synthetic_survival_rows,
    run_signed_suite,
)


def test_cox_uncertainty_matches_lifelines_patient_sandwich_after_row_selection(
    tmp_path,
):
    _, original = sealed_survival(tmp_path)
    body = original.model_dump(mode="json", exclude={"execution_contract_sha256"})
    dependence = PlannedDependenceRequirement(
        group_source="patient", group_derivation="identity"
    )
    body["dependence"] = dependence.model_dump(mode="json")
    authority = build_current_case_scientific_runtime_authority(body)
    rng = np.random.default_rng(731)
    frame = pd.DataFrame(
        {
            authority.derived_time_column: rng.uniform(1, 80, 240),
            authority.derived_event_column: rng.binomial(1, 0.6, 240),
            "treatment": rng.binomial(1, 0.5, 240),
            "age": rng.normal(60, 12, 240),
        }
    )
    source = frame.assign(patient=np.repeat(np.arange(60), 4))
    frame = frame.drop(index=[0, 4, 8])
    bind_groups(frame, source, dependence)
    actual, failed = _adjusted_cox_fit(frame, sealed=authority)
    expected = CoxPHFitter().fit(
        frame.assign(patient=source.loc[frame.index, "patient"]),
        duration_col=authority.derived_time_column,
        event_col=authority.derived_event_column,
        cluster_col="patient",
        robust=True,
    )
    assert failed is None
    np.testing.assert_allclose(actual.summary["se(coef)"], expected.summary["se(coef)"])
    ph = clustered_ph_test(
        frame,
        duration_col=authority.derived_time_column,
        event_col=authority.derived_event_column,
        covariates=["treatment", "age"],
    )
    assert set(ph.covariate) == {"treatment", "age", "global"}
    assert ph.p_value.between(0, 1).all()


def test_signed_suite_executes_and_records_all_stays_and_patient_structure(tmp_path):
    _, original = sealed_survival(tmp_path)
    body = original.model_dump(mode="json", exclude={"execution_contract_sha256"})
    body["dependence"] = PlannedDependenceRequirement(
        group_source="patient", group_derivation="identity"
    ).model_dump(mode="json")
    authority = build_current_case_scientific_runtime_authority(body)
    frame = synthetic_survival_rows(600).assign(patient=np.repeat(np.arange(60), 10))
    result = run_signed_suite(authority, frame, tmp_path / "suite")
    receipt = json.loads(
        (tmp_path / "suite" / "landmark_survival_runtime_receipt.json").read_text()
    )
    structure = receipt["repeated_stay_structure"]
    assert structure["n_patients"] == 60
    assert structure["patients_with_multiple_stays"] == 60
    assert structure["n_stays"] == result["n_complete_case"]
    cox = pd.read_csv(tmp_path / "suite" / "landmark_cox_summary.csv")
    assert cox.standard_error.gt(0).all()
