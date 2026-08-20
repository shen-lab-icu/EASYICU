from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    build_runtime_scientific_projection,
    load_default_case_protocol,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    CurrentCaseScientificAuthorityError,
    LandmarkSplineRuntimeAuthority,
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.execution.runners.landmark_spline_executor import (
    run_landmark_spline_association,
)
from easyicu.research_agent.execution.runners.source_feasibility_executor import (
    run_source_feasibility_fail_closed,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.schema import AnalysisPlan


def _authority(task_id: str):
    projection = build_runtime_scientific_projection(
        load_default_case_protocol(task_id)
    )
    authority = load_current_case_scientific_runtime_authority(
        projection.deterministic_execution_contract or {}
    )
    return projection, authority


def _e2_plan(authority: LandmarkSplineRuntimeAuthority) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Estimate the signed landmark association.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "01_signed_primary",
                    "planned_analysis_role": "primary",
                    "intent": authority.plan_intent,
                    "inputs": [
                        "dataset:analysis_cohort",
                        *authority.required_columns,
                    ],
                    "expected_outputs": list(authority.plan_outputs),
                    "method": authority.plan_method,
                    "scientific_capability": "association_freeform_v1",
                    "icu_rule_refs": [authority.plan_rule_ref],
                }
            ],
        }
    )


def _h2_plan(authority: SourceFeasibilityRuntimeAuthority) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Assess whether the contrast is identifiable.",
            "analysis_type": "causal_inference",
            "steps": [
                {
                    "step_id": "01_signed_feasibility",
                    "planned_analysis_role": "auxiliary",
                    "intent": authority.plan_intent,
                    "inputs": [],
                    "expected_outputs": list(authority.plan_outputs),
                    "method": authority.plan_method,
                    "icu_rule_refs": [authority.plan_rule_ref],
                }
            ],
        }
    )


def test_e2_plan_and_runtime_are_bound_to_one_signed_contract(tmp_path: Path) -> None:
    projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    plan = _e2_plan(authority)
    authority.validate_plan(plan)

    drifted_step = plan.steps[0].model_copy(
        update={"method": "linear_logistic_regression"}
    )
    with pytest.raises(CurrentCaseScientificAuthorityError, match="method"):
        authority.validate_plan(plan.model_copy(update={"steps": [drifted_step]}))

    rng = np.random.default_rng(20260810)
    n = 320
    lactate = rng.lognormal(mean=0.55, sigma=0.45, size=n)
    age = rng.normal(64.0, 13.0, size=n)
    sex = rng.choice(["F", "M"], size=n).astype(object)
    sex[0] = None
    charlson = rng.poisson(3.0, size=n).astype(float)
    probability = 1.0 / (
        1.0 + np.exp(-(-3.6 + 0.45 * lactate + 0.018 * (age - 60.0)))
    )
    death = rng.binomial(1, probability, size=n)
    death_time = np.where(death == 1, rng.uniform(30.0, 180.0, size=n), np.nan)
    frame = pd.DataFrame(
        {
            "lact_max": lactate,
            "death": death,
            "death_time": death_time,
            "los_icu": rng.uniform(1.2, 8.0, size=n),
            "age": age,
            "sex": sex,
            "charlson_first": charlson,
        }
    )
    summary = run_landmark_spline_association(
        frame=frame,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path,
    )
    assert summary["status"] == "ok"
    assert set(summary["output_files"]) == set(authority.plan_outputs)
    receipt = summary["scientific_runtime_receipt"]
    assert receipt["execution_contract_sha256"] == (
        authority.execution_contract_sha256
    )
    assert receipt["runtime_projection_sha256"] == (
        projection.runtime_projection_sha256
    )
    assert summary["n_primary_population"] == n
    assert summary["n_complete_case"] == n - 1


def test_h2_plan_forbids_effect_work_and_runtime_emits_no_estimate(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("h2_vasopressor_causal")
    assert isinstance(authority, SourceFeasibilityRuntimeAuthority)
    plan = _h2_plan(authority)
    authority.validate_plan(plan)

    forbidden = plan.steps[0].model_copy(
        update={
            "step_id": "02_psm",
            "method": "propensity_score_matching",
            "intent": "Construct a control arm and estimate a causal effect.",
        }
    )
    with pytest.raises(CurrentCaseScientificAuthorityError, match="forbidden"):
        authority.validate_plan(
            plan.model_copy(update={"steps": [plan.steps[0], forbidden]})
        )

    summary = run_source_feasibility_fail_closed(
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path,
    )
    assert summary["status"] == "ok"
    assert summary["scientific_decision"] == "blocked_by_source_authority"
    assert summary["reason_code"] == "H2_VERIFIED_NON_USE_UNAVAILABLE"
    assert summary["effect_estimate"] is None
    table = pd.read_csv(tmp_path / "h2_source_feasibility.csv")
    assert table.loc[0, "causal_contrast_authorized"] in (False, 0)
    assert pd.isna(table.loc[0, "effect_estimate"])


def test_signed_current_case_contracts_are_selected_by_the_real_execution_router() -> None:
    for task_id, expected_kind, plan_factory in (
        (
            "e2_lactate_mortality",
            "signed_landmark_spline_association",
            _e2_plan,
        ),
        (
            "h2_vasopressor_causal",
            "signed_source_feasibility_fail_closed",
            _h2_plan,
        ),
    ):
        projection, authority = _authority(task_id)
        plan = plan_factory(authority)
        selected = select_standard_executor(
            plan.steps[0],
            plan=plan,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256=(
                projection.runtime_projection_sha256
            ),
        )
        assert selected is not None
        assert selected.analysis_kind == expected_kind
        assert authority.execution_contract_sha256 in selected.code
        assert projection.runtime_projection_sha256 in selected.code

        rebuilt_step = plan.steps[0].model_copy(deep=True)
        assert rebuilt_step is not plan.steps[0]
        rebuilt = select_standard_executor(
            rebuilt_step,
            plan=plan,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256=(
                projection.runtime_projection_sha256
            ),
        )
        assert rebuilt is not None
        assert rebuilt.analysis_kind == expected_kind


def test_pipeline_config_requires_the_signed_contract_and_projection_as_a_pair(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("e2_lactate_mortality")
    config = PipelineConfig(
        workdir=tmp_path,
        current_case_scientific_runtime_authority=authority.model_dump(mode="json"),
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert config.current_case_scientific_runtime_authority is not None

    with pytest.raises(ValueError, match="configured together"):
        PipelineConfig(
            workdir=tmp_path,
            current_case_scientific_runtime_authority=authority.model_dump(
                mode="json"
            ),
        )
