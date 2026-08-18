"""The E1 sensitivity grid is host-compiled and reuses the verified model owner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.figure2_canonical9.e1_scientific_acceptance import (
    build_e1_model_grid_runtime_projection,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    AssociationModelGridRuntimeAuthority,
    CurrentCaseScientificAuthorityError,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    run_adjusted_association_from_env,
)
from easyicu.research_agent.execution.runners.association_model_grid_executor import (
    AssociationModelGridError,
    run_association_model_grid,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.schema import AnalysisPlan

_FIXTURE = Path(__file__).parent / "fixtures" / "real_plan_steps_fresh17_fresh19.json"
_COVARIATES = ["age", "sex", "charlson_max"]


def _model_terms() -> list[dict[str, object]]:
    return [
        {
            "name": "sep3_sofa2_max",
            "role": "exposure",
            "coding": "binary",
            "levels": ["0", "1"],
            "reference_level": "0",
            "transform": "treatment_contrast",
        },
        {
            "name": "age",
            "role": "covariate",
            "coding": "continuous",
            "transform": "identity",
        },
        {
            "name": "sex",
            "role": "covariate",
            "coding": "binary",
            "levels": ["0", "1"],
            "reference_level": "0",
            "transform": "treatment_contrast",
        },
        {
            "name": "charlson_max",
            "role": "covariate",
            "coding": "continuous",
            "transform": "identity",
        },
    ]


def _authority_and_plan():
    projection = build_e1_model_grid_runtime_projection()
    authority = load_current_case_scientific_runtime_authority(
        projection["deterministic_execution_contract"]
    )
    assert isinstance(authority, AssociationModelGridRuntimeAuthority)

    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    recorded = next(
        item for item in document["plans"] if item["label"] == "fresh19"
    )["plan"]
    parent = next(
        item
        for item in recorded["steps"]
        if item["step_id"] == "07_primary_adjusted_association"
    )
    parent["model_requirements"][0].update(
        covariates=_COVARIATES,
        model_terms=_model_terms(),
    )
    candidate = {
        "step_id": "08_sensitivity",
        "planned_analysis_role": "sensitivity",
        "intent": "Planner-level sensitivity intent.",
        "inputs": [],
        "expected_outputs": [authority.output_product],
        "method": "sensitivity_analysis",
        "sensitivity_spec_ids": list(authority.sensitivity_ids),
        "icu_rule_refs": [],
    }
    draft = AnalysisPlan.model_validate(
        {
            "research_question": "Estimate the prespecified association.",
            "analysis_type": "association_study",
            "steps": [parent, candidate],
        }
    )
    bound, findings = ScientificRuntimeAuthorities(
        current_case=authority,
        trajectory=None,
    ).bind_plan(draft)
    return projection, authority, bound, findings


def _cohort(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(20260817)
    exposure = rng.integers(0, 2, n)
    age = rng.normal(64.0, 12.0, n)
    sex = rng.integers(0, 2, n)
    charlson = rng.poisson(3.0, n)
    readmission = rng.integers(0, 2, n)
    probability = 1.0 / (
        1.0
        + np.exp(
            -(
                -3.0
                + 0.8 * exposure
                + 0.02 * (age - 64.0)
                + 0.1 * charlson
            )
        )
    )
    death = rng.binomial(1, probability, n)
    death_time = np.where(death == 1, rng.uniform(-2.0, 120.0, n), np.nan)
    return pd.DataFrame(
        {
            "sep3_sofa2_max": exposure,
            "age": age,
            "sex": sex,
            "charlson_max": charlson,
            "death": death,
            "death_time": death_time,
            "icu_readmission": readmission,
        }
    )


def _parent_binding(
    *,
    tmp_path: Path,
    frame: pd.DataFrame,
    authority: AssociationModelGridRuntimeAuthority,
    plan: AnalysisPlan,
) -> tuple[Path, dict[str, object]]:
    run_dir = (tmp_path / "run").resolve()
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    requirement = plan.steps[0].model_requirements[0]
    run_adjusted_association_from_env(
        requirement_id=requirement.requirement_id,
        exposure=requirement.exposure_source,
        outcome=requirement.outcome,
        covariates=requirement.covariates,
        model_terms=requirement.model_terms,
        estimator_kind="logistic",
        analysis_set=requirement.analysis_set,
        analysis_role=requirement.analysis_role,
        method_family=requirement.method_family,
        frame=frame,
        cohort_path=Path("cohort.parquet"),
        output_dir=parent_dir,
        emit_step_summary=False,
    )
    source = parent_dir / "adjusted_association_estimates.csv"
    artifact = evidence_dir / source.name
    artifact.write_bytes(source.read_bytes())
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    columns = pd.read_csv(artifact).columns.tolist()
    input_key = authority.parent_product
    product = input_key.partition(":")[2]
    evidence_id = "parent_adjusted_association"
    binding = {
        "relative_path": artifact.relative_to(run_dir).as_posix(),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": evidence_id,
        "product": product,
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": evidence_id,
            "input_key": input_key,
            "product": product,
            "sha256": digest,
        },
        "product_contract": {"columns": columns, "row_count": 1},
        "consumption_contract": {
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
    }
    return run_dir, {
        "step_id": plan.steps[1].step_id,
        "inputs": {input_key: binding},
    }


def test_host_compiles_the_exact_grid_and_the_real_router_claims_it() -> None:
    projection, authority, plan, findings = _authority_and_plan()

    authority.validate_plan(plan)
    assert findings[0].detail["reason_code"] == (
        "association_model_grid_host_compiled"
    )
    selected = select_standard_executor(
        plan.steps[1],
        plan=plan,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection[
            "runtime_projection_sha256"
        ],
    )
    assert selected is not None
    assert selected.analysis_kind == "association_model_grid"
    assert selected.consumed_input_keys == (
        authority.cohort_product,
        authority.parent_product,
    )

    drifted = plan.model_copy(
        update={
            "steps": [
                plan.steps[0],
                plan.steps[1].model_copy(update={"method": "generated_python"}),
            ]
        }
    )
    with pytest.raises(CurrentCaseScientificAuthorityError, match="method"):
        authority.validate_plan(drifted)


def test_grid_reuses_the_parent_fit_and_emits_all_signed_variants(
    tmp_path: Path,
) -> None:
    projection, authority, plan, _ = _authority_and_plan()
    frame = _cohort()
    run_dir, manifest = _parent_binding(
        tmp_path=tmp_path,
        frame=frame,
        authority=authority,
        plan=plan,
    )
    out_dir = tmp_path / "outputs"
    summary = run_association_model_grid(
        frame=frame,
        cohort_path=Path("cohort.parquet"),
        authority=authority,
        runtime_projection_sha256=projection["runtime_projection_sha256"],
        parent_requirement=plan.steps[0].model_requirements[0],
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=plan.steps[1].step_id,
    )

    table = pd.read_csv(out_dir / "e1_scientific_sensitivity.csv")
    assert summary["status"] == "ok"
    assert table["analysis_id"].tolist() == list(authority.sensitivity_ids)
    assert table.loc[0, "n_stays"] == len(frame)
    assert table.loc[1, "n_stays"] < len(frame)
    assert table.loc[2, "n_stays"] < len(frame)
    assert summary["basis_receipts"]["flexible_age_charlson"]
    assert summary["scientific_runtime_receipt"]["adapter"] == (
        "adjusted_association_executor/statsmodels"
    )

    parent_path = run_dir / manifest["inputs"][authority.parent_product][
        "relative_path"
    ]
    parent = pd.read_csv(parent_path)
    parent.loc[0, "estimate"] *= 2.0
    parent.to_csv(parent_path, index=False)
    binding = manifest["inputs"][authority.parent_product]
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    binding["sha256"] = digest
    binding["identity_row"]["sha256"] = digest
    binding["consumption_contract"]["artifact_sha256"] = digest
    with pytest.raises(AssociationModelGridError, match="disagrees"):
        run_association_model_grid(
            frame=frame,
            cohort_path=Path("cohort.parquet"),
            authority=authority,
            runtime_projection_sha256=projection["runtime_projection_sha256"],
            parent_requirement=plan.steps[0].model_requirements[0],
            out_dir=tmp_path / "tampered",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=plan.steps[1].step_id,
        )
