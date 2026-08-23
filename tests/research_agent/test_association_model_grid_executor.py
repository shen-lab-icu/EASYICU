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
from easyicu.research_agent.authority.plan_input_closure import (
    close_measurement_companion_inputs,
)
from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
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
from easyicu.research_agent.contracts.declared_product import (
    declared_product_contract_findings,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.plan_utils import effect_output_authorized
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

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


def test_signed_standard_selection_authorizes_grid_effect_outputs_only_at_runtime():
    _projection, _authority, plan, _findings = _authority_and_plan()
    step = plan.steps[1]
    record = {
        "deterministic_standard_analysis": "association_model_grid",
        "deterministic_standard_selection_reason": (
            "signed_association_model_grid_contract_preflight"
        ),
        "standard_executor_candidates": {"claimed_by": "association_model_grid"},
    }

    assert effect_output_authorized(step) is False
    assert effect_output_authorized(step, step_record=record) is True
    assert effect_output_authorized(
        step, step_record={**record, "deterministic_standard_analysis": "other"}
    ) is False


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

    rebuilt_step = plan.steps[1].model_copy(deep=True)
    assert rebuilt_step is not plan.steps[1]
    rebuilt_selection = select_standard_executor(
        rebuilt_step,
        plan=plan,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection[
            "runtime_projection_sha256"
        ],
    )
    assert rebuilt_selection is not None
    assert rebuilt_selection.analysis_kind == "association_model_grid"

    receipt_selection = select_standard_executor(
        rebuilt_step,
        plan=plan,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection[
            "runtime_projection_sha256"
        ],
        plausibility_scope=FlagOnlyPlausibilityScope(
            step_id=rebuilt_step.step_id,
            expected_columns=("age",),
            source_contracts_sha256="0" * 64,
            authority_kind="test_raw_input_contract",
        ),
    )
    assert receipt_selection is not None
    compile(receipt_selection.code, "<association_model_grid>", "exec")
    assert "plausibility_audit" in receipt_selection.code

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


def test_runtime_grid_rebinds_after_generic_measurement_input_closure() -> None:
    _, authority, bound, _ = _authority_and_plan()
    context = ResearchContext(
        research_question=bound.research_question,
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[
            ConceptDescriptor(name=name, dtype="float64")
            for name in (
                "sep3_sofa2_max",
                "sep3_sofa2_measured",
                "sep3_sofa2_n",
            )
        ],
    )
    closed, findings = close_measurement_companion_inputs(
        plan=bound,
        context=context,
    )
    assert findings
    with pytest.raises(CurrentCaseScientificAuthorityError, match="inputs"):
        authority.validate_plan(closed)

    rebound, _ = ScientificRuntimeAuthorities(
        current_case=authority,
        trajectory=None,
    ).bind_plan(closed)

    authority.validate_plan(rebound)
    assert set(rebound.steps[1].inputs) == {
        *authority.required_columns(rebound),
        authority.cohort_product,
        authority.parent_product,
    }


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
    assert set(table["exposure"]) == {"sep3_sofa2_max"}
    assert table["is_reference"].sum() == 1
    assert table.loc[table["is_reference"], "analysis_id"].item() == (
        summary["scientific_runtime_receipt"]["reference_variant_id"]
    )
    assert set(table["outcome"]) == {"death"}
    assert set(table["adjustment_covariates"]) == {"age;sex;charlson_max"}
    assert table["fitted_covariates"].str.len().gt(0).all()
    assert table["landmark_hours"].notna().any()
    assert summary["basis_receipts"]["flexible_age_charlson"]
    assert summary["scientific_runtime_receipt"]["adapter"] == (
        "adjusted_association_executor/statsmodels"
    )
    assert summary["scientific_runtime_receipt"]["adjustment_covariates"] == (
        _COVARIATES
    )
    selection_record = {
        "deterministic_standard_analysis": "association_model_grid",
        "deterministic_standard_selection_reason": (
            "signed_association_model_grid_contract_preflight"
        ),
        "standard_executor_candidates": {"claimed_by": "association_model_grid"},
    }
    product_findings = declared_product_contract_findings(
        step=plan.steps[1],
        step_summary=summary,
        effect_method_authorized=effect_output_authorized(
            plan.steps[1], step_record=selection_record
        ),
        out_dir=out_dir,
    )
    assert not any(
        finding.detail.get("kind") == "unauthorized_effect_product"
        for finding in product_findings
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
